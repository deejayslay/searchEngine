import pickle
import json
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import wordnet
import nltk
import math
from collections import defaultdict
from bs4 import BeautifulSoup


class Query:
    def __init__(self, q):
        self.basepath = "WEBPAGES_RAW"
        self._stopword_set = None  ## Set of stopwords
        with open("stopwords.txt") as f:
            self._stopword_set = {line.rstrip() for line in f}
        f.close()

        infile = open("inverted_index", 'rb')  # need to open both this and the bi_inverted_index
        self.inverted_index = pickle.load(infile)
        infile.close()

        infile = open("bi_inverted_index", 'rb')  # need to open both this and the bi_inverted_index
        self.bi_inverted_index = pickle.load(infile)
        infile.close()

        self.query = str(q)
        # tokenize and lemmatize query
        self.query_words = []
        self.lemmatize_tokenize()

        # tf-idfs for query
        self.query_tf_idfs = {}
        self.tf_idf()

        ## Total number of results for a query
        self.total_results_num = 0

        ## Implementing bi-gram index for retrieval
        self.bi_query_words = []
        for i in range(len(self.query_words)):
            if i+1 < len(self.query_words):
                bi = "".join([self.query_words[i], " ", self.query_words[i+1]])
                self.bi_query_words.append(bi)
        self.bi_query_tf_idfs = {}
        self.bi_tf_idf()


    def get_urls(self):
        urls = {}
        j = open('WEBPAGES_RAW/bookkeeping.json')
        data = json.load(j)
        top_results = self.cos_sim_helper()       ## this is a dict of the top 20 URLS, and their 600 char (approx. 100 word) descriptions
        if len(top_results.keys()) == 0:
            print("No Results")
            return None
        for docID in top_results.keys():
            url = data[docID]
            urls[url] = {}
            urls[url]["title"] = top_results[docID]["title"]
            urls[url]["desc"] = top_results[docID]["desc"]
        j.close()
        num_res = self.get_num_results()
        print(num_res)
        for item in urls:
            print(item)
        return urls

    def get_num_results(self):
        return self.total_results_num

    ## Lemmatizes and tokenizes the query, inserts tokens into a list of token strings
    def lemmatize_tokenize(self):
        lemmatizer = WordNetLemmatizer()
        tk = re.split('[^a-z0-9]+', self.query.lower())

        for item in tk:
            if item.isnumeric():
                pass
            elif (item.isalnum()) and (item.isascii()) and (item not in self._stopword_set) and (len(item) > 1):
                pos = self.get_wordnet_pos(item)
                # takes care of adverbs ending in ly
                if pos == 'r' and item.endswith('ly'):
                    if item.endswith('ily'):
                        item = item[0:-3] + item[-1]
                    else:
                        item = item[:-2]
                        # new word is now an adjective
                    pos == 'a'
                item = lemmatizer.lemmatize(item, pos)
                self.query_words.append(item)

    # https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    ## Gets part of speech (POS tag) of each token
    def get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    ## Get query tokens tf-idfs
    def tf_idf(self):
        for word in self.query_words:
            if word in self.inverted_index.keys():
                for docID in self.inverted_index[word]:
                    for key, value in self.inverted_index[word][docID].items():
                        if key == 'tf-idf':
                            self.query_tf_idfs[word] = (1 + math.log(1)) * (math.log(37497 / len(self.inverted_index[word])))

    ## Get query tokens tf-idfs (bi-gram)
    def bi_tf_idf(self):
        for word in self.bi_query_words:
            if word in self.bi_inverted_index.keys():
                for docID in self.bi_inverted_index[word]:
                    for key, value in self.bi_inverted_index[word][docID].items():
                        if key == 'tf-idf':
                            self.bi_query_tf_idfs[word] = (1 + math.log(1)) * (
                                math.log(37497 / len(self.bi_inverted_index[word])))

    ## Cosine Similarity and Scoring for 1-gram
    def cos_sim_helper(self):
        ## Scores from bi-gram computations
        bi_gram_scores = self.bi_cos_sim()

        # minimizes num of calculations
        ## to check if doc contains at least half of query words
        ## ex query: lion bane stststst
        ## as long as the doc contains at least 2/3 of those terms
        search_threshold = math.ceil(len(self.query_words) / 2)

        ## Left Hand Side of Equation
        queryLength = 0.0
        normalized_query_wt = {}    ## dict of normalized query wts
        ## Compute query length
        for word in self.query_tf_idfs.keys():
            queryLength += math.pow(self.query_tf_idfs[word], 2)
        queryLength = math.sqrt(queryLength)    ## final query length
        ## Compute normalized query weights for each query term
        for word in self.query_tf_idfs.keys():
            normalized_query_wt[word] = self.query_tf_idfs[word] / queryLength

        ## Right Hand Side of Equation
        doc_contains = defaultdict(dict)    ## each docID has a dict of tokens that match query tokens
        docLength = 0.0
        ## Generates dict of docID and their associated matching query tokens
        for word in self.query_words:
            for docID in self.inverted_index[word].keys():
                doc_contains[docID][word] = 0
        ## Computes normalized document wts
        for docID in doc_contains:
            for word in doc_contains[docID]:
                if len(doc_contains[docID]) >= search_threshold:
                    docLength += math.pow(self.inverted_index[word][docID]['tf-idf'], 2)
        docLength = math.sqrt(docLength)    ## final document length

        ## If no results, return empty dict
        if docLength == 0:
            return {}

        for docID in doc_contains:
            if len(doc_contains[docID]) >= search_threshold:
                for word in doc_contains[docID]:
                    doc_contains[docID][word] += self.inverted_index[word][docID]['tf-idf'] / docLength      ## normalized document wt

        ## Dictionary of doc scores
        score_dict = {}
        for docID in doc_contains.keys():
            if len(doc_contains[docID]) >= search_threshold:
                score = 0.0
                for word in doc_contains[docID]:
                    score += normalized_query_wt[word] * doc_contains[docID][word]
                ## html tag importance is given a weight of .1 in the score calculation
                if docID in bi_gram_scores.keys():
                    score_dict[docID] = math.sqrt(score) + (.1 * self.inverted_index[word][docID]['avg_weight']) + (.1 * bi_gram_scores[docID])
                else:
                    score_dict[docID] = math.sqrt(score) + (.1 * self.inverted_index[word][docID]['avg_weight'])

        ## Sorts list of docs by descending score
        sorted_scores = sorted(score_dict.items(), key=lambda x: -x[1])

        ## Total number of results for query
        self.total_results_num = len(sorted_scores)

        ## Dictionary of Top 20 Results
        result_dict = {}

        ## If there's 20 or more results, return the top 20 results that are at least 1200 chars in length
        ## Reasoning: 1200 is approximately 200 words on avg which is what we've determined is a meaningful
        ## amount of text in a document for our search
        counter = 0
        ## If there's less than 20 results, return all of them
        max_if_lt_20 = 20
        if len(sorted_scores) < 20:
            max_if_lt_20 = len(sorted_scores)
        for res in sorted_scores:
            pathname = "".join([self.basepath, "/", res[0]])
            path = open(pathname, 'r', encoding='utf8')
            contents = path.read()
            soup = BeautifulSoup(contents, 'lxml')
            text = "".join([soup.get_text()])
            # https://drumcoder.co.uk/blog/2012/jul/13/removing-non-ascii-chars-string-python/
            # only includes ascii characters, and gets rid of newline characters
            text = "".join([c for c in text if ord(c) < 128]).replace('\n', ' ')
            if len(text) >= 1200 and counter < 20 and counter < max_if_lt_20:
                counter += 1
                result_dict[res[0]] = {}
                result_dict[res[0]]["desc"] = text[0:601]

                title_tag = soup.find('title')
                if title_tag:
                    result_dict[res[0]]["title"] = title_tag.text.lower()
                else:
                    result_dict[res[0]]["title"] = "No meta title given"

            elif counter >= 20 or counter >= max_if_lt_20:
                break
        return result_dict

    ## Cosine Similarity and Scoring from 2-gram
    def bi_cos_sim(self):
        ## to check if doc contains at least half of query words
        ## ex query: lion bane stststst
        ## as long as the doc contains at least 2/3 of those terms
        search_threshold = math.ceil(len(self.bi_query_words) / 2)

        ## Left Hand Side of Equation
        queryLength = 0.0
        normalized_query_wt = {}  ## dict of normalized query wts
        ## Compute query length
        for word in self.bi_query_tf_idfs.keys():
            queryLength += math.pow(self.bi_query_tf_idfs[word], 2)
        queryLength = math.sqrt(queryLength)  ## final query length
        ## Compute normalized query weights for each query term
        for word in self.bi_query_tf_idfs.keys():
            normalized_query_wt[word] = self.bi_query_tf_idfs[word] / queryLength

        ## Right Hand Side of Equation
        doc_contains = defaultdict(dict)  ## each docID has a list of tokens that match query tokens
        docLength = 0.0
        ## Generates dict of docID and their associated matching query tokens
        for word in self.bi_query_words:
            for docID in self.bi_inverted_index[word].keys():
                doc_contains[docID][word] = 0
        ## Computes normalized document wts
        for docID in doc_contains:
            for word in doc_contains[docID]:
                if len(doc_contains[docID]) >= search_threshold:
                    docLength += math.pow(self.bi_inverted_index[word][docID]['tf-idf'], 2)
        docLength = math.sqrt(docLength)  ## final document length

        ## If no results, return empty dict
        if docLength == 0:
            return {}

        for docID in doc_contains:
            if len(doc_contains[docID]) >= search_threshold:
                for word in doc_contains[docID]:
                    doc_contains[docID][word] += self.bi_inverted_index[word][docID]['tf-idf'] / docLength  ## normalized document wt

        ## Dictionary of doc scores
        score_dict = {}
        for docID in doc_contains.keys():
            if len(doc_contains[docID]) >= search_threshold:
                score = 0.0
                for word in doc_contains[docID]:
                    score += normalized_query_wt[word] * doc_contains[docID][word]
                ## html tag importance is given a weight of .3 in the score calculation
                score_dict[docID] = math.sqrt(score)
        return score_dict