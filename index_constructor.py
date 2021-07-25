from bs4 import BeautifulSoup
import nltk
import os
import re
from nltk.stem import WordNetLemmatizer
from _collections import defaultdict
from nltk.corpus import wordnet
import pickle
import math


class Index:
    def __init__(self):
        self.basepath = "WEBPAGES_RAW"
        self.CORPUS_SIZE = 37497
        self._stopword_set = None  ## Set of stopwords
        with open("stopwords.txt") as f:
            self._stopword_set = {line.rstrip() for line in f}
        f.close()

        ## 1-gram inverted index
        self._index_dict = defaultdict(dict)
        ## 2-gram inverted index
        self._bi_dict = defaultdict(dict)

    def run(self):
        # loops through all folders
        for folder in os.listdir(self.basepath):
            if os.path.isdir(os.path.join(self.basepath, folder)):
                # loops through all files
                for filename in os.listdir(os.path.join(self.basepath, folder)):
                    pathname = "".join([self.basepath, "/", folder, "/", filename])
                    doc_ID = "".join([folder, "/", filename])
                    path = open(pathname, 'r', encoding='utf8')
                    self.parser(path, pathname, doc_ID)
                print(folder)
        self.tf_idf()
        self.bi_tf_idf()

        fn = "inverted_index"
        outfile = open(fn, "wb")
        pickle.dump(self._index_dict, outfile)
        outfile.close()

        filename = "bi_inverted_index"
        outfile_bi = open(filename, "wb")
        pickle.dump(self._bi_dict, outfile_bi)
        outfile_bi.close()

    def parser(self, path, pathname, doc_ID):
        contents = path.read()
        soup = BeautifulSoup(contents, 'lxml')
        self.get_titles(soup, pathname, doc_ID)
        self.get_body(soup, pathname, doc_ID)

    ## Importance weights
    ## Title: 1.0
    ## H1: .6
    ## H2: .57
    ## H3: .54
    ## H4 - H6: .5
    ## Strong/Bold: .3
    ## Body: .1
    def get_titles(self, soup, pathname, doc_ID):
        title_tag = soup.find('title')
        # if there is a title
        if title_tag:
            title_tag = title_tag.text.lower()
            self.lemmatize_tokenize(pathname, title_tag, 1.0, doc_ID)

    def get_body(self, soup, pathname, doc_ID):
        h1_text = ""
        h2_text = ""
        h3_text = ""
        h4_h6_text = ""
        strong_bold_text = ""
        rem_body_text = ""

        # if there is a body tag
        if soup.find('body'):
            # check all children tags within body tag
            for tag in soup.find('body').find_all():
                # if there's text
                if tag.text:
                    # check for h1 tags
                    if tag.name == "h1":
                        h1_text = "".join([h1_text, tag.text.lower(), " "])
                    # check for h2 tags
                    elif tag.name == "h2":
                        h2_text = "".join([h2_text, tag.text.lower(), " "])
                    # check for h3 tags
                    elif tag.name == "h3":
                        h3_text = "".join([h3_text, tag.text.lower(), " "])
                    # check for h4, h5, h6 tags
                    elif tag.name in ["h4", "h5", "h6"]:
                        h4_h6_text = "".join([h4_h6_text, tag.text.lower(), " "])
                    # check for strong, bold tags
                    elif tag.name in ["strong", "bold"]:
                        strong_bold_text = "".join([strong_bold_text, tag.text.lower(), " "])
                    # check for any remaining tags that has text (eg. p tag)
                    else:
                        rem_body_text = "".join([rem_body_text, tag.text.lower(), " "])

        self.lemmatize_tokenize(pathname, h1_text, .6, doc_ID)
        self.lemmatize_tokenize(pathname, h2_text, .57, doc_ID)
        self.lemmatize_tokenize(pathname, h3_text, .54, doc_ID)
        self.lemmatize_tokenize(pathname, h4_h6_text, .5, doc_ID)
        self.lemmatize_tokenize(pathname, strong_bold_text, .3, doc_ID)
        self.lemmatize_tokenize(pathname, rem_body_text, .1, doc_ID)

    def lemmatize_tokenize(self, pathname, text, weight, doc_ID):
        lemmatizer = WordNetLemmatizer()
        ## Tokenize for alphanumeric characters using regular expressions
        tk = re.split('[^a-z0-9]+', text)
        # all tokens to use for bigrams
        modified_tk = []

        # for each token
        for item in tk:
            # for posting data shown below
            nested_dict = {}
            if item.isnumeric():  # if token is only numeric, throw out
                pass
            elif (item.isalnum()) and (item.isascii()) and (item not in self._stopword_set) and (len(item) > 1):
                pos = self.get_wordnet_pos(item)  # get pos (part of speech) for token
                # takes care of adverbs ending in ly
                if pos == 'r' and item.endswith('ly'):
                    if item.endswith('ily'):
                        item = item[0:-3] + item[-1]
                    else:
                        item = item[:-2]
                        # new word is now an adjective
                    pos == 'a'
                item = lemmatizer.lemmatize(item, pos)
                modified_tk.append(item)
                # update tf and weights
                if doc_ID in self._index_dict[item]:
                    self._index_dict[item][doc_ID]['tf'] += 1
                    self._index_dict[item][doc_ID]['weights'] += weight
                # set postings list
                else:
                    # set to 0 for now
                    nested_dict['tf-idf'] = 0
                    nested_dict['avg_weight'] = 0
                    nested_dict['tf'] = 1
                    nested_dict['weights'] = weight
                    self._index_dict[item][doc_ID] = nested_dict

        # creates bi-gram index
        for i in range(len(modified_tk)):
            nested_dict = {}
            # if last element not reached
            if i + 1 < len(modified_tk):
                bi = "".join([modified_tk[i], " ", modified_tk[i + 1]])
                if doc_ID in self._bi_dict[bi]:
                    self._bi_dict[bi][doc_ID]['tf'] += 1
                    self._bi_dict[bi][doc_ID]['weights'] += weight
                else:
                    # set to 0 for now
                    nested_dict['tf-idf'] = 0
                    nested_dict['avg_weight'] = 0
                    nested_dict['tf'] = 1
                    nested_dict['weights'] = weight
                    self._bi_dict[bi][doc_ID] = nested_dict

                    # https://www.machinelearningplus.com/nlp/lemmatization-examples-python/

    # gets part of speech for a given word
    def get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    # calculate avg_weight and tf_idf for all words in the index
    def tf_idf(self):
        for k, v in self._index_dict.items():
            for k2, v2 in v.items():
                for key, value in v2.items():
                    if key == 'weights':
                        self._index_dict[k][k2]['avg_weight'] = value / (self._index_dict[k][k2]['tf'])
                    if key == 'tf-idf':
                        # formula taken from lecture slides
                        self._index_dict[k][k2]['tf-idf'] = (1 + math.log(self._index_dict[k][k2]['tf'])) * (
                            math.log(self.CORPUS_SIZE / len(self._index_dict[k])))

    # calculate avg_weight and tf_idf for all words in the bi-gram index
    def bi_tf_idf(self):
        for k, v in self._bi_dict.items():
            for k2, v2 in v.items():
                for key, value in v2.items():
                    if key == 'weights':
                        self._bi_dict[k][k2]['avg_weight'] = value / (self._bi_dict[k][k2]['tf'])
                    if key == 'tf-idf':
                        # formula taken from lecture slides
                        self._bi_dict[k][k2]['tf-idf'] = (1 + math.log(self._bi_dict[k][k2]['tf'])) * (
                            math.log(self.CORPUS_SIZE / len(self._bi_dict[k])))


if __name__ == "__main__":
    ind = Index()
    ind.run()
