import tkinter as tk
from tkinter import Scrollbar
from basic_query import Query


class SearchEngineGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Search")
        self.root.geometry("800x420")

        self.entry1 = tk.Entry(self.root, width=50)  # use.get to get query
        self.entry1.grid(row=0, column=0)

        self.scroll = Scrollbar(self.root)
        self.scroll.grid(row=1, column=1, sticky='ns')

        self.display_box = tk.Text(self.root, wrap=tk.WORD, yscrollcommand=self.scroll.set)
        self.display_box.grid(row=1, column=0)

        self.search_b = tk.Button(self.root, width=20, text='Search', command=self.search_Basic_Query)
        self.search_b.grid(row=0, column=1)

        self.display_box.configure(yscrollcommand=self.scroll.set)
        self.scroll.config(command=self.display_box.yview)

        self.root.mainloop()

    def search_Basic_Query(self):
        self.display_box.delete('1.0', tk.END)    # clears search results for every new query
        q = Query(str(self.entry1.get()))
        results = q.get_urls()
        if results is None:
            ## No results suggestions from Google search
            self.display_box.insert(tk.END, "Your search did not match any documents \n\nSuggestions:" \
                   "\n\nMake sure all words are spelled correctly.\nTry different keywords.\n" \
                   "Try more general keywords.")
        else:
            display = ""
            for res in results:
                display = "".join([display, "Title: \n", results[res]["title"], "\nURL: \n", res, "\nDescription: \n", results[res]["desc"], "\n\n"])
            self.display_box.insert(tk.END, display)
        self.display_box.insert(tk.END, "\n\nSearch Engine Found: " + str(q.get_num_results()) + " results.")


if __name__ == "__main__":
    g = SearchEngineGUI()