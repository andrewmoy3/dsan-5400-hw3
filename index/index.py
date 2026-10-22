import argparse
import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import string


def parse_collection(path):
    """
    Parse an XML collection of <doc> elements into (docid, text) pairs. Assumes the XML is formed into <doc> elements with <docid> and <msgtext> children.
    :param path: path to the XML file
    :return: a list of (docid, text) tuples
    """
    root = ET.parse(path).getroot()
    docs = []
    for doc in root.iter("doc"):
        docid = int(doc.findtext("docid"))
        text = doc.findtext("msgtext", default="")
        docs.append((docid, text))
    return docs

class Tokenizer:
    def tokenize(self, text): 
        raise NotImplementedError("Tokenizer subclasses must implement the tokenize method.")

    def normalize(self, tokens):
        tokens = [t.lower().strip(string.punctuation) for t in tokens]
        return [t for t in tokens if t]  

class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text):
        return text.split()

class OffTheShelfTokenizer(Tokenizer):
    def __init__(self):
        # only download the punkt models if they aren't installed yet
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

    def tokenize(self, text):
        return word_tokenize(text)

class NGramTokenizer(Tokenizer):
    def __init__(self, n=4):
        self.n = n

    def tokenize(self, text):
        return [text[i:i+self.n] for i in range(len(text)-self.n+1)]

def search(index, words, tokenizer, stemmer=None):
    """
    Return the docids of documents containing every query word
    :param index: dict mapping term -> list of (docid, tf) pairs
    :param words: list of query words
    :param tokenizer: the Tokenizer used to build the index
    :param stemmer: the stemmer used to build the index, or None
    :return: a sorted list of docids
    """
    results = []
    # tokenize each word separately so n-grams don't span two query words
    for word in words:
        terms = tokenizer.normalize(tokenizer.tokenize(word))
        if stemmer:
            terms = [stemmer.stem(term) for term in terms]
        for term in terms:
            results.append({docid for docid, _ in index.get(term, [])})
    return sorted(set.intersection(*results)) if results else []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inverted index")
    parser.add_argument("-f", required=True, help="Path to XML collection")
    parser.add_argument("-t", required=True, help="Tokenizer to use")
    parser.add_argument("-s", action="store_true", help="Apply stemming")
    parser.add_argument("-q", nargs="+", help="Query word(s); returns docids containing all of them")
    args = parser.parse_args()

    # (A) Read in and parse the XML collection
    docs = parse_collection(args.f)
    print(f"Parsed {len(docs)} documents from {args.f}")

    # (B) Initialize the tokenizer
    if args.t == "whitespace":
        print("Using whitespace tokenizer")
        tokenizer = WhitespaceTokenizer()
    elif args.t == "nltk":
        print("Using nltk tokenizer")
        tokenizer = OffTheShelfTokenizer()
    elif args.t == "ngram":
        print("Using n-gram tokenizer")
        tokenizer = NGramTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer: {args.t}")

    stemmer = PorterStemmer()
    index = {}
    for text in docs:
        docid, tokens = text
        token_list = tokenizer.normalize(tokenizer.tokenize(tokens))
        if args.s:
            stem_list = [stemmer.stem(token) for token in token_list]
        else:
            stem_list = token_list

        tfs = Counter(stem_list)

        for token in tfs:
            if token in index:
                index[token] += [(docid, tfs[token])]
            else:
                index[token] = [(docid, tfs[token])]

    # (E-G) Search the index if query words were given
    if args.q:
        docids = search(index, args.q, tokenizer, stemmer if args.s else None)
        print(f"Docs containing {' AND '.join(args.q)}: {docids}")

