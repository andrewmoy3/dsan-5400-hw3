import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

class NaiveBayes:
    """Multinomial Naive Bayes classifier."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.vocabulary = None
        self.priors = None
        self.likelihoods = None

    def train(self, df):
        """
        Fit vocabulary, priors and likelihoods from a labeled DataFrame.
        :param df: a pandas DataFrame
        :return: the vocabulary dict, a list of priors and a numpy array of likelihoods
        """
        # Create a dictionary that maps whitespace-separated tokens in the
        #  file to a unique index. Also, create variables for the number of
        #  documents and the number of classes. Use df.shape for the vocabulary
        #  and the nunique() method for the number of classes
        self.vocabulary = {}
        n_docs = df.shape[0]
        n_classes = df["author"].nunique()
        # loop through text files
        for text in df["text"]:
            for word in text.split():
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(self.vocabulary) # len of the vocab gives an index

        self.priors = []
        for i in range(n_classes):
            self.priors.append(df[df["author"] == i].shape[0] / n_docs)

        # Create a matrix containing all 0s called training_matrix of size
        #  (n_docs, len(vocabulary)), then fill it with the counts of each word
        #  for each document. This is the bag-of-words matrix for all the documents
        training_matrix = np.zeros((n_docs, len(self.vocabulary)))
        for i, text in enumerate(df["text"]):
            for word in text.split():
                if word in self.vocabulary:
                    training_matrix[i, self.vocabulary[word]] += 1

        # Get word counts for both classes
        word_counts_per_class = np.zeros((n_classes, len(self.vocabulary)))
        for i in range(n_classes):
            class_documents = training_matrix[df["author"] == i]
            word_counts_per_class[i] = np.sum(class_documents, axis=0)

        # Initialize a matrix to store the likelihoods
        self.likelihoods = np.zeros((n_classes, len(self.vocabulary)))

        # Then fill it in using Lidstone smoothing
        for i in range(n_classes):
            denominator = np.sum(word_counts_per_class[i]) + self.alpha * len(self.vocabulary)
            numerator = word_counts_per_class[i] + self.alpha
            self.likelihoods[i] = numerator / denominator
        
        return self.vocabulary, self.priors, self.likelihoods

    def test(self, df):
        """
        Return class predictions for each document in df.
        :param df: a pandas DataFrame
        :return: a numpy array of predictions
        """
        log_priors = np.log(self.priors)
        log_likelihoods = np.log(self.likelihoods)

        class_predictions = []
        for text in df["text"]:
            test_vector = np.zeros(shape=(len(self.vocabulary)))
            # Fill test_vector with counts for the words that appear in the
            #  vocabulary
            for word in text.split():
                if word in self.vocabulary:
                    test_vector[self.vocabulary[word]] += 1
            # Compute predictions p(y|text) (in log space to avoid underflow):
            #  log P(y|text) ~ log P(y) + sum_w count(w) * log P(w|y)
            preds = log_priors + log_likelihoods @ test_vector
            # Then get your predictions, yhat
            yhat = np.argmax(preds)
            class_predictions.append(yhat)
        return [int(x) for x in class_predictions]



