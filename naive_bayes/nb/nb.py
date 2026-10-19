import numpy as np


class NaiveBayes:
    """Multinomial Naive Bayes classifier."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def train(self, df):
        """Fit vocabulary, priors and likelihoods from a labeled DataFrame."""
        # TODO: move train_nb() from Assignment 2 here
        raise NotImplementedError

    def test(self, df):
        """Return class predictions for each document in df."""
        # TODO: move test() from Assignment 2 here
        raise NotImplementedError
