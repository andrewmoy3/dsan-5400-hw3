from nb.nb import NaiveBayes
import pytest
import pandas as pd

@pytest.fixture
def model():
    # load tests/data/docs.txt into a DataFrame and train
    df = pd.read_csv("tests/data/docs.txt")
    nb = NaiveBayes(alpha=1)  # textbook uses add-1 smoothing
    nb.train(df)

    return nb

def test_nb_class(model):
    # 1. priors match the textbook
    assert model.priors[0] == 0.6
    assert model.priors[1] == 0.4

def test_nb_likelihoods(model):
    # 2. likelihoods match the textbook
    neg, pos = model.likelihoods[0], model.likelihoods[1]
    vocab = model.vocabulary

    assert neg[vocab["predictable"]] == pytest.approx((1 + 1) / (14 + 20))
    assert pos[vocab["predictable"]] == pytest.approx((0 + 1) / (9 + 20))
    assert neg[vocab["no"]] == pytest.approx((1 + 1) / (14 + 20))
    assert pos[vocab["no"]] == pytest.approx((0 + 1) / (9 + 20))
    assert neg[vocab["fun"]] == pytest.approx((0 + 1) / (14 + 20))
    assert pos[vocab["fun"]] == pytest.approx((1 + 1) / (9 + 20))

def test_nb_likelihood_sums(model):
    # 3. likelihoods sum to 1 for each class
    neg, pos = model.likelihoods[0], model.likelihoods[1]
    assert sum(neg) == pytest.approx(1.0)
    assert sum(pos) == pytest.approx(1.0)
