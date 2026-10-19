import argparse

from nb.nb import NaiveBayes
from nb.utils.load_data import build_dataframe


def main():
    parser = argparse.ArgumentParser(description="Naive Bayes Algorithm")
    parser.add_argument("-f", "--indir", required=True, help="Data directory")
    args = parser.parse_args()

    # TODO: build DataFrames, train, test, print predictions


if __name__ == "__main__":
    main()
