import argparse
import numpy as np
from nb.nb import NaiveBayes
from nb.utils.load_data import build_dataframe

if __name__ == "__main__":
    # (a) Ask the user for an input directory to load the data from.
    parser = argparse.ArgumentParser(description="Naive Bayes Algorithm")
    parser.add_argument("-f", "--indir", required=True, help="Data directory")
    args = parser.parse_args()
    data_dir = args.indir

    # (b) Build the DataFrame from the text within the data subfolders.
    (df_train, df_test) = build_dataframe(data_dir)

    # (c) Produce estimates for the priors and likelihoods.
    # (d) Run the train method from your NaiveBayes class.
    nb = NaiveBayes()
    nb.train(df_train)

    print("Priors:", np.round(nb.priors, 3))
    print("Likelihoods:", np.array2string(nb.likelihoods, precision=3))

    # (e) Run the test method from your NaiveBayes class.
    preds = nb.test(df_test)

    # (f) Print out the predictions
    print("Predictions:", preds)
