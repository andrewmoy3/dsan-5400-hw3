# naive_bayes

A multinomial Naive Bayes classifier that predicts whether a presidential
speech was given by Kennedy or Johnson. The `NaiveBayes` class (`nb/nb.py`)
learns class priors and Lidstone-smoothed word likelihoods from labeled
speeches with `train`, then classifies unseen speeches with `test`.

## Project structure

```
naive_bayes/
├── data/                   # johnson/, kennedy/ (training) and unlabeled/ (test)
├── nb/
│   ├── nb.py               # NaiveBayes class
│   ├── utils/load_data.py  # build_dataframe(): reads speeches into DataFrames
│   └── bin/main.py         # command-line entry point
├── tests/
│   ├── data/docs.txt       # Jurafsky & Martin worked-example documents
│   └── test_naive_bayes.py # checks priors, likelihoods, and that likelihoods sum to 1
├── environment.yml
├── pytest.ini
└── setup.py
```

## Install

```bash
conda env create -f environment.yml
conda activate dsan5400
pip install -e .
```

## Run

```bash
python -m nb.bin.main -f data
```

`-f` is the data directory containing the `kennedy/`, `johnson/`, and
`unlabeled/` subfolders. The script prints the priors, the likelihoods, and
the predicted label for each unlabeled speech (0 = Kennedy, 1 = Johnson).

## Test

```bash
pytest tests/
```

The tests train on the five documents from the Jurafsky & Martin Naive Bayes
worked example with add-1 smoothing and check the results against the
textbook's values.
