## 1a. Take a look at my attempt at Naive Bayes. How is it different from yours? What could you have done differently? What do you think I could have done better? Provide some comments.

1. In your calculation of the likelihoods, you add a plus one to the size of the vocabulary in the denominator, which denormalizes the values. The sum of the likelihoods no longer sums to one.

2. Building the vocabulary in the train method uses a set, rather than a dictionary or list, which results in different vocabulary indices between runs. This is not a problem for the model, but may make it more difficult to debug or run tests on with expected values.

3. There is an unnecessary double nested loop. It counts the number of times a token appears in a class for each token, rather than aggregating counts of words via a dictionary through n pass throughs, where n is the number of classes.

```python
    for token, idx in vocabulary.items():
        for auth in range(n_classes):
            count_token_idx_in_class_auth = sum(np.squeeze(training_matrix[
                                        np.where(df['author'] == auth), idx]))
            likelihoods[auth, idx] = (alpha +
                count_token_idx_in_class_auth) / \
                (alpha * (len(vocabulary) + 1) + word_counts_per_class[auth])
```

## 1c. Test Output

```
 (dsan5400) amoy1@amoy123:~/dsan/5400/hw3/dsan-5400-hw3/naive_bayes$ pytest tests/ -s
=========================== test session starts ============================
platform linux -- Python 3.14.8, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/amoy1/dsan/5400/hw3/dsan-5400-hw3/naive_bayes
configfile: pytest.ini
plugins: anyio-4.14.2
collected 3 items

tests/test_naive_bayes.py ...

============================ 3 passed in 2.15s =============================
```

## 1d. Final Output
```
(dsan5400) amoy1@amoy123:~/dsan/5400/hw3/dsan-5400-hw3/naive_bayes$ uv run nb/bin/main.py -f data/
Priors: [0.353 0.647]
Likelihoods: [[7.646e-04 9.427e-07 9.522e-05 ... 1.037e-05 1.037e-05 1.037e-05]
 [2.996e-03 4.558e-06 3.194e-04 ... 4.143e-07 4.143e-07 4.143e-07]]
Predictions: [1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
```

## 2e. Using each index, what document ids are returned for the query “system”? Note that this search should be case-insensitive, so that documents containing “system” and “System” are both returned. Note that in this question, we are not concerned with ranking the documents in any order. We’re simply returning the documents that contain the query terms, but we’re checking for the presence of those terms using our inverted index. Provide some comments.

Using whitespace tokenize
Docs containing system: [3, 1127, 2552, 3319, 5653, 8772, 8935, 14398, 16515, 16638]

Using nltk tokenizer
Docs containing system: [3, 1127, 2552, 3319, 5653, 8772, 8935, 14398, 16515, 16638]

Using n-gram tokenizer
Docs containing system: [3, 1127, 2552, 3319, 5653, 8772, 8935, 14398, 16515, 16638]

All three tokenizers return the same document ids, which is promising. For the n-gram tokenizer, we have to do some extra work because we have to find where the four character sections overlap to match the searched term.

## 2f. Using each index, what document ids are returned for the query “compatibility”? Provide some comments.

Using n-gram tokenizer
Docs containing compatibility: [8772, 14398]

Using nltk tokenizer
Docs containing compatibility: [8772, 14398]

Using whitespace tokenizer
Docs containing compatibility: [8772, 14398]

For compatibility, the three tokenizers return the same document ids, which is again promising. Beyond that, it is effectively the same problem as before, where we have to stitch the four character sections together.


## 2g. Using each index, what document ids are returned for the query terms “system” and “compatibility”? There should be at least one document returned. Provide some comments.
 
Using whitespace tokenizer
Docs containing compatibility AND system: [8772, 14398]

Using nltk tokenizer
Docs containing compatibility AND system: [8772, 14398]

Using n-gram tokenizer
Docs containing compatibility AND system: [8772, 14398]

This output contains the same two documents as the previous output, which checks out because those same two documents are listed for the search for 'system' as well. 
