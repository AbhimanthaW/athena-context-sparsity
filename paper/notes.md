## Structure

1. Introduction

2. Background
   2.1 N-gram language models
   2.2 Conditional entropy and additional context
   2.3 Finite-data sparsity
   2.4 Good-Turing intuition and occupancy statistics
   2.5 Smoothing, backoff, and interpolation

3. Research Questions and Hypotheses

4. Methods
   4.1 Corpus and preprocessing
   4.2 Train/validation/test protocol
   4.3 Nested training subsets
   4.4 N-gram models and add-alpha estimation
   4.5 Hyperparameter selection
   4.6 Singleton-history and distinct-history statistics
   4.7 Held-out unseen-context rate
   4.8 Marginal context value
   4.9 Paired moving-block bootstrap
   4.10 Interpolation robustness experiment

5. Results
   5.1 Fixed-order generalisation reversals
   5.2 Sparsity as model order and data size vary
   5.3 Training occupancy versus held-out coverage
   5.4 Failure of the marginal-harm hypothesis
   5.5 Estimator robustness through interpolation

6. Discussion
   6.1 Coverage prediction is not marginal-value prediction
   6.2 Why fixed-order models reverse
   6.3 Estimation error versus informational value
   6.4 What interpolation reveals
   6.5 Relation to classical language-model smoothing

7. Limitations

8. External Replication
   [second corpus, once completed]

9. Conclusion