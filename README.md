# When Does More Context Hurt?

Research project investigating whether training-only context sparsity
can predict the marginal held-out benefit of increasing n-gram order.

## Research Question

Does the training-only singleton-history rate provide information about
the sign or magnitude of the marginal held-out benefit of increasing
n-gram order?

## Core Quantities

For history length k:

S_k = f_1^(k) / N_k

where f_1^(k) is the number of histories observed exactly once and N_k
is the total number of training-history occurrences.

For model order n:

Delta H_n = H_test(q_n) - H_test(q_(n-1))

Delta H_n < 0 means additional context improved held-out prediction.
Delta H_n > 0 means additional context degraded held-out prediction.

S_k is treated as a Good–Turing-inspired singleton-history diagnostic,
not as a theoretically guaranteed missing-mass estimator.

## Status

Day 1 — theory, experimental design, and implementation validation.

No final experimental results have been produced yet.

## Planned Experiment

- n-gram orders: 1–6
- multiple training-corpus sizes
- fixed train/validation/test split
- tuned add-alpha smoothing
- held-out cross-entropy and perplexity
- singleton-history statistics
- unseen held-out history rate
- marginal cross-entropy change between consecutive model orders

## Repository Structure

Briefly describe src/, tests/, experiments/, results/, figures/, notes/,
paper/, and data/.

## Reproducibility

Environment and experiment configuration are version-controlled.
Dataset provenance and preprocessing decisions are documented under data/.