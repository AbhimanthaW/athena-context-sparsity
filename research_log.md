# Research Log

## 2026-08-14 — Study design

### Primary question

Does the training-only singleton-history rate provide information about the sign or magnitude of the marginal held-out benefit of increasing n-gram order, beyond what is already explained by model order and training-set size?

### Central statistic

S<sub>k</sub> = f<sub>1</sub><sup>(k)</sup> / N<sub>k</sub>

This is treated as a Good–Turing-inspired singleton-history diagnostic, not as a guaranteed unbiased estimator of missing mass because overlapping language histories are dependent.

### Primary outcome

ΔH<sub>n</sub> = H<sub>test</sub>(q<sub>n</sub>) - H<sub>test</sub>(q<sub>n-1</sub>)

### Main methodological risks

- dependence between overlapping histories
- smoothing confounding
- vocabulary definition
- corpus-specific effects
- model order and training size confounding pooled correlations
- uncertainty around very small Delta H values
- leakage between train, validation, and test

## 2026-08-15 — Theory and Good–Turing motivation

### Question investigated

Is f1/N theoretically appropriate as a training-only measure of
context sparsity for this experiment?

### Learning / result

Under an i.i.d. discrete sampling model, the singleton fraction f1/N
is closely connected to expected missing probability mass.

However, consecutive n-gram histories are dependent and often overlap,
so the classical binomial derivation does not directly apply.

### Decision

Use:

S_k = f_1^(k) / N_k

as the primary training-only singleton-history rate.

Describe it as a Good–Turing-inspired sparsity diagnostic rather than
as a theoretically guaranteed or unbiased missing-mass estimator.

### Consequence for experiment

Measure the actual held-out unseen-history rate U_k separately and test
whether S_k tracks it.

Analyse S_k against Delta H_n, but do not interpret correlation as
causation.

### Unresolved issues

- how strongly S_k tracks held-out unseen-history rate
- whether S_k provides information beyond model order and training size
- sensitivity to smoothing method

## 2026-08-17 — Baseline pipeline implementation and methodological review

### Objective

Implement the preprocessing, n-gram estimation, occupancy-statistics, and
held-out evaluation components required for the main experiment.

### Components implemented

- deterministic corpus preprocessing
- sequential train/validation/test splitting
- vocabulary construction using the training partition only
- fixed `<UNK>` mapping across all dataset sizes and held-out partitions
- deterministic block-based nested training subsets
- arbitrary-order n-gram estimation
- add-alpha smoothing
- training-history occupancy statistics
- held-out cross-entropy and perplexity
- held-out unseen-history rate

### Methodological issue identified

The initial evaluation implementation allowed each n-gram order to score
a different number of held-out target tokens because order-n models began
evaluation after their own history length.

This would make comparisons such as

\[
\Delta H_n = H_n - H_{n-1}
\]

slightly confounded by differences in the evaluation sample.

### Correction

Evaluation was changed so that all model orders use exactly the same target
positions.

Because the maximum tested order is \(n=6\), the maximum history length is

\[
k_{\max}=5.
\]

All models therefore begin held-out evaluation at target position 5 within
each evaluation sequence, while each model uses only the amount of context
required by its own order.

Thus the number and identity of held-out prediction targets are identical
for \(n=1,\ldots,6\).

### Additional implementation changes

Preprocessing is now controlled by `experiments/config.json`, making the
experiment configuration the single source of truth.

A local seeded random-number generator is used for deterministic training
block shuffling without changing global random state.

Nested training subsets are represented using one shuffled block sequence
plus subset block counts rather than duplicating blocks for every training
fraction.

`NGramModel.fit()` now resets existing counts before fitting so repeated calls
cannot silently accumulate duplicate observations.

The implementation requires \(\alpha>0\), avoiding undefined unsmoothed MLE
behaviour for completely unseen histories.

Training occupancy counting was aligned exactly with the history-target
events used by the n-gram estimator.

### Processed data artifacts

The preprocessing pipeline produces:

- `cleaned_sequences.txt` — tokenised corpus before vocabulary mapping
- `train.txt` — vocabulary-normalised training partition
- `val.txt` — vocabulary-normalised validation partition
- `test.txt` — vocabulary-normalised test partition
- `vocab.json` — frozen training-derived vocabulary
- `dataset_blocks.json` — machine-readable block-structured experiment data

### Current status

Stage 4 baseline implementation is complete.

No main experimental results have been inspected.

The next stage is correctness validation with manually verifiable unit tests
before any validation sweep or test-set evaluation is permitted.