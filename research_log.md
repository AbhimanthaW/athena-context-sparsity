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