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

## 2026-08-17 — Stage 5 correctness validation

### Objective

Validate the preprocessing, n-gram estimation, occupancy-statistics, and
held-out evaluation pipeline before running experiments on the Shakespeare
corpus.

### Validation performed

A pytest suite was implemented covering:

- deterministic tokenisation
- training-only vocabulary construction
- `<UNK>` mapping
- deterministic block shuffling
- nested training subsets
- preservation of artificial block boundaries
- manually verified n-gram counts
- reset-on-fit behaviour
- add-alpha probability normalisation
- uniform prediction for unseen histories
- occupancy statistics \(N_k\), \(T_k\), \(f_1^{(k)}\), \(S_k\), and \(D_k\)
- held-out cross-entropy
- perplexity
- unseen-history rate \(U_k\)
- identical held-out target positions for \(n=1,\ldots,6\)

### Result

All 23 unit tests passed.

\[
23/23 \text{ tests passed}
\]

No experimental held-out results have yet been inspected.

### Decision

The baseline implementation is considered sufficiently validated to begin
corpus-level sanity checks and preliminary model runs.

Any later changes to core preprocessing, counting, probability estimation,
or evaluation logic must be documented and followed by rerunning the full
test suite.

## 2026-08-17 — Corpus preprocessing sanity check

### Objective

Verify the corpus-level outputs of the frozen preprocessing pipeline before
training any experimental language models.

### Corpus statistics

Total tokenised corpus size:

\[
N = 988{,}444
\]

Sequential split:

- Training: 790,755 tokens
- Validation: 98,844 tokens
- Test: 98,845 tokens

This corresponds approximately to the preregistered 80/10/10 split.

### Vocabulary

The vocabulary was constructed exclusively from the full training partition
using minimum training frequency:

\[
C(w) \geq 2.
\]

Frozen vocabulary size:

\[
|V| = 13{,}938
\]

including `<UNK>`.

### Unknown-token rates

- Training: 1.0419%
- Validation: 5.3731%
- Test: 4.3715%

The higher held-out `<UNK>` rates are expected because the vocabulary is
derived exclusively from training data and the corpus is split
sequentially. No vocabulary information from validation or test was used.

### Training blocks

The 790,755-token training partition produced 791 training blocks using an
approximately 1,000-token block size.

Nested training subsets:

| Fraction | Blocks | Tokens |
|---|---:|---:|
| 5% | 40 | 40,000 |
| 10% | 79 | 79,000 |
| 20% | 158 | 158,000 |
| 40% | 316 | 316,000 |
| 80% | 633 | 632,755 |
| 100% | 791 | 790,755 |

The 80% subset contains the final partial 755-token block because blocks were
shuffled before nested subsets were selected.

### Assessment

The corpus statistics are internally consistent and no obvious preprocessing
failure was detected.

The validation and test partitions exhibit higher `<UNK>` rates than the
training partition, which will be retained rather than adjusted because
changing the vocabulary using held-out data would introduce leakage.

### Next step

Run a single end-to-end smoke experiment using:

\[
m=20\%,\qquad n=3,\qquad \alpha=0.1
\]

before implementing the full validation sweep.

## 2026-08-17 — Real-data smoke test

### Objective

Verify the complete validated n-gram pipeline on one real Shakespeare model
before beginning the full validation experiment.

This run was intended only as an implementation and data sanity check.
It was not used to select hyperparameters or evaluate the research hypotheses.

### Condition

The preregistered smoke-test condition was:

\[
m=20\%,\qquad n=3,\qquad \alpha=0.1.
\]

Therefore the history length was:

\[
k=n-1=2.
\]

Training tokens:

\[
158{,}000.
\]

Validation tokens:

\[
98{,}844.
\]

### Training occupancy

The trigram training subset produced:

\[
N_2=157{,}684
\]

total history occurrences,

\[
T_2=85{,}635
\]

distinct observed histories, and

\[
f_1^{(2)}=67{,}647
\]

singleton histories.

Therefore:

\[
S_2
=
\frac{f_1^{(2)}}{N_2}
=
0.429004.
\]

The distinct-history rate was:

\[
D_2
=
\frac{T_2}{N_2}
=
0.543080.
\]

### Validation evaluation

Held-out validation cross-entropy was:

\[
H_{\mathrm{val}}
=
9.148831
\text{ nats/token}.
\]

Validation perplexity was:

\[
PP_{\mathrm{val}}
=
9403.445376.
\]

The actual validation unseen-history occurrence rate was:

\[
U_2
=
0.437995.
\]

The evaluator scored:

\[
N_{\mathrm{eval}}
=
98{,}839
\]

target tokens, exactly matching the expected shared-target evaluation count:

\[
98{,}844-5=98{,}839.
\]

### Manual inspection

The most common inspected histories and their successors were linguistically
and structurally plausible.

Examples included:

- `in the`
- `i am`
- `i have`

with plausible successor distributions such as `castle`, `palace`, `a`,
`not`, and `been`.

No evidence of artificial cross-block n-grams, malformed token IDs, or
incorrect history construction was observed.

### Sanity assessment

All real-data invariants passed:

- \(0 \leq S_2 \leq 1\)
- \(0 \leq D_2 \leq 1\)
- \(0 \leq U_2 \leq 1\)
- cross-entropy was finite
- perplexity was finite
- held-out target count was correct
- inspected histories and successors were plausible

The observed numerical proximity between

\[
S_2=0.429004
\]

and

\[
U_2=0.437995
\]

is noted but is not interpreted as evidence for the research hypothesis
because this was a single preregistered smoke-test condition.

### Decision

The real-data pipeline is approved for the full validation sweep.

The test partition remains untouched.

The next stage is to evaluate all preregistered combinations of training size,
model order, and add-\(\alpha\) smoothing strength on validation data only,
then freeze the selected smoothing parameter for every \((m,n)\) condition
before test evaluation.

## 2026-08-17 — Initial validation sweep

### Objective

Evaluate the preregistered add-alpha smoothing grid using validation data only
for every combination of training-corpus size and n-gram order.

The test partition remained untouched.

### Experimental grid

Training fractions:

\[
m\in\{5\%,10\%,20\%,40\%,80\%,100\%\}
\]

Model orders:

\[
n\in\{1,2,3,4,5,6\}
\]

Initial smoothing grid:

\[
\alpha\in\{0.001,0.01,0.1,1.0\}.
\]

Total validation conditions:

\[
6\times6\times4=144.
\]

### Result integrity

All 144 conditions completed successfully.

Each model was evaluated on exactly:

\[
98{,}839
\]

shared validation target tokens.

The sweep produced 36 provisional hyperparameter selections, one for every
\((m,n)\) condition.

No test-set performance was evaluated.

### Preliminary methodological observations

Training singleton-history rate behaved qualitatively as expected:

- \(S_k\) increased strongly with history length;
- \(S_k\) generally decreased as training size increased.

Validation unseen-history rate \(U_k\) showed a similar qualitative pattern.

These observations are treated as validation-stage sanity checks rather than
final hypothesis tests.

### Hyperparameter-grid issue

The initial grid did not adequately bracket the validation optima.

Most models with \(n\geq4\) selected:

\[
\alpha=0.001,
\]

the smallest tested value.

Several unigram conditions selected:

\[
\alpha=1.0,
\]

the largest tested value.

Therefore the current selections cannot yet be treated as properly frozen
validation optima because better values may lie outside the tested range.

### Decision

Before any test evaluation, expand the validation-only smoothing grid to:

\[
\alpha\in
\{0.0001,0.001,0.01,0.1,1.0,10.0\}.
\]

The validation sweep will be rerun using this expanded grid.

If the selected values are no longer systematically clipped at the grid
boundaries, the resulting hyperparameters will be frozen and committed before
the test set is evaluated.

## 2026-08-17 — Expanded validation sweep and hyperparameter freeze

### Objective

Resolve the boundary-selection problem identified during the initial
validation sweep before any use of the test partition.

The original smoothing grid was:

\[
\alpha\in\{0.001,0.01,0.1,1.0\}.
\]

Because several validation-optimal values occurred at the boundaries of this
grid, it was expanded before test evaluation.

### Expanded smoothing grid

The revised validation-only grid was:

\[
\alpha\in
\{0.0001,0.001,0.01,0.1,1.0,10.0\}.
\]

Across:

\[
6
\]

training sizes,

\[
6
\]

model orders, and

\[
6
\]

smoothing strengths, the expanded sweep evaluated:

\[
6\times6\times6=216
\]

validation conditions.

### Validation integrity

All 216 conditions completed successfully.

Each condition was evaluated on exactly:

\[
N_{\mathrm{val}}=98{,}839
\]

shared validation target positions.

The full sweep produced:

- 216 validation-result rows;
- 36 selected \((m,n)\) hyperparameter configurations.

The test partition remained completely untouched.

The full unit-test suite was also rerun before the expanded experiment:

\[
23/23
\]

tests passed.

### Selected smoothing pattern

The expanded grid produced a stable order-dependent pattern.

Broadly:

- unigram models preferred \(\alpha\approx1\);
- bigram models preferred \(\alpha=0.01\);
- trigram models preferred \(\alpha=0.001\) or \(0.01\);
- four-gram and five-gram models preferred \(\alpha=0.001\);
- six-gram models preferred \(\alpha=0.0001\).

The previous upper-boundary issue was therefore resolved.

Six-gram models continued to select the smallest tested value,

\[
\alpha=0.0001.
\]

However, the validation improvement relative to

\[
\alpha=0.001
\]

was less than approximately

\[
0.001\text{ nats/token}
\]

for every training-size condition.

Because this difference is very small and continued expansion would provide
little practical information while increasing hyperparameter search
flexibility, the smoothing grid will not be expanded further.

### Decision

The expanded grid is frozen as the final smoothing search space:

\[
\boxed{
\alpha\in
\{0.0001,0.001,0.01,0.1,1.0,10.0\}
}
\]

For every training fraction \(m\) and model order \(n\), the selected
hyperparameter is:

\[
\alpha^*(m,n)
=
\arg\min_{\alpha}
H_{\mathrm{val}}(m,n,\alpha).
\]

These 36 hyperparameter selections are now frozen.

No changes to smoothing parameters, model orders, training fractions,
vocabulary construction, tokenisation, or evaluation alignment will be made
in response to subsequent test-set performance.

### Next step

Commit the frozen validation results and selected hyperparameters.

Only after this commit will the test partition be evaluated using exactly one
preselected model for each of the 36 \((m,n)\) conditions.

## 2026-08-17 — Locked primary test evaluation

### Objective

Evaluate the final validation-selected n-gram models on the held-out test
partition exactly once, without any further hyperparameter selection.

### Frozen experimental state

The final smoothing parameters were selected using validation data only and
committed before test evaluation.

SHA-256 of the frozen hyperparameter-selection file:

`a04a261d18fa30c35e5291759bc751802ccddb75d260c22e639b71d5ef7a1d19`

The test evaluation used exactly one frozen smoothing parameter for each of
the 36 combinations of training size and n-gram order.

### Test evaluation

Number of final model conditions:

\[
36
\]

Shared held-out target positions per model:

\[
N_{\mathrm{test}}=98{,}840.
\]

All models were evaluated on exactly the same target positions.

No hyperparameters or methodological choices were changed after observing
test performance.

### Transition dataset

For every training size, consecutive model orders were compared using

\[
\Delta H_n
=
H_n-H_{n-1}.
\]

With six training sizes and five order transitions, this produced:

\[
6\times5=30
\]

transition observations.

### Initial structural observation

All 30 observed transitions had

\[
\Delta H_n>0.
\]

Thus, under the tested fixed-order add-alpha estimator, every increase in
model order increased held-out cross-entropy for the corresponding test
condition.

This observation will be analysed formally in the next stage rather than
used to modify the experimental methodology.

### Data integrity

The final outputs were written to:

- `results/main_results.csv`
- `results/transitions.csv`

The test set is now considered consumed for the primary experiment.

No subsequent methodological or hyperparameter changes will be made in
response to these test results.

### H3 calibration against held-out unseen-history rate

Although \(S_k\) and the distinct-history baseline \(D_k\) produced identical
Spearman rank correlations with \(U_k\), they differed in numerical
calibration.

Across all 30 nontrivial \((m,k)\) conditions:

\[
\operatorname{MAE}(S_k,U_k)=0.01995
\]

compared with:

\[
\operatorname{MAE}(D_k,U_k)=0.02737.
\]

Similarly:

\[
\operatorname{RMSE}(S_k,U_k)=0.02436
\]

compared with:

\[
\operatorname{RMSE}(D_k,U_k)=0.04007.
\]

The advantage of \(S_k\) was concentrated primarily at shorter history
lengths \(k=1,2\). At higher history lengths \(k=3,4,5\), \(D_k\) was
numerically closer to the held-out unseen-history rate.

Therefore the evidence supports strong coverage-tracking ability for both
training-only occupancy diagnostics, with better pooled calibration for
\(S_k\), but does not support claiming that singleton-history rate uniquely
dominates the simpler distinct-history baseline.

### H4 final primary result

Transition-specific Spearman correlations between training-only
singleton-history rate and marginal held-out loss were:

\[
\rho_{1\to2}=1.000
\]

\[
\rho_{2\to3}=-0.943
\]

\[
\rho_{3\to4}=-1.000
\]

\[
\rho_{4\to5}=-1.000
\]

\[
\rho_{5\to6}=-1.000.
\]

The median transition-specific correlation was:

\[
\boxed{-1.000}.
\]

This is opposite to the preregistered directional expectation for four of
five transitions.

Because training size and \(S_k\) are nearly monotonically coupled within
each transition, these correlations do not establish predictive information
from \(S_k\) beyond training-set size.

## 2026-08-17 — Paired block-bootstrap uncertainty analysis

### Objective

Quantify uncertainty in the 30 observed marginal held-out loss differences

\[
\Delta H_n = H_n-H_{n-1}
\]

without changing any frozen models or hyperparameters.

Because adjacent models were evaluated on exactly the same held-out target
positions, uncertainty was estimated from paired per-target loss differences:

\[
d_t =
\ell_{n,t}-\ell_{n-1,t}.
\]

### Frozen design

The bootstrap design was committed before intervals were inspected.

Primary configuration:

- method: paired circular moving-block bootstrap;
- random seed: 42;
- bootstrap replicates: 5,000;
- primary block length: 1,000 held-out target positions;
- confidence level: 95%;
- sensitivity block lengths: 500 and 2,000.

The frozen hyperparameter artifact was verified against SHA-256:

`a04a261d18fa30c35e5291759bc751802ccddb75d260c22e639b71d5ef7a1d19`

All reconstructed per-target losses reproduced the previously frozen
test cross-entropies and transition-level Delta H values before
bootstrapping.

### Primary result

For the primary block length:

\[
L=1000,
\]

all:

\[
30/30
\]

95% percentile bootstrap intervals were entirely above zero.

No interval contained zero and no interval was entirely below zero.

Thus all observed positive marginal-loss effects were stable under the
primary held-out block-resampling procedure.

### Sensitivity analysis

The same qualitative result held for both alternative block lengths:

\[
L=500:
\quad 30/30\text{ intervals entirely above zero}
\]

and

\[
L=2000:
\quad 30/30\text{ intervals entirely above zero}.
\]

Thus the sign-level conclusion was insensitive to the tested block-length
choices.

### Small-effect examples

The smallest observed marginal loss occurred for the 5% training condition
on the \(5\to6\) transition:

\[
\Delta H_6=0.00551
\]

with primary 95% interval:

\[
[0.00333,\ 0.00835].
\]

For the full-data \(1\to2\) transition:

\[
\Delta H_2=0.05089
\]

with primary 95% interval:

\[
[0.01294,\ 0.08749].
\]

Although some individual bootstrap replicates for the smallest-margin
conditions were non-positive, their 95% percentile intervals remained
strictly above zero.

### Interpretation limit

These intervals quantify uncertainty produced by resampling contiguous
blocks of the frozen held-out sequence.

They do not include uncertainty from:

- alternative training-set samples;
- corpus choice;
- preprocessing decisions;
- vocabulary construction;
- hyperparameter selection;
- smoothing-estimator choice.

Therefore this analysis strengthens the within-experiment evidence for
generalisation reversals but does not establish that increasing context is
universally harmful.

### Decision

The uncertainty analysis is frozen.

The next stage will test whether the observed reversals remain under a
preplanned lower-order interpolation estimator, addressing the possibility
that the result is specific to fixed-order add-alpha smoothing.

## 2026-08-17 — Validation-selected lower-order interpolation

### Objective

Test whether the generalisation reversals observed under fixed-order
add-alpha estimation are sensitive to using lower-order interpolation.

For each transition

\[
n-1\rightarrow n,
\]

the interpolated distribution is

\[
q_{\lambda,n}
=
(1-\lambda)q_{n-1}
+
\lambda q_n,
\]

where \(\lambda\) is the weight assigned to the higher-order model.

The component models retained their previously frozen validation-selected
add-alpha smoothing parameters.

### Frozen interpolation grid

The validation-only interpolation grid was:

\[
\lambda\in
\{0.0,0.1,0.2,\ldots,0.9,1.0\}.
\]

Across six training sizes and five order transitions this produced:

\[
6\times5\times11=330
\]

validation conditions.

Every condition was evaluated on exactly:

\[
N_{\mathrm{val}}=98{,}839
\]

shared validation target positions.

The test set was not evaluated during this stage.

### Selection

One lambda was selected for each of the 30
(training-size, order-transition) conditions by minimum validation
cross-entropy.

Exact ties were resolved in favor of the smaller higher-order weight.

Selected-weight distribution:

- \(\lambda=0\): 3/30
- \(0<\lambda<1\): 27/30
- \(\lambda=1\): 0/30

Thus validation never preferred using the higher-order model alone.

The selected interpolation artifact has SHA-256:

`2e7fc62042dab3a024194555ff2f7ebb91a1efd81c5cf3b8ad43676f1029587c`

### Interpretation note

Because lambda=0 was included in the candidate grid, the selected
interpolated model cannot be worse than the lower-order model on the same
validation data.

Therefore validation improvements relative to the lower-order model are a
model-selection property and are not treated as evidence of generalisation.

The relevant robustness question is whether validation-selected interpolation
improves or preserves performance on the held-out test sequence.

### Decision

The 30 interpolation weights are frozen.

No interpolation weights will be modified in response to test performance.

## 2026-08-17 — Locked interpolation robustness test

### Objective

Evaluate the 30 validation-selected lower-order interpolation weights on the
held-out test sequence without any test-driven changes.

The interpolated model for transition \(n-1\rightarrow n\) was

\[
q_{\lambda,n}
=
(1-\lambda)q_{n-1}
+
\lambda q_n,
\]

where the component models retained their previously frozen add-alpha
parameters and lambda was selected using validation data only.

### Frozen artifacts

Primary hyperparameter SHA-256:

`a04a261d18fa30c35e5291759bc751802ccddb75d260c22e639b71d5ef7a1d19`

Interpolation-selection SHA-256:

`2e7fc62042dab3a024194555ff2f7ebb91a1efd81c5cf3b8ad43676f1029587c`

Each interpolated model was evaluated on the same:

\[
N_{\mathrm{test}}=98{,}840
\]

held-out target positions used in the primary experiment.

### Main robustness result

Under the original fixed-order add-alpha models:

\[
30/30
\]

order increases had:

\[
\Delta H_n>0.
\]

After validation-selected lower-order interpolation:

- 27/30 transitions had \(\Delta H_n^{\mathrm{interp}}<0\);
- 3/30 transitions had \(\Delta H_n^{\mathrm{interp}}=0\);
- 0/30 transitions had \(\Delta H_n^{\mathrm{interp}}>0\).

The three exact-zero transitions corresponded to validation selections with

\[
\lambda=0,
\]

so the interpolated model was exactly identical to the lower-order model in
those cases.

Interpolation improved test cross-entropy relative to the original
higher-order fixed-order model in:

\[
30/30
\]

conditions.

### Interpretation

The universal generalisation reversals observed in the primary fixed-order
experiment are therefore not robust to estimator choice.

Higher-order context was harmful when sparse higher-order models were forced
to predict using fixed-order add-alpha smoothing, but validation-selected
lower-order interpolation either improved on the lower-order baseline or
reduced exactly to it in every tested condition.

This suggests that the original reversals primarily reflect finite-sample
estimation and sparse-context handling rather than evidence that additional
context itself is intrinsically harmful.

### Remaining uncertainty

Several interpolated improvements are very small, particularly for the
highest-order transitions.

The preplanned next step is therefore paired moving-block bootstrap
uncertainty analysis of the frozen interpolated loss differences using the
same bootstrap design as the primary analysis.

No interpolation weights or other experimental choices will be modified.