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