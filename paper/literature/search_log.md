## Literature Search Log

### 2026-08-17 — Core-paper verification

Core prior work verified:

- Good (1953): frequency-of-frequency estimation and singleton/missing-mass motivation.
- Katz (1987): sparse m-gram probability estimation and recursive backoff.
- Kneser & Ney (1995): improved lower-order backoff distributions.
- MacKay & Peto (1995): hierarchical probabilistic interpretation of smoothing.
- Chen & Goodman (1996): systematic smoothing comparison over training size, corpus and n-gram order.
- Pereira, Singer & Tishby (1995): prediction suffix trees and variable-order mixtures.
- Saul & Pereira (1997): mixed-order models inserted between conventional n-gram orders.
- McAllester & Schapire (2000): convergence theory for Good-Turing singleton/missing-mass estimators.
- Chandra, Thangaraj & Rajaraman (2021/2024): Good-Turing under Markov dependence.

### Layer A — n-gram order versus training-data size

Representative searches:

```text
higher order n-gram training data size cross entropy model order
n-gram model order held-out perplexity
optimal n-gram order sparse data
model order training size n-gram cross entropy
```

#### Important findings

Chen & Goodman already systematically vary training size and model order when evaluating smoothing. 

Takahashi & Tanaka-Ishii explicitly map n-gram context/order against training-data size and show that the best order increases with available data. 

Siivola & Pellom similarly report that smaller data makes higher-order n-grams less useful. 

#### Conclusion

**The broad phenomenon is established.**

Athena must not claim novelty for demonstrating that high-order n-grams can generalise poorly with insufficient data.

---

### Layer B — held-out coverage versus performance

Representative searches:

```text
n-gram coverage perplexity
unseen context rate language model perplexity
n-gram hit rate model order perplexity
context coverage higher-order language model
```

#### Important findings

Saul & Pereira explicitly isolate predictions involving unseen word combinations and measure how alternate backoff models affect their perplexity. 

Classical smoothing literature also directly treats unseen events as a central source of estimation failure. 

#### Current conclusion

Coverage/performance relationships are clearly established in broad form.

However, this search pass has **not located an exact prior analysis** using:

\[
S_k
=
f_1^{(k)}/N_k
\]

as a **training-only predictor of held-out history occurrence coverage \(U_k\)** across fixed history lengths and training sizes.

Status:

\[
\boxed{\text{possible narrow contribution, not yet proven novel}}
\]

---

### Layer C — singleton-history statistics

Representative searches:

```text
singleton counts n-gram language model
frequency of frequencies context language model
Good-Turing history counts n-gram
singleton context language model
missing mass n-gram history
singleton-history language model
```

#### Findings

Good-Turing and later missing-mass theory establish the singleton fraction as a missing-mass statistic under appropriate sampling assumptions. 

Markov-dependent variants have also been studied theoretically. 

#### Current conclusion

I did **not** locate, in this pass, a paper centered on Athena's exact statistic:

\[
S_k
=
\frac{\#\{\text{distinct length-}k\text{ histories occurring once}\}}
{\#\{\text{length-}k\text{ training occurrences}\}}
\]

and its association with adjacent-order held-out marginal loss.

That absence is **not proof of novelty**.

Next search should citation-chase papers that discuss:

- count-of-counts by context order;
- n-gram history occupancy;
- unseen context prediction;
- model-order selection statistics.

---

### Layer D — fixed versus interpolated/mixed order

Representative searches:

```text
fixed order versus interpolated n-gram
mixed-order Markov perplexity
variable order n-gram sparse data
interpolation higher order context language model
```

#### Findings

This area is heavily established.

Pereira et al. use variable-order prediction suffix trees and mixtures. 

Saul & Pereira directly construct mixed-order models between ordinary n-gram orders. 

Chen & Goodman show the importance of interpolation and smoothing choice across order/data regimes. 

Chelba et al. later show that the ability to exploit longer n-gram contexts depends strongly on the estimator used. 

#### Conclusion

Athena's interpolation method itself is **not a contribution**.

The possible contribution is the controlled diagnostic contrast:

\[
\text{universal fixed-order reversals}
\]

followed by:

\[
\text{no clear harms under frozen validation-selected interpolation},
\]

combined with the occupancy analysis.

---