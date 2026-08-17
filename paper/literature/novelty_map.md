
This is the most important file, frankly.

| Claim / idea | Status after first search | Closest prior work | Athena's defensible addition |
|---|---|---|---|
| Higher-order n-grams become sparse | **Established** | Katz; Chen & Goodman; Kneser-Ney | None |
| High order can hurt with insufficient data | **Established** | Chen & Goodman; Takahashi & Tanaka-Ishii; Siivola & Pellom | Broader replication only |
| Best n-gram order depends on training size | **Established** | Takahashi & Tanaka-Ishii; Chen & Goodman | None as a broad claim |
| Additive smoothing is weak for sparse n-grams | **Established** | Chen & Goodman | None |
| Lower-order backoff/interpolation mitigates sparsity | **Established** | Katz; Kneser-Ney; Chen & Goodman | None |
| Variable/mixed order can beat globally fixed order | **Established** | Pereira et al.; Saul & Pereira | None |
| Singleton fraction relates to missing mass | **Established** | Good; McAllester & Schapire | None |
| Good-Turing under dependent samples needs special treatment | **Established** | Chandra et al. | None |
| \(S_k\) strongly tracks held-out unseen-history occurrence \(U_k\) | **Potentially narrower empirical contribution** | Good-Turing/missing-mass work is close, but not exact | Controlled measurement across \(k\) and \(m\) |
| \(S_k\) is better calibrated than \(D_k\) overall for \(U_k\) | **Potentially narrow empirical result** | No exact match found yet | Baseline comparison |
| \(S_k\) predicts \(\Delta H_n\) in the expected positive direction | **Our hypothesis was falsified** | No exact diagnostic found yet | Negative result |
| Coverage prediction does not imply marginal-context-value prediction | **Potentially interesting contribution** | Related smoothing literature, no exact formulation found yet | Empirical separation of \(S_k\to U_k\) from \(S_k\to\Delta H_n\) |
| Fixed add-\(\alpha\) reversals disappear under interpolation | **Empirical extension, not new method** | Katz; Chen & Goodman; Saul & Pereira; Chelba et al. | Frozen controlled before/after experiment |
| \(26/27\) nonzero interpolations show clear primary-bootstrap benefit | **Experiment-specific result** | Related mixed-order literature | Quantitative robustness result on our protocol |

---