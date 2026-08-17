## Literature Review: Context Sparsity, Missing Mass, and Estimation in N-Gram Language Models

### Purpose

This review examines prior work relevant to the following questions:

1. How does increasing n-gram order interact with finite training data and sparsity?
2. What does singleton frequency measure, particularly in relation to missing probability mass?
3. Can training occupancy statistics be expected to predict held-out context coverage?
4. Is poor higher-order generalisation fundamentally caused by additional context, or by the estimator used to exploit that context?
5. Has prior work directly connected a training-only singleton-history statistic to the **marginal held-out benefit of increasing n-gram order**?

The review distinguishes established ideas from the narrower empirical questions investigated in Project Athena.

---

## Good (1953): *The Population Frequencies of Species and the Estimation of Population Parameters*

### Bibliographic information

**Author:** I. J. Good  
**Year:** 1953  
**Venue:** *Biometrika*, 40(3–4), 237–264  
**DOI:** 10.1093/biomet/40.3-4.237

### Problem

Good studies estimation in large discrete populations when many categories are rare or absent from a finite sample.

The setup uses the **frequency of frequencies**:

\[
n_r
=
\#\{\text{types observed exactly }r\text{ times}\}.
\]

The paper explicitly notes literary vocabulary as an application. 

### Method

Rather than treating the empirical frequency

\[
\frac{r}{N}
\]

as reliable when \(r\) is small, Good uses statistics describing how many categories occur once, twice, and so forth.

The classical Good-Turing adjusted-count relation is:

\[
r^*
=
(r+1)\frac{n_{r+1}}{n_r}.
\]

The singleton frequency plays a particularly important role in estimating probability associated with categories not represented in the sample. Chen & Goodman also summarize this frequency-of-frequency construction in the language-model setting. 

### Relevant definition for Athena

Athena defines:

\[
S_k
=
\frac{f_1^{(k)}}{N_k},
\]

where:

- \(N_k\) is the number of observed training occurrences of histories of length \(k\);
- \(f_1^{(k)}\) is the number of distinct histories occurring exactly once.

This has the same structural form as a Good-Turing singleton statistic.

### Main relevance

Good establishes the statistical motivation for using singleton frequency as information about unseen probability mass.

### Relation to Athena

Athena borrows the **occupancy intuition**, not a new Good-Turing estimator.

The hypothesis is that a high fraction of singleton histories may indicate that the space of possible contexts is poorly covered.

### Difference from Athena

Good's theoretical setup treats observations using a sampling model appropriate to species/frequency estimation. Athena's \(k\)-token histories:

- overlap;
- are sequentially dependent;
- come from natural language;
- are not iid samples of history types.

Therefore:

\[
S_k
\]

must not automatically be identified with a true missing-mass estimator.

Athena uses it as a **Good-Turing-motivated training-only sparsity diagnostic**.

### Threat to novelty

**High** for any claim that singleton frequency or frequency-of-frequency estimation is new.

**Low** for Athena's exact empirical question relating history occupancy to adjacent-order marginal held-out loss.

### Manuscript citation use

Cite for:

- frequency-of-frequency estimation;
- singleton/missing-mass motivation;
- statistical origin of \(S_k\).

### Open question

How far does the singleton/missing-mass relationship survive when observations are dependent rather than iid?

That is addressed much more directly by Chandra et al.

---

## Katz (1987): *Estimation of Probabilities from Sparse Data for the Language Model Component of a Speech Recognizer*

### Bibliographic information

**Author:** Slava M. Katz  
**Year:** 1987  
**Venue:** *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 35(3)  
**DOI:** 10.1109/TASSP.1987.1165125

### Problem

Katz addresses exactly the classical sparse-data problem in m-gram language models: many combinations that can occur in language are rare or absent from the finite training corpus. The paper proposes a recursive probability-estimation procedure designed for this setting. 

### Method

The central idea is **backoff**.

When sufficient high-order evidence exists, the model uses it. When a high-order event is unseen or unreliable, probability estimation falls back to a less specific lower-order distribution.

Katz incorporates Good-Turing-style discounting and lower-order probability information into a recursive estimator. Chen & Goodman identify Katz smoothing as one of the principal techniques developed to combine information from different orders under sparse data. 

### Main result relevant to Athena

Sparse high-order counts should not simply be trusted as independent estimates.

Reliable lower-order information can be used when high-order observations provide insufficient evidence.

### Relation to Athena

Athena's primary add-\(\alpha\) experiment deliberately used a simple fixed-order estimator.

That produced:

\[
30/30
\]

higher-order generalisation reversals.

The interpolation experiment then allowed lower-order evidence to contribute and eliminated every observed penalty.

This places Athena directly in the statistical problem Katz addressed.

### Difference from Athena

Athena does **not** propose a replacement for Katz backoff.

Instead it asks a diagnostic question:

> What happens to observed order reversals, occupancy statistics, and marginal context value under a deliberately controlled fixed-order estimator, and what changes when lower-order information is restored?

### Threat to novelty

**Very high** to any claim that backing off under sparse context is novel.

**Moderate** to the estimator-dependence result broadly.

**Low/uncertain** to the exact \(S_k \rightarrow \Delta H_n\) diagnostic experiment.

### Manuscript citation use

Cite for:

- sparse n-gram estimation;
- recursive lower-order backoff;
- historical motivation for estimator robustness.

---

## Kneser & Ney (1995): *Improved Backing-Off for M-Gram Language Modeling*

### Bibliographic information

**Authors:** Reinhard Kneser and Hermann Ney  
**Year:** 1995  
**Venue:** ICASSP  
**Pages:** 181–184

The RWTH Aachen publication archive confirms the original conference record. 

### Problem

Traditional backoff does not merely face the question of **whether** to use a lower-order distribution.

There is also the question:

> What should that lower-order distribution actually represent when it is being used as a backoff model?

### Method

Kneser-Ney modifies the lower-order distribution so that it is specifically appropriate for redistribution/backoff rather than simply reproducing ordinary lower-order maximum-likelihood frequencies.

### Relation to Athena

The paper reinforces a major lesson from Athena's interpolation experiment:

\[
\boxed{\text{estimator design matters}}
\]

Two models may use nominally the same context information while generalising very differently depending on how sparse contexts are handled.

### Difference from Athena

Athena does not implement Kneser-Ney in the primary experiment.

Its robustness method is deliberately simpler:

\[
q_n^{\mathrm{interp}}
=
(1-\lambda)q_{n-1}
+
\lambda q_n.
\]

This makes the causal comparison easier to interpret experimentally, but it is not a competitive smoothing algorithm.

### Threat to novelty

**Very high** to any claim that using lower-order evidence to repair sparse higher-order models is new.

**Low** to the occupancy/marginal-value diagnostic.

### Manuscript citation use

Cite when discussing:

- sophisticated alternatives to naive add-\(\alpha\);
- why lower-order probability construction matters;
- limitations of Athena's deliberately simple estimator.

---

## MacKay & Peto (1995): *A Hierarchical Dirichlet Language Model*

### Bibliographic information

**Authors:** David J. C. MacKay and Linda C. Bauman Peto  
**Year:** 1995  
**Venue:** *Natural Language Engineering*, 1(3), 289–308  
**DOI:** 10.1017/S1351324900000218

Cambridge records the article in the September 1995 issue. 

### Problem

The paper asks whether smoothing can be understood through a coherent probabilistic hierarchical model rather than merely as an engineering correction to empirical frequencies.

### Method

MacKay and Peto construct a hierarchical probabilistic model based on Dirichlet distributions whose predictions behave similarly to language-model smoothing.

The hierarchy permits information sharing rather than estimating each conditional distribution in isolation.

### Main result

The hierarchical model was compared with smoothing on a corpus of approximately two million words and achieved roughly comparable prediction accuracy while using fewer computational resources. 

### Relation to Athena

The statistical interpretation is highly relevant.

Athena's fixed-order models treat increasingly specific history-conditioned distributions as increasingly separate estimation problems.

Hierarchical or interpolated methods instead allow evidence to be shared between levels.

### Difference from Athena

Athena is not proposing a Bayesian hierarchy.

The experiment instead demonstrates empirically how dramatically the apparent value of longer context can change when lower-order information is permitted to contribute.

### Threat to novelty

**Medium** to broad claims about statistical sharing solving sparse estimation.

**Low** to Athena's exact occupancy diagnostic.

### Manuscript citation use

Useful for the Discussion section:

> the estimator failure observed under fixed-order add-\(\alpha\) can be understood as failure to share statistical strength between related conditional distributions.

---

## Chen & Goodman (1996): *An Empirical Study of Smoothing Techniques for Language Modeling*

### Bibliographic information

**Authors:** Stanley F. Chen and Joshua Goodman  
**Year:** 1996  
**Venue:** 34th Annual Meeting of the ACL  
**Pages:** 310–318  
**ACL ID:** P96-1041  
**DOI:** 10.3115/981863.981904 


### Problem

Chen & Goodman conduct a systematic empirical comparison of n-gram smoothing methods.

This is probably the **most important novelty-threat paper** for Athena.

### Experimental dimensions

Their study explicitly varies:

- smoothing algorithm;
- training-data size;
- corpus;
- n-gram order;
- smoothing hyperparameters.

Performance is evaluated using held-out/test cross-entropy. 

### Key observation

They find that the relative quality of smoothing methods depends strongly on:

\[
\text{training size}
\]

and:

\[
\text{model order}.
\]

They also emphasize that ordinary maximum-likelihood estimates become poor when training data are small relative to the size of the model being estimated. 

### Additive smoothing

Their discussion is especially relevant because Athena's primary estimator is add-\(\alpha\).

Chen & Goodman note serious deficiencies of additive smoothing and explain why lower-order interpolation can produce more realistic predictions than assigning unseen events probabilities without considering lower-order frequency structure. 

### Good-Turing

They also make an important distinction:

Good-Turing frequency adjustment alone is not sufficient for high-quality n-gram smoothing because good language-model estimation also requires combining lower- and higher-order information. 

### Relation to Athena

Strong overlap exists with:

\[
(\text{training size},\text{model order})
\rightarrow
H_{\mathrm{test}}.
\]

Therefore Athena **cannot claim novelty merely for studying how order and training size interact**.

### Difference from Athena

Their primary question is:

> Which smoothing algorithms perform best under different circumstances?

Athena's narrower diagnostic question is:

\[
S_k
\longrightarrow
U_k
\]

versus:

\[
S_k
\longrightarrow
\Delta H_n.
\]

Athena additionally:

- considers orders \(1\) through \(6\), not merely bigram/trigram comparison;
- explicitly measures held-out unseen-history occurrence rate \(U_k\);
- studies training history occupancy \(S_k\);
- uses adjacent-order marginal loss
  \[
  \Delta H_n=H_n-H_{n-1};
  \]
- finds a sharp separation between predicting **coverage** and predicting **marginal context value**;
- performs a frozen fixed-estimator versus interpolation comparison.

### Threat to novelty

\[
\boxed{\textbf{HIGH}}
\]

Any novelty statement must explicitly distinguish Athena from Chen & Goodman.

### Manuscript citation use

This paper should appear repeatedly in:

- Background;
- Related Work;
- Methods justification;
- Discussion;
- Limitations.

---

## Pereira, Singer & Tishby (1995): *Beyond Word N-Grams*

### Bibliographic information

**Authors:** Fernando Pereira, Yoram Singer, Naftali Tishby  
**Year:** **1995 conference version**  
**Venue:** Third Workshop on Very Large Corpora  
**ACL ID:** W95-0108

The official ACL version is from 1995. A later arXiv version appeared in 1996, which explains the date ambiguity from our earlier discussion. 

### Problem

A globally fixed context length creates two opposing failures:

- low orders cannot represent useful longer-range dependencies;
- high orders cause model size and sparsity to explode.

### Method

The paper uses **prediction suffix trees (PSTs)**.

Instead of requiring every prediction to use exactly the same context length, context can be extended adaptively when additional history improves prediction.

The authors also construct mixtures over possible prediction suffix trees. 

### Main result

The mixture formulation performs better than relying on a single selected tree, and compact PST mixture models achieve competitive prediction performance on several corpora. 

### Relation to Athena

This is directly relevant to one of our final interpretations:

> globally forcing every prediction to use the same amount of context is statistically crude.

Athena's interpolation result likewise indicates that additional context should not necessarily be trusted with weight \(1\).

### Difference from Athena

PSTs solve an adaptive modeling problem.

Athena performs a controlled diagnostic experiment in fixed-order models.

Athena is therefore **not proposing variable-order language modeling**.

### Threat to novelty

**High** to broad claims that “using different amounts of context adaptively is better than fixed order.”

**Low** to the singleton-history diagnostic question.

---

## Saul & Pereira (1997): *Aggregate and Mixed-Order Markov Models for Statistical Language Processing*

### Bibliographic information

**Authors:** Lawrence Saul and Fernando Pereira  
**Year:** 1997  
**Venue:** Second Conference on Empirical Methods in Natural Language Processing  
**ACL ID:** W97-0309 


### Problem

The authors explicitly investigate models whose size and predictive accuracy fall **between conventional n-gram orders**.

They note that the parameter space of n-gram models grows extremely rapidly with order and that sparse estimation becomes especially severe for large vocabularies. 

### Method

They introduce:

- aggregate Markov models;
- mixed-order Markov models.

These models are inserted between ordinary n-gram orders inside a smoothing/backoff system.

### Important empirical result

Their mixed-order backoff model substantially reduces perplexity on previously unseen combinations. In the reported experiment, only a subset of predictions required backing off, yet the improved backoff model also reduced overall test perplexity substantially. 

### Relation to Athena

This is extremely close conceptually to Athena's final robustness result.

Athena observed:

\[
\text{fixed high order}
\quad\rightarrow\quad
\text{large penalty}
\]

while allowing mixed lower/higher information produced:

\[
\text{benefit or fallback}.
\]

### Difference from Athena

Saul & Pereira design better intermediate models.

Athena's contribution is not a new mixed-order estimator.

Instead, Athena uses a deliberately simple interpolation as a **robustness intervention** to diagnose why the fixed-order reversal occurred.

### Threat to novelty

**High** to any “mixed-order estimation solves higher-order sparsity” novelty claim.

**Low/medium** to our controlled before/after diagnostic design.

---

## McAllester & Schapire (2000): *On the Convergence Rate of Good-Turing Estimators*

### Bibliographic information

**Authors:** David McAllester and Robert E. Schapire  
**Preliminary version:** COLT 2000  
**Later draft:** 2001

The authors' paper explicitly states that its preliminary version appeared at COLT 2000. 

### Problem

Good-Turing was widely used, but its finite-sample convergence behavior required stronger theoretical characterization.

### Missing mass

Let:

\[
M_0
\]

be the total true probability assigned to outcomes not represented in the sample.

Let:

\[
G_0
\]

be the fraction of sample observations belonging to types that appear exactly once.

Then:

\[
G_0
=
\frac{f_1}{N}.
\]

McAllester & Schapire describe the singleton fraction as the Good-Turing missing-mass estimate and derive high-probability convergence bounds. 

### Assumption

Their core derivation considers a sample constructed by **independent draws** from an unknown discrete distribution. 

This matters enormously for Athena.

### Relation to Athena

The form:

\[
S_k=\frac{f_1^{(k)}}{N_k}
\]

is mathematically analogous to \(G_0\) if history types were sampled iid.

### Difference from Athena

Our history occurrences are dependent and overlapping.

Therefore the theorem does not directly justify interpreting:

\[
S_k
\]

as an unbiased or concentrated estimator of test unseen-history rate.

### Threat to novelty

**Very high** to any statistical novelty claim about the singleton fraction itself.

**Very useful** for defending why \(S_k\) is theoretically motivated.

### Manuscript citation use

This should support the exact wording:

> “We use a singleton-history statistic motivated by Good-Turing missing-mass estimation, rather than treating it as a theoretically valid missing-mass estimator for overlapping language histories.”

---

## Chandra, Thangaraj & Rajaraman: *How good is Good-Turing for Markov samples?*

### Bibliographic information

**Authors:** Prafulla Chandra, Andrew Thangaraj, Nived Rajaraman  
**Initial preprint:** 2021  
**Accepted:** TMLR, 2024 


### Problem

The paper directly attacks the assumption problem above.

Good-Turing's classic guarantees are most straightforward under iid sampling, while applications such as language contain temporal dependence.

### Method

The authors study missing-mass estimation when observations come from a Markov chain.

They analyze the relationship among:

- stationary probabilities;
- transition structure;
- spectral properties;
- convergence of the Good-Turing estimator.

They also investigate transition matrices derived from real text corpora including New York Times and Charles Dickens data. 

### Main relevance

Good-Turing-style singleton estimation can remain meaningful under some dependent processes, but its behavior depends on the dependence structure.

There is no license here to pretend dependency does not matter.

### Relation to Athena

Natural-language history sequences are temporally dependent.

This paper gives us a principled citation for saying:

> the iid Good-Turing interpretation cannot automatically be transferred to our history-occurrence process.

### Difference from Athena

Their target is **missing stationary mass under Markov sampling**.

Athena's \(U_k\) is:

\[
U_k
=
\frac{\text{held-out history occurrences unseen in training}}
{\text{held-out target occurrences}},
\]

and the objects being counted are overlapping length-\(k\) histories.

Those are related but not identical quantities.

### Threat to novelty

**High** to claims that Athena newly studies Good-Turing under dependent language data.

**Low** to the specific \(S_k,U_k,\Delta H_n\) relationship.

### Manuscript citation use

Essential for:

- Good-Turing caveat;
- dependence limitation;
- discussion of why \(\rho(S_k,U_k)=1\) is an empirical observation rather than a theorem.

---

# Additional high-priority papers found by the novelty search

These were **not** in our original nine. They now need to be read because they are unusually close to parts of Athena.

---

## Takahashi & Tanaka-Ishii (2018): *Cross Entropy of Neural Language Models at Infinity—A New Bound of the Entropy Rate*

### Why this is important

This paper explicitly varies both:

\[
\text{training-data size}
\]

and:

\[
\text{context length / n-gram order}.
\]

For n-gram models, they evaluate orders across a much wider range than bigram/trigram and show that the best-performing order rises as more training data becomes available. For example, their reported character-level experiments find a lower best \(n\) with small data and a higher best \(n\) with much larger data. 

### Novelty consequence

This substantially weakens any claim that Athena newly demonstrates:

> “higher n-gram order can become harmful under limited data.”

That phenomenon is established.

### What they apparently do not provide

From the sections located in this search pass, they do not appear to center:

\[
S_k=\frac{f_1}{N}
\]

as a training-only context diagnostic, nor the distinction:

\[
S_k\rightarrow U_k
\]

versus:

\[
S_k\rightarrow\Delta H_n.
\]

That is where our narrower question remains interesting.

### Threat to novelty

\[
\boxed{\textbf{VERY HIGH}}
\]

for the original title/question.

---

## Chelba, Norouzi & Bengio (2017): *N-gram Language Modeling using Recurrent Neural Network Estimation*

This paper is another important estimator-dependence reference.

Their experiments report that classical Katz/Kneser-Ney backoff estimators do not simply continue improving as n-gram order grows, while their LSTM-based smoothing can exploit longer n-gram contexts more effectively. 

### Novelty consequence

The proposition:

> “whether longer context helps depends on how the conditional distribution is estimated”

is **not new**.

Our value lies in the controlled demonstration and occupancy diagnostics, not the general observation.

---

## Goodman (2001): *A Bit of Progress in Language Modeling*

Goodman's large empirical study explores higher-order n-grams, smoothing, model combinations, data-size interactions, and other language-model improvements. It explicitly reports significant interactions among modeling techniques. 

This should probably become paper #12 in our review set.

---

## Siivola & Pellom (2005): *Growing an n-gram language model*

This work avoids fixing a single maximal practical order by selectively adding useful n-grams and reports experiments across different training-set sizes. It explicitly notes that less training data reduces the usefulness of higher-order n-grams. 

This further confirms that:

\[
\text{data size}\times\text{context order}
\]

is established territory.

---