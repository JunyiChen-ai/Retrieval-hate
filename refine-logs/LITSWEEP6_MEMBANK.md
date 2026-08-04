# LITSWEEP-6 — THE MEMORY BANK AS AN ACTIVE OBJECT

> **ERRATUM POINTER (2026-08-05, F120):** the "deployed head train LOO" triple `0.9406 / 0.8915 / 0.8154` used in this file (lines 177, 880, 886) is a **protocol-mixed pooled mean** over val-selected **and** final-epoch checkpoints (MHC-EN final-epoch only), not a deployed-protocol LOO triple. Measured: MHC-ZH **0.9303** (not 0.8915), HateMM **0.9404** on the deployed `-LoRA-curric` lineage, MHC-EN 0.8154 unchanged. See `TARGET_FINDINGS.md` F120. This record is left as written.

**Date:** 2026-07-27 NZST · **Agent:** litsweep-6 membank · **Cost: $0** (WebSearch/WebFetch +
local record reads only; zero GPU, zero SLURM, zero Modal, zero code execution beyond `python3`
reads of `state/directions_tried.json`). Repo sha at writing `8a9d484` (working tree dirty).
**Test-split contact: NONE** — no dataset file of any split was opened.

**Lens.** Shaping, synthesising, subsampling, calibrating **the bank itself** — as opposed to the
eval-time operators (F89), the vote depth (F94) and the decision rule (F95) that are now closed.

> ### ▸ SUPERSESSION NOTE — appended 2026-07-28 by orchestrator ruling. READ BEFORE RUNNING ANY BAR BELOW.
>
> *Nothing in this record is rewritten. This note supersedes **one clause** of its frozen bars.*
>
> **The bar "exchange rate ≥ 1.2 on the pathology population" is REFUTED AS A SCREENING CRITERION** and
> must not be used to license a run. It appears at **four** sites below — the frozen-bar blocks of C1,
> C2, C3 and C5 — each marked in place with `[SUPERSEDED as a SCREEN 2026-07-28 …]`.
>
> **Why.** `refine-logs/VSW_PREGATE_RECORD.md` §6 measured **ER = 6.0000 on HateMM — 5× this bar — and
> the arm still FAILED.** A rate is scale-free and cannot bound the quantity the goal is denominated
> in: the rate is purchasable only by shrinking the population it acts on (HateMM precision
> **0.8571 at 21 changed → 0.5696 at 79 changed**), and the product stays pinned.
>
> **The correct law and the replacement screen:**
> ```
> net = changed × (2·precision − 1)
> ```
> **Screen on NET ITEMS against 22.3 (HateMM) / 17.4 (MHC-ZH) / 16.5 (MHC-EN)**
> (`refine-logs/LITSWEEP7_LANDING_SITE.md`). The exchange rate remains a **useful diagnostic to report**
> — it is how the trade-off was found — but it is **not** a gate. Any candidate below whose screen is
> phrased as a rate must be re-phrased as a net-item count before it is run.
>
> This note does **not** touch this record's other bars, its candidate list, or MEMBANK-C4, which
> remains live and untouched.

**Citation discipline.** Every paper below was verified by **fetching its arXiv abstract page or
venue page during this sweep** and reading the abstract. §8 is the verification log, including which
fields (venue, code URL) were and were not stated on the fetched page. Three sub-agents produced the
raw candidate pool; **I re-fetched every citation I ranked** rather than trusting the relay.
Nothing here is cited from memory.

---

## §0. THE ARITHMETIC THAT SELECTS THE CANDIDATES

Before any paper: the campaign's own measurements already eliminate most of the search space, and
the eliminations are structural, not budgetary. Writing them as a filter:

**(i) The support-set argument.** Deletion, reweighting, quotas, CSLS, whitening and length-direction
excision are all operators on the *weights or the metric over a fixed support set*. If the
(short transcript × hate) cell holds almost nothing inside the query's top-20 radius, **no weight
vector and no re-metrication can put something there** — reweighting an empty support is still empty.
That is exactly the observed signature: T1 class-balanced quota **degenerate** (identical predictions
215/215 HateMM, 149/149 ZH), T2a CSLS **inert**, T3 1-D length excision **inert**, T2b whitening
**negative** (F89). Only two levers change *which items are retrievable at all*: **synthesis into the
bank** (C2), and nothing else. Deletion cannot create.

**(ii) The shape-cost argument (F95 control 2b).** Replacing the rank-weighted top-20 average with
"shortlist per class, take the best" costs **−0.0293 to −0.0437 acc before any scorer runs**. Any
candidate that abandons the averaging must first earn that back. Corollary, and this is the useful
half: **a candidate that keeps the deployed shape and changes only what is summed, or only how the
summands are weighted, pays zero shape cost by construction.** C1 and C3 are the two ways to do that.

**(iii) The exchange-rate law.** Every mechanism that surfaces the pathology *symmetrically* pays for
it at par or worse: image-stream substitution 11-14 fixed / 40-43 broken; F89 T2b/T4 1-5 fixed;
F95's learned verifier 31-54 fixed / 47-58 broken, exchange rate 0.53-0.95, never above 1.17 anywhere
in 36 cells. A 10× increase in core errors *reached* did not move the exchange rate. **Reaching the
pathology is not the hard part and no candidate should be sold on reaching it.** The bar is the
exchange rate, and every pregate below reports it.
**[CORRECTED 2026-07-28 — the last sentence is wrong and is superseded. F105/VSW measured ER = 6.0000
on HateMM and the arm still failed: the rate is NOT the bar. `net = changed × (2·precision − 1)`, and
the bar is NET ITEMS ≥ 22.3 / 17.4 / 16.5. Reporting the rate stays useful as a diagnostic. See the
supersession note at the top of this file.]**

**(iv) The interaction-share fact (F95 §4.1), and what it does NOT license.** Only **26.6-37.7 %** of
the deployed cosine's score variance is query×bank interaction; 62-73 % is item-level offsets. That
looks like an invitation to hubness correction. It is not: the offset term splits into a **query**
offset, which is *rank-invariant within a query and therefore cannot flip a single prediction*, and a
**bank-entry** offset, which is exactly what CSLS removes — and CSLS was measured **inert**. The
hubness axis is closed by arithmetic, not by budget. I ran the 2024-2026 hubness literature anyway
(§7.1) and recommend against spending on it.

**(v) What F94 leaves.** Ranks 11-20 flip **zero** predictions; the noise is at ranks 1-5 where the
labels are wrong. So any aggregation candidate must act on **ranks 1-5**, and any candidate whose
effect is concentrated in the tail is pre-dead.

**(vi) The one unexamined object.** F89 de-biased the **geometry** (keys, metric, directions). F94
re-cut the **depth**. F95 replaced the **decision rule**. C2/C5 shape the **membership**. Nothing in
63 dead entries has ever touched **the label field the vote transports** — the quantity
`(2·lab_i − 1)` that sits inside the sum. That is C1, and it is the gap this sweep found.

Together these say: candidates must be **(a)** bank-membership changes, **(b)** label-field changes,
or **(c)** aggregation changes that preserve the 20-neighbour average. Everything else is priced dead.

---

## §1. CANDIDATE C1 — NUISANCE-RESIDUAL VOTE (de-bias the label field, not the geometry)

**Rank 1 of 5.** Attacks **CP1**. $0 pregate, **zero new parameters** in the deployed path.

### (a) Papers

1. **FT2Ra: A Fine-Tuning-Inspired Approach to Retrieval-Augmented Code Completion.**
   Qi Guo, Xiaohong Li, Xiaofei Xie, Shangqing Liu, Ze Tang, Ruitao Feng, Junjie Wang, Jidong Ge,
   Lei Bu. **ISSTA 2024** (venue stated in arXiv comments). arXiv:**2404.01554**, 2 Apr 2024.
   No code URL on the abstract page.
2. **Long-Tail Crisis in Nearest Neighbor Language Models.** Yuto Nishida, Makoto Morishita,
   Hiroyuki Deguchi, Hidetaka Kamigaito, Taro Watanabe. **Findings of NAACL 2025.**
   arXiv:**2503.22426**, 28 Mar 2025. (Supporting diagnosis, not the mechanism.)
3. **Delving into Deep Imbalanced Regression.** Yuzhe Yang, Kaiwen Zha, Ying-Cong Chen, Hao Wang,
   Dina Katabi. **ICML 2021 (Long Oral).** arXiv:**2102.09554**. Code
   `https://github.com/YyzHarry/imbalanced-regression` (stated on the page). (Supplies the
   *ordered-covariate* estimator for the base model — see (d).)

### (b) Mechanism in three sentences

FT2Ra derives from the gradient of fine-tuning that the quantity worth retrieving is not a
neighbour's **label** but its **delta logit** — the residual between that neighbour's true label and
what the base model already predicts for it. At inference it retrieves neighbours, looks up each
neighbour's stored residual, aggregates the residuals with the ordinary retrieval weights, and adds
the result to the base prediction. Out comes a prediction corrected by *what the base model gets
wrong in this neighbourhood*, rather than by *what the neighbourhood's labels are*.

### (c) Mapping to the measured pathology, and the distinctness argument

CP1 says the bank's class prior is a function of transcript length: `P(hate | 0-1 words) = 0.1096`
rising monotonically to `0.5538` at 401+ words (ERRPAT-HateMM §2/§4.3), and retrieval is itself
length-organised. A vote over raw signed labels therefore **transports the neighbourhood's
length-conditioned base rate wholesale**: a short hateful query retrieves short neighbours that are
~89 % non-hate and is confidently inverted, even though the correct analogue is sitting at median
rank 1.5 (11 of 22 ZH core errors at rank 1).

Transport residuals instead. With `p̂_i = P̂(hate | length_i)` fitted on **train only, leave-one-out**,
replace the summand `s_i = 2·lab_i − 1` by `r_i = s_i − (2·p̂_i − 1)`. A short non-hate bank item
carries `r = −0.22` instead of `−1`; a short **hate** item carries `r = +1.78` instead of `+1`. The
component of the neighbourhood's evidence that is explained by the shared nuisance cancels *before*
the sum, and only the part the covariate cannot explain is transported. Worked on the exact ERRPAT
configuration (correct analogue at rank 1, nineteen wrong-class neighbours at ranks 2-20, rank
weights `[20..1]`): the normalised vote moves from **−0.81 to −0.10** — an 8× compression toward the
boundary, concentrated entirely on the population F94 identified as the live one (ranks 1-5).

Why it is not each dead precedent, in the order a reviewer will raise them:

* **vs F89 (all five).** Those operators edit the **keys**: a quota over retrieved keys, a hubness
  offset on similarities, a whitener, a direction removed from the key space. Every one is
  **label-blind** and every one is a map of the *geometry*. C1 does not touch a key, a similarity, a
  rank or the retrieval at all — the identical 20 neighbours in the identical order are retrieved,
  and only the **label field being summed** changes. F89's own two mechanism facts *support* C1
  rather than threaten it: F89c/F89d found the length organisation **is not carried by any single
  linear direction** and whitening **amplifies** it — i.e. the nuisance is not removable from the
  geometry, which is precisely the argument for removing it from the field instead.
* **vs global thresholds / score-level logistic recalibration (dead).** Those are **monotone maps of
  one scalar** — the final vote — and a monotone map of a scalar cannot reorder anything; it can only
  move a threshold. C1 modifies **each summand individually** by an item-specific amount before
  aggregation, which changes the vote's ordering across items. This is the same distinction FT2Ra's
  own analysis makes, and it retro-explains why score-level recalibration was dead: a single global
  map cannot undo a **neighbourhood-varying** prior.
* **vs F82 (vote-side Offensive reweighting, "any monotone weighting, any tau").** F82 reweighted the
  label value by a **class-level constant** (the Offensive class gets weight τ). C1's correction is
  **item-level and covariate-driven** — two bank items with the same gold label receive different
  summands. Not a monotone function of the label; not in F82's family. (Flagging this explicitly
  because it is the closest ban and a reviewer will reach for it.)
* **vs F63 (label propagation / diffusion).** LP is multi-hop and diffuses labels between bank items.
  C1 is strictly **one-hop** — the same single hop the deployed vote already performs — and nothing
  propagates between bank items. The topology is unchanged; only the transported quantity differs.
* **vs the pseudo-label ban.** No item is added, deleted, or given a label it did not have. Every
  `lab_i` is the gold train label; `p̂_i` is a *correction term*, not a label.

### (d) Transplant

**Changes:** one vector. Precompute `p̂` for the bank once and store it beside `bank_lab`; in
`mechfix_ops.deployed_vote`, `(2*lab − 1)` becomes `(2*lab − 1) − (2*p̂ − 1)`.
**Stays:** encoder, head, key space, FAISS index, k=20, rank weights `[20..1]`, threshold at 0,
both protocols, every floor.

**The base model must NOT be the trained head.** **[ERRATUM — see the appended ERRATUM at the end of
this file. 0.998 is F47's **CLIP** head; the deployed **Qwen** heads measure 0.9406 / 0.8915 / 0.8154.
The design rule below is still sound — 0.82-0.94 is high enough that residuals against the trained head
would be badly attenuated — but it should not have been justified at 0.998.]** F47 measured the RGCL head's leave-one-out accuracy
on its own train split at **0.998**; residuals against it are ≈ 0 by construction and the vote would
collapse to noise. This is the single most likely way to run C1 and get a meaningless null, so it is
written into the design: the base model is a **nuisance-only** predictor, deliberately weak, fitted
on the covariate alone.

Three declared base-model arms, in increasing order of capacity:
- **B-a (primary):** univariate logistic on `log(1 + n_words)`, LOO over train. 2 parameters.
- **B-b:** ordered-bin base rate with **FDS-style smoothing across neighbouring bins** (arXiv:2102.09554)
  — our covariate is continuous and monotone in the prior, which is an imbalanced-*regression*
  geometry, not a categorical long-tail one; smoothing across adjacent bins is the estimator that
  matches it, and it is what makes the 5-20-item bins usable.
- **B-c:** logistic on `[log(1+n_words), log duration]`. 3 parameters.

Word counts already exist in-repo (`errpat_mhc_en.py:178` computes `n_tr_word`;
`errpat_hatemm_ceilings.py:140` computes `nw`; ZH uses transcript character length,
`errpat_zh_c2_settle.py:12`), and `mechfix_ops.fit_length_direction(train_keys, length_scalar)`
shows the length scalar is already plumbed through the F89-frozen harness.

### (e) $0 / CPU pregate

Reuse the **F95 harness verbatim**: `StratifiedKFold(5, shuffle=True, random_state=0)` over train
items, item-disjoint, floor = `mechfix_ops.deployed_vote` (sha256 `635c1312…c83fc8d`, 15/15 floor
parity at 4 dp) on the same fitting-fold bank. `p̂` is fitted **inside the fitting folds only**; the
held-out fold never contributes to the base model. Train split only; `dev_seen`/`test_seen` unopened.
Cost: **minutes of CPU, ≤8 threads, $0.** Full version: **0 GPU-h** — C1 changes no trained object,
so the formal ceremony is a re-evaluation of existing checkpoints, not a retrain.

**Frozen bars (declare before running, F95 style):**
1. **Primary:** pooled held-out-item Δacc vs deployed ≥ **+0.010** on ≥1 dataset, 5/5 folds Δ ≥ 0,
   ≥3/5 strictly positive. (1 item = 0.0013/0.0017/0.0018, so +0.010 = 7.4/5.8/5.5 items.)
2. **Exchange rate ≥ 1.2** on the pathology population (deployed-wrong items whose nearest
   same-gold-class bank item is within rank 5) — set above F95's ceiling of 1.17 *on purpose*: a
   candidate that lands inside the band every symmetric operator has already occupied has told us
   nothing new.
   **[SUPERSEDED as a SCREEN 2026-07-28 — see the supersession note at the top; screen on NET ITEMS 22.3/17.4/16.5, not on a rate]**
3. **Degeneracy control (fires a KILL, not a caveat):** report `sd(p̂)` over the bank and the AUC of
   `p̂` against the gold train label. If `p̂` is near-constant, C1 is a global threshold shift in
   disguise and is dead by the existing ban regardless of its Δ.
4. **Stratum-honesty control:** report Δacc **separately for the short-transcript and long-transcript
   halves**. C1 necessarily makes short queries more likely to be called hate. If the gain is
   short-recall bought at exactly matching long-precision cost, that is the exchange-rate law again
   and must be reported as such, not netted out.
5. **Class-balance sanity** (F95 control 4): decision positive rate vs bank positive rate.

### (f) Honest risk

**Most likely killer: the exchange-rate law (iii).** C1 is, viewed unsympathetically, a
*stratum-conditional prior correction implemented inside the vote*; it will move short-transcript
queries toward "hate" as a population, and the campaign's history is that population-level moves pay
for themselves. **F89a is the specific threat**: "the local class prior is not separable from the
retrieval signal in the cone-collapsed space". If separability fails at the field level as it did at
the geometry level, the residual is a near-constant shift and bar 3 fires.

**What distinguishes survival, observable early:** C1 differs from F89's T1 in that T1 was
*degenerate* — it changed **no** prediction (215/215, 149/149) — whereas C1 is continuous and
item-level and **will** change predictions. So the informative outcome is available cheaply either
way. The early tell is bar 3 (`sd(p̂)` and its AUC) together with the **sign pattern of the first
fold's exchange rate**: an exchange rate ≥ 1.2 in fold 1 is the first number in this campaign that
would break law-(iii), and an exchange rate landing in 0.5-1.1 again is the ninth confirmation of it.

---

## §2. CANDIDATE C2 — CELL-CONDITIONAL SYNTHESIS INTO THE BANK

**Rank 2 of 5.** Attacks **CP1**. Highest mechanism-novelty in the sweep; survival honestly lower
than C1. This is the *only* candidate that changes which items are retrievable.

### (a) Papers

1. **FeTrIL: Feature Translation for Exemplar-Free Class-Incremental Learning.** Grégoire Petit,
   Adrian Popescu, Hugo Schindler, David Picard, Bertrand Delezoide. **WACV 2023** (venue attested by
   the CVF Open Access page; the arXiv page lists cs.CV only). arXiv:**2211.13131**, v1 23 Nov 2022,
   v2 28 Nov 2023. Code `https://github.com/GregoirePetit/FeTrIL` (**not** stated on the arXiv page —
   verified on GitHub).
2. **Free Lunch for Few-shot Learning: Distribution Calibration.** Shuo Yang, Lu Liu, Min Xu.
   **ICLR 2021.** arXiv:**2101.06395**. No code URL on the abstract page.
3. **Delving into Deep Imbalanced Regression** (FDS/LDS), as §1. arXiv:**2102.09554**, **ICML 2021
   Long Oral**, code stated.
4. **Simplicial SMOTE: Oversampling Solution to the Imbalanced Learning Problem.** Oleg Kachan,
   Andrey Savchenko, Gleb Gusev. **KDD 2025 (research track).** arXiv:**2503.03418**, 5 Mar 2025.
   No code URL on the page. (The 2025 frontier of the interpolation family — and, per §7.3, the arm
   I expect to fail.)
5. **Decision Boundary-aware Generation for Long-tailed Learning.** Jiacheng Yang, Ruichi Zhang,
   Chikai Shang, Mengke Li, Xinyi Shang, Junlong Gao, Yonggang Zhang, Yang Lu. **CVPR 2026** (stated
   as accepted). arXiv:**2605.01468**, 2 May 2026. Code `https://github.com/keepdigitalabc-svg/DBG`
   (stated on the page). (Names the failure mode C2 must design against.)
6. **Bias-Corrected Data Synthesis for Imbalanced Learning.** Pengfei Lyu, Zhengchi Ma, Linjun Zhang,
   Anru R. Zhang. arXiv:**2510.26046**, v1 30 Oct 2025, v2 Feb 2026. **No venue stated.** No code URL.
   (The estimator-level correction; the diagnostic to run if a first pass measures negative.)
7. **FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning.**
   Dipam Goswami, Yuyang Liu, Bartłomiej Twardowski, Joost van de Weijer. **NeurIPS 2023.**
   arXiv:**2309.14062**. Code `https://github.com/dipamgoswami/FeCAM` (stated). (Included as the
   **counter-evidence**, see (f), and as the numerical tooling — shrinkage + Tukey — that any
   attempt at d=7168 with 5-20 samples requires.)

### (b) Mechanism in three sentences

FeTrIL: in goes a frozen-feature population from a data-**rich** cell plus the stored centroid of a
data-**poor** cell; the rich cell's features are rigidly translated by (poor-centroid − rich-centroid);
out come pseudo-features that carry the rich cell's *intra-cell diversity* while sitting at the poor
cell's *location*. Distribution Calibration does the second-moment version: in goes the poor cell's
handful of features plus rich-cell covariances, a Tukey transform plus similarity-weighted statistics
transfer produces a calibrated Gaussian, and out come samples from it. FDS supplies the estimator for
our case specifically — because our nuisance is **ordered and continuous**, statistics are borrowed
from *adjacent bins* rather than from an arbitrary "similar class".

### (c) Mapping and distinctness

CP1's mechanical content is that the (short × hate) cell is nearly empty, so no local estimate of the
class prior in that region can be right. Synthesis is the only operator in the entire family that
changes **membership**: it puts hate-labelled mass into the short-transcript region so that the
deployed vote — untouched — retrieves it.

* **vs F78 (deletion-based curation, measured null) and W2-E prototype-select.** Deletion is
  monotone-decreasing in support. It can only remove; the pathology is absence. These are opposite
  operators on the same object and a null for one is not evidence for the other.
* **vs F89 / F94 / F95.** All three act *after* retrieval on a fixed candidate set. C2 acts *before*
  it, and the deployed decision rule is unchanged — so it pays **zero** F95-control-2b shape cost
  and inherits the 85 %-protecting averaging intact. This is the specific reason to prefer synthesis
  over any further decision-rule work.
* **vs the pseudo-label ban** (the ban a reviewer will fire first). The ban is *"kNN-vote-pool
  expansion via pseudo-labels"* — creating a label for an item that did not have one. C2's synthetic
  points are **within-class translations of labelled parents along the nuisance axis**: a
  (long × hate) train item is moved toward the short-transcript region and **keeps its own hate
  label**. No unlabelled item is ever scored, no external or cross-dataset item is touched, and the
  class of every synthetic point is the class of its own gold parent.
  **Consequence, binding:** **M2m-style majority→minority translation is EXCLUDED** from this
  candidate. M2m (arXiv:2004.00431, CVPR 2020) and its 2025 feature-space port (arXiv:2508.06420,
  IGARSS 2025) translate a *majority-labelled* parent and then call it minority — that assigns a
  label the parent did not have, and while it does not trip the ban literally, it is close enough
  that it needs a user ruling before anyone spends on it. C2 as specified never does this.
  (Its second liability is mechanical anyway: M2m's translation is steered by a trained classifier's
  gradients, and here the classifier/vote *is* the broken object.)

### (d) Transplant

**Changes:** the bank gains rows. `bank_keys ← [bank_keys ; Z_syn]`, `bank_lab ← [bank_lab ; y_syn]`
where `y_syn` is inherited. **Stays:** everything else — encoder, head, retrieval, k=20, weights,
threshold.

Pipeline, all CPU: stratify train by (gold class × length quartile); identify donor-rich and
recipient-poor cells; PCA to 128-d fitted on **fitting-fold items only**; synthesise in the reduced
space (FeTrIL translation as arm S1; DC/FeCAM shrinkage-Gaussian sampling as arm S2); project back;
**re-L2-normalise** — non-negotiable, since the bank is L2-normalised and FAISS-inner-product scored,
so an unnormalised synthetic row is either never retrieved or always retrieved. Enforce a minimum
cosine separation among synthetic rows and between synthetic and real rows: in a **rank-weighted**
vote, twenty near-copies of one parent are not twenty pieces of evidence, and the space is
cone-collapsed (deployed top-1 cosine 0.9439-0.9686, F91).

An anchor-local variant is worth declaring in the same freeze, because it follows directly from
ERRPAT: rather than translating onto the *cell centroid*, translate onto the **retrieved rank-1.5
analogue's location**, which is the only construction that guarantees the synthetic mass lands
*inside the query's actual top-20 radius* rather than merely inside the right cell.

### (e) $0 / CPU pregate

Same 5-fold item-disjoint harness. **Synthetic rows are built from fitting-fold parents only and
placed only in the fitting-fold bank**; held-out items never parent a synthetic row. Cost: **$0,
minutes.** Full version: **0 GPU-h** for the bank-side change itself (the head is not retrained);
budget ~0.3 GPU-h only if the ceremony requires a same-path floor re-mint, per the F78 door-closer.

**Frozen bars:**
1. Primary Δacc ≥ **+0.010**, 5/5 fold signs ≥ 0, ≥3/5 strictly positive, ≥1 dataset.
2. **Exchange rate ≥ 1.2** on the pathology population. **[SUPERSEDED as a SCREEN 2026-07-28 — see the supersession note at the top; screen on NET ITEMS 22.3/17.4/16.5, not on a rate]**
3. **Retrieval-quality read (the mechanism check, and the paper's real deliverable):** median top-20
   true-label purity on the pathology population, **before vs after**, currently 0.12-0.22. If purity
   does not move, the synthesis did not land where CP1 says it must, and the accuracy read is
   uninterpretable.
4. **Occupancy control:** fraction of top-20 slots held by synthetic rows, and Δacc as a function of
   the synthetic:real ratio ρ ∈ {0.1, 0.25, 0.5}. A gain that only appears at ρ = 0.5 is the bank
   being replaced by its own smoothing, not repaired.
5. **Near-duplicate control:** max cosine between each synthetic row and its parent; a distribution
   piled at >0.99 means the interpolants are copies and the arm is void (this is the Blagus-Lusa
   failure made observable).

### (f) Honest risk

**Two named, published hard negatives, both of which I recommend be written into the prereg rather
than discovered later.**

**Blagus & Lusa, "SMOTE for high-dimensional class-imbalanced data", BMC Bioinformatics 14:106
(2013), DOI 10.1186/1471-2105-14-106** — verified via PMC3648438 — states that SMOTE **should not be
used with k-NN without variable selection** because it strongly biases classification toward the
minority class; that it helps k-NN in high dimensions *only if* variable selection is done first,
with the benefit **larger for larger k**; and that it **decreases data variability and introduces
correlation between samples**. Our configuration (k = 20, rank-weighted, d = 7168, ~550-744 items) is
precisely the one that paper warns about. This is why dimensionality reduction before synthesis is
written into (d) as a precondition, not an option — and why the interpolation arm (Simplicial SMOTE)
is ranked below the translation and statistics-transfer arms rather than above them.

**FeCAM (NeurIPS 2023)** reports that *modelling feature covariance beats sampling features from
normal distributions* in the frozen-feature few-shot regime — direct evidence against the DC arm.
Two things salvage it here and both should be stated up front: FeCAM's comparison is at the
**classifier** level, where a Mahalanobis metric can substitute for synthesis, whereas our decision
is a **retrieval vote over discrete bank rows** and there is no metric that conjures a bank row — we
have already measured that the metric-level fix (whitening) is negative and *amplifies* the length
organisation. And FeCAM's shrinkage + Tukey machinery is the tooling the DC arm needs at d = 7168
with 5-20 samples regardless.

**Third risk, from DBG (CVPR 2026): "latent non-local feature mixing"** — head-to-tail translation
can place synthetic points where they straddle the boundary. In a softmax classifier that is a
regularisation cost; **in a kNN vote it is worse**, because a boundary-straddling row silently
occupies top-20 slots for queries of *both* classes. DBG's answer is boundary-aware placement. We
already own a placement filter that is paid for and currently has no job: the **F95 pair verifier**
(pair-AUC +0.13-0.27, within-query +0.16-0.23). Using it to *accept or reject synthetic rows* is
squarely inside what F95 left legal — it is relation scoring, not a decision rule and not an accuracy
claim in itself.

**What distinguishes survival:** bar 3. If top-20 purity on the pathology population moves and
accuracy does not, C2 becomes the **tenth law-I datum** and closes the bank-membership axis cleanly.
If purity does not move, the synthesis never landed and the axis is untested — which is why bar 3
must be read before bar 1.

---

## §3. CANDIDATE C3 — LEARNED AGGREGATION OVER THE NEIGHBOURHOOD PROFILE

**Rank 3 of 5.** Attacks **CP2** without ever replacing the average with a max.

### (a) Papers

1. **Adaptive Nearest Neighbor Machine Translation.** Xin Zheng, Zhirui Zhang, Junliang Guo,
   Shujian Huang, Boxing Chen, Weihua Luo, Jiajun Chen. **ACL-IJCNLP 2021** (main, short).
   arXiv:**2105.13022**, 27 May 2021. Code `https://github.com/zhengxxn/adaptive-knn-mt`.
   Abstract states the Meta-k Network "can be efficiently trained with only a few training samples".
2. **Learning Kernel-Smoothed Machine Translation with Retrieved Examples** (KSTER). Qingnan Jiang,
   Mingxuan Wang, Jun Cao, Shanbo Cheng, Shujian Huang, Lei Li. **EMNLP 2021.** arXiv:**2109.09991**.
   Code `https://github.com/jiangqn/KSTER`. *(Honesty note: the abstract I fetched confirms
   kernel-smoothed online adaptation and the overfitting-to-retrieved-examples motivation; it does
   **not** itself state the per-query bandwidth/mixing-weight prediction. That detail is from the
   body and must be re-checked in the PDF before it is cited as such.)*
3. **PNI: Industrial Anomaly Detection using Position and Neighborhood Information.** Jaehyeok Bae,
   Jae-Han Lee, Seyun Kim. **ICCV 2023.** arXiv:**2211.12634**, v1 22 Nov 2022, v3 30 Mar 2023.
   Code `https://github.com/wogur110/PNI_Anomaly_Detection` (not stated on the arXiv page).
4. **StructCore: Structure-Aware Image-Level Scoring for Training-Free Unsupervised Anomaly
   Detection.** Joongwon Chae, Lihui Luo, Yang Liu, Runming Wang, Dongmei Yu, Zeming Liang, Xi Yuan,
   Dayan Zhang, Zhenglin Chen, Peiwu Qin, Ilmoon Chae. arXiv:**2602.17048**, v1 19 Feb 2026, v2
   21 Feb 2026. **No venue stated, no code URL stated** — flagged.
5. Supporting: **Why do Nearest Neighbor Language Models Work?** Frank F. Xu, Uri Alon, Graham Neubig,
   arXiv:**2301.02828**; **Great Memory, Shallow Reasoning: Limits of kNN-LMs**, Shangyi Geng,
   Wenting Zhao, Alexander M. Rush, arXiv:**2408.11815**, code
   `https://github.com/GSYfate/knnlm-limits/` (stated).

### (b) Mechanism in three sentences

Meta-k retrieves a fixed upper bound of K neighbours once and builds a **configuration vector** from
the neighbourhood alone — the K distances plus the count of distinct labels among the top-i for each
i, i.e. a purity-prefix profile. A ~0.6k-3k-parameter MLP maps that vector to a distribution over
candidate neighbourhood sizes, and the prediction is the **soft mixture** of the corresponding kNN
distributions; PNI and StructCore are the same move in anomaly detection (learn `p(class | retrieved
configuration)`; consume a **vector descriptor of the score profile** rather than one extreme of it).
Nothing about the base model, the datastore or the retrieval changes — only the combination rule.

### (c) Mapping and distinctness

CP2 says the fixed profile `w = [20..1]` is right on ~85 % of items and wrong on ~15 %. That is
literally the statement that **the right weighting is item-dependent**, and a *global* grid search
over one scalar cannot express it.

* **vs F94 (k-sweep, closed both directions).** F94 chose **one global k** and found the plateau; its
  strongest upper bound is per-seed **oracle** k at **+0.0145**. That bounds the *k-selection*
  family. C3 is not in it: a soft mixture over k spans a **convex family of monotone weight
  profiles**, and a profile network spans **non-monotone** rules the k-grid cannot reach — for
  instance "when the rank-1 margin is large, trust rank 1", which is exactly the ERRPAT
  configuration and is unreachable by any single global k (F94 proved `k ≤ 3` **is** 1-NN and costs
  −0.016 to −0.039 *globally*, but never priced it *conditionally*). **This is the load-bearing
  distinctness claim for C3 and it should be attacked first in review.**
* **vs F95.** No shortlist, no max, no per-class quota — all 20 neighbours keep contributing, so the
  −0.0293/−0.0437 shape cost is never paid.
* **vs F47 (per-item selection dead at all 3 supervision sources).** F47's target was *"is operator A
  correct on this item?"*, which degenerates because the head memorises train (LOO 0.998). C3's
  target is **the gold label**, which memorisation cannot degenerate, and the fit is item-disjoint
  LOO so the aggregator never sees its own query.
* **vs global calibration.** The input is the **local configuration**, not the output score.

### (d) Transplant

**Changes:** the scalar weight vector becomes a learned function.
`v = Σ_i r_i · cos_i · g_θ(profile)_i / Σ_i g_θ(profile)_i`, where the profile is
`[cos_1..cos_20 ; s_1..s_20 ; purity-prefix counts ; (optionally) Δlength_1..Δlength_20]`.
**Stays:** encoder, head, keys, retrieval, k = 20, threshold. θ is ~1-3k parameters, CPU-trained.
Note `r_i` is written rather than `s_i` so that **C3 composes with C1** — the two are orthogonal (C1
changes what is summed, C3 changes the weights) and the composition should be a declared arm.

### (e) $0 / CPU pregate

Same 5-fold item-disjoint harness; θ fitted on fitting-fold profiles only. 549-744 LOO examples of a
~60-d → 1 problem. **$0, CPU, minutes.** Full version: **0 GPU-h**.

**Frozen bars:**
1. Δacc ≥ **+0.010**, 5/5 fold signs, ≥3/5 strictly positive, ≥1 dataset.
2. **The control that decides whether C3 means anything (declare in advance, F95-2b style):** the
   learned aggregator must beat **the best fixed monotone profile chosen on the fitting folds** — not
   merely the deployed `[20..1]`. Beating `[20..1]` but not the best fixed profile is a win for
   *profile tuning*, which F94 has effectively closed, and must be reported as such.
3. **Non-monotonicity read:** what fraction of held-out queries receive a **non-monotone** learned
   weight profile, and is the Δ concentrated on them? If the learned profile is monotone almost
   everywhere, C3 has collapsed into F94's family and is dead by that precedent.
4. Exchange rate ≥ 1.2 on the pathology population; class-balance sanity. **[SUPERSEDED as a SCREEN 2026-07-28 — see the supersession note at the top; screen on NET ITEMS 22.3/17.4/16.5, not on a rate]**

### (f) Honest risk

**Most likely killer: F47's law plus F94's oracle bound.** The profile is a deterministic function of
what the vote already consumes, so C3 introduces **no new signal source** — and "per-item selection
without a new signal source" is exactly what F47/F49/F66 closed. The counter-argument is that C3
**selects nothing**: it emits a continuous weight for every neighbour, with no branch and no operator
choice. Whether reviewers accept that distinction is the real gate, and it should be settled *before*
the pregate runs, not after.

Second risk: overfitting at n = 549-744. Meta-k's own evidence is favourable (the abstract states
"only a few training samples"; the body's reported figure of ~100 training sentences should be
re-verified in the PDF before citation), and 1-3k parameters on 549-744 LOO examples is the smallest
learned object this campaign has ever fitted — but bar 2 is what protects against fitting fold noise.

**Early observable:** bar 3. Non-monotone profiles concentrated on the ERRPAT population = alive.
Monotone everywhere = it re-derived F94 and should be killed on the spot.

---

## §4. CANDIDATE C4 — AGGREGATE-THEN-COMPARE (class-conditional subspace residual)

**Rank 4 of 5.** Attacks **CP2** with a functional form that is neither the mean nor the max.

### (a) Papers

1. **ProCon: Projection-Consistency Memory for Training-Free Anomaly Detection.** Joongwon Chae,
   Lihui Luo, Yang Liu, Dongmei Yu, Peiwu Qin, Runming Wang, Ilmoon Chae. arXiv:**2607.04894**,
   6 Jul 2026, cs.CV. **No venue stated** (preprint). Code `https://github.com/jw-chae/Procon`
   (stated on the page).
2. **SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling.** Camile Lendering,
   Erkut Akdag, Egor Bondarev. **CVPR 2026** (stated as accepted in the comments).
   arXiv:**2602.23013**, 26 Feb 2026. Code `https://github.com/CLendering/SubspaceAD` (stated).
3. **Learning to Compare: Relation Network for Few-Shot Learning.** Flood Sung, Yongxin Yang,
   Li Zhang, Tao Xiang, Philip H.S. Torr, Timothy M. Hospedales. **CVPR 2018.** arXiv:**1711.06025**.
   (The lineage the team lead asked about; see (c) for what it actually contributes.)

### (b) Mechanism in three sentences

ProCon stops treating the bank as a nearest-neighbour lookup table and instead **softly projects the
query onto the span of its retrieved memory vectors**, using the **projection residual** as the
evidence; residuals are aggregated by **median** across seed-perturbed banks and fused by consensus
across layers. SubspaceAD is the same object with the bank discarded entirely — fit a PCA subspace to
the reference features and score by reconstruction residual. RelationNet supplies the composition
order that unifies them: in the k-shot case it **aggregates the support set first** (element-wise
into one class representation) and computes **one** relation per class, instead of computing k
relations and pooling them.

### (c) Mapping and distinctness

This is the direct answer to the question F95 left open. F95 measured **compare-then-aggregate**
(score each pair, then max or mean-top-3) and that architecture lost −0.029 to −0.044 to the shape
change alone. **Aggregate-then-compare has never been measured here.** Its mechanical appeal against
CP2 is exact: a single correct analogue can **span** the query — contributing to the class's
representation — without needing to *out-vote* nineteen wrong-class neighbours, and without the
brittleness of a max over individually noisy pair scores.

* **vs F89.** A projection residual onto a **query-dependent** subspace cannot be written as
  `cos(Az, Az')` for any fixed `A`; it is not in the re-metrication family those five operators span.
* **vs F95.** Different composition order, and the F95 ban is explicitly about *nomination + per-pair
  verification*; §5's routing note also forecloses other **pair-scorer architectures**, which C4 is
  not — there is no pair scorer at all.
* **vs F63.** No propagation between bank items and no multi-hop anything; one hop, one query.
* Median aggregation across perturbed banks is a **robust order statistic**, neither the mean the
  vote uses nor the max F95 killed — and it is the only verified instantiation I found of the
  "trimmed/robust aggregation of neighbour evidence" the brief asked for (see §7.4 on conformal).

### (d) Transplant

**Changes:** the decision rule, replaced by a residual gap. Retrieve the deployed top-20; split by
gold label; for each class `c` form a rank-`r` basis from that class's retrieved members (`r` small,
ridge-regularised); predict 1 iff `residual_0 − residual_1 > 0`. **Stays:** encoder, head, keys,
retrieval, k = 20 — the *candidate set is exactly the deployed one*, which is what keeps this out of
F95's nomination family.

**Warning that must be in the freeze:** with ~10 vectors per class in a 128-256-d reduced space, both
spans are near-universal and both residuals go to ≈ 0. `r ∈ {1, 2, 3, 5}` and ridge regularisation
are therefore **arms, not tuning**, and must be declared before the run.

### (e) $0 / CPU pregate

Same 5-fold item-disjoint harness; **training-free** (a least-squares projection). **$0, minutes.**
Full version: **0 GPU-h**.

**Frozen bars:** (1) Δacc ≥ +0.010, 5/5 fold signs, ≥3/5 positive; (2) exchange rate ≥ 1.2 **[SUPERSEDED as a SCREEN 2026-07-28 — see the supersession note at the top; screen on NET ITEMS 22.3/17.4/16.5, not on a rate]**;
(3) **degeneracy control fired before anything else** — the distribution of `residual_0 − residual_1`
must be non-degenerate at some declared `r`; if the two residuals are near-identical at every `r`,
the arm is void and reports nothing about aggregate-then-compare; (4) class-balance sanity.

### (f) Honest risk

**Most likely killer: degeneracy, then the exchange-rate law.** In a cone-collapsed space (top-1
cosine 0.9439-0.9686) with 10 vectors per class, the two class spans may be indistinguishable — this
would be a *harness* null, not a mechanism null, and bar 3 exists to say so honestly rather than
bank a fake kill.

Secondary risk: this is the candidate most likely to be argued into F95's ban by a strict reader.
The defence is textual and mechanical — F95 banned *nomination + per-pair verification*, and C4
neither re-nominates (same candidate set) nor scores pairs (no pair function exists) — but it should
be pre-cleared in the recon rather than argued after a positive result.

**Early observable:** at the best `r`, does the residual gap separate the pathology population at all?
If yes and accuracy still does not move, that is another law-I datum on a genuinely new functional
form. If no, kill on bar 3 and record it as untested-by-degeneracy.

---

## §5. CANDIDATE C5 — PER-ENTRY SOFT RELIABILITY WEIGHTS ON THE BANK

**Rank 5 of 5.** Attacks **CP1/CP3** weakly. Included because it is the honest continuous relaxation
of a measured null, and because it has a **second, non-performance job** that no other candidate has.

### (a) Papers

1. **SoftPatch: Unsupervised Anomaly Detection with Noisy Data.** Xi Jiang, Ying Chen, Qiang Nie,
   Yong Liu, Jianlin Liu, Bin-Bin Gao, Jun Liu, Chengjie Wang, Feng Zheng. **NeurIPS 2022.**
   arXiv:**2403.14233**.
2. **SoftPatch+: Fully Unsupervised Anomaly Classification and Segmentation.** Chengjie Wang,
   Xi Jiang, Bin-Bin Gao, Zhenye Gan, Yong Liu, Feng Zheng, Lizhuang Ma. **Pattern Recognition**
   (venue stated in comments), arXiv:**2412.20870**, v1 30 Dec 2024, v2 13 Jan 2025. Code
   `https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch` (stated).
3. **Improving Retrieval-Augmented Large Language Models via Data Importance Learning.**
   Xiaozhong Lyu, Stefan Grafberger, Samantha Biegel, Shaopeng Wei, Meng Cao, Sebastian Schelter,
   Ce Zhang. arXiv:**2307.03027**, 6 Jul 2023. **No venue stated.**

### (b) Mechanism in three sentences

Noise discriminators compute an outlier/density score for every candidate bank entry **before**
coreset construction; the score is **stored in the bank alongside the entry** and used at inference to
*scale* that entry's contribution, softening the decision boundary. Nothing is deleted — SoftPatch's
explicit argument is that noisy entries **cannot be removed completely**, so removal is the wrong
operator. Lyu et al. give the supervised analogue: a polynomial-time multilinear-extension importance
per corpus entry, used to prune **or reweight**, reporting 33.3 % → 37.7 % (prune) and 36.9 %
(reweight).

### (c) Mapping and distinctness

SoftPatch's stated argument is a direct refutation of the *operator* our F78 recon parked: deletion is
binary and irreversible, and the pathology is graded. A per-entry weight `α_i ∈ (0,1]` folded into the
vote (`Σ r_i · cos_i · w_i · α_i`) is the continuous relaxation. With gold labels available we can
define something SoftPatch cannot: **down-weight bank entries whose label agreement with their own
neighbourhood is explained by the length stratum rather than by content** — a direct CP1
counter-weight computable at zero GPU.

**But be honest about the family adjacency:** this is a *reweighting* of a fixed support, which is
exactly filter (i) above. It cannot put mass where there is none. That is why it ranks last.

### (d) Transplant

**Changes:** one scalar per bank row, multiplied into the vote. **Stays:** everything else.
Composes with C2 (synthetic rows enter at `α < 1`) and with C1 (orthogonal terms).

### (e) $0 / CPU pregate

Same harness; `α` fitted on fitting folds only. **$0, minutes.** Full version: **0 GPU-h**.
**Frozen bars:** as C1, plus a **spread control** (`sd(α)`; a near-constant `α` is a no-op) and a
**support control** — report Δacc restricted to items whose top-20 contains ≥1 same-gold-class entry
versus those where it does not, since C5 can by construction do nothing for the latter.

**Explicitly NOT recommended:** Lyu et al.'s validation-set-defined utility. Our dev splits are
n = 107/78/80, and the campaign has already measured that a 78-item dev selection costs ~2 accuracy
points of noise (F56/F45). Any `α` fitted against dev utility will fit noise. If C5 runs at all, `α`
must be defined by a **train-LOO, label-supervised, closed-form** statistic — never by dev selection.

### (f) Honest risk

**Most likely killer: filter (i) plus the F78 precedent.** The continuous-vs-binary gap is real but
small, and F89's T1 already showed that reweighting the *class* composition of the retrieved set can
be exactly degenerate.

**Its actual value is elsewhere, and that is why it is on the list at all:** a per-entry reliability
weight is a first-class object for **pillar ④ (auditable/editable archive memory)** — it is the
natural quantitative companion to the human-in-the-loop deletion story, and F95 explicitly left the
pair verifier legal as an **evidence ranker** for that pillar. If C5's accuracy read is null, the
record is still usable as method-chapter material. Run it last, or run it as a ride-along inside C2's
pregate at zero marginal cost.

---

## §6. RANKING

Ordered by (mechanism-novelty for the paper × survival probability). All five pregates are $0/CPU on
banked train-split features; **none of the five needs a GPU even in its full version** — a change from
every previous litsweep, and a direct consequence of restricting the lens to the bank and the vote.

| # | candidate | pathology | novelty | survival | pregate | full version | first thing that kills it |
|---|---|---|---|---|---|---|---|
| **C1** | nuisance-residual vote (de-bias the **label field**) | CP1 | high — no dead entry has ever touched the transported quantity | **highest** | $0 CPU | 0 GPU-h, 0 new params | `sd(p̂)` ≈ 0 ⇒ it is a global threshold (bar 3) |
| **C2** | cell-conditional **synthesis into the bank** | CP1 | **highest** — see §9 | medium-low | $0 CPU | 0 GPU-h | purity on the pathology population does not move (bar 3); Blagus-Lusa near-duplicates |
| **C3** | learned aggregation over the **neighbourhood profile** | CP2 | medium-high | medium | $0 CPU | 0 GPU-h | learned profile is monotone everywhere ⇒ re-derived F94 (bar 3) |
| **C4** | **aggregate-then-compare** subspace residual | CP2 | high | medium-low | $0 CPU | 0 GPU-h | both class residuals degenerate at every rank r (bar 3) |
| **C5** | per-entry **soft reliability weights** | CP1/CP3 | low-medium | low | $0 CPU | 0 GPU-h | filter (i): reweighting an empty support is still empty |

**Recommended order:** C1 and C4 first — both are $0, both are decisive within their own bars, and
they attack different pathologies (CP1 field, CP2 form) so they can run in parallel without
interfering. C3 next, but **only after its bar-2 control is agreed**, since without it a positive is
uninterpretable. C2 is the paper's best story and the one worth a real prereg, but it should be
written *after* C1's result, because if C1 shows the label field is correctable then C2's placement
criterion can use `p̂` and becomes much better targeted. C5 as a ride-along inside C2.

**Composition note:** C1 × C3 is orthogonal by construction (what is summed × how it is weighted) and
C2 is upstream of both. If two of the three clear their bars individually, the composition is a legal
single ceremony rather than three.

---

## §7. VERIFIED AND REJECTED — the sub-areas the brief asked for that I am recommending against

Each of these was searched and the frontier verified; each is rejected on measured or arithmetic
grounds, not on effort. Recording them so they are not re-swept.

### §7.1 Hubness correction beyond CSLS (2024-2026) — **closed by arithmetic**
Frontier: **Nearest Neighbor Normalization Improves Multimodal Retrieval** (Neil Chowdhury,
Franklin Wang, Sumedh Shenoy, Douwe Kiela, Sarah Schwettmann, Tristan Thrush, **EMNLP 2024**,
arXiv:2410.24114) and **NeighborRetr** (CVPR 2025). The F95 §4.1 decomposition looks like an
invitation here, but filter (iv) closes it: the query-side offset is rank-invariant *within* a query
and cannot flip a prediction, and the bank-side offset is what CSLS removes — measured **inert**
(F89 T2a). NNN is a one-sided additive variant of the same correction. **Predicted inert; report the
arithmetic instead of running it.**

### §7.2 Coreset / bank-construction objectives (PatchCore lineage) — **dead end**
Everything verified in 2025-2026 is efficiency, memory-bounding or streaming: ASO PatchCore
(SCITEPRESS 2026), Sequential PatchCore (arXiv:2501.09579), CADIC (arXiv:2511.08634), on-device
continual variants (arXiv:2512.13497). **CADIC's own result is that incremental coreset ≈ greedy
k-center and both merely beat random** — the field's evidence says the selection objective is not the
binding constraint. FSLC (BMVC 2025, proceedings p. 922) builds its bank by test-time score-based
filtration = transductive deletion curation = our measured null. Also verified and excluded:
**Mahalanobis PatchCore** (arXiv:2605.27748) is global whitening — our measured-negative operator,
and its gains are memory not accuracy; **CIF** (AAAI 2026, arXiv:2511.05966) is training-free
hypergraph message passing over the memory graph — our killed F63 diffusion.

### §7.3 SMOTE-family interpolation as a primary lever — **expect measured null**
Verified frontier is genuine (Simplicial SMOTE, KDD 2025, arXiv:2503.03418; survey arXiv:2502.08960),
but at top-1 cosine 0.94-0.99 the convex hull of a 5-20-item cell is nearly a point, so interpolants
are near-duplicates that consume top-20 slots without adding information — worse than nothing in a
*rank-weighted* vote. Blagus & Lusa (2013) is the published hard negative for exactly SMOTE + k-NN in
high dimensions. **Kept inside C2 as a secondary arm only, ranked below translation and
statistics-transfer.**

### §7.4 Conformal / trimmed / robust aggregation — **structurally the wrong tool**
Verified frontier: **DANCE: Doubly Adaptive Neighborhood Conformal Estimation** (Brandon R. Feng,
Brian J. Reich, Daniel Beaglehole, Xihaier Luo, David Keetae Park, Shinjae Yoo, Zhechao Huang,
Xueyu Mao, Olcay Boz, Jungeum Kim, arXiv:**2602.20652**, 24 Feb 2026, stat.ML, no venue/code stated) —
a doubly locally-adaptive nearest-neighbour conformal algorithm using two nonconformity scores on the
embedded representation. It is real, recent and well-matched to our object, and it **produces
prediction sets, not point predictions**: its currency is set-size efficiency under a coverage
guarantee. Our bar is **point accuracy (+0.030 on ≥2 datasets)**, which conformal machinery does not
target and cannot deliver. Also: conformal calibration needs a held-out calibration split, and our
dev splits are n = 107/78/80. **Rejected for the performance goal.** Retained as a *possible*
selective-prediction/abstention story for the auditability pillar, where coverage is the right
currency — but that is a different claim object and needs a user ruling, since selective prediction
changes what the main table means. The genuinely transplantable half of "robust aggregation" —
**median/trimmed statistics over the retrieved evidence** — survives inside **C4** via ProCon.

### §7.5 Datastore pruning / compression / distillation (kNN-MT lineage) — **dead end**
Every headline verified is *retain quality at lower cost*, never *improve*: arXiv:2211.04052
(Findings of ACL 2023), arXiv:2204.06175 (ACL 2022), arXiv:2109.04212 (EMNLP 2021). Their datastores
are 10⁶-10⁹ entries; ours is 549-744, where there is no redundancy to harvest — and deletion-based
pruning is already our measured null.

### §7.6 Datastore key **revision** (RevisedKey / INK) — **deprioritised, adjacent to closed**
**Bridging the Domain Gaps in Context Representations for kNN-MT** (arXiv:2305.16599, ACL 2023) and
**INK** (arXiv:2306.06381, ACL 2023) move the **datastore keys one-sidedly** — query and its
datastore twin may end up in different places — which is genuinely not NCA (F75, which applies one
shared map to both sides). But it is a learned re-metrication of a 7168-d key space fitted on 549-744
items: the highest overfitting risk in the sweep, on the axis nearest to closed territory. Also noted:
**Revisiting Nearest Neighbor for Tabular Data** (arXiv:2407.03257, ICLR 2025) is a modernised NCA —
same axis F75 closed. **Do not spend before C1-C4.**

### §7.7 Generative feature-space synthesis (diffusion / flow) — **dead end at our n**
Verified: LatentDiff (arXiv:2509.23240), latent diffusion for long-tail (arXiv:2404.04517, L3D-IVU @
CVPR 2024 workshop), latent-space flow for tabular (arXiv:2511.16571). Conceptually the best-matched
mechanism in the whole sweep — feature-space, covariate-conditional, region-targeted — and every one
fits its generator on 10³-10⁶ features. At 549-744 vectors in 7168-d a diffusion/flow model
memorises. The transferable finding is the tabular paper's own ("smaller embeddings improve minority
recall"), which collapses to "reduce dimension, then use the cheap parametric methods" = C2.
**Do not spend GPU here.**

### §7.8 Datastore scaling — **closed by user constraint, not measurement**
arXiv:2407.12854 (NeurIPS 2024) reports monotone gains with a trillion-token datastore. Requires
external corpora; banned 2026-07-14.

### §7.9 TabR — **flagged as out-of-budget, one piece salvageable**
**TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023** (Yury Gorishniy, Ivan Rubachev,
Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, Artem Babenko, arXiv:**2307.14338**, ICLR 2024,
code `https://github.com/yandex-research/tabular-dl-tabr`). Its value module is built on the
**difference vector** `W_K(x̃) − W_K(x̃_i)` — a pure interaction object, invariant to additive
per-query offsets, which is directly responsive to the F95 §4.1 finding, and it *re-values* each
neighbour rather than *re-ranking* it (so it is not F95's pair scorer). But its experimental grid runs
~10k-1.2M objects, ~15× our budget at the smallest. **Only the minimal form** — a rank-8-32 key
projection plus a small function of the difference, heavily regularised — is in scope, and that
belongs as a declared arm inside C3 rather than as its own candidate.

---

## §8. CITATION VERIFICATION LOG

Every row was verified by fetching the URL shown during this sweep and reading the abstract.
"Venue" records what the fetched page stated; where a venue or code URL was **not** on the page it is
marked, so no claim rests on an unverified field.

| paper | arXiv / DOI | fetched | venue on page | code URL on page |
|---|---|---|---|---|
| FT2Ra | 2404.01554 | ✔ | ISSTA 2024 (comments) | no |
| Adaptive kNN-MT (Meta-k) | 2105.13022 | ✔ | ACL-IJCNLP 2021 | yes (relay: zhengxxn/adaptive-knn-mt) |
| FeTrIL | 2211.13131 | ✔ | cs.CV only — WACV 2023 from CVF page | no (GitHub only) |
| Free Lunch / Distribution Calibration | 2101.06395 | ✔ | ICLR 2021 | no |
| Delving into Deep Imbalanced Regression (FDS) | 2102.09554 | ✔ | ICML 2021 Long Oral | yes |
| Long-Tail Crisis in kNN-LM | 2503.22426 | ✔ | Findings of NAACL 2025 | not on page |
| ProCon | 2607.04894 | ✔ | **none (preprint)** | yes (jw-chae/Procon) |
| SubspaceAD | 2602.23013 | ✔ | CVPR 2026 (comments) | yes (CLendering/SubspaceAD) |
| StructCore | 2602.17048 | ✔ | **none**, cs.CV | **none** |
| SoftPatch+ | 2412.20870 | ✔ | Pattern Recognition | yes (Tencent Youtu) |
| PNI | 2211.12634 | ✔ | cs.CV — ICCV 2023 from CVF/IEEE | not on page |
| Simplicial SMOTE | 2503.03418 | ✔ | KDD 2025 research track | no |
| Great Memory, Shallow Reasoning | 2408.11815 | ✔ | cs.CL preprint | yes |
| KSTER | 2109.09991 | ✔ | EMNLP 2021 | yes |
| RelationNet | 1711.06025 | ✔ | CVPR 2018 | not on page |
| DANCE | 2602.20652 | ✔ | **none**, stat.ML | **none** |

Verified via relay only (sub-agent fetched, I did not re-fetch — **treat as needing re-verification
before any of these enters a prereg or a paper**): SoftPatch NeurIPS 2022 (2403.14233), DBG CVPR 2026
(2605.01468), Bias-Corrected Data Synthesis (2510.26046), FeCAM NeurIPS 2023 (2309.14062), TabR
(2307.14338), RevisedKey (2305.16599), INK (2306.06381), Lyu et al. (2307.03027), CRAD ECCV 2024
(2402.18293), DINOSaur/CAD (2605.24251), Blagus & Lusa BMC Bioinformatics 14:106 (via PMC3648438),
NNN EMNLP 2024 (2410.24114), and every item in §7.2/§7.5/§7.7.

---

## §9. THE NOVELTY OBSERVATION WORTH KEEPING

Independent of which candidate survives, the sweep turned up one gap worth stating in the paper.

**Nobody synthesises into the retrieval structure.** Across the SMOTE lineage, feature-space
diffusion, distribution calibration, long-tail geometry and group-robustness literatures, synthesis
targets a **training set** — points are generated so a classifier can be *fitted* on them. I found no
verified 2024-2026 work that inserts synthetic entries **into a retrieval index / memory bank /
datastore** and measures **retrieval quality** as the outcome. (The nearest verified neighbour, SuS-X,
ICCV 2023, arXiv:2211.16198, constructs a support set from an external generator or LAION retrieval —
both banned here.) Our setup can measure exactly that, and already owns the metric: top-20 true-label
purity on the pathology population, currently 0.12-0.22.

Two further items are paper-grade regardless of outcome. **The AD memory-bank lineage cannot state
CP1** — it is normal-only, so there is no second class whose local prior can be skewed by a nuisance
covariate; the closest published thing is position-conditioning (PNI; Position-Aware PatchCore,
*J. Japan Soc. Precision Engineering* 91(12):1130-1135, 2025). And **the kNN-LM lineage has
independently measured our law**: "Long-Tail Crisis" (Findings of NAACL 2025) shows the retrieved
distribution tracks the datastore's marginal rather than the query's true class, with the failure
concentrated where the marginal is against you — our `0.11 → 0.55` length-stratified prior with
`frequency` replaced by `length`; and "Great Memory, Shallow Reasoning" (arXiv:2408.11815) shows via
**oracle retrieval** that kNN-LMs still fail with perfect retrieval. That is our
"the right analogue is retrieved at rank ~1.5 and then out-voted", published, in another modality.
The F95 §4.1 variance decomposition is the sharper, quantitative version of the same claim and is
**ours**.

---

## §10. LIMITATIONS OF THIS SWEEP

1. **No measurement of any kind.** Every survival estimate is an argument, not a number. The five
   pregates exist precisely because these arguments are not evidence.
2. **Relay verification for the §8 lower block.** Sixteen citations I re-fetched myself; the rest were
   fetched by sub-agents. They must be re-verified before entering a prereg or a paper.
3. **KSTER's per-query bandwidth claim is body-level, not abstract-level** (§3(a) note 2) and is not
   yet verified from the PDF.
4. **Meta-k's "~100 training sentences" figure is relay-only.** The fetched abstract says only "a few
   training samples". Do not cite the number without checking the PDF.
5. **C3's admissibility is a judgement call, not a measurement.** Whether "continuous per-neighbour
   weights" escapes the F47/F49/F66 selection ban should be ruled on before its pregate runs.
6. **C2's ban-distinctness is the load-bearing legal claim in this document.** Within-class
   translation along the nuisance axis is, in my reading, clearly outside the pseudo-label ban — but
   it is close enough that a user ruling before spending is the right sequence, and M2m-style
   cross-class translation is excluded here for exactly that reason.
7. **Sub-area coverage is uneven.** Industrial AD, kNN-LM/MT and feature synthesis were swept in
   depth; conformal aggregation and the RelationNet lineage were swept by me directly but more
   briefly (two searches plus targeted verification), and hubness was closed by argument after
   locating the frontier rather than by exhaustive search.
8. **No test-split quantity, no oracle, no dataset file of any split was read.** Nothing was written
   outside `refine-logs/`; `state/` untouched.

---

## ⚠ ERRATUM (appended 2026-07-28, closeout) — the inherited "head memorises train at LOO ≈ 0.998" premise is a **CLIP** number

**No verdict moves.** This is a framing correction to an inherited premise, not to any measurement
taken in this record.

**The error.** This record repeats — from `mechnov_pairverify.py:21-25` / F95 — that the trained RGCL
head *"memorises its own train split (LOO train acc 0.998, F47)"*, and uses it to justify screening in
the **raw** encoder key space rather than the deployed head space.

**0.998 is F47's CLIP head, not the deployed head.** F47's own `ban_scope`
(`directions_tried.json:171`) reads *"train-supervised = memorization-degenerate target, **CLIP LOO
0.998**"*, and the memory index pairs it with *"vs **Qwen 0.800**"*. The deployed system does not use
the CLIP head.

**The deployed Qwen heads, newly computed** (`INSTRUMENT_VALIDATION_RECON.md` §0.2, F111; re-read from
`scripts/analysis/mechfix_{hatemm,zh,en}_OUT.json` → `train_side_sanity.deployed_loo_train_acc`):

| | HateMM | MHC-ZH | MHC-EN |
|---|---|---|---|
| **deployed head train LOO** | **0.9406** | **0.8915** | **0.8154** |
| raw-arena deployed train LOO | 0.8441 | 0.8480 | 0.7796 |
| gap between the two arenas | +0.0965 | +0.0435 | +0.0358 |

**The two arenas differ by 3.6–9.7 accuracy points on the same train items, not by the 0.998-vs-0.84
chasm the premise asserts.** The argument *"a train-side screen in head space measures memorisation"*
is therefore **weaker than stated — downgraded, not vacated**: 0.9406 against a 0.8441 raw floor still
means the head reproduces its own train split far better than its deployed test behaviour.

**CONSEQUENCE 1 — the raw-space screening justification is superseded.** The saturation claim applies
**only to full-train LOO**. `HEADSPACE_TRANSFER_PREGATE.md` (F113) demonstrates the fix nobody used:
**train the head on 4/5 of the train split and query it with the held-out fifth.** That **fold-head
arena is unsaturated**, is a strictly better proxy for deployment than the raw arena, and costs
**~35 s of CPU per fold-head**. The existing `mechfix_ops` / `vsw_pregate` battery runs in it
unmodified. **The head space was available the whole time**, and F113 recommends it become the default
`$0` pregate arena.

**CONSEQUENCE 2 — F107's Q1 argument depended on this figure and has been adjudicated.**
`HEADCOV_PREGATE_RECORD.md` §6.1 claimed *"the objective is already at its optimum on its own training
signal, with ≤0.002 of headroom"*. On the corrected figures the remaining train-side headroom is
**0.0594 / 0.1085 / 0.1846** — 30× / 54× / 92× larger. That step is **RETRACTED**; F107's conclusion
(the metric channel is closed) **survives but is SCOPED and WEAKENED — it is now empirical, not
analytic**, resting on the F75/NCA isomorphism (a measured GPU negative) plus a weak observational
conversion bound (R² = 0.027, MHC-ZH dev only) plus F113's head-space fitting evidence. **F107 must no
longer be cited as a theory-level door-closer.** See `HEADCOV_PREGATE_RECORD.md` §6.1 ERRATUM in full.

**Provenance note.** `scripts/analysis/mechnov_pairverify.py:21-25` still carries the wrong premise and
has been **deliberately left byte-identical**: its sha256 `77b0defd…b7240d` is asserted at run time by
five scripts, so editing even a comment would break the reproducibility of F95, F97, F98, F105, F112
and F113 at once. The correction lives in `MECHNOV_PAIRVERIFY_PREGATE.md` §E.1–E.3.

*Authority: `INSTRUMENT_VALIDATION_RECON.md` §0.2 (F111) · `HEADSPACE_TRANSFER_PREGATE.md` §8 (F113).
Ledger: F114. `$0` — no GPU, no SLURM, no Modal, no training, no test contact.*
