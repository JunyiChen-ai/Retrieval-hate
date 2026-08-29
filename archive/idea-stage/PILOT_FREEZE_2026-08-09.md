# Pilot freeze — idea-discovery Phase 2, 2026-08-09

**Written BEFORE any candidate metric was computed.** Three pilots, all CPU-level, all zero
test-set contact, each a single submission. Decision rules below are frozen and will be
transcribed unedited into the results.

Red lines in force: (1) zero test-set contact; (2) decision rules frozen before results are seen;
(3) blind design — no candidate endpoint was computed while designing or implementing;
(4) one submission per pilot, no re-runs, no tuning after seeing numbers.

Logging convention: `logging/runs/<name>/run.{log,pid}`, background process.

---

## P-A — Disagreement retrievability gate

**What family it gates.** Every vote-based candidate (agreement-shaped retrieval geometry,
dissent-preserving memory, contested-item abstention). All of them assume a **necessary
condition**: that an item's *contestedness* is predictable from its retrieval neighbourhood.
If a neighbourhood cannot carry disagreement, none of the family has a mechanism.

**Data.** MultiHateClip EN (train 549 + val 80 = 629) and ZH (train 579 + val 78 = 657).
`test.jsonl` is never opened. Votes joined from the upstream official release
`{English,Chinese}_data/annotation/{train,valid}.tsv` by `Video_ID` (join verified 100 %).

**Features.** Frozen CLIP ViT-L/14-336 caches already on disk:
`data/CLIP_Embedding/{MHC,MHC_zh}/{train,dev_seen}_openai_clip-vit-large-patch14-336_HF.pt`.
Key = `[l2(img_feats) ‖ l2(text_feats)]`, similarity = dot product (= sum of the two per-block
cosines). This is arm 0 of the late-interaction pilot, chosen so the result is comparable to it.

**Targets (both frozen, both binary, computed per item from the raw vote list).**
- **T1 `non_unanimous`** — the vote multiset contains ≥2 distinct labels.
- **T2 `binary_split`** — the votes disagree after mapping to this project's binary protocol
  (`Normal`, `Counter Narrative` → 0; `Offensive`, `Hateful` → 1). T2 ⊂ T1.

**Predictor.** Leave-one-out over the pooled train+val set. For query *i*, retrieve k=20 nearest
neighbours (excluding *i*), score
`s_i = Σ_j w_ij · t_j / Σ_j w_ij`, where `t_j` is the neighbour's contestedness indicator and
`w_ij` is the similarity. **Similarity-weighted mean, deliberately continuous and
non-saturating** — per the P2 forensic transferable rule, a bounded neighbour *count* is
degenerate by construction and must not be used as a selection score.
Neighbour ordering ties broken lexicographically by `video_id`.

**Endpoints.**
- **E1** — AUROC of `s_i` predicting T1, per language.
- **E2** — AUROC of `s_i` predicting T2, per language (secondary).
- **E3 (the discriminator)** — AUROC of a **label-only hardness baseline** predicting T1:
  `h_i = 1 − |p_i − 0.5| · 2` where `p_i` is the similarity-weighted fraction of harmful-labelled
  neighbours. This is what a system with **no access to votes** already knows.
  The reported increment is `Δ = AUROC(s) − AUROC(h)`.
- **Null control** — repeat E1 with the contestedness targets randomly permuted (seed 20260909);
  AUROC must land within [0.45, 0.55] or the pilot is void.
- Uncertainty: 2000-resample bootstrap over queries, same draws for `s` and `h` (paired).

**Frozen decision rule.**
- **GO** — in **both** languages: `AUROC(s) on T1 ≥ 0.60` with bootstrap 95 % LB `> 0.55`,
  **and** `Δ ≥ +0.03`.
- **AMBIGUOUS** — the AUROC bar is met in both languages but `Δ < +0.03`; or the full GO
  condition holds in exactly one language.
- **NO-GO** — `AUROC(s) < 0.60` in either language, or `Δ ≤ 0` in both.

**Reading.** NO-GO means the vote data, though real, is not *retrievable* — the disagreement
family loses its mechanism and must be closed. AMBIGUOUS with high AUROC but low Δ means
contestedness is real but already implied by label geometry, i.e. the votes add no new
information and the idea reduces to hardness-aware retrieval, which is not novel.

---

## P-B — Near-duplicate and label-conflict census

**What it gates.** The "duplicate-conflict memory" candidate and the "naturally occurring minimal
pairs" slot; it is also a validity check on every retrieval number this project has produced
(if reposts are casting repeated votes, neighbourhood purity is inflated).

**Data.** **Train splits only**, all datasets with cached CLIP embeddings: HateMM, MHC-EN,
MHC-ZH, HateClipSeg, ImpliHateVid. `dev_seen`/`val`/`test` are not opened.
**The train↔test contamination audit is deliberately NOT run here** — it would require test-set
contact. It is deferred to a separately authorised, separately pre-registered step.

**Similarity.** Two frozen measures computed on L2-normalised blocks:
`c_img` (visual cosine) and `c_txt` (transcript/title text cosine). A repost re-encoded or
re-narrated keeps visual similarity while text may drift, so `c_img` is primary.

**Endpoints.**
- **N1** — count of pairs at `c_img ≥ {0.85, 0.90, 0.95}`, within-dataset and cross-dataset.
- **N2** — of those, **conflicting pairs**: near-duplicate pairs whose binary labels differ.
- **N3** — among MHC conflicting pairs, how many have a `Counter Narrative` vote on one side.
- **N4 (discriminator against false positives)** — for every flagged pair, token-level Jaccard
  overlap of the transcripts. A genuine repost shares transcript; two unrelated talking-head
  videos do not. Report the conflicting-pair counts **additionally filtered** at
  `Jaccard ≥ 0.5` as the conservative count.

**Frozen decision rule** (on the conservative count: `c_img ≥ 0.90` **and** `Jaccard ≥ 0.5`):
- **ALIVE** — ≥ 30 conflicting pairs across the train pools.
- **AMBIGUOUS** — 10–29 conflicting pairs (would need manual verification of a sample).
- **DEAD** — < 10 conflicting pairs.

**Known limitation, stated in advance.** High cosine on mean-pooled 8-frame CLIP features is
**not** proof of near-duplication; same-genre talking-head footage can reach 0.9. N4 is the
mitigation, and any surviving claim requires visual verification of a sample. The census
measures an upper bound on duplication and a lower bound on verified conflict.

---

## P-C — On-screen-text provenance separability

**What it gates.** The "provenance-typed OCR" candidate. Prior bounds: mean-pooled untyped OCR
into a linear head = **+0.0094** (AMBIGUOUS, sub-threshold); the same vector through the learned
fusion MLP = **−0.0246** (NO-GO); OCR inside a retrieval key with a max aggregator = **−0.018**.
The open question is whether the OCR channel is weak because it is *undifferentiated*.

**Data.** HateMM **train only** (744 videos), the existing OCR cache `ocr_windows_K30.jsonl`
with per-detection boxes and confidences, SHA-256 verified against `data/OCR/SHA256SUMS.json`.
Reuses the frozen OCR fusion pilot's folds, head, seeds and filter (`conf ≥ 0.5, len ≥ 2`).

**Provenance typing rule (frozen, unsupervised, no label access).** Group detections across the
30 windows into tracks by normalised box-centre proximity (≤ 0.05 in both axes) and text
similarity (token Jaccard ≥ 0.6). A track is **overlay-like** if it persists in ≥ 50 % of the
windows that contain any text **and** its box-centre standard deviation is ≤ 0.05 in both axes.
All other detections are **scene-like**.

**Arms** (identical folds / head / seeds to the OCR fusion pilot; three seeds):
| arm | input |
|---|---|
| 0 | baseline `[l2(img) ‖ l2(txt)]`, 1792-d |
| 1 | + untyped mean-pooled OCR block (replicates the +0.0094 result), 2560-d |
| 1c | **parameter-matched control**: the untyped OCR block duplicated into two blocks, 3328-d |
| 2 | + **typed**: overlay-mean block ‖ scene-mean block, 3328-d |

Arm 1c exists so that arm 2 − arm 1c isolates provenance typing from the added dimensionality —
the capacity confound that the A0 ±OCR pilot could not separate.

**Endpoints.**
- **O1** — descriptive: share of OCR text mass classified overlay vs scene; coverage per class.
- **O2** — AUROC of overlay-text-presence and of scene-text-presence against the video label,
  separately (descriptive, non-gating).
- **O3 (gating)** — OOF macro-F1 over the 744 train videos, seed-paired deltas.

**Frozen decision rule** (primary quantity = seed-mean `arm2 − arm1c`):
- **GO** — `≥ +0.010` and positive on 3/3 seeds.
- **AMBIGUOUS** — `+0.003 … +0.010`, or mixed sign with a positive mean.
- **NO-GO** — `≤ +0.003`.

**Reading.** NO-GO means on-screen text's weakness in this pipeline is not a typing problem, and
slot #7 closes. GO means provenance is a real axis and is worth a mechanism-level design.

---

## What no pilot here can establish

None of these three produces an accuracy claim, and none touches a test split. P-A and P-B are
necessary-condition gates: passing them does not make an idea publishable, it only means the idea
still has a mechanism to build on. P-C bounds one fusion of typed OCR, not the typing concept.

---

# P-A-v2 — strong-baseline retest of the disagreement retrievability gate

**Appended 2026-08-09, written BEFORE any P-A-v2 quantity was computed.** Blind: no arm's
endpoint existed at the time this section was written; the implementation was smoke-tested on
synthetic data and on label-permuted data only. Single submission, no tuning after results.

## Why

P-A returned GO, but its discriminator `E3` was a 20-NN similarity-weighted label fraction — a
baseline with no trained parameters. The P-A result section states this limitation itself
("the hardness baseline may be unfairly weak"). P-A-v2 replaces that baseline with a **trained
discriminative disagreement predictor** and asks the sharper question: *does the retrieval
neighbourhood carry disagreement information beyond what a proper classifier extracts from the
same frozen features?* This retest can only lower the standing of the vote-retrieval family; it
cannot raise it.

## Inherited unchanged from P-A

- **Data.** MHC-EN train+val (629) and MHC-ZH train+val (657), pooled per language.
  `test.jsonl` never opened; the path guard that HALTs on any path containing `test` is re-armed.
- **Features.** Same frozen CLIP ViT-L/14-336 caches, same key
  `X_base = [l2(img_feats) ‖ l2(text_feats)]` (1792-d), same dot-product similarity.
- **Targets.** Identical to P-A, recomputed by the same code path:
  **T1 `non_unanimous`** (≥2 distinct raw vote labels) is the **primary and only gating target**;
  **T2 `binary_split`** is reported as a secondary, non-gating endpoint.
  Same vote parsing, same alias map (`No` → `Normal`), same binary protocol.

## Protocol change (the whole point of v2)

P-A used leave-one-out with the neighbour pool = every other item. Trained arms cannot be
evaluated that way at reasonable cost, and LOO would also give the retrieval arm a pool the
trained arms do not have. **v2 puts every arm on one shared cross-validated protocol:**

- **Folds.** Stratified 5-fold on T1, per language. **3 fold-randomisation seeds**:
  `20260910, 20260911, 20260912`. All arms see byte-identical folds within a seed.
- **Neighbour pool is fold-restricted.** For a held-out query, the k=20 neighbours are drawn from
  the **training folds only**. For a training row (needed to fit the retrieval arm), neighbours
  are drawn from the training folds excluding itself. So no arm ever sees a held-out item's votes.
  This is *stricter* than P-A's LOO pool and the two numbers are therefore not interchangeable;
  the fold-restricted version of P-A's original scalar is reported for continuity.
- **Everything is OOF.** Every reported score for item *i* comes from a model that never saw *i*.

## Feature blocks (frozen)

- **`X_base`** — 1792-d, as above.
- **`X_unc`** — 2-d model-uncertainty block: `[H(p), |p − 0.5|]` where `H` is binary entropy and
  `p` is an **out-of-fold probability of the binary harmful label** from an A0-style predictor.
  No A0 OOF prediction file exists for MHC, so the cheapest path is taken and frozen: the same
  logistic-regression recipe below, fit on `X_base` against the binary label, **nested** — within
  each outer training fold an inner stratified 5-fold produces `p` for the training rows, and the
  model refit on the whole outer training fold produces `p` for the held-out rows. Uses **labels
  only, never votes**.
- **`X_nbr`** — 8-d retrieval-neighbourhood vote block, computed on the fold-restricted pool with
  P-A's similarity weights `w`:
  1. `s_T1` — similarity-weighted mean of neighbour T1 (**this is P-A's original scalar `s`**)
  2. `s_T2` — similarity-weighted mean of neighbour T2
  3. `u_T1` — unweighted mean of neighbour T1
  4. `sd_T1` — unweighted std of neighbour T1
  5. `s_frac` — similarity-weighted mean of the neighbour's **harmful-vote fraction**
  6. `s_disp` — similarity-weighted mean of the neighbour's vote dispersion `2f(1−f)`
  7. `w_mean` — mean neighbour similarity (neighbourhood density)
  8. `w_max` — max neighbour similarity
  Blocks 5–6 are vote-level, not label-level: they are what a vote-free system cannot compute.

## Arms

| arm | input | what it is |
|---|---|---|
| **B1** | `X_base` | trained hardness/disagreement baseline on the frozen features |
| **B2** | `X_base ‖ X_unc` | B1 + the model's own uncertainty (the stronger baseline) |
| **C** | `X_nbr` | the candidate: retrieval-neighbourhood vote signal, trained |
| **D** | `X_base ‖ X_nbr` | combination, for the increment over B1 |
| *C0* | `s_T1` used directly as a score | descriptive only, non-gating: P-A's scalar under v2 folds |

**Classifier (frozen, one recipe for every arm so no arm is advantaged).**
`sklearn.linear_model.LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000,
class_weight='balanced')`. Features standardised with outer-training-fold mean/std. Inverse
regularisation strength selected **per outer fold by inner stratified 5-fold CV** over
`C ∈ {0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}`, maximising inner mean AUROC. Logistic regression
is chosen over the shallow MLP; that choice is frozen here and not revisited.

## Endpoints

- **A1 (gating)** — OOF AUROC of each arm predicting **T1**, per language, **averaged over the
  3 seeds**.
- **A2** — the same on T2 (secondary, non-gating).
- **Increments** — `C − B1`, `D − B1`, and, non-gating, `C − B2` and `D − B2`.
- **Uncertainty** — paired bootstrap over queries, **2000 resamples, seed 20260913**, identical
  draws for every arm. Within a resample, each arm's AUROC is computed per seed and then averaged
  over the 3 seeds; the increment for that resample is the difference of those seed-averaged
  values. Percentile 95 % CI.
- **Null control** — repeat arm **C** with T1 permuted (seed `20260909`, as in P-A) at fold seed
  `20260910`; OOF AUROC must land in [0.45, 0.55] or the retest is flagged VOID.

## Frozen decision rule

Per language *L*, on T1, seed-averaged point estimates:

```
condition(L) :=  AUROC(C, L) >= AUROC(B1, L)
             AND lower bound of the 95% CI of (D - B1) at L  >  0
```

- **GO-STRONG** — `condition(L)` holds in **both** languages.
- **GO-ZH-ONLY** — `condition(L)` holds **only** in ZH.
- **GO-EN-ONLY** — `condition(L)` holds **only** in EN.
- **KILL** — `AUROC(C, L) < AUROC(B1, L)` in **both** languages. Reading: the neighbourhood signal
  is only a shadow of feature quality, and the Human-Agreement Retrieval family loses its
  mechanism.
- **AMBIGUOUS** — anything else (e.g. C beats B1 somewhere but no increment CI clears zero).

Precedence, since the clauses can co-fire: `GO-STRONG > GO-ZH-ONLY / GO-EN-ONLY > KILL >
AMBIGUOUS`. The commissioning brief also admits a looser literal reading — "condition holds in at
least one language → GO-STRONG". That reading is **also** reported as a raw flag
(`literal_at_least_one_flag`) so the record can be re-adjudicated without re-running.

## What P-A-v2 cannot establish

No accuracy claim, no test-set number. A KILL closes the vote-retrieval family's *mechanism*
claim; it does not say the votes are uninformative, only that a trained model on the same frozen
features already extracts what the neighbourhood carries. A GO does not make the family
publishable — it restores the necessary condition against a fair baseline, nothing more.

---

# LEG2-KILL — capped kill test of Human-Agreement leg (ii): agreement-weighted contrastive training

**Written 2026-08-09, BEFORE implementation and BEFORE any candidate metric was computed on real
data.** The only real-data quantity looked at before this document was fixed is the timing of one
synthetic head fit (1.5 s), used to size the compute budget.

## Standing declaration (reproduced verbatim in the result file and in IDEA_REPORT §6.1)

> This experiment is an **adaptively selected** hypothesis. It was chosen *after* the frozen
> P-A-v2 gate failed in both languages, because leg (ii) is the only leg that gate did not touch.
> It **inherits no prior GO**. The original P-A / P-A-v2 gate **stays failed regardless of this
> outcome**. A positive result grants the label **"exploratory"** only — never "recommended",
> never a main-conference claim, and it remains single-dataset, adaptively-selected evidence for a
> low-novelty kernel instantiation until independently confirmed on data this project has not
> touched. A negative result **permanently closes the entire Human-Agreement family**
> (legs i, ii, iii), with no revival branch.

External-review conditions this design is executing (IDEA_REPORT §6.5, §6.8): new pre-registration;
primary comparator = **GenSCL/LDL distributional contrastive**, *not* hard-label RGCL; all loss /
temperature / smoothing degrees of freedom frozen; **futility rule — one failure closes it**;
placebo arm mandatory (§6.7 item 7).

**Defects deliberately NOT fixed.** §6.7 items 1 (similarity/certainty conflation), 2 (4-class
geometry vs binary task), 3 (marginalised hard-label equivalence), 4 (contestedness confounded with
vote count), 5 (`q` is a 2-vote histogram, not a population distribution). The kill test measures
the **original mechanism as specified**. A corrected kernel is a different experiment and is out of
scope for this round.

## Data

MultiHateClip EN and ZH. Pooled `train + val` only, exactly the sets P-A/P-A-v2 used
(EN 549+80 = 629, ZH 579+78 = 657); `test.jsonl` is never opened and P-A's path guard is re-armed
(any path component containing `test` HALTs). Votes from the official release
`data/gt/mhc_votes/mhc_{English,Chinese}_{train,valid}.tsv`, parsed by the **unchanged** P-A
functions (`parse_votes`, `load_votes`, `load_lang`), including the frozen alias `No → Normal`.
Features: the frozen CLIP ViT-L/14-336 cache, `X = [l2(img_feats) ‖ l2(text_feats)]`, 1792-d,
standardised with outer-training-fold mean/std. **Everything is out-of-fold; no test contact.**

**Vote distribution `q_i`** — the empirical 4-class vote histogram over the frozen class order
`[Hateful, Offensive, Normal, Counter Narrative]`, normalised to sum 1. **No smoothing, no prior,
no re-weighting** (§6.7 item 5 is a stated defect, not a thing to repair here).

## Model (one architecture, one optimiser, identical for every arm)

```
trunk : Linear(1792 -> 128) -> ReLU -> Dropout(0.2)      -> h
proj  : Linear(128 -> 64) -> L2-normalise                -> z   (contrastive space)
clf   : Linear(128 -> 1)                                 -> logit
```
Adam, lr 1e-3, weight_decay 1e-2, **400 full-batch steps** (the outer training fold is one batch, so
the anchor set `A(i)` is every other training item). PyTorch default init, seeded by
`fold_seed + fold_index`. Classification loss = `BCEWithLogits` with
`pos_weight = n_neg/n_pos` computed on the outer training fold. Decision threshold **0.5**, frozen.

## The contrastive objective (GenSCL Eq. 2, verbatim form)

Kim, Lee, Chang & Park, *Generalized Supervised Contrastive Learning*, arXiv **2206.00384**, Eq. 2:

```
L_gen = mean_i [ -(1/|A(i)|) * sum_{j in A(i)} simY(y_i, y_j) * log( exp(z_i . z_j / tau)
                                                / sum_{a in A(i)} exp(z_i . z_a / tau) ) ]
```

with `tau = 0.1` frozen. The **only** thing that differs between arms B, C and D is the label
similarity `simY`. The `1/|A(i)|` prefactor and the absence of row-normalisation are kept exactly as
published — no re-normalisation of the kernel is applied to any arm.

Total loss: `L = BCE + lambda * L_gen`.

## Arms (identical folds, identical seeds, identical tuning budget)

| arm | `simY(q_i, q_j)` | what it is |
|---|---|---|
| **A** | — (`lambda = 0`, no contrastive term) | hard-label baseline head, current BCE recipe |
| **B** | `q_i.q_j / (‖q_i‖ ‖q_j‖)` — **cosine** | **the primary comparator**: GenSCL's published label-similarity function applied to the vote distributions (the strongest existing distributional-contrastive baseline) |
| **C** | `q_i.q_j` — **raw inner product** = `P(Y_i = Y_j)` | **the candidate mechanism**: expected inter-annotator agreement as the pair topology |
| **D** | `q_pi(i).q_pi(j)`, `pi` a global permutation | **shuffled-`q` placebo**: identical kernel family and identical marginal distribution of `q`, destroyed alignment with features and labels |

**What C − B isolates, stated in advance.** After the cosine normalisation is removed, C and B
differ by exactly the per-item certainty factors `‖q_i‖‖q_j‖`. C therefore down-weights contested
items as positives — this **is** external-review defect §6.7 item 1, and it is the quantity under
test. C − B is the whole scientific content of leg (ii); C − A is not (A is context only).

**Global-scale confound and how it is handled.** `q_i.q_j <= cos(q_i,q_j)`, so C's contrastive term
is uniformly smaller than B's. `lambda` is therefore selected **per arm, per outer fold** from the
frozen grid `lambda in {0.1, 0.3, 1.0, 3.0}` by inner **stratified 3-fold CV** maximising inner
macro-F1 — an identical budget (4 values x 3 inner folds = 12 extra fits) for B, C and D. This
absorbs the global scale difference so the contrast is about the *relative* weighting across pairs,
not about the kernel's magnitude. Arm A has no `lambda` and gets no tuning.

**Placebo construction.** `pi` is drawn once per (language, fold-seed) with
`numpy.random.default_rng(20260920 + seed_index)`, applied to the whole pooled set **before**
folding, permuting `q` only — hard labels `y` are untouched, so D's BCE term is identical to
B's and C's. Runtime assertions: `K_B != K_C` and `K_D` is a permutation of `K_C`.

## Protocol

Stratified **5-fold** on the binary label, **3 fold seeds `20260914 / 20260915 / 20260916`**, every
arm on byte-identical folds, per language. All reported quantities are **out-of-fold** over the
pooled train+val set.

## Endpoints

- **Primary (gating).** Binary **macro-F1** of the OOF hard predictions (`p > 0.5`) against the
  project binary label. Per seed `s`, the endpoint is the **mean of the two languages**:
  `M(arm, s) = 0.5 * (macroF1_EN + macroF1_ZH)`. Per-language values are reported descriptively but
  **do not** create a per-language decision branch (no "at least one language" reading exists in
  this freeze).
- **Secondary (non-gating), distribution-prediction quality.** Against the vote-derived binary soft
  target `f_i` = harmful-vote fraction (`Offensive`, `Hateful` -> harm):
  (a) mean `KL( Bernoulli(f_i) ‖ Bernoulli(p_i) )`, `p` clipped to `[1e-6, 1-1e-6]`, lower better;
  (b) macro **soft-F1**, `F1_1 = 2*sum(p_i f_i)/(sum p_i + sum f_i)`,
  `F1_0 = 2*sum((1-p_i)(1-f_i))/(sum(1-p_i) + sum(1-f_i))`, averaged.

## Frozen decision rule (no AMBIGUOUS branch — futility rule in force)

```
d_CB(s) = M(C,s) - M(B,s)        d_CD(s) = M(C,s) - M(D,s)      s = 1..3

pass_CB := mean_s d_CB(s) >= +0.005  AND  d_CB(s) > 0 for all 3 seeds
pass_CD := mean_s d_CD(s) >= +0.005  AND  d_CD(s) > 0 for all 3 seeds

EXPLORATORY-GO   iff  pass_CB AND pass_CD
FAMILY-CLOSED    otherwise            (any other outcome, including ties and mixed signs)
```

`EXPLORATORY-GO` grants only the "exploratory" label of the standing declaration above.
`FAMILY-CLOSED` **permanently closes Human-Agreement Retrieval (all three legs)**: no re-run, no
re-tuning, no re-specification of this hypothesis on this data.

## Red lines

(1) Zero test-set contact, guard armed. (2) This rule is frozen before results are seen and is
transcribed unedited into `idea-stage/LEG2_KILL_RESULT.md`. (3) Blind design — no candidate endpoint
was computed on real data while designing or implementing; implementation is validated only with
synthetic data and a label-permuted smoke. (4) **One submission.** No re-runs, no post-hoc tuning,
no arm added after the fact.

## What this experiment cannot establish

No test-set number, no accuracy claim, no video-specific claim (§6.7 item 8 stands: this is a
modality-agnostic loss on frozen pooled CLIP features). An `EXPLORATORY-GO` would still be
adaptively selected, single-collection evidence — EN and ZH from one collection are **not**
independent replications (§6.7 item 6) and seeds are optimisation replicates, not resampling of the
population. A `FAMILY-CLOSED` closes the *candidate*, not the value of the vote data, which remains
assigned to the evaluation-validity / resource track.

---
---

# §C8 — Prosody-as-operator binding (round-4 pilot, frozen 2026-08-09)

**Written BEFORE any implementation line and BEFORE any candidate metric was computed on real
data.** Population counts and cache shapes below were read from disk during design (they are
facts about assets, not endpoints) and are recorded here so they cannot be quietly revised later.

Candidate: **C8** in `idea-stage/IDEA_REPORT.md` §7.1 / §7.4 / §7.8, ranked first in the round-4
queue ("cheapest remaining, cleanest falsification").

## C8.0 The hypothesis, restated from §7.4 without softening

Phase 1 measured audio's **marginal** utility (audio-only accuracy; the gain from concatenating an
audio vector) and judged the audio prior weak. §7.4's claim is that this is **the wrong estimand**:
a weak marginal is fully compatible with a strong **conditional interaction**, because prosody is
not a fourth evidence stream but an *operator* on transcript meaning — the same sentence said with
mockery, threat or slogan-chanting delivery changes label without being recoverable from the audio
alone. If that is true, the effect must live in the region where the text is **genuinely
ambiguous** (hateful-or-not on its wording), and a global marginal average must dilute it to
invisibility.

This pilot tests exactly that and nothing else. It does **not** test whether an operator-structured
model beats a concat model at the system level, and no result here licenses a test-set number.

## C8.1 Data — HateMM train split only

- **Population**: `data/gt/HateMM/train.jsonl`, 744 rows, label base rate 0.4005.
- **Empty-transcript handling — frozen**: the **39** train rows whose `text` is whitespace-only are
  **excluded** from the analysis population, in both fitting and evaluation, for every arm.
  Analysable N = **705**.
  Justification, frozen in advance (`refine-logs/EMPTY_TEXT_AUDIT_2026-08-09.md` §2a–§2d): those 39
  rows have `Title == "" and Transcript == ""` upstream, so their CLIP text vector is one identical
  constant point, and that point is 92.3 % non-hate against a 40.1 % base rate. On a constant text
  vector a text×prosody interaction term degenerates into a pure prosody main effect, and the
  boundary-band construction (§C8.4) is undefined for them. Keeping them could only manufacture a
  GO. Their exclusion is therefore the **least GO-favouring** reading.
  A labelled **non-gating** sensitivity re-runs the primary arm on all 744 and is reported; it
  cannot change the verdict.
- **`dev_seen` / val is not used.** `test_seen` / `test.jsonl` is never opened. `pilot_a`'s path
  guard is armed: any path containing `test` HALTs. (`clap_..._trainval.pt` contains no `test`
  substring; it is loaded and then row-restricted to the 705 train ids.)

## C8.2 Features — named explicitly, no silent substitution (§7.4 asset correction)

Per the standing correction in §7.4/§7.7-5: **CLAP is cached for HateMM only**; MHC / MHC_zh carry
Whisper-encoder audio. This pilot is HateMM-only, so the correction does not force any substitution
— but every representation used is named here.

| role | tensor | dim | source |
|---|---|---|---|
| **text** | `text_feats` | 768 | `data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt` (CLIP ViT-L/14-336 text tower) |
| **prosody, arm P (primary sense of "prosody")** | `egemaps` | 88 | `data/audio/HateMM/egemaps_v02_trainval.pt` (openSMILE eGeMAPSv02 Functionals: F0, jitter, shimmer, loudness, HNR, spectral slope …) |
| **prosody, arm C (the cached CLAP asset §7.8 budgeted)** | `proj` | 1024 | `data/audio/HateMM/clap_larger_clap_general_trainval.pt` (`laion/larger_clap_general`, mean⊕max over 10 s windows) |

**Both arms are declared here, before results, and both are gating** (see §C8.5 for why, and for
the multiplicity handling). eGeMAPS is what the word *prosody* denotes; CLAP is a general
audio-semantic embedding and is the asset the round-4 budget named. Reporting one and hiding the
other is the failure mode this section exists to prevent.

**Whisper-large-v3 encoder features (`whisper_..._trainval.pt`, 2560-d) are pre-registered as
EXCLUDED**, for a stated reason: the Whisper encoder is trained to carry phonetic/lexical content,
so a "text × Whisper-audio" interaction is partly a text × text interaction and cannot separate the
conditional-prosody hypothesis from transcript redundancy. This exclusion is frozen and may not be
revisited after seeing results.

## C8.3 Arms

All heads are `sklearn` logistic regression, `C = 1.0`, `max_iter = 5000`, `lbfgs`. All
preprocessing (standardisation, PCA, interaction standardisation) is fitted **on the training fold
only** and applied to the held-out fold.

| arm | features | purpose |
|---|---|---|
| **M0** | `PCA_64(z(text))` | text-only head; also defines the boundary band |
| **M1** | `PCA_64(z(text)) ‖ PCA_16(z(prosody))` | **marginal** arm — the Phase-1 "concatenate an audio vector" replication |
| **M2** | M1 features `‖ z(outer(t_8, p_8))` (64 extra terms) | **conditional** arm — bilinear text×prosody gate on top of both main effects |

`t_8` / `p_8` are the first 8 components of the same fold-fitted text / prosody PCAs, each rescaled
to unit variance on the training fold; their 8×8 outer product is standardised on the training
fold. M2 ⊃ M1 by construction, so **M2 − M1 isolates the interaction** and nothing else.

Frozen dims: `d_text = 64`, `d_pros = 16`, interaction `8 × 8 = 64`. No dim is tuned after results.

## C8.4 Boundary band — frozen definition and bandwidth

Out-of-fold probabilities come from **5-fold stratified CV over videos** (each video held out
exactly once). Seeds **20260901, 20260902, 20260903** (3 seeds; a seed re-draws the CV partition).

**Band = the middle 30 % of the analysable population ranked by M0's OOF probability**, i.e. rank
quantiles [0.35, 0.65]. Expected size 0.30 × 705 ≈ **212** items. The band is computed from **M0
only** — it never sees prosody, the interaction, or the M1/M2 predictions — and is therefore
identical across arms and across placebo replicates, which keeps every comparison paired.

**VOID clause (frozen)**: if either class has fewer than 20 members inside the band for any seed,
the pilot is VOID and reported as such — not as a KILL and not as a GO.

Non-gating sensitivities: middle 20 % and middle 40 %. Reported, labelled, cannot move the verdict.

## C8.5 Endpoint, and the one metric pinned

**Gating endpoint: `Δ_int = AUC_band(M2) − AUC_band(M1)`**, ROC-AUC of the OOF probabilities
restricted to the band, averaged over the 3 seeds.

**AUC is the pinned metric; macro-F1 is reported as a labelled non-gating secondary.** Reason,
frozen in advance: the band is constructed to sit at the decision boundary, so macro-F1 at a fixed
0.5 threshold there is dominated by threshold placement rather than by whether the interaction
carries information; the pilot's question is a ranking question. The bar transfers unchanged at
**+0.010**.

Also reported, **non-gating, for context only**:
- `Δ_marg = AUC_full(M1) − AUC_full(M0)` over all 705 — the Phase-1 "prior difference" replication;
- `Δ_int` evaluated on the **complement** of the band (the outer 70 %) — the dilution claim predicts
  it is smaller than inside the band. This is a *reading aid*, not a condition.

## C8.6 Placebo (the discriminator)

Prosody rows are **permuted across videos within label strata** (hate / non-hate separately;
HateMM is single-language, so §7.4's "label × language strata" reduces to label strata). Permuting
within label preserves any label-marginal information prosody carries and destroys only its
**pairing with the text** — which is precisely the operator hypothesis.

The permutation is applied to the prosody matrix once per replicate, and the **entire** pipeline
(M1 and M2, same folds, same band) is re-run on it, so the placebo statistic is
`Δ_int^perm = AUC_band(M2_perm) − AUC_band(M1_perm)`: same model capacity, same feature count, same
band, destroyed pairing.

**Replicates: 3 seeds × 10 permutations = 30**, permutation seeds `20260901..3 × {0..9}`.

## C8.7 Frozen decision rule — GO / KILL, no AMBIGUOUS

Evaluated **independently for arm P (eGeMAPS) and arm C (CLAP)**. An arm PASSES iff all three hold:

```
(a)  mean over 3 seeds of  Δ_int  ≥  +0.010          (AUC, inside the band)
(b)  all 3 seeds have      Δ_int  >  0               (3/3 same sign)
(c)  mean Δ_int  >  P95 of the 30 placebo Δ_int^perm (placebo does not reproduce it)
```

```
GO     iff  arm P PASSES  OR  arm C PASSES
KILL   otherwise
```

**Why the OR, stated before results.** This pilot's job is to *close* a family. Two representations
are tested, so an OR raises the false-GO rate; that is deliberate and it is the safe direction —
a false GO costs one pre-registration and CPU-minutes downstream, whereas a false KILL closes the
entire audio-operator family permanently on a single representation choice. Making the **KILL**
the conjunction (both arms fail) is the conservative reading of a family-closing gate. Condition
(c) is an independent per-arm null and is what actually protects against noise.

**KILL closes the whole audio-operator family** on this project's data: prosody-as-operator,
FiLM/gating/bilinear audio conditioning, and any successor whose mechanism is "audio modulates
text". No re-run, no re-tuning, no re-specification of this hypothesis on this dataset.

**GO** means only: the conditional estimand shows a boundary-band interaction that a within-label
permutation does not reproduce. It is *not* a system-level claim, *not* a test-set claim, and it
sends C8 to a full pre-registration, not to a paper.

## C8.8 Red lines and discipline

(1) **Zero test-set contact**, guard armed, touched-path list and input SHA256s written into the
output JSON. (2) This rule is frozen before results and is transcribed **unedited** into
`idea-stage/C8_PROSODY_RESULT.md`. (3) **Blind design** — no candidate endpoint was computed on
real data while designing or implementing; implementation is validated with a synthetic
positive-control (a planted text×prosody interaction, which the pipeline must detect) and a
label-permuted smoke (which it must not). (4) **One submission**, background, logs at
`logging/runs/c8_prosody/run.{log,pid}`.

## C8.9 What this pilot cannot establish

It cannot show that an operator-structured architecture beats concatenation at the system level; it
cannot show the interaction generalises beyond HateMM (MHC/MHC_zh have no CLAP — §7.4); it cannot
distinguish "prosody modulates meaning" from "some audio property co-varies with an unmeasured
confounder that happens to be text-conditional"; and a single dataset with 705 usable rows and
~212 in the band supports no effect-size claim, only a presence/absence gate. Seeds are CV-partition
replicates, not resampling of the population.
