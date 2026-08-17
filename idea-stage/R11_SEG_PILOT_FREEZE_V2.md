# R11-SEG v2 — pre-registration (FROZEN)

**Date frozen**: 2026-08-18 · **Supersedes**: nothing. `R11_SEG_PILOT_FREEZE.md` (v1, commit
`96635e1`) ran to completion under single-submission discipline and is reported in
`R11_SEG_PILOT_RESULT.md`. This is a **second, separate** pre-registration.
**Inputs**: v1's result, and `idea-stage/R11_SEG_NOVELTY_CHECK.md` (commit `6ad6b32`), which rated
the direction **(c)** as specified and gave a revision path to **(b)**.
**Status**: frozen; committed to git before `scripts/r11_seg/run_v2.py` is executed. No v2 arm
metric exists at the time of writing.

---

## 0. Why there is a v2, stated honestly

v1 answered its own question and the answer was no: a TAS-family causal temporal model does not beat
a per-window independent head (`+0.270`, CI [−1.371, +1.967]) and loses to the causal broadcast
control (`−0.828`). The novelty check independently says the v1 design was **one mechanism short** —
it was architecture transfer with no component that consumes the coverage prior, on a task three
papers already occupy (HateClipSeg's own baselines, StreamSense WWW 2026, and **SafeLens AAAI-26**,
which has v1's exact modality set on the exact corpus and scores segments *independently*).

v2 exists to test the two things v1 could not:

- **(a) the work carrier** — does *any* temporal-structure arm beat the per-window independent head?
  v1 tested the GTEA-line TAS shape and the LSTR-family attention shape and got null on both. v2
  adds the **dense action detection** shape (per-instant multi-label sigmoid, MS-TCT / PAT line),
  which the novelty check correctly identifies as the closer architectural fit to a multi-hot
  toxicity timeline and which is *not* on the invalid list either.
- **(b) the novelty carrier** — does a **coverage-budget constrained decode**, driven by a
  video-level score, add anything on top of per-instant scores? This is the component none of the
  three occupants has, and it is the only thing here that couples the analysis to the method.

**Both are required.** (a) alone is a first-application. (b) alone is a decoder with nothing to
decode. If (a) fails again and (b) fails, the sub-direction is closed.

---

## 1. Motivation text discipline (binding on every document this round produces)

The novelty check overturned six of the landscape's twelve "structurally invalid" families
(softmax-over-time attention, ActionFormer/TriDet focal negatives, DETR no-object, auxiliary
background class at coverage 0.8, top-K MIL, outer-inner completeness). Binding consequences:

1. **The only permitted form of the structural claim** is the narrowed one:
   *"objectives that force low foreground density, or that manufacture negatives from relative
   within-video scores, become statistically inconsistent as foreground coverage approaches 1."*
   The twelve-family universal must not appear in any pre-registration, result document or paper.
2. **The "TAS has no background class, therefore it fits" argument is dropped.** Under a multi-hot
   sigmoid, "all categories off" *is* the background state re-encoded; removing the explicit class
   changes the parameterisation, not the problem. The surviving reasons to look at TAS are data
   scale and the absence of intra-video contrastive negatives — and §4's `E3` tests the second one
   rather than asserting it.
3. **"TAS is the only surviving family" is deleted.** Dense action detection (`1507.05738`,
   MS-TCT `2112.03902`, PAT `2308.05051`) is a fourth family that is not on the invalid list, and it
   is a baseline in §3, not a footnote.
4. **The perfect-oracle framing is replaced by a learned broadcast control** (already `A1` in v1,
   carried into v2), with coverage-stratified reporting. v1's post-hoc perfect-oracle number
   (79.42) is retained only as a labelled ceiling.

---

## 2. Causality discipline (binding)

HateClipSeg's online per-timestamp task is **causal**. Two protocols are reported and never mixed:

- **ONLINE protocol** — every component may read windows `0..k` only. Causal convolutions,
  unidirectional context, and **forward filtering** (the HMM forward recursion is causal; Viterbi is
  not). All v2 gates are in this protocol.
- **OFFLINE protocol** — acausal components allowed (full-sequence Viterbi, bidirectional context).
  Reported in a separately labelled table, **never compared to StreamSense's 72.06 or to LSTR's
  62.75**, and never used for any gate.

Every v2 model carries the same in-run causality assertion as v1 (perturb windows ≥ 20, require
exactly zero change at windows < 20). Any arm that fails the assertion is reported as OFFLINE.

---

## 3. Arms

Substrate, features, split, grid, seeds, threshold and evaluation are **identical to v1** so that v2
numbers sit in the same table as v1's: `p11_split.json` 237/39/119, canonical K=30 grid,
per-timestamp 0.25 s evaluation, input `ALL` (3586-d) and `V` (1024-d), 12 seeds **2200-2211**,
video-clustered paired bootstrap 10 000 resamples seed **2299**, threshold 0.5, val-selected epoch.

### 3.1 Carried from v1 (re-run so that per-seed scores are saved for the decoder)

| id | arm | claim it serves |
|---|---|---|
| `A1` | `BCAST-CAUSAL` learned broadcast control | control |
| `A2` | `PERWIN` per-window independent head — **the scorer the decoders sit on** | (a) comparator |
| `A4` | `CTCN` causal MS-TCN | (a) |

### 3.2 New — (a), the work carrier

| id | arm | definition |
|---|---|---|
| `B2` | **`DENSE`** — dense action detection shape, minimal MS-TCT/PAT-style, **causal** | causal dilated temporal conv backbone with a multi-scale (stride-1/2/4) temporal branch, per-window **multi-hot sigmoid over the 5 offensive categories** (hateful, insulting, sexual, violence, harm), trained with per-instant BCE on the multi-hot target. Binary read-out for the online task = `1 − Π_c (1 − p_c)`. No background class, no top-k, no intra-video negatives. |

### 3.3 New — (b), the novelty carrier: coverage-budget constrained decoding

Training-free, applied **on top of `A2`'s per-window scores** (and reported on `A4`'s too). The
contrast is against the *same scores* decoded without the constraint, so the decoder is the only
thing that changes.

**Ingredients, all fitted on TRAIN only:**
- **Coverage budget `ĉ(v)`** — a video-level ridge regression predicting a video's offensive window
  fraction from its **causal prefix mean** feature. In the ONLINE protocol at window `k` the budget
  is `ĉ_k` computed from `mean(x_0..x_k)`; in the OFFLINE protocol it is computed once from the
  whole-video mean. This is the "video-level hate score sets the decode's foreground budget"
  coupling: it is the only place the coverage prior enters, and it is what none of the three
  occupants has.
- **Transition confidences `T`** — the 2×2 empirical transition matrix of `y_win` on TRAIN
  (per-window label persistence), used as an HMM transition prior.
- **Per-class duration bounds normalised to video length** — the TRAIN 5th/95th percentile of run
  length, expressed as a fraction of `K`, enforced as a minimum-duration constraint on emitted runs.

| id | arm | protocol | definition |
|---|---|---|---|
| `C0` | `UNCONSTRAINED` | ONLINE | `A2` scores thresholded at 0.5. Identical to v1's `A2` by construction; re-stated as the decoder baseline. |
| `C1` | **`COVBUD-ONLINE`** | ONLINE | causal HMM **forward filtering** with transition prior `T`, plus a per-video causal budget: at window `k`, the decision threshold is the `(1 − ĉ_k)`-quantile of the prefix scores `{p_0..p_k}`. Both components use only the past. |
| `C1a` | `COVBUD-ONLINE, budget only` | ONLINE | the quantile budget without the transition prior — isolates which half does the work |
| `C1b` | `COVBUD-ONLINE, transition only` | ONLINE | forward filtering without the budget — isolates the other half |
| `C2` | `COVBUD-OFFLINE` | **OFFLINE** | full-sequence Viterbi with `T`, a global budget `ĉ(v)` from the whole-video mean, and the duration-bound constraint. Reported separately, never compared to published numbers, never a gate. |
| `C3` | `ORACLE-BUDGET` | diagnostic | `C1` with the **gold** coverage fraction substituted for `ĉ`. Labelled an oracle; it is the ceiling of the budget idea and is **not** evidence for it (round-11 rule: do not treat a large oracle as evidence). |

### 3.4 New — the controlled objective test for the narrowed Part A claim

| id | arm | definition |
|---|---|---|
| `E1` | `OBJ-BCE` | `A4` trained with plain per-instant BCE only. (= v1's `A4`.) |
| `E2` | `OBJ-INTRA` | `A4` plus a **UniVTG-style score-derived intra-video negative** term: per video, anchor = mean projected embedding of the windows the *current model* scores highest (top-⌈0.2K⌉); negatives = the windows the current model scores lowest (bottom-⌈0.2K⌉) **of the same video**, InfoNCE with temperature 0.07, weight 0.1. Negatives are chosen by relative within-video score, exactly as the mechanism is published — not by gold label. |

**The falsifiable prediction.** The narrowed claim says this term becomes inconsistent as coverage
→ 1. HateClipSeg's mean coverage is 0.45, so the claim predicts `E2 − E1 ≈ 0` overall and
`E2 − E1 < 0` on the **high-coverage stratum**. That stratified contrast is the test; a null on the
pooled contrast alone neither confirms nor refutes it.

---

## 4. Evaluation and stratification

Primary metric unchanged: per-timestamp (0.25 s) **macro-F1** on the 119-video test split,
seed-averaged over 12 seeds, video-clustered paired bootstrap.

**Coverage strata**, fixed here before any v2 number exists, on the gold offensive window fraction
of each test video: `LOW` = `[0, 0.25)`, `MID` = `[0.25, 0.75)`, `HIGH` = `[0.75, 1.0]`.
Also reported: **single-span vs multi-span** (a video is multi-span if it has ≥ 2 contiguous
offensive runs). Strata are reported for every arm; only the strata named in §5 are gates.

---

## 5. Decision rule (frozen; no v2 arm number exists at the time of writing)

Two independent gates, both required for the direction to stand. δ = **+1.0 macro-F1** as in v1.

**(a) the work gate** — `Δ_a = max(B2, A4) − A2` on input `ALL`, ONLINE protocol.
- pass iff `Δ_a > 0`, 95% CI excludes zero, and the point estimate `≥ δ`.

**(b) the novelty gate** — `Δ_b = C1 − C0` on input `ALL`, ONLINE protocol, both decoders on the
identical `A2` scores.
- pass iff `Δ_b > 0` and 95% CI excludes zero, **or** `Δ_b` is non-negative overall
  (CI lower bound ≥ −0.5, i.e. "does not hurt") **and** `Δ_b > 0` with CI excluding zero on the
  **multi-span** subset. The second branch is the pre-declared "at least does not hurt and helps
  where there is something to decode" condition.

| verdict | condition |
|---|---|
| **GO** | (a) passes **and** (b) passes |
| **AMBIGUOUS** | exactly one of (a), (b) passes |
| **KILL** | neither passes |

Arm-to-claim assignment, so the result document cannot blur them:
`A1, C0` = controls · `A2` = the "no temporal structure" comparator and the scorer under the
decoders · `A4, B2` = claim (a), temporal structure · `C1, C1a, C1b` = claim (b), coverage budget ·
`C2` = offline protocol only, no claim · `C3` = labelled oracle, no claim ·
`E1, E2` = the Part A objective test, no claim about (a) or (b).

**Not gates, reported anyway**: `C1a` vs `C1b` (which half of the decoder works), `C2` (offline),
`C3` (oracle ceiling), `E2 − E1` pooled and by coverage stratum, everything on input `V`, and all
coverage / span strata.

**Comparability, restated and binding**: no v2 number is comparable to StreamSense 72.06, LSTR
62.75, or SafeLens (which publishes no benchmark table in its PDF at all). Internal arm-vs-arm on
the identical frozen 90.8% subset only, with the `DATASET_hateclipseg.md §4` selection-bias
statement attached.

---

## 6. Carried unchanged from v1

- The **B3 matched-head pre-check** was executed under v1 and returned **NOT REPRODUCED**
  (audio − visual on moment label `−0.165` [−5.741, +5.452]; visual − audio on the boundary proxy
  `−1.183` [−4.013, +1.938]). It is **not** re-run. Its circular-shift control stands as v1's
  measurement: audio carries moment-level information (`+3.301` [+0.710, +5.888] over its own
  within-video shuffle) and the CLIP visual channel does not (`−0.275` [−3.659, +2.967]).
- The four red lines, implemented exactly as in v1 §7: zero test-label tuning (val carries all
  selection, threshold fixed at 0.5, test read once by the final scripted pass with an in-run
  disjointness assertion), decision rule frozen before running, no v2 arm metric computed at freeze
  time, single submission with any re-run recorded as a deviation.

## 7. Files

`scripts/r11_seg/run_v2.py` → `idea-stage/r11_seg/out/results_v2.json`;
log `logging/runs/r11_seg/run_v2.log`; result appended to `idea-stage/R11_SEG_PILOT_RESULT.md`.
