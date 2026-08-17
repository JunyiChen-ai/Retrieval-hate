# LEG2-KILL — result: **FAMILY-CLOSED**

Capped kill test of Human-Agreement Retrieval **leg (ii)** — the agreement-weighted contrastive
training objective `s_ij = q_i^T q_j`. Run 2026-08-09, single submission, CPU, 753 s.

---

## Standing declaration (transcribed verbatim from the freeze; read before any number below)

> This experiment is an **adaptively selected** hypothesis. It was chosen *after* the frozen
> P-A-v2 gate failed in both languages, because leg (ii) is the only leg that gate did not touch.
> It **inherits no prior GO**. The original P-A / P-A-v2 gate **stays failed regardless of this
> outcome**. A positive result grants the label **"exploratory"** only — never "recommended",
> never a main-conference claim, and it remains single-dataset, adaptively-selected evidence for a
> low-novelty kernel instantiation until independently confirmed on data this project has not
> touched. A negative result **permanently closes the entire Human-Agreement family**
> (legs i, ii, iii), with no revival branch.

This run executes the external reviewer's own conditions (`IDEA_REPORT.md` §6.5, §6.8): a new
pre-registration; primary comparator **GenSCL/LDL**, not hard-label RGCL; every loss / temperature /
smoothing degree of freedom frozen in advance; a **futility rule** (one failure closes it);
a mandatory placebo arm (§6.7 item 7).

---

## Verdict

**FAMILY-CLOSED.** The frozen rule required `C` to beat the GenSCL comparator `B` **and** the
shuffled-`q` placebo `D` by ≥ +0.005 macro-F1 with 3/3 seeds agreeing in sign. `C` beat neither on
the mean and was positive in **0/3** seeds against `B` and **1/3** against `D`.

Frozen rule, transcribed unedited from `idea-stage/PILOT_FREEZE_2026-08-09.md` §LEG2-KILL:

```
M(arm, s) = 0.5 * (macroF1_EN(arm,s) + macroF1_ZH(arm,s))
d_CB(s) = M(C,s) - M(B,s)        d_CD(s) = M(C,s) - M(D,s)      s = 1..3

pass_CB := mean_s d_CB(s) >= +0.005  AND  d_CB(s) > 0 for all 3 seeds
pass_CD := mean_s d_CD(s) >= +0.005  AND  d_CD(s) > 0 for all 3 seeds

EXPLORATORY-GO   iff  pass_CB AND pass_CD
FAMILY-CLOSED    otherwise            (any other outcome, including ties and mixed signs)
```

| gate quantity | seed 20260914 | seed 20260915 | seed 20260916 | mean | required | met |
|---|---|---|---|---|---|---|
| `d_CB` = C − B | −0.00002 | −0.01231 | −0.00286 | **−0.00506** | ≥ +0.005 and 3/3 > 0 | **no** (0/3) |
| `d_CD` = C − D | +0.00136 | −0.00911 | −0.00328 | **−0.00368** | ≥ +0.005 and 3/3 > 0 | **no** (1/3) |

`pass_CB = false`, `pass_CD = false` → **FAMILY-CLOSED**. There is no AMBIGUOUS branch and no
re-run: **Human-Agreement Retrieval (legs i, ii, iii) is permanently closed.**

---

## Design, as executed

**Data.** MultiHateClip EN (549 train + 80 val = 629) and ZH (579 + 78 = 657), pooled train+val,
all out-of-fold. `test.jsonl` never opened; P-A's path guard armed — the 12 paths touched are
listed in the JSON and none contains `test`. Official votes
`data/gt/mhc_votes/mhc_{English,Chinese}_{train,valid}.tsv`, parsed by the unchanged P-A functions
(alias `No → Normal` fired once, in ZH). Features: frozen CLIP ViT-L/14-336,
`X = [l2(img) ‖ l2(txt)]`, 1792-d, standardised on the outer training fold.

`q_i` = empirical 4-class vote histogram over `[Hateful, Offensive, Normal, Counter Narrative]`,
**no smoothing**. Vote counts: EN `{2: 526, 3: 102, 4: 1}`, ZH `{2: 485, 3: 165, 4: 7}`;
unanimous items 78.9 % EN / 68.3 % ZH; binary base rate 0.3068 EN / 0.3166 ZH.

**Objective.** GenSCL Eq. 2 (Kim, Lee, Chang & Park, arXiv **2206.00384**), verbatim form,
full-batch anchors, `tau = 0.1`:

```
L_gen = mean_i [ -(1/|A(i)|) * sum_{j in A(i)} simY(q_i, q_j)
                 * log( exp(z_i.z_j/tau) / sum_{a in A(i)} exp(z_i.z_a/tau) ) ]
L     = BCE(pos_weight = n_neg/n_pos) + lambda * L_gen
```

The **only** difference between arms B, C and D is `simY`. No re-normalisation of the kernel was
applied to any arm; the `1/|A(i)|` prefactor is kept as published.

| arm | `simY(q_i, q_j)` | role |
|---|---|---|
| **A** | — (`lambda = 0`) | hard-label BCE baseline head |
| **B** | `q_i.q_j / (‖q_i‖‖q_j‖)` (cosine) | **primary comparator** — GenSCL's published label-similarity function on the vote distributions |
| **C** | `q_i.q_j` = `P(Y_i = Y_j)` | **candidate mechanism** — expected inter-annotator agreement as the pair topology |
| **D** | `q_pi(i).q_pi(j)` | shuffled-`q` placebo (global permutation per language per seed, seed `20260920 + s`; labels untouched, so D's BCE term is identical to B's and C's) |

Runtime assertions passed: `K_B != K_C`, and `K_D` is an exact permutation of `K_C` (identical
multiset of entries). Mean `K_B − K_C` gap: 0.046 EN / 0.077 ZH — the two kernels are materially
different, which is what makes C − B a real test.

**What C − B isolates (stated in the freeze, before results).** Removing the cosine normalisation
leaves exactly the per-item certainty factors `‖q_i‖‖q_j‖`, i.e. C systematically down-weights
contested items as positives. That **is** external-review defect §6.7 item 1, and it is the quantity
under test. `C − A` is context only and was never part of the gate.

**Head (identical for every arm).** `Linear(1792→128) → ReLU → Dropout(0.2)` → `Linear(128→64)` +
L2-norm for the contrastive space, `Linear(128→1)` for the logit. Adam, lr 1e-3, wd 1e-2, 400
full-batch steps, threshold 0.5. Stratified 5-fold, 3 fold seeds `20260914/15/16`, byte-identical
folds across arms. `lambda` selected per arm per outer fold by inner stratified 3-fold CV over the
frozen grid `{0.1, 0.3, 1.0, 3.0}` — an identical 12-fit budget for B, C and D (arm A has no
`lambda` and gets no tuning). This absorbs the fact that `q_i.q_j <= cos(q_i,q_j)` uniformly, so the
contrast is about the *relative* weighting across pairs, not kernel magnitude.

---

## Results

### Primary — OOF binary macro-F1 (gating)

| | EN | ZH | **M = mean of the two (gate endpoint)** |
|---|---|---|---|
| **A** — BCE only | 0.7016 | 0.7243 | **0.7130** |
| **B** — GenSCL cosine (comparator) | **0.7142** | **0.7335** | **0.7239** |
| **C** — `q_i^T q_j` (candidate) | 0.7104 | 0.7272 | **0.7188** |
| **D** — shuffled-`q` placebo | 0.7130 | 0.7319 | **0.7225** |

Per-seed macro-F1 (seeds 20260914 / 20260915 / 20260916):

| arm | EN | ZH | `M` |
|---|---|---|---|
| A | 0.6870 / 0.6921 / 0.7257 | 0.7211 / 0.7171 / 0.7348 | 0.7040 / 0.7046 / 0.7302 |
| B | 0.7013 / 0.7089 / 0.7324 | 0.7197 / 0.7376 / 0.7433 | 0.7105 / 0.7232 / 0.7379 |
| C | 0.6998 / 0.7004 / 0.7310 | 0.7211 / 0.7214 / 0.7391 | 0.7104 / 0.7109 / 0.7350 |
| D | 0.6970 / 0.7074 / 0.7347 | 0.7212 / 0.7327 / 0.7419 | 0.7091 / 0.7200 / 0.7383 |

**C loses to B in both languages** (−0.0038 EN, −0.0063 ZH) and **to the placebo in both**
(−0.0026 EN, −0.0047 ZH). The direction is consistent; only the magnitude is small.

### Secondary — distribution-prediction quality (non-gating)

Against the vote-derived soft target `f_i` = harmful-vote fraction.

| arm | KL (EN) | KL (ZH) | soft-F1 (EN) | soft-F1 (ZH) |
|---|---|---|---|---|
| A | 0.6082 | 0.6582 | 0.6873 | 0.6891 |
| B | **0.5838** | 0.6456 | 0.6879 | **0.6893** |
| C | 0.5916 | **0.6439** | 0.6877 | 0.6888 |
| D | 0.5907 | 0.6459 | 0.6856 | 0.6885 |

No rescue. C is worse than B on EN KL, marginally better on ZH KL (−0.0017, inside the placebo's own
spread), and soft-F1 is flat to three decimals across all four arms. Nothing here would have changed
the verdict even if the secondary had been gating, which it was not.

### Chosen-`lambda` histogram (15 outer folds per arm per language)

| | 0.1 | 0.3 | 1.0 | 3.0 |
|---|---|---|---|---|
| EN B | 2 | 1 | 2 | **10** |
| EN C | 3 | 3 | 3 | 6 |
| EN D | 2 | 3 | 5 | 5 |
| ZH B | 2 | 4 | 3 | 6 |
| ZH C | 1 | 5 | 2 | 7 |
| ZH D | 2 | 3 | 4 | 6 |

---

## The result that is larger than the verdict

**The placebo reproduces almost the entire contrastive gain.** Over the BCE baseline A, the
GenSCL comparator gains `+0.0109` and the **shuffled-vote placebo gains `+0.0095`** — 87 % of it —
while the candidate mechanism gains `+0.0058`. Whatever the contrastive term is doing for this head,
it is **not** carrying annotator-agreement information: a kernel built from votes randomly reassigned
to other videos does essentially as well.

This is a descriptive reading of the frozen arms, not a new analysis and not a gating quantity, and
it rests on differences of the same order as the seed spread (see qualification 2). But it is the
direct answer to §6.7 item 7 — *"a positive result would not identify the mechanism"* — running in
the other direction: there was no mechanism-specific effect to identify.

---

## Honest qualifications

1. **`lambda` grid-edge saturation.** The top of the frozen grid (`3.0`) was selected most often for
   every arm, and for EN arm B on 10/15 folds. A wider grid could raise **B** further; since B is
   the comparator C had to beat, that can only **deepen** the kill. Same structural caveat as
   P-A-v2's `C = 0.003` edge, and it points the same way.
2. **The effect sizes are small relative to seed spread.** `M` moves 0.704 → 0.738 across fold seeds
   within a single arm, while the gate quantities are −0.005 and −0.004. No confidence interval was
   pre-registered — the frozen rule is a threshold-plus-sign rule, and adding a bootstrap now would
   be exactly the post-hoc analysis the red lines forbid. So what this run establishes is
   **"no advantage at or above the pre-registered effect size, with no seed agreeing in sign"**, not
   "C is significantly worse than B". The futility rule was adopted knowing this: the reviewer's
   condition was that one failure closes the question, and it did.
3. **Defects deliberately left unfixed** (per the freeze, per §6.7): the kernel conflates similarity
   with certainty (item 1); the 4-class geometry fights the binary task (item 2); `q_i^T q_j` may be
   nothing but marginalised hard-label training (item 3); contestedness is confounded with vote count
   (item 4); with 2 votes for 79 % EN / 68 % ZH of items, `q_i` is a noisy histogram, not a population
   distribution (item 5). This run tested the **original mechanism as specified**. A corrected kernel
   would be a different experiment — and under FAMILY-CLOSED it is not one this project will run.
4. **Scope.** No test-set number, no accuracy claim, no video-specific claim (§6.7 item 8 stands:
   this is a modality-agnostic loss on frozen pooled CLIP features). EN and ZH come from one
   collection and are not independent replications (§6.7 item 6); the 3 seeds are optimisation
   replicates, not resampling of the population.
5. **What is *not* closed.** The MultiHateClip vote data itself. FAMILY-CLOSED closes the *candidate
   mechanism*; the votes stay assigned to the resource / evaluation-validity track, where §6.4 and
   §6.6 already place them.

---

## Red-line compliance

| red line | evidence |
|---|---|
| zero test-set contact | guard armed at start; `paths_touched` = 12 files, all `train` / `val` / `valid` / `dev_seen`; no path component contains `test` |
| decision rule frozen before results | `idea-stage/PILOT_FREEZE_2026-08-09.md` §LEG2-KILL, sha256 `14c803d1bbf408c193c0676dda7b46b7fb8f6117c3150287f1e32cade0a2f902` at implementation time; the rule is transcribed unedited above and is reproduced verbatim in `leg2_kill.py:verdict()` |
| blind design | only synthetic and label-permuted smokes were run during implementation; the sole real-data quantity consulted before the freeze was the wall-clock cost of one synthetic head fit (1.5 s) |
| single submission | one launch, `logging/runs/leg2_kill/run.{log,pid}`, 753 s, no re-runs, no re-tuning, no arm added after the fact |

**Implementation smokes.** Synthetic (random features, random votes, 3 seeds): all four arms
0.466–0.489 macro-F1, i.e. chance, verdict machinery returns FAMILY-CLOSED as expected on noise.
Label-permuted on real features (1 seed, seed 999, not a frozen seed): A/B/C/D = 0.497/0.502/0.500/0.495
EN and 0.507/0.514/0.525/0.520 ZH — no leakage path.

---

## Reproducibility

| artifact | path |
|---|---|
| frozen rules (written before implementation) | `idea-stage/PILOT_FREEZE_2026-08-09.md` §LEG2-KILL |
| implementation | `idea-stage/leg2_kill.py` |
| raw results | `idea-stage/leg2_kill.json` |
| run log / pid | `logging/runs/leg2_kill/run.log`, `run.pid` |
| upstream context | `idea-stage/IDEA_REPORT.md` §3.1b (P-A-v2 KILL), §5.2 (leg-(ii) novelty retraction), §6 (external review), §6.9 (this result) |
| comparator paper | Kim, Lee, Chang & Park, *Generalized Supervised Contrastive Learning*, arXiv 2206.00384 (Eq. 2; label similarity = cosine of the label vectors) |
