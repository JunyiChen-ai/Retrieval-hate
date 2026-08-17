# Round-4 pilot freeze — decision rules fixed before any implementation line exists

**Frozen 2026-08-10, before a single line of pilot code was written.**
Source of the rules: the cross-model triage verdict `idea-stage/triage_r4_verdict_2026-08-10.md`
(`gpt-5.6-sol`, `model_reasoning_effort: xhigh`, thread `019fe784-eefa-7fd1-b53b-67753e528bc0`).
The rules below are transcribed from that verdict and are **not** re-negotiable after results exist.

Two pilots are authorised. **There is no third pilot** — the jury explicitly declined the third slot.

## Standing protocol (applies to both)

- Train on `train`, select epoch on `val` (validation macro-F1), **report `test`**. Seeds 0/1/2.
- Single submission per pilot. Every cell reported, including losing ones.
- Base head = `idea-stage/r4_harness.py` (`Head`, align/Hadamard fusion, map 1024, proj 1024,
  3 layers, dropout 0.2/0.4/0.1, AdamW lr 1e-4, bs 64, 30 epochs, warmup 5). Validated against
  `RGCL_ABLATION_RESULT.md`: HateMM/CLIP 0.8013 vs 0.7993, ImpliHateVid/CLIP 0.9068 vs 0.9118,
  HateMM/Qwen 0.8588 vs 0.8640.
- **Confirmatory-by-construction disclosure**: both pilots were motivated by measurements already
  taken on these test sets today (`IDEA_REPORT` §8.4). Per the jury, this is *survivable for a
  disclosed pilot* but a GO licenses only an untouched-external confirmation build and a novelty
  sweep — **never a paper-level claim from these four test sets alone**.
- Threshold rule, both pilots: each method × seed picks its threshold by maximum **validation**
  macro-F1; ties resolved toward the threshold closest to 0.5.
- `DeltaROC_d = mean_seed[test_ROC(method) - test_ROC(frozen_comparator_d)]`; `DeltaF1_d`
  analogously on macro-F1 at the validation-chosen threshold. `MeanDelta*` = unweighted mean over
  the four datasets/cells.

---

## Pilot R4-1 — F1 MDL (Monotone Disagreement Lattice)

**Mechanism.** Fit a monotone piecewise-linear lattice over the per-encoder out-of-fold logits that
is pinned to the validation-best single encoder wherever the encoders agree, and is free to learn a
non-additive correction only where they disagree.

**Scope (frozen).** Datasets HateMM, MHC-EN, MHC-ZH, ImpliHateVid. Encoders CLIP ViT-L/336, frozen
Qwen2.5-VL-7B, LoRA-Qwen where cached — **ImpliHateVid gets CLIP + Qwen only** (no LoRA cache).
Five stratified out-of-fold train logits per encoder; val/test logits from a full-train model.

**Model (frozen, no grid permitted).** One lattice, 4 knots per axis, knots at the train-OOF
empirical 0, 1/3, 2/3, 1 quantiles of each axis. Monotone non-decreasing along every axis.
Objective = pairwise logistic rank loss + BCE calibration + concordant-region identity penalty
toward the validation-best encoder, **identity penalty weight fixed at 1.0**. No lattice-size and no
loss-weight search. Epoch selection on validation macro-F1.

**Comparators (frozen).** validation-best single encoder · mean probability · mean logit ·
non-negative validation-weighted logit average · logistic stacker · 2-layer ReLU MLP stacker with
parameter count ≥ the lattice's. Per dataset, the frozen comparator is the one with the **highest
mean validation ROC over the three seeds**, ties broken in the conservative order
`MLP > logistic > weighted > mean-logit > mean-probability > single`. **The comparator is frozen
before any test metric is read.** All comparators' test results are reported regardless.

**Null (frozen).** 200 deterministic repetitions, run after the primary rules are frozen. Within each
split and each hard-label stratum, independently permute every **non-reference** encoder's logit rows
while holding the validation-best reference encoder fixed; refit the identical lattice. This
preserves each encoder's class-conditional score distribution and ROC while destroying item-level
complementarity. Test labels enter only to build and report this post-hoc null, never to select a
model. `Null95` = 95th percentile of `max(0, MeanDeltaROC_null)`.

**GO — every clause required:**

1. `DeltaROC_MHC-EN >= +0.010`
2. `MeanDeltaROC >= +0.010` **and** `MeanDeltaROC >= 3 * Null95`
3. at least 3 of 4 `DeltaROC_d` strictly positive, and none below `-0.005`
4. `MeanDeltaF1 >= +0.010`, with no dataset below `-0.005`

**KILL.** Failure of any clause. Named explicitly by the jury: beating mean averaging but **not** the
validation-selected trained stacker is a KILL; a macro-F1-only gain with ROC below the bar is a KILL;
an MHC-EN miss **cannot** be rescued by HateMM.

**Interpretation, fixed in advance.** A GO licenses an untouched-external-dataset build plus a novelty
sweep over monotone ensemble aggregation — not a claim. A KILL closes the disagreement-lattice
mechanism while leaving the descriptive encoder-complementarity finding intact.

---

## Pilot R4-2 — B1 JLR (Jackknife Lower-Bound Rank Head)

Runs **only after** R4-1's folds and predictions are complete, reusing the identical fold assignment.

**Mechanism.** Replace pointwise BCE with a pairwise objective on the **leave-one-block-out lower
confidence bound** of each hate/non-hate margin, so an ordering supported by only a few training
items is discounted relative to one that is stable across folds.

**Scope (frozen).** Seeds 0/1/2 and exactly four pre-declared cells: HateMM/LoRA-Qwen,
MHC-EN/frozen-Qwen, MHC-ZH/LoRA-Qwen, ImpliHateVid/CLIP. Five seeded stratified folds. Five heads;
head *k* receives BCE and pairwise gradients only from items outside fold *k*. For each sampled
positive–negative pair, over the heads eligible for both items:

```
softplus( -( mean_eligible_margin - 1.0 * sd_eligible_margin ) ) + 0.1 * mean_eligible_BCE
```

Inference averages the five logits. **The coefficient 1.0, the BCE weight 0.1, the fold count, the
architecture and the pair-sampling rule are fixed; there is no grid.** Epoch selection on validation
macro-F1.

**Comparators (frozen).** (a) ordinary single BCE head; (b) five-head leave-one-fold-out BCE ensemble
with identical inference averaging; (c) the same joint pairwise ensemble with the sd coefficient set
to **zero**; (d) a single head with ordinary pairwise-AUC loss + 0.1 BCE. Per dataset the frozen
comparator is the highest mean validation ROC across seeds; ties prefer the higher-capacity five-head
BCE ensemble, then the coefficient-zero pairwise ensemble, then single pairwise, then single BCE.
All test cells reported.

**Null (frozen).** On MHC-EN, 20 pre-seeded null trainings. For the lower-confidence-bound term only,
independently permute each eligible head's item scores within hard label before assembling cross-head
margins; each head's BCE data, label marginals, architecture and inference averaging are unchanged.
`Null95` = 95th percentile of `max(0, DeltaROC_MHC-EN_null)`.

**GO — every clause required:**

1. `DeltaROC_MHC-EN >= +0.010` **and** `>= 3 * Null95`
2. `MeanDeltaROC >= +0.010` over the four fixed cells
3. at least 3 of 4 `DeltaROC_d` strictly positive, none below `-0.005`
4. `MeanDeltaF1 >= +0.005`, no dataset losing more than `0.005`

**KILL.** Anything short of all four clauses. Explicitly: a gain over the single BCE head that
vanishes against either five-head comparator is a KILL. A GO is permission for a targeted novelty
search only — if robust-AUC or ensemble-LCB prior art already contains the mechanism, the candidate
stays dead regardless of its numbers.

---

## Removed from the queue by the feasibility gate + jury (recorded, not re-litigable this round)

- **T1 PRES** — OCR window vectors exist for **HateMM only** (`pilot_ocr_window_vecs.npz` 6565×768
  train+val, `test_ocr_window_vecs.npz` 2111×768); HateClipSeg has windows but **no train/test split
  at all** (395 items, one partition) and a 395/395 constant text channel; MHC/MHC-ZH/ImpliHateVid
  have no OCR cache. Its decisive test-background-vs-train-background comparison is structurally
  single-dataset, on the contaminated dataset. Jury: *"a diagnostic wearing a method's clothes"*,
  score 2.0/10, removed rather than demoted.
- **I1 IPPO** — ImpliHateVid is the only dataset in the project with hate-subtype annotation, so any
  result is one-dataset; and both its error target and its functional form came from today's test
  recon. Jury 4.2/10, *"clean local diagnostic"*, not a methods-pilot.
- **B2 PCD** (4.8, best reserve) — cannot be frozen until the policy cone is mathematically attached
  to the deployed nonlinear Hadamard-fusion head; next action is paper-and-pencil spec + novelty
  check, not GPU.
- **I3 CNV** 4.0 · **B3 NTC** 3.8 · **I2 SHC** 3.4 — occupied baselines and/or one-dataset with a
  hostile trained comparator.
- **R1 B-SRTD** — the round's highest-scoring idea (7.0) but **not pilotable**: re-verified today,
  `data/Counterfactual/MHC/train_twins.jsonl` = 168 records **all label=1**, `MHC_zh` = 180 records
  **all label=1**, one intervention axis. Requires building a balanced two-axis lattice first.
- **R2 EAPD** (6.4) and **F2 SCRA** / **T3 JRSA** — annotation build / weeks of theory.

## ID-leakage rule (armed for any ImpliHateVid work)

ImpliHateVid ids encode both subtype and label (`EX_`/`IM_`/`NH_`), and HateMM ids likewise. Subtype
is parsed into a **train-only** table; all id strings are dropped before folds or tensors are built;
folds come from a seeded permutation of row indices, never from id sorting or id hashing; at
inference the loader exposes numeric features only, and subtype is re-joined **after** prediction
solely to report per-stratum metrics. An assertion fails if any id or subtype field enters a batch.

---

## AMENDMENT (deviation D1, 2026-08-10) — R4-1 clause 2 null replaced by jury ruling

Raised by the mandatory smoke test **before any primary run**; record
`idea-stage/R4_DEVIATION_D1_2026-08-10.md`, ruling `idea-stage/R4_DEVIATION_D1_RULING.md`
(same jury thread `019fe784-eefa-7fd1-b53b-67753e528bc0`).

**Defect.** The frozen within-hard-label permutation null preserves each encoder's class-conditional
distribution and marginal ROC as stated, but replaces the observed inter-encoder copula with
*conditional independence* — the best case for score combination, not the worst. It manufactures
combination gain (synthetic: mean-of-two ROC 0.8266 → 0.8785; real 2-rep smoke on MHC-ZH:
`DeltaROC_null` +0.0549, +0.0692), making `3 × Null95` unattainable for any mechanism. Jury:
**"a false-KILL generator and a blocking defect. Do not apply it literally."**

**Jury's reasoning for not substituting another permutation.** No non-arbitrary permutation can
simultaneously hold every encoder's empirical class-conditional distribution fixed, destroy
item-level complementarity, and define a canonical "no-complementarity" distribution — complementarity
lives in the joint dependence structure, and independence / comonotonicity / rank-correlation
preservation / parametric copulas each impose a different substantive assumption. The jury declined
to replace one arbitrary copula with another.

**Replacement — paired stratified joint-row bootstrap:**

1. Complete all 12 primary cells unchanged; retain each test item's hard label, MDL score and
   frozen-comparator score for every seed.
2. **10,000 repetitions**, `numpy.random.default_rng(20260810)`.
3. Dataset order exactly `HateMM, MHC-EN, MHC-ZH, ImpliHateVid`; **positives drawn before negatives**.
4. Within each dataset sample with replacement exactly the original number of positive and of
   negative test items.
5. **The same sampled joint rows** are applied to MDL, the comparator, and all seeds. Methods and
   encoders are never resampled independently.
6. Per dataset and seed recompute both ROC AUCs and their difference; average over seeds, then take
   the unweighted mean over datasets.
7. `LCB95 = numpy.quantile(MeanDeltaROC_boot, 0.05, method="linear")`. **No truncation at zero, no
   ×3, no `Null95`.**

**Amended clause 2:** `MeanDeltaROC >= +0.010` **and** `LCB95 > 0.000`. Failure of either conjunct is
a KILL. **Clauses 1, 3, 4 and every other frozen rule are unchanged.** Any permutation-null output
already produced is labelled *invalid diagnostic; excluded from verdict*.

**Partial unblinding.** MHC-ZH seed 0 (1 of 12 primary cells) was printed by the smoke run before the
defect was identified; values are recorded verbatim in the deviation file. Jury ruling: **"the round
survives with disclosure; no restart or scope change is required"** — the replacement rule is a
standard paired uncertainty calculation, is not tuned to that cell, and relaxes no remaining clause.
Required handling, adopted: carry those predictions forward unchanged (or reproduce them
deterministically under identical code and seed, output suppressed); run the remaining cells exactly
once; **suppress per-cell test output until all predictions and comparator choices are saved**; and
reproduce this disclosure in the final report.
