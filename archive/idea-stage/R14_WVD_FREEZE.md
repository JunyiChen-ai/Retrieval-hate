# R14-WVD — pre-registration (frozen before any pilot code exists)

Round 12 of idea discovery. Sub-direction: **hateful video temporal localization, proposal level**.
Date 2026-08-18. Arena: HateClipSeg, local 395-video subset, frozen split
`data/gt/HateClipSeg/p11_split.json` (237 / 39 / 119).

This document is committed **before** `scripts/r14_loc/run_wvd.py` is written. The reconnaissance
scripts `scripts/r14_loc/recon_decode.py` and `scripts/r14_loc/recon_headroom.py` already exist and
their outputs (M1-M7 in §1) are the motivation; they are descriptive and are **not** gates.

---

## 1. The measurements this pilot is built on (train/val only, test never opened)

| id | measurement |
|---|---|
| M1 | Whole-video broadcast with a perfect video-level classifier scores proposal **F1@tIoU0.5 = 10.9** on train. The proposal task is **not** degenerate, unlike the online per-timestamp task (79.42 macro-F1 broadcast ceiling, round 11). |
| M2 | Unit-grid representation ceilings (oracle labels → merge runs), F1@tIoU 0.3/0.5/0.7: gold segments 100/100/100 · uniform 1 s 99.9/99.5/99.1 · uniform 2 s 98.2/98.2/95.7 · **uniform 8 s (K=30) 93.9/87.6/68.1** · Whisper-chunk grid 58.5/50.2/42.1. |
| M3 | Gold segment boundaries are not freely recoverable: Whisper-large-v3 chunk boundaries reach 32% recall / 27% precision at 1 s tolerance. |
| M4 | Synthetic transfer function at fixed within-video AUC ≈ 0.64: F1@0.5 = 4.3 (threshold only) → 38.3 (tuned decode) for i.i.d. errors, 22.2 for AR-1(ρ=0.8) errors. |
| M5 | **On real scores the decode leverage collapses.** Per-window head, val n=39: wv-AUC 0.671, per-window error lag-1 autocorrelation 0.334, between-video share of score variance 0.519. F1@0.5: naive 21.6 → tuned decode 23.8 → per-video z-normalised 25.5. Broadcast control 7.1. Oracle window labels through the same decoder 87.0. |
| M6 | 2×2 oracle substitution on val: model level + model residual 23.8 · **gold level + model residual 28.1** · **model level + gold residual 86.5** · gold+gold 87.0. The gap is +62.7 from the within-video residual, +4.3 from the video-level term, +2.2 from the decode. |
| M7 | Single-channel within-video AUC (val): audio 0.623 > visual 0.587 ≈ CLIP-text(ASR) 0.583 > CLIP-text(OCR) 0.572; all four 0.671. |

**Thesis under test.** Proposal-level hateful-video localization is a *within-video discrimination*
problem. Round 11 closed the temporal-operator axis and the decode axis is now priced at 2-4 points
(M5). The open question is whether **within-video discrimination itself** can be moved on frozen
features, and by which of three levers.

M6 is an **oracle substitution, not a causal decomposition** — substituting labels for a residual
must recover labels. It bounds where the headroom is; it does not prove the headroom is learnable.
That is exactly what this pilot tests, and the KILL rule below is written so that a null closes the
direction rather than being explained away.

---

## 2. Design — one pre-registered 2×2×2 factorial

**Arena.** HateClipSeg **train split only** (237 videos), 5-fold **video-grouped** cross-validation
inside train, folds assigned by `sorted(train_ids)` hashed with a fixed permutation under seed 4210.
Every reported number is out-of-fold. **The 39-video val split and the 119-video test split are not
opened by the pilot runner**; a path/id guard asserts this and prints the assertion.

**Grid.** The canonical frozen K=30 uniform window grid from round 11
(`idea-stage/r11_seg/out/grid_labels.npz`), median ≈ 8.0 s. Justification: M2 puts the
representation ceiling of this grid at F1@0.5 = 87.6 against a current 23.8, so output resolution is
not the binding constraint and changing it would confound the factorial.

**Fixed across all cells.** Head = round 11's `PerWin` (Linear→GELU→Dropout(0.1) to 256, then
256→256→GELU→2). AdamW lr 1e-3, weight decay 1e-2, full-batch, **exactly 40 epochs, no early
stopping and no per-fold model selection** (identical optimisation budget in every cell, no
selection channel). Audio channel = `w2v` (wav2vec2-large-robust-12-ft-emotion-msp-dim, masked-mean
per window) in every cell. Visual channel = CLIP-L/14-336 `subclipK30` in every cell. Feature
z-scoring fitted on the training folds only.

### Factor A — objective

- **A0 `BCE`** — 2-class softmax cross-entropy on the per-window binary offensive label (round 11's
  D2 convention).
- **A1 `BCE+WVR`** — A0 **plus** a within-video pairwise ranking term
  `softplus(-(s_p - s_n))` over same-video window pairs with opposite gold labels, where `s` is the
  positive-class logit margin. Each video's pair set is normalised to **equal total weight per
  video** (divide by `m_v·(n_v-m_v)`), so long or balanced videos do not dominate. Videos with no
  within-video label variation contribute nothing. **λ = 1.0, fixed, never tuned.**

A1 is not the round-11 `E2` family: its negatives are **certified by gold segment labels**, not
manufactured from the model's own scores, so the narrowed high-coverage inconsistency claim does not
apply. It is also deliberately a *hybrid* — BCE is retained — because M6 assigns 4.3 points to the
video-level term and a pure conditional objective would discard it.

### Factor B — text substrate

- **B0 `CLIPTXT`** — the round-11 CLIP text-tower embeddings of the per-window ASR text and the
  per-window OCR text (`idea-stage/r11_seg/out/text_feats.npz`), with their presence masks.
- **B1 `HATETXT`** — the same two texts encoded by frozen
  **`cardiffnlp/twitter-roberta-base-hate-latest`**, mean-pooled last hidden state (768-d, attention
  masked, `pooler` never used because it is randomly initialised in this checkpoint), truncation 256
  tokens. Same presence masks. The encoder is **declared here and will not be swapped**.

### Factor C — representation

- **C0 `ABS`** — absolute per-window features (round 11's construction).
- **C1 `ABS+REL`** — C0 concatenated with, for each of the four channels, the **leave-one-out
  video-relative** vector `x_vi − mean_{j≠i} x_vj` and the scalar within-video rank in [0,1] of
  `cos(x_vi, LOO centroid)`. Entirely **label-free**; uses no gold information at any point.

**Cells** = 8. **Seeds** = 5, `4200, 4201, 4202, 4203, 4204` (outside every banned range:
0-119, 400-429, 500-529, 600-629, 700-729, 1300-1524, 2200-2211, 41000-41029).

---

## 3. Endpoints

**Primary — video-macro within-video AUC (`wv-AUC`).** For every out-of-fold video that has both
labels among its 30 windows, the ROC-AUC of its window scores against its window labels; averaged
over videos with equal weight. A whole-video broadcast predictor is 0.500 on this read-out by
construction, so it cannot be inflated by video-level separability. One number per (cell, seed);
reported as mean ± sd over the 5 seeds.

**Secondary — proposal F1@tIoU {0.3, 0.5, 0.7}**, out-of-fold, under a decoder **frozen here**:
no smoothing (w = 1), merge gaps ≤ 5 s, drop intervals shorter than 12 s, and a
**prevalence-matched threshold** — τ chosen on the *training folds* of that fold so that the
fraction of training-fold windows scored ≥ τ equals the training-fold positive rate. This rule is
label-free at evaluation time, is identical for every cell, and handles the fact that A1 arms are
not probability-calibrated. Intervals are matched greedily 1-to-1 against gold toxic blocks
(adjacent toxic segments merged).

**Reported but not a gate:** per-window macro-F1 (to test the prediction that A1 raises wv-AUC while
*lowering* the round-11 metric), between-video share of score variance, seed sd.

---

## 4. Decision rules — frozen before any number exists

Let `Δ_A` = mean over the four (B,C) combinations of `[A1 − A0]` in OOF wv-AUC, seed-averaged;
`Δ_B` and `Δ_C` defined analogously. CIs are **video-clustered paired bootstrap**, 10 000
resamples, seed 4299, resampling out-of-fold videos.

- **Smallest worthwhile main effect: δ = +0.020 wv-AUC.**
- **GO on a factor** iff its main effect is **≥ +0.020 and its 95% CI excludes 0**.
- **KILL the round's mechanism thesis** iff **all three** main effects are either `< +0.010` or have
  a CI containing 0. In that case the conclusion recorded is: *within-video discrimination on this
  frozen-feature substrate does not move under an objective change, a text-encoder change, or a
  video-relative representation*, and the proposal-level localization direction is closed for this
  substrate.
- **Factor B alone passing is not a mechanism.** If only `Δ_B` clears δ, the finding is recorded as
  **baseline hygiene** (the field's and this project's text substrate was inadequate), explicitly
  not as a candidate method, per the method-paper-only rule.
- Interactions `A×B`, `A×C`, `B×C` are reported **descriptively only** and can never convert a
  failed main effect into a GO.
- The secondary proposal-F1 endpoint can **veto** but never **create** a GO: if a factor clears δ on
  wv-AUC but its F1@tIoU0.5 main effect is negative with a CI excluding zero, the verdict is
  recorded as *within-video ordering improved without delivering intervals* and the factor does not
  advance.

**Noise floor.** Round 11 measured ±0.5 macro-F1 of GPU nondeterminism on this substrate. Five seeds
per cell are run and the per-cell seed sd is reported; if seed sd on wv-AUC exceeds 0.010 the
CI-based half of every rule governs and the point estimate alone is never sufficient.

---

## 5. Test discipline

- The runner loads `p11_split.json`, asserts the three id sets are disjoint, and **uses only
  `train`**. It asserts that no val or test id enters any tensor it fits or scores, and prints both
  assertions.
- No file whose name contains `dev_seen` is opened. The `test_seen_*` caches are the *only* names
  the HateClipSeg feature files ship under (they are the full 395-video caches, not a test split);
  the split is applied by id after loading, and the id-level assertion above is the real guard.
- No number from this pilot may be reported as a held-out result. Anything that survives goes to a
  separate, later, single-submission run on val and then test.

## 6. Cost and execution

Zero cloud, zero API. Local RTX 5090, background `setsid nohup` with
`logging/runs/r14_wvd/{run.log, run.pid}`. One text-encoding pass (~24 000 short strings, minutes),
then 8 cells × 5 folds × 5 seeds of a two-layer head on 30-window sequences. Estimated well under
one hour. Single submission of the frozen runner; a crash-and-fix that changes no arm, threshold,
seed, metric or decision rule is recorded as a deviation, not a re-freeze.
