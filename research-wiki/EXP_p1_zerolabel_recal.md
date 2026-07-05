# P1 — Zero-label drift recalibration via MLLM prior estimation

Front: P1 (campaign goal = give the MLLM a *real* method role: removing it must
measurably cost something). Builds directly on `EVAL_temporal_memory_W4.md`.

Established finding we build on (W4): on the EN (MHC) upload-date temporal split the
drift is **calibration / prior drift, not concept drift**. Temporal macro-F1 0.6273 vs
random-split floor 0.7113; ROC intact (0.8484); label prior shifts train 0.339 → test
0.242; at t=0.5 only 8.7% of test scores clear the vote threshold while the true positive
rate is 24.2%. W4 showed **k=20 LABELLED** new-era samples recover the drop fully
(0.7336). ZH shows no F1 drop (static 0.7779 ≥ random 0.7641): its prior also shifts
(0.340 → 0.248) but its score distribution co-shifts so that t=0.5 already predicts
~28.9% positive ≈ the true 24.8%, i.e. its operating point is already well-calibrated.

**This experiment replaces the k=20 LABELLED recalibration with a ZERO-LABEL one** driven
by an MLLM prior estimate, and asks whether that estimate is good enough to recover the EN
drop without harming the ZH control.

---

## PRE-REGISTRATION (locked before evaluating condition (c); written 2026-07-06)

### Method under test
1. **Vote score** `s(x)` = rank-weighted top-20 neighbour-label fraction against the STATIC
   temporal-train kNN memory (identical to W4; decision `s ≥ t`, default `t = 0.5`).
2. **MLLM verdict** `v(x) ∈ {HARMFUL, BENIGN}` from the video's structured archive ALONE
   (`judge_archive_harmful.py`, Qwen2.5-VL-7B, greedy, fixed prompt). The dataset **label is
   never shown**. Per-video accuracy ~0.7 is acceptable — this is an aggregate prior task.
3. **Adjusted classify-and-count** (bias correction, all inputs free to the method):
   - `TPR = P(v=HARMFUL | gold=1)`, `FPR = P(v=HARMFUL | gold=0)` estimated on the
     **TRAIN era only** (train labels are always available to a deployed system);
   - `CC = P(v=HARMFUL)` on the **unlabeled TEST era**;
   - `p̂ = clamp((CC − FPR) / (TPR − FPR), 0, 1)`.
4. **Threshold recalibration by quantile matching**: choose `t` so the predicted positive
   rate on the unlabeled test era equals `p̂`. Zero labels, zero retraining.
5. **Drift trigger** — recalibrate only if the trigger fires, else keep `t = 0.5`:
   - **Primary (as instructed):** `|p̂ − train_prior| > τ`, `τ = 0.05`.
   - **Refined (pre-registered improvement, still zero-label):** `|p̂ − r₀| > τ`, where
     `r₀ = frac(s_test ≥ 0.5)` is the model's CURRENT operating-point positive rate on the
     unlabeled test stream. Motivation (from the W4 ZH score-distribution fact, established
     before any MLLM verdict): the quantity recalibration actually fixes is the mismatch
     between the current operating point and the estimated prior, not the gap to the (stale)
     train prior. We expect the primary trigger to fire on BOTH languages (both priors
     shift), while the refined trigger should fire on EN (r₀=0.087 far from p̂≈0.24) and
     decline on ZH (r₀=0.289 near p̂≈0.25).

Leak discipline: test-era gold labels are used ONLY for (i) final metric computation and
(ii) the diagnostic oracle prior / verdict-quality numbers, never inside the threshold
selection. `t` is never tuned on test labels.

### Conditions (one test measurement each)
EN temporal split (primary):
- **(a) static** `t=0.5` — must reproduce W4 0.6273.
- **(b) k=20 labelled recal** — must reproduce W4 0.7336 (5 seeds, exact W4 mechanism).
- **(c) MLLM zero-label recal (ours)** — forced, primary-gated, refined-gated.
- **(d) oracle-prior recal** — quantile-match to the TRUE test prior (upper bound).
- **(e) uncorrected classify-and-count** — quantile-match to raw `CC` (no TPR/FPR
  correction; shows the correction matters).
- **(f) sensitivity** — recompute (c) with `TPR/FPR` from only 50% of the train era
  (20 subsamples; report `p̂` and F1 mean±std).

ZH temporal split (control):
- **(a) static** `t=0.5`.
- **(c) full pipeline with trigger** — report whether each trigger fires, and the forced
  (ignore-trigger) F1 change.

Report macro-F1 / accuracy for all; report `p̂` vs the true test prior (the
prior-estimation error is a headline number); report the MLLM verdict TPR/FPR on train vs
(diagnostic) test era to check cross-era stability. Single temporal checkpoint per language
(as in W4) → bootstrap 95% CIs over test samples for the single-threshold conditions.

### Success criteria (pre-registered)
1. Reproduction of (a) and (b) exact.
2. `|p̂ − true test prior| ≤ 0.07`.
3. Zero-label (c) recovers `≥ 60%` of the (b)−(a) gap on EN.
4. ZH control unharmed: the trigger declines, OR the forced version changes F1 by `< 0.01`.

If it fails, report the failure mechanism honestly (e.g. MLLM verdict TPR/FPR unstable
across eras — that itself is a finding).

---

## Assets & protocol
- Splits: `data/gt/{MHC,MHC_zh}_temporal/{train,val,test}.jsonl` (upload-date split).
  EN priors: train 0.3388 (186/549), val 0.2125, test 0.2422 (39/161).
  ZH priors: train 0.3402 (197/579), val 0.2436, test 0.2483 (37/149).
- Archives: `data/Archive/{MHC,MHC_zh}/v2/{train,dev_seen,test_seen}_*.jsonl`. Every
  temporal id is covered (0 missing in v2). Verdicts are keyed by the union of all three
  random-split archive files → one verdict per video, re-indexed into the temporal split.
- Head/checkpoints (W4, val-selected, warmup≥5; restored from B2, verified to reproduce
  0.6273 / 0.7336 / oracle 0.7646 on EN):
  - EN `logging/Retrieval/MHC_temporal/.../ckpt/best_model_16_0.8125.pt`
  - ZH `logging/Retrieval/MHC_zh_temporal/.../ckpt/best_model_27_0.8589….pt`
- Code (new; no `src/` edits): `scripts/analysis/judge_archive_harmful.py`,
  `scripts/slurm/judge_archive_harmful.sbatch`, `scripts/analysis/p1_prior_recal.py`.
- Judging job: diag **12352** (COMPLETED, probes correct), full **12353**.

---

## RESULTS

Full judging job **12353** COMPLETED (Qwen2.5-VL-7B, greedy; EN 790 / ZH 806 archives, v2).
Analysis JSON: `logging/temporal_memory/{MHC,MHC_zh}_temporal_p1_zerolabel.json`.
Single temporal checkpoint per language (as W4); CIs are 95% percentile bootstrap over test
samples (2000 resamples) at the fixed operating point.

### Success-criteria scorecard
| # | criterion | outcome |
|---|---|---|
| 1 | reproduce (a) & (b) exact | **PASS** — EN (a)=0.6273, (b)=0.7336; ZH (a)=0.7779, (b)=0.7239 (matches W4, incl. recal-hurts-ZH) |
| 2 | \|p̂ − true prior\| ≤ 0.07 | **FAIL** — EN err **0.221** (p̂=0.021), ZH err **0.179** (p̂=0.427) |
| 3 | zero-label (c) recovers ≥60% of (b)−(a) gap (EN) | **FAIL** — corrected (c)=0.4798 < static 0.6273 (recovery −139%, i.e. actively harmful) |
| 4 | ZH control unharmed | **FAIL** — trigger fires (both variants); forced recal 0.7779→0.7229 (ΔF1 −0.055 ≫ 0.01) |

### Main table — macro-F1 (acc), EN primary / ZH control
| condition | EN macro-F1 (acc) | EN CI95 | ZH macro-F1 (acc) | ZH CI95 |
|---|---|---|---|---|
| (a) static t=0.5 | 0.6273 (0.795) | [0.529, 0.723] | **0.7779 (0.826)** | [0.700, 0.851] |
| (b) k=20 labelled recal | **0.7336 (0.786)** | ±0.019 (5 seeds) | 0.7239 (0.809) | ±0.024 |
| (c) zero-label corrected — forced | 0.4798 (0.764) | [0.423, 0.552] | 0.7229 (0.752) | [0.645, 0.797] |
| (c) zero-label corrected — gated (primary=refined) | 0.4798 (fires) | — | 0.7229 (fires) | — |
| (d) oracle-prior recal | 0.7124 (0.789) | [0.626, 0.787] | 0.7483 (0.812) | [0.664, 0.821] |
| (e) uncorrected classify-and-count | 0.7156 (0.758) | [0.634, 0.787] | 0.6686 (0.691) | [0.591, 0.745] |
| (f) sensitivity (TPR/FPR from 50% train) | 0.5303 ± 0.1041 | p̂ 0.046±0.055 | 0.7238 ± 0.0285 | p̂ 0.425±0.044 |

### Prior estimation (headline diagnostic)
| | train_prior | true_test_prior | r₀=frac(s≥0.5) | CC_test | **p̂ (corrected)** | \|p̂−true\| |
|---|---|---|---|---|---|---|
| EN | 0.339 | 0.242 | 0.087 | 0.379 | **0.021** | **0.221** |
| ZH | 0.340 | 0.248 | 0.289 | 0.497 | **0.427** | **0.179** |

MLLM verdict error rates, **train era (used by method) vs test era (diagnostic only)**:
| | train TPR | test TPR | train FPR | test FPR | train acc | test acc |
|---|---|---|---|---|---|---|
| EN | 0.699 | 0.821 | **0.372** | **0.238** | 0.652 | 0.776 |
| ZH | 0.741 | 0.784 | **0.314** | **0.402** | 0.705 | 0.644 |

### What happened (failure mechanism — clean and well-attributed)
1. **The recalibration mechanism is sound; only the prior source is bad.** With a *known*
   prior, quantile matching recovers the EN drop: oracle-prior (d)=0.7124 recovers 80% of the
   (b)−(a) gap, and labelled k=20 (b)=0.7336 recovers it fully. So the bottleneck is
   *prior estimation*, not the O(1) threshold knob the retrieval memory exposes.
2. **The MLLM verdict TPR/FPR are not stable across the temporal eras**, so the adjusted
   classify-and-count with *train-era* rates is badly biased — and in **opposite directions**
   on the two languages:
   - EN: train FPR 0.372 ≈ CC_test 0.379, so `p̂=(CC−FPR)/(TPR−FPR)` collapses to ≈0.02,
     even though the MLLM's *actual* test-era FPR is only 0.238. Quantile-matching to p̂≈0.02
     predicts almost no positives → (c)=0.4798, **worse than static**.
   - ZH: train FPR 0.314 < CC_test 0.497 with test-era FPR rising to 0.402, so the correction
     *over*-inflates p̂ to 0.427 (true 0.248).
3. **The bias-correction step back-fires.** The pre-registered "correction matters" ablation
   (e) shows the *uncorrected* count is **better on EN** (0.7156 vs corrected 0.4798) — the raw
   MLLM harmful-call rate (0.379) is closer to the true prior than the "corrected" 0.021. So
   correction helps only where CC already overshoots (ZH) and hurts where it is near-right (EN);
   it is not a reliable operation under era-drifting error rates.
4. **The control cannot be protected by the prior estimate.** ZH must *not* be recalibrated
   (static 0.7779 is already best; oracle-prior only 0.7483). But the MLLM's estimate (p̂=0.427,
   or raw CC=0.497) sits far from both the true prior (0.248) *and* the current operating rate
   r₀=0.289, so **both** triggers fire and every recalibration variant lowers ZH F1 by
   0.05–0.11. The MLLM actively *misleads* on the no-drift language.
5. Sensitivity (f) confirms robustness of the *failure*: with TPR/FPR from half the train era,
   EN p̂=0.046±0.055 (F1 0.53±0.10) and ZH p̂=0.425±0.044 — the poor estimate is stable, i.e.
   not a small-sample fluke but a genuine era-shift of the estimator.

### Verdict (plain language)
**Zero-label MLLM prior estimation, as specified, does NOT earn the MLLM a method role here.**
The *idea* — expose the operating point as an O(1) knob and re-aim it at the new-era prior — is
validated (oracle-prior + labelled k=20 both recover the EN drop), but the MLLM cannot supply a
prior accurate enough to drive it: its per-video verdict is mediocre (train acc 0.65–0.70) and,
decisively, its false-positive rate **drifts across the very temporal boundary we are adapting to**
(EN FPR 0.372→0.238; ZH 0.314→0.402), which is exactly what breaks adjusted classify-and-count.
The result fails pre-registered criteria (2), (3), (4). The one honest positive is that the raw
(uncorrected) MLLM harmful-rate carries usable *drift-direction* signal on the drifting language
(EN gap-recovery 0.83), but it over-calls on the stable language (ZH −0.11) and so is unsafe to
deploy unguarded — and no zero-label trigger built on this estimate can tell the two cases apart.
Recommendation: drop MLLM prior-estimation for recalibration; if the memory's threshold knob is
kept as a contribution, drive it with the small-labelled k=20 mechanism (W4), which is cheap,
reversible, and — unlike the MLLM estimate — actually works.

### Jobs / artifacts / repro
- Judging: diag **12352** (probes correct), full **12353** (COMPLETED). Verdicts:
  `scripts/analysis/p1_out/harmful_verdicts.json`.
- Analysis (CPU, coordinator-approved scores-on-disk style, like W4):
  `logging/temporal_memory/{MHC,MHC_zh}_temporal_p1_zerolabel.json`.
- Code (no `src/` edits): `scripts/analysis/judge_archive_harmful.py`,
  `scripts/slurm/judge_archive_harmful.sbatch`, `scripts/analysis/p1_prior_recal.py`.
- Reproduces W4 exactly (checked): EN (a) 0.6273, (b) 0.7336, oracle-threshold 0.7646.
