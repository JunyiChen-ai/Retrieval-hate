# EXP: P10 — amplify the MLLM localization role (calibrate on HateMM spans, test once on HateClipSeg)

> **Status: PRE-REGISTERED (design frozen before the HateClipSeg test pass).**
> P6 (`research-wiki/EXP_p6_mllm_localization.md`) is the campaign's one positive MLLM
> method-role: reading frames+ASR, the MLLM localizes hate WITHIN videos significantly better
> than the CLIP-visual memory scorer (HateClipSeg within-video AUC 0.5435, CI [0.533,0.554],
> p=5.4e-8; paired over memory +0.030, p=0.007) — but MODEST. With all nine accuracy routes
> refuted, P10 asks whether the localization gain can be made **substantial** and honest, by
> tuning the scorer on a **calibration dataset** and testing the single promoted config **once**
> on the held-out P6 harness.

## Two-dataset protocol (this is what makes free iteration legal)

- **CALIBRATION = HateMM gold `hate_snippet` spans** (`data/gt/HateMM/hate_spans.json`; 427
  hateful videos, 391 with both-class seconds; span median 32s, ~46% coverage — non-trivial;
  provenance `EVAL_localization_hatemm.md` §1). A P6-style within-video localization eval
  (`scripts/analysis/p10_eval_hatemm.py`): windows → 1-fps seconds (label = second-midpoint in a
  gold span), within-video AUC over hateful videos with both-class seconds + AP-hateonly + a
  random control. **On this set I iterate freely and exhaustively** (window K, prompt wording,
  few-shot exemplars, ASR-weighting/aggregation, scorer model) — **every config is logged**
  (no silent shopping). No HateClipSeg contact.
- **HELD-OUT TEST = HateClipSeg**, ONE pass with the single promoted config on the frozen P6
  harness (`p6_eval_localization.py`; same 395-video split, same within-video AUC + CI + AP,
  same controls incl. the memory-scores row and random).

## Anchor (compute FIRST — sets the bar)

The frozen P6 config = the P3 scorer (`score_segments_mllm.py`, Qwen2.5-VL-7B, frames + K-window
ASR → integer 0–3), K=30/M=120. Its HateMM-calibration within-video AUC is the **anchor**; the
promotion bar is stated relative to it. (Data point already in hand: the same scorer at K=4 gives
HateMM wv-AUC 0.5478, CI [0.533,0.563], p=3.9e-8, n=389.)

**ANCHOR (P6 config, K=30, HateMM): wv-AUC = `<filled after job 12474>`.**

## Iteration grid (HateMM calibration — logged, cheap→expensive)

CPU-only (re-aggregate existing scores, no re-scoring):
- **A-gate**: zero-weight windows with no speech (localization is speech-borne per P6 mechanism).
- **A-lex**: weight each window score by an ASR hate-lexicon hit count (`HateClipSeg/lexicons.json`
  is EN; used read-only as a generic cue, NOT a HateClipSeg label).
- **A-fuse**: combine K=4 + K=30 anchor scores (coarse×fine).

GPU (one scoring pass each; `p10_score_segments.py`, prompt/model variants):
- **K60**: K=60/M=120 — finer localization windows.
- **fewshot**: K=30 + in-context 0–3 rating exemplars in the prompt.
- **speech**: K=30 + a speech-focused prompt (rate the SPOKEN hate in this window).
- **32B**: K=30 + Qwen2.5-VL-32B-Instruct (bf16, 1×A100-80G) — stronger scorer, only if a 7B
  variant is close to the bar.

## Pre-registered promotion bar

A config is promoted to the single HateClipSeg test **iff**, on the HateMM calibration set, its
paired within-video AUC beats the **anchor** (P6 config) by **≥ +0.04** with the paired
bootstrap 95% CI **excluding 0**. If several clear, the highest paired Δ is promoted. **If none
clears, P10 dies calibration-side — the HateClipSeg test is never touched and P6 stands as-is.**

## Pre-registered substantial bar (the goal's bar, on the HateClipSeg test)

- **HateClipSeg wv-AUC ≥ 0.60** (vs P6's 0.5435, memory 0.514) = **clear success** — substantial,
  novel MLLM localization role.
- **0.56 ≤ wv-AUC < 0.60** with CI excluding P6's 0.5435 = **modest amplification** (honest
  report; user decides if it's enough).
- **wv-AUC < 0.56** = amplification did not transfer; **P6 stands as-is**.

One test touch total. HateClipSeg controls (memory row, random) recomputed unchanged.

## Hard rules

SLURM only (no `--time`, `HF_HUB_OFFLINE=1`, `WANDB_MODE=disabled`), foreground `sacct` polling;
calibration scoring uses hateful-only gt (`data/gt_p10hate/`, 427 vids) to bound GPU; ASR re-binned
on CPU from the stored word timestamps (`p10_rebin_asr.py`, no Whisper re-run). No `.pt` in git;
32B cache deleted after use; quota watch. Report the full HateMM calibration leaderboard BEFORE
the test pass.

---

## HateMM CALIBRATION LEADERBOARD

_(anchor + every iterated config appended here before any test pass)_

## HateClipSeg TEST (promoted config only)

_(one pass; only if a config clears the promotion bar)_
