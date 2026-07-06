# EXP: P6 — MLLM evidence scores for span-free temporal localization

> **Status: PRE-REGISTERED (design frozen before any MLLM score is evaluated).**
> Motivation, pipeline, conditions, metrics, and the success bar are committed before running
> the scoring/eval. Results are appended in `## RESULTS`. Numbers by
> `scripts/analysis/p6_eval_localization.py`; MLLM scores by the frozen P3 scorer
> `scripts/analysis/score_segments_mllm.py` (prompt/model/greedy UNCHANGED); ASR by
> `src/utils/generate_segment_asr_HF.py`.

## Motivation

The existing zero-training localization capability (`research-wiki/EVAL_localization_hateclipseg.md`)
is an **existence proof only**: cross-dataset consensus-kNN scoring of CLIP-visual windows gives
best within-video mean-AUC 0.526 (CI [0.505, 0.547], sign-test p=0.0066 — significant in exactly
1 of 4 cells) and +0.088 frame-AP over random. It is weak because **CLIP-visual memory keys are
blind to speech-borne hate**, and HateClipSeg hate is largely spoken. P6 asks: does an MLLM that
**reads frames + ASR per window** localize substantially better? If yes, the MLLM earns a real,
removable role in the localization capability; if not, the capability stays existence-proof grade
and the MLLM-localization role is an honest kill.

## Data & harness (reuse EVAL_localization_hateclipseg.md EXACTLY)

- HateClipSeg alive subset: **395 videos**, cleaned gold segments (`data/gt/HateClipSeg/
  gold_segments.json`, 10,572 kept segments tiling [0,D); gold spans are **VALIDATION ONLY** — no
  HateClipSeg label enters any scoring path). Our declared split = all 395 (zero-training, no
  leakage). Same broadcast control, same 1-fps second→window mapping `min(K−1,⌊mK/D⌋)`, same
  within-video / frame / segment protocols and estimators, imported read-only from
  `scripts/analysis/eval_localization_hateclipseg.py`.
- **Granularity = K=30, M=120** (window median ≈ 8s, density-matched to the gold-segment median
  8.12s; ≈ 11,850 windows). This is the granularity where within-video localization is measurable
  (30 rankable windows/video) and where ASR is localized to ~8s; it matches the existing doc's
  K=30 memory numbers for a same-granularity head-to-head. Choosing K=30 (not P3's default K=4) is
  the **window-alignment adaptation** anticipated in the brief — it changes only the window count,
  not the P3 prompt/model/decoding, which are frozen.

## Pipeline (two GPU jobs, then CPU eval)

1. **ASR** — `generate_segment_asr_HF.py --dataset HateClipSeg --splits test --num_frames 120
   --num_subclips 30 --timestamps word` → `data/ASR/HateClipSeg/test_seen_asrK30_whisper-large-v3.jsonl`
   (Whisper large-v3, word timestamps binned to the 30 windows; language=en).
2. **MLLM scoring** — the **frozen** P3 scorer `score_segments_mllm.py --dataset HateClipSeg
   --splits test --num_frames 120 --num_subclips 30 --asr_tag asrK30_whisper-large-v3`
   → `data/MLLM_scores/HateClipSeg/test_seen_segscoreK30_qwen.jsonl` (Qwen2.5-VL-7B, greedy, each
   window scored IN ISOLATION on its ≤4 frames + its window ASR → integer hate-evidence density
   0..3). Prompt/model/decoding identical to P3 — **not retuned for HateClipSeg** (zero-shot; the
   rubric says "hate evidence density", so only a **threshold-free ranking** metric is valid, no
   operating point to tune).
3. **Eval** (CPU) — `p6_eval_localization.py` builds the [395, 30] window-score matrix for each
   condition and runs the SAME metric functions.

## Conditions (one evaluation pass, no metric shopping)

| id | condition | window-score source |
|----|-----------|---------------------|
| **a** | memory (baseline, reproduce) | consensus-kNN, `knn_hatemm_subclip` @ K=30 (existing doc's best memory config at this K; `knn_hatemm_video` also reported) — cached `loc_out_hcs/scores_knn_*_K30.npz` |
| **b** | **MLLM (ours)** | P3 scorer integer scores 0..3 |
| **c** | combination (pre-registered, ONE rule) | **per-video rank-average of (a) and (b)** — within each video, average the two windows' ranks (normalised rank/(K−1)); frozen now, no alternatives |
| **d** | random control | `np.random.RandomState(0)` |
| **e** | broadcast control | per-video mean of (b) broadcast to all windows (pooled-metric control; within-video AUC = 0.5 by construction) |

## Metrics (same estimators as the existing doc)

- **within-video mean AUC** over videos with both classes (the sharp localization diagnostic) +
  **bootstrap 10k 95% CI** + one-sided sign-test vs 0.5. **This is the PRIMARY metric.**
- frame-level AP/AUC (protocol-full + toxiconly) and segment-level AP/AUC (duration-weighted),
  reported as supporting/pooled evidence. Per the existing doc, pooled AP mostly reflects
  video-level toxicity **density**, not within-video localization — so it is secondary here.

## Pre-registered success bar

**PRIMARY (MLLM earns a removable localization role) — ALL of:**
1. within-video mean-AUC(**b**) > within-video mean-AUC(**a**) AND > (**d**);
2. **b**'s 95% bootstrap CI **excludes 0.5** and sign-test p < 0.05.

**SECONDARY (supporting, NOT required):**
3. frame-full AP(**b**) − AP(**a**) ≥ **+0.176** (= 2× the existing +0.088 headline delta) — the
   pre-registered "substantial" AP bar; stated now, acknowledged as demanding (pooled AP is
   density-dominated, so within-video AUC is the real localization test);
4. within-video AUC(**c**) ≥ max(**a**,**b**).

**KILL:** if (1)/(2) fail — the MLLM does not beat memory + random on within-video AUC with CI
excluding null — the localization capability **stays existence-proof grade**; that is an honest
kill of the MLLM-localization role, reported with the mechanism.

## Hard rules

GPU via SLURM (no `--time`, `HF_HUB_OFFLINE=1`, `WANDB_MODE=disabled`); one ASR job + one scoring
job; poll `sacct`; resume-safe, FORCE=False, no cache overwrite; no `.pt` in git; disk under quota;
commit (no push). The scorer is byte-frozen from P3 (only `--num_frames/--num_subclips/--asr_tag`
differ, i.e. window alignment).

---

## RESULTS

_(appended after the scoring + eval runs)_
