# EXP_p3_evidence_pooling — MLLM evidence-density weighted pooling of the video embedding

**Status:** PRE-REGISTERED (code + gates fixed before any GPU run) · **Started:** 2026-07-06 ·
**Owner:** subagent P3 (方法开发, src/ 只读复用, 新缓存不覆盖旧缓存)

**Front (P3):** the direct attack on the diagnosed EN root cause. W2/EXP_mm established that
**71% of MHC-EN true-Hateful evidence is carried by speech / on-screen-text**, localized in a
minority of a video's segments, while the whole-video embedding the retrieval head consumes is a
**MEAN pool over frames/segments** — which dilutes that localized evidence. Segment *supervision*
was tried and closed as a channel on EN (EXP_mm_segment_keys: 3/3 seed below floor). Segment-aware
*POOLING of the input embedding* — reweighting the mean by where the evidence actually is — was
**never tried**. That is this experiment.

The reweighting signal is an MLLM (Qwen2.5-VL-7B) reading each segment and scoring its
hate-evidence density 0–3. This gives the MLLM a **real method role**: not a label, not a judge of
the final decision, but an *input-processing* signal that concentrates the frozen-encoder video
embedding onto evidence-bearing segments before any training or retrieval happens.

---

## 0. Pre-registration (written and committed BEFORE the scoring jobs are submitted)

Everything in §0 is fixed now. No knob below is tuned on any test measurement. The primary/secondary
weighting are both declared here; the training arms are declared here; the probe gate and the
success criteria are declared here.

### 0.1 MLLM segment scoring (unsupervised input processing — no labels, no leakage)

- Model: `Qwen/Qwen2.5-VL-7B-Instruct`, frozen, greedy (`do_sample=False`), fixed prompt, offline.
- Granularity: the **same** K=4 contiguous windows over M=16 uniformly-sampled frames used by
  `generate_subclip_embedding_HF.py` (identical `_sample_frame_indices` + `_window_bounds`), so the
  frames the MLLM scores are exactly the frames pooled. Each window is scored **in isolation**
  (only that window's ≤4 frames + that window's Whisper ASR text), so the score is the density of
  hate evidence *within that segment*, not cross-window context.
- Output: integer in {0,1,2,3}. 0 = no evidence of hate; 1 = ambiguous/mild; 2 = clear but not
  explicit; 3 = explicit hate evidence. A parse failure or refusal → score 0 (recorded, audited).
- Coverage: **every** video in train+val+test of each dataset is scored. This is unsupervised
  processing of the model input; labels are never read, so scoring the val/test splits carries no
  leakage (identical status to CLIP feature extraction, ASR, and the E0b archive).
- ASR text channel: per-window Whisper-large-v3 transcript (`data/ASR/<DS>/<split>_asrK4_*.jsonl`),
  same window-boundary contract. Windows with no ASR text pass `(no speech)`; the MLLM still sees
  the frames (and can read on-screen text from them).

### 0.2 Weighted pooling (the intervention)

Let `s = [s_0..s_{K-1}]` be a video's per-segment scores and `e_i` its K sub-clip CLIP embeddings
(from the existing `*_subclipK4_*` cache; the sub-clip visual features are **not** re-extracted).
The pooled video **image** embedding is

```
  v_img = Σ_i w_i · e_i
```

- **PRIMARY** weighting: `w = softmax(s / T)`, `T = 1.0` (pre-registered, not tuned).
- **SECONDARY** (single pre-registered milder variant): `w = (1 + s) / Σ_j (1 + s_j)`.

The **text** stream is unchanged (copied byte-for-byte from the whole-video cache). Only `img_feats`
is replaced. Everything else in the cache (`ids`, `labels`, order) is copied unchanged, so the cache
is a drop-in `--model` swap for `run_rac.py`.

Floor = uniform weights `w_i = 1/K` = **mean pooling of the same sub-clips**.

Note (frame-count provenance, recorded honestly): the sub-clip cache pools **M=16** frames (4 per
window); the paper's whole-video cache pools **M=8** frames. So the mean-of-sub-clips floor is a
16-frame mean and will NOT be bit-for-bit equal to the published 8-frame whole-video cache
(measured max |Δ|=2.38 on MHC train img_feats). This experiment's floor is therefore the
**16-frame mean-of-sub-clips**, trained fresh; the published 0.7826/0.7113 (8-frame) is reported
alongside for context but the pre-registered A/B is *mean-vs-weighted over the identical 16-frame
sub-clip set* — a clean single-variable comparison (only the pooling weights differ).

### 0.3 Sanity (must pass before any probe or training)

1. **Equal-weights reproduces mean pooling bit-for-bit.** Feeding a constant score vector to the
   weighting function yields `w_i = 1/K` and `Σ_i (1/K) e_i`, which must be `torch.equal` to the
   mean-pool cache's `img_feats`. (Pre-verified on MHC train: uniform-weighted-sum == sub-clip mean,
   max |Δ| = 0.0.) The builder asserts this on every produced dataset/split.
2. **Text/labels/ids untouched.** The weighted and floor caches must have `torch.equal` text_feats,
   equal labels, identical id order (asserted).

### 0.4 Probe gate (mandatory, BEFORE any training — probe-before-train discipline)

On the **TRAIN split only**, raw frozen-encoder space, **no trained head**. Leave-one-out kNN vote
(cosine similarity, similarity-weighted "arithmetic" vote, k=20 to match training `--topk 20`):

- **PRIMARY gate metric:** LOO kNN accuracy of the **video embedding** = concat
  `[l2n(v_img) | l2n(text)]` (the representation the head actually consumes), weighted-pooled vs
  mean-pooled. **Gate: weighted ≥ mean on EN train.**
- **Diagnostics reported (not gates):** img-only LOO kNN accuracy (most sensitive to the pooling
  change since text is shared), LOO accuracy at k∈{1,5,10,20}, and macro-F1.
- **Score-concentration check (reported):** do the MLLM scores concentrate on hateful videos —
  i.e. is the within-video score variance (or max score) higher for true-Hateful than for benign
  train videos? A pooling reweight can only help if the scores actually separate evidence.

**Gate decision:** if the PRIMARY gate FAILS on EN, the EN training arm is **stopped and reported
as a probe kill** (a verdict is a finding). ZH and HateMM proceed to training only if their own
PRIMARY gate passes. (HateMM needs its ASR generated first.)

### 0.5 Training protocol (only for datasets whose probe passes)

- Encoder space: **CLIP** (caches exist). Qwen-space sub-clip regeneration only if CLIP-space shows
  a real win (expensive; not pre-authorized here).
- Seeds: **{0,1,2}** from the start. 30 epochs. Standard recipe (identical to
  `experiments_video.sh` / `train_consensus_mm.sbatch`: `--fusion_mode align --loss triplet
  --metric cos --hybrid_loss True --majority_voting arithmetic --topk 20 --warmup 5 --lambda_seg 0
  --dropout 0.2 0.4 0.1`). LAMBDA_SEG=0 (whole-video path; segment loss inert).
- New GROUP: `RAC_video_p3pool`. FORCE=False (existing groups/ckpts can never be overwritten).
  Conditions distinguished by `--exp_comment` (`_p3mean` / `_p3wsoftT1` / `_p3wmild`) so same-seed
  runs never collide.
- Conditions:
  - `floor`  = mean pooling (`_p3mean`) — must reproduce the 16-frame mean numbers.
  - `weighted` (PRIMARY) = softmax T=1.0 (`_p3wsoftT1`).
  - `weighted-mild` (SECONDARY) = normalize(1+s) (`_p3wmild`), reported separately, never mix-selected
    with PRIMARY.
- Report BOTH protocols (val-selected warmup≥5 AND final-epoch), acc + macro-F1, paired per-seed
  deltas. One test measurement per cell.

### 0.6 Success criteria (pre-registered)

1. Equal-weights reproduces floor **bit-for-bit** (sanity §0.3.1). [Hard prerequisite.]
2. Probe gate passes before any training (§0.4). [Hard prerequisite per dataset.]
3. `weighted` beats `floor` with **≥2/3 seeds positive AND mean paired ΔF1 > 1 pt** (the ~1.6-video
   noise floor on these test sets) on **at least one dataset** under **BOTH** protocols, with **no
   >1pt harm** elsewhere.

Anything weaker than (3) = reported as **within-noise, no claim**. If the probe or training says no,
that is an honest kill and is reported as the verdict. No cross-seed ensembles; no test-set tuning.

### 0.7 Datasets & floors to beat (CLIP-space, val-selected, published)

| dataset | acc | macro-F1 | note |
|---|---|---|---|
| MHC-EN | 0.7826 | 0.7113 | 8-frame whole-video floor (16-frame mean-of-subclips floor trained fresh here) |
| MHC_zh | 0.8054 | 0.7706 | ZH evidence is visual/on-screen-text; EN root cause may not transfer |
| HateMM | 0.828 (acc) | — | needs ASR generated first |

---

## 1. Sanity results

(pending — filled by `scripts/analysis/p3_probe.py --sanity` on the built caches)

## 2. Probe results

(pending — filled per dataset after scoring)

## 3. Training results

(pending — gated on §2)

## 4. Verdict

(pending)
