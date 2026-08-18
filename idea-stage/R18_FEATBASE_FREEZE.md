# R18-FEATBASE — freeze

**Date** 2026-08-18 · **Type** preparation round (feature-base probe), *not* a method round ·
**Cost** ¥0 (no cloud, no paid API, no annotation; HuggingFace checkpoint download only) ·
**Hardware** local RTX 5090, conda `HateVideo`, torch 2.7.1+cu128.

This file is committed **before any detector run on the new features exists**. Everything below —
arms, seeds, statistic, bar, and the sentence that must be written on failure — is fixed here.

---

## 0. Why this round exists

`idea-stage/R16_DETBASE_RESULT.md` established a competent detector base on HateClipSeg:
official ActionFormer on 4-FPS CLIP-L/14-336, `rawseg` ground truth, 237/39/119 split,
test **F1@tIoU 0.5 = 42.02 ± 0.73** for V⊕A⊕T early fusion (38.22 ± 1.63 visual only).
R14 (`R14_WVD_RESULT.md`) and R17 (`R17_OCRV_RESULT.md`) then killed both span-scoring
mechanisms available on that base: within-video ranking objectives (Δ −0.0052 wv-AUC) and a
dense OCR channel (Δ −0.16 F1). The head/objective axis on this substrate is spent.

One lever from the TAL literature has never been pulled here: **the feature backbone**.
OpenTAD's controlled study (arXiv 2502.20361, Table 4) holds the detection head fixed
(ActionFormer) and swaps the video feature on THUMOS-14:

| feature | THUMOS-14 mAP with ActionFormer |
|---|---|
| TSN-R50 | 49.79 |
| VideoSwin-L | 58.78 |
| SlowFast-R101 | 63.04 |
| VideoMAE-H | 67.23 |
| **InternVideo2-6B** | **72.36** |

`[read-method, fetched from the paper's HTML 2026-08-18]`. The same study prices five years of
neck/head architecture at ~0.5 mAP. Feature swap is worth **+22.6**; the head axis is noise.
This round measures how much of that lever survives on HateClipSeg localization.

**This is the last unpulled lever on the temporal-localization goal.** If the feature base does
not move the number either, the goal is unreachable under this project's current constraints and
this round is the evidence for that.

## 1. Feature choice and its justification

**Chosen: VideoMAEv2 ViT-g/14, checkpoint `vit_g_hybrid_pt_1200e_k710_ft`** — UnlabeledHybrid-1M
self-supervised pretraining (1200 epochs, dual masking) followed by supervised Kinetics-710
fine-tuning. 1.01 B parameters, `embed_dim` 1408, `depth` 40, `tubelet_size` 2, 16 frames,
224×224, patch 14.

Why this one, in order of the constraints that bind:

1. **InternVideo2-6B is the strongest in OpenTAD's table but is not extractable here.** 6 B
   parameters at ~6× the per-clip cost of ViT-g puts the extraction well past the round's ~3 h
   GPU budget, and the `OpenGVLab/InternVideo2-Stage2_6B` weights are not a plain public
   download. VideoMAE-H at 67.23 already carries **+17.4 of OpenTAD's +22.6**, and ViT-g is the
   larger sibling of that entry.
2. **VideoMAEv2-g is *the* feature backbone of the current TAL state of the art.** AdaTAD
   (arXiv 2311.17241), the top THUMOS-14 entry in the landscape table (76.9 avg mAP / 80.9@0.5),
   is VideoMAEv2-g. The published THUMOS/ActivityNet VideoMAEv2 features are extracted from
   exactly this `vit_g_hybrid_pt_1200e_k710_ft` checkpoint.
3. **The supervised-finetuned checkpoint, not the raw MAE pretrain.** `OpenGVLab/VideoMAEv2-giant`
   on HuggingFace is the self-supervised pretrain only; MAE features without supervised
   fine-tuning are known to be weak under a frozen-feature protocol. The K710-finetuned weights
   are mirrored at `Sam3000/vit_g_hybrid_pt_1200e_k710_ft.pth` (2.03 GB). **Verified before use:**
   the state dict is `patch_embed.proj.weight (1408,3,2,14,14)`, 40 blocks, `fc_norm`,
   `head.weight (710,1408)` — the exact K710 architecture — and it loads into the official
   `OpenGVLab/VideoMAEv2-giant` `VisionTransformer` with **zero missing and zero unexpected
   keys**. Functional check on a real HateClipSeg video: mean top-1 softmax 0.126 and mean
   predictive entropy **4.77 nats against a 6.57-nat uniform**, with the same K710 class returned
   for 9 consecutive clips. A corrupted or randomly-initialised checkpoint cannot do that.
4. **Architecturally orthogonal to the incumbent.** CLIP-L/14-336 is an image tower applied
   frame-by-frame; it has no motion. VideoMAEv2-g is a spatio-temporal tower over 16-frame
   clips. This is the substitution OpenTAD's table is about.

## 2. Extraction contract

`scripts/r18_featbase/extract_dense_vmae.py`.

- Decode at **8 FPS**, short-side resize to 224 (bicubic) + centre crop 224, ImageNet
  mean/std (`preprocessor_config.json` of the official repo).
- Output row `i` is the **same instant as CLIP row `i`**, `t = i/4 s`. The array length `T` is
  asserted equal to the existing `dense4fps_clipL336/<vid>.npy` length, so the two arrays
  concatenate index-by-index and every ActionFormer setting (`feat_stride: 1`, `num_frames: 0`,
  `default_fps: 4`, `max_seq_len: 2304`) is carried over **unchanged** from R16.
- The clip fed to the encoder for row `i` is 8-FPS frames `[2i−8, 2i+8)`, edge-clamped: a
  **2.0 s window centred on t** (centre offset −62 ms, a quarter of one output step).
- Feature = `fc_norm(mean over tokens)`, 1408-d, the standard VideoMAEv2 feature read-out.
- **`--hop 2`.** Measured throughput is 36 clip/s, so evaluating the encoder on every 4-FPS row
  costs 3.0 h — the whole round's budget. The encoder is therefore evaluated on **every second
  row (2 Hz)** and the result is **linearly interpolated back onto the full 4-FPS grid**
  (≈1.5 h). The grid the detector sees is unchanged at 4 Hz; only the encoder's evaluation rate
  is halved. Adjacent 4-FPS windows already overlap by 87.5% of their 2.0 s receptive field, so
  a 0.5 s hop is 75% overlap. **Declared as a deviation from a pure 4 Hz extraction up front.**
- Attention is routed through `F.scaled_dot_product_attention` in place of the repo's manual
  softmax — numerically equivalent, and the only code change to the released model.
- `yt_NzvfkIYS5Yg` (0 decodable frames) gets an all-zero `(T,1408)` array, exactly as in R16 D2,
  so all systems see the identical 119 test videos.

Feature streams are concatenated **raw, without normalisation**, exactly as R16 and R17 did.
Recorded scales: CLIP 1.028 RMS, Wav2Vec-Emotion 0.078, BERT 0.493, VideoMAEv2-g 0.799. The new
stream sits inside the range the existing fusion already spans.

## 3. Arms

Ground truth is **`rawseg`** throughout — the convention R16 §3 established as the paper's, and
the only one on which this project's detector is in the published regime. Everything except the
input feature file is byte-identical to R16's `vat_rawseg` run.

**Primary family** (the 42.02 base):

| arm | input | dim | feature dir |
|---|---|---|---|
| **A1** `clip_vat` | CLIP ⊕ A ⊕ T | 2816 | `dense4fps_vat` |
| **A2** `mae_vat` | **VideoMAEv2-g** ⊕ A ⊕ T | 3200 | `dense4fps_mat` |
| **A3** `maeclip_vat` | **VideoMAEv2-g** ⊕ CLIP ⊕ A ⊕ T | 4224 | `dense4fps_mvat` |

A1 is a re-run, not R16's stored number: it uses this round's seeds so that all three arms are
**seed-paired**.

**Secondary family, declared here, descriptive only, does not gate anything** (visual-only is
the setting OpenTAD's table is measured in):

| arm | input | dim | feature dir |
|---|---|---|---|
| **B1** `clip_v` | CLIP | 1024 | `dense4fps_clipL336` |
| **B2** `mae_v` | VideoMAEv2-g | 1408 | `dense4fps_vmaev2g` |

## 4. Protocol, seeds, and what may be selected

Identical to R16 §4:

1. Split `p11_split.json`, 237 train / 39 val / 119 test, unchanged.
2. Config `third_party/actionformer/configs/hateclipseg_clip.yaml` with only `input_dim` and
   `feat_folder` changed per arm. **Nothing is swept, nothing is tuned.**
3. Selectable on **val only**: the training epoch (of the 35 run) and one global proposal-score
   threshold, both maximising val F1@tIoU 0.5, then applied unchanged at 0.3 and 0.7 and on test.
4. **Test is opened once per arm**, after selection, through the `--touch-test` path. No
   test-derived quantity re-enters any choice.
5. **Seeds `7300, 7301, 7302`**, the same three in every arm. Fresh: prior rounds consumed
   0-89, 42, 4299-4310, 4399, 5100-5102, 6200-6202, 6210-6212, 6280, 6299.
6. Metric `scripts/r16_detbase/eval_f1.py`, unchanged.

## 5. The decision rule

Endpoint: **test F1@tIoU 0.5**, mean over the three seeds.

For each candidate arm `X ∈ {A2, A3}` form the **seed-paired** difference
`d_s = F1(X, seed s) − F1(A1, seed s)`, `s ∈ {7300, 7301, 7302}`; report `d̄` and the
Student-t 95% CI `d̄ ± 4.303 · sd(d)/√3`.

> **The bar: the feature axis is worth reopening a method round iff, for at least one of A2 and
> A3, `d̄ ≥ +2.0` and the lower bound of that CI is `> 0`.**

Both conditions, on the primary family, on test F1@tIoU 0.5. The secondary family B1/B2, the
other tIoU thresholds, the proposal-pool recall and every geometry statistic are **descriptive**
and may not be substituted for this endpoint.

**Pre-committed failure sentence.** If neither A2 nor A3 clears the bar, the result file must
say, verbatim:

> The feature backbone is the last lever the TAL literature offers, it is worth +22.6 mAP on
> THUMOS-14 under a fixed head, and on HateClipSeg localization it does not move the number.
> Together with R14 (span scoring) and R17 (input channels), all three axes of the detector are
> now measured and none responds. The temporal-localization goal is not reachable under this
> project's constraints, and further work on it requires a scope change ruled by the user, not
> another substrate.

## 6. Secondary diagnostic, declared, not a gate

`scripts/r16_detbase/diagnose.py` re-run per arm: **proposal-pool recall** at tIoU 0.3/0.5/0.7
(what fraction of gold segments *some* proposal covers, ignoring score), pool size, and
intervals-per-video / median length. This answers the question the primary endpoint cannot: if
the feature moves the number, is it moving **candidate coverage** or **ranking quality**?
R16's reference values for CLIP-visual on `rawseg` are pool recall 97.42 / 90.68 / 64.97 over a
23 800-proposal pool.

## 7. Deviation policy

Anything not written above that has to change during execution is written into the result file
as a numbered deviation before the number it affects is reported. Documentation staleness is
fixed on sight and never blocks (project rule, 2026-08-05).

## 8. Artifacts

| artifact | path |
|---|---|
| this freeze | `idea-stage/R18_FEATBASE_FREEZE.md` |
| extractor | `scripts/r18_featbase/extract_dense_vmae.py` |
| stream fusion | `scripts/r18_featbase/fuse_feats.py` |
| detector runner | `scripts/r16_detbase/run_af.py` (unchanged, re-used) |
| features | `data/CLIP_Embedding/HateClipSeg/dense4fps_{vmaev2g,mat,mvat}/` |
| results | `idea-stage/r18_featbase/out/res_{clip_vat,mae_vat,maeclip_vat,clip_v,mae_v}.json` |
| logs | `logging/runs/r18_{extract,fuse,af}/` |
| result write-up | `idea-stage/R18_FEATBASE_RESULT.md` |
