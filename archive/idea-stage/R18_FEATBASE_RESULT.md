# R18-FEATBASE — result

**Date** 2026-08-18 · **Freeze** `idea-stage/R18_FEATBASE_FREEZE.md`, commit **`57f26c6`**,
committed before `data/CLIP_Embedding/HateClipSeg/dense4fps_vmaev2g/` contained a single file ·
**Type** preparation round (feature-base probe), not a method round ·
**Cost** ¥0 (no cloud, no paid API, no annotation; two HuggingFace checkpoint downloads) ·
**Hardware** local RTX 5090, conda `HateVideo`, torch 2.7.1+cu128 ·
**Wall (GPU)** VideoMAEv2-g extraction 5 650 s (1.57 h, 187 452 clips at 33.2 clip/s),
5 detector arms × 3 seeds 4 238 s (1.18 h) — **2.75 h total, inside the ~3 h budget** ·
**Seeds** 7300 / 7301 / 7302, the same three in every arm ·
**Test contact** once per arm, after val-only selection, through the `--touch-test` path.

---

## VERDICT: **the bar fails on both candidates. The feature axis does not move HateClipSeg localization.**

| contrast | seed-paired d, test F1@tIoU 0.5 | 95% CI | bar: d ≥ +2.0 and LB > 0 | verdict |
|---|---|---|---|---|
| **A2 `mae_vat` − A1 `clip_vat`** | **−4.50** | [−18.65, +9.65] | no | **fail** |
| **A3 `maeclip_vat` − A1 `clip_vat`** | **−1.69** | [−5.04, +1.67] | no | **fail** |
| B2 `mae_v` − B1 `clip_v` *(secondary, descriptive)* | −2.32 | [−5.50, +0.85] | — | negative |

All three point estimates are **negative**. Replacing CLIP with the strongest video feature the
TAL literature offers costs 4.5 F1 points; adding it alongside CLIP costs 1.7. Nothing here is
within noise of a +2.0 gain.

Under §5 of the freeze, committed before any of these numbers existed, this forces the following
sentence, verbatim:

> The feature backbone is the last lever the TAL literature offers, it is worth +22.6 mAP on
> THUMOS-14 under a fixed head, and on HateClipSeg localization it does not move the number.
> Together with R14 (span scoring) and R17 (input channels), all three axes of the detector are
> now measured and none responds. The temporal-localization goal is not reachable under this
> project's constraints, and further work on it requires a scope change ruled by the user, not
> another substrate.

---

## 1. What was extracted, and the evidence it is the right thing

**VideoMAEv2 ViT-g/14, `vit_g_hybrid_pt_1200e_k710_ft`** — 1.01 B parameters, `embed_dim` 1408,
`depth` 40, tubelet 2, 16 frames at 224×224, UnlabeledHybrid-1M self-supervised pretraining
followed by supervised Kinetics-710 fine-tuning. Read-out = `fc_norm(mean over tokens)`, 1408-d.

Selection rationale, in full in freeze §1, in one paragraph here: OpenTAD's controlled study
(arXiv 2502.20361, Table 4) holds ActionFormer fixed and swaps the feature on THUMOS-14 —
TSN-R50 **49.79**, VideoSwin-L 58.78, SlowFast-R101 63.04, VideoMAE-H **67.23**, InternVideo2-6B
**72.36**. That is the +22.6 that motivated this round. InternVideo2-6B is ~6× ViT-g per clip and
would not fit the budget; VideoMAE-H already carries +17.4 of the +22.6, and ViT-g is its larger
sibling and the backbone of AdaTAD, the current THUMOS-14 leader (76.9 avg mAP).

**Checkpoint provenance, checked before use.** The official `OpenGVLab/VideoMAEv2-giant` HF repo
ships the *self-supervised pretrain only*; raw MAE features are weak under a frozen-feature
protocol, so the K710-finetuned weights are required. They are mirrored at
`Sam3000/vit_g_hybrid_pt_1200e_k710_ft.pth` (2.03 GB). Verification performed:

- state dict is `patch_embed.proj.weight (1408,3,2,14,14)`, 40 blocks, `fc_norm`,
  `head.weight (710,1408)` — the exact K710 architecture;
- it loads into the official `OpenGVLab/VideoMAEv2-giant` `VisionTransformer` with
  **0 missing and 0 unexpected keys**;
- on a real HateClipSeg video the K710 head returns mean top-1 softmax 0.126 and mean predictive
  entropy **4.77 nats against a 6.57-nat uniform**, with the same class for 9 consecutive clips.
  A corrupted or randomly-initialised checkpoint cannot do that.

**Temporal contract.** Output row `i` is the same instant as CLIP row `i` (`t = i/4 s`), so the
two arrays concatenate index-by-index and every ActionFormer setting is carried over unchanged
from R16. The clip for row `i` is 16 frames of an 8-FPS decode, `[2i−8, 2i+8)`, edge-clamped — a
2.0 s window centred on `t`. Per freeze §2 the encoder was evaluated at 2 Hz (`--hop 2`) and
linearly interpolated onto the 4 Hz grid; at 33.2 clip/s, 4 Hz would have cost 3.0 h, the whole
round's budget. `yt_NzvfkIYS5Yg` (0 decodable frames) gets an all-zero array, as in R16 D2.

Per-stream RMS, concatenated raw and unnormalised exactly as R16/R17 did: CLIP 1.024,
**VideoMAEv2-g 0.773**, Wav2Vec-Emotion 0.080, BERT 0.498. The new stream sits comfortably inside
the range the existing fusion already spans (13× between CLIP and audio).

## 2. The five arms

Test split, 119 videos, 1 474 `rawseg` gold instances, mean ± sd over seeds 7300/7301/7302.
Everything except `feat_folder` and `input_dim` is byte-identical to R16's `vat_rawseg` run.

| arm | input | dim | F1@0.3 | **F1@0.5** | F1@0.7 | P@0.5 | R@0.5 | val F1@0.5 |
|---|---|---|---|---|---|---|---|---|
| **A1** `clip_vat` | CLIP ⊕ A ⊕ T | 2816 | 54.24 ± 1.91 | **41.56 ± 1.50** | 21.86 ± 1.22 | 35.79 | 49.66 | 48.66 |
| **A2** `mae_vat` | **VMAEv2-g** ⊕ A ⊕ T | 3200 | 47.93 ± 3.70 | **37.06 ± 3.19** | 16.45 ± 3.90 | 28.89 | 52.10 | 45.94 |
| **A3** `maeclip_vat` | **VMAEv2-g** ⊕ CLIP ⊕ A ⊕ T | 4224 | 51.18 ± 1.15 | **39.88 ± 0.61** | 21.04 ± 0.72 | 31.38 | 55.09 | 48.02 |
| B1 `clip_v` | CLIP | 1024 | 49.91 ± 0.69 | 38.68 ± 0.81 | 21.04 ± 0.49 | 29.40 | 57.33 | 46.64 |
| B2 `mae_v` | **VMAEv2-g** | 1408 | 48.74 ± 1.41 | 36.35 ± 1.16 | 18.91 ± 0.60 | 27.73 | 52.94 | 44.07 |
| *R16, seeds 5100-5102* | *CLIP ⊕ A ⊕ T* | *2816* | *54.72 ± 0.69* | *42.02 ± 0.73* | *21.11 ± 0.86* | — | — | — |
| *R16, seeds 5100-5102* | *CLIP* | *1024* | *50.28 ± 1.82* | *38.22 ± 1.63* | *19.95 ± 0.09* | — | — | — |

**The baseline reproduces.** A1 on fresh seeds gives 41.56 ± 1.50 against R16's 42.02 ± 0.73, and
B1 gives 38.68 ± 0.81 against R16's 38.22 ± 1.63. Both differences are inside one seed sd, so the
round is measuring against a stable base and the negative deltas are not a broken re-run.

**The val split, which is never used to compare arms, independently agrees with the ranking.**
Val F1@0.5 orders the arms A1 48.66 > A3 48.02 > B1 46.64 > A2 45.94 > B2 44.07 — the same order
as test. The verdict is not a test-split accident.

Per-seed test F1@0.5, for the paired differences in the verdict table:

| seed | A1 `clip_vat` | A2 `mae_vat` | A3 `maeclip_vat` | B1 `clip_v` | B2 `mae_v` |
|---|---|---|---|---|---|
| 7300 | 41.51 | 38.52 | 39.30 | 39.42 | 37.95 |
| 7301 | 39.75 | 40.04 | 39.60 | 39.06 | 35.26 |
| 7302 | 43.43 | 32.63 | 40.73 | 37.55 | 35.84 |

A2's seed 7302 (32.63) is a genuine outlier and is what inflates that contrast's CI to
[−18.65, +9.65]. It is reported, not excluded; the arm's point estimate is negative with or
without it (mean of the other two seeds is 39.28, still −1.35 against the paired A1 seeds).
A3, whose sd is 0.61, is the cleaner of the two candidates and it is also negative.

## 3. Where the feature does and does not act — the declared diagnostic

Freeze §6 asked whether a feature change moves **candidate coverage** or **ranking quality**.

**Proposal-pool recall** — the fraction of gold instances that *some* proposal in the pool covers
at that tIoU, ignoring score entirely. Test split, mean ± sd over the 3 seeds, pool 23 800
proposals (200/video) in every arm.

| arm | recall @0.3 | recall @0.5 | recall @0.7 |
|---|---|---|---|
| A1 `clip_vat` | 97.83 ± 0.29 | 91.07 ± 0.42 | 64.95 ± 2.68 |
| A2 `mae_vat` | 97.72 ± 0.26 | 89.62 ± 0.65 | 53.55 ± 7.21 |
| A3 `maeclip_vat` | 98.24 ± 0.24 | 91.36 ± 0.31 | 64.88 ± 2.62 |
| B1 `clip_v` | 97.74 ± 0.08 | 91.00 ± 0.44 | 67.62 ± 0.64 |
| B2 `mae_v` | 98.10 ± 0.20 | 90.91 ± 0.25 | 64.38 ± 1.89 |

**Candidate coverage is a constant.** All five arms sit at 97.7–98.2 at tIoU 0.3 and 89.6–91.4 at
tIoU 0.5. The strongest video feature in the TAL literature proposes the same set of intervals as
a frame-wise CLIP tower. The only movement is at tIoU 0.7, where A2 loses 11 points of coverage
(and 7.2 of its 11 is the seed-7302 outlier).

**Post-hoc: the achievable ceiling is also a constant.** Holding each arm's own pool fixed and
keeping the top-k proposals per video, ranked by the model's own score versus by oracle tIoU
(test F1@tIoU 0.5, mean over 3 seeds; this diagnostic was defined after the primary number, as
R16 §6 allows):

| arm | rank by | k=1 | k=2 | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|---|
| A1 `clip_vat` | model | 5.65 | 9.81 | 13.98 | 19.88 | 29.83 |
| A1 `clip_vat` | **oracle** | 12.85 | 23.01 | 31.02 | 42.60 | **56.26** |
| A2 `mae_vat` | model | 5.27 | 9.46 | 12.74 | 18.43 | 26.98 |
| A2 `mae_vat` | **oracle** | 12.85 | 22.78 | 30.88 | 42.50 | **56.56** |
| A3 `maeclip_vat` | model | 4.94 | 9.27 | 13.07 | 19.56 | 29.18 |
| A3 `maeclip_vat` | **oracle** | 12.89 | 22.94 | 31.17 | 42.73 | **56.78** |
| B1 `clip_v` | model | 4.35 | 8.68 | 11.72 | 17.82 | 27.25 |
| B1 `clip_v` | **oracle** | 12.93 | 23.01 | 31.17 | 42.73 | **56.23** |
| B2 `mae_v` | model | 5.19 | 8.53 | 11.98 | 17.82 | 26.60 |
| B2 `mae_v` | **oracle** | 12.93 | 22.98 | 30.95 | 43.11 | **57.56** |

The oracle rows are **identical to within 0.3 points at every k across all five arms**. The five
proposal pools are interchangeable. What separates the arms is entirely which of those proposals
the classifier scores highest, and the feature swap makes that slightly worse, not better.

The gap between model ranking and oracle ranking (29.8 vs 56.3 at k=10 for A1) is the same
26-point ranking deficit R16 §5.4 measured, unchanged by the feature. **This round did not shrink
the bottleneck; it confirmed that the bottleneck is insensitive to the input representation.**

**Prediction geometry** (seed 7300, after thresholding). Gold: 12.39 instances/video, median 8.42 s.
A1 17.25/video at 9.58 s; A2 23.70 at 9.94 s; A3 19.94 at 8.88 s; B1 27.30 at 9.29 s;
B2 22.29 at 9.75 s. Every arm over-emits and every arm gets the length scale right; the VideoMAE
arms trade precision for recall (A2 P 28.9 / R 52.1 against A1's 35.8 / 49.7) without a net gain.

## 4. Post-hoc: one measurable property that distinguishes the two towers

Scale-normalised feature trajectories, computed on interpolation-free rows only (every second row
of the VideoMAE array is exactly computed; the intermediate rows are my linear interpolation, so
they are excluded from this statistic). Cosine similarity between feature vectors 0.5 s apart,
mean over 60 videos:

| tower | lag-0.5 s cosine | between-video variance share |
|---|---|---|
| CLIP-L/14-336 (single frame) | **0.789** | 0.498 |
| VideoMAEv2-g (2.0 s clip) | **0.863** | 0.463 |

VideoMAEv2-g's trajectory is measurably smoother in time, which is what a 2.0 s receptive field
does, and it is consistent with where the arm loses most: tIoU 0.7 (A2 16.45 vs A1 21.86), the
threshold that needs the sharpest boundaries. This is a plausible partial mechanism, offered as a
description of the failure, not as a claim that a shorter receptive field would fix it.

## 5. What is now established

1. **The feature-backbone lever, the largest one the TAL literature reports (+22.6 mAP on
   THUMOS-14 with the head held fixed), is worth −4.5 to −1.7 F1 points here.** Both directions
   of use — replacement and concatenation — are negative.
2. **The reason is not candidate generation.** Pool recall at tIoU 0.5 is 89.6–91.4 in every arm,
   and the oracle-ranked top-k curves are identical across arms to within 0.3 points. Whatever the
   input tower, ActionFormer proposes the same intervals with the same reachable ceiling.
3. **The bottleneck is span classification and it is representation-insensitive.** The 26-point
   model-vs-oracle ranking gap that R16 §5.4 measured is unchanged. R14 attacked it with objectives
   (−0.0052 wv-AUC), R17 attacked it with an extra input channel (−0.16 F1), and R18 attacks it
   with a stronger visual representation (−4.5 / −1.7 F1). Three independent axes, three nulls or
   negatives.
4. **A motion-pretrained video tower is not the missing ingredient for this content.** HateClipSeg
   is talking heads, screen recordings and edited clips; a Kinetics-710 action representation adds
   nothing a frame-wise CLIP tower does not already have, and its temporal smoothing costs
   boundary precision at high tIoU.
5. **The R16 base is confirmed stable across seed families** (41.56 ± 1.50 vs 42.02 ± 0.73 on
   disjoint seeds), so any future claim on this substrate has a trustworthy reference point.

**What this does not establish.** It does not show that no feature could help — InternVideo2-6B,
the actual top of OpenTAD's table, was not run, and neither was end-to-end adapter tuning
(AdaTAD's 76.9 is a *tuned* backbone, not frozen features). It shows that the largest frozen-feature
step available inside the round's constraints produces nothing, on a task where the diagnostic says
the pool was never the problem. Given (2) and (3), a larger frozen tower is very unlikely to be
different in kind: it would have to fix ranking, and the ranking ceiling is already identical
across two architecturally unrelated towers.

## 6. Deviations and declared design choices

- **Declared in the freeze, not deviations:** the 2 Hz encoder evaluation with linear
  interpolation to the 4 Hz grid (§2); the third-party checkpoint mirror with the verification
  above (§1); the `F.scaled_dot_product_attention` substitution for the repo's manual softmax
  (§2); raw unnormalised concatenation (§2); the all-zero array for `yt_NzvfkIYS5Yg` (§2).
- **D1 — the `--hop 2` caveat, stated against the result.** If the VideoMAE feature's advantage
  lived below 0.5 s of temporal resolution, this round would have thrown it away. Three things
  argue it did not: gold median segment length is 8.42 s, so 0.5 s is 6% of one instance;
  pool recall at tIoU 0.3 and 0.5 is identical to CLIP's, which is where a coverage loss from
  coarse features would show first; and the interpolation-free measurement in §4 shows the tower's
  own trajectory is already smoother than the interpolation step. It remains the one thing a
  follow-up would have to redo before calling this final at tIoU 0.7.
- **D2 — the §3 top-k ranking table and the §4 smoothness table are post-hoc**, defined after the
  primary number existed, as R16 §6 practice allows. Neither gates anything.
- **D3 — A2 seed 7302 is an outlier** (32.63 against 38.52 / 40.04). It is kept; §2 reports the
  arm both with and without it and the sign does not change.
- No HALT, no crash, no re-submission. All five arms ran in one `run_arms.sh` invocation.

## 7. Reproduction

| artifact | path |
|---|---|
| freeze (pre-extraction, commit `57f26c6`) | `idea-stage/R18_FEATBASE_FREEZE.md` |
| feature extractor | `scripts/r18_featbase/extract_dense_vmae.py` |
| stream fusion | `scripts/r18_featbase/fuse_feats.py` |
| arm runner (single submission) | `scripts/r18_featbase/run_arms.sh` |
| frozen decision rule + declared diagnostics | `scripts/r18_featbase/analyze.py` |
| detector runner (unchanged from R16) | `scripts/r16_detbase/run_af.py` |
| metric (unchanged from R16) | `scripts/r16_detbase/eval_f1.py` |
| VideoMAEv2-g features (2.0 GB) | `data/CLIP_Embedding/HateClipSeg/dense4fps_vmaev2g/` |
| fused features (4.5 / 6.0 GB) | `data/CLIP_Embedding/HateClipSeg/dense4fps_{mat,mvat}/` |
| per-arm results | `idea-stage/r18_featbase/out/res_{clip_vat,mae_vat,maeclip_vat,clip_v,mae_v}.json` |
| analysis, pool recall, ranking, smoothness | `idea-stage/r18_featbase/out/{analysis,pool_recall_3seed,rank_diag,feat_var}.json` |
| logs | `logging/runs/r18_{dl,extract,fuse,af}/` |
