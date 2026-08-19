# REPRO_CAMPAIGN_RESULTS — label-free frame-level baseline reproduction

Protocol: `idea-stage/REPRO_CAMPAIGN_FREEZE.md` (frozen 74b9d87, deviations D1–D2 in its §12).
Assets: `idea-stage/repro_campaign/PHASE_A_STATUS.md`. Table schema is freeze §14.
Machine: single RTX 5090, conda `HateVideo`, torch 2.7.1+cu128. Zero paid API spend.

**This is a baseline table, not a candidate trial** (freeze §0). No row here receives a GO/KILL
verdict and no decision rule in this file selects a winner.

Batch 1 (2026-08-19): **Wave 0 item 1 — ZS-CLIP**, all four datasets.

---

## Method as run — ZS-CLIP (baseline row defined inside LAVAD, CVPR 2024)

For frame `f` and a prompt pair `(p_normal, p_hateful)`:

```
s_n = cos(img(f), txt(p_normal))
s_h = cos(img(f), txt(p_hateful))
score(f) = softmax_h([s_n, s_h] * logit_scale) = sigmoid(logit_scale * (s_h - s_n)),  logit_scale = 100
```

**Implementation choice, recorded (campaign brief asked for it).** The official LAVAD repo
(`third_party/lavad`) ships **no ZS-CLIP script** — ZS-CLIP exists only as a baseline row in the
LAVAD paper — so there is no "exact official formula" to copy; the two candidate conventions are
softmax over the two prompts and the raw difference `s_h - s_n`. They are related by a strictly
increasing map, so **frame ROC-AUC and PR-AUC are identical under either**. The choice is therefore
not load-bearing for any number in this table. We store the difference and report the softmax
probability.

**Image embeddings.** The Phase A cache stores `CLIPVisionModel.pooler_output` (1024-d,
pre-projection). `CLIPModel.get_image_features` is exactly `visual_projection(pooler_output)`, so
applying the frozen 1024→768 projection to the cache reconstructs the joint-space embedding with no
re-decoding. Verified against a live decode of 4 frames of `bit_0EHvMSiEHVoc`:
**mean cosine 0.999994** (residual is the fp16 extraction vs fp32 live encode).
Model `openai/clip-vit-large-patch14-336`, revision `ce19dc9`.

**Prompt sets.** The `main` pair is the one the campaign brief fixed. The other three are the
prompt-sensitivity appendix and are **all reported**; none was selected on any metric (freeze §10
red line 1). No prompt is tuned on any split.

| set | negative prompt | positive prompt |
|---|---|---|
| `main` | a normal video frame | a hateful video frame |
| `sens_hs` | a normal video frame | a video frame containing hate speech |
| `sens_off` | a normal video frame | a video frame with offensive content |
| `sens_lavad` | a normal video frame | an anomalous video frame *(LAVAD's own anomaly wording)* |

HateClipSeg additionally gets one pair per released toxic class
(`hateful / insulting / sexual / violence / harm`, each against `a normal video frame`), aggregated
to any-toxic by **max over the five class scores** (`hcs6_anytoxic_max`). `hcs_hateful` is the same
prompt string as `main`, so those two rows coincide by construction.

**`+text (ours)` variant** — our declared adaptation, freeze §8. ZS-CLIP's native window is a single
frame, which would carry roughly one ASR word, so the injected text window is the **OCR cache's own
K=30 grid**: window `k` covers `[k·D/30, (k+1)·D/30)`; the string is the ASR words whose word-level
timestamps overlap that window, in time order, then the OCR window text, joined by `" | "`. The
string is encoded by the **same CLIP text tower** and scored against the same prompt pair, giving a
text score in `[0,1]` per window. Fusion is a fixed, untuned **equal-weight average** of the visual
and text probabilities; a frame whose window has no text keeps the visual score alone. No text is
transcribed or OCR'd per method — the frozen caches only (ASR K=4 whisper-large-v3, OCR K=30
PaddleOCR). MHC-ZH text is injected in Chinese as cached. Window text coverage: HateMM 95.7%,
MHC-EN 96.6%, MHC-ZH 97.9%, HateClipSeg 97.3%.

**Controls** (freeze §3) are recomputed inside the same evaluator on exactly the same frame pool as
the method rows, so they differ in the third-to-fourth decimal from the Phase A table, which pooled
the frames of videos this method has no features for. Frame counts here are `min(T_gt, T_feat)` per
video (freeze §1) and exclude the two HateMM videos with no video stream (deviation D2).

**Stratification** (freeze §14): `single_span` = videos with ≤1 gold span, `multi_span` = videos
with ≥2 gold spans **plus all zero-span videos** (which supply the negatives; without them the pool
has no negative frames at all). The MHC-ZH test `multi_span` pool contains no span-carrying video at
all — it is single-class and its ROC/AP are undefined, reported `n/a` rather than dropped.

`F1@tIoU` is `n/a` for every row: ZS-CLIP emits a score curve, not intervals, and freeze §2 forbids
inventing a threshold for it.

Reproduce: `python scripts/repro_campaign/zs_clip.py --stage all`
(stages `prompts / visual / text / eval`, each idempotent; raw per-row JSON in
`idea-stage/repro_zs_clip/results.json`, logs in `logging/runs/repro_zsclip/`).

---
## A. Headline rows — main prompt pair, test split

| method | wave | dataset | split | supervision | variant | prompt_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GOLD_BROADCAST | — | HateMM | test | control | control | n/a | video | 0.8857 | 0.5831 | n/a | n/a | n/a | 1.0000 | 116908 | 0.2422 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| RANDOM_UNIFORM | — | HateMM | test | control | control | n/a | 4 fps | 0.5002 ± 0.0019 | 0.2424 ± 0.0014 | n/a | n/a | n/a | 0.0000 | 116908 | 0.2422 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateMM | test | label-free | base | main | 4 fps | 0.5368 | 0.2775 | n/a | n/a | n/a | 0.1031 | 116908 | 0.2422 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateMM | test | label-free | +text (ours) | main | 4 fps | 0.5555 | 0.2789 | n/a | n/a | n/a | 0.1071 | 116908 | 0.2422 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.957 |
| GOLD_BROADCAST | — | MHC-EN | test | control | control | n/a | video | 0.9427 | 0.7664 | n/a | n/a | n/a | 1.0000 | 22336 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| RANDOM_UNIFORM | — | MHC-EN | test | control | control | n/a | 4 fps | 0.5002 ± 0.0034 | 0.2736 ± 0.0026 | n/a | n/a | n/a | 0.0000 | 22336 | 0.2734 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-EN | test | label-free | base | main | 4 fps | 0.5013 | 0.2678 | n/a | n/a | n/a | -0.0118 | 22336 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-EN | test | label-free | +text (ours) | main | 4 fps | 0.4982 | 0.2622 | n/a | n/a | n/a | -0.0231 | 22336 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.964 |
| GOLD_BROADCAST | — | MHC-ZH | test | control | control | n/a | video | 0.9842 | 0.9191 | n/a | n/a | n/a | 1.0000 | 18195 | 0.2649 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| RANDOM_UNIFORM | — | MHC-ZH | test | control | control | n/a | 4 fps | 0.4986 ± 0.0050 | 0.2646 ± 0.0035 | n/a | n/a | n/a | 0.0000 | 18195 | 0.2649 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-ZH | test | label-free | base | main | 4 fps | 0.6075 | 0.3406 | n/a | n/a | n/a | 0.1162 | 18195 | 0.2649 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-ZH | test | label-free | +text (ours) | main | 4 fps | 0.5908 | 0.3268 | n/a | n/a | n/a | 0.0949 | 18195 | 0.2649 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.990 |
| GOLD_BROADCAST | — | HateClipSeg | test | control | control | n/a | video | 0.6260 | 0.5433 | n/a | n/a | n/a | 1.0000 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| RANDOM_UNIFORM | — | HateClipSeg | test | control | control | n/a | 4 fps | 0.5004 ± 0.0019 | 0.4711 ± 0.0013 | n/a | n/a | n/a | 0.0000 | 114021 | 0.4709 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | base | main | 4 fps | 0.4990 | 0.4612 | n/a | n/a | n/a | -0.1364 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | +text (ours) | main | 4 fps | 0.5258 | 0.5003 | n/a | n/a | n/a | 0.4044 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.970 |

## B. Full corpus — main prompt pair

| method | wave | dataset | split | supervision | variant | prompt_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GOLD_BROADCAST | — | HateMM | full | control | control | n/a | video | 0.9033 | 0.6742 | n/a | n/a | n/a | 1.0000 | 623788 | 0.2858 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| RANDOM_UNIFORM | — | HateMM | full | control | control | n/a | 4 fps | 0.5002 ± 0.0007 | 0.2859 ± 0.0006 | n/a | n/a | n/a | 0.0000 | 623788 | 0.2858 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateMM | full | label-free | base | main | 4 fps | 0.5406 | 0.3135 | n/a | n/a | n/a | 0.0709 | 623788 | 0.2858 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateMM | full | label-free | +text (ours) | main | 4 fps | 0.5474 | 0.3147 | n/a | n/a | n/a | 0.0740 | 623788 | 0.2858 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.957 |
| GOLD_BROADCAST | — | MHC-EN | full | control | control | n/a | video | 0.9536 | 0.7767 | n/a | n/a | n/a | 1.0000 | 110728 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| RANDOM_UNIFORM | — | MHC-EN | full | control | control | n/a | 4 fps | 0.5002 ± 0.0024 | 0.2445 ± 0.0015 | n/a | n/a | n/a | 0.0000 | 110728 | 0.2441 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-EN | full | label-free | base | main | 4 fps | 0.5474 | 0.2767 | n/a | n/a | n/a | 0.0605 | 110728 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-EN | full | label-free | +text (ours) | main | 4 fps | 0.5594 | 0.2854 | n/a | n/a | n/a | 0.0768 | 110728 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.966 |
| GOLD_BROADCAST | — | MHC-ZH | full | control | control | n/a | video | 0.9709 | 0.8537 | n/a | n/a | n/a | 1.0000 | 102130 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| RANDOM_UNIFORM | — | MHC-ZH | full | control | control | n/a | 4 fps | 0.5005 ± 0.0023 | 0.2543 ± 0.0014 | n/a | n/a | n/a | 0.0000 | 102130 | 0.2538 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-ZH | full | label-free | base | main | 4 fps | 0.5611 | 0.2953 | n/a | n/a | n/a | 0.0683 | 102130 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-ZH | full | label-free | +text (ours) | main | 4 fps | 0.5542 | 0.2887 | n/a | n/a | n/a | 0.0574 | 102130 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.979 |
| GOLD_BROADCAST | — | HateClipSeg | full | control | control | n/a | video | 0.6164 | 0.5296 | n/a | n/a | n/a | 1.0000 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| RANDOM_UNIFORM | — | HateClipSeg | full | control | control | n/a | 4 fps | 0.4996 ± 0.0009 | 0.4632 ± 0.0008 | n/a | n/a | n/a | 0.0000 | 375250 | 0.4635 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | base | main | 4 fps | 0.5119 | 0.4674 | n/a | n/a | n/a | 0.0634 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | +text (ours) | main | 4 fps | 0.5378 | 0.5019 | n/a | n/a | n/a | 0.5828 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.973 |

## C. Prompt-sensitivity appendix — every prompt pair, both splits

| method | wave | dataset | split | supervision | variant | prompt_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ZS-CLIP | 0 | HateMM | full | label-free | base | main | 4 fps | 0.5406 | 0.3135 | n/a | n/a | n/a | 0.0709 | 623788 | 0.2858 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateMM | full | label-free | +text (ours) | main | 4 fps | 0.5474 | 0.3147 | n/a | n/a | n/a | 0.0740 | 623788 | 0.2858 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.957 |
| ZS-CLIP | 0 | HateMM | full | label-free | base | sens_hs | 4 fps | 0.5241 | 0.3078 | n/a | n/a | n/a | 0.0563 | 623788 | 0.2858 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateMM | full | label-free | +text (ours) | sens_hs | 4 fps | 0.5296 | 0.2914 | n/a | n/a | n/a | 0.0140 | 623788 | 0.2858 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.957 |
| ZS-CLIP | 0 | HateMM | full | label-free | base | sens_off | 4 fps | 0.5780 | 0.3532 | n/a | n/a | n/a | 0.1731 | 623788 | 0.2858 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateMM | full | label-free | +text (ours) | sens_off | 4 fps | 0.6237 | 0.3748 | n/a | n/a | n/a | 0.2289 | 623788 | 0.2858 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.957 |
| ZS-CLIP | 0 | HateMM | full | label-free | base | sens_lavad | 4 fps | 0.3973 | 0.2405 | n/a | n/a | n/a | -0.1170 | 623788 | 0.2858 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateMM | full | label-free | +text (ours) | sens_lavad | 4 fps | 0.3821 | 0.2263 | n/a | n/a | n/a | -0.1536 | 623788 | 0.2858 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.957 |
| ZS-CLIP | 0 | HateMM | test | label-free | base | main | 4 fps | 0.5368 | 0.2775 | n/a | n/a | n/a | 0.1031 | 116908 | 0.2422 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateMM | test | label-free | +text (ours) | main | 4 fps | 0.5555 | 0.2789 | n/a | n/a | n/a | 0.1071 | 116908 | 0.2422 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.957 |
| ZS-CLIP | 0 | HateMM | test | label-free | base | sens_hs | 4 fps | 0.5428 | 0.2948 | n/a | n/a | n/a | 0.1537 | 116908 | 0.2422 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateMM | test | label-free | +text (ours) | sens_hs | 4 fps | 0.5710 | 0.2826 | n/a | n/a | n/a | 0.1180 | 116908 | 0.2422 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.957 |
| ZS-CLIP | 0 | HateMM | test | label-free | base | sens_off | 4 fps | 0.5573 | 0.2888 | n/a | n/a | n/a | 0.1362 | 116908 | 0.2422 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateMM | test | label-free | +text (ours) | sens_off | 4 fps | 0.6357 | 0.3283 | n/a | n/a | n/a | 0.2522 | 116908 | 0.2422 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.957 |
| ZS-CLIP | 0 | HateMM | test | label-free | base | sens_lavad | 4 fps | 0.4245 | 0.2188 | n/a | n/a | n/a | -0.0693 | 116908 | 0.2422 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateMM | test | label-free | +text (ours) | sens_lavad | 4 fps | 0.3873 | 0.1899 | n/a | n/a | n/a | -0.1541 | 116908 | 0.2422 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.957 |
| ZS-CLIP | 0 | MHC-EN | full | label-free | base | main | 4 fps | 0.5474 | 0.2767 | n/a | n/a | n/a | 0.0605 | 110728 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-EN | full | label-free | +text (ours) | main | 4 fps | 0.5594 | 0.2854 | n/a | n/a | n/a | 0.0768 | 110728 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.966 |
| ZS-CLIP | 0 | MHC-EN | full | label-free | base | sens_hs | 4 fps | 0.5654 | 0.2844 | n/a | n/a | n/a | 0.0749 | 110728 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-EN | full | label-free | +text (ours) | sens_hs | 4 fps | 0.5776 | 0.3137 | n/a | n/a | n/a | 0.1300 | 110728 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.966 |
| ZS-CLIP | 0 | MHC-EN | full | label-free | base | sens_off | 4 fps | 0.5588 | 0.2841 | n/a | n/a | n/a | 0.0745 | 110728 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-EN | full | label-free | +text (ours) | sens_off | 4 fps | 0.5944 | 0.3110 | n/a | n/a | n/a | 0.1249 | 110728 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.966 |
| ZS-CLIP | 0 | MHC-EN | full | label-free | base | sens_lavad | 4 fps | 0.4742 | 0.2260 | n/a | n/a | n/a | -0.0347 | 110728 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-EN | full | label-free | +text (ours) | sens_lavad | 4 fps | 0.4580 | 0.2249 | n/a | n/a | n/a | -0.0367 | 110728 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.966 |
| ZS-CLIP | 0 | MHC-EN | test | label-free | base | main | 4 fps | 0.5013 | 0.2678 | n/a | n/a | n/a | -0.0118 | 22336 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-EN | test | label-free | +text (ours) | main | 4 fps | 0.4982 | 0.2622 | n/a | n/a | n/a | -0.0231 | 22336 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.964 |
| ZS-CLIP | 0 | MHC-EN | test | label-free | base | sens_hs | 4 fps | 0.5149 | 0.2793 | n/a | n/a | n/a | 0.0116 | 22336 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-EN | test | label-free | +text (ours) | sens_hs | 4 fps | 0.5257 | 0.2814 | n/a | n/a | n/a | 0.0158 | 22336 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.964 |
| ZS-CLIP | 0 | MHC-EN | test | label-free | base | sens_off | 4 fps | 0.5116 | 0.2755 | n/a | n/a | n/a | 0.0039 | 22336 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-EN | test | label-free | +text (ours) | sens_off | 4 fps | 0.5733 | 0.3145 | n/a | n/a | n/a | 0.0829 | 22336 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.964 |
| ZS-CLIP | 0 | MHC-EN | test | label-free | base | sens_lavad | 4 fps | 0.4319 | 0.2318 | n/a | n/a | n/a | -0.0847 | 22336 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-EN | test | label-free | +text (ours) | sens_lavad | 4 fps | 0.4067 | 0.2198 | n/a | n/a | n/a | -0.1092 | 22336 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.964 |
| ZS-CLIP | 0 | MHC-ZH | full | label-free | base | main | 4 fps | 0.5611 | 0.2953 | n/a | n/a | n/a | 0.0683 | 102130 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-ZH | full | label-free | +text (ours) | main | 4 fps | 0.5542 | 0.2887 | n/a | n/a | n/a | 0.0574 | 102130 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.979 |
| ZS-CLIP | 0 | MHC-ZH | full | label-free | base | sens_hs | 4 fps | 0.4919 | 0.2467 | n/a | n/a | n/a | -0.0126 | 102130 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-ZH | full | label-free | +text (ours) | sens_hs | 4 fps | 0.4804 | 0.2395 | n/a | n/a | n/a | -0.0247 | 102130 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.979 |
| ZS-CLIP | 0 | MHC-ZH | full | label-free | base | sens_off | 4 fps | 0.5614 | 0.2982 | n/a | n/a | n/a | 0.0732 | 102130 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-ZH | full | label-free | +text (ours) | sens_off | 4 fps | 0.5104 | 0.2622 | n/a | n/a | n/a | 0.0132 | 102130 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.979 |
| ZS-CLIP | 0 | MHC-ZH | full | label-free | base | sens_lavad | 4 fps | 0.5262 | 0.2711 | n/a | n/a | n/a | 0.0281 | 102130 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-ZH | full | label-free | +text (ours) | sens_lavad | 4 fps | 0.5214 | 0.2684 | n/a | n/a | n/a | 0.0236 | 102130 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.979 |
| ZS-CLIP | 0 | MHC-ZH | test | label-free | base | main | 4 fps | 0.6075 | 0.3406 | n/a | n/a | n/a | 0.1162 | 18195 | 0.2649 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-ZH | test | label-free | +text (ours) | main | 4 fps | 0.5908 | 0.3268 | n/a | n/a | n/a | 0.0949 | 18195 | 0.2649 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.990 |
| ZS-CLIP | 0 | MHC-ZH | test | label-free | base | sens_hs | 4 fps | 0.5223 | 0.2845 | n/a | n/a | n/a | 0.0303 | 18195 | 0.2649 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-ZH | test | label-free | +text (ours) | sens_hs | 4 fps | 0.5192 | 0.2767 | n/a | n/a | n/a | 0.0185 | 18195 | 0.2649 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.990 |
| ZS-CLIP | 0 | MHC-ZH | test | label-free | base | sens_off | 4 fps | 0.6149 | 0.3563 | n/a | n/a | n/a | 0.1401 | 18195 | 0.2649 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-ZH | test | label-free | +text (ours) | sens_off | 4 fps | 0.5848 | 0.3183 | n/a | n/a | n/a | 0.0821 | 18195 | 0.2649 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.990 |
| ZS-CLIP | 0 | MHC-ZH | test | label-free | base | sens_lavad | 4 fps | 0.5597 | 0.3079 | n/a | n/a | n/a | 0.0661 | 18195 | 0.2649 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | MHC-ZH | test | label-free | +text (ours) | sens_lavad | 4 fps | 0.5478 | 0.2964 | n/a | n/a | n/a | 0.0486 | 18195 | 0.2649 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.990 |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | base | main | 4 fps | 0.5119 | 0.4674 | n/a | n/a | n/a | 0.0634 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | +text (ours) | main | 4 fps | 0.5378 | 0.5019 | n/a | n/a | n/a | 0.5828 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.973 |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | base | sens_hs | 4 fps | 0.5343 | 0.4920 | n/a | n/a | n/a | 0.4331 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | +text (ours) | sens_hs | 4 fps | 0.5462 | 0.4964 | n/a | n/a | n/a | 0.4992 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.973 |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | base | sens_off | 4 fps | 0.5640 | 0.5291 | n/a | n/a | n/a | 0.9911 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | +text (ours) | sens_off | 4 fps | 0.5868 | 0.5425 | n/a | n/a | n/a | 1.1933 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.973 |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | base | sens_lavad | 4 fps | 0.4603 | 0.4304 | n/a | n/a | n/a | -0.4951 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | +text (ours) | sens_lavad | 4 fps | 0.4436 | 0.4222 | n/a | n/a | n/a | -0.6183 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.973 |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | base | main | 4 fps | 0.4990 | 0.4612 | n/a | n/a | n/a | -0.1364 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | +text (ours) | main | 4 fps | 0.5258 | 0.5003 | n/a | n/a | n/a | 0.4044 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.970 |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | base | sens_hs | 4 fps | 0.5094 | 0.4915 | n/a | n/a | n/a | 0.2826 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | +text (ours) | sens_hs | 4 fps | 0.5259 | 0.4938 | n/a | n/a | n/a | 0.3152 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.970 |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | base | sens_off | 4 fps | 0.5512 | 0.5367 | n/a | n/a | n/a | 0.9084 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | +text (ours) | sens_off | 4 fps | 0.5798 | 0.5485 | n/a | n/a | n/a | 1.0719 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.970 |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | base | sens_lavad | 4 fps | 0.4620 | 0.4464 | n/a | n/a | n/a | -0.3414 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | +text (ours) | sens_lavad | 4 fps | 0.4303 | 0.4269 | n/a | n/a | n/a | -0.6110 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.970 |

## D. HateClipSeg 6-class appendix (per-class prompt pair, any-toxic = max over the 5 toxic classes)

| method | wave | dataset | split | supervision | variant | prompt_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ZS-CLIP | 0 | HateClipSeg | full | label-free | base | hcs_hateful | 4 fps | 0.5119 | 0.4674 | n/a | n/a | n/a | 0.0634 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | +text (ours) | hcs_hateful | 4 fps | 0.5378 | 0.5019 | n/a | n/a | n/a | 0.5828 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.973 |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | base | hcs_insulting | 4 fps | 0.5280 | 0.4969 | n/a | n/a | n/a | 0.5069 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | +text (ours) | hcs_insulting | 4 fps | 0.5338 | 0.4983 | n/a | n/a | n/a | 0.5284 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.973 |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | base | hcs_sexual | 4 fps | 0.4869 | 0.4617 | n/a | n/a | n/a | -0.0239 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | +text (ours) | hcs_sexual | 4 fps | 0.4874 | 0.4639 | n/a | n/a | n/a | 0.0102 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.973 |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | base | hcs_violence | 4 fps | 0.4982 | 0.4589 | n/a | n/a | n/a | -0.0647 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | +text (ours) | hcs_violence | 4 fps | 0.5109 | 0.4789 | n/a | n/a | n/a | 0.2366 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.973 |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | base | hcs_harm | 4 fps | 0.4913 | 0.4547 | n/a | n/a | n/a | -0.1282 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | +text (ours) | hcs_harm | 4 fps | 0.4861 | 0.4580 | n/a | n/a | n/a | -0.0795 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.973 |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | base | hcs6_anytoxic_max | 4 fps | 0.5217 | 0.4755 | n/a | n/a | n/a | 0.1854 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | full | label-free | +text (ours) | hcs6_anytoxic_max | 4 fps | 0.5318 | 0.4882 | n/a | n/a | n/a | 0.3752 | 375250 | 0.4635 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.973 |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | base | hcs_hateful | 4 fps | 0.4990 | 0.4612 | n/a | n/a | n/a | -0.1364 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | +text (ours) | hcs_hateful | 4 fps | 0.5258 | 0.5003 | n/a | n/a | n/a | 0.4044 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.970 |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | base | hcs_insulting | 4 fps | 0.5396 | 0.5359 | n/a | n/a | n/a | 0.8979 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | +text (ours) | hcs_insulting | 4 fps | 0.5542 | 0.5332 | n/a | n/a | n/a | 0.8597 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.970 |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | base | hcs_sexual | 4 fps | 0.5006 | 0.4872 | n/a | n/a | n/a | 0.2234 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | +text (ours) | hcs_sexual | 4 fps | 0.5061 | 0.4886 | n/a | n/a | n/a | 0.2421 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.970 |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | base | hcs_violence | 4 fps | 0.5088 | 0.4695 | n/a | n/a | n/a | -0.0217 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | +text (ours) | hcs_violence | 4 fps | 0.5098 | 0.5027 | n/a | n/a | n/a | 0.4382 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.970 |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | base | hcs_harm | 4 fps | 0.5091 | 0.4640 | n/a | n/a | n/a | -0.0975 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | +text (ours) | hcs_harm | 4 fps | 0.4882 | 0.4643 | n/a | n/a | n/a | -0.0936 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.970 |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | base | hcs6_anytoxic_max | 4 fps | 0.5449 | 0.4833 | n/a | n/a | n/a | 0.1691 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ |  |
| ZS-CLIP | 0 | HateClipSeg | test | label-free | +text (ours) | hcs6_anytoxic_max | 4 fps | 0.5519 | 0.5124 | n/a | n/a | n/a | 0.5724 | 114021 | 0.4709 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | text_cov=0.970 |

## E. Stratified sub-tables — single-span vs multi-span (main prompt pair)

| method | wave | dataset | split | supervision | variant | prompt_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GOLD_BROADCAST | — | HateMM | full | control | control | n/a | video | 0.9547 | 0.7880 | n/a | n/a | n/a | 1.0000 | 528406 | 0.2520 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  |
| RANDOM_UNIFORM | — | HateMM | full | control | control | n/a | 4 fps | 0.5000 ± 0.0010 | 0.2520 ± 0.0006 | n/a | n/a | n/a | 0.0000 | 528406 | 0.2520 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  |
| ZS-CLIP | 0 | HateMM | full | label-free | base | main | 4 fps | 0.5357 | 0.2755 | n/a | n/a | n/a | 0.0439 | 528406 | 0.2520 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span |
| ZS-CLIP | 0 | HateMM | full | label-free | +text (ours) | main | 4 fps | 0.5466 | 0.2789 | n/a | n/a | n/a | 0.0503 | 528406 | 0.2520 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  text_cov=0.955 |
| GOLD_BROADCAST | — | HateMM | full | control | control | n/a | video | 0.9386 | 0.4727 | n/a | n/a | n/a | 1.0000 | 454771 | 0.0991 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span  |
| RANDOM_UNIFORM | — | HateMM | full | control | control | n/a | 4 fps | 0.5000 ± 0.0013 | 0.0991 ± 0.0004 | n/a | n/a | n/a | 0.0000 | 454771 | 0.0991 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span  |
| ZS-CLIP | 0 | HateMM | full | label-free | base | main | 4 fps | 0.5625 | 0.1228 | n/a | n/a | n/a | 0.0635 | 454771 | 0.0991 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span |
| ZS-CLIP | 0 | HateMM | full | label-free | +text (ours) | main | 4 fps | 0.5580 | 0.1178 | n/a | n/a | n/a | 0.0503 | 454771 | 0.0991 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span  text_cov=0.953 |
| GOLD_BROADCAST | — | HateMM | test | control | control | n/a | video | 0.9585 | 0.7517 | n/a | n/a | n/a | 1.0000 | 93251 | 0.2008 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  |
| RANDOM_UNIFORM | — | HateMM | test | control | control | n/a | 4 fps | 0.5006 ± 0.0026 | 0.2013 ± 0.0013 | n/a | n/a | n/a | 0.0000 | 93251 | 0.2008 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  |
| ZS-CLIP | 0 | HateMM | test | label-free | base | main | 4 fps | 0.5202 | 0.2043 | n/a | n/a | n/a | 0.0056 | 93251 | 0.2008 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span |
| ZS-CLIP | 0 | HateMM | test | label-free | +text (ours) | main | 4 fps | 0.5557 | 0.2255 | n/a | n/a | n/a | 0.0440 | 93251 | 0.2008 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  text_cov=0.953 |
| GOLD_BROADCAST | — | HateMM | test | control | control | n/a | video | 0.9147 | 0.4055 | n/a | n/a | n/a | 1.0000 | 91996 | 0.1043 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span  |
| RANDOM_UNIFORM | — | HateMM | test | control | control | n/a | 4 fps | 0.5003 ± 0.0035 | 0.1045 ± 0.0010 | n/a | n/a | n/a | 0.0000 | 91996 | 0.1043 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span  |
| ZS-CLIP | 0 | HateMM | test | label-free | base | main | 4 fps | 0.5695 | 0.2139 | n/a | n/a | n/a | 0.3634 | 91996 | 0.1043 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span |
| ZS-CLIP | 0 | HateMM | test | label-free | +text (ours) | main | 4 fps | 0.5555 | 0.1491 | n/a | n/a | n/a | 0.1482 | 91996 | 0.1043 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span  text_cov=0.953 |
| GOLD_BROADCAST | — | MHC-EN | full | control | control | n/a | video | 0.9594 | 0.7934 | n/a | n/a | n/a | 1.0000 | 108453 | 0.2379 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  |
| RANDOM_UNIFORM | — | MHC-EN | full | control | control | n/a | 4 fps | 0.4998 ± 0.0019 | 0.2380 ± 0.0012 | n/a | n/a | n/a | 0.0000 | 108453 | 0.2379 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  |
| ZS-CLIP | 0 | MHC-EN | full | label-free | base | main | 4 fps | 0.5409 | 0.2646 | n/a | n/a | n/a | 0.0479 | 108453 | 0.2379 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span |
| ZS-CLIP | 0 | MHC-EN | full | label-free | +text (ours) | main | 4 fps | 0.5535 | 0.2739 | n/a | n/a | n/a | 0.0646 | 108453 | 0.2379 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  text_cov=0.965 |
| GOLD_BROADCAST | — | MHC-EN | full | control | control | n/a | video | 0.9932 | 0.5380 | n/a | n/a | n/a | 1.0000 | 78205 | 0.0157 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span  |
| RANDOM_UNIFORM | — | MHC-EN | full | control | control | n/a | 4 fps | 0.5031 ± 0.0071 | 0.0160 ± 0.0003 | n/a | n/a | n/a | 0.0000 | 78205 | 0.0157 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span  |
| ZS-CLIP | 0 | MHC-EN | full | label-free | base | main | 4 fps | 0.7017 | 0.0356 | n/a | n/a | n/a | 0.0375 | 78205 | 0.0157 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span |
| ZS-CLIP | 0 | MHC-EN | full | label-free | +text (ours) | main | 4 fps | 0.6912 | 0.0329 | n/a | n/a | n/a | 0.0323 | 78205 | 0.0157 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span  text_cov=0.967 |
| GOLD_BROADCAST | — | MHC-EN | test | control | control | n/a | video | 0.9498 | 0.7801 | n/a | n/a | n/a | 1.0000 | 21668 | 0.2628 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  |
| RANDOM_UNIFORM | — | MHC-EN | test | control | control | n/a | 4 fps | 0.5003 ± 0.0047 | 0.2630 ± 0.0032 | n/a | n/a | n/a | 0.0000 | 21668 | 0.2628 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  |
| ZS-CLIP | 0 | MHC-EN | test | label-free | base | main | 4 fps | 0.4901 | 0.2472 | n/a | n/a | n/a | -0.0305 | 21668 | 0.2628 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span |
| ZS-CLIP | 0 | MHC-EN | test | label-free | +text (ours) | main | 4 fps | 0.4883 | 0.2453 | n/a | n/a | n/a | -0.0341 | 21668 | 0.2628 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  text_cov=0.963 |
| GOLD_BROADCAST | — | MHC-EN | test | control | control | n/a | video | 0.9912 | 0.6168 | n/a | n/a | n/a | 1.0000 | 15036 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span  |
| RANDOM_UNIFORM | — | MHC-EN | test | control | control | n/a | 4 fps | 0.4976 ± 0.0142 | 0.0282 ± 0.0016 | n/a | n/a | n/a | 0.0000 | 15036 | 0.0274 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span  |
| ZS-CLIP | 0 | MHC-EN | test | label-free | base | main | 4 fps | 0.6357 | 0.0628 | n/a | n/a | n/a | 0.0587 | 15036 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span |
| ZS-CLIP | 0 | MHC-EN | test | label-free | +text (ours) | main | 4 fps | 0.6124 | 0.0444 | n/a | n/a | n/a | 0.0274 | 15036 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span  text_cov=0.962 |
| GOLD_BROADCAST | — | MHC-ZH | full | control | control | n/a | video | 0.9733 | 0.8644 | n/a | n/a | n/a | 1.0000 | 101597 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  |
| RANDOM_UNIFORM | — | MHC-ZH | full | control | control | n/a | 4 fps | 0.4994 ± 0.0023 | 0.2535 ± 0.0016 | n/a | n/a | n/a | 0.0000 | 101597 | 0.2538 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  |
| ZS-CLIP | 0 | MHC-ZH | full | label-free | base | main | 4 fps | 0.5609 | 0.2951 | n/a | n/a | n/a | 0.0682 | 101597 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span |
| ZS-CLIP | 0 | MHC-ZH | full | label-free | +text (ours) | main | 4 fps | 0.5540 | 0.2886 | n/a | n/a | n/a | 0.0575 | 101597 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  text_cov=0.979 |
| GOLD_BROADCAST | — | MHC-ZH | full | control | control | n/a | video | 0.9959 | 0.1868 | n/a | n/a | n/a | 1.0000 | 72497 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span  |
| RANDOM_UNIFORM | — | MHC-ZH | full | control | control | n/a | 4 fps | 0.5047 ± 0.0157 | 0.0022 ± 0.0006 | n/a | n/a | n/a | 0.0000 | 72497 | 0.0019 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span  |
| ZS-CLIP | 0 | MHC-ZH | full | label-free | base | main | 4 fps | 0.5951 | 0.0023 | n/a | n/a | n/a | 0.0006 | 72497 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span |
| ZS-CLIP | 0 | MHC-ZH | full | label-free | +text (ours) | main | 4 fps | 0.5696 | 0.0021 | n/a | n/a | n/a | -0.0005 | 72497 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span  text_cov=0.977 |
| GOLD_BROADCAST | — | MHC-ZH | test | control | control | n/a | video | 0.9842 | 0.9191 | n/a | n/a | n/a | 1.0000 | 18195 | 0.2649 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  |
| RANDOM_UNIFORM | — | MHC-ZH | test | control | control | n/a | 4 fps | 0.4986 ± 0.0050 | 0.2646 ± 0.0035 | n/a | n/a | n/a | 0.0000 | 18195 | 0.2649 | 20 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  |
| ZS-CLIP | 0 | MHC-ZH | test | label-free | base | main | 4 fps | 0.6075 | 0.3406 | n/a | n/a | n/a | 0.1162 | 18195 | 0.2649 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span |
| ZS-CLIP | 0 | MHC-ZH | test | label-free | +text (ours) | main | 4 fps | 0.5908 | 0.3268 | n/a | n/a | n/a | 0.0949 | 18195 | 0.2649 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=single_span  text_cov=0.990 |
| ZS-CLIP | 0 | MHC-ZH | test | label-free | base | all | 4 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12952 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_zs_clip/ | stratum=multi_span single-class pool, metrics undefined |

---

## F. Alignment against LELA's published ZS-CLIP row

LELA (arXiv 2602.09637) reports ZS-CLIP frame ROC-AUC **0.5367** on HateMM and **0.5449** on
MultiHateClip, without stating the split, the language of its MultiHateClip column, the prompt
wording, or the CLIP backbone. The campaign brief set the agreement band at ±0.05.

| target | LELA | ours, full corpus | \|diff\| | ours, test split | \|diff\| |
|---|---|---|---|---|---|
| HateMM ROC | 0.5367 | **0.5406** | 0.0039 | **0.5368** | **0.0001** |
| MultiHateClip ROC vs MHC-EN | 0.5449 | **0.5474** | 0.0025 | 0.5013 | 0.0436 |
| MultiHateClip ROC vs MHC-ZH | 0.5449 | 0.5611 | 0.0162 | 0.6075 | 0.0626 |

**Conclusion: the transplant agrees.** Both full-corpus numbers land inside ±0.004 of LELA's, an
order of magnitude tighter than the ±0.05 band, on both datasets at once — with a prompt pair we
fixed before seeing any number and a backbone/protocol we could not read off their paper. The
MHC-EN full-corpus match (0.0025) is the closer of the two MultiHateClip candidates, which is weak
evidence that LELA's "MultiHateClip" column is the English half or a pool dominated by it, but the
ZH number is also inside the band and this is not decisive. The only comparison outside ±0.05 is
MHC-ZH test (0.0626), a 149-video / 18k-frame pool where the method is 0.05–0.06 above its own
full-corpus value; ordinary small-pool variation, not a transplant failure. Freeze §7's formal
±0.03 transplant gate is defined only for LAVAD and URF-HVAA and is `n/a` for ZS-CLIP; this section
is the informal equivalent.

## G. What the numbers say

1. **ZS-CLIP is a floor, as expected, and it is a floor for a specific reason.** On the main prompt
   pair the test-split ROC is 0.5368 / 0.5013 / 0.6075 / 0.4990 (HateMM / MHC-EN / MHC-ZH /
   HateClipSeg) against a random floor of 0.500 and a gold-broadcast ceiling of 0.886 / 0.943 /
   0.984 / 0.626. Oracle-normalised AP is 0.10 / −0.01 / 0.12 / −0.14: the method recovers roughly
   a tenth of the gap between chance and a *zero-localisation* video-level oracle, and on two of the
   four datasets it recovers none of it.
2. **The prompt matters more than the method.** Across the four prompt pairs, test ROC on HateMM
   spans 0.4245–0.5573 and on HateClipSeg 0.4620–0.5512 — a range wider than the method's entire
   margin over chance. LAVAD's own `anomalous` wording is **below chance on three of the four
   datasets** (test ROC 0.4245 HateMM / 0.4319 MHC-EN / 0.5597 MHC-ZH / 0.4620 HateClipSeg, and
   below chance on MHC-EN and HateClipSeg full corpus too), which is the concrete form of the
   domain-transfer problem: hateful content is not visually anomalous. Nothing here is selected on a metric — the
   main row is the pre-fixed one, and it is not the best one.
3. **Injecting the ASR/OCR text helps where the frames are weakest, and hurts where the visual
   prompt already works.** `+text` on the main pair moves test ROC +0.019 on HateMM, −0.003 on
   MHC-EN, −0.017 on MHC-ZH, +0.027 on HateClipSeg. The largest single effect anywhere in the table
   is `sens_off` + text on HateMM test: 0.5573 → 0.6357 ROC, AP_norm 0.14 → 0.25. Read carefully:
   an equal-weight, untuned average of a text channel that CLIP's 77-token text tower was never
   built to score is already competitive with the visual channel it is averaged into.
4. **The stratified split reproduces the coverage degeneracy the landscape document predicted.**
   On MHC-EN the multi-span stratum has a 1.6% positive base rate and a broadcast ceiling of AP
   0.538 at ROC 0.993, and ZS-CLIP reaches ROC 0.702 there while sitting at 0.541 on the single-span
   stratum. The frame-level ROC of these benchmarks is dominated by which videos carry spans, not by
   where inside a video the span is.
5. **HateClipSeg's per-class appendix shows no class is carried by the visual channel.** Test ROC by
   class: hateful 0.499, insulting 0.540, sexual 0.501, violence 0.509, harm 0.509; the any-toxic
   max aggregate reaches 0.545, above every individual class but still 0.08 below the broadcast
   ceiling of 0.626 — and that ceiling is itself the lowest of the four datasets, because
   HateClipSeg's spans cover 46% of all frames.

Consequence for any future localisation claim of ours: on these four benchmarks the interval
between "chance" and "a perfect video-level classifier with zero localisation" is narrow
(AP 0.286→0.674 on HateMM, 0.471→0.543 on HateClipSeg), and the CLIP frame-similarity floor sits
inside the bottom tenth of it. A method that beats ZS-CLIP has cleared very little; the number worth
quoting against is the broadcast ceiling.


---

Batch 3 (2026-08-19): **Wave 0 item 2 — ZS-ImageBind**, image / video / audio, all four datasets.
Wave 0 item 4 (Qwen2.5-VL-7B native grounding) is still running and lands in a later batch.

## H. Method as run — ZS-ImageBind (image / video / audio)

LAVAD's zero-shot ImageBind baseline: cosine similarity of a unit-normalised ImageBind embedding
against the two text prompts `["normal", "hateful"]`, fixed before any number existed and never
tuned. The reported score is the softmax over the pair, a strictly increasing function of
`sim(hateful) − sim(normal)`, so the ranking metrics do not depend on that choice.

The Phase A dense cache is CLIP, not ImageBind, so all three channels were re-encoded from raw
video with ImageBind-Huge (`imagebind_huge.pth`, 4.8 GB) using the transforms shipped in
`lavad/libs/ImageBind/imagebind/data.py`:

| channel | unit | native rate | transform source |
|---|---|---|---|
| image | one frame | 4 fps | `load_and_transform_vision_data` (Resize short side 224 bicubic, CenterCrop 224, CLIP mean/std) |
| video | one 2 s clip, 2 frames | 0.5 fps | `load_and_transform_video_data` (`UniformTemporalSubsample(2)` inside ImageBind's native `clip_duration=2`) |
| audio | one 2 s clip | 0.5 fps | `load_and_transform_audio_data` (`waveform2melspec`, 128 mel bins, target_length 204, normalise −4.268 / 9.138) |

**The audio row is an extension, not a LAVAD baseline row.** LAVAD's published ZS-ImageBind baseline
is visual. ImageBind supports audio natively and audio matters in this domain, so the audio channel
is run and reported, marked as beyond the LAVAD baseline table.

**Two declared deviations from the shipped loaders.** (1) The video channel uses the centre crop
only, not `SpatialCrop(224, num_crops=3)`, so one decode serves both visual channels. (2) Clips tile
the whole video at a 2 s stride instead of `ConstantClipsPerVideoSampler` spreading a fixed number
of clips over it — a per-clip curve is the point here, a fixed clip count is not. The 0.5 fps
channels are broadcast piecewise-constant onto the 4 fps grid (freeze §1); the evaluator's broadcast
path was verified to reproduce the 4 fps result exactly on the gold oracle.

Embeddings are stored float16, one `.npy` per video per channel, under
`data/CLIP_Embedding/<DS>/imagebind_{image,video,audio}/`.

**Data-integrity finding, recorded because it touches an already-published row.** One HateClipSeg
video, `yt_NzvfkIYS5Yg`, is a **138 KB partial download**: ffprobe reports 273.9 s but neither
ffmpeg nor PyAV can decode a single frame (`partial file` / `Cannot determine format of input
stream 0:0 after EOF`). The pre-existing Phase A cache nevertheless holds a full-length
**all-zero** `(1096, 1024)` array for it in both `dense4fps_clipL336` and `dense4fps_w2vemo`, and
the video **is in the HateClipSeg test split**. A sweep of all eight caches found this to be the
only fabricated visual array anywhere (the 15 all-zero HateMM `w2vemo` arrays are the documented
no-audio videos, none of them in a split). The extractor here refuses to fabricate: the video has
no ImageBind embedding, is counted as missing (0.8% of the HateClipSeg test pool) and is dropped
from the pool. The ZS-CLIP rows above scored it as ~1096 frames of constant similarity; the effect
is bounded by 1% of that dataset's test frames but the rows are not strictly comparable to these
on HateClipSeg.

Reproduce: `bash` the extractor over the four datasets, then
`python scripts/repro_campaign/eval_frame.py --method imagebind --split {test,all}`
(raw JSON in `idea-stage/repro_campaign/eval_imagebind_{test,all}.json`,
embeddings in `data/CLIP_Embedding/<DS>/imagebind_{image,video,audio}/`, 3.0 GB,
logs in `logging/runs/repro_imagebind/`).

### H.1 Headline rows — test split

| method | wave | dataset | split | supervision | variant | query_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GOLD_BROADCAST | — | HateMM | test | control | control | n/a | video | 0.8857 | 0.5829 | n/a | n/a | n/a | 1.0000 | 116975 | 0.2421 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | HateMM | test | control | control | n/a | 4 fps | 0.5003 ± 0.0019 | 0.2423 ± 0.0013 | n/a | n/a | n/a | 0.0000 | 116975 | 0.2421 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| ZS-ImageBind (image) | 0 | HateMM | test | label-free | base | base | 4 fps | 0.5919 | 0.3143 | n/a | n/a | n/a | 0.2115 | 116908 | 0.2422 | 1 | n/a | §4 | idea-stage/repro_campaign/ |  |
| ZS-ImageBind (video) | 0 | HateMM | test | label-free | base | base | 0.5 fps | 0.5907 | 0.3100 | n/a | n/a | n/a | 0.1990 | 116931 | 0.2422 | 1 | n/a | §4 | idea-stage/repro_campaign/ |  |
| ZS-ImageBind (audio) | 0 | HateMM | test | label-free | base | base | 0.5 fps | 0.5654 | 0.2906 | n/a | n/a | n/a | 0.1422 | 116972 | 0.2421 | 1 | n/a | §4 | idea-stage/repro_campaign/ |  |
| GOLD_BROADCAST | — | MHC-EN | test | control | control | n/a | video | 0.9427 | 0.7664 | n/a | n/a | n/a | 1.0000 | 22337 | 0.2734 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | MHC-EN | test | control | control | n/a | 4 fps | 0.5004 ± 0.0034 | 0.2737 ± 0.0026 | n/a | n/a | n/a | 0.0000 | 22337 | 0.2734 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| ZS-ImageBind (image) | 0 | MHC-EN | test | label-free | base | base | 4 fps | 0.5938 | 0.3286 | n/a | n/a | n/a | 0.1119 | 22336 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_campaign/ |  |
| ZS-ImageBind (video) | 0 | MHC-EN | test | label-free | base | base | 0.5 fps | 0.5636 | 0.3056 | n/a | n/a | n/a | 0.0653 | 22337 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_campaign/ |  |
| ZS-ImageBind (audio) | 0 | MHC-EN | test | label-free | base | base | 0.5 fps | 0.6157 | 0.3678 | n/a | n/a | n/a | 0.1914 | 22337 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_campaign/ |  |
| GOLD_BROADCAST | — | MHC-ZH | test | control | control | n/a | video | 0.9842 | 0.9191 | n/a | n/a | n/a | 1.0000 | 18199 | 0.2648 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | MHC-ZH | test | control | control | n/a | 4 fps | 0.4985 ± 0.0052 | 0.2646 ± 0.0038 | n/a | n/a | n/a | 0.0000 | 18199 | 0.2648 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| ZS-ImageBind (image) | 0 | MHC-ZH | test | label-free | base | base | 4 fps | 0.5975 | 0.3580 | n/a | n/a | n/a | 0.1424 | 18195 | 0.2649 | 1 | n/a | §4 | idea-stage/repro_campaign/ |  |
| ZS-ImageBind (video) | 0 | MHC-ZH | test | label-free | base | base | 0.5 fps | 0.5727 | 0.3546 | n/a | n/a | n/a | 0.1372 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_campaign/ |  |
| ZS-ImageBind (audio) | 0 | MHC-ZH | test | label-free | base | base | 0.5 fps | 0.6527 | 0.3958 | n/a | n/a | n/a | 0.2000 | 18186 | 0.2650 | 1 | n/a | §4 | idea-stage/repro_campaign/ |  |
| GOLD_BROADCAST | — | HateClipSeg | test | control | control | n/a | video | 0.6260 | 0.5437 | n/a | n/a | n/a | 1.0000 | 114097 | 0.4712 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | HateClipSeg | test | control | control | n/a | 4 fps | 0.5009 ± 0.0021 | 0.4721 ± 0.0016 | n/a | n/a | n/a | 0.0000 | 114097 | 0.4712 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| ZS-ImageBind (image) | 0 | HateClipSeg | test | label-free | base | base | 4 fps | 0.5926 | 0.5535 | n/a | n/a | n/a | 1.0946 | 112926 | 0.4730 | 1 | n/a | §4 | idea-stage/repro_campaign/ | missing 1/119 (0.8%) dropped, not interpolated |
| ZS-ImageBind (video) | 0 | HateClipSeg | test | label-free | base | base | 0.5 fps | 0.5814 | 0.5431 | n/a | n/a | n/a | 0.9534 | 112928 | 0.4730 | 1 | n/a | §4 | idea-stage/repro_campaign/ | missing 1/119 (0.8%) dropped, not interpolated |
| ZS-ImageBind (audio) | 0 | HateClipSeg | test | label-free | base | base | 0.5 fps | 0.5652 | 0.5121 | n/a | n/a | n/a | 0.5288 | 112979 | 0.4732 | 1 | n/a | §4 | idea-stage/repro_campaign/ | missing 1/119 (0.8%) dropped, not interpolated |

### H.2 Full corpus

| method | wave | dataset | split | supervision | variant | query_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ZS-ImageBind (image) | 0 | HateMM | all | label-free | base | base | 4 fps | 0.5869 | 0.3546 | n/a | n/a | n/a | 0.1770 | 623788 | 0.2858 | 1 | n/a | §4 | idea-stage/repro_campaign/ | missing 2/1083 (0.2%) dropped, not interpolated |
| ZS-ImageBind (video) | 0 | HateMM | all | label-free | base | base | 0.5 fps | 0.5997 | 0.3618 | n/a | n/a | n/a | 0.1958 | 623861 | 0.2857 | 1 | n/a | §4 | idea-stage/repro_campaign/ | missing 2/1083 (0.2%) dropped, not interpolated |
| ZS-ImageBind (audio) | 0 | HateMM | all | label-free | base | base | 0.5 fps | 0.5827 | 0.3768 | n/a | n/a | n/a | 0.2310 | 622663 | 0.2874 | 1 | n/a | §4 | idea-stage/repro_campaign/ | missing 15/1083 (1.4%) dropped, not interpolated |
| ZS-ImageBind (image) | 0 | MHC-EN | all | label-free | base | base | 4 fps | 0.5874 | 0.3168 | n/a | n/a | n/a | 0.1365 | 110728 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_campaign/ |  |
| ZS-ImageBind (video) | 0 | MHC-EN | all | label-free | base | base | 0.5 fps | 0.5666 | 0.2967 | n/a | n/a | n/a | 0.0987 | 110735 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_campaign/ |  |
| ZS-ImageBind (audio) | 0 | MHC-EN | all | label-free | base | base | 0.5 fps | 0.6078 | 0.3274 | n/a | n/a | n/a | 0.1563 | 110735 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_campaign/ |  |
| ZS-ImageBind (image) | 0 | MHC-ZH | all | label-free | base | base | 4 fps | 0.5598 | 0.3065 | n/a | n/a | n/a | 0.0879 | 102130 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_campaign/ |  |
| ZS-ImageBind (video) | 0 | MHC-ZH | all | label-free | base | base | 0.5 fps | 0.5490 | 0.3010 | n/a | n/a | n/a | 0.0788 | 102152 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_campaign/ |  |
| ZS-ImageBind (audio) | 0 | MHC-ZH | all | label-free | base | base | 0.5 fps | 0.6139 | 0.3474 | n/a | n/a | n/a | 0.1560 | 102140 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_campaign/ |  |
| ZS-ImageBind (image) | 0 | HateClipSeg | all | label-free | base | base | 4 fps | 0.5889 | 0.5485 | n/a | n/a | n/a | 1.2697 | 374155 | 0.4641 | 1 | n/a | §4 | idea-stage/repro_campaign/ | missing 1/395 (0.2%) dropped, not interpolated |
| ZS-ImageBind (video) | 0 | HateClipSeg | all | label-free | base | base | 0.5 fps | 0.5856 | 0.5451 | n/a | n/a | n/a | 1.2190 | 374161 | 0.4641 | 1 | n/a | §4 | idea-stage/repro_campaign/ | missing 1/395 (0.2%) dropped, not interpolated |
| ZS-ImageBind (audio) | 0 | HateClipSeg | all | label-free | base | base | 0.5 fps | 0.5696 | 0.5118 | n/a | n/a | n/a | 0.7156 | 374155 | 0.4642 | 1 | n/a | §4 | idea-stage/repro_campaign/ | missing 1/395 (0.2%) dropped, not interpolated |

### H.3 Stratified — single-span vs multi-span (HateMM / MHC only)

Same convention as section E: `single_span` = videos with ≤1 gold span, `multi_span` = videos with
≥2 gold spans plus all zero-span videos, which supply the negative frames. Frame counts match
section E's exactly, which is an independent check that the two methods are pooled identically.

| method | wave | dataset | split | supervision | variant | query_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ZS-ImageBind (image) | 0 | HateMM | test | label-free | base | base | 4 fps | 0.6431 | 0.3243 | n/a | n/a | n/a | 0.2242 | 93251 | 0.2008 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (video) | 0 | HateMM | test | label-free | base | base | 0.5 fps | 0.6340 | 0.3149 | n/a | n/a | n/a | 0.2072 | 93271 | 0.2008 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (audio) | 0 | HateMM | test | label-free | base | base | 0.5 fps | 0.5878 | 0.2667 | n/a | n/a | n/a | 0.1198 | 93312 | 0.2007 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (image) | 0 | MHC-EN | test | label-free | base | base | 4 fps | 0.5851 | 0.3129 | n/a | n/a | n/a | 0.0969 | 21668 | 0.2628 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (video) | 0 | MHC-EN | test | label-free | base | base | 0.5 fps | 0.5563 | 0.2923 | n/a | n/a | n/a | 0.0570 | 21669 | 0.2628 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (audio) | 0 | MHC-EN | test | label-free | base | base | 0.5 fps | 0.6137 | 0.3541 | n/a | n/a | n/a | 0.1765 | 21669 | 0.2628 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (image) | 0 | MHC-ZH | test | label-free | base | base | 4 fps | 0.5975 | 0.3580 | n/a | n/a | n/a | 0.1424 | 18195 | 0.2649 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (video) | 0 | MHC-ZH | test | label-free | base | base | 0.5 fps | 0.5727 | 0.3546 | n/a | n/a | n/a | 0.1372 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (audio) | 0 | MHC-ZH | test | label-free | base | base | 0.5 fps | 0.6527 | 0.3958 | n/a | n/a | n/a | 0.2000 | 18186 | 0.2650 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (image) | 0 | HateMM | test | label-free | base | base | 4 fps | 0.5588 | 0.1247 | n/a | n/a | n/a | 0.0679 | 91996 | 0.1043 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span |
| ZS-ImageBind (video) | 0 | HateMM | test | label-free | base | base | 0.5 fps | 0.5771 | 0.1276 | n/a | n/a | n/a | 0.0776 | 92013 | 0.1042 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span |
| ZS-ImageBind (audio) | 0 | HateMM | test | label-free | base | base | 0.5 fps | 0.5365 | 0.1213 | n/a | n/a | n/a | 0.0567 | 92052 | 0.1042 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span |
| ZS-ImageBind (image) | 0 | MHC-EN | test | label-free | base | base | 4 fps | 0.8197 | 0.0868 | n/a | n/a | n/a | 0.1008 | 15036 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span |
| ZS-ImageBind (video) | 0 | MHC-EN | test | label-free | base | base | 0.5 fps | 0.7611 | 0.0668 | n/a | n/a | n/a | 0.0668 | 15037 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span |
| ZS-ImageBind (audio) | 0 | MHC-EN | test | label-free | base | base | 0.5 fps | 0.6644 | 0.0500 | n/a | n/a | n/a | 0.0383 | 15037 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span |
| ZS-ImageBind (image) | 0 | MHC-ZH | test | label-free | base | base | 4 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12952 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span single-class pool, metrics undefined |
| ZS-ImageBind (video) | 0 | MHC-ZH | test | label-free | base | base | 0.5 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12955 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span single-class pool, metrics undefined |
| ZS-ImageBind (audio) | 0 | MHC-ZH | test | label-free | base | base | 0.5 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12942 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span single-class pool, metrics undefined |
| ZS-ImageBind (image) | 0 | HateMM | all | label-free | base | base | 4 fps | 0.6061 | 0.3369 | n/a | n/a | n/a | 0.1584 | 528406 | 0.2520 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (video) | 0 | HateMM | all | label-free | base | base | 0.5 fps | 0.6171 | 0.3431 | n/a | n/a | n/a | 0.1700 | 528472 | 0.2520 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (audio) | 0 | HateMM | all | label-free | base | base | 0.5 fps | 0.5980 | 0.3619 | n/a | n/a | n/a | 0.2035 | 526139 | 0.2531 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (image) | 0 | MHC-EN | all | label-free | base | base | 4 fps | 0.5841 | 0.3089 | n/a | n/a | n/a | 0.1278 | 108453 | 0.2379 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (video) | 0 | MHC-EN | all | label-free | base | base | 0.5 fps | 0.5640 | 0.2894 | n/a | n/a | n/a | 0.0926 | 108460 | 0.2379 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (audio) | 0 | MHC-EN | all | label-free | base | base | 0.5 fps | 0.6095 | 0.3213 | n/a | n/a | n/a | 0.1500 | 108460 | 0.2379 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (image) | 0 | MHC-ZH | all | label-free | base | base | 4 fps | 0.5607 | 0.3074 | n/a | n/a | n/a | 0.0879 | 101597 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (video) | 0 | MHC-ZH | all | label-free | base | base | 0.5 fps | 0.5505 | 0.3021 | n/a | n/a | n/a | 0.0791 | 101619 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (audio) | 0 | MHC-ZH | all | label-free | base | base | 0.5 fps | 0.6146 | 0.3478 | n/a | n/a | n/a | 0.1540 | 101607 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=single_span |
| ZS-ImageBind (image) | 0 | HateMM | all | label-free | base | base | 4 fps | 0.5562 | 0.1152 | n/a | n/a | n/a | 0.0430 | 454771 | 0.0991 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span |
| ZS-ImageBind (video) | 0 | HateMM | all | label-free | base | base | 0.5 fps | 0.5795 | 0.1218 | n/a | n/a | n/a | 0.0607 | 454814 | 0.0991 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span |
| ZS-ImageBind (audio) | 0 | HateMM | all | label-free | base | base | 0.5 fps | 0.5581 | 0.1230 | n/a | n/a | n/a | 0.0590 | 453690 | 0.1010 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span |
| ZS-ImageBind (image) | 0 | MHC-EN | all | label-free | base | base | 4 fps | 0.7294 | 0.0359 | n/a | n/a | n/a | 0.0388 | 78205 | 0.0157 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span |
| ZS-ImageBind (video) | 0 | MHC-EN | all | label-free | base | base | 0.5 fps | 0.6767 | 0.0293 | n/a | n/a | n/a | 0.0262 | 78208 | 0.0157 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span |
| ZS-ImageBind (audio) | 0 | MHC-EN | all | label-free | base | base | 0.5 fps | 0.5912 | 0.0221 | n/a | n/a | n/a | 0.0123 | 78208 | 0.0157 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span |
| ZS-ImageBind (image) | 0 | MHC-ZH | all | label-free | base | base | 4 fps | 0.5947 | 0.0023 | n/a | n/a | n/a | 0.0023 | 72497 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span |
| ZS-ImageBind (video) | 0 | MHC-ZH | all | label-free | base | base | 0.5 fps | 0.4912 | 0.0024 | n/a | n/a | n/a | 0.0030 | 72513 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span |
| ZS-ImageBind (audio) | 0 | MHC-ZH | all | label-free | base | base | 0.5 fps | 0.7996 | 0.0154 | n/a | n/a | n/a | 0.0730 | 72500 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_campaign/ | stratum=multi_span |

### H.4 Alignment against LELA's published ZS-ImageBind row

LELA (arXiv 2602.09637) reports ZS-ImageBind frame ROC-AUC **0.5683** on HateMM and **0.5753** on
MultiHateClip, without stating the split, the language of its MultiHateClip column, the prompt
wording, or which ImageBind modality it used. The campaign brief set the agreement band at ±0.05.
The image channel is the natural counterpart to a published frame-level visual row.

| target | LELA | ours (image), full corpus | \|diff\| | ours (image), test split | \|diff\| |
|---|---|---|---|---|---|
| HateMM ROC | 0.5683 | **0.5869** | 0.0186 | **0.5919** | 0.0236 |
| MultiHateClip ROC vs MHC-EN | 0.5753 | **0.5874** | **0.0121** | 0.5938 | 0.0185 |
| MultiHateClip ROC vs MHC-ZH | 0.5753 | 0.5598 | 0.0155 | 0.5975 | 0.0222 |

**The transplant agrees.** Every one of the six comparisons lands inside ±0.024, well within the
±0.05 band, on a prompt pair fixed before any number existed. The video channel is also inside the
band on HateMM (0.5997, diff 0.0314). As with ZS-CLIP, the MHC-EN full-corpus figure is the closer
of the two MultiHateClip candidates (0.0121 vs 0.0155), but the margin is too small to identify
which language LELA pooled. Freeze §7's formal ±0.03 gate is defined only for LAVAD and URF-HVAA
and is `n/a` here; this section is the informal equivalent.

### H.5 What the numbers say

1. **ZS-ImageBind is a floor, and it is a slightly higher floor than ZS-CLIP.** Test-split image ROC
   is 0.5919 / 0.5938 / 0.5975 / 0.5926 (HateMM / MHC-EN / MHC-ZH / HateClipSeg) against a random
   floor of 0.500 and gold-broadcast ceilings of 0.886 / 0.943 / 0.984 / 0.626. ZS-CLIP's main-pair
   test row was 0.5368 / 0.5013 / 0.6075 / 0.4990. ImageBind is the more consistent of the two: it
   is 0.09–0.10 above chance on all four datasets, whereas ZS-CLIP was at chance on two of them.
   Oracle-normalised AP is still only 0.21 / 0.11 / 0.14 / 1.09 — on the three datasets with a
   meaningful ceiling it recovers a tenth to a fifth of the chance-to-video-oracle gap.
2. **Audio is the strongest channel on both MultiHateClip halves, and it is the channel LAVAD's
   baseline table does not have.** Test ROC: MHC-EN audio 0.6157 vs image 0.5938; MHC-ZH audio
   0.6527 vs image 0.5975 — the single best zero-shot number anywhere in this campaign so far, and
   +0.045 / +0.055 over the visual channel on the same videos. On HateMM and HateClipSeg the
   ordering reverses (audio 0.5654 / 0.5652 vs image 0.5919 / 0.5926). Read against the OCR ruling:
   the modality that carries hate in these corpora is not the one the published visual baselines
   score.
3. **The image and video channels are nearly interchangeable, so the 2 s temporal window buys
   nothing.** Test ROC differs by 0.001–0.030 between them on every dataset, with image ahead on
   three of four. ImageBind's video encoder sees 2 frames per 2 s clip, so this is evidence that at
   this scale the extra frame adds no localisation signal over a single frame — an eight-fold
   cheaper channel would have reported the same thing.
4. **The stratification reproduces the coverage degeneracy again, and more sharply than ZS-CLIP.**
   On MHC-EN test the multi-span stratum has a 2.7% positive base rate and ImageBind's image channel
   reaches ROC 0.8197 there while sitting at 0.5851 on the single-span stratum; on MHC-EN full corpus
   the same contrast is 0.7294 vs 0.5841. The frame-level ROC of these benchmarks is still dominated
   by which videos carry spans, not by where inside a video the span sits.
5. **AP_norm above 1 on HateClipSeg is a statement about the benchmark, not the method.**
   HateClipSeg's broadcast ceiling is AP 0.5437 against a base rate of 0.4712, so the whole
   chance-to-oracle interval is 0.07 wide; ImageBind's 0.5535 clears it. A video-level oracle is
   simply not a useful ceiling on a dataset where the annotated spans cover 47% of all frames.


---

Batch 4 (2026-08-19): **Wave 0 item 4 — Qwen2.5-VL-7B native temporal grounding**, all four
datasets. This completes Wave 0's label-free floors.

## I. Method as run — Qwen2.5-VL-7B native temporal grounding


The harness is `scripts/repro_campaign/qwen25vl_grounding.py`, smoke-tested in Phase A. Its
`pre_prompt`, `post_prompt`, greedy decoding, `max_new_tokens=50`, 32 uniformly sampled frames and
the `extract_time` regex cascade are copied verbatim from the lmms-eval
`temporal_grounding_charades` task, which is the harness TempSamp-R1 reports its Qwen2.5-VL-7B
zero-shot row under. `scripts/repro_campaign/run_qwen_grounding.py` adds only the corpus loop,
resume, and decode reuse across queries.

**Query text, recorded because the freeze does not fix one.** Freeze §9 names the method and its
prompt convention but never writes down the query sentence, so the campaign brief's default is used
verbatim for the main row:

```
the moment containing hateful content targeting a person or group
```

HateClipSeg additionally gets one query per released class, same sentence frame:
`the moment containing {normal, non-toxic | hateful | insulting | sexual | violent | harmful}
content`. Each per-class row is scored against **that class's own frame labels**, rebuilt from
`gold_segments.json` on the 4 fps grid (the union of classes 1–5 reproduces the stored any-toxic
array exactly, and the gold-interval oracle scores F1 = 1.0 at every tIoU for every class).

**Model precision — declared deviation.** The model is loaded in **4-bit NF4** (the Phase A smoke
configuration, `MODEL_ASSETS_STATUS §3.7`), not bf16. Reason: at the lmms-eval default of 32 frames
and `max_pixels=151200`, one bf16 forward peaks at ~24 GiB and then asks for a further 7.9 GiB
block, which does not fit the 32 GiB card at all, let alone beside the parallel extraction job. The
NF4 run peaks at 27 GiB. Everything else — prompt, frame count, pixel budget, decoding — is the
published configuration.

**Decoder fallback — declared deviation, and the reason it was necessary.** `qwen_vl_utils` reads
video with decord and silently falls back to torchvision when decord fails; torchvision then
reports a nonsense frame count and refuses `nframes=32`. On a 60-video sample per dataset that
failure hits **27% of MHC-EN, 7% of MHC-ZH and 7% of HateClipSeg** containers, so taking the
harness at face value would have recorded roughly a tenth of the corpus as model failures that were
really decode failures. Videos that decord rejects are decoded with PyAV at the same uniform frame
indices decord would have used, and the count of fallback videos is reported.

**Score curve and the AUC it produces.** The model emits one interval per (video, query). The
interval is clipped to `[0, D)` and rasterised to a **binary 0/1 curve** on the 4 fps grid. Read the
frame ROC-AUC and PR-AUC accordingly: a two-valued score has a single operating point, so its ROC
curve is one interior vertex and its ROC-AUC equals `(TPR + TNR) / 2`, i.e. balanced accuracy at
that operating point — not the threshold-swept quantity the same column holds for a score-curve
method. The comparison against ZS-CLIP's and ZS-ImageBind's continuous curves is therefore a
comparison of a *committed* prediction against a *ranked* one, which favours neither in a
straightforward way. `F1@tIoU` is the metric this method is actually built for, and unlike the
score-curve methods it is reported rather than `n/a`.

**Missing videos.** A refusal, an unparseable generation or a decode failure is recorded with
`span=null` and the video is **dropped from the pool, never interpolated** (freeze §14).

**Run record.** 5,452 generations over 3,082 videos (one main query everywhere, plus six per-class
queries on HateClipSeg), 4 h 20 min wall clock on one RTX 5090. **Zero refusals and zero
unparseable generations** — `extract_time` returned an interval for every generation the model
produced, so the LLM-refusal failure mode that `MODEL_ASSETS_STATUS §3.11a` flags for the
Llama-based scorers does not arise here.

**Missing videos: 16 of 3,082 (0.5%).** Two are the D2 audio-only HateMM containers. The other
fourteen are files truncated mid-download (`partial file`, `Invalid NAL unit size`) on which the C
video decoder does not raise but **kills the process outright**; this stopped the run twice before
the driver was given an in-flight marker and a supervisor, after which each crash retires exactly
one id as a decode failure and the run continues. Per dataset: HateMM 3/1083, MHC-EN 11/792,
MHC-ZH 3/814, HateClipSeg 1/395. All are dropped from the pool, never interpolated.

Reproduce: `bash scripts/repro_campaign/run_qwen_supervised.sh`, then
`python scripts/repro_campaign/eval_frame.py --method qwen_grounding --split {test,all} --qkeys main,c0_normal,...`
(raw generations in `idea-stage/repro_qwen_ground/raw/`, logs in `logging/runs/repro_qwen_ground/`).

### I.1 Headline rows — main query, test split

| method | wave | dataset | split | supervision | variant | query_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GOLD_BROADCAST | — | HateMM | test | control | control | n/a | video | 0.8857 | 0.5829 | n/a | n/a | n/a | 1.0000 | 116975 | 0.2421 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | HateMM | test | control | control | n/a | 4 fps | 0.5003 ± 0.0019 | 0.2423 ± 0.0013 | n/a | n/a | n/a | 0.0000 | 116975 | 0.2421 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| Qwen2.5-VL-7B grounding | 0 | HateMM | test | label-free | base | query=main | 4 fps | 0.5185 | 0.2522 | 0.0549 | 0.0299 | 0.0200 | 0.0297 | 116975 | 0.2421 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point |
| GOLD_BROADCAST | — | MHC-EN | test | control | control | n/a | video | 0.9427 | 0.7664 | n/a | n/a | n/a | 1.0000 | 22337 | 0.2734 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | MHC-EN | test | control | control | n/a | 4 fps | 0.5004 ± 0.0034 | 0.2737 ± 0.0026 | n/a | n/a | n/a | 0.0000 | 22337 | 0.2734 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| Qwen2.5-VL-7B grounding | 0 | MHC-EN | test | label-free | base | query=main | 4 fps | 0.5221 | 0.2806 | 0.0558 | 0.0279 | 0.0186 | 0.0222 | 21896 | 0.2696 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 2/161 (1.2%) dropped, not interpolated |
| GOLD_BROADCAST | — | MHC-ZH | test | control | control | n/a | video | 0.9842 | 0.9191 | n/a | n/a | n/a | 1.0000 | 18199 | 0.2648 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | MHC-ZH | test | control | control | n/a | 4 fps | 0.4985 ± 0.0052 | 0.2646 ± 0.0038 | n/a | n/a | n/a | 0.0000 | 18199 | 0.2648 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| Qwen2.5-VL-7B grounding | 0 | MHC-ZH | test | label-free | base | query=main | 4 fps | 0.5113 | 0.2699 | 0.0718 | 0.0205 | 0.0000 | 0.0077 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point |
| GOLD_BROADCAST | — | HateClipSeg | test | control | control | n/a | video | 0.6260 | 0.5437 | n/a | n/a | n/a | 1.0000 | 114097 | 0.4712 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | HateClipSeg | test | control | control | n/a | 4 fps | 0.5009 ± 0.0021 | 0.4721 ± 0.0016 | n/a | n/a | n/a | 0.0000 | 114097 | 0.4712 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| Qwen2.5-VL-7B grounding | 0 | HateClipSeg | test | label-free | base | query=main | 4 fps | 0.5030 | 0.4750 | 0.0298 | 0.0128 | 0.0043 | 0.0231 | 113002 | 0.4733 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 1/119 (0.8%) dropped, not interpolated |

### I.2 Full corpus — main query

| method | wave | dataset | split | supervision | variant | query_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B grounding | 0 | HateMM | all | label-free | base | query=main | 4 fps | 0.5245 | 0.3018 | 0.0679 | 0.0453 | 0.0226 | 0.0400 | 622316 | 0.2862 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 3/1083 (0.3%) dropped, not interpolated |
| Qwen2.5-VL-7B grounding | 0 | MHC-EN | all | label-free | base | query=main | 4 fps | 0.5119 | 0.2501 | 0.0560 | 0.0270 | 0.0097 | 0.0094 | 108488 | 0.2450 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 11/792 (1.4%) dropped, not interpolated |
| Qwen2.5-VL-7B grounding | 0 | MHC-ZH | all | label-free | base | query=main | 4 fps | 0.5155 | 0.2608 | 0.0633 | 0.0279 | 0.0112 | 0.0116 | 101557 | 0.2539 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 3/814 (0.4%) dropped, not interpolated |
| Qwen2.5-VL-7B grounding | 0 | HateClipSeg | all | label-free | base | query=main | 4 fps | 0.5049 | 0.4671 | 0.0267 | 0.0115 | 0.0013 | 0.0439 | 374235 | 0.4642 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 1/395 (0.2%) dropped, not interpolated |

### I.3 HateClipSeg 6-class appendix

One query per released class, each scored against **that class's own frame labels** rebuilt from
`gold_segments.json` (the `c0_normal` row is therefore scored against the normal class, whose base
rate is the complement of any-toxic).

| method | wave | dataset | split | supervision | variant | query_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B grounding | 0 | HateClipSeg | test | label-free | base | query=c0_normal | 4 fps | 0.5014 | 0.5274 | 0.0158 | 0.0000 | 0.0000 | 0.0514 | 113002 | 0.5267 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 1/119 (0.8%) dropped, not interpolated |
| Qwen2.5-VL-7B grounding | 0 | HateClipSeg | test | label-free | base | query=c1_hateful | 4 fps | 0.5008 | 0.2005 | 0.0133 | 0.0067 | 0.0067 | 0.0010 | 113002 | 0.2002 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 1/119 (0.8%) dropped, not interpolated |
| Qwen2.5-VL-7B grounding | 0 | HateClipSeg | test | label-free | base | query=c2_insulting | 4 fps | 0.5011 | 0.2577 | 0.0330 | 0.0000 | 0.0000 | 0.0032 | 113002 | 0.2572 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 1/119 (0.8%) dropped, not interpolated |
| Qwen2.5-VL-7B grounding | 0 | HateClipSeg | test | label-free | base | query=c3_sexual | 4 fps | 0.5006 | 0.0573 | 0.0000 | 0.0000 | 0.0000 | 0.0003 | 113002 | 0.0572 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 1/119 (0.8%) dropped, not interpolated |
| Qwen2.5-VL-7B grounding | 0 | HateClipSeg | test | label-free | base | query=c4_violence | 4 fps | 0.5004 | 0.1368 | 0.0154 | 0.0077 | 0.0000 | 0.0005 | 113002 | 0.1367 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 1/119 (0.8%) dropped, not interpolated |
| Qwen2.5-VL-7B grounding | 0 | HateClipSeg | test | label-free | base | query=c5_harm | 4 fps | 0.5009 | 0.0117 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 113002 | 0.0117 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 1/119 (0.8%) dropped, not interpolated |
| Qwen2.5-VL-7B grounding | 0 | HateClipSeg | all | label-free | base | query=c0_normal | 4 fps | 0.4985 | 0.5351 | 0.0082 | 0.0012 | 0.0000 | -0.0338 | 374235 | 0.5358 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 1/395 (0.2%) dropped, not interpolated |
| Qwen2.5-VL-7B grounding | 0 | HateClipSeg | all | label-free | base | query=c1_hateful | 4 fps | 0.5052 | 0.2119 | 0.0166 | 0.0062 | 0.0021 | 0.0092 | 374235 | 0.2095 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 1/395 (0.2%) dropped, not interpolated |
| Qwen2.5-VL-7B grounding | 0 | HateClipSeg | all | label-free | base | query=c2_insulting | 4 fps | 0.5009 | 0.2570 | 0.0273 | 0.0080 | 0.0016 | 0.0027 | 374235 | 0.2566 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 1/395 (0.2%) dropped, not interpolated |
| Qwen2.5-VL-7B grounding | 0 | HateClipSeg | all | label-free | base | query=c3_sexual | 4 fps | 0.5014 | 0.0366 | 0.0000 | 0.0000 | 0.0000 | 0.0006 | 374235 | 0.0365 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 1/395 (0.2%) dropped, not interpolated |
| Qwen2.5-VL-7B grounding | 0 | HateClipSeg | all | label-free | base | query=c4_violence | 4 fps | 0.5051 | 0.1224 | 0.0244 | 0.0073 | 0.0000 | 0.0107 | 374235 | 0.1207 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 1/395 (0.2%) dropped, not interpolated |
| Qwen2.5-VL-7B grounding | 0 | HateClipSeg | all | label-free | base | query=c5_harm | 4 fps | 0.4983 | 0.0050 | 0.0000 | 0.0000 | 0.0000 | -0.0001 | 374235 | 0.0050 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | binary 0/1 curve: ROC-AUC = balanced accuracy at one operating point; missing 1/395 (0.2%) dropped, not interpolated |

### I.4 Stratified — single-span vs multi-span (HateMM / MHC only)

| method | wave | dataset | split | supervision | variant | query_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B grounding | 0 | HateMM | test | label-free | base | query=main | 4 fps | 0.5282 | 0.2166 | 0.0656 | 0.0492 | 0.0328 | 0.0289 | 93315 | 0.2007 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | stratum=single_span |
| Qwen2.5-VL-7B grounding | 0 | MHC-EN | test | label-free | base | query=main | 4 fps | 0.5242 | 0.2706 | 0.0585 | 0.0293 | 0.0195 | 0.0230 | 21228 | 0.2587 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | stratum=single_span |
| Qwen2.5-VL-7B grounding | 0 | MHC-ZH | test | label-free | base | query=main | 4 fps | 0.5113 | 0.2699 | 0.0718 | 0.0205 | 0.0000 | 0.0077 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | stratum=single_span |
| Qwen2.5-VL-7B grounding | 0 | HateMM | test | label-free | base | query=main | 4 fps | 0.5025 | 0.1047 | 0.0209 | 0.0000 | 0.0000 | 0.0017 | 92052 | 0.1042 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | stratum=multi_span |
| Qwen2.5-VL-7B grounding | 0 | MHC-EN | test | label-free | base | query=main | 4 fps | 0.4909 | 0.0274 | 0.0000 | 0.0000 | 0.0000 | -0.0007 | 14799 | 0.0278 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | stratum=multi_span |
| Qwen2.5-VL-7B grounding | 0 | MHC-ZH | test | label-free | base | query=main | 4 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12955 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | stratum=multi_span single-class pool, metrics undefined |
| Qwen2.5-VL-7B grounding | 0 | HateMM | all | label-free | base | query=main | 4 fps | 0.5245 | 0.2670 | 0.0688 | 0.0516 | 0.0313 | 0.0281 | 528575 | 0.2520 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | stratum=single_span |
| Qwen2.5-VL-7B grounding | 0 | MHC-EN | all | label-free | base | query=main | 4 fps | 0.5136 | 0.2450 | 0.0561 | 0.0261 | 0.0080 | 0.0103 | 106340 | 0.2392 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | stratum=single_span |
| Qwen2.5-VL-7B grounding | 0 | MHC-ZH | all | label-free | base | query=main | 4 fps | 0.5155 | 0.2608 | 0.0602 | 0.0263 | 0.0113 | 0.0114 | 101024 | 0.2539 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | stratum=single_span |
| Qwen2.5-VL-7B grounding | 0 | HateMM | all | label-free | base | query=main | 4 fps | 0.5289 | 0.1095 | 0.0309 | 0.0146 | 0.0016 | 0.0274 | 453238 | 0.0991 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | stratum=multi_span |
| Qwen2.5-VL-7B grounding | 0 | MHC-EN | all | label-free | base | query=main | 4 fps | 0.4808 | 0.0146 | 0.0034 | 0.0034 | 0.0034 | -0.0007 | 76841 | 0.0149 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | stratum=multi_span |
| Qwen2.5-VL-7B grounding | 0 | MHC-ZH | all | label-free | base | query=main | 4 fps | 0.5139 | 0.0020 | 0.0071 | 0.0036 | 0.0000 | 0.0003 | 72057 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_qwen_ground/ | stratum=multi_span |

### I.5 What the numbers say

1. **Qwen2.5-VL's native grounding is at chance on all four datasets.** Test-split frame ROC-AUC is
   0.5185 / 0.5221 / 0.5113 / 0.5030 (HateMM / MHC-EN / MHC-ZH / HateClipSeg) against a random floor
   of 0.500; oracle-normalised AP is 0.030 / 0.022 / 0.008 / 0.023, i.e. it recovers under 3% of the
   chance-to-video-oracle gap everywhere. It is the weakest of the three Wave 0 floors: ZS-CLIP
   reached 0.5368 / 0.5013 / 0.6075 / 0.4990 and ZS-ImageBind 0.5919 / 0.5938 / 0.5975 / 0.5926.
   The result is not a refusal artefact — every generation parsed.
2. **F1@tIoU is near zero, and most of that is a length mismatch rather than a placement error.**
   Test F1@0.3 / 0.5 / 0.7 is 0.055 / 0.030 / 0.020 on HateMM and 0.030 / 0.013 / 0.004 on
   HateClipSeg. The model's predicted intervals have a **median length of 4.2–5.9 s** against a
   **median gold span of 19–21 s**. Since one predicted interval of length L can reach at best
   `min(L,G)/max(L,G)` IoU against a gold span of length G, the predicted lengths alone cap the
   fraction of videos that *could* reach tIoU 0.5 at **24% (HateMM), 16% (MHC-EN), 11% (MHC-ZH),
   48% (HateClipSeg)** even with perfect placement. The method attains roughly a tenth to a fifth of
   even that capped ceiling.
3. **The model answers largely without looking.** A single identical generation,
   `"The event happens in 19.8 - 25.7 seconds."`, is returned for **12.6% of HateMM, 19.3% of MHC-EN
   and 10.6% of MHC-ZH videos**; on HateClipSeg the modal answer is
   `"The event happens in 15.8 - 21.9 seconds."` (9.4%). Median predicted start is 16.8–18.9 s on
   three datasets regardless of video length. This is the Charades-STA prior — short activity
   intervals early in a short clip — transferred verbatim to videos whose hateful spans are long and
   distributed differently. It also explains why the per-class HateClipSeg rows are flat at
   ROC 0.4983–0.5052: the query text barely moves the answer.
4. **Read the ROC-AUC column with the binary-curve caveat.** This method commits to one interval, so
   its frame score is two-valued and its ROC-AUC is balanced accuracy at a single operating point,
   not a threshold-swept ranking quality. That normally *flatters* a method relative to a continuous
   curve at the same quality, which makes 0.50–0.52 a weaker result than the same figure from
   ZS-CLIP or ZS-ImageBind, not a comparable one.
5. **Consequence for the campaign.** The one Wave 0 method that natively emits intervals — the only
   one for which `F1@tIoU` is even defined — cannot localise hate spans in these corpora at all. Any
   future localisation claim of ours therefore has no meaningful interval-level zero-shot baseline to
   clear on these benchmarks; the honest comparison remains the gold-broadcast ceiling of §3.

---

## M. Method as run — LaGoVAD (ICLR 2026)

LaGoVAD's premise is that the anomaly is **defined at inference time by free text**, so the
campaign runs it the way the paper intends: a written definition of hateful content is the query,
and the model's per-frame similarity to that definition is the score.

**Code path.** `third_party/LaGoVAD-PreVAD` @ `e2b93f85`. The repo ships one entry point,
`src/end2end_inference.py`, a single-video demo whose query list is hard-coded to the six
XD-Violence class names and whose only output is a PNG. The campaign patch
(`scripts/repro_campaign/patches/LaGoVAD-PreVAD.patch`) adds `--queries`, which routes the forward
through the `cap_*` head — the free-text head — instead of the `cls_*` (soft-prompted class-name)
head, and dumps the raw curves. `scripts/repro_campaign/run_lagovad.py` is the corpus loop around
exactly that code path; the model class, the released `best.ckpt`, the CLIP ViT-B/16 visual tower,
the every-8th-frame sampling and the 224×224 square resize are the demo's.

**Frame sampling and native rate.** Upstream extracts every 8th decoded frame
(`FRAME_INTERVAL = 8`) as a JPEG and lets `CLIPProcessor(size=(224,224), do_center_crop=True)`
resize it, which for a 224 crop box is a plain square resize. We stream the same selection out of
ffmpeg (`select=not(mod(n,8)),scale=224:224`) instead of writing ~1.1 M JPEGs to disk. Checked
against the upstream JPEG path on `bit_0EHvMSiEHVoc`: same frame count (680) and **mean cosine
1.000000** between the two feature sets. The native rate is therefore `fps/8`, which varies per
video (2.9–7.5 samples/s across the corpora); each video's own rate is recorded and its curve is
broadcast piecewise-constant onto the 4 fps grid by the shared evaluator's own broadcast function
(freeze §1).

**Long videos.** `max_position_embeddings` is 1536. 49 HateMM videos exceed that at `fps/8`
(the longest is 5,808 s). Those are processed in consecutive non-overlapping chunks of 1,500
samples and the per-frame outputs concatenated; the temporal encoder is a 2-layer transformer, so
this removes attention across chunk boundaries and nothing else. Recorded as our adaptation.

**Score = raw logit, not the demo's sigmoid.** `SimScoreHead` divides the cosine similarity by a
learned temperature before returning it, and the demo then applies `sigmoid`. The resulting logits
are wide enough that the sigmoid saturates to 0/1 in float32 and destroys the within-video ranking.
ROC-AUC and AP are rank metrics and the sigmoid is strictly increasing, so the raw logit is the
same measurement without the saturation; we store the raw logit.

**Query grouping is part of the design, not an accident of batching.** LaGoVAD fuses text and
vision with co-attention in which the *visual* stream attends over the whole text set
(`vis_attn(v_feat, t_feat, t_feat)`), so a query's score depends on which other queries share the
forward pass. The groups are therefore fixed in advance:

- each definition **alone**, giving the `main`, `sens_short`, `sens_vad` and `normal` rows;
- each hate definition **paired with the `normal` definition** — the two-class set-up upstream's own
  `Normal + anomaly` class list uses — giving the `<def>_pair` row (the hate similarity from that
  paired forward) and the `<def>_vsnormal` row (hate minus normal from the same forward);
- HateClipSeg's **six released classes in one forward**, the multi-class set-up the model was built
  for, giving the `c0…c5` rows.

**Definition texts, frozen in `scripts/repro_campaign/run_lagovad.py` and committed before the
first run** (commit `aea5e5f`), so no wording could be chosen after seeing a number. All are
reported; none is selected.

| key | text |
|---|---|
| `main` | Hateful content: speech, imagery or on-screen text that attacks, dehumanizes, demeans or incites hatred or violence against a person or a group because of a protected attribute such as race, ethnicity, national origin, religion, gender, sexual orientation or disability. |
| `sens_short` | A person or a group is being attacked, insulted or dehumanized because of their race, religion, gender, sexuality or nationality. |
| `sens_vad` | Abnormal event: hateful or abusive behaviour directed at a person or a group, including slurs, harassment, threats and violence. |
| `normal` | Normal behavior, views or content that does not involve any hateful, abusive, offensive or otherwise unexpected event. |
| `c0_normal` | Normal, non-toxic content that does not attack, insult, sexualise or harm anyone. |
| `c1_hateful` | Hateful content that attacks or dehumanizes a person or a group because of race, ethnicity, religion, gender, sexual orientation, nationality or disability. |
| `c2_insulting` | Insulting content: mocking, humiliating, name-calling or otherwise demeaning a person or a group. |
| `c3_sexual` | Sexual content: nudity, sexual acts, or sexually explicit language and imagery. |
| `c4_violence` | Violent content: physical fighting, assault, weapons, blood, injury or killing. |
| `c5_harm` | Harmful content: self-harm, suicide, dangerous acts, or content that encourages people to harm themselves or others. |

**Correction to a Phase A observation.** `MODEL_ASSETS_STATUS §3.1` recorded that LaGoVAD's binary
head is constant across frames and that the per-frame signal lives only in the similarity matrix.
Both halves of that were an artefact of a **corrupted `openai/clip-vit-base-patch16` checkpoint**:
the cached `pytorch_model.bin` had the right byte count and wrong content (max |w| = 3.7 × 10¹⁹), so
CLIP returned one identical image embedding for every frame of every video, and therefore every
downstream curve — binary head and similarity matrix alike — was flat. With the verified
checkpoint, both vary. The cache was audited by re-hashing every blob against its own filename (an
HF cache blob is stored under its sha256): 7 of 55 weight files were corrupt, all repaired and
re-verified. See `idea-stage/repro_campaign/hf_cache_audit.txt` and
`scripts/repro_campaign/{audit_hf_cache.sh,hf_refetch.py}`.

`F1@tIoU` is `n/a` for every LaGoVAD row: it emits a score curve, not intervals, and freeze §2
forbids inventing a threshold for it.

Reproduce: `python scripts/repro_campaign/run_lagovad.py --stage extract` then `--stage infer`,
then `scripts/repro_campaign/eval_frame.py --method curves --curve-dir idea-stage/repro_lagovad/curves`
(driver `scripts/repro_campaign/run_lagovad_chain.sh`, logs in `logging/runs/repro_lagovad_*`).

### M.1 Headline rows — test split

| method | wave | dataset | split | supervision | variant | query_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GOLD_BROADCAST | — | HateMM | test | control | control | n/a | video | 0.8857 | 0.5829 | n/a | n/a | n/a | 1.0000 | 116975 | 0.2421 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | HateMM | test | control | control | n/a | 4 fps | 0.5003 ± 0.0019 | 0.2423 ± 0.0013 | n/a | n/a | n/a | 0.0000 | 116975 | 0.2421 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| GOLD_BROADCAST | — | MHC-EN | test | control | control | n/a | video | 0.9427 | 0.7664 | n/a | n/a | n/a | 1.0000 | 22337 | 0.2734 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | MHC-EN | test | control | control | n/a | 4 fps | 0.5004 ± 0.0034 | 0.2737 ± 0.0026 | n/a | n/a | n/a | 0.0000 | 22337 | 0.2734 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| GOLD_BROADCAST | — | MHC-ZH | test | control | control | n/a | video | 0.9842 | 0.9191 | n/a | n/a | n/a | 1.0000 | 18199 | 0.2648 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | MHC-ZH | test | control | control | n/a | 4 fps | 0.4985 ± 0.0052 | 0.2646 ± 0.0038 | n/a | n/a | n/a | 0.0000 | 18199 | 0.2648 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| GOLD_BROADCAST | — | HateClipSeg | test | control | control | n/a | video | 0.6260 | 0.5437 | n/a | n/a | n/a | 1.0000 | 114097 | 0.4712 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | HateClipSeg | test | control | control | n/a | 4 fps | 0.5009 ± 0.0021 | 0.4721 ± 0.0016 | n/a | n/a | n/a | 0.0000 | 114097 | 0.4712 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | main | 4 fps | 0.5579 | 0.3047 | n/a | n/a | n/a | 0.1836 | 116975 | 0.2421 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_short | 4 fps | 0.5229 | 0.2661 | n/a | n/a | n/a | 0.0703 | 116975 | 0.2421 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.5507 | 0.2651 | n/a | n/a | n/a | 0.0674 | 116975 | 0.2421 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | normal | 4 fps | 0.4879 | 0.2372 | n/a | n/a | n/a | -0.0143 | 116975 | 0.2421 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | main_pair | 4 fps | 0.5458 | 0.2913 | n/a | n/a | n/a | 0.1445 | 116975 | 0.2421 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.5336 | 0.2849 | n/a | n/a | n/a | 0.1256 | 116975 | 0.2421 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.5193 | 0.2773 | n/a | n/a | n/a | 0.1032 | 116975 | 0.2421 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.5176 | 0.2680 | n/a | n/a | n/a | 0.0762 | 116975 | 0.2421 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.5399 | 0.2689 | n/a | n/a | n/a | 0.0787 | 116975 | 0.2421 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.5485 | 0.2811 | n/a | n/a | n/a | 0.1143 | 116975 | 0.2421 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | bin | 4 fps | 0.4989 | 0.2317 | n/a | n/a | n/a | -0.0306 | 116975 | 0.2421 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | main | 4 fps | 0.5239 | 0.2617 | n/a | n/a | n/a | -0.0238 | 22337 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_short | 4 fps | 0.5149 | 0.2644 | n/a | n/a | n/a | -0.0182 | 22337 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.4236 | 0.2256 | n/a | n/a | n/a | -0.0969 | 22337 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | normal | 4 fps | 0.3844 | 0.2175 | n/a | n/a | n/a | -0.1133 | 22337 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | main_pair | 4 fps | 0.4893 | 0.2480 | n/a | n/a | n/a | -0.0515 | 22337 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.5885 | 0.3033 | n/a | n/a | n/a | 0.0607 | 22337 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.5050 | 0.2552 | n/a | n/a | n/a | -0.0370 | 22337 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.5862 | 0.3015 | n/a | n/a | n/a | 0.0569 | 22337 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.4123 | 0.2229 | n/a | n/a | n/a | -0.1024 | 22337 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.5796 | 0.3166 | n/a | n/a | n/a | 0.0875 | 22337 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | bin | 4 fps | 0.6058 | 0.3490 | n/a | n/a | n/a | 0.1534 | 22337 | 0.2734 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | main | 4 fps | 0.5965 | 0.3118 | n/a | n/a | n/a | 0.0717 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_short | 4 fps | 0.6432 | 0.3698 | n/a | n/a | n/a | 0.1605 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.4593 | 0.2303 | n/a | n/a | n/a | -0.0528 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | normal | 4 fps | 0.4303 | 0.2208 | n/a | n/a | n/a | -0.0673 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | main_pair | 4 fps | 0.5911 | 0.3128 | n/a | n/a | n/a | 0.0733 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.5786 | 0.2981 | n/a | n/a | n/a | 0.0508 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.6479 | 0.3523 | n/a | n/a | n/a | 0.1337 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.6117 | 0.3349 | n/a | n/a | n/a | 0.1070 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.4717 | 0.2362 | n/a | n/a | n/a | -0.0438 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.6480 | 0.3744 | n/a | n/a | n/a | 0.1674 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | bin | 4 fps | 0.5673 | 0.2895 | n/a | n/a | n/a | 0.0377 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | HateClipSeg | test | aux-temporal-pretrain | base | main | 4 fps | 0.5000 | 0.4666 | n/a | n/a | n/a | -0.0918 | 113002 | 0.4733 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/119 (0.8%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | test | aux-temporal-pretrain | base | sens_short | 4 fps | 0.5428 | 0.5280 | n/a | n/a | n/a | 0.7435 | 113002 | 0.4733 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/119 (0.8%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | test | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.4973 | 0.4693 | n/a | n/a | n/a | -0.0553 | 113002 | 0.4733 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/119 (0.8%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | test | aux-temporal-pretrain | base | normal | 4 fps | 0.4680 | 0.4634 | n/a | n/a | n/a | -0.1353 | 113002 | 0.4733 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/119 (0.8%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | test | aux-temporal-pretrain | base | main_pair | 4 fps | 0.4937 | 0.4575 | n/a | n/a | n/a | -0.2153 | 113002 | 0.4733 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/119 (0.8%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | test | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.5261 | 0.4969 | n/a | n/a | n/a | 0.3209 | 113002 | 0.4733 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/119 (0.8%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | test | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.5417 | 0.5152 | n/a | n/a | n/a | 0.5685 | 113002 | 0.4733 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/119 (0.8%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | test | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.5460 | 0.5367 | n/a | n/a | n/a | 0.8617 | 113002 | 0.4733 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/119 (0.8%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | test | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.5094 | 0.4758 | n/a | n/a | n/a | 0.0334 | 113002 | 0.4733 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/119 (0.8%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | test | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.5508 | 0.5416 | n/a | n/a | n/a | 0.9285 | 113002 | 0.4733 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/119 (0.8%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | test | aux-temporal-pretrain | base | bin | 4 fps | 0.5431 | 0.5499 | n/a | n/a | n/a | 1.0402 | 113002 | 0.4733 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/119 (0.8%) dropped, not interpolated |

### M.2 Full corpus

| method | wave | dataset | split | supervision | variant | query_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | main | 4 fps | 0.5225 | 0.3099 | n/a | n/a | n/a | 0.0624 | 624110 | 0.2856 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_short | 4 fps | 0.4938 | 0.2836 | n/a | n/a | n/a | -0.0052 | 624110 | 0.2856 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.5105 | 0.2946 | n/a | n/a | n/a | 0.0232 | 624110 | 0.2856 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | normal | 4 fps | 0.4947 | 0.3039 | n/a | n/a | n/a | 0.0472 | 624110 | 0.2856 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | main_pair | 4 fps | 0.5178 | 0.3116 | n/a | n/a | n/a | 0.0670 | 624110 | 0.2856 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.5106 | 0.3050 | n/a | n/a | n/a | 0.0499 | 624110 | 0.2856 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.4875 | 0.2908 | n/a | n/a | n/a | 0.0134 | 624110 | 0.2856 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.4910 | 0.3007 | n/a | n/a | n/a | 0.0389 | 624110 | 0.2856 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.5040 | 0.2945 | n/a | n/a | n/a | 0.0229 | 624110 | 0.2856 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.4880 | 0.2789 | n/a | n/a | n/a | -0.0174 | 624110 | 0.2856 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | bin | 4 fps | 0.4921 | 0.2753 | n/a | n/a | n/a | -0.0267 | 624110 | 0.2856 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | main | 4 fps | 0.5605 | 0.2780 | n/a | n/a | n/a | 0.0637 | 110735 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_short | 4 fps | 0.5793 | 0.2979 | n/a | n/a | n/a | 0.1010 | 110735 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.4721 | 0.2192 | n/a | n/a | n/a | -0.0468 | 110735 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | normal | 4 fps | 0.4354 | 0.2064 | n/a | n/a | n/a | -0.0708 | 110735 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | main_pair | 4 fps | 0.5233 | 0.2518 | n/a | n/a | n/a | 0.0145 | 110735 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.5798 | 0.2870 | n/a | n/a | n/a | 0.0806 | 110735 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.5667 | 0.2797 | n/a | n/a | n/a | 0.0668 | 110735 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.5974 | 0.3192 | n/a | n/a | n/a | 0.1411 | 110735 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.4710 | 0.2146 | n/a | n/a | n/a | -0.0554 | 110735 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.5885 | 0.3107 | n/a | n/a | n/a | 0.1250 | 110735 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | bin | 4 fps | 0.5822 | 0.3062 | n/a | n/a | n/a | 0.1166 | 110735 | 0.2441 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | main | 4 fps | 0.5694 | 0.3006 | n/a | n/a | n/a | 0.0780 | 102153 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_short | 4 fps | 0.6002 | 0.3367 | n/a | n/a | n/a | 0.1382 | 102153 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.4619 | 0.2252 | n/a | n/a | n/a | -0.0476 | 102153 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | normal | 4 fps | 0.4421 | 0.2223 | n/a | n/a | n/a | -0.0525 | 102153 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | main_pair | 4 fps | 0.5602 | 0.2816 | n/a | n/a | n/a | 0.0464 | 102153 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.5742 | 0.3175 | n/a | n/a | n/a | 0.1062 | 102153 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.5980 | 0.3156 | n/a | n/a | n/a | 0.1030 | 102153 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.5922 | 0.3383 | n/a | n/a | n/a | 0.1409 | 102153 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.4665 | 0.2267 | n/a | n/a | n/a | -0.0452 | 102153 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.6102 | 0.3192 | n/a | n/a | n/a | 0.1090 | 102153 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | bin | 4 fps | 0.5724 | 0.3135 | n/a | n/a | n/a | 0.0996 | 102153 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ |  |
| LaGoVAD | 1 | HateClipSeg | all | aux-temporal-pretrain | base | main | 4 fps | 0.5433 | 0.4854 | n/a | n/a | n/a | 0.3194 | 374235 | 0.4642 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/395 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | all | aux-temporal-pretrain | base | sens_short | 4 fps | 0.5497 | 0.5150 | n/a | n/a | n/a | 0.7645 | 374235 | 0.4642 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/395 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | all | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.4408 | 0.4255 | n/a | n/a | n/a | -0.5817 | 374235 | 0.4642 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/395 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | all | aux-temporal-pretrain | base | normal | 4 fps | 0.4285 | 0.4140 | n/a | n/a | n/a | -0.7557 | 374235 | 0.4642 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/395 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | all | aux-temporal-pretrain | base | main_pair | 4 fps | 0.5290 | 0.4759 | n/a | n/a | n/a | 0.1765 | 374235 | 0.4642 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/395 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | all | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.5657 | 0.5108 | n/a | n/a | n/a | 0.7017 | 374235 | 0.4642 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/395 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | all | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.5476 | 0.5011 | n/a | n/a | n/a | 0.5554 | 374235 | 0.4642 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/395 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | all | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.5700 | 0.5413 | n/a | n/a | n/a | 1.1596 | 374235 | 0.4642 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/395 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | all | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.4456 | 0.4263 | n/a | n/a | n/a | -0.5702 | 374235 | 0.4642 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/395 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | all | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.5647 | 0.5381 | n/a | n/a | n/a | 1.1118 | 374235 | 0.4642 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/395 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | all | aux-temporal-pretrain | base | bin | 4 fps | 0.5785 | 0.5523 | n/a | n/a | n/a | 1.3251 | 374235 | 0.4642 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/395 (0.2%) dropped, not interpolated |

### M.3 HateClipSeg 6-class appendix (one forward over the six released classes, each scored against its own frame labels)

| method | wave | dataset | split | supervision | variant | query_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LaGoVAD | 1 | HateClipSeg | test | aux-temporal-pretrain | base | c0_normal | 4 fps | 0.5373 | 0.5500 | n/a | n/a | n/a | 1.5663 | 113002 | 0.5267 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/119 (0.8%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | test | aux-temporal-pretrain | base | c1_hateful | 4 fps | 0.6137 | 0.2462 | n/a | n/a | n/a | 0.1766 | 113002 | 0.2002 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/119 (0.8%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | test | aux-temporal-pretrain | base | c2_insulting | 4 fps | 0.5480 | 0.2755 | n/a | n/a | n/a | 0.1301 | 113002 | 0.2572 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/119 (0.8%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | test | aux-temporal-pretrain | base | c3_sexual | 4 fps | 0.5619 | 0.0831 | n/a | n/a | n/a | 0.1132 | 113002 | 0.0572 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/119 (0.8%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | test | aux-temporal-pretrain | base | c4_violence | 4 fps | 0.3970 | 0.1163 | n/a | n/a | n/a | -0.1112 | 113002 | 0.1367 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/119 (0.8%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | test | aux-temporal-pretrain | base | c5_harm | 4 fps | 0.1843 | 0.0067 | n/a | n/a | n/a | -0.0261 | 113002 | 0.0117 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/119 (0.8%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | all | aux-temporal-pretrain | base | c0_normal | 4 fps | 0.5688 | 0.5859 | n/a | n/a | n/a | 2.4745 | 374235 | 0.5358 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/395 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | all | aux-temporal-pretrain | base | c1_hateful | 4 fps | 0.6357 | 0.2817 | n/a | n/a | n/a | 0.2820 | 374235 | 0.2095 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/395 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | all | aux-temporal-pretrain | base | c2_insulting | 4 fps | 0.5301 | 0.2657 | n/a | n/a | n/a | 0.0626 | 374235 | 0.2566 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/395 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | all | aux-temporal-pretrain | base | c3_sexual | 4 fps | 0.5824 | 0.0740 | n/a | n/a | n/a | 0.1878 | 374235 | 0.0365 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/395 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | all | aux-temporal-pretrain | base | c4_violence | 4 fps | 0.4262 | 0.1118 | n/a | n/a | n/a | -0.0562 | 374235 | 0.1207 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/395 (0.2%) dropped, not interpolated |
| LaGoVAD | 1 | HateClipSeg | all | aux-temporal-pretrain | base | c5_harm | 4 fps | 0.3791 | 0.0040 | n/a | n/a | n/a | -0.0060 | 374235 | 0.0050 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 1/395 (0.2%) dropped, not interpolated |

### M.4 Stratified — single-span vs multi-span (HateMM / MHC only)

| method | wave | dataset | split | supervision | variant | query_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | main | 4 fps | 0.6120 | 0.2968 | n/a | n/a | n/a | 0.1746 | 93315 | 0.2007 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | main | 4 fps | 0.4689 | 0.1044 | n/a | n/a | n/a | 0.0007 | 92052 | 0.1042 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_short | 4 fps | 0.5759 | 0.2585 | n/a | n/a | n/a | 0.1050 | 93315 | 0.2007 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_short | 4 fps | 0.4202 | 0.0842 | n/a | n/a | n/a | -0.0665 | 92052 | 0.1042 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.5368 | 0.2217 | n/a | n/a | n/a | 0.0382 | 93315 | 0.2007 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.5765 | 0.1177 | n/a | n/a | n/a | 0.0449 | 92052 | 0.1042 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | normal | 4 fps | 0.4518 | 0.1869 | n/a | n/a | n/a | -0.0251 | 93315 | 0.2007 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | normal | 4 fps | 0.5393 | 0.1152 | n/a | n/a | n/a | 0.0366 | 92052 | 0.1042 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | main_pair | 4 fps | 0.6021 | 0.2929 | n/a | n/a | n/a | 0.1675 | 93315 | 0.2007 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | main_pair | 4 fps | 0.4360 | 0.0851 | n/a | n/a | n/a | -0.0635 | 92052 | 0.1042 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.6027 | 0.3121 | n/a | n/a | n/a | 0.2024 | 93315 | 0.2007 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.4147 | 0.0927 | n/a | n/a | n/a | -0.0381 | 92052 | 0.1042 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.5871 | 0.2850 | n/a | n/a | n/a | 0.1531 | 93315 | 0.2007 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.3828 | 0.0778 | n/a | n/a | n/a | -0.0876 | 92052 | 0.1042 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.5866 | 0.2833 | n/a | n/a | n/a | 0.1501 | 93315 | 0.2007 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.3954 | 0.0842 | n/a | n/a | n/a | -0.0665 | 92052 | 0.1042 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.5311 | 0.2320 | n/a | n/a | n/a | 0.0569 | 93315 | 0.2007 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.5488 | 0.1111 | n/a | n/a | n/a | 0.0228 | 92052 | 0.1042 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.6143 | 0.2846 | n/a | n/a | n/a | 0.1524 | 93315 | 0.2007 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.4379 | 0.0865 | n/a | n/a | n/a | -0.0588 | 92052 | 0.1042 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | bin | 4 fps | 0.5393 | 0.2046 | n/a | n/a | n/a | 0.0071 | 93315 | 0.2007 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | HateMM | test | aux-temporal-pretrain | base | bin | 4 fps | 0.4242 | 0.0888 | n/a | n/a | n/a | -0.0512 | 92052 | 0.1042 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | main | 4 fps | 0.5064 | 0.2422 | n/a | n/a | n/a | -0.0399 | 21669 | 0.2628 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | main | 4 fps | 0.8407 | 0.0739 | n/a | n/a | n/a | 0.0789 | 15037 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_short | 4 fps | 0.4977 | 0.2421 | n/a | n/a | n/a | -0.0400 | 21669 | 0.2628 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_short | 4 fps | 0.8435 | 0.1004 | n/a | n/a | n/a | 0.1238 | 15037 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.4247 | 0.2176 | n/a | n/a | n/a | -0.0874 | 21669 | 0.2628 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.4370 | 0.0221 | n/a | n/a | n/a | -0.0090 | 15037 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | normal | 4 fps | 0.3906 | 0.2113 | n/a | n/a | n/a | -0.0996 | 21669 | 0.2628 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | normal | 4 fps | 0.2524 | 0.0172 | n/a | n/a | n/a | -0.0173 | 15037 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | main_pair | 4 fps | 0.4789 | 0.2344 | n/a | n/a | n/a | -0.0550 | 21669 | 0.2628 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | main_pair | 4 fps | 0.7024 | 0.0412 | n/a | n/a | n/a | 0.0233 | 15037 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.5773 | 0.2857 | n/a | n/a | n/a | 0.0442 | 21669 | 0.2628 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.8071 | 0.0613 | n/a | n/a | n/a | 0.0576 | 15037 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.4928 | 0.2387 | n/a | n/a | n/a | -0.0465 | 21669 | 0.2628 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.7628 | 0.0625 | n/a | n/a | n/a | 0.0595 | 15037 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.5738 | 0.2814 | n/a | n/a | n/a | 0.0359 | 21669 | 0.2628 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.8342 | 0.0762 | n/a | n/a | n/a | 0.0828 | 15037 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.4135 | 0.2150 | n/a | n/a | n/a | -0.0925 | 21669 | 0.2628 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.4298 | 0.0218 | n/a | n/a | n/a | -0.0095 | 15037 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.5724 | 0.2981 | n/a | n/a | n/a | 0.0682 | 21669 | 0.2628 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.7716 | 0.0832 | n/a | n/a | n/a | 0.0947 | 15037 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | bin | 4 fps | 0.5949 | 0.3321 | n/a | n/a | n/a | 0.1340 | 21669 | 0.2628 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | test | aux-temporal-pretrain | base | bin | 4 fps | 0.7768 | 0.0574 | n/a | n/a | n/a | 0.0509 | 15037 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | main | 4 fps | 0.5965 | 0.3118 | n/a | n/a | n/a | 0.0717 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | main | 4 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12955 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span single-class pool, metrics undefined |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_short | 4 fps | 0.6432 | 0.3698 | n/a | n/a | n/a | 0.1605 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_short | 4 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12955 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span single-class pool, metrics undefined |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.4593 | 0.2303 | n/a | n/a | n/a | -0.0528 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_vad | 4 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12955 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span single-class pool, metrics undefined |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | normal | 4 fps | 0.4303 | 0.2208 | n/a | n/a | n/a | -0.0673 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | normal | 4 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12955 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span single-class pool, metrics undefined |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | main_pair | 4 fps | 0.5911 | 0.3128 | n/a | n/a | n/a | 0.0733 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | main_pair | 4 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12955 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span single-class pool, metrics undefined |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.5786 | 0.2981 | n/a | n/a | n/a | 0.0508 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | main_vsnormal | 4 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12955 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span single-class pool, metrics undefined |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.6479 | 0.3523 | n/a | n/a | n/a | 0.1337 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_short_pair | 4 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12955 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span single-class pool, metrics undefined |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.6117 | 0.3349 | n/a | n/a | n/a | 0.1070 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12955 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span single-class pool, metrics undefined |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.4717 | 0.2362 | n/a | n/a | n/a | -0.0438 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12955 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span single-class pool, metrics undefined |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.6480 | 0.3744 | n/a | n/a | n/a | 0.1674 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12955 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span single-class pool, metrics undefined |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | bin | 4 fps | 0.5673 | 0.2895 | n/a | n/a | n/a | 0.0377 | 18199 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | test | aux-temporal-pretrain | base | bin | 4 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12955 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span single-class pool, metrics undefined |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | main | 4 fps | 0.5248 | 0.2724 | n/a | n/a | n/a | 0.0382 | 528575 | 0.2520 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=single_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | main | 4 fps | 0.5216 | 0.1268 | n/a | n/a | n/a | 0.0744 | 455032 | 0.0991 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=multi_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_short | 4 fps | 0.4937 | 0.2513 | n/a | n/a | n/a | -0.0012 | 528575 | 0.2520 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=single_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_short | 4 fps | 0.4885 | 0.1000 | n/a | n/a | n/a | 0.0024 | 455032 | 0.0991 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=multi_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.5144 | 0.2644 | n/a | n/a | n/a | 0.0233 | 528575 | 0.2520 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=single_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.5155 | 0.1064 | n/a | n/a | n/a | 0.0197 | 455032 | 0.0991 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=multi_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | normal | 4 fps | 0.4951 | 0.2717 | n/a | n/a | n/a | 0.0368 | 528575 | 0.2520 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=single_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | normal | 4 fps | 0.5057 | 0.1235 | n/a | n/a | n/a | 0.0655 | 455032 | 0.0991 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=multi_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | main_pair | 4 fps | 0.5210 | 0.2740 | n/a | n/a | n/a | 0.0412 | 528575 | 0.2520 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=single_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | main_pair | 4 fps | 0.5007 | 0.1164 | n/a | n/a | n/a | 0.0464 | 455032 | 0.0991 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=multi_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.5116 | 0.2752 | n/a | n/a | n/a | 0.0434 | 528575 | 0.2520 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=single_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.5017 | 0.1063 | n/a | n/a | n/a | 0.0193 | 455032 | 0.0991 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=multi_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.4892 | 0.2615 | n/a | n/a | n/a | 0.0179 | 528575 | 0.2520 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=single_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.4688 | 0.0956 | n/a | n/a | n/a | -0.0095 | 455032 | 0.0991 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=multi_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.4918 | 0.2756 | n/a | n/a | n/a | 0.0442 | 528575 | 0.2520 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=single_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.4777 | 0.0965 | n/a | n/a | n/a | -0.0069 | 455032 | 0.0991 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=multi_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.5083 | 0.2642 | n/a | n/a | n/a | 0.0229 | 528575 | 0.2520 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=single_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.5060 | 0.1078 | n/a | n/a | n/a | 0.0233 | 455032 | 0.0991 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=multi_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.4925 | 0.2537 | n/a | n/a | n/a | 0.0033 | 528575 | 0.2520 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=single_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.4630 | 0.0868 | n/a | n/a | n/a | -0.0330 | 455032 | 0.0991 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=multi_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | bin | 4 fps | 0.4929 | 0.2439 | n/a | n/a | n/a | -0.0151 | 528575 | 0.2520 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=single_span |
| LaGoVAD | 1 | HateMM | all | aux-temporal-pretrain | base | bin | 4 fps | 0.4829 | 0.0930 | n/a | n/a | n/a | -0.0165 | 455032 | 0.0991 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | missing 2/1083 (0.2%) dropped, not interpolated; stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | main | 4 fps | 0.5521 | 0.2626 | n/a | n/a | n/a | 0.0443 | 108460 | 0.2379 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | main | 4 fps | 0.8148 | 0.0711 | n/a | n/a | n/a | 0.1062 | 78208 | 0.0157 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_short | 4 fps | 0.5726 | 0.2841 | n/a | n/a | n/a | 0.0831 | 108460 | 0.2379 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_short | 4 fps | 0.7759 | 0.0529 | n/a | n/a | n/a | 0.0714 | 78208 | 0.0157 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.4703 | 0.2134 | n/a | n/a | n/a | -0.0442 | 108460 | 0.2379 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.5197 | 0.0146 | n/a | n/a | n/a | -0.0020 | 78208 | 0.0157 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | normal | 4 fps | 0.4352 | 0.2014 | n/a | n/a | n/a | -0.0658 | 108460 | 0.2379 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | normal | 4 fps | 0.3997 | 0.0117 | n/a | n/a | n/a | -0.0075 | 78208 | 0.0157 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | main_pair | 4 fps | 0.5144 | 0.2381 | n/a | n/a | n/a | 0.0002 | 108460 | 0.2379 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | main_pair | 4 fps | 0.7834 | 0.0575 | n/a | n/a | n/a | 0.0800 | 78208 | 0.0157 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.5736 | 0.2764 | n/a | n/a | n/a | 0.0692 | 108460 | 0.2379 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.7833 | 0.0401 | n/a | n/a | n/a | 0.0467 | 78208 | 0.0157 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.5596 | 0.2663 | n/a | n/a | n/a | 0.0511 | 108460 | 0.2379 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.7681 | 0.0469 | n/a | n/a | n/a | 0.0597 | 78208 | 0.0157 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.5918 | 0.3087 | n/a | n/a | n/a | 0.1274 | 108460 | 0.2379 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.7823 | 0.0403 | n/a | n/a | n/a | 0.0473 | 78208 | 0.0157 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.4686 | 0.2085 | n/a | n/a | n/a | -0.0531 | 108460 | 0.2379 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.5308 | 0.0150 | n/a | n/a | n/a | -0.0013 | 78208 | 0.0157 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.5839 | 0.3017 | n/a | n/a | n/a | 0.1148 | 108460 | 0.2379 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.7793 | 0.0376 | n/a | n/a | n/a | 0.0421 | 78208 | 0.0157 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | bin | 4 fps | 0.5788 | 0.2984 | n/a | n/a | n/a | 0.1089 | 108460 | 0.2379 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-EN | all | aux-temporal-pretrain | base | bin | 4 fps | 0.6997 | 0.0287 | n/a | n/a | n/a | 0.0249 | 78208 | 0.0157 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | main | 4 fps | 0.5691 | 0.2998 | n/a | n/a | n/a | 0.0754 | 101620 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | main | 4 fps | 0.8778 | 0.0155 | n/a | n/a | n/a | 0.0737 | 72513 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_short | 4 fps | 0.6003 | 0.3374 | n/a | n/a | n/a | 0.1369 | 101620 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_short | 4 fps | 0.8971 | 0.0181 | n/a | n/a | n/a | 0.0875 | 72513 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.4623 | 0.2254 | n/a | n/a | n/a | -0.0465 | 101620 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_vad | 4 fps | 0.3688 | 0.0014 | n/a | n/a | n/a | -0.0028 | 72513 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | normal | 4 fps | 0.4424 | 0.2224 | n/a | n/a | n/a | -0.0513 | 101620 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | normal | 4 fps | 0.3025 | 0.0012 | n/a | n/a | n/a | -0.0035 | 72513 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | main_pair | 4 fps | 0.5598 | 0.2808 | n/a | n/a | n/a | 0.0443 | 101620 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | main_pair | 4 fps | 0.8070 | 0.0093 | n/a | n/a | n/a | 0.0404 | 72513 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.5738 | 0.3168 | n/a | n/a | n/a | 0.1032 | 101620 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | main_vsnormal | 4 fps | 0.8417 | 0.0119 | n/a | n/a | n/a | 0.0543 | 72513 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.5979 | 0.3152 | n/a | n/a | n/a | 0.1006 | 101620 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_short_pair | 4 fps | 0.8891 | 0.0115 | n/a | n/a | n/a | 0.0522 | 72513 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.5919 | 0.3383 | n/a | n/a | n/a | 0.1384 | 101620 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_short_vsnormal | 4 fps | 0.8653 | 0.0164 | n/a | n/a | n/a | 0.0785 | 72513 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.4669 | 0.2268 | n/a | n/a | n/a | -0.0442 | 101620 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_vad_pair | 4 fps | 0.3738 | 0.0014 | n/a | n/a | n/a | -0.0028 | 72513 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.6104 | 0.3194 | n/a | n/a | n/a | 0.1074 | 101620 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | sens_vad_vsnormal | 4 fps | 0.8310 | 0.0055 | n/a | n/a | n/a | 0.0198 | 72513 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | bin | 4 fps | 0.5718 | 0.3135 | n/a | n/a | n/a | 0.0978 | 101620 | 0.2538 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=single_span |
| LaGoVAD | 1 | MHC-ZH | all | aux-temporal-pretrain | base | bin | 4 fps | 0.8153 | 0.0102 | n/a | n/a | n/a | 0.0450 | 72513 | 0.0019 | 1 | n/a | §4 | idea-stage/repro_lagovad/ | stratum=multi_span |

### M.5 What the numbers say

1. **LaGoVAD clears the random floor on three of the four corpora, and by very little.** With the main
   definition on the test split, frame ROC-AUC is 0.5579 (HateMM), 0.5239 (MHC-EN), 0.5965 (MHC-ZH),
   0.5000 (HateClipSeg), against a random floor of 0.500 and a gold-broadcast ceiling of 0.8857 /
   0.9427 / 0.9842 / 0.6260. Oracle-normalised AP for the same rows is 0.184 / −0.024 / 0.072 /
   −0.092: the method recovers under a fifth of the chance-to-video-oracle gap on its best dataset
   and none of it on two others. On HateClipSeg the main definition is exactly at chance to four
   decimal places.

2. **The free-text definition is not what carries the signal, and on two datasets it is not even the
   best row.** LaGoVAD's premise is that a written definition selects the anomaly at inference. On
   MHC-EN the strongest test row is `bin` — the *binary* anomaly head, which takes no text at all —
   at ROC 0.6058 / AP 0.3490 / AP_norm 0.1534, ahead of every one of the ten text rows. On
   HateClipSeg `bin` is again the top row (ROC 0.5431, AP_norm 1.0402). A text-free head beating
   every textual query is evidence that what transfers here is the checkpoint's generic
   surveillance-anomaly prior, not the hate definition we wrote.

3. **Definition wording moves the number more than the choice of method does.** Across the four
   test-split datasets the spread between the three hate definitions is 0.52–0.64 ROC on MHC-ZH
   (`sens_short` 0.6432 vs `sens_vad` 0.4593, a 0.18 swing) and 0.42–0.52 on MHC-EN. The three
   definitions describe the same concept in different registers; a method whose output swings that
   far on paraphrase is not reading the definition so much as reacting to its surface form. All
   three are reported precisely so this is visible; none was chosen after the fact.

4. **The `normal` reference row scores *below* chance on three datasets** (0.4879 / 0.3844 / 0.4303
   / 0.4680), which is the one internally consistent signal in the table: the similarity to a
   "nothing unusual happening" definition is anti-correlated with the hateful frames, as it should
   be. It is also why the `_vsnormal` contrast rows beat their solo counterparts on MHC-EN
   (0.5885 vs 0.5239) and HateClipSeg (0.5261 vs 0.5000) — subtracting the normal row removes a
   video-level offset that the raw similarity carries.

5. **Full corpus is not kinder than the test split.** On the full corpora the main definition reads
   0.5225 / 0.5605 / 0.5694 / 0.5433, i.e. HateMM drops from 0.5579 to 0.5225 while HateClipSeg
   rises from 0.5000 to 0.5433. The test-split and full-corpus numbers differ by up to 0.043 on the
   same variant, which is worth remembering before reading any single figure in this section as
   stable.

6. **Consequence for the campaign.** LaGoVAD is the first `aux-temporal-pretrain` entry in the table
   and the first method whose whole design is "define the anomaly in words". On these corpora it
   lands between the Wave 0 floors — above Qwen2.5-VL's 0.50–0.52, below ZS-ImageBind's 0.59
   everywhere except MHC-ZH — while its own text-free head is competitive with its text rows. The
   honest reading is that a definition-conditioned VAD checkpoint trained on XD-Violence does not
   transfer its language conditioning to hate speech, and that the gold-broadcast ceiling remains
   the only comparison in this table with real headroom in it.

**Recorded for completeness:** while wiring the evaluator's `curves` front-end, a two-video plumbing
check on HateClipSeg printed AUCs before the full run. No method setting was changed after it — the
definitions, the grouping and the score convention were already frozen and committed at `aea5e5f`
and `1eb5366` — but freeze §10 red line 3 asks smoke tests to check shape and range only, so the
deviation is stated rather than left unmentioned.

### M.6 Harness defects found while running this method, and what was changed

Recorded here rather than in a commit message alone, because three of them
affected rows other than LaGoVAD's.

| # | Defect | Effect | Fix |
|---|---|---|---|
| 1 | `openai/clip-vit-base-patch16` cached with the right byte count and wrong content (max \|w\| = 3.7 × 10¹⁹) | CLIP returned **one identical embedding for every frame of every video**; `MODEL_ASSETS_STATUS §3.1` recorded the resulting flat curves as a property of LaGoVAD's binary head | `audit_hf_cache.sh` re-hashes every blob against its own filename (an HF blob is stored under its sha256): 7 of 55 weight files corrupt, all repaired and re-verified by `hf_refetch.py` |
| 2 | The in-flight crash marker could not distinguish a decoder crash from an operator `SIGTERM` or an OOM-kill | Each stop silently retired one healthy video. **12 of the 14** ids the Wave 0 Qwen row excludes decode cleanly | An id is retired only after taking the process down **twice** (freeze §12 **D3**) |
| 3 | Runners had no `set -e`; the AV²A supervisor printed the smoke's exit code without checking it and emitted `RUN COMPLETE` on driver `rc=0` regardless of output | A chain could report success it had not earned — as the LAVAD chain did, exiting `rc=0` having written 5 curves for one dataset | `set -euo pipefail` plus per-dataset curve-count guards in the LaGoVAD chain, the UniTime converter and the AV²A supervisor; each verified to pass on real data and fail on a truncation |
| 4 | `decord` cannot open 25.5% of MHC-EN containers | A quarter of that dataset would have been recorded as method failures | `decord_fallback.py` tries the real reader first and falls back to PyAV; 275 of 3,084 containers need it |

**A retracted claim, kept visible because the mistake is instructive.** While
diagnosing URF's OOM I read VideoLLaMA3's `VisionAttention`, saw it add a *bool*
mask to float logits (`True` → `1.0`), measured that 44–56% of a row's attention
mass escapes its own block, and reported that our forced flash → sdpa adaptation
had changed the model's semantics. **That was wrong.** The class that actually
runs is `VisionSdpaAttention`, a *subclass* selected by
`VISION_ATTENTION_CLASSES[config._attn_implementation]`; it builds the same bool
tensor but **passes** it to `F.scaled_dot_product_attention` as `attn_mask`
instead of adding it, and for SDPA a bool mask means "True = attend, False =
−inf". Verified here: sdpa-with-bool-mask reproduces independently computed
per-block attention to **0.000e+00**, and the URF worker measured it against the
published flash path at 4.44e-16. So the sdpa path is genuinely block-diagonal,
there is no fidelity choice to make, and §L needs only "no sm_120 flash wheel, so
sdpa; semantics identical". The eager class I analysed is dead code.

The error is worth recording because it is the same species as the defects in the
table: I inferred behaviour from reading source and reported it as a measurement,
without checking which class the config actually instantiates. The memory patch I
wrote off the back of it was removed — it targeted the unused class, and its
`type(m).__name__ == "VisionAttention"` test would not have matched the subclass
anyway, so `install()` would have returned 0 while reporting success. That
silent-no-op shape is now itself a check (`patch_applied`).

**The pattern worth carrying forward.** Defects 1 and 2 are *silent correctness*
failures: right shape, right range, wrong content, no exception, clean exit code.
None is caught by asking "did it run?". The question that separates them from a
working component is whether the output **varies with the input**, which a
collapsed encoder cannot do.
`scripts/repro_campaign/discrimination_check.py` makes that a cheap smoke-test
assertion (`curve_varies`, `embeddings_discriminate`, `scores_separate_items`).
Replayed against the real corrupt-CLIP failure it fires on both the collapsed
embeddings (mean off-diagonal cosine 1.000000) and the resulting flat curve, and
it passes on the repaired LaGoVAD output — while deliberately *not* failing
Qwen's 19% modal answer, which is a finding rather than a fault.

**LaGoVAD's own artifacts were re-verified after all of the above**: 3,081 curves
of 3,084, the three missing being exactly the two audio-only HateMM containers
and the truncated `yt_NzvfkIYS5Yg`; test pools full at 215 / 161 / 149 / 118. No
row in §M is affected by any of these defects.

## K. Method as run — LAVAD (CVPR 2024)

The harness is `scripts/repro_campaign/lavad_chain.py` (stages 02-06) on top of
`scripts/repro_campaign/blip2_caption.py` (stage 01) and
`scripts/repro_campaign/extract_frames_1fps.py` (stage 00), driven by
`scripts/repro_campaign/run_lavad_wave1.sh`. Every prompt, window geometry, dedup rule,
neighbour count and weighting is taken from `third_party/lavad` @ `1ad46c66`; what changed is
listed below, and nothing else did.

**The chain, stage for stage.** `00_extract_frames.sh` writes JPEG frames; `01_caption.sh`
captions each frame with BLIP-2; `02_create_index.sh` builds a per-video ImageBind **text** index
over the captions; `03_clean_captions.sh` re-assigns each frame the caption whose ImageBind text
embedding is nearest that frame's ImageBind **image** embedding, giving a nested
`{center: {frame: cleaned caption}}`; `04_query_llm.sh` runs Llama-2-13b-chat twice, first to
summarise the ten cleaned captions of a 10 s window into one sentence, then to score that summary
0-1 with the law-enforcement prompt; `05_create_summary_index.sh` indexes the summaries;
`06_refine_anomaly_scores.sh` embeds each 10 s window with ImageBind's **video** path, retrieves
the ten nearest summaries in the same video, and `src/eval.py` combines their scores with
`softmax(similarity)` weights before `np.repeat`-ing each centre's score across the interval it
represents. That final curve is what is scored here.

**Adaptation 1 — a 1 fps frame grid, our stand-in for `frame_interval=16`.** LAVAD extracts every
native frame: `MODEL_ASSETS_STATUS §2` measured 283 MB of JPEG for one 222 s video, so the four
corpora would need 700-800 GB, which does not fit beside the other campaign jobs. Frames are
instead extracted at `fps=1`, so `frames/<DS>/<vid>/000123.jpg` is the content at `t = 123 s` and
one caption exists per second. Centres are every captioned frame, i.e. `native_rate = 1.0`
samples/s against LAVAD's 1.875/s (16 frames at its assumed 30 fps). The 10 s clip window and its
10 uniform samples are unchanged, and at 1 fps those 10 samples are exactly the 10 frames of the
window — which is why the same JPEGs serve stages 01, 03 and 06 without a second decode. Total
cost on disk: **4.0 GB, not 700-800 GB.**

**Extraction path — ffmpeg, and no video was lost to a decoder.** Frames come from
`ffmpeg -vf fps=1 -q:v 2`, not decord. This matters because decord cannot open a substantial share
of the released MultiHateClip containers — 27% of MHC-EN on the sample the Wave 0 Qwen2.5-VL row
measured. ffmpeg opened **215/215 HateMM, 161/161 MHC-EN, 149/149 MHC-ZH and 118/119 HateClipSeg**
test videos. The single loss, `yt_NzvfkIYS5Yg`, is a truncated source file that ffmpeg itself
rejects (exit 183); that is a media fault, not a decoder choice.

**Adaptation 2 — a single captioner, and what it costs (the `§3.11b` decision).**
`01_caption.sh` lists five BLIP-2 variants and `02_create_index.sh` indexes all five; only
`Salesforce/blip2-opt-6.7b-coco` is on disk, the other four being ~120 GB of download and a 5x
multiplier on both the captioning and the indexing stages. **Decision: run the chain over the one
captioner and label every row `single-captioner`.** The likely effect on the LELA alignment is
worth stating precisely rather than hand-waving, because it is the largest single departure in
this port. Stage 03's retrieval pool shrinks from up to five candidate captions per frame to one,
so caption *cleaning* can still substitute a caption across **frames** — which is the mechanism
LAVAD's ablation credits most of the gain to — but no longer across **captioners**. The
ensemble's role in the published pipeline is to widen the candidate set that the image embedding
chooses from; with one captioner the cleaning step becomes "pick the moment in this video whose
caption best matches this frame" rather than "pick the best of five descriptions of the best
matching moment". We therefore expect a *lower* score than the published one if the ensemble
contributes, and we should not read a shortfall on the §7 check as evidence of a coding error
before checking this. If a row misses tolerance, this and the evaluation pool (below) are the two
explanations to rule out first.

**Adaptation 3 — greedy decoding, so the run is deterministic.** The shipped `04_query_llm.sh`
never passes `--temperature`, so it inherits the `0.6` default, i.e. sampling; under freeze §6
that would make LAVAD a three-seed method and triple an already tight single-GPU budget. Decoding
is greedy (`temperature = 0`, which is also Meta's own `chat_completion` semantics for
`temperature == 0`), the run is a single run, and the §7 transplant comparison is not contaminated
by decode luck. Greedy decoding also makes the content-keyed generation cache exactly lossless:
identical prompts must give identical generations, so caching them is a speed-up and not an
approximation.

**Adaptation 4 — the `llama_hf` shim and the ungated mirror.** Both `Llama.build` call sites are
redirected to `scripts/repro_campaign/shim/llama_hf` (`MODEL_ASSETS_STATUS §3.2`), which maps
`libs/llama/llama-2-13b-chat/` to `NousResearch/Llama-2-13b-chat-hf` because every `meta-llama/*`
repo is gated and no token is configured here (§3.8). That mirror ships no `tokenizer.chat_template`,
so the shim falls back to the canonical Llama-2 `[INST] <<SYS>>…` layout, which is byte-for-byte
what Meta's own `ChatFormat` emits for a `[system, user]` pair (§3.12) — verified on this machine.
The 13B runs in **NF4** (~7.5 GiB) rather than bf16 (26 GiB), because bf16 weights plus a batch-32
KV cache do not fit a 32 GiB card. Meta caps generation at `total_len = min(max_seq_len,
max_gen_len + max_prompt_len)`; that is reproduced exactly, with the repo's own `max_seq_len = 512`.

**Adaptation 5 — FAISS replaced by an exact matrix product.** `IndexFlatIP` over `normalize_L2`-ed
vectors *is* cosine similarity, so a per-video matmul with `argsort` returns the same neighbours
in the same order without writing index files. Two further changes are exact-equivalence
speed-ups, each verified numerically rather than assumed:
  * ImageBind's video loader emits 15 sub-clips per window, but only **3 are distinct** — the
    `ConstantClipsPerVideoSampler` asks for 5 clips of 2 s from a 10-frame "video" that
    `FrameVideo` calls 0.33 s long, so all 5 clips are the same frames and only `SpatialCrop`'s 3
    crops differ (equivalence classes `{0,3,6,9,12}`, `{1,4,7,10,13}`, `{2,5,8,11,14}`). ImageBind
    reduces the clip axis with `mean(dim=1)`, so averaging the 3 distinct crops equals averaging
    all 15, at a fifth of the cost.
  * `UniformTemporalSubsample(2)` keeps `linspace(0, T-1, 2)`, i.e. the **first and last frame**
    of each window, so handing the loader those two frames gives a bit-identical tensor
    (max |diff| = 0.0, checked on real frames) while reading 2 JPEGs per window instead of 10.

**Adaptation 6 — refusals are masked, never interpolated** (`MODEL_ASSETS_STATUS §3.11a`,
mandatory items 1 and 2). LAVAD's `_parse_score` turns anything without a `[x]` into `-1`, and its
`_interpolate_unmatched_scores` then runs `np.interp` over the remaining points, so a refusal
becomes a linear blend of its neighbours and nothing in the output distinguishes it from a real
score. **`np.interp` is never called here.** Every `-1` is recorded with its raw generation text
in `data/lavad/score_refusals/`, the frame is left unscored, and the shared evaluator drops exactly
those frames and reports `coverage` beside every AUC. Refusal rates are broken down by GT label in
§K.5, which is the breakdown that matters: refusals concentrate on violent and group-hostile
descriptions, which in these corpora are the positive class.

Note carefully what this does and does not do to the two variants. The `raw` row is the stage-04b
score with refused frames dropped, so its coverage is below 1. The `base` row is LAVAD's *own*
stage-06 refinement, which replaces **every** centre's score — refused or not — with a
similarity-weighted average of the ten visually nearest summaries' scores; a refused frame
therefore still has a defined refined score, computed only over the neighbours that did answer.
That is the published mechanism, not a harness patch, and it is why `base` coverage is 1.000 while
`raw` coverage is not.

**Adaptation 7 — the evaluation pool is the test split, not the full corpus (declared deviation).**
The four corpora are 84 h of video and the two Llama passes alone are ~135k generations at 1 fps;
running LAVAD *and* URF-HVAA over all splits does not fit the single shared RTX 5090. Both methods
are therefore evaluated on the **frozen test split only** — 644 videos, 18.9 h, 67,647 one-second
centres — which is what freeze §5 calls the headline anyway ("Headline table = test split"), so
no rule is broken; but there is **no full-corpus row for LAVAD or URF**, unlike the Wave 0
methods. This matters for §7: LELA does not say which pool it evaluated, and ZS-CLIP §A/§B already
show the test and full-corpus pools can differ by up to 0.05 on the same method and dataset
(MHC-ZH ROC 0.6075 test vs 0.5611 full). **A miss on the §7 check must therefore be read against
the pool difference before it is read as a bad transplant.**

**Nothing else changed.** `context_prompt`, `format_prompt` and `summary_prompt` are byte-for-byte
the strings in `04_query_llm.sh`; `clip_duration = 10`, `num_samples = 10`, `num_neighbors = 10`,
`index_dim = 1024`, `batch_size = 32` are the shipped values; the captioner is unconditional with
the checkpoint's own generation config, as `image_captioner.py` runs it.

**One upstream bug we did not inherit.** `06_refine_anomaly_scores.sh` asks FAISS for 10
neighbours unconditionally. A video with fewer than 10 *distinct* summaries gets `-1` back for the
missing ones, and LAVAD's `file_names[-1]` then silently selects the last index entry. We clamp
the neighbour count to the index size and renormalise the weights instead; the number of videos
where the clamp bites is reported in §K.5.

**Run record.** One RTX 5090, single run, greedy. Stage 01 BLIP-2: **643 videos, 67,647 captions,
43.6 min at 25.9 img/s**; one video lost, `yt_NzvfkIYS5Yg`, a truncated source ffmpeg rejects.
Stage 02+03 ImageBind: 643 videos, 30.4 min. Stage 04a Llama-2-13b-chat NF4: 637 videos,
**66,477 dialogs, 47,102 generations, 19,375 served from the prompt cache** (29%), 0 truncations,
0 OOM, 260.7 min. Stage 04b: 638 videos, 66,666 dialogs, 37,262 generations, 29,404 cached (44%),
67.8 min. Stage 05+06 ImageBind: 637 videos, 96.0 min, **145 videos had fewer than 10 distinct
summaries** and had their neighbour count clamped. Productive GPU time **8.3 h**.

**What looked wrong, and was fixed before the numbers were produced.** A first attempt at batch 64
died of CUDA OOM inside stage 04a; the runner's `|| echo` then let stages 04b-06 run on the 6
videos that had made it through, and the chain exited 0 having written 5 curves for one dataset.
The run reported here is a clean re-run at batch 48 with `expandable_segments`. Three guards were
added first: a failed stage now exits the chain, `curves` raises when any dataset yields zero
curves, and both scorers halve the batch on OOM and retry (greedy decoding makes the result
independent of how the work is split, so batch size can no longer change a number or wedge a run).

Reproduce: `bash scripts/repro_campaign/run_lavad_wave1.sh test 1`, then
`python scripts/repro_campaign/eval_frame.py --method curves --curve-dir idea-stage/repro_lavad/curves --method-name LAVAD --variants base,raw --split test`.

### K.1 Headline rows — test split

`base` is the full published chain (stage-06 refined score). `raw` is the stage-04b LLM score
before refinement — LAVAD's own ablation, reported as a diagnostic, not as a competing row.

| method | wave | dataset | split | supervision | variant | query_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GOLD_BROADCAST | — | HateMM | test | control | control | n/a | video | 0.8857 | 0.5829 | n/a | n/a | n/a | 1.0000 | 116975 | 0.2421 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | HateMM | test | control | control | n/a | 4 fps | 0.5003 ± 0.0019 | 0.2423 ± 0.0013 | n/a | n/a | n/a | 0.0000 | 116975 | 0.2421 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| GOLD_BROADCAST | — | MHC-EN | test | control | control | n/a | video | 0.9427 | 0.7664 | n/a | n/a | n/a | 1.0000 | 22337 | 0.2734 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | MHC-EN | test | control | control | n/a | 4 fps | 0.5004 ± 0.0034 | 0.2737 ± 0.0026 | n/a | n/a | n/a | 0.0000 | 22337 | 0.2734 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| GOLD_BROADCAST | — | MHC-ZH | test | control | control | n/a | video | 0.9842 | 0.9191 | n/a | n/a | n/a | 1.0000 | 18199 | 0.2648 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | MHC-ZH | test | control | control | n/a | 4 fps | 0.4985 ± 0.0052 | 0.2646 ± 0.0038 | n/a | n/a | n/a | 0.0000 | 18199 | 0.2648 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| GOLD_BROADCAST | — | HateClipSeg | test | control | control | n/a | video | 0.6260 | 0.5437 | n/a | n/a | n/a | 1.0000 | 114097 | 0.4712 | 1 | n/a | §4+D1 | idea-stage/repro_campaign/ | zero-temporal-resolution ceiling, full GT pool |
| RANDOM_UNIFORM | — | HateClipSeg | test | control | control | n/a | 4 fps | 0.5009 ± 0.0021 | 0.4721 ± 0.0016 | n/a | n/a | n/a | 0.0000 | 114097 | 0.4712 | 20 | n/a | §4 | idea-stage/repro_campaign/ | U(0,1) per frame, 20 seeds, full GT pool |
| LAVAD | 1 | HateMM | test | label-free | base | base | 1 fps | 0.5587 | 0.2909 | n/a | n/a | n/a | 0.1424 | 116841 | 0.2424 | 1 | OUT_OF_TOLERANCE | §4 | idea-stage/repro_lavad/ | single-captioner |
| LAVAD | 1 | HateMM | test | label-free | base | raw | 1 fps | 0.5040 | 0.2445 | n/a | n/a | n/a | 0.0123 | 115553 | 0.2403 | 1 | OUT_OF_TOLERANCE | §4 | idea-stage/repro_lavad/ | single-captioner; coverage=0.989 |
| LAVAD | 1 | MHC-EN | test | label-free | base | base | 1 fps | 0.5559 | 0.3107 | n/a | n/a | n/a | 0.0680 | 22099 | 0.2760 | 1 | OUT_OF_TOLERANCE | §4 | idea-stage/repro_lavad/ | single-captioner; missing 1/161 (0.6%) dropped, not interpolated |
| LAVAD | 1 | MHC-EN | test | label-free | base | raw | 1 fps | 0.5077 | 0.2801 | n/a | n/a | n/a | 0.0103 | 21971 | 0.2749 | 1 | OUT_OF_TOLERANCE | §4 | idea-stage/repro_lavad/ | single-captioner; coverage=0.994; missing 1/161 (0.6%) dropped, not interpolated |
| LAVAD | 1 | MHC-ZH | test | label-free | base | base | 1 fps | 0.4923 | 0.2634 | n/a | n/a | n/a | -0.0022 | 18150 | 0.2648 | 1 | OUT_OF_TOLERANCE | §4 | idea-stage/repro_lavad/ | single-captioner |
| LAVAD | 1 | MHC-ZH | test | label-free | base | raw | 1 fps | 0.5159 | 0.2737 | n/a | n/a | n/a | 0.0133 | 18127 | 0.2650 | 1 | OUT_OF_TOLERANCE | §4 | idea-stage/repro_lavad/ | single-captioner; coverage=0.999 |
| LAVAD | 1 | HateClipSeg | test | label-free | base | base | 1 fps | 0.5768 | 0.5464 | n/a | n/a | n/a | 0.9980 | 112883 | 0.4730 | 1 | n/a | §4 | idea-stage/repro_lavad/ | single-captioner; coverage=1.000; missing 1/119 (0.8%) dropped, not interpolated |
| LAVAD | 1 | HateClipSeg | test | label-free | base | raw | 1 fps | 0.5453 | 0.5018 | n/a | n/a | n/a | 0.4699 | 111243 | 0.4671 | 1 | n/a | §4 | idea-stage/repro_lavad/ | single-captioner; coverage=0.985; missing 1/119 (0.8%) dropped, not interpolated |

### K.2 Full corpus — not run, and why

There is **no full-corpus row for LAVAD**. Freeze §5 makes the test split the headline and does not
require a full-corpus evaluation per method, so this breaks no rule, but it does differ from the
Wave 0 sections, which report both. The two Llama passes at a 1 s grid are ~133k dialogs on the
test split alone and took 5.5 h of the 8.3 h; the full corpus is 4.5x that for LAVAD and again for
URF-HVAA, which does not fit one shared GPU. The consequence for §7 is spelled out in K.4.

### K.3 Stratified — single-span vs multi-span (HateMM / MHC only)

| method | wave | dataset | split | supervision | variant | query_set | native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | gt_convention | run_dir | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LAVAD | 1 | HateMM | test | label-free | base | base | 1 fps | 0.5657 | 0.2540 | n/a | n/a | n/a | 0.0963 | 93190 | 0.2010 | 1 | n/a | §4 | idea-stage/repro_lavad/ | single-captioner; stratum=single_span |
| LAVAD | 1 | HateMM | test | label-free | base | base | 1 fps | 0.5844 | 0.1428 | n/a | n/a | n/a | 0.1277 | 91945 | 0.1043 | 1 | n/a | §4 | idea-stage/repro_lavad/ | single-captioner; stratum=multi_span |
| LAVAD | 1 | HateMM | test | label-free | base | raw | 1 fps | 0.4938 | 0.2005 | n/a | n/a | n/a | 0.0016 | 92170 | 0.1996 | 1 | n/a | §4 | idea-stage/repro_lavad/ | single-captioner; coverage=0.989; stratum=single_span |
| LAVAD | 1 | HateMM | test | label-free | base | raw | 1 fps | 0.5372 | 0.1130 | n/a | n/a | n/a | 0.0356 | 91577 | 0.1023 | 1 | n/a | §4 | idea-stage/repro_lavad/ | single-captioner; coverage=0.989; stratum=multi_span |
| LAVAD | 1 | MHC-EN | test | label-free | base | base | 1 fps | 0.5579 | 0.3022 | n/a | n/a | n/a | 0.0686 | 21433 | 0.2653 | 1 | n/a | §4 | idea-stage/repro_lavad/ | single-captioner; missing 1/161 (0.6%) dropped, not interpolated; stratum=single_span |
| LAVAD | 1 | MHC-EN | test | label-free | base | base | 1 fps | 0.5774 | 0.0317 | n/a | n/a | n/a | 0.0072 | 15012 | 0.0274 | 1 | n/a | §4 | idea-stage/repro_lavad/ | single-captioner; missing 1/161 (0.6%) dropped, not interpolated; stratum=multi_span |
| LAVAD | 1 | MHC-EN | test | label-free | base | raw | 1 fps | 0.5035 | 0.2680 | n/a | n/a | n/a | 0.0073 | 21309 | 0.2641 | 1 | n/a | §4 | idea-stage/repro_lavad/ | single-captioner; coverage=0.994; missing 1/161 (0.6%) dropped, not interpolated; stratum=single_span |
| LAVAD | 1 | MHC-EN | test | label-free | base | raw | 1 fps | 0.5681 | 0.0321 | n/a | n/a | n/a | 0.0076 | 14948 | 0.0276 | 1 | n/a | §4 | idea-stage/repro_lavad/ | single-captioner; coverage=0.994; missing 1/161 (0.6%) dropped, not interpolated; stratum=multi_span |
| LAVAD | 1 | MHC-ZH | test | label-free | base | base | 1 fps | 0.4923 | 0.2634 | n/a | n/a | n/a | -0.0022 | 18150 | 0.2648 | 1 | n/a | §4 | idea-stage/repro_lavad/ | single-captioner; stratum=single_span |
| LAVAD | 1 | MHC-ZH | test | label-free | base | base | 1 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12919 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_lavad/ | single-captioner; stratum=multi_span single-class pool, metrics undefined |
| LAVAD | 1 | MHC-ZH | test | label-free | base | raw | 1 fps | 0.5159 | 0.2737 | n/a | n/a | n/a | 0.0133 | 18127 | 0.2650 | 1 | n/a | §4 | idea-stage/repro_lavad/ | single-captioner; coverage=0.999; stratum=single_span |
| LAVAD | 1 | MHC-ZH | test | label-free | base | raw | 1 fps | n/a | n/a | n/a | n/a | n/a | n/a | 12908 | 0.0000 | 1 | n/a | §4 | idea-stage/repro_lavad/ | single-captioner; coverage=0.999; stratum=multi_span single-class pool, metrics undefined |

### K.4 Transplant fidelity (freeze §7) — the campaign's only pass/fail

Target: within **±0.03 absolute on both metrics** against LELA (arXiv 2602.09637). LELA does not
state which MultiHateClip language it pooled, so both are compared against the same target.

| method | dataset | LELA PR-AUC | ours PR-AUC (AP) | |diff| | LELA ROC-AUC | ours ROC-AUC | |diff| | verdict | ours PR-AUC (trapezoid, LAVAD's own convention) | note |
|---|---|---|---|---|---|---|---|---|---|---|
| LAVAD | HateMM | 0.5781 | 0.2909 | 0.2872 | 0.6163 | 0.5587 | 0.0576 | **OUT_OF_TOLERANCE** | 0.2829 |  |
| LAVAD | MHC-EN | 0.5865 | 0.3107 | 0.2758 | 0.6302 | 0.5559 | 0.0743 | **OUT_OF_TOLERANCE** | 0.3149 | LELA's 'MultiHateClip' column, language unstated |
| LAVAD | MHC-ZH | 0.5865 | 0.2634 | 0.3231 | 0.6302 | 0.4923 | 0.1379 | **OUT_OF_TOLERANCE** | 0.2637 | LELA's 'MultiHateClip' column, language unstated |

**Verdict: `OUT_OF_TOLERANCE` on all three rows.** Reported, not hidden. The investigation the
freeze requires follows, and it separates cleanly into two different-sized problems.

**ROC-AUC misses by 0.058 / 0.074 / 0.138.** That is outside tolerance but the right order of
magnitude, and it is the comparison worth trusting, because ROC-AUC does not depend on the positive
base rate of the pool.

**PR-AUC misses by ~0.29, and that gap is dominated by the evaluation pool, not by the port.**
Average precision is prevalence-dependent, and LELA states no pool. Measured on **our own HateMM
curve, unchanged**, simply re-pooled:

| pool | positive base rate | frame AP | frame ROC-AUC |
|---|---|---|---|
| test split, all 215 videos (our headline) | 0.2424 | 0.2909 | 0.5587 |
| test split, the 83 videos with an annotated span | 0.5833 | **0.5770** | 0.4708 |
| mean of per-video AP over those 83 videos | — | 0.6123 | — |

The same scores give **AP 0.291 or 0.577** depending only on which frames are pooled — a factor of
two, with the method untouched — and the span-positive pool lands 0.0011 from LELA's 0.5781. We do
**not** claim that is LELA's protocol: their ROC-AUC on that pool would be 0.4708, not 0.6163, so no
single pool we can construct reproduces both of their numbers. What the table does establish is
that **a PR-AUC comparison against a paper that does not state its frame pool is not interpretable
at ±0.03**, and that the honest reading of K.4 is "ROC-AUC is 0.06-0.14 low; PR-AUC is not
comparable as published".

**Would a full-corpus re-run settle it?** No, and the arithmetic says so without spending the GPU.
The coordinator's condition was to suspect the pool before the transplant and to re-run a dataset
on the full corpus if that would decide it. The full-corpus HateMM pool has base rate 0.2858
against the test split's 0.2424 — 18% higher. AP scales roughly with prevalence at fixed ranking
quality, so the full corpus would move our 0.2909 to roughly 0.34, nowhere near 0.578. The
sensitivity that matters is 0.24 → 0.58 (which frames are pooled), not 0.24 → 0.29 (which split).
A ~3 h full-corpus re-run of MHC-EN was therefore not spent.

**Three further candidate explanations, in the order we would test them.**
1. **The single-captioner adaptation** (see above). LAVAD's stage 03 retrieval pool is 1 caption
   per frame instead of up to 5. This can only lower our number, and is the largest deliberate
   departure in the port.
2. **The 1 s centre grid** against LAVAD's 1.875 samples/s. This costs temporal resolution, but
   K.5 shows the refined curve is almost constant within a video anyway, so it is unlikely to be
   worth 0.06 ROC-AUC.
3. **LELA's own port may have hit the refusal wall silently.** `MODEL_ASSETS_STATUS §3.11a` notes
   that LAVAD's `_interpolate_unmatched_scores` replaces refusals with a linear blend of their
   neighbours and nothing in the output distinguishes them. We mask instead (K.5). On 1.2% of
   frames the difference is small, but it moves in the direction of a *higher* published number.

### K.5 Refusals — rate by dataset and by ground-truth label

`_parse_score` returning -1 is recorded with its raw generation text and the frame is left
unscored. **`np.interp` is never called.** Positive/negative is the gold label at the centre's own
instant.

| dataset | scored centres | refusals | refusal rate | on positive frames | on negative frames | enrichment (pos/neg) |
|---|---|---|---|---|---|---|
| HateMM | 29,243 | 322 | 0.0110 | 138/7,080 = 0.0195 | 184/22,163 = 0.0083 | 2.35x |
| MHC-EN | 5,604 | 82 | 0.0146 | 17/1,527 = 0.0111 | 65/4,077 = 0.0159 | 0.70x |
| MHC-ZH | 4,558 | 6 | 0.0013 | 1/1,208 = 0.0008 | 5/3,350 = 0.0015 | 0.55x |
| HateClipSeg | 28,242 | 413 | 0.0146 | 362/13,360 = 0.0271 | 51/14,882 = 0.0034 | 7.91x |
| **all four** | 67,647 | 823 | 0.0122 | 518/23,175 = 0.0224 | 305/44,472 = 0.0069 | 3.26x |

**Reading it.** The overall rate is **1.22%**, under the ~2% below which `MODEL_ASSETS_STATUS
§3.11a` says the remaining countermeasures are unnecessary — but it is **not label-balanced**, which
is the other half of that condition. Refusals are **3.26x more likely on a positive frame than a
negative one** overall, and **7.91x on HateClipSeg**, exactly the concentration §3.11a predicted:
Llama-2-13b-chat declines on descriptions of violence and group-directed hostility, which in these
corpora are the positive class. Because we mask rather than interpolate, this costs coverage
(`raw` coverage 0.985-0.999) rather than biasing the score; had `np.interp` been left in, 2.24% of
positive frames would have been silently replaced by a blend of their mostly-negative neighbours.
MHC-ZH is the outlier at 0.13%, consistent with BLIP-2 describing Chinese-language video in
generic English that trips no safety heuristic.

The most common refusals are the standard Llama-2 safety strings, e.g. *"I cannot provide a rating
for the scene as it is not appropriate to use offensive language"* (188 occurrences) and
*"I apologize, but I cannot provide a rating for the scene you described..."* (86). Raw text for
every refusal is in `data/lavad/score_refusals/`.

Because the rate is label-imbalanced, §3.11a item 3 — a content-moderation prompt reframing run as
a **paired** variant next to the verbatim prompt — is justified. It is implemented
(`lavad_chain.py --prompt mod`, prompt frozen in the file) and is reported in K.6 if it ran; it is
never a silent substitution for the verbatim row.

### K.6 What the numbers say

1. **LAVAD clears the random floor on three of four datasets, and is at or below it on MHC-ZH.**
   Test frame ROC-AUC 0.5587 / 0.5559 / 0.4923 / 0.5768 (HateMM / MHC-EN / MHC-ZH / HateClipSeg)
   against a 0.500 floor. Oracle-normalised AP is 0.142 / 0.068 / -0.002 / 0.998 — it recovers 14%
   of the chance-to-video-oracle gap on HateMM and essentially none on MHC. **Read the HateClipSeg
   `AP_norm` of 0.998 with care:** HateClipSeg's broadcast ceiling (AP 0.5437) sits only 0.073
   above its base rate (0.4712), so the normaliser's denominator is tiny and the ratio is unstable;
   the raw AP gain there is 0.073, not a near-perfect result.
2. **Against the other campaign floors, LAVAD is mid-pack, not ahead.** On HateMM its 0.5587 sits
   between ZS-CLIP (0.5368) and ZS-ImageBind (0.5919), and above Qwen2.5-VL grounding (0.5185). A
   seven-stage chain with two large language models, a captioner and a multimodal retriever does
   not beat a single ImageBind forward pass on these corpora.
3. **The gain comes from the refinement stage, and the refinement works by flattening the curve.**
   Refined vs raw ROC-AUC is 0.559 vs 0.504 (HateMM), 0.556 vs 0.508 (MHC-EN), 0.577 vs 0.545
   (HateClipSeg). Over the same videos the **median within-video standard deviation of the score
   falls from 0.048 to 0.007**, a 7x flattening, and the share of videos whose whole curve varies
   by less than 0.01 rises from 23% to 57% (81% on MHC-ZH). Stage 06 replaces each moment's score
   with a similarity-weighted average of the ten visually nearest moments *in the same video*, so
   on corpora where the hateful content is diffuse it converges to a per-video constant. **What
   improves the metric is therefore video-level smoothing, not localisation** — the opposite of
   what a frame-level baseline is supposed to demonstrate.
4. **MHC-ZH is the failure case and the reason is legible.** ROC-AUC 0.4923 is below chance and
   `AP_norm` is -0.002. LAVAD is an English-only chain: BLIP-2 emits English captions for
   Chinese-language video, and the Chinese on-screen text that carries much of the hate signal is
   not read at all. 81% of MHC-ZH videos end with a curve flatter than 0.01, the highest of the
   four corpora.
5. **The transplant is out of tolerance, and the PR-AUC half of it is not diagnostic.** ROC-AUC is
   0.06-0.14 below LELA. PR-AUC is ~0.29 below, but the same curve yields 0.291 or 0.577 depending
   only on which frames are pooled (K.4), so that half of the ±0.03 check cannot be evaluated
   against a paper that does not state its pool. Anyone re-using LELA's LAVAD row as a baseline
   should state the frame pool alongside it.
6. **Refusals are rare but land where they hurt.** 1.22% overall, 3.26x enriched on positive
   frames, 7.91x on HateClipSeg. The rate is too low to explain the transplant gap, but it is
   large enough that a harness which interpolates instead of masking would report a *different*
   number without saying so — which is the failure mode §3.11a warned about and the reason
   coverage is printed next to every AUC here.
7. **Consequence for the campaign.** LAVAD is the strongest label-free chain reproduced so far on
   HateClipSeg (0.5768) and mid-pack elsewhere, but every dataset remains far below the
   zero-temporal-resolution gold-broadcast ceiling (0.8857 / 0.9427 / 0.9842 / 0.6260 ROC-AUC).
   The gap between "knows which video is hateful" and "knows when" is untouched by this method.
