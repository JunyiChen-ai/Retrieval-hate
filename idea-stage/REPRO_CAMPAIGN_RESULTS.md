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
