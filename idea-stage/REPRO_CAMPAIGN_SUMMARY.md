# REPRO_CAMPAIGN_SUMMARY — label-free frame-level baseline reproduction, Waves 0/1/2

Protocol: `idea-stage/REPRO_CAMPAIGN_FREEZE.md` (frozen, deviations D1–D4 in its §12).
Per-method detail: `idea-stage/REPRO_CAMPAIGN_RESULTS.md` §A–§M, plus the Wave 2 section files listed in §8.
Assets: `idea-stage/repro_campaign/MODEL_ASSETS_STATUS.md`.
Machine: single RTX 5090, conda `HateVideo`, torch 2.7.1+cu128. **Zero paid API spend.**

**This is a baseline table, not a candidate trial** (freeze §0). No row here receives a GO/KILL
verdict and no decision rule in this file selects a winner. The only pass/fail in the whole campaign
is the transplant-fidelity check of freeze §7, reported in §4.

Every number in §1 and §2 is generated from the raw evaluator JSON by
`scripts/repro_campaign/summary_table.py`, never transcribed by hand. Re-running that script after
a pending method lands refills its row automatically:

```
python scripts/repro_campaign/summary_table.py --split test --csv idea-stage/repro_campaign/summary_test.csv
python scripts/repro_campaign/summary_table.py --split all
```

> **Read `AP_norm` with the §5 footnote in hand. On HateClipSeg it is not interpretable at all.**

---

## 0. Status of the roster

| method | wave | supervision | state |
|---|---|---|---|
| ZS-CLIP | 0 | label-free | **done** |
| ZS-ImageBind (image / video / audio) | 0 | label-free | **done** |
| Qwen2.5-VL-7B native grounding | 0 | label-free | **done** (12 false exclusions, freeze D3, not yet re-run) |
| LAVAD | 1 | label-free | **done** |
| URF-HVAA | 1 | label-free | **done** |
| LaGoVAD | 1 | aux-temporal-pretrain | **done** |
| AV²A | 1 | label-free | **done** (evt_* rows single-seed, freeze D4) |
| UniTime | 1 | aux-temporal-pretrain | **running** — corpus decode in flight |
| MULDE | 2 | one-class | **done** (headline `clipL336` stream only, deviation M-1) |
| CLAP | 2 | unlabelled | **running** — port complete, FedAvg grid in flight |
| T3AL | 2 | label-free | **queued** behind UniTime on the single card |
| SeViLA Localizer | 2 | aux-temporal-pretrain | **queued** behind UniTime on the single card |
| OV-AVEL, FLAM, FineLAP, BaGLM | 2 | install-gated | **ran** on one video; corpus run not scheduled |
| OmniVTG | 2 | install-gated | **needs the free card**; everything up to the engine forward verified |
| ZS-STVG, DASM, LAVIDA | 2 | install-gated | **dropped**, reasons in §6 |

Run state, pointers and ETAs for everything still moving are in §8.

---

## 1. Master table — test split (headline)

Frame ROC-AUC / frame PR-AUC (average precision), 4 dp, pooled over the whole evaluated split.
`not run` marks a method whose corpus run has not produced its evaluator JSON yet.

MULDE is stochastic and carries three seeds (freeze §6); the row below shows **seed 20250819** so the
table stays one number per cell. Its three-seed mean ± sd — HateMM **0.5989 ± 0.0031**, MHC-EN
**0.4737 ± 0.0117**, MHC-ZH 0.5102 ± 0.0028, HateClipSeg 0.5276 ± 0.0066 — is in §O.3, and the seed
spread is small enough that no ranking in this table turns on which seed is shown.

| method | wave | supervision | variant | native_rate | HateMM ROC / AP | MHC-EN ROC / AP | MHC-ZH ROC / AP | HateClipSeg ROC / AP |
|---|---|---|---|---|---|---|---|---|
| **GOLD_BROADCAST** | — | control | control | video | 0.8857 / 0.5831 | 0.9427 / 0.7664 | 0.9842 / 0.9191 | 0.6260 / 0.5433 |
| **RANDOM_UNIFORM** | — | control | control | 4 fps | 0.5002 / 0.2424 | 0.5002 / 0.2736 | 0.4986 / 0.2646 | 0.5004 / 0.4711 |
| **ZS-CLIP** | 0 | label-free | base (prompt=main) | 4 fps | 0.5368 / 0.2775 | 0.5013 / 0.2678 | 0.6075 / 0.3406 | 0.4990 / 0.4612 |
| **ZS-ImageBind (image)** | 0 | label-free | base | 4 fps | 0.5919 / 0.3143 | 0.5938 / 0.3286 | 0.5975 / 0.3580 | 0.5926 / 0.5535 |
| **ZS-ImageBind (video)** | 0 | label-free | base | 0.5 fps | 0.5907 / 0.3100 | 0.5636 / 0.3056 | 0.5727 / 0.3546 | 0.5814 / 0.5431 |
| **ZS-ImageBind (audio)** | 0 | label-free | base | 0.5 fps | 0.5654 / 0.2906 | 0.6157 / 0.3678 | 0.6527 / 0.3958 | 0.5652 / 0.5121 |
| **Qwen2.5-VL-7B grounding** | 0 | label-free | query=main | interval | 0.5185 / 0.2522 | 0.5221 / 0.2806 | 0.5113 / 0.2699 | 0.5030 / 0.4750 |
| **LAVAD** | 1 | label-free | base | 1 fps | 0.5587 / 0.2909 | 0.5559 / 0.3107 | 0.4923 / 0.2634 | 0.5768 / 0.5464 |
| **URF-HVAA** | 1 | label-free | base | 0.1 fps | 0.5744 / 0.3183 | 0.5493 / 0.2973 | 0.5454 / 0.2868 | 0.5863 / 0.5528 |
| **LaGoVAD** | 1 | aux-temporal-pretrain | main | 0.5 fps | 0.5579 / 0.3047 | 0.5239 / 0.2617 | 0.5965 / 0.3118 | 0.5000 / 0.4666 |
| **AV²A** | 1 | label-free | sim_combined | 1 fps / 0.1 fps | 0.5393 / 0.2520 | 0.5310 / 0.3224 | 0.5595 / 0.3206 | 0.4860 / 0.4680 |
| **UniTime** | 1 | aux-temporal-pretrain | window | interval | not run | not run | not run | not run |
| **MULDE** | 2 | one-class | clipL336_s0 | 4 fps | 0.6002 / 0.3090 | 0.4869 / 0.2585 | 0.5129 / 0.2499 | 0.5326 / 0.5000 |
| **CLAP** | 2 | unlabelled | main | 32 seg/video | not run | not run | not run | not run |
| **T3AL** | 2 | label-free | main | interval | not run | not run | not run | not run |
| **SeViLA Localizer** | 2 | aux-temporal-pretrain | main | 1 fps | not run | not run | not run | not run |

Secondary variants each method's own section calls out (never a substitute for the headline row above):

| method | wave | supervision | variant | native_rate | HateMM ROC / AP | MHC-EN ROC / AP | MHC-ZH ROC / AP | HateClipSeg ROC / AP |
|---|---|---|---|---|---|---|---|---|
| LaGoVAD (bin, text-free head) | 1 | aux-temporal-pretrain | bin | 0.5 fps | 0.4989 / 0.2317 | 0.6058 / 0.3490 | 0.5673 / 0.2895 | 0.5431 / 0.5499 |
| LAVAD (raw, pre-refinement) | 1 | label-free | raw | 1 fps | 0.5040 / 0.2445 | 0.5077 / 0.2801 | 0.5159 / 0.2737 | 0.5453 / 0.5018 |
| URF-HVAA (round1, pre-refinement) | 1 | label-free | round1 | 0.1 fps | 0.5771 / 0.3105 | 0.5664 / 0.3187 | 0.5397 / 0.2893 | 0.5854 / 0.5534 |
| AV²A (sim_video) | 1 | label-free | sim_video | 1 fps | 0.5591 / 0.2650 | 0.4957 / 0.2598 | 0.6199 / 0.3601 | 0.5135 / 0.4689 |
| AV²A (sim_audio) | 1 | label-free | sim_audio | 0.1 fps | 0.4922 / 0.2509 | 0.5391 / 0.3087 | 0.5173 / 0.2877 | 0.4815 / 0.4904 |
| UniTime (mr_seg) | 1 | aux-temporal-pretrain | seg | segment | not run | not run | not run | not run |


## 2. Master table — full corpus

Reported where a method ran the whole corpus. LAVAD and URF-HVAA deliberately did not (§K.2);
T3AL has no full-corpus row by design (its deviation T-8).

> **MULDE's full-corpus row is contaminated by construction and is not a baseline number.** It was
> fitted on the non-hateful videos of the train split, and the full corpus contains those very
> videos. The size of the inflation tracks the share of the corpus it trained on almost exactly:
> MHC-EN 48.9% fitted → **+0.358** ROC over its test row, MHC-ZH 47.5% → +0.199, HateMM 41.3% →
> +0.202, HateClipSeg 7.6% → **+0.030**. That is a memorisation curve, not a generalisation result.
> **MULDE's honest figures are the test-split rows in §1.**

| method | wave | supervision | variant | native_rate | HateMM ROC / AP | MHC-EN ROC / AP | MHC-ZH ROC / AP | HateClipSeg ROC / AP |
|---|---|---|---|---|---|---|---|---|
| **GOLD_BROADCAST** | — | control | control | video | 0.9033 / 0.6742 | 0.9536 / 0.7767 | 0.9709 / 0.8537 | 0.6164 / 0.5296 |
| **RANDOM_UNIFORM** | — | control | control | 4 fps | 0.5002 / 0.2859 | 0.5002 / 0.2445 | 0.5005 / 0.2543 | 0.4996 / 0.4632 |
| **ZS-CLIP** | 0 | label-free | base (prompt=main) | 4 fps | 0.5406 / 0.3135 | 0.5474 / 0.2767 | 0.5611 / 0.2953 | 0.5119 / 0.4674 |
| **ZS-ImageBind (image)** | 0 | label-free | base | 4 fps | 0.5869 / 0.3546 | 0.5874 / 0.3168 | 0.5598 / 0.3065 | 0.5889 / 0.5485 |
| **ZS-ImageBind (video)** | 0 | label-free | base | 0.5 fps | 0.5997 / 0.3618 | 0.5666 / 0.2967 | 0.5490 / 0.3010 | 0.5856 / 0.5451 |
| **ZS-ImageBind (audio)** | 0 | label-free | base | 0.5 fps | 0.5827 / 0.3768 | 0.6078 / 0.3274 | 0.6139 / 0.3474 | 0.5696 / 0.5118 |
| **Qwen2.5-VL-7B grounding** | 0 | label-free | query=main | interval | 0.5245 / 0.3018 | 0.5119 / 0.2501 | 0.5155 / 0.2608 | 0.5049 / 0.4671 |
| **LAVAD** | 1 | label-free | base | 1 fps | not run | not run | not run | not run |
| **URF-HVAA** | 1 | label-free | base | 0.1 fps | not run | not run | not run | not run |
| **LaGoVAD** | 1 | aux-temporal-pretrain | main | 0.5 fps | 0.5225 / 0.3099 | 0.5605 / 0.2780 | 0.5694 / 0.3006 | 0.5433 / 0.4854 |
| **AV²A** | 1 | label-free | sim_combined | 1 fps / 0.1 fps | 0.4659 / 0.2614 | 0.5520 / 0.2772 | 0.5352 / 0.2815 | 0.5015 / 0.4700 |
| **UniTime** | 1 | aux-temporal-pretrain | window | interval | not run | not run | not run | not run |
| **MULDE** | 2 | one-class | clipL336_s0 | 4 fps | 0.8054 / 0.5551 | 0.8361 / 0.5085 | 0.7028 / 0.4374 | 0.5591 / 0.5032 |
| **CLAP** | 2 | unlabelled | main | 32 seg/video | not run | not run | not run | not run |
| **T3AL** | 2 | label-free | main | interval | not run | not run | not run | not run |
| **SeViLA Localizer** | 2 | aux-temporal-pretrain | main | 1 fps | not run | not run | not run | not run |

Secondary variants each method's own section calls out (never a substitute for the headline row above):

| method | wave | supervision | variant | native_rate | HateMM ROC / AP | MHC-EN ROC / AP | MHC-ZH ROC / AP | HateClipSeg ROC / AP |
|---|---|---|---|---|---|---|---|---|
| LaGoVAD (bin, text-free head) | 1 | aux-temporal-pretrain | bin | 0.5 fps | 0.4921 / 0.2753 | 0.5822 / 0.3062 | 0.5724 / 0.3135 | 0.5785 / 0.5523 |
| LAVAD (raw, pre-refinement) | 1 | label-free | raw | 1 fps | not run | not run | not run | not run |
| URF-HVAA (round1, pre-refinement) | 1 | label-free | round1 | 0.1 fps | not run | not run | not run | not run |
| AV²A (sim_video) | 1 | label-free | sim_video | 1 fps | 0.5602 / 0.3214 | 0.5127 / 0.2462 | 0.5532 / 0.2820 | 0.5176 / 0.4686 |
| AV²A (sim_audio) | 1 | label-free | sim_audio | 0.1 fps | 0.4644 / 0.2811 | 0.5177 / 0.2487 | 0.5032 / 0.2588 | 0.4931 / 0.4722 |
| UniTime (mr_seg) | 1 | aux-temporal-pretrain | seg | segment | not run | not run | not run | not run |


## 3. Controls (freeze §3), and what they mean

Both control rows are computed inside the same evaluator, on the same frame pool as the method rows.

| dataset | test base rate | random ROC / AP | gold-broadcast ROC / AP | chance→oracle AP gap |
|---|---|---|---|---|
| HateMM | 0.2422 | 0.5002 / 0.2424 | 0.8857 / 0.5831 | 0.3407 |
| MHC-EN | 0.2734 | 0.5002 / 0.2736 | 0.9427 / 0.7664 | 0.4930 |
| MHC-ZH | 0.2649 | 0.4986 / 0.2646 | 0.9842 / 0.9191 | 0.6542 |
| HateClipSeg | 0.4709 | 0.5004 / 0.4711 | 0.6260 / 0.5433 | **0.0724** |

`GOLD_BROADCAST` is a perfect video-level classifier with **zero** localisation ability: score 1 on
every frame of every gold-positive video, 0 elsewhere. It is a ceiling for this campaign and a floor
for the benchmarks — the gap between it and chance is the entire amount of headroom a frame-level
method can win, and on HateClipSeg that gap is 0.07 of AP because the annotated spans cover 47% of
all frames.

**Every method reproduced so far sits in the bottom fifth of that interval on every dataset.** The
best label-free test ROC-AUC in the campaign is URF-HVAA at 0.5744 / 0.5493 / 0.5454 / 0.5863,
against broadcast ceilings of 0.8857 / 0.9427 / 0.9842 / 0.6260.

## 4. Alignment against LELA's published rows

LELA (arXiv 2602.09637) is the only third party to have ported any of these methods to our datasets.
Freeze §7 sets a formal ±0.03 gate for LAVAD and URF-HVAA; ZS-CLIP and ZS-ImageBind get the informal
±0.05 check the campaign brief specified.

| method | dataset | LELA ROC | ours ROC | \|diff\| | LELA PR | ours AP | \|diff\| | verdict |
|---|---|---|---|---|---|---|---|---|
| ZS-CLIP | HateMM | 0.5367 | 0.5406 (full) | **0.0039** | — | — | — | agrees |
| ZS-CLIP | MultiHateClip | 0.5449 | 0.5474 (EN full) | **0.0025** | — | — | — | agrees |
| ZS-ImageBind | HateMM | 0.5683 | 0.5869 (full) | 0.0186 | — | — | — | agrees |
| ZS-ImageBind | MultiHateClip | 0.5753 | 0.5874 (EN full) | 0.0121 | — | — | — | agrees |
| LAVAD | HateMM | 0.6163 | 0.5587 | 0.0576 | 0.5781 | 0.2909 | 0.2872 | OUT_OF_TOLERANCE |
| LAVAD | MHC-EN | 0.6302 | 0.5559 | 0.0743 | 0.5865 | 0.3107 | 0.2758 | OUT_OF_TOLERANCE |
| LAVAD | MHC-ZH | 0.6302 | 0.4923 | 0.1379 | 0.5865 | 0.2634 | 0.3231 | OUT_OF_TOLERANCE |
| URF-HVAA | HateMM | 0.5674 | 0.5744 | **0.0070** | 0.6239 | 0.3183 | 0.3056 | OUT_OF_TOLERANCE (ROC passes) |
| URF-HVAA | MHC-EN | 0.5626 | 0.5493 | **0.0133** | 0.6147 | 0.2973 | 0.3174 | OUT_OF_TOLERANCE (ROC passes) |
| URF-HVAA | MHC-ZH | 0.5626 | 0.5454 | **0.0172** | 0.6147 | 0.2868 | 0.3279 | OUT_OF_TOLERANCE (ROC passes) |

**The single most useful result of the alignment work: the ~0.3 PR-AUC offset is a property of the
comparison, not of either port.** Two independently built chains — different repos, different
captioners, different LLMs — miss LELA's PR-AUC by almost exactly the same amount while URF
reproduces its ROC-AUC to within 0.007–0.017. Average precision depends on the positive base rate of
the frame pool, and LELA states no pool. Measured on **our own HateMM curve, unchanged, simply
re-pooled**:

| pool | base rate | frame AP | frame ROC-AUC |
|---|---|---|---|
| LAVAD, test split, all 215 videos (our headline) | 0.2424 | 0.2909 | 0.5587 |
| LAVAD, test split, the 83 videos carrying a span | 0.5833 | **0.5770** | 0.4708 |
| URF-HVAA, test split, all videos | 0.2485 | 0.3183 | 0.5744 |
| URF-HVAA, test split, span-carrying videos only | 0.5902 | **0.6334** | **0.5585** |
| **LELA's published URF-HVAA HateMM row** | (unstated) | **0.6239** | **0.5674** |

On the span-positive pool URF lands within **0.0095** on PR-AUC and **0.0089** on ROC-AUC of LELA
simultaneously. We do not claim that is LELA's protocol — LAVAD's ROC on the same pool would be
0.4708, not 0.6163, so no single pool reproduces both of their rows for both methods. What it does
establish is that **a PR-AUC comparison against a paper that does not state its frame pool is not
interpretable at ±0.03**. Anyone re-using LELA's numbers as a baseline should state the pool
alongside them.

## 5. The `AP_norm` footnote, and why HateClipSeg is the exception

`AP_norm = (AP − base_rate) / (AP_broadcast − base_rate)` reads 0 at the random floor and 1 at the
zero-localisation video-level ceiling. It is a descriptive rescaling (freeze §2), not a metric to
optimise, and **it is not comparable across datasets.**

The denominator is 0.34 (HateMM), 0.49 (MHC-EN) and 0.65 (MHC-ZH) on the test split, but only
**0.0724 on HateClipSeg**, because a 47% positive base rate leaves the video-level oracle almost no
room above chance. Dividing by 0.07 does not remove a nuisance, it **amplifies noise about 14×**: a
0.01 wobble in AP moves HateClipSeg's `AP_norm` by 0.136, against 0.015 on MHC-ZH. Concretely, the
LaGoVAD HateClipSeg rows span `AP_norm` −0.22 to +1.04 while their raw AP spans only 0.4575 to
0.5499, and both LAVAD (0.998) and ZS-ImageBind (1.09) produce readings at or above 1 that mean
nothing more than "AP touched a ceiling sitting 0.07 above chance".

**On HateClipSeg, read raw `frame_PR_AUC` against `base_rate` and treat `AP_norm` as unusable.**
The evaluator now emits `AP_norm_denom` and `AP_norm_reliable` (`False` when the gap is under 0.15)
with every row so the column cannot be misread again. Raised by the LAVAD worker after a 0.998
reading; applies to every method's HateClipSeg rows in this campaign.

## 6. Lessons

### 6.1 Four bad artifacts that did not raise an error

| # | what went wrong | how it presented | how it was caught, and the fix |
|---|---|---|---|
| 1 | the cached `openai/clip-vit-base-patch16` weight blob had the right byte count and the wrong content (max abs weight 3.7 × 10¹⁹) | CLIP returned **one identical embedding for every frame of every video**; the resulting flat curves were written up in `MODEL_ASSETS_STATUS §3.1` as a property of LaGoVAD's binary head | `audit_hf_cache.sh` re-hashes every blob against its own filename (an HF blob is stored under its sha256): **7 of 55 weight files corrupt**, all repaired by `hf_refetch.py` and re-verified |
| 2 | the in-flight crash marker could not tell a decoder crash from an operator `SIGTERM`, an OOM-kill or a reboot | every stop silently retired one healthy video as `decode_all_backends_failed`; **12 of the 14 ids the Wave 0 Qwen row excludes decode cleanly** under both ffmpeg and PyAV | freeze deviation **D3**: an id is retired only after taking the process down **twice** (`.crashcount.json`), and a caught signal clears the marker and exits retiring nothing |
| 3 | runner scripts had no `set -e`, and the AV²A supervisor printed the smoke's exit code without checking it, emitting `RUN COMPLETE` on driver `rc=0` regardless of what was written | the LAVAD chain exited **rc=0 having written 5 curves for one dataset** | `set -euo pipefail` plus per-dataset curve-count guards in the LaGoVAD chain, the UniTime converter and the AV²A supervisor, each verified to pass on real data and fail on a truncation |
| 4 | `decord` cannot open 25.5% of MultiHateClip-EN containers | a quarter of that dataset would have been recorded as **method** failures rather than **reader** failures | `decord_fallback.py` tries the real reader first and falls back to PyAV |

### 6.2 The audit principle: exit 0 is not evidence

Defects 1 and 2 are *silent correctness* failures — right shape, right range, wrong content, no
exception, clean exit code. "Did it run?" does not separate them from a working component. The
question that does is **does the output vary with the input**, which a collapsed encoder cannot do.

`scripts/repro_campaign/discrimination_check.py` turns that into a cheap smoke assertion
(`curve_varies`, `embeddings_discriminate`, `scores_separate_items`). Replayed against the real
corrupt-CLIP failure it fires on both the collapsed embeddings (mean off-diagonal cosine 1.000000)
and the resulting flat curve, and it passes on the repaired output — while deliberately **not**
firing on Qwen2.5-VL's 19% modal answer, which is a finding rather than a fault.

Companion case, same species and worth keeping visible: while diagnosing URF's OOM, VideoLLaMA3's
`VisionAttention` was read, seen to add a bool mask to float logits, measured at 44–56% attention
leakage, and reported as evidence that the forced flash→sdpa adaptation had changed the model's
semantics. The class that actually runs is `VisionSdpaAttention`, a subclass selected by
`VISION_ATTENTION_CLASSES[config._attn_implementation]`, which **passes** the same bool tensor to
`F.scaled_dot_product_attention` as `attn_mask`, where `True` means attend. Verified at 0.000e+00
against independently computed per-block attention. Behaviour was inferred from source and reported
as a measurement; the memory patch written off the back of it targeted dead code and would have
returned 0 while reporting success.

### 6.3 decord / PyAV census

**275 of 3,084 containers** across the four corpora need the PyAV fallback; on MultiHateClip-EN
alone that is **25.5%** of the dataset. Two HateMM files (`hate_video_147`, `hate_video_292`) have
no video stream at all — a property of the released media, recorded as freeze deviation **D2** — and
six further HateMM files have no decodable audio. None of the eight is in a frozen split, so no
headline table is affected.

### 6.4 The refusal prediction inverted at scale

`MODEL_ASSETS_STATUS §3.11a` predicted, from **three hand-picked provocative captions**, that
Llama-3.1-8B-Instruct (URF's backbone) would refuse frequently on the positive class while
Llama-2-13b-chat (LAVAD's) would not, and recommended swapping the model on that basis. Measured on
the full runs:

| scorer | dialogs | refusals | rate | (a)'s prediction |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct (URF-HVAA) | 6,455 | 5 | **0.077%** | frequent refusals on the positive class |
| Llama-2-13b-chat (LAVAD) | 67,647 | 823 | **1.22%** | "did not refuse on any of the four test captions" |

Llama-2 refuses about **16× more often** — the reverse of the prediction, and the opposite of the
basis for its recommendation. Three adversarially chosen examples cannot estimate a rate, and the
write-up presented a property of the probe as a property of the model.

What survives is the **mechanism** (`_parse_score` → −1 → `np.interp` silently replaces a refusal
with a blend of its neighbours; the campaign masks instead and reports `coverage` beside every AUC)
and the **label-imbalance** finding (LAVAD's refusals are 3.26× enriched on positive frames overall
and 7.91× on HateClipSeg). The recommended countermeasure — a content-moderation reframing of the
prompt — was run as a paired variant and moved the overall refusal rate by **−0.03 percentage
points** (1.22% → 1.19%), *raising* it on MHC-EN. It did move the metric by up to ±0.05 ROC-AUC,
which makes it a prompt-sensitivity result rather than a fix.

This is the third claim in `MODEL_ASSETS_STATUS` overturned by measurement, after the "LaGoVAD's
binary head is constant" reading (which was the corrupt checkpoint of 6.1 #1) and the
attention-semantics reading of 6.2. All three failed the same way: **a small unrepresentative
observation reported as a property of the model.**

### 6.5 Two findings from the install-gated group

- **OV-AVEL's native output is a hard 0/1 agreement flag, not a score**, so any ranking metric needs
  the continuous similarity underneath, reported as our adaptation — the same situation as LaGoVAD's
  binary-head row.
- **BaGLM's real blocker is semantic, not dependency-level.** Both named install blockers dissolved
  (`torchcodec==0.4.0` decodes our mp4 against the stock system ffmpeg; flash-attn only needs
  `use_flash_attn=False`). But `bayes_filter.py` smooths per-segment posteriors with
  **step-prerequisite** probabilities, a prior that presumes an ordered procedure. The video-segment
  stage transfers to hate video; the filter the paper is named for does not, without inventing a
  step ordering.

### 6.6 Two of the reproduced repos select their read-out on test labels

This one is a property of the released code, not of our port, and it turned up independently in two
Wave 2 methods:

- **MULDE.** Upstream's own evaluation loop arg-maxes the noise scale and the GMM aggregation on
  **test** AUC. Freeze §10 red line 1 forbids that, so the selection was moved to the val split. The
  cost is visible: our HateMM row is 0.5989 where a naive port that kept upstream's loop would have
  printed a higher number, and the val→test drop is −0.047 to −0.102 on three of four datasets.
- **CLAP.** `evaluate_ucf` loads UCF-Crime frame labels from `labels/gt-ucf-RTFM.npy` and `test_ucf`
  reports the **max** AUC over client models before and after local training. The in-loop read-out
  was disabled and the row reports the single aggregated global model at the frozen round.

Neither repo hides this and neither is unusual in the VAD literature, where "we report the best
epoch" is common. It matters here for one specific reason: **any published number produced that way
is not comparable to a number produced under this campaign's protocol**, and the gap is the same
order as the differences between methods in §1. When one of these rows looks lower than its paper's
figure, this is the first thing to check.

## 7. Does the literature mechanism work in the hate domain? One sentence per method

Each verdict rests only on that method's own section; the pointer is given so the evidence is one
click away. "No" here means the published mechanism does not deliver frame-level localisation on
these corpora — it is not a judgement of the method on the benchmarks it was built for (freeze §0).

| method | mechanism under test | works? | the evidence, in one sentence |
|---|---|---|---|
| **ZS-CLIP** (§A–G) | frame-level CLIP similarity to a hate prompt | **no** | at or below chance on two of four corpora (0.5368 / 0.5013 / 0.6075 / 0.4990), and the prompt wording moves ROC further than the method's entire margin over chance — LAVAD's own "anomalous" wording is below chance on three of four, which is the concrete form of the domain gap: hateful content is not visually anomalous. |
| **ZS-ImageBind** (§H) | joint-embedding similarity across image / video / audio | **partly** | the most consistent zero-shot floor, 0.09–0.10 above chance on all four datasets, and its **audio** channel is the strongest zero-shot cell in the campaign on both MultiHateClip halves (0.6157 EN, 0.6527 ZH) — the modality that carries hate here is not the one the published visual baselines score. |
| **Qwen2.5-VL-7B grounding** (§I) | native temporal grounding, emits intervals | **no** | at chance on all four (0.5030–0.5221) with every generation parsing, a single identical answer returned for 10–19% of videos, and a median predicted interval of 4.2–5.9 s against a median gold span of 19–21 s — the Charades-STA prior transferred verbatim. |
| **LAVAD** (§K) | captioner → LLM anomaly scoring → retrieval refinement | **no, and its gain is not localisation** | refinement raises HateMM ROC 0.504 → 0.559 while flattening the within-video curve **sevenfold** (median sd 0.048 → 0.007, 57% of videos varying by under 0.01), so what improves the metric is video-level smoothing; below chance on MHC-ZH, where an English-only captioner cannot read the Chinese on-screen text. |
| **URF-HVAA** (§L) | video-captioner → LLM scoring → tag-conditioned re-scoring | **weakly, and the gain is a captioner gain** | best label-free ROC in the campaign (0.5744 / 0.5493 / 0.5454 / 0.5863) at one tenth LAVAD's sampling rate, with its largest advantage on MHC-ZH (+0.053) where a video captioner reads on-screen text a frame captioner throws away — but the paper's headline re-scoring gate admits only **42 of 642 videos**, because it is calibrated for UCF-Crime's score distribution, not ours. |
| **LaGoVAD** (§M) | a written definition selects the anomaly at inference | **no** | on MHC-EN and HateClipSeg the strongest row is `bin`, the **text-free** binary head, ahead of all ten text rows, and paraphrasing the same definition swings MHC-ZH ROC from 0.4593 to 0.6432 — what transfers is a generic surveillance-anomaly prior, not the hate definition. |
| **AV²A** (§J) | training-free open-vocabulary audio-visual event localisation with score-level early fusion | **no** | all 24 (dataset × variant) cells sit at or near chance, the audio branch is at or below the random floor on two corpora, and the headline early fusion is **below the better of its two inputs on all four datasets**; its best F1@tIoU comes from the variant with the *worst* frame ranking, because a 10 s audio window happens to match the gold span length without locating anything. |
| **UniTime** (§N) | universal temporal grounding, six hate categories as six queries | *pending* | corpus run in flight; §8 of this file carries the state. |
| **MULDE** (§O) | one-class multi-scale density estimation of normality | **no** | beats the random floor on only two of four corpora (HateMM 0.5989 ± 0.0031, HateClipSeg 0.5276 ± 0.0066), sits at chance on MHC-ZH (0.5102 ± 0.0028) and **below** chance on MHC-EN (0.4737 ± 0.0117); its best result recovers 19% of the chance-to-oracle gap, and the strong-looking 0.80–0.83 full-corpus figure is memorisation of the negatives it was fitted on. Its HateMM row is nonetheless the highest frame ROC-AUC in the campaign — which says a little one-class supervision buys about what a whole captioner-plus-LLM chain does, not that the mechanism transfers. |
| **CLAP** (§P) | coarse-to-fine pseudo-labels from an unlabelled pool, federated | *pending* | port complete and documented; FedAvg grid in flight. |
| **T3AL** (§Q) | test-time adaptation of a VLM for zero-shot localisation | *pending* | queued behind UniTime; one finding already recorded, that upstream's `get_indices` degenerates to comparing a set against itself on any video under 400 feature vectors, i.e. **most MHC videos at 4 fps**. |
| **SeViLA Localizer** (§R) | frame-wise yes/no VQA keyframe scoring | *pending* | queued behind UniTime; the §R verdict is deliberately left unwritten rather than guessed. |

### The campaign's own headline finding

Read as a whole, the table is a statement about the **benchmarks** more than about the methods.
The interval between chance and a perfect video-level classifier with zero localisation ability is
narrow — frame AP 0.2424 → 0.5831 on HateMM test, and only 0.4711 → 0.5433 on HateClipSeg — and
**every one of the eight completed methods sits in the bottom fifth of it.** Three further
regularities hold across methods:

1. **Coverage degeneracy dominates the stratified numbers.** On MHC-EN test the multi-span stratum
   has a 1.6–2.8% positive base rate, where ZS-ImageBind reaches ROC 0.8197 against 0.5851 on the
   single-span stratum. Frame-level ROC on these corpora is driven by *which videos carry spans*,
   not by where inside a video the span sits.
2. **Wording matters more than method choice.** ZS-CLIP spans 0.4245–0.5573 test ROC on HateMM
   across four prompt pairs; LaGoVAD spans 0.4593–0.6432 on MHC-ZH across three paraphrases of one
   definition; LAVAD swings ±0.05 ROC on one sentence of system prompt. These ranges exceed the gaps
   between several of the methods in §1.
3. **The audio channel is the under-used one.** ZS-ImageBind's audio row is the best zero-shot cell
   on both MultiHateClip halves, and none of the published visual baselines has an audio row at all.

Consequence for any future localisation claim of ours: beating the label-free floor means clearing
ROC-AUC ~0.55–0.59, which is a low bar; the number worth quoting against is the gold-broadcast
ceiling, which none of these methods touches.

## 8. What is still running, and how to collect it

All pending runs are detached, resume-safe and start unattended. The single RTX 5090 is
serialised by `scripts/repro_campaign/gpu_queue.sh`, so the two queued methods begin by themselves
when UniTime releases the lock. **Nothing needs to be restarted.**

| method | state | launcher | log | ETA |
|---|---|---|---|---|
| UniTime | decoding the corpus, 110/3754 generations at 2026-08-21 22:12 | `scripts/repro_campaign/run_unitime_wave1.sh` | `logging/runs/repro_unitime/run.log` | ~30 h from restart |
| CLAP | FedAvg grid, HateMM chain complete | `scripts/repro_campaign/run_clap_chain.sh` | `logging/runs/repro_clap/run.log` | ~5–7 h |
| T3AL | parked in the GPU queue | `scripts/repro_campaign/t3al_launch.sh` | `logging/runs/repro_t3al/run.log` | starts when UniTime releases, then ~9 h |
| SeViLA | parked in the GPU queue | `scripts/repro_campaign/run_sevila_wave2.sh` | `logging/runs/repro_sevila/run.log` | starts when UniTime releases, then ~1–3 h |

To finish this document once a run lands: score it through the shared evaluator (each section's own
"Reproduce" block carries the exact command), then re-run `summary_table.py` as shown at the head of
this file — §1 and §2 refill from the evaluator JSON automatically. The per-method sections live in
`idea-stage/repro_campaign/sections/` (`O_mulde.md`, `P_clap.md`, `Q_t3al.md`, `R_sevila.md`,
`S_install_gated.md`) and fold into `REPRO_CAMPAIGN_RESULTS.md` as §O–§S.

### A scheduling note worth recording

UniTime's decode rate fell from **0.064 to 0.034 generations/s** over the first hour, roughly
halving, while MULDE and CLAP trained on the same card under a 10% memory cap. Part of that is
dataset composition — the run moved from 35 s MultiHateClip videos to 144 s HateMM ones, which also
trigger the expensive segment-retrieval pass — but part is straightforward SM contention. The
trade was taken deliberately: serialising everything would have finished UniTime ~8 h sooner while
leaving MULDE and CLAP entirely unstarted, so total wall-clock to *all* results is lower with the
overlap. The two large-model Wave 2 methods were **not** overlapped, on the campaign's standing
"one big model on the card at a time" rule.

One scope cut was made for the same budget reason and is recorded here as well as in §O: MULDE's
audio (`w2vemo`) and concatenated (`clip+w2vemo`) feature variants were dropped, keeping only the
headline visual `clipL336` stream. The campaign brief made those two conditional on being cheap.
The ~10 h figure the ruling quoted was an overestimate — epoch checkpoints share a single training
run, so the grid is 4 trainings per (dataset, variant), and the measured all-variants total was
~6.8 h. The cut still stands on its reason (the priority run was being starved), and the corrected
arithmetic is recorded here rather than left as a stale number. The cut was decided **before any test number existed and without
looking at any metric**, so it touches no red line.
