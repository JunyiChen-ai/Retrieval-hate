# Label-free frame-level baselines — reproducibility shortlist (2026-08-18)

Scope requested: baselines that emit a **per-frame / per-timestamp** hateful-or-anomalous score,
trained with **no labels of any kind** (video-level, frame-level, timestamp-level), published at
**CCF-A or CORE A\***, with **reachable official open-source code**. Domain is not restricted to
hateful video — video anomaly detection/localization, violence detection, harmful-moment detection
and training-free temporal grounding all count as near neighbours.

Zero training was run for this document. It is literature reconnaissance only.

---

## 0. Honest assessment of how small the intersection is

**The three-way intersection (label-free × A-venue × open code) inside the hateful-video domain is empty.**

Verified facts behind that statement:

| In-domain paper | Frame-level? | Label-free? | Venue (verified) | Verdict |
|---|---|---|---|---|
| LELA — *Towards Training-free Multimodal Hate Localisation with LLMs* (arXiv 2602.09637) | yes | yes | **arXiv only** (no venue in v1 comments) | fails venue |
| MARS — *Training-Free and Interpretable Hateful Video Detection* (arXiv 2601.15115) | no (video-level) | yes | **ICASSP 2026** (CCF-B) | fails venue + granularity |
| MultiHateLoc (arXiv 2512.10408) | yes | **no** — video-level MIL labels | **WWW 2026** (CCF-A) | fails label-free |
| HateClipSeg / DeHate / MultiHateClip | dataset papers | n/a | n/a | not methods |

So every entry below comes from the **video anomaly detection (VAD) neighbourhood**. That is the
right neighbourhood: VAD frame-level ROC-AUC / AP is exactly the protocol MultiHateLoc and LELA
adopted for hate localisation, and LELA has already ported two of the entries below (LAVAD,
URF-HVAA) onto HateMM and MultiHateClip, so the transfer is empirically demonstrated rather than
speculative.

Counting only entries that pass **all three** hard filters:

- **11 standalone A-venue papers, fully zero-label, with official code** — 6 unconditional
  anomaly-score methods (§1A) and 5 query-conditioned methods (§1B).
- **+2 boundary cases** whose checkpoints were pretrained on a large *auxiliary* temporally-labelled
  corpus but which use zero target labels (UniVTG, SeViLA) — flagged, your call.
- **+4 zero-shot baselines that exist only as rows inside A-venue papers** (ZS-CLIP, ZS-ImageBind,
  LLaVA-1.5 direct query, VideoLLaMA3+Llama). Peer-reviewed numbers, code inside an A-venue repo,
  but no paper of their own.
- **+7 normal-samples-only / unlabeled-pool methods** (§2), pending your ruling on whether
  "train on the non-hate videos" counts as label use.

So 25 rows total, of which 11 are unambiguous. Two structural observations worth carrying:

1. **The axis is young and homogeneous.** All 11 unambiguous entries are frozen-model inference
   pipelines (the sole exception being T3AL, which does discardable test-time gradient adaptation on
   the unlabeled test video), and 10 of the 11 were published in the last 36 months. Worse, five of
   the six §1A entries — LAVAD, URF-HVAA, VADTree, EventVAD, MoniTor — share one skeleton: caption
   windows → LLM scores each window → temporal smoothing. Reproducing three of those is closer to
   reproducing one pipeline three ways than to covering three independent hypotheses. That is the
   main reason the §1B query-conditioned line earns a slot in the recommended order below.
2. **Every entry scores visual frames only.** None of the 25 reads speech or on-screen text. Against
   this project's own Gate-C finding — 30.1% of misses are "on-screen text has the evidence, speech
   does not", `on_screen_text` OR 2.29 — a pure CLIP/BLIP frame-similarity curve is measuring the
   wrong channel. Whichever entry gets ported, the ASR and OCR caches on disk have to be bolted into
   the query or caption stream, or the number will be floor-level for a reason that has nothing to
   do with the method.

**Read this alongside `research-wiki/TEMPORAL_SPAN_LANDSCAPE_2026-08-18.md`**, which measured the
benchmarks these baselines would be scored on. Its finding materially changes what a reproduction is
worth: on HateMM a degenerate oracle with perfect video-level classification and *zero* localisation
ability scores frame-level AP **0.675**, above MultiHateLoc's published 0.645; on MultiHateClip the
same oracle scores 0.786 (EN) / 0.853 (ZH) against MultiHateLoc's 0.445. **HateClipSeg is the only
hateful-video benchmark where frame-level localisation is a real task** (degenerate-oracle AP 0.530,
median 3 disjoint toxic blocks per video). Consequence for this document: reproducing a baseline
*on HateMM* mostly re-measures video-level classification. Reproduce on **HateClipSeg**.

---

## 1. Fully zero-label (no target-dataset labels of any kind, not even "this video is normal")

Split into two mechanism families, because they have different interfaces:
**1A** takes no query and emits an unconditional anomaly/hate score per frame;
**1B** takes a free-text query and emits a per-frame relevance curve conditioned on it.

### 1A. Unconditional anomaly-score line

| # | Method | Venue + year | Domain | Label-free class | Per-frame output | Datasets & headline numbers | Code + status | Port cost to HateClipSeg / HateMM |
|---|---|---|---|---|---|---|---|---|
| 1 | **LAVAD** | **CVPR 2024** | VAD (surveillance/violence) | fully zero-label, training-free | LLM emits a 0–1 anomaly score per captioned frame, then refined by cross-modal similarity over a 10 s window | UCF-Crime **80.28** AUC; XD-Violence **62.01** AP / **85.36** AUC. Training-free baselines it defines: ZS-CLIP 53.16, ZS-ImageBind(I) 53.65, (V) 55.78, LLaVA-1.5 72.84 | [lucazanella/lavad](https://github.com/lucazanella/lavad) — ~151★, last push 2024-07-15, official | **LOW–MED.** Needs raw mp4 (have both), BLIP-2 ensemble + Llama-2-13B-chat + ImageBind. Only change is the prompt ("anomalous" → "hateful"). Already ported by LELA: HateMM ROC-AUC **0.6163** / PR-AUC 0.5781; MHC 0.6302 / 0.5865. 13B in fp16 = 26 GB — tight on a single 32 GB card, use 4-bit or swap the LLM |
| 2 | **URF-HVAA** (Lin et al., "unified reasoning framework") | **NeurIPS 2025** (poster) | VAD + localisation + explanation | fully zero-label, training-free | per-frame score from LLM scoring of per-16-frame captions, then a suspicious-window pass and score-gated refinement | UCF-Crime **84.28** AUC; XD-Violence **91.34** AUC / **68.07** AP; UBnormal 68.98; MSAD 85.9 / 76.4 | [Rathgrith/URF-HVAA](https://github.com/Rathgrith/URF-HVAA) — ~10★, last push 2025-12-10, official, full code + precomputed captions/scores | **LOW.** Built directly on LAVAD's codebase and data layout, so porting LAVAD gets this nearly free. Uses VideoLLaMA3-7B + Llama-3.1-8B — fits one 3090, comfortable on a 5090. Captioning ≈ 20 h / 3090 for UCF-Crime; HateClipSeg is 395 videos so far cheaper. Already ported by LELA: HateMM PR-AUC **0.6239** / ROC 0.5674; MHC 0.6147 / 0.5626 |
| 3 | **VADTree** | **NeurIPS 2025** (poster) | VAD | fully zero-label, training-free | multi-granularity per-node scores from a hierarchical event tree, fused to a per-frame score | UCF-Crime **84.74** AUC; XD-Violence **67.82** AP / **90.44** AUC (VADTree\* 68.85 / 90.55) | [wenlongli10/VADTree](https://github.com/wenlongli10/VADTree) — ~19★, last push 2026-06-09, official | **MED.** Extra dependency: a pretrained Generic Event Boundary Detection model to build the tree. Attractive for hate video precisely because hate spans have very variable length (HateMM spans range from a few s to whole-video), which is the failure mode VADTree targets |
| 4 | **EventVAD** | **ACM MM 2025** | VAD | fully zero-label, training-free | event-segment score from a 7B MLLM, broadcast to frames | UCF-Crime **82.03** AUC; XD-Violence **64.04** AP / **87.51** AUC | [YihuaJerry/EventVAD](https://github.com/YihuaJerry/EventVAD) — ~536★, last push 2025-07-09, official | **MED.** Two conda envs (event segmentation + scoring). Needs RAFT optical flow + CLIP features + a dynamic graph before the MLLM. The optical-flow stage encodes a surveillance prior (motion discontinuity = event boundary) that is weaker in talking-head hate video |
| 5 | **MoniTor** | **NeurIPS 2025** | VAD, **online/streaming** | fully zero-label, training-free | streaming per-frame score from a VLM with an LSTM-style predictor and a scoring queue | Online setting: UCF-Crime **82.57** AUC; XD-Violence **55.01** AP / **79.11** AUC (vs online-LAVAD 76.06 / 52.63 / 76.01) | [YsTvT/MoniTor](https://github.com/YsTvT/MoniTor) — ~30★, last push 2025-10-01, official | **MED.** Only entry designed for causal/streaming scoring. Interesting for hate video if you care about "flag as it plays", less so for offline benchmark AUC where the offline methods dominate |
| 6 | **O-VAD** | **ECCV 2026** (CORE A\*) | **industrial** VAD | fully zero-label, training-free, agentic | tracks per-object state trajectories, flags abnormal objects in grounded frames + emits a report | three IVAD datasets; beats frontier VLMs and fine-tuned VAD. Exact numbers not in the abstract | [yuanapril/OVAD-ECCV26](https://github.com/yuanapril/OVAD-ECCV26) + [project page](https://o-vad.github.io) — 0★, created 2026-06/07, official but brand-new | **HIGH / poor fit.** The mechanism is object state evolution under physics and procedural constraints. Hateful video has no such object dynamics — the signal is speech, on-screen text and symbols. Listed for completeness only |
| 7 | **GtS** (Glance-then-Scrutinize, VAGU) | **AAAI 2026** | VAD grounding + understanding | fully zero-label, training-free, text-prompt guided | coarse glance pass then fine scrutiny pass over candidate intervals; outputs time intervals + scores | Introduces the VAGU benchmark; journal extension arXiv 2608.11260 | **No public repo found** (GitHub search for VAGU / Glance-then-Scrutinize returned nothing) | **BLOCKED** — fails filter 3 today. Listed because it is A-venue + label-free and the prompt-guided interface would accept "a moment containing hate speech" directly. Re-check for a code release |
| 8 | **ZS-CLIP** (frame-text cosine vs. two prompts) | defined as a baseline **inside LAVAD, CVPR 2024** | VAD | fully zero-label | cosine similarity of each frame embedding against "normal"/"anomalous" text embeddings | UCF-Crime 53.16 AUC; XD-Violence 17.83 AP / 38.21 AUC. On hate: HateMM ROC 0.5367, MHC 0.5449 (LELA) | code lives inside [lucazanella/lavad](https://github.com/lucazanella/lavad) | **NEAR-ZERO.** `data/CLIP_Embedding/HateClipSeg/dense4fps_clipL336/*.npy` already holds dense 4 fps CLIP-L/336 frame features for all 395 videos. This is a dot product plus a text encoder call — hours, not days. Chance-level performance, so it is a floor, not a competitor |
| 9 | **ZS-ImageBind** (image and video variants) | baseline inside **LAVAD, CVPR 2024** | VAD | fully zero-label | cosine similarity in ImageBind space, image-level or 10 s video-clip-level | UCF-Crime 53.65 / 55.78 AUC; XD-Violence 27.25 / 25.36 AP. On hate: HateMM ROC 0.5683, MHC 0.5753 (LELA) | inside [lucazanella/lavad](https://github.com/lucazanella/lavad) | **LOW.** Needs the ImageBind checkpoint and a re-encode from raw frames (existing cache is CLIP, not ImageBind). Adds the audio modality for free, which matters for hate video — HateMM hate is often carried by speech |
| 10 | **LLaVA-1.5 direct query** | baseline inside **LAVAD, CVPR 2024** | VAD | fully zero-label | MLLM asked per frame/segment to emit a 0–1 score | UCF-Crime 72.84 AUC; XD-Violence 50.26 AP / 79.62 AUC. On hate: HateMM ROC 0.5529, MHC 0.5438 (LELA) | inside [lucazanella/lavad](https://github.com/lucazanella/lavad) | **LOW.** This is the "zero-shot MLLM scores each segment" baseline the brief asked about. Note it exists in the literature **only as a baseline defined inside A-venue papers**, never as its own A-venue paper. `data/MLLM_scores/{HateClipSeg,HateMM,...}` suggests scaffolding already exists |
| 11 | **VideoLLaMA3-7B + Llama-3.1-8B (ZS)** | baseline inside **URF-HVAA, NeurIPS 2025** | VAD | fully zero-label | caption-then-score, no refinement | MSAD 78.7 AUC / 68.5 AP | inside [Rathgrith/URF-HVAA](https://github.com/Rathgrith/URF-HVAA) | **LOW.** The natural ablation floor for entry 2 |

### 1B. Query-conditioned line (free text → per-frame relevance curve)

These come from training-free temporal grounding / zero-shot temporal action localization /
training-free keyframe selection. All of them internally build a per-frame curve; some expose it as
the output, some only use it to pick frames.

| # | Method | Venue + year | Label-free class | Per-frame output | Datasets & headline numbers | Code + status | Port cost to HateClipSeg / HateMM |
|---|---|---|---|---|---|---|---|
| 12 | **UniVTG** (zero-shot inference) | **ICCV 2023** | **boundary** — zero target labels, but the checkpoint is pretrained on 4.2 M temporal annotations (Ego4D narrations, VideoCC, 1.5 M CLIP-teacher pseudo-curves) | **best output form in the document**: three heads per 2 s clip — foreground probability, boundary offsets, and an explicit **saliency score** curve | ZS QVHighlights R1@0.5 25.16, mAP-avg 10.87, HD mAP 35.96; ZS Charades-STA R@0.5 **25.22**, mIoU 27.12; TACoS collapses (R@0.5 1.27); ZS highlight YouTube-HL mAP 53.9, TVSum top-5 mAP 67.2 | [showlab/UniVTG](https://github.com/showlab/UniVTG) — ~379★ (most-starred here), last push 2024-05-08, official, checkpoint released | **LOWEST of section 1B.** Inputs are CLIP ViT-B/32 + SlowFast R-50 features — no LLM, no API. Free-form query via the CLIP text encoder (77-token limit). The saliency head is natively the curve you want to score against `gold_segments.json` |
| 13 | **TFVTG** | **ECCV 2024** (CORE A\*) | fully zero-label, training-free | 3 fps BLIP-2 Q-Former query-frame similarity, Gaussian-smoothed, first-differenced into a dynamic (onset) score plus a static (in- vs out-of-segment) score | Charades-STA R@1 IoU 0.3/0.5/0.7/mIoU **67.04 / 49.97 / 24.32 / 44.51**; ActivityNet-Captions 49.34 / 27.02 / 13.39 / 34.10; OOD-Charades R@0.5 45.9 — best zero-shot grounding numbers found | [minghangz/TFVTG](https://github.com/minghangz/TFVTG) — ~57★, last push 2024-09-13, official | **MED.** Needs a GPT-4-Turbo call per query for sub-event decomposition. Two structural mismatches: the scoring signal is purely visual BLIP-2 similarity, and it assumes **one** target moment per query while HateClipSeg has a median of 3 disjoint toxic blocks |
| 14 | **T3AL** | **CVPR 2024** | fully zero-label on target — does gradient test-time adaptation on the single **unlabeled** test video (≤50 steps, only prompt vectors + temperature), then discards it | per-frame cosine similarity between CoCa frame embeddings and the text embedding → 1-D curve → threshold → text-guided region suppression via frame captions | THUMOS14 50/50 mAP avg **10.4** (@0.5 = 8.9); 75/25 avg 9.2; ActivityNet-1.3 avg **14.3**. Naive CoCa frame classification gets 1.1–3.4, so 3–9× above the naive floor; base-class-trained STALE gets 22.2–23.8 | [benedettaliberatori/T3AL](https://github.com/benedettaliberatori/T3AL) — ~79★, last push 2024-09-11, official | **MED.** Single model (CoCa ViT-L/14) does classification, proposals and captioning; runs on one V100. Catch: the video-level pseudo-labeling step takes an **argmax over a closed class list**, so you must supply a candidate set (`{racist statement, misogynistic statement, neutral talking, …}`) rather than one free query |
| 15 | **SeViLA** (Localizer) | **NeurIPS 2023** | **boundary, stricter than UniVTG** — the Localizer is pretrained on QVHighlights videos + queries + **human temporal span labels**; zero target labels | frozen BLIP-2 prompted per frame with a yes/no relevance question; the "yes" logit is the per-frame score (32 sampled frames) | Headline results are QA accuracy (NExT-QA, STAR, How2QA, VLEP, TVQA) + QVHighlights moment retrieval; no comparable Charades R@1 IoU 0.5 | [Yui010206/SeViLA](https://github.com/Yui010206/SeViLA) — ~199★, last push 2024-01-14 (stale), official | **LOW.** Cheapest *prompted-VLM* per-frame scorer here, and the interface is ideal: "Does this frame show someone making a racist statement?" is directly expressible as the localizer prompt |
| 16 | **AKS** (Adaptive Keyframe Sampling) | **CVPR 2025** | fully zero-label, plug-and-play, nothing trained | explicit per-frame prompt-relevance score `s(Q, F_t)` at 1 fps (the paper plots these curves), consumed by a recursive bin-splitting relevance/coverage tradeoff | **No IoU/mAP numbers at all** — evaluated only as downstream QA: LLaVA-Video-7B + AKS gets LongVideoBench val 62.7, Video-MME 65.3 | [ncTimTang/AKS](https://github.com/ncTimTang/AKS) — ~231★, last push 2025-12-19 (actively maintained), official | **LOW.** BLIP-ITM score, trivially accepts arbitrary text. Main weakness: the curve is validated only as "helps QA", never as a temporal prediction |
| 17 | **BOLT** | **CVPR 2025** | fully zero-label, inference-time only | 1 fps CLIP-L/14 query-frame cosine similarity, sliding-window smoothed, treated as a density and sampled by **inverse transform sampling** | Video-MME 53.8 → **56.1**; MLVU 58.9 → **63.4**. Again **no IoU/mAP** | [sming256/BOLT](https://github.com/sming256/BOLT) — ~55★, last push 2026-02-05, official | **LOW.** The ITS formulation is directly reusable if you want soft segment weights rather than hard top-k |
| 18 | **T\*** (temporal search) | **CVPR 2025** | fully zero-label at inference (the LV-Haystack dataset ships human annotations; the method does not use them) | maintains and iteratively refines a **temporal sampling distribution over all frames**, by recasting temporal search as spatial search with an open-vocab detector on tiled frame grids | LV-Haystack temporal/visual F1 (prior search SOTA reached only 2.1% temporal F1 on the LongVideoBench subset); GPT-4o 50.5→53.1, LLaVA-OV-72B 56.5→62.4 | [mll-lab-nu/TStar](https://github.com/mll-lab-nu/TStar) — ~97★, last push 2026-03-23, official | **DO NOT PORT.** Step one asks a VLM to extract a *target object* and *cue object* from the question, then searches for those objects. "Someone making a racist statement" has no detectable object |

### Section 1 notes

- Entries 8–11 are **baselines defined inside A-venue papers**, not standalone A-venue papers.
  They pass the spirit of the venue filter (the numbers are peer-reviewed and the code sits in an
  A-venue repo) but not the letter. Flagging so the call is yours.
- Answering the brief's question directly: **there is no CCF-A / CORE-A\* paper whose contribution
  is "zero-shot MLLM per-segment scoring for harmful/hateful video."** That configuration exists
  only (a) as a baseline row inside VAD papers, and (b) as arXiv/ICASSP-tier hate-domain papers
  (LELA, MARS).
- Only 3 of the 7 entries in §1B (TFVTG, T3AL, UniVTG) report IoU-based localisation numbers at
  all. AKS, BOLT, T\* and SeViLA are frame-selection-for-QA papers — their per-frame curves are
  validated as "helps downstream QA accuracy", never as temporal predictions. Treat their curves as
  plausible but unproven.
- §1B methods need a **query**, and hate is not one concept. HateClipSeg's `gold_segments.json`
  carries a 6-way label vector per segment, so the natural query set is one prompt per category
  rather than a single "is this hateful" string. That also fixes T3AL's closed-class-list
  requirement for free.

---

## 2. Only normal samples (one-class / semi-supervised) — needs your ruling

These train on the target domain using **only non-anomalous videos**. No anomaly labels are ever
seen, but asserting "this pool is clean" is itself label information. For hateful video the
analogue is "train on the non-hate videos only", which on HateMM means consuming the video-level
`label == 0` annotation.

| # | Method | Venue + year | Label-free class | Per-frame output | Datasets & numbers | Code + status | Port cost |
|---|---|---|---|---|---|---|---|
| 19 | **MULDE** | **CVPR 2024** | normal-only (B1) | multiscale log-density via denoising score matching, GMM-combined into a per-frame score | frame-centric: ShanghaiTech 81.3, **UCF-Crime 78.5**, UBnormal 72.8 micro-AUC; object-centric: Ped2 99.7 / Avenue 94.3 / SHTech 86.7 | [jakubmicorek/MULDE-…](https://github.com/jakubmicorek/MULDE-Multiscale-Log-Density-Estimation-via-Denoising-Score-Matching-for-Video-Anomaly-Detection) — ~50★, last push 2024-06-19, official | **LOWEST in the whole document.** Frame-centric mode is a shallow MLP + GMM over an arbitrary per-clip feature vector. Swapping Hiera-L features for the CLIP-L/336 4 fps cache on disk is a config change. Proven on UCF-Crime, so it is not Ped2-only |
| 20 | **CLAP** | **CVPR 2024** | **fully unlabeled pool (B2)** — no "these are normal" assumption | segment score from pseudo-labels generated by internal clustering + hypothesis testing, broadcast to frames | UCF-Crime **80.91** AUC, XD-Violence **81.71** AUC | [AnasEmad11/CLAP](https://github.com/AnasEmad11/CLAP) — ~20★, last push 2024-09-30, official | **LOW.** Interface is a per-segment feature vector only. Arguably belongs in section 1: the paper states videos are used without any labels. Placed here because the mechanism still assumes anomalies are the minority of the pool, which is roughly true on HateMM/HateClipSeg but worth checking |
| 21 | **AnomalyRuler** | **ECCV 2024** | normal-only, **few-shot** (~10 normal reference frames) | per-frame binary decision + score, from LLM-induced normality rules | Ped2 97.9, Avenue 89.7, SHTech 85.2, UBnormal 71.9 AUC. Cross-dataset zero-shot variant: UBnormal 65.4 | [Yuchen413/AnomalyRuler](https://github.com/Yuchen413/AnomalyRuler) — ~105★, last push 2024-12-16, official | **MED.** The rule-induction idea maps cleanly ("induce rules for what a non-hateful video looks like"), and the normal-sample budget is tiny. But the rules are scene-descriptive, tuned for pedestrian surveillance; hate is semantic, not scene-statistical. A cross-domain rule-transfer variant already exists in the repo |
| 22 | **Joint-VAD** | **ECCV 2024** | fully unlabeled pool (B2) | STG-NF log-likelihood + RTFM snippet score, interleaved via adaptive thresholding on pseudo-labels | ShanghaiTech 82.57 / 88.52 AUC; UBnormal 74.82 / 63.26 | [benedictstar/Joint-VAD](https://github.com/benedictstar/Joint-VAD) — ~10★, last push 2025-10-25, official | **HIGH.** Requires human pose/skeleton data. Pedestrian-surveillance prior, poor fit |
| 23 | **MemAE** | **ICCV 2019** | normal-only | memory-augmented autoencoder reconstruction error per frame | Avenue 83.3, SHTech 71.2 AUC | [donggong1/memae-anomaly-detection](https://github.com/donggong1/memae-anomaly-detection) — ~500★, last push 2022-08-01, official | **MED.** Simplest architecture here (raw frame cubes, no flow, no detector) but reconstruction error on RGB is meaningless for hate — a hateful talking-head frame reconstructs as easily as a benign one. Include only as a negative control |
| 24 | **MNAD** | **CVPR 2020** | normal-only | reconstruction/prediction error + memory compactness term per frame | Avenue 88.5, SHTech 70.5 AUC | [cvlab-yonsei/MNAD](https://github.com/cvlab-yonsei/MNAD) — ~360★, last push 2024-11-04, official | **MED.** Same objection as MemAE |
| 25 | **AED-MAE** | **CVPR 2024** | normal-only | teacher–student reconstruction discrepancy + synthetic-anomaly head, per frame | Ped2 / Avenue / SHTech / UBnormal; 1655 FPS. Numbers not verified from a primary source | [ristea/aed-mae](https://github.com/ristea/aed-mae) — ~57★, last push 2024-11-28, official | **MED.** Cheapest modern one-class model to actually run; same appearance-prior objection |

Also verified in this class but weaker fit, listed without full rows: **SSPCAB** (CVPR 2022, a
plug-in block not a detector, [ristea/sspcab](https://github.com/ristea/sspcab) ~160★);
**STG-NF** (ICCV 2023, human-pose-graphs only — effectively unportable,
[orhir/STG-NF](https://github.com/orhir/STG-NF) ~115★); **SSMTL** (CVPR 2021, needs an object
detector, [lilygeorgescu/AED-SSMTL](https://github.com/lilygeorgescu/AED-SSMTL) ~60★);
**HF²-VAD** (ICCV 2021, flow + detector, [LiUzHiAn/hf2vad](https://github.com/LiUzHiAn/hf2vad)
~140★); **Jigsaw-VAD** (ECCV 2022, object detector,
[gdwang08/Jigsaw-VAD](https://github.com/gdwang08/Jigsaw-VAD) ~57★); **Future Frame Prediction**
(CVPR 2018, TF 1.x, [StevenLiuWen/ano_pred_cvpr2018](https://github.com/StevenLiuWen/ano_pred_cvpr2018)
~460★); **MA-PDM** (AAAI 2025, [henrryzh1/MA-PDM](https://github.com/henrryzh1/MA-PDM) ~16★);
**ADSM** (ICCV 2025, [Bbeholder/ADSM](https://github.com/Bbeholder/ADSM) 3★, README is a stub);
**LPGB** (ICLR 2025, [AllenYLJiang/Local-Patterns-Generalize-Better](https://github.com/AllenYLJiang/Local-Patterns-Generalize-Better)
4★, low-confidence official). Numbers for the last four were not verified from a primary source.

---

## 3. Boundary cases — your ruling required

| Method | Venue | What the supervision actually is | Verdict I would default to |
|---|---|---|---|
| **LAVIDA** — *No Need For Real Anomaly* | **CVPR 2026** | Trains end-to-end, but **only on synthetic pseudo-anomalies** built by pasting segmented objects. Zero real VAD data, zero target labels. Frame-level *and* pixel-level output. Code: [VitaminCreed/LAVIDA](https://github.com/VitaminCreed/LAVIDA) ~27★, last push 2026-02-25, official | **Counts as label-free.** A training run exists but consumes no target labels. The pseudo-anomaly generator ("paste an out-of-context object") is a visual-oddity prior that does not obviously produce hate cues, so expect the port to underperform |
| **HiProbe-VAD** | **ACM MM 2025** | "Tuning-free" MLLM, but trains a **logistic-regression scorer on ~1 % of the labelled training set** and sets its threshold from that same few-shot set. Verified from §4.1.2 of the paper | **Fails filter 1.** Uses video-level labels, just very few. Worth flagging because the numbers are strong (UCF-Crime 86.72 with InternVL2.5, XD 82.15 AP) and 1 % is a cheap concession if you ever relax the rule. Code: [CebCai/HiProbeVAD](https://github.com/CebCai/HiProbeVAD) 4★, last push 2026-02-02 |
| **VERA** | **CVPR 2025** | "Verbalized learning" optimises guiding questions using **coarsely labelled (video-level) training data**. Confirmed independently: URF-HVAA's Table 2 marks VERA `Zero-shot ✗ / Training-free ✓` | **Fails filter 1.** Rejected. Code: [vera-framework/VERA](https://github.com/vera-framework/VERA) ~86★, last push 2026-03-23 |
| **C2FPL** | **WACV 2024** | Genuine fully-unsupervised (B2) on precomputed features, official code [AnasEmad11/C2FPL](https://github.com/AnasEmad11/C2FPL) ~21★ | **Fails filter 2 only.** WACV is CORE A, not A\*, and not CCF-A. Cheap to add if the venue bar moves |
| **AnyAnomaly** | **WACV 2026** | Genuinely zero-shot, customisable-criterion LVLM VAD — the mechanism ("define your own anomaly in text") is the best conceptual fit for "define hate in text" of anything found | **Fails filter 2 only.** WACV. Same note as above |

### 3.1 The zero-shot *image*-anomaly line — checked, and it does not belong on the list

The brief flagged AnomalyCLIP and the zero-shot anomaly-detection line. Venues all verified from
each paper's own arXiv comment field, and every one clears the venue bar:

| Method | Venue (verified) | Supervision | Output granularity |
|---|---|---|---|
| **WinCLIP** | **CVPR 2023** | zero-shot variant is genuinely training-free (window-based CLIP + hand-crafted prompt ensembles); few-shot variant uses normal reference images | per-pixel anomaly map |
| **AnomalyCLIP** | **ICLR 2024** | learns object-agnostic **text prompts on an auxiliary labelled AD dataset**, then transfers zero-shot to the target | per-pixel |
| **AnomalyGPT** | **AAAI 2024** | trains on simulated anomalies + auxiliary data with an LLM decoder | per-pixel + text |
| **AdaCLIP** | **ECCV 2024** | hybrid learnable prompts trained on auxiliary labelled AD data | per-pixel |
| **InCTRL** | **CVPR 2024** | in-context residual learning trained on auxiliary data; needs few-shot normal prompts at test time | image + region |

**All five are excluded, for two independent reasons.**

1. **Granularity.** Every one produces a per-pixel spatial anomaly map for a still image. None is a
   video method, and I found no temporal extension for any of them. The brief asks for a per-frame
   or per-timestamp score; converting a defect segmentation map into a hate-over-time curve is not a
   port, it is a different method.
2. **Domain.** These detect *structural and textural defects* on industrial parts (MVTec-AD, VisA)
   and medical scans — scratches, dents, missing components. Hateful video is a semantic and social
   judgement about speech, symbols and on-screen text. The shared word "anomaly" is doing all the
   work in that analogy and none of the mechanism transfers.

Four of the five would also land in this boundary section on supervision grounds anyway
(auxiliary-dataset prompt training). Recording the check so this axis does not get re-opened.
Caveat: the GitHub API rate-limited before their repos could be enumerated, so their code status is
**not** verified in this session — but since they are excluded on granularity and domain regardless,
that verification is not worth spending on.

---

## 4. Explicitly rejected, with reasons

**Rejected on label-free (uses video-level anomaly labels):** the entire MIL-ranking lineage from
Sultani et al. onward — MIST, RTFM, MGFN, UR-DMU, S3R, CLIP-TSA, **VadCLIP**, STPrompt, OVVAD,
Holmes-VAD / Holmes-VAU. Also **MultiHateLoc** (WWW 2026), which is the in-domain frame-level
weakly-supervised SOTA and therefore the number to beat, not a baseline to reproduce
(HateMM frame mAP 0.645 / AUC 0.799; MultiHateClip 0.445 / 0.750).

Two more near-misses on this filter, both from the grounding line:

- **STALE** (ECCV 2022, "Zero-Shot Temporal Action Detection via Vision-Language Prompting",
  [sauradip/STALE](https://github.com/sauradip/STALE) ~116★). "Zero-shot" here means *unseen
  classes*; it still fine-tunes on the target dataset's **base classes with full timestamp labels**.
  T3AL's §3 shows it drops >15 mAP cross-dataset.
- **PSVL** (ICCV 2021, [gistvision/PSVL](https://github.com/gistvision/PSVL) ~48★). Label-free
  pseudo-supervision, but it *trains an NLVL model on the target video collection*, and its
  pseudo-queries are detector nouns plus co-occurrence verbs — nowhere near "racist statement".
  TFVTG measures it at Charades OOD R@0.5 = **3.0**, i.e. it collapses off-distribution.

**Rejected on venue:** LELA (arXiv), MARS (ICASSP 2026), SlowFastVAD (arXiv), Flashback (arXiv),
AnyAnomaly (WACV 2026), C2FPL (WACV 2024), VLAVAD (BMVC 2024), CKNN (CIKM 2024),
**ZeroTA** (DBLP shows CoRR only), **FreeZAD** (CoRR only), **VTG-GPT** (CoRR only; journal version
is MDPI Applied Sciences — would otherwise qualify at Charades R@0.5 43.68), **Luo et al.**
zero-shot video moment retrieval (WACV 2024, Charades R@0.5 42.93), *Unsupervised Video Highlight
Detection from Audio and Visual Recurrence* (WACV 2025), *Unsupervised Modality-Transferable Video
Highlight Detection* (IEEE TIP 2024), and a long tail of TCSVT / Pattern Recognition /
Neurocomputing journal work.

**Rejected on no code:** **GtS/VAGU** (AAAI 2026), **GCL** (CVPR 2022), **Self-Trained Deep Ordinal
Regression** (CVPR 2020), **LANP** (ECCV 2024), **PZVMR** (ACM MM 2022). All five have verified
A-venues and are genuinely label-free or unsupervised; exhaustive GitHub search found no official
*or* third-party implementation.

**Rejected on task shape:** *Contrastive Learning for Unsupervised Video Highlight Detection*
(CVPR 2022) — label-free, but trains on target videos and has **no text-query mechanism**, so it
cannot be pointed at hate. **VideoTree** (CVPR 2025,
[Ziyang412/VideoTree](https://github.com/Ziyang412/VideoTree) ~167★) — training-free query-adaptive
clustering, but produces cluster-level relevance for QA, not a temporal curve, and reports no
localisation metrics.

---

## 5. What is already on disk (grounds the port-cost column)

- `~/data/HateClipSeg/videos/` — 395 raw mp4.
- `~/data/HateMM/video/` — 1083 raw mp4.
- `data/CLIP_Embedding/HateClipSeg/dense4fps_clipL336/*.npy` — **dense 4 fps CLIP-L/336 frame
  features for all 395 HateClipSeg videos**, plus parallel `dense4fps_*` caches for BERT, OCR-BERT,
  VideoMAE-v2-g, wav2vec-emotion and several fusion variants.
- `data/gt/HateClipSeg/gold_segments.json` — segment boundaries with 6-way label vectors;
  `video_durations.jsonl`.
- `data/gt/HateMM/hate_spans.json` — per-video `duration` + list of `[start, end]` hate spans +
  video label. Frame-level ROC-AUC / AP is directly computable from both.
- `data/ASR/{HateClipSeg,HateMM,MHC,MHC_zh}` — whisper-large-v3 transcripts, windowed (K=4, K=30).
- `data/OCR/{HateClipSeg,HateMM,…}` — OCR windows (channel unblocked 2026-08-08).
- `data/MLLM_scores/{HateClipSeg,HateMM,MHC,MHC_zh}` — existing MLLM scoring scaffolding.

Consequence: the CLIP-similarity baselines (entry 8) and the feature-vector one-class models
(entries 19–20) can run **without touching a video file**. The captioning pipelines (entries 1–5)
and the grounding methods (12–18) must re-read raw mp4, which is local-only under the data-boundary
rule; the Claude-API exemption of 2026-08-07 does allow raw frames into the Claude API if you want a
hosted captioner instead of a local BLIP-2/VideoLLaMA3.

---

## 6. Recommended reproduction order

**1. LAVAD (CVPR 2024) → URF-HVAA (NeurIPS 2025) as one work item.**
URF-HVAA is built on LAVAD's codebase and reuses its data layout, so the expensive part — frame
extraction, captioning, the per-video score-file format — is paid once and yields two A-venue
label-free frame-level baselines. Both already have published HateMM and MultiHateClip frame-level
numbers from LELA (LAVAD ROC-AUC 0.6163 / URF-HVAA PR-AUC 0.6239 on HateMM), which gives a
reproduction target to check your port against instead of flying blind. URF-HVAA runs on a single
3090-class GPU (VideoLLaMA3-7B + Llama-3.1-8B), so it fits the 5090 workstation without
quantisation gymnastics. Neither has ever been run on **HateClipSeg**, which is the segment-level
dataset — that is an unclaimed number.

**2. UniVTG zero-shot inference (ICCV 2023) — the query-conditioned axis.**
Cheapest real method in the document: CLIP ViT-B/32 + SlowFast features, no LLM, no paid API,
released checkpoint, 379★. Its saliency head *is* a per-2s-clip score curve, so no output adapter is
needed — it lines up directly against `gold_segments.json`. Two reasons it earns a slot over the
higher-scoring VAD entries: it is a genuinely different mechanism (query-conditioned rather than
unconditional anomaly scoring), so it is not a fourth rerun of the caption-then-LLM-score skeleton;
and per `TEMPORAL_SPAN_LANDSCAPE_2026-08-18.md` "query-conditioned grounding over a hateful video"
is an **empty slot** in the field. Use HateClipSeg's 6-way segment label vector as the query set —
one prompt per category — rather than a single "is this hateful" string. Caveat to record before
running: it is the boundary case (checkpoint pretrained on 4.2 M auxiliary temporal annotations, zero
target labels), and its TACoS collapse (R@0.5 = 1.27) shows it degrades badly off-distribution.

**3. ZS-CLIP frame-similarity floor (baseline inside LAVAD, CVPR 2024).**
Near-zero cost: the dense 4 fps CLIP-L/336 cache for all 395 HateClipSeg videos is already on disk,
so this is a text-encoder call plus a dot product — hours, no GPU queue, no video reads. It
establishes the chance-level floor (expect ROC-AUC ≈ 0.53–0.55, matching LELA's HateMM 0.5367) that
every later claim of "our method localises hate" must clear. It is third by importance and first by
schedule: run it while item 1 is still extracting frames.

**Pending your §2 ruling, the fourth pick is MULDE frame-centric (CVPR 2024).** It is the only entry
whose interface is literally "an array of per-clip feature vectors" — a shallow MLP plus GMM trained
by denoising score matching, consuming the existing CLIP cache as-is, minutes of compute. It reports
UCF-Crime 78.5 micro-AUC, so it is not restricted to Ped2-style fixed-camera scenes. It needs the
one-class ruling, because training on HateMM's `label == 0` pool consumes the video-level label. If
you rule that out, substitute **CLAP** (CVPR 2024), which makes no clean-normal assumption and has
the same feature-vector interface.

Deliberately *not* in the top 3:

- **VADTree and EventVAD** have the strongest raw VAD numbers, but each adds a non-trivial external
  dependency — a GEBD checkpoint and a RAFT optical-flow stage — and those dependencies are exactly
  the surveillance-motion priors least likely to survive the move to talking-head hate video.
- **TFVTG** has the best zero-shot grounding numbers in §1B, but it bills a GPT-4-Turbo call per
  query and assumes one target moment per query, while HateClipSeg's median video has 3 disjoint
  toxic blocks.
- The **reconstruction-based one-class family** (MemAE / MNAD / AED-MAE): RGB reconstruction error
  is close to uninformative about hate — a hateful talking head reconstructs exactly as well as a
  benign one. Worth running once as a negative control, never as a contender.
- **T\*** should not be ported at all: its first step extracts a target object from the query, and
  "someone making a racist statement" has no detectable object.

One caveat that applies to all three picks: every method in this document scores **visual frames
only**. The ASR and OCR caches already on disk are not optional extras here — without them the
curve is blind to the modality that Gate-C identified as the dominant evidence gap.

---

## 7. Direct answers to the starting points in the brief

| Starting point | Finding |
|---|---|
| "LAVAD probably qualifies" | **Confirmed.** CVPR 2024, fully training-free, official code, 151★. Entry 1. Better still, its NeurIPS 2025 successor URF-HVAA reuses its codebase, and both already have published HateMM/MultiHateClip frame-level numbers via LELA |
| "Is there an A-venue version of zero-shot MLLM per-segment scoring (Qwen-style per-window QA)?" | **No.** It exists only as *baseline rows* inside A-venue VAD papers (LLaVA-1.5 direct query in LAVAD; VideoLLaMA3+Llama in URF-HVAA; GLM-4.1V-9B-Thinking ZS-CoT in URF-HVAA's Table 2 at UCF 61.80 / XD 72.73 AUC), and as arXiv/ICASSP-tier hate-domain work (LELA, MARS). No A-venue paper claims it as its contribution |
| "VadCLIP is weakly supervised — is there an A-venue zero-shot CLIP-scoring line?" | **Confirmed VadCLIP is out** (video-level labels). The zero-shot CLIP line exists but only as the ZS-CLIP baseline row (entry 8), scoring near chance: UCF 53.16 AUC, HateMM ROC 0.5367. The genuine A-venue query-conditioned CLIP methods are in §1B — UniVTG (ICCV 2023) and BOLT (CVPR 2025) |
| "AnomalyCLIP / zero-shot anomaly line" | **Checked and excluded — see §3.1.** WinCLIP (CVPR 2023), AnomalyCLIP (ICLR 2024), AnomalyGPT (AAAI 2024), AdaCLIP (ECCV 2024), InCTRL (CVPR 2024) all clear the venue bar, but all five are **still-image, per-pixel defect detectors** with no temporal extension, and four of the five train prompts on an auxiliary labelled dataset. Excluded on granularity and domain, not just supervision |
| "arXiv 2604.09327 (SEBB transplanted to VAD) — check its supervision" | **The ID does not match that description.** 2604.09327 is *From Frames to Events: Rethinking Evaluation in Human-Centric Video Anomaly Detection* (Rashvand, Yao, Danesh Pazho, Rahimi Ardabili, Tabkhi) — a **pose-based VAD evaluation-methodology paper** that ports Temporal Action Localization metrics to event-level VAD scoring. **No mention of SEBB anywhere.** It proposes no detector, so it is not a baseline candidate. It is however directly relevant to the degenerate-benchmark problem in `TEMPORAL_SPAN_LANDSCAPE_2026-08-18.md` — event-level rather than frame-level scoring is exactly the fix for "one long contiguous span makes frame-AUC meaningless". Flagging with the standing constraint in mind: this project does not write metric papers, so treat it as a tool, not a direction |
| "Training-free temporal grounding counts as a near neighbour" | **Yes, and it is the more interesting half.** §1B has 5 fully-zero-label A-venue methods with code plus 2 boundary cases. It is a genuinely different mechanism from the VAD line and, per the landscape doc, query-conditioned grounding over hateful video is unoccupied |

---

## 8. Verification status per claim

- **Read method / primary source verified:** LAVAD (result tables read from the CVPR PDF),
  EventVAD (Tables 1–2 read from the arXiv PDF), URF-HVAA (Table 2 read from the arXiv PDF),
  VADTree (result table read), MoniTor (result table read), HiProbe-VAD (§4.1.2 supervision claim
  read directly), LELA (Tables 1–2 read — source of all HateMM/MHC frame-level numbers here),
  and from §1B: TFVTG, T3AL, UniVTG, AKS, BOLT, SeViLA (supervision sections read explicitly),
  MULDE and CLAP (methods read via ar5iv).
- **Venue verification.** DBLP was used where reachable; it returned HTTP 503 for a large part of
  this session, so the fallback was the paper's own arXiv `comment` field (which states the
  acceptance venue) or the official proceedings / project page. Every venue claim here traces to one
  of those three. No venue in this document is a guess.
- **Repo checked via the GitHub API** (stars, `pushed_at`, file listing): every repo linked above.
  Two findings worth recording: `Rathgrith/URF-ZS-HVAA` has been **renamed to `Rathgrith/URF-HVAA`**
  (the NeurIPS project page still advertises the dead URL), and the GitHub search API rate-limited
  near the end of the session, so O-VAD's repo contents were not enumerated — only its existence and
  description were confirmed.
- **Read abstract only:** the §2 "weaker fit" tail (SSPCAB, STG-NF, SSMTL, HF²-VAD, Jigsaw-VAD,
  Future Frame Prediction, MA-PDM, ADSM, LPGB), LAVIDA, T\*, and O-VAD.
- **Unverified numbers, explicitly flagged:** AED-MAE, MA-PDM, ADSM, LPGB, O-VAD. Star counts and
  `pushed_at` dates are as of 2026-08-18 and will drift.
- **Zero-shot image-anomaly line (§3.1):** venues verified from each paper's arXiv comment field;
  supervision and output granularity read from abstracts. Their GitHub repos were **not** checked —
  the API rate-limited — but the line is excluded on granularity and domain regardless.
- **Sources used:** arXiv API (metadata + `comment` venue field), arXiv/CVF PDFs read locally with
  `pdftotext` for all result tables, HuggingFace papers search API for discovery, GitHub REST API for
  repo status, DBLP where reachable, OpenReview for the URF-HVAA citation. Semantic Scholar and
  OpenAlex were avoided as rate-limit-prone. Web search quota was exhausted partway through, so the
  second half of the sweep ran on arXiv + GitHub + direct page fetches only.
