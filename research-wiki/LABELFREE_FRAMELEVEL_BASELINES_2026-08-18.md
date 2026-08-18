# Label-free frame-level baselines — reproducibility shortlist (2026-08-18)

> **Revision, same day, second sweep.** The first pass built its spine from LAVAD (CVPR 2024) and
> UniVTG (ICCV 2023) and noted that five of the six §1A entries share one skeleton
> (caption windows → LLM scores each window → smoothing). A follow-up sweep restricted to
> **2025-01 … 2026-08** was run to find mechanisms that are actually new: agentic/tool-using
> detectors, reasoning-LLM scorers, test-time scaling, next-generation training-free grounding,
> streaming, and audio-visual. Results are in the new **§1C**, and they change §6's recommended
> order. Two other things changed: **§3.2's venue-integrity problem is now solved** (there is a
> definitive, machine-readable way to separate CVPR 2026 main from Findings — see §3.2), and the
> **audio channel is no longer empty** (§1C.5).

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
- **+13 entries from the 2025–2026 second sweep** (§1C), all passing all three filters: 8 visual
  (rows 26–33, §1C.1) and **5 audio / audio-visual** (rows 34–38, §1C.5), which are the first
  entries in this document that read the audio channel at all. Separately tabulated: 9 A-venue
  label-free frame-level entries that **fail only on code** (§1C.2), and the 2026 arXiv crop that
  currently holds the best numbers in the field (§1C.3).

So 25 rows total, of which 11 are unambiguous. Two structural observations worth carrying — **both
were partly overturned by the second sweep, see §1C**:

1. **The axis is young and homogeneous.** All 11 unambiguous entries are frozen-model inference
   pipelines (the sole exception being T3AL, which does discardable test-time gradient adaptation on
   the unlabeled test video), and 10 of the 11 were published in the last 36 months. Worse, five of
   the six §1A entries — LAVAD, URF-HVAA, VADTree, EventVAD, MoniTor — share one skeleton: caption
   windows → LLM scores each window → temporal smoothing. Reproducing three of those is closer to
   reproducing one pipeline three ways than to covering three independent hypotheses. That is the
   main reason the §1B query-conditioned line earns a slot in the recommended order below.
   *Update from §1C: the homogeneity is real but the sweep found four genuinely different A-venue
   mechanisms — agentic tool-use with memory (PANDA), synthetic-anomaly end-to-end training
   (LAVIDA), cross-video test-time memory (Memory Matters), and free-text-redefinable detection
   trained on a disjoint auxiliary corpus (LaGoVAD). Only LaGoVAD ships runnable code.*
2. **Every entry scores visual frames only.** None of the 25 reads speech or on-screen text. Against
   this project's own Gate-C finding — 30.1% of misses are "on-screen text has the evidence, speech
   does not", `on_screen_text` OR 2.29 — a pure CLIP/BLIP frame-similarity curve is measuring the
   wrong channel. Whichever entry gets ported, the ASR and OCR caches on disk have to be bolted into
   the query or caption stream, or the number will be floor-level for a reason that has nothing to
   do with the method.
   *Update from §1C.5: the second sweep found the audio half of this gap does have A-venue,
   training-free, frame-level, open-vocabulary methods with code (AV²A and OV-AVEL, both CVPR 2025).
   They score **acoustic events**, not linguistic content, so they are a cheap audio floor and
   citable prior art, not a hate localiser. The visual-only claim about the original 25 stands.*

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
| 5 | **MoniTor** | **NeurIPS 2025** | VAD, **online/streaming** | fully zero-label, training-free | streaming per-frame score from a VLM with an LSTM-style predictor and a scoring queue | Online setting: UCF-Crime **82.57** AUC; XD-Violence **55.01** AP / **79.11** AUC (vs online-LAVAD 76.06 / 52.63 / 76.01) | [YsTvT/MoniTor](https://github.com/YsTvT/MoniTor) — ~30★, last push 2025-10-01. **CORRECTION (second sweep): this repo is a documentation stub** — README says "Code will be available" and it has not been touched since before the arXiv posting. Filter 3 effectively fails | **MED → BLOCKED.** Only entry designed for causal/streaming scoring, but not currently reproducible. See §1C.3 for ESOM, the arXiv-only successor that beats it by +3.6 AUC / +16.7 AP causally |
| 6 | **O-VAD** | **ECCV 2026 — CONFIRMED main conference** (ECCV has no Findings track, see §3.2) | **industrial** VAD | fully zero-label, training-free, agentic | tracks per-object state trajectories, flags abnormal objects in grounded frames + emits a report | Phys-AD 0.584, LiquidAD 0.692, IPAD 0.565 video AUROC | **CORRECTION (second sweep): the repo linked here is the wrong one.** `yuanapril/OVAD-ECCV26` (0★) holds only README + index.html. The real one is [o-vad/O-VAD](https://github.com/o-vad/O-VAD) — 2★, last push 2026-07-26, **complete implementation**. See §1C.1 row 33 | **HIGH / poor fit.** The mechanism is object state evolution under physics and procedural constraints. Hateful video has no such object dynamics — the signal is speech, on-screen text and symbols. Now passes all three filters; still listed for completeness only |
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
| 12 | **UniVTG** (zero-shot inference) — **SUPERSEDED by UniTime (NeurIPS 2025), §1C.1 row 27; removed from the recommended order, see §6.0** | **ICCV 2023** | **boundary** — zero target labels, but the checkpoint is pretrained on 4.2 M temporal annotations (Ego4D narrations, VideoCC, 1.5 M CLIP-teacher pseudo-curves) | **best output form in the document**: three heads per 2 s clip — foreground probability, boundary offsets, and an explicit **saliency score** curve | ZS QVHighlights R1@0.5 25.16, mAP-avg 10.87, HD mAP 35.96; ZS Charades-STA R@0.5 **25.22**, mIoU 27.12; TACoS collapses (R@0.5 1.27); ZS highlight YouTube-HL mAP 53.9, TVSum top-5 mAP 67.2 | [showlab/UniVTG](https://github.com/showlab/UniVTG) — ~379★ (most-starred here), last push 2024-05-08, official, checkpoint released | **LOWEST of section 1B.** Inputs are CLIP ViT-B/32 + SlowFast R-50 features — no LLM, no API. Free-form query via the CLIP text encoder (77-token limit). The saliency head is natively the curve you want to score against `gold_segments.json` |
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
  (LELA, MARS). *Re-confirmed by the 2025–2026 sweep: the reasoning-LLM papers that look like this
  (Vad-R1, VAU-R1) all train on target-derived data, and the naive version has a measured recall
  collapse — see §1C.4.*
- Only 3 of the 7 entries in §1B (TFVTG, T3AL, UniVTG) report IoU-based localisation numbers at
  all. AKS, BOLT, T\* and SeViLA are frame-selection-for-QA papers — their per-frame curves are
  validated as "helps downstream QA accuracy", never as temporal predictions. Treat their curves as
  plausible but unproven.
- §1B methods need a **query**, and hate is not one concept. HateClipSeg's `gold_segments.json`
  carries a 6-way label vector per segment, so the natural query set is one prompt per category
  rather than a single "is this hateful" string. That also fixes T3AL's closed-class-list
  requirement for free.
- **Every §1B method assumes one target moment per query, and that assumption is now measured.**
  *Towards One-to-Many Temporal Grounding* (arXiv 2606.06294) reports that MLLMs optimised for
  one-to-one grounding "yield near-zero scores" when a query maps to multiple disjoint segments, for
  lack of event-cardinality perception. CoMET-Bench (arXiv 2606.15320) puts numbers on it: on
  conditional multi-event grounding the **grounding-specialised 7B models collapse** — TRACE
  F1@0.5 = 3.8, DisTime 0.5, TimeLens 5.4, LITA 0.4, all with a false-positive rate of 93–100%
  (they always predict something) — while frontier general models reach only 10.1 (GPT-5), 10.1
  (Gemini 2.5 Pro), 14.6 (Gemini 3 Flash), and the best training-free agent reaches 19.0.
  HateClipSeg's median video has **3 disjoint toxic blocks**. Both papers are arXiv-only so neither
  is a citable baseline, but the measurement is the most important caveat in this document for the
  §1B line.

---

## 1C. The 2025–2026 mechanism refresh (second sweep, same day)

Window: **2025-01-01 to 2026-08-18**. Same three hard filters. Same output requirement (per-frame or
per-timestamp score). Directions probed: agentic/multi-step VAD, reasoning-LLM scoring, test-time
scaling / test-time training, next-generation training-free grounding, streaming/online, and
audio-visual.

**Headline: the mechanisms did move, but the three filters did not move with them.** The recurring
failure is not novelty and it is not accuracy — it is **filter 3**. At A-venues the 2025–2026 crop
repeatedly ships a paper and an empty repository, while the entries with the strongest numbers are
2026 arXiv preprints with no venue yet. Four of the six best new mechanisms below have no runnable
code today.

### 1C.1 New entries that pass all three filters

| # | Method | Venue + year (verified how) | Direction | Mechanism, one sentence | Label-free class | Per-frame output | Datasets & headline numbers | Code + status | Port cost |
|---|---|---|---|---|---|---|---|---|---|
| 26 | **LaGoVAD** — *Language-guided Open-world VAD* | **ICLR 2026** (PDF header "Published as a conference paper at ICLR 2026"; arXiv 2503.13160 `journal_ref`) | VAD, free-text-redefinable | trains a CLIP detector **only on the new PreVAD corpus** (35,279 web/stream videos) with dynamic video synthesis + negative-mining contrastive regularisation, so the anomaly definition is re-specified in free text at inference | **(c) auxiliary-pretrained, zero target labels** — PreVAD sources verified to contain **no UCF-Crime and no XD-Violence** | yes | UCF-Crime **81.12** AUC; **XD-Violence 74.25 AP**; MSAD 90.41; UBnormal 58.07; DoTA 62.60; TAD 89.56; LAD 78.91 | [Kamino666/LaGoVAD-PreVAD](https://github.com/Kamino666/LaGoVAD-PreVAD) — ~31★, code Feb-2026, **weights May-2026, dataset released** | **LOW–MED.** The only new entry whose interface is literally "write down what counts as anomalous, in text" — the closest mechanical match in the whole document to "define hate in text". **XD-Violence AP 74.25 is +6.2 over URF-HVAA**, the best label-free XD number with released code |
| 27 | **UniTime** — *Universal VTG with generative MLLMs* | **NeurIPS 2025** (papers.nips.cc main-conference listing) | grounding | MLLM that **interleaves timestamp tokens with video tokens** and does adaptive frame scaling; the `-Zero` variant is pretrained with every in-domain corpus removed | **(b) UniVTG class** — zero target labels | intervals (timestamp tokens) | ZS Charades-STA R@0.5 **59.09**, R@0.7 31.88, **mIoU 52.19**; ANet 14.14 / 27.31 mIoU; QVH 31.48 / 43.71 | [lzq5/UniTime](https://github.com/lzq5/UniTime) — ~55★, last push 2026-05-20 | **MED. This is UniVTG's actual successor** and it more than doubles UniVTG's zero-shot Charades R@0.5 (59.09 vs 25.22). Cost is an MLLM forward per video instead of CLIP+SlowFast features |
| 28 | **OmniVTG** | **CVPR 2026 MAIN** (CVF open-access `/content/CVPR2026/`, **not** `/content/CVPR2026F/` — see §3.2) | grounding | builds a 2,124-hour auto-annotated open-world YouTube/Bilibili grounding corpus, then SFT + CoT + GRPO on it; Charades/ANet/QVH/TVGBench all held out | **(b)** zero target labels | intervals | ZS Charades R@0.3/0.5/0.7 **78.3 / 63.2 / 37.0**; ANet 60.3 / 39.8 / 21.4; QVH 82.8 / 67.0 / 47.3 | [oceanflowlab/OmniVTG](https://github.com/oceanflowlab/OmniVTG) — ~64★, last push 2026-05-28, full code + verl | **MED.** Best zero-shot grounding numbers with code in the document. Caveat: the zero-shot claim rests on its corpus not overlapping Charades/ANet, and the paper publishes **no overlap audit** |
| 29 | **NumPro** | **CVPR 2025** (arXiv 2411.10332 comment; DBLP `CVPR`) | grounding | **burns the frame index onto each frame** so a frozen video-LLM can read the timestamp off the image, like page numbers | **(a) fully training-free** (the NumPro variant; NumPro-FT is trained) | intervals | ZS Charades: Qwen2-VL-7B 5.4 → **36.8** R@0.5 with NumPro; GPT-4o 32.0 → 35.5; ANet Qwen2-VL 9.4 → 26.4 | [yongliang-wu/NumPro](https://github.com/yongliang-wu/NumPro) — ~149★, last push 2026-08-01 | **LOWEST of §1C.1.** A rendering trick plus one MLLM call. Also the single best source of the "general MLLM emits timestamps zero-shot" baseline table (§1C.6) |
| 30 | **BAGLM** — *Training-free Online Video Step Grounding* | **NeurIPS 2025** (DBLP `conf/nips/ZanellaMWTR25`) | grounding, **online** | LAVAD's own first author; zero-shot LMM scoring of instructional steps in a **causal/online** pass | **(a) fully training-free** | yes | instructional step grounding (not VAD/VTG benchmarks) | [lucazanella/baglm](https://github.com/lucazanella/baglm) — ~12★, last push 2025-10-26 | **MED / indirect.** The only A-venue, label-free, open-source *temporal* method the LAVAD lineage produced after LAVAD. Task is step grounding, so numbers do not transfer; the online formulation does |
| 31 | **ZS-STVG** (LLaVA-Next-STVG) | **NeurIPS 2025** (papers.nips.cc main listing) | spatio-temporal grounding | decomposes the query into attribute and action sub-queries, then optimises **latent spatial/temporal prompts at inference** via logit-guided re-attention | **(a) fully training-free** | per-frame temporal highlighting | HCSTVG-v1 m_vIoU 24.8, v2 27.7, VidSTG 18.0. **No Charades-STA number** — different task | [zaiquanyang/LLaVA_Next_STVG](https://github.com/zaiquanyang/LLaVA_Next_STVG) — ~21★, last push 2025-12-05 | **MED.** Emits tubes, not curves. Listed because the "optimise latent prompts at inference, no gradients on weights" mechanism is a clean test-time family member with code |
| 32 | **LAVIDA** — *No Need For Real Anomaly* | **CVPR 2026 MAIN — now CONFIRMED** (present in `openaccess.thecvf.com/CVPR2026?day=all`, paper id suffix `_CVPR_2026_paper`; absent from the `CVPR2026_findings` listing) | VAD | end-to-end MLLM + SAM2 trained **solely on pseudo-anomalies** made by pasting segmented objects; reverse-attention token compression | **zero target labels** (a training run exists, consumes no VAD data) | **frame *and* pixel** | UBnormal **76.45** AUC; ShanghaiTech **85.28**; UCF-Crime **82.18**; **XD-Violence AP 90.62** — above weakly-supervised Holmes-VAU (88.96) and 28 points above LAVAD | [VitaminCreed/LAVIDA](https://github.com/VitaminCreed/LAVIDA) — ~27★, last push 2026-02-25. Real code (`train_acc.py`, `model/`, `sam2/`, `clip/`, `loss/`) but README says data-prep and usage instructions are "waiting for further updates" and **no checkpoint is released** | **HIGH.** Filter 3 passes on the letter and fails on the spirit: nothing is runnable without writing the data pipeline yourself and re-training. The pseudo-anomaly generator ("paste an out-of-context object") is a visual-oddity prior with no obvious hate cue, so expect the port to underperform its XD number badly |
| 33 | **O-VAD** — *corrected entry, supersedes §1A row 6* | **ECCV 2026 CONFIRMED** (arXiv 2607.18142 comment "Accepted to ECCV 2026"; **ECCV 2026 has no Findings track** — verified from its CfP, which lists workshops/tutorials/demos/doctoral-consortium as separate submission categories, not alternative acceptance tracks) | industrial VAD, agentic | ground → track → reason: VLM object inventory + SAM3 masks, CropFormer/SAM2/FC-CLIP tubelet tracking, then six-step CoT over the state graph | **(a) fully training-free, agentic** | video, frame *and* object level report | three IVAD datasets: Phys-AD 0.584, LiquidAD 0.692, IPAD 0.565 video AUROC | **The link in §1A row 6 is wrong.** `yuanapril/OVAD-ECCV26` (0★) holds only `README.md` + `index.html`. The real repo is **[o-vad/O-VAD](https://github.com/o-vad/O-VAD)** — 2★, last push 2026-07-26, complete (`pipeline.py`, `quick_run.py`, `configs/`, `dataset/`, `eval/`, `TubeletGraph/`, `thirdparty/`) | **HIGH / poor fit, unchanged.** Object state evolution under physics and procedural constraints; hateful video has no such object dynamics. Now passes all three filters, still recommended against |

### 1C.2 A-venue, label-free, frame-level — and **no runnable code today**

This is the largest and most frustrating group in the sweep. Every row here would be a candidate if
the code existed. Worth a re-check in a few months.

| Method | Venue + year (verified) | Mechanism | Numbers | Code status |
|---|---|---|---|---|
| **PANDA** — *Generalist VAD via Agentic AI Engineer* (arXiv 2509.26386) | **NeurIPS 2025** (OpenReview `venue = NeurIPS 2025 poster`; README "[NeurIPS 2025] Accepted Paper") | The answer to "agentic VAD, who else besides O-VAD": scene-aware **RAG strategy planning** → goal-driven heuristic reasoning → **tool-augmented self-reflection** (super-resolution, object detection, retrieval, web search) → **self-improving chain-of-memory**. Training-free *and* "manual-free". 1 fps score curve, mean filter w=10 | offline UCF **84.89** AUC / XD **70.16** AP / UBnormal **75.78** / CSAD 73.12; online 82.57 / 63.57 / 72.41 / 71.25. Beats LAVAD, EventVAD, URF-HVAA and VADTree on UCF and XD AP; its online mode beats MoniTor's XD AP by +8.6 | **[showlab/PANDA](https://github.com/showlab/PANDA) — 33★, last push 2025-10-02, contains `README.md` + `LICENSE` + `assets/` and nothing else.** The paper text says "Code is released at …"; verified via the GitHub contents API that it is not |
| **Memory Matters (LLT)** — *Boosting Training-Free Zero-Shot TAL with a Learnable Lookup Table* | **CVPR 2026 MAIN** (CVF `/content/CVPR2026/`) | **The A-venue T3AL successor.** Recasts training-free ZS-TAL as memory-augmented retrieval: a lookup table accumulates high-confidence action-positive candidates **across historical test videos**, a learnable residual adapts the retrieved item to the current video, refined **frame activation scores** select frames and re-tune the text prototypes | **Beats T3AL on both benchmarks.** THUMOS'14 50/50 avg mAP **12.6** vs T3AL 10.4; 75/25 **12.8** vs 9.2. ActivityNet 50/50 **15.7** vs 14.3; 75/25 **17.1** vs 15.4. Even its no-optimisation variant (LLT_{T=0}, 11.8 / 12.1) beats full T3AL | **No repository found.** GitHub search returns nothing; the paper contains no code statement. It is built directly on T3AL's protocol (same threshold rule, same CoCa-style VLM), so a reimplementation on top of T3AL's repo is the realistic path |
| **DART** — *Difficulty-Adaptive Routing for Zero-Shot VTG* (arXiv 2607.00672) | **ECCV 2026** (arXiv comment; no ECCV Findings track) | Query-conditioned **Determinantal Point Process** kernel does double duty — selects diverse query-relevant keyframes, and its **spectral entropy is a difficulty score**. Simple queries go to a Fast path; complex ones to a Slow path with Temporal Markup Prompting (global event analysis → per-frame temporal role annotation → boundary extraction) | **Best zero-shot grounding numbers in the document.** Charades R@0.3/0.5/0.7/mIoU **70.98 / 52.04 / 29.45 / 48.93**; ANet **54.90 / 32.14 / 18.11 / 39.89**. Beats TFVTG by +2.1 R@0.5 and +4.4 mIoU on Charades, +5.8 mIoU on ANet. Ablation: w/o DPP 44.70, Fast-only 44.34, Slow-only 47.53 | **No repository.** GitHub search returns 0; [dart-vtg.github.io](https://dart-vtg.github.io/) has no code link. Painful, because the recipe is cheap and open: **LLaVA-1.6-7B, 3 FPS, one A100, no paid API, 12 frames and 3.9 s per query** — against TFVTG's 86 frames, 4.7 s **and a GPT-4-Turbo call**. Caveat if reimplemented: its DPP bandwidths and routing threshold were **grid-searched on the Charades-STA validation split** |
| **MoniTor** — *already §1A row 5, status corrected* | NeurIPS 2025 | streaming per-frame score, LSTM-style predictor + scoring queue | online UCF 82.57 / XD 55.01 AP | **[YsTvT/MoniTor](https://github.com/YsTvT/MoniTor) is a documentation stub** — README says "Code will be available", untouched since 2025-10-01, i.e. before the arXiv posting. The §1A row's "official" label overstates it |
| **GranAlign** (arXiv 2601.00584) | **AAAI 2026** (arXiv comment) | training-free zero-shot moment retrieval: granularity-based query rewriting + query-aware caption generation, pairing multi-level queries with query-agnostic and query-aware captions | Charades **59.1 / 39.6 / 22.7 / 38.0**; ANet 50.3 / 34.0 / 16.5 / 33.1 — **below TFVTG's 2024 numbers on Charades by 6.5 mIoU** | No repository found |
| **VTimeCoT** (arXiv 2510.14672) | **ICCV 2025** (CVF proceedings listing) | training-free "thinking by drawing": a plug-in **progress-bar overlay tool** plus a highlighting tool let GPT-4o/Qwen2-VL annotate time progression visually | Charades **74.06 / 51.02 / 22.45 / 46.78** with GPT-4o — a real but marginal +1.05 R@0.5 / +2.27 mIoU over TFVTG | Project page only, "Code (Coming soon)", no repo under the `vtimecot` org besides the site |
| **Self-SiMS** (arXiv 2607.19027) | **ECCV 2026** (arXiv comment) | training-free span-proposal scoring | Charades 62.7 / 39.7 / 21.0 / **41.9** — below TFVTG | No repository found |
| **Moment-GPT** (arXiv 2501.07972) | **AAAI 2025** (arXiv comment) | off-the-shelf MLLMs, explicit frame scorer + span scorer | Charades 58.2 / 38.4 / 21.6 / 36.5 — below TFVTG | No repository found |
| **GtS / VAGU** — *already §1A row 7* | AAAI 2026 | glance-then-scrutinise, prompt-guided | journal extension arXiv 2608.11260 | Still no repo. Unchanged |

### 1C.3 Better numbers, wrong venue — the 2026 arXiv crop

These are the strongest label-free frame-level results that exist anywhere right now. **None is
citable as a baseline**, and several are visibly under review. Recorded so the field position is
honest and so the re-check list is concrete.

| Method | Status | Mechanism | Numbers |
|---|---|---|---|
| **CEAVAD** (2608.09908) | arXiv, 2026-08-10; [lessiYin/CEAVAD](https://github.com/lessiYin/CEAVAD) 0★ | **Contrastive event adjudication**: instead of scoring an anomaly concept, it pairs each hazard mechanism with a generic normal account *and* a mechanism-specific benign counterpart, then adjudicates which explanation the video evidence supports | **The best training-free VAD numbers found**: UCF-Crime **86.15** ROC-AUC, UBnormal **77.55**, XD-Violence **93.40** AUC / **79.52 AP**. Its own "Direct Qwen" floor: UCF 80.60 / UBnormal 70.41 |
| **ESOM** (2604.07772) | arXiv, no venue; [Kamino666/ESOM_OpenDef-Bench](https://github.com/Kamino666/ESOM_OpenDef-Bench) 3★ | streaming open-world VAD with an explicit frame-level score module, causal | **Online UCF 86.18 AUC / XD 71.68 AP at RTF 0.52** — +3.6 AUC and +16.7 AP over MoniTor. The cleanest MoniTor successor that exists, and unrefereed |
| **Flashback** (2505.15205) | **ICLR 2026 REJECTED** (OpenReview `venueid = ICLR.cc/2026/Conference/Rejected_Submission`); no official repo | memory-driven, frozen LLM offline + frozen ImageBind/PerceptionEncoder online | UCF **87.29** / XD **75.13 AP** — the highest headline in the space. Two disqualifiers beyond the venue: no code, and despite the "real-time" branding it **averages segment scores over the whole video with Gaussian smoothing**, so the number is not causal and not comparable to MoniTor/ESOM |
| **REZE** (2608.04480) | arXiv 2026-08, no code | training-free per-clip score sequence with a task-swappable aggregation rule | Charades **80.59 / 64.92 / 37.80, mIoU 55.66** (Qwen3-VL-8B). **But its own ablation shows frozen Qwen3-VL-8B emitting timestamps directly already scores 62.88 R@0.5 / 53.30 mIoU** — the mechanism adds ~+2.0 |
| **MTLA** *Propose and Attend* (2607.05978) | arXiv, [TalRemez/MTLA](https://github.com/TalRemez/MTLA) 4★ | training-free post-hoc grounding confidence from **multi-token localised attention** over a frozen Qwen3-VL-8B | Charades 76.3 / 55.4 / 29.4 at N=16 rollouts; +11.4 over its own single-rollout backbone, i.e. bought with 16× test-time compute |
| **FV-Action** *Your VLM Already Knows When* (2608.08315) | arXiv, no code | training-free coarse-to-fine scan of the per-clip "Yes"-probability | Charades 70.1 / **56.8** / 29.7 (Qwen2.5-VL-7B); QVH mIoU 45.2 |
| **MLLMs Know When Before Speaking** (2605.21954) | arXiv, project page only | finds sparse **Temporal Grounding Heads** whose prefill cross-attention already concentrates on the ground-truth interval, then re-invokes the model with context cropped to it — the "perception–generation gap" | gives a **debiased frame-level relevance signal**, which is exactly the output form this document wants |
| **TAG** (2508.07925) | **BMVC 2025** — fails venue only; [Nuetee/TAG](https://github.com/Nuetee/TAG) 24★, last push 2025-11-18 | BLIP-2 temporal pooling + **temporal coherence clustering** + similarity adjustment. **No LLM anywhere, no paid API.** Explicitly targets *segment fragmentation* | Charades 67.82 / 48.58 / 26.67 / **45.69** — beats TFVTG's mIoU while deleting its GPT-4-Turbo dependency. The cheapest code-available thing that clears the 2024 bar |
| **FreeZAD / AdaZAD** (2501.13795) | **IEEE TMM** (journal) — fails venue only; [Chaolei98/FreeZAD](https://github.com/Chaolei98/FreeZAD) 18★, complete | FreeZAD = snippet actionness + calibration, 0 trained params, 128.9 FPS; AdaZAD = test-time adaptation on the unlabelled test video | **AdaZAD is the only method that cleanly beats T3AL on both**: THUMOS 14.1 / 14.0 vs 10.4 / 9.2; ANet 20.0 / 19.3 vs 14.3 |
| **OZ-TAL / VFEAL** (2605.09976) | arXiv (DBLP CoRR); [Chaolei98/OZ-TAL](https://github.com/Chaolei98/OZ-TAL) 0★, **baseline only — VFEAL not released** | fully training-free, gradient-free, causal state machine over frame-wise K+1 classification | THUMOS 50/50 **9.24** vs T3AL 10.4 — its own text concedes "comparable". ActivityNet collapses to **4.99** vs 14.3. The contribution is the causal constraint, not the score |
| **QVAD** (2604.03040) · **AgenticVAU** (2608.03779) · **SphereVAD** (2605.08003) · **Cerberus** (2510.16290) · **TRACES** (2511.00580) | all arXiv-only | question-centric agentic prompt-updating dialogue; four-agent explore/verify with an anchor-registry evidence memory; geodesic inference on the unit sphere; cascaded VLMs; temporal recall | the 2026 agentic VAD wave. Prolific, unrefereed |
| **CoMET-Agent** (2606.15320) | arXiv-only | training-free agentic **multi-event** search-and-aggregate | +6.1 F1@0.5 over GPT-5 on conditional multi-event grounding. The only method in the sweep that targets the multi-disjoint-segment regime HateClipSeg actually has |

### 1C.4 Honest scoreboard: is "new mechanism" the same as "better number"?

The brief asked for this explicitly. Split by direction:

- **Agentic VAD — new mechanism, better number, no code.** PANDA (NeurIPS 2025) beats the entire
  §1A caption-then-score line on UCF-Crime and XD-Violence AP, and it does it with a genuinely
  different architecture (RAG planning, tool use, cross-video memory). Its repo is empty. **The
  answer to "who else besides O-VAD" is: PANDA, and nobody else at an A-venue** — QVAD, AgenticVAU,
  ESOM and CEAVAD are all 2026 arXiv. Worth knowing about PANDA's own ablation: swapping the
  reasoning LLM across GPT-4o (84.97), Gemini 2 Flash, DeepSeek-V3 and Qwen2.5-72B (84.03) moves
  UCF-Crime AUC by **about one point**. The scaffolding carries the result, not the frontier model.
- **Reasoning-LLM scoring — the honest answer is "no A-venue method, and the zero-shot numbers are
  weak."** There is still no CCF-A/A\* paper whose contribution is "reasoning MLLM scores each
  segment". The 2025 "R1"-style video-anomaly-reasoning papers (Vad-R1 at NeurIPS 2025, VAU-R1)
  **fail filter 1** — they SFT and RL-tune on data built from UCF-Crime and XD-Violence, and Vad-R1
  reports no frame-level AUC at all. Meanwhile *Are Multimodal LLMs Ready for Surveillance?*
  (arXiv 2603.04727) measures the naive version and finds a **recall collapse**: zero-shot MLLMs are
  confidently biased toward "normal", and peak F1 on ShanghaiTech only moves from **0.09 to 0.64**
  once class-specific instructions are supplied. The GLM-4.1V-9B-Thinking ZS-CoT row already in this
  document (UCF 61.80 / XD 72.73 AUC, from URF-HVAA's Table 2) is consistent with that.
- **Test-time scaling — the T3AL line finally has an A-venue successor, and the fashionable
  "test-time scaling" papers are not it.** Memory Matters (CVPR 2026 main) beats T3AL on both
  benchmarks and is training-free; it has no code. Everything marketed as test-time scaling for
  video — CyberV, TimeSearch-R, Zoom-Zero, TimeScope, ZoomV — reports **zero** THUMOS/ActivityNet
  zero-shot TAL mAP; they optimise long-video QA accuracy. Treating them as T3AL successors is a
  category error. Reinforcing this, *The Illusion of Progress?* (**NeurIPS 2025 Datasets &
  Benchmarks**) benchmarks 8 episodic and 7 online TTA methods over 15 datasets and finds most
  **fail to surpass the plain zero-shot baseline** on fine-grained tasks — a refereed warning
  against expecting test-time adaptation machinery to pay off on a target like hateful video.
- **Training-free grounding — fancier, and mostly worse.** Of the A-venue fully-training-free
  entries, **not one beats TFVTG (ECCV 2024)** except VTimeCoT by a marginal +1.05 R@0.5:
  Moment-GPT (AAAI 2025) 38.4, NumPro (CVPR 2025) 36.8, GranAlign (AAAI 2026) 39.6, Self-SiMS
  (ECCV 2026) 39.7 — all roughly **10 points below** TFVTG's 49.97. DART (ECCV 2026) does beat it
  properly, at 52.04, and has no code. The arXiv crop (REZE 64.92, FV-Action 56.8, MTLA 55.4) beats
  it clearly, but **the gain is backbone drift, not mechanism**: REZE's own ablation puts frozen
  Qwen3-VL-8B direct timestamp emission at 62.88 R@0.5 already, so its machinery adds ~+2. On a
  fixed backbone the 2025–2026 mechanisms are worth roughly **+2 to +5 points, not +15**.
  One correction to carry: **the 49.97 bar is soft** — Self-SiMS reproduced TFVTG at a 1-FPS budget
  and got 29.8 R@0.5 / 38.2 mIoU. TFVTG's own number needs 3 FPS *and* GPT-4-Turbo query rephrasing.
  Quote the bar with its FPS and its LLM or expect pushback.
- **Where the real numbers are: label-free-but-trained (class b/c).** UniTime-Zero 59.09,
  OmniVTG 63.2, LaGoVAD XD AP 74.25. If the binding constraint is "no *target* labels" rather than
  "no training", this class is 10–15 points ahead of everything training-free, and three of its
  members are at A-venues **with working code**. That is the single biggest strategic finding of
  this sweep.
- **Streaming — led by an unrefereed preprint.** ESOM (86.18 / 71.68, causal) beats MoniTor
  (82.57 / 55.01) decisively but is arXiv-only; MoniTor's own repo is a stub; PANDA's online mode is
  A-venue but has no code. The online + training-free niche is three papers deep and none of them is
  currently reproducible.

### 1C.5 Audio-visual — the channel this project actually needs

The first pass recorded "none of the 25 reads speech or on-screen text" as a structural gap. The
sweep found the gap is **half-filled**: A-venue, training-free, frame-level, open-vocabulary
audio-visual localisers with code do exist.

| # | Method | Venue + year (verified) | Mechanism | Label-free class | Per-frame output | Free-text query? | Numbers | Code |
|---|---|---|---|---|---|---|---|---|
| 34 | **AV²A** — *Adapting to the Unknown* | **CVPR 2025 MAIN** (DBLP `conf/cvpr/ShaarSCW25`, pp. 3142–3151, DOI 10.1109/CVPR52734.2025.00299; CVF `/content/CVPR2025/`) | freezes CLIP **and** CLAP, fuses their text-similarity scores at score level, then walks a per-category decision threshold across the video using a running confusion matrix over prior segments | **(a) fully training-free**, zero gradient steps | **yes, per 1 s — and separately for audio-only, visual-only and audio-visual** | **yes**, label set defined per video at inference, separate audio-side and visual-side prompts | LLP seg-Type@AV 22.0 → **52.4** (LanguageBind); audio seg-F1 20.3 → **40.9**; AVE acc 32.9 → **72.8** | [eitan159/AV2A](https://github.com/eitan159/AV2A) — ~6★, last commit 2025-10-21 |
| 35 | **OV-AVEL** | **CVPR 2025 MAIN** (DBLP `conf/cvpr/ZhouGGMHZCW25`; arXiv comment) | embeds audio, video and free-text category prompts into **one ImageBind joint space** and takes segment-level tri-modal similarity | **(a) fully training-free** (the training-free baseline) | yes, 10 × 1 s segments | yes | OV-AVEBench training-free 59.2 acc / 46.7 seg-F1 / 34.0 event-F1; unseen split 59.8 / 47.3 / 34.0 | [jasongief/OV-AVEL](https://github.com/jasongief/OV-AVEL) — ~46★, last commit 2025-03-07 |
| 36 | **DASM** — *Detect Any Sound* | **ACM MM 2025** (arXiv comment + DOI 10.1145/3746027.3755574 + DBLP) | frame-level **retrieval** against a query vector from frozen MGA-CLAP; dual-stream decoder splits clip-level recognition from temporal localisation | **(b)** trained on AudioSet-Strong base classes, zero target labels | **yes, 50 fps (20 ms)** | yes — **text *or* audio** prompt | novel-class PSDS_r **33.9** (+7.8 over CLAP methods); closed-set PSDS 50.9; DESED cross-dataset ZS PSDS1 42.2 | [cai525/Transformer4SED](https://github.com/cai525/Transformer4SED) — ~105★, last commit 2026-02-10, **weights on HF `CPF2/detect_any_sound`, MIT** |
| 37 | **FLAM** | **ICML 2025** (arXiv comment) | open-vocabulary contrastive audio-language model with a **calibrated frame-wise objective** plus logit adjustment, so a text query yields a per-frame presence curve | **(b)** aux-pretrained | yes, frame-wise | yes | frame-wise open-vocabulary sound-event localisation | [adobe-research/openflam](https://github.com/adobe-research/openflam) — ~111★ (weights non-commercial) |
| 38 | **FineLAP** | **ACL 2026 MAIN** (ACL Anthology `2026.acl-long.473`, pp. 10393–10408) | reconciles clip-level, frame-level and synthetic supervision into one language-audio pretraining recipe | **(b)** | yes, ~160 ms frames | yes, free-text phrase grounding | text-to-audio grounding ≈ 0.649 | [xiquan-li/FineLAP](https://github.com/xiquan-li/FineLAP) — ~36★ |

**The catch is semantic and it is serious.** Every one of these localises **acoustic events**, not
linguistic content. Their text towers are CLAP / ImageBind / MGA-CLAP, trained on AudioSet-style
sound-event captions. "Someone making a racist statement" is not an acoustic category — it is a
property of transcribed words. CLAP will score "shouting", "angry speech" or "male speech", and
nothing in these models separates a slur from a greeting at the same volume. The absolute ceiling
confirms it: genuinely zero-shot frame-level audio grounding sits at **PSDS1 10.1 on
AudioSet-Strong** (MGA-CLAP, ACM MM 2024 — passes every filter except the date window). So AV²A and
OV-AVEL are a **cheap audio floor and the right prior-art citation**, and they will not replace
ASR + a text model on the speech channel.

Near-misses: **FlexSED** (open-vocab SED, code, weights) fails only on venue (WASPAA 2025);
**TimeAudio** (AAAI 2026) and **SpotSound** emit timestamps as generated text tokens, so there is no
per-frame score to threshold; **PreFM** (NeurIPS 2025) is trained on UnAV-100/LLP; **FATE**
(2608.01310) is arXiv-only with no DBLP record; *What's Making That Sound Right Now?* (ICCV 2025)
localises **spatially**, not temporally. **AnyAnomaly** (free-text anomaly, per-frame) still fails on
WACV.

**One hard negative worth recording:** the intersection {zero-shot} × {uses audio} ×
{XD-Violence} is **empty** across 2025-01…2026-08. Every zero-shot XD-Violence result is
visual-only; every audio-using XD-Violence method is weakly supervised. Probed from six query
directions; this is a real gap, not a search artifact.

### 1C.6 General MLLMs emitting timestamps — the citable off-the-shelf baseline

The brief asked whether a general MLLM (Qwen/Gemini/GPT) directly emitting timestamps has been
systematically evaluated at an A-venue, so it can be quoted as "the strongest off-the-shelf zero-shot
baseline". **Yes — three A-venue sources, and they disagree with each other, so cite the source and
the harness, never the bare number.**

| Source (venue, verified) | Model | Charades-STA R@0.3 / 0.5 / 0.7 / mIoU | Note |
|---|---|---|---|
| **NumPro, CVPR 2025** (2411.10332) | GPT-4o | 55.0 / **32.0** / 11.5 / 35.4 | The only A-venue paper whose purpose is a systematic "general MLLMs emit timestamps" table |
| NumPro, CVPR 2025 | Qwen2-VL-7B | 8.7 / 5.4 / 2.4 / 7.9 | |
| NumPro, CVPR 2025 | Qwen2-VL-**72B** | 0.0 / 0.0 / 0.0 / 0.2 | Bigger is not better; output format collapses |
| NumPro, CVPR 2025 | LLaVA-OneVision-7B / LLaVA-Video-7B / LongVA-7B-DPO | 7.9 / 2.7 / 10.1 R@0.5 | |
| **TempSamp-R1, NeurIPS 2025** (2509.18056) | **Qwen2.5-VL-7B native, zero-shot** | 73.4 / **54.4** / 30.3 / **49.7** | lmms-eval harness. Charades/ANet rows are **reproduced from another paper** by the authors' own footnote; only QVHighlights is their implementation |
| **VideoChat-R1.5, NeurIPS 2025** (2509.21100) | Qwen2.5-VL-7B | — / **42.9** / 26.2 / — | **Same model, 11.5 points lower.** The spread is prompt/harness, nothing else |
| **TimeLens, CVPR 2026 MAIN** (2512.14698) | Qwen2.5-VL-7B | 59.4 / 38.2 / 18.1 / 43.6 | Explicitly framed as "does not introduce a novel method but establishes a baseline" |
| TimeLens, CVPR 2026 main | GPT-4o / GPT-5 / Gemini-2.0-Flash / Gemini-2.5-Pro | 27.9 / 18.3 / 29.0 / 25.5 R@0.5 | **Frontier proprietary models score *worse* than a 7B open model on original Charades-STA.** TimeLens's diagnosis: the original annotations are noisy — on their re-annotated Charades-TimeLens the ordering flips (Gemini-2.5-Pro 61.1, Qwen3-VL-8B 53.4, GPT-4o 44.5) |

Four consequences:

1. **Qwen2.5-VL-7B native grounding is the strongest reproducible off-the-shelf zero-shot baseline**,
   and it is an A-venue row. Cite **TempSamp-R1** for it, state the harness, and note the
   VideoChat-R1.5 disagreement in the same breath.
2. **GPT-4o/GPT-5/Gemini direct timestamp emission is not a strong baseline** — on original
   Charades-STA they sit below a 7B open model, and DART's Table 1 puts open 7B video-LLMs
   (VideoChat-7B 6.5 mIoU, VideoLLaMA-7B 7.1, VideoChatGPT-7B 13.7) far below purpose-built
   training-free pipelines at 44–49. "Just ask Gemini for the timestamps" is not a competitive
   baseline for this task.
3. **Qwen3-VL / Qwen3-Omni / GPT-5 temporal grounding has no A-venue evaluation at all.** The best
   number for Qwen3-VL-8B (79.87 / 62.88 / 32.15, mIoU 53.30) comes from REZE, which is arXiv-only.
4. **On the VAD side there is no A-venue GPT-4o or Gemini zero-shot row at all.** The open-weight
   zero-shot rows that do exist: LLaVA-1.5 72.84 AUC / 50.26 AP, Video-LLaMA2 74.42 / 53.57,
   LAVAD 80.28 / 62.01. Note LAVAD's UCF AUC is reported as 79.21 by ESOM and 78.33 by EventVAD's
   reproduction, and its XD AP as 60.02 in MoniTor's table — **cross-paper VAD numbers are not
   safely poolable; re-run rather than cite across papers.**

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
| **LAVIDA** — *No Need For Real Anomaly* | **CVPR 2026 MAIN CONFERENCE — RESOLVED, see §3.2 and §1C.1 row 32** | Trains end-to-end, but **only on synthetic pseudo-anomalies** built by pasting segmented objects. Zero real VAD data, zero target labels. Frame-level *and* pixel-level output. Code: [VitaminCreed/LAVIDA](https://github.com/VitaminCreed/LAVIDA) ~27★, last push 2026-02-25, official | **Counts as label-free, and the venue is now confirmed main-track.** Numbers extracted in §1C.1: XD-Violence AP **90.62**, the highest in this document. The pseudo-anomaly generator ("paste an out-of-context object") is a visual-oddity prior that does not obviously produce hate cues, so expect the port to underperform. Practical blocker is now filter 3 in spirit: the repo has code but no data-prep instructions and no released checkpoint |
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

Three further findings from this line that are worth keeping even though the line is excluded:

- **LogSAD** (*Towards Training-free Anomaly Detection with Vision and Language Foundation Models*,
  **CVPR 2025**, DOI 10.1109/CVPR52734.2025.01416, [zhang0jhon/LogSAD](https://github.com/zhang0jhon/LogSAD)
  ~101★) is the strongest **genuinely** training-free member — no gradient step anywhere, GPT-4V for
  offline proposal generation over frozen CLIP + DINOv2 + SAM. Its headline MVTec-AD/VisA numbers
  (97.0/97.6 and 93.0/98.1) are **4-shot** — they consume target normal reference images. Its true
  zero-shot number is MVTec-LOCO 90.2 image-AUROC. Still image-only, so still excluded, but it is
  the right citation if you ever need "training-free dense scoring is possible".
- **The auxiliary training in this literature is done on the benchmark's *test* split.** VCP-CLIP
  states it outright ("to evaluate ZSAS performance on other datasets, we employ weights trained on
  VisA's test sets"), KAnoCLIP likewise; the rest inherit the APRIL-GAN/AnomalyCLIP protocol
  silently. It is structurally unavoidable — the test split is the only one with anomalous images
  and pixel masks — but it means the "zero-shot" label in this line rests on cross-dataset transfer
  from *another benchmark's test set with full pixel supervision*. Worth knowing before adopting any
  of their framing.
- **AnomalyVFM** breaks the two-category framing: it is trained **solely on synthetic data**
  (10,000 FLUX-generated images, low-rank adapters on a frozen RADIOv2.5 backbone), touching no
  MVTec/VisA labels, and claims a 9-dataset average image-AUROC of 94.1.
  ([MaticFuc/AnomalyVFM](https://github.com/MaticFuc/AnomalyVFM) ~62★, real code; **venue
  unverified**.) That is the same move LAVIDA makes on the video side — synthetic anomalies instead
  of target labels — so it is now a two-domain pattern rather than a one-off.

### 3.2 Venue-integrity: the CVPR 2026 Findings problem, **now solved**

> **Update, second sweep.** The procedure below was written when the CVPR 2026 program API was the
> only known check and it returns 403. There is a better one, and it is definitive.
>
> **`openaccess.thecvf.com` publishes the two proceedings as separate listings:**
> - main conference — `https://openaccess.thecvf.com/CVPR2026?day=all`, **2,295 papers**, paper
>   paths `/content/CVPR2026/html/…_CVPR_2026_paper.html`
> - Findings — `https://openaccess.thecvf.com/CVPR2026_findings?day=all`, **942 papers**, paper
>   paths `/content/CVPR2026F/html/…_CVPRF_2026_paper.html`
> - workshops — `CVPR2026_workshops`, separate again
>
> So the check is: fetch the main listing and grep for the title. The `_CVPR_2026_paper` vs
> `_CVPRF_2026_paper` suffix is the discriminator. **Use this instead of DBLP, OpenReview or the
> arXiv comment field for any CVPR 2026 claim.**
>
> **Resolved this way:**
> - **LAVIDA** (*No Need For Real Anomaly*) — present in the **main** listing. The document's
>   longest-standing open item is closed: it is a genuine CVPR 2026 main-conference paper.
> - **OmniVTG** and **TimeLens** — both **main**. Safe to cite (§1C.1, §1C.6).
> - **Memory Matters** (learnable lookup table for training-free ZS-TAL) — **main** (§1C.2).
> - Calibration example of a paper that *is* Findings, i.e. workshop proceedings and a venue FAIL:
>   *ConfDiff: Confidence-Guided Representation Diffusion for Video Moment Retrieval*.
>
> **ECCV 2026 does not have this problem.** Verified from the ECCV 2026 Call for Papers: there is a
> single main technical program, and workshops / tutorials / demos / doctoral consortium are
> separate submission categories, not alternative acceptance tracks for regular papers. An arXiv
> comment reading "Accepted to ECCV 2026" is therefore sufficient. This clears **O-VAD** (§1C.1
> row 33), **DART** and **Self-SiMS** (§1C.2).
>
> **The trap is live and it caught things.** Two papers in this sweep advertise CVPR 2026 in their
> arXiv comment and are workshop papers: *Text-guided Fine-Grained Video Anomaly Detection*
> (arXiv 2511.00524) — comment reads verbatim "Accepted by CVPR 2026 **SVC Workshop**" — and
> *GRAZE* (2604.01383, CVPR 2026 CVSports Workshop). **GridVAD** (2603.25467) is an ECCV 2026 LVOS
> **Workshop** paper. A fourth trap class turned up outside CVPR: **SafeLens: Segment-Level Hate
> Speech Detection in Online Videos** is indexed by DBLP with the bare venue string "AAAI" but is a
> **3-page AAAI 2026 Demonstration Track** paper. Bare DBLP venue strings are not sufficient either.

### 3.2.1 Original note (superseded above, kept for the record)

**Venue-integrity warning: "CVPR 2026" in an arXiv comment is not proof of the main track**

CVPR 2026 introduced a **Findings Track** at decision time. From the official Author Guidelines:
*"a venue for technically sound papers with solid experimental validation, even if their novelty is
more incremental"* … *"**Findings papers will appear in the workshop proceedings.**"* Roughly 147
arXiv papers currently carry a Findings comment, and the phrasings are inconsistent — some write
only *"Accepted to CVPR 2026"* and mention Findings solely in the body.

Compounding this: **DBLP has no `conf/cvpr/cvpr2026` volume yet and OpenAlex has no CVPR 2026
proceedings**, so "absent from DBLP" currently carries *zero* negative signal for CVPR 2026.

The reliable check is the conference's own program data:
`https://cvpr.thecvf.com/static/virtual/data/cvpr-2026-orals-posters.json`. A main-conference entry
has `eventtype: Poster|Oral`, a real `session`, and `sourceurl` pointing at the
`thecvf.com/CVPR/2026/Conference` OpenReview group. Verified that way in this session:
**Alert-CLIP** — *Abnormality-aware Latent-Enhanced Representation Tuning of CLIP for Video Anomaly
Detection* (BUPT et al., event id 36334, Poster Session 5) is a genuine CVPR 2026 main-conference
poster. It is a **video** anomaly detection paper and therefore in scope for this document; its
supervision is unverified and the title ("representation tuning") implies training, so it most
likely belongs with the weakly-supervised rejects — flagging as the one open item.

**LAVIDA could not be checked** *(at the time — since resolved via the CVF listing, see §3.2)*: the
program JSON is paginated at 200 of 5163 entries, the paginating API returns 403 from this
environment (`{"detail":"Authentication credentials were not provided."}` even with a browser
user-agent), `/virtual/2026/search` is a client-side shell that renders nothing server-side, and
OpenReview's search API returns no CVPR 2026 note for it. All four of those dead ends are still
dead ends — the CVF open-access route in §3.2 is what works.

**Alert-CLIP — open item now closed.** Its CVPR 2026 main-conference status is confirmed (present in
the CVF main listing). Its supervision is now read from the abstract: the first of its three
alignment stages is **"video-label alignment, which reshapes the semantic space"**. It uses labels.
**Fails filter 1** and joins the weakly-supervised rejects in §4, exactly as the title implied.

Scope of the damage: **none of the four originally recommended reproductions is affected.** LAVAD
(CVPR 2024), URF-HVAA (NeurIPS 2025), UniVTG (ICCV 2023), ZS-CLIP (inside LAVAD) and the fallback
MULDE/CLAP (CVPR 2024) all have fully indexed pre-2026 venues.

---

## 4. Explicitly rejected, with reasons

**Rejected on label-free (uses video-level anomaly labels):** the entire MIL-ranking lineage from
Sultani et al. onward — MIST, RTFM, MGFN, UR-DMU, S3R, CLIP-TSA, **VadCLIP**, STPrompt, OVVAD,
Holmes-VAD / Holmes-VAU. Also **MultiHateLoc** (WWW 2026), which is the in-domain frame-level
weakly-supervised SOTA and therefore the number to beat, not a baseline to reproduce
(HateMM frame mAP 0.645 / AUC 0.799; MultiHateClip 0.445 / 0.750).

**Open item CLOSED:** **Alert-CLIP** (CVPR 2026 main conference) — abstract now read from the CVF
proceedings. Stage 1 of its method is "video-label alignment". **It uses labels; rejected on
filter 1**, joining the list above.

**New rejects on label-free from the 2025–2026 sweep:**

- **Vad-R1** (NeurIPS 2025, arXiv 2505.19877) — the flagship "video anomaly reasoning" paper. SFT on
  1,755 CoT-annotated videos plus AVA-GRPO RL on 6,448 **video-level weak labels**, and its
  Vad-Reasoning corpus is built *from* UCF-Crime and XD-Violence. It also reports no frame-level AUC
  at all. **VAU-R1** (arXiv-only) fails the same way.
- **Alert-CLIP**, **Fine-VAD**, **TLMA**, *Learning from Noisy Supervision*, *The Road Less Seen*,
  *Weakly Supervised VAD with Anomaly-Connected Components* — the CVPR 2026 main-track VAD crop is
  overwhelmingly weakly supervised.
- **TF-CADE** (CVPR 2026 main, zero-shot temporal action detection) — despite the "TF" prefix, its
  §Implementation says "we train our model for 25 epochs" on base classes. **The STALE trap**:
  "zero-shot" means unseen *classes*, not unseen labels.
- **SteerVAD** (ICLR 2026) — meta-controller consumes 1% of the target training set. Same shape as
  HiProbe-VAD.
- **TimeLens** (CVPR 2026 main), **TimeLens2**, **MeCo** (ICLR 2026), **TimePLE**, **UniversalVTG**,
  **VideoTG-R1**, **One-to-Many Temporal Grounding**, **ZoomV**, **TimeSearch-R**, **Zoom-Zero**,
  **TimeScope** — the entire 2025–2026 RL/SFT grounding wave trains on target-benchmark train
  splits. TimeLens is still worth citing for its **baseline table** (§1C.6), not as a method.
- **PreFM** (NeurIPS 2025) — audio-visual, trained on UnAV-100/LLP.
- **TANDEM** (arXiv 2601.11178) — in-domain and temporal ("timestamps and target identities" for
  multimodal hate speech), tandem RL between vision-language and audio-language models. Fails
  label-free *and* venue (AAAI-**ICWSM** 2027).

**In-domain re-sweep, 2025-01…2026-08 — clean negative, confirmed twice.** Nothing new in the
hateful/harmful/toxic video domain passes all three filters. The new A-venue in-domain work found is
uniformly supervised: **SCANNER** (AAAI 2026 main, first test-time-adaptation framework for hate
video detection, but whole-video and source-trained), **StreamSense** (WWW 2026, segment-level
streaming moderation *with public code*, fails only on label-free), **IARE** (SIGIR 2026 main, DPO
fine-tuning), **MM-HSD** (ACM MM 2025, video-level — but directly relevant to this project's OCR
decision, since it uses **on-screen text as the cross-modal-attention query** and the other
modalities as keys, and that configuration is what produces its gain), **ImpliHateVid** (ACL 2025
main), **SAGE** (ACL 2026 main), *Cross-Modal Transfer from Memes to Videos* (WWW 2025), and
**SafeLens** (AAAI 2026 **Demonstration Track**, 3 pages — a venue trap, see §3.2). The shape of the
gap is unchanged and now measured from two independent sweeps: **label-free + frame-level exists
only at arXiv tier** (LELA, no code); **A-venue + frame-level exists only supervised**
(MultiHateLoc, StreamSense); and **label-free + A-venue + open-source + frame-level** exists only as
generic video-anomaly work, never in-domain.

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
*or* third-party implementation. **The 2025–2026 sweep more than doubled this list** — see §1C.2 for
PANDA (NeurIPS 2025, repo is README + LICENSE only), Memory Matters (CVPR 2026 main), DART
(ECCV 2026), GranAlign (AAAI 2026), VTimeCoT (ICCV 2025), Self-SiMS (ECCV 2026), Moment-GPT
(AAAI 2025), and the correction that **MoniTor's repo is also a stub**. Absence of code is a
negative search result, not proof; all of these are worth a re-check in a few months.

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

> **Revised after the 2025–2026 sweep.** The first version of this list is kept below as §6.1 so the
> change is auditable. Two picks are replaced and one is inserted; ZS-CLIP and the URF-HVAA half of
> pick 1 survive.

### 6.0 Current recommended order

**1. LaGoVAD (ICLR 2026) — NEW, and it takes over as the lead VAD pick from LAVAD.**
Three reasons, in order of weight.
*Interface.* LaGoVAD's entire premise is that the anomaly definition is **written in free text at
inference time** and can be re-specified per dataset. That is the closest mechanical match in this
whole document to what this project needs — one prompt per HateClipSeg category, from
`gold_segments.json`'s 6-way label vector — and unlike §1B it does not assume one moment per query.
*Number.* XD-Violence AP **74.25**, which is +12.2 over LAVAD, +6.2 over URF-HVAA, and the best
label-free XD figure with released code. UCF-Crime 81.12 is unremarkable, but XD-Violence is the
violence-and-audio-adjacent benchmark, so it is the more relevant of the two here.
*Cost.* A CLIP-based detector, not an LLM pipeline — no captioning stage, no paid API, and the
weights, code and the PreVAD corpus are all released.
Supervision class is **(c)**: zero target labels, but the checkpoint saw 35,279 auxiliary web/stream
videos with weak labels. That is the same concession UniVTG already carried, for a much better
number. Two things to check before running: the paper reports **no near-duplicate audit** of PreVAD
against UCF/XD, and its Table 2 also contains `PreVAD+UCF` and `PreVAD+XD` rows that **do** use
target labels — the label-free row is the plain `PreVAD` one, 81.12 / 74.25.

**2. URF-HVAA (NeurIPS 2025), standalone — LAVAD demoted to its ablation floor.**
Unchanged in substance, changed in framing. URF-HVAA remains the strongest label-free UCF-Crime
number with working code (84.28 AUC / 68.07 XD AP), it runs on a single 3090-class GPU
(VideoLLaMA3-7B + Llama-3.1-8B), and it already has published HateMM/MultiHateClip frame-level
numbers via LELA (PR-AUC 0.6239) to check a port against. What changed: the sweep confirmed
**nothing in the 2025–2026 agentic or reasoning wave beats it under all three filters** — PANDA and
CEAVAD do beat it but neither is reproducible — so it is the frontier, not a stepping stone.
**LAVAD is no longer worth reproducing for its own sake.** Its repo has been dead since 2024-07-15,
its numbers are the weakest in the line, and its UCF AUC is reported three different ways across
papers (80.28 own / 79.21 in ESOM / 78.33 in EventVAD's reproduction). Run it only as URF-HVAA's ablation floor, which URF-HVAA's codebase
gives you nearly free.

**3. ZS-CLIP frame-similarity floor — unchanged, still first by schedule.**
Near-zero cost: the dense 4 fps CLIP-L/336 cache for all 395 HateClipSeg videos is already on disk,
so this is a text-encoder call plus a dot product — hours, no GPU queue, no video reads. It
establishes the chance-level floor (expect ROC-AUC ≈ 0.53–0.55, matching LELA's HateMM 0.5367) that
every later claim of "our method localises hate" must clear. Run it while item 1 is downloading.

**4. UniTime (NeurIPS 2025) — NEW, replaces UniVTG outright on the grounding axis.**
UniVTG is **retired from the recommended order**. The argument for keeping it was "cheapest real
method, saliency head is natively the curve you want". The argument against is now decisive:
UniTime is by the same task lineage, is also class (b) with zero target labels, has released code
(55★), and scores **zero-shot Charades-STA R@0.5 = 59.09 / mIoU 52.19 against UniVTG's 25.22** —
more than double. UniVTG's TACoS collapse (R@0.5 = 1.27) already showed it degrades badly
off-distribution, and hateful video is as off-distribution as it gets. The cost difference is real
(an MLLM forward per video rather than CLIP+SlowFast features) but it buys 34 points.
If a **fully training-free** grounding entry is wanted instead of class (b), the honest answer from
§1C.4 is that **TFVTG (ECCV 2024) is still the best one with code** — every A-venue training-free
method published since sits ~10 points below it, and the two that beat it (DART, VTimeCoT) have no
code. Cite TFVTG's 49.97 with its 3 FPS budget and its GPT-4-Turbo dependency attached.

**5. AV²A (CVPR 2025) — NEW, the audio channel.**
This is the cheapest way to stop the "every baseline scores visual frames only" objection from being
fatal. AV²A is fully training-free (frozen CLIP + frozen CLAP, zero gradient steps), takes a
**user-defined free-text label set per video**, and emits per-second scores **separately for
audio-only, visual-only and audio-visual** — which is directly the decomposition this project's
Gate-C analysis calls for. Run it as an audio floor, and read §1C.5's caveat before believing any
number: CLAP scores *acoustic events*, so it can detect "shouting" and cannot detect "slur".
Expect it to be a floor, like ZS-CLIP, not a contender. **OV-AVEL (CVPR 2025)** is the drop-in
alternative if ImageBind's joint space is preferable to score-level CLIP+CLAP fusion.

**Pending the §2 ruling, the fallback pick is still MULDE frame-centric (CVPR 2024)**, or **CLAP
(CVPR 2024)** if training on HateMM's `label == 0` pool is ruled out. Unchanged.

**Deliberately not in the order, and why:**

- **PANDA (NeurIPS 2025)** would be pick 1 or 2 on merit — agentic, training-free, best A-venue VAD
  numbers, online mode included. Its repository contains a README and a LICENSE. **Re-check monthly**;
  if the code lands it displaces URF-HVAA.
- **DART (ECCV 2026)** would be the grounding pick on merit (Charades 52.04 R@0.5, LLaVA-1.6-7B,
  no paid API, 3.9 s/query). No code. Same re-check note.
- **Memory Matters (CVPR 2026 main)** is the first A-venue method to beat T3AL on both benchmarks
  and is training-free. No code, but it is built on T3AL's protocol, so it is the one no-code entry
  that is realistically reimplementable — on top of T3AL's existing repo.
- **LAVIDA (CVPR 2026 main)** has the highest number anywhere in this document (XD-Violence AP
  90.62) and a confirmed main-track venue. It is not pick 1 because the repo ships training code
  with **no data-preparation instructions and no checkpoint**, so the true cost is "write the
  pipeline and re-train", and because its pseudo-anomaly prior — paste an out-of-context object —
  has no obvious hate analogue.
- **VADTree and EventVAD** — unchanged reasoning (GEBD checkpoint, RAFT optical flow, surveillance
  motion priors). Note EventVAD has by far the healthiest codebase in the document (536★) if a
  second caption-then-score entry is ever wanted.
- **The whole test-time-scaling wave** (CyberV, TimeSearch-R, Zoom-Zero, TimeScope) — reports no
  temporal-localisation metric at all, and *The Illusion of Progress?* (NeurIPS 2025 D&B) finds TTA
  methods mostly fail to beat the plain zero-shot baseline on fine-grained tasks.
- **T\*** — unchanged, still should not be ported: step one extracts a target object from the query,
  and "someone making a racist statement" has no detectable object.
- **The reconstruction-based one-class family** (MemAE / MNAD / AED-MAE) — unchanged, negative
  control only.

**The caveat that applies to every pick is unchanged and is now better evidenced.** Every visual
entry scores frames only; the ASR and OCR caches on disk have to be bolted into the query or caption
stream. And per §1's new note, **every §1B-style method assumes one moment per query while
HateClipSeg's median video has 3 disjoint toxic blocks** — a regime in which grounding-specialised
models measurably collapse to F1@0.5 below 6 with a 93–100% false-positive rate.

### 6.1 Original order (superseded by §6.0, kept for the record)

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

### 7.1 Direct answers to the second sweep's questions (2025–2026)

| Question | Finding |
|---|---|
| "The recommended picks are old and the mechanism is naive — is there anything newer?" | **Yes, and the constraint that bites is code, not novelty.** §1C.1 lists eight new entries that pass all three filters; §1C.2 lists nine that pass label-free + venue + frame-level and fail only on code, including the two best mechanisms found (PANDA, DART). Three picks in §6 changed as a result |
| "Agentic / multi-step VAD — O-VAD is on the list, who else?" | **PANDA (NeurIPS 2025), and at an A-venue nobody else.** RAG strategy planning + tool-augmented self-reflection (super-resolution, detection, retrieval, web search) + self-improving chain-of-memory. Beats every §1A entry: UCF 84.89 / XD AP 70.16 / UBnormal 75.78. **Repository is README + LICENSE only.** The wider 2026 agentic wave — QVAD, AgenticVAU, CEAVAD, ESOM — is entirely arXiv-only. O-VAD itself is now confirmed ECCV 2026 main *and* its real repo (`o-vad/O-VAD`) is complete — the link in §1A row 6 was wrong |
| "Has GLM-4.1V-Thinking-style CoT scoring been methodised at an A-venue?" | **No.** The 2025 reasoning wave (Vad-R1 NeurIPS 2025, VAU-R1) all **fails label-free** — SFT + RL on corpora built from UCF-Crime and XD-Violence — and Vad-R1 reports no frame-level AUC at all. The naive version is measured and it is weak: *Are MLLMs Ready for Surveillance?* finds a recall collapse, ShanghaiTech peak F1 **0.09 → 0.64** only with class-specific instructions |
| "Test-time scaling / TTT for zero-shot localisation — T3AL's successors?" | **Memory Matters (CVPR 2026 main) is the successor, and it is not what "test-time scaling" usually means.** Cross-video test-time memory via a learnable lookup table; **beats T3AL on both benchmarks** (THUMOS 12.6 vs 10.4, ANet 15.7 vs 14.3), training-free, **no code**. The papers actually branded test-time scaling (CyberV, TimeSearch-R, Zoom-Zero, TimeScope) report no localisation metric at all. **AdaZAD** (IEEE TMM) also beats T3AL cleanly (THUMOS 14.1) and fails only on venue. **OZ-TAL** does *not* beat T3AL (9.24 vs 10.4; ActivityNet 4.99 vs 14.3) and its method code is unreleased |
| "Training-free grounding after TFVTG — what is next-generation?" | **Fancier, and mostly worse.** No A-venue fully-training-free method published since TFVTG beats it except VTimeCoT by +1.05 R@0.5 (no code). Moment-GPT 38.4, NumPro 36.8, GranAlign 39.6, Self-SiMS 39.7 — all ~10 points below. **DART (ECCV 2026) beats it properly at 52.04 and has no code.** The arXiv crop beats it, but its own ablations show the gain is **backbone drift, not mechanism** (~+2 on a fixed backbone) |
| "Qwen2.5-VL native grounding as a zero-shot baseline — is there an A-venue definition?" | **Yes, and two A-venue sources disagree by 11.5 points.** TempSamp-R1 (NeurIPS 2025) reports Charades R@0.5 **54.4**; VideoChat-R1.5 (NeurIPS 2025) reports **42.9** for the same model. Cite the source and the harness. NumPro (CVPR 2025) is the systematic table for proprietary models, and its headline is that **GPT-4o direct timestamp emission (32.0) is worse than a 7B open model**. **Qwen3-VL / Qwen3-Omni / GPT-5 have no A-venue temporal-grounding evaluation at all** |
| "Streaming / online zero-shot after MoniTor" | **The niche is three papers deep and none is reproducible.** ESOM (86.18 AUC / 71.68 AP causal) beats MoniTor decisively but is arXiv-only; PANDA's online mode is A-venue with no code; **MoniTor's own repo is a stub** ("Code will be available", untouched since before the arXiv posting). Flashback's 87.29 / 75.13 is not a legitimate online result — it averages non-causally over the whole video — and it was **rejected at ICLR 2026** |
| "Audio-visual zero-shot — we care about audio" | **The axis is not empty: AV²A and OV-AVEL, both CVPR 2025 main, training-free, per-second, open-vocabulary, with code** (§1C.5). AV²A even scores audio-only and visual-only separately. **But they localise acoustic events, not linguistic content** — CLAP-style text towers cannot represent "racist statement". Use as an audio floor and prior-art citation. Hard negative: {zero-shot} × {uses audio} × {XD-Violence} is **empty** across the whole window |
| "2026 successors of URF-HVAA / VADTree / EventVAD / MoniTor / LAVAD?" | **None exists.** Checked each first author's full 2026 arXiv history, OpenAlex output, repo commit logs and DBLP. Two same-group follow-ups turned up, neither a successor: **BAGLM** (NeurIPS 2025, LAVAD's first author — training-free *online* step grounding, added as §1C.1 row 30) and **TASLE/MSLoc** (ICML 2026, MoniTor's lab — supervised AI-generated-content forensics). LAVAD's repo has been dead two years, EventVAD's over one; only VADTree is still maintained |

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
- **2026-venue claims (§3.2):** LAVIDA's and O-VAD's conference *tracks* are unverified, because the
  CVPR 2026 program API returns 403 from this environment and DBLP has not yet indexed either
  conference. Every other venue in this document is from a fully indexed year. Alert-CLIP's
  main-conference status was confirmed directly from the CVPR 2026 program JSON.
- **Sources used:** arXiv API (metadata + `comment` venue field), arXiv/CVF PDFs read locally with
  `pdftotext` for all result tables, HuggingFace papers search API for discovery, GitHub REST API for
  repo status, DBLP where reachable, OpenReview for the URF-HVAA citation. Semantic Scholar and
  OpenAlex were avoided as rate-limit-prone. Web search quota was exhausted partway through, so the
  second half of the sweep ran on arXiv + GitHub + direct page fetches only.

### 8.1 Verification status for the second sweep (§1C, and the §3.2/§4/§6/§7 revisions)

- **Read from the paper's own tables via `pdftotext`, in this session:** DART (Tables 1, 5, 6, 7 and
  implementation details), LAVIDA (Table 1), CEAVAD (Table 1), PANDA (Table 1 + LLM-backbone
  ablation), Memory Matters (Tables 1–2), TF-CADE (implementation section), CoMET-Bench (main
  results table). Everything else in §1C was verified by the parallel sweeps and is attributed below.
- **Venue verification, method by method.** CVPR 2026 claims (LAVIDA, OmniVTG, TimeLens, Memory
  Matters, Alert-CLIP, TF-CADE) were resolved against the **CVF open-access main vs Findings
  listings** — the definitive procedure documented in §3.2, established during this sweep. ICCV 2025
  and NeurIPS 2025 claims were checked against the CVF and papers.nips.cc proceedings listings, which
  is stronger evidence than DBLP. PANDA's NeurIPS 2025 status was confirmed twice (OpenReview
  `venue` field + repo README). Everything else rests on the arXiv `comment` field plus DBLP or
  OpenReview.
- **ECCV 2026 has no Findings track** — verified from the ECCV 2026 Call for Papers, which lists a
  single main technical program with workshops/tutorials/demos/doctoral-consortium as separate
  submission categories. This is what clears O-VAD, DART and Self-SiMS.
- **Repo status re-checked 2026-08-18 and unchanged** for all five originally-listed 2025 entries:
  lavad 151★/2024-07-15, URF-HVAA 10★/2025-12-10, VADTree 19★/2026-06-09, EventVAD 536★/2025-07-09,
  MoniTor 30★/2025-10-01.
- **Empty-repo claims were verified via the GitHub contents API**, not inferred from star counts:
  `showlab/PANDA` = README + LICENSE + assets; `yuanapril/OVAD-ECCV26` = README + index.html;
  `o-vad/O-VAD` = full implementation. `VitaminCreed/LAVIDA` has real code but its README states data
  preparation and usage instructions are still pending and no checkpoint is released.
- **Explicitly unverified, flagged in place:** the arXiv-only 2026 crop in §1C.3 (CEAVAD, ESOM,
  QVAD, AgenticVAU, SphereVAD, REZE, MTLA, FV-Action, MarkIt, CoMET) — silent `comment` fields and
  silent Semantic Scholar entries mean they may be under review; absence of a venue is not proof of
  rejection. Flashback is the one exception, where OpenReview shows an explicit ICLR 2026 rejection.
- **Number-pooling hazard, now measured.** LAVAD's UCF-Crime AUC appears as 80.28 (own paper), 79.21
  (ESOM's table) and 78.33 (EventVAD's reproduction); its XD AP is 62.01 everywhere except 60.02 in
  MoniTor's table. VADTree is 67.82 in its own paper and 68.85 in SphereVAD's. Qwen2.5-VL-7B's
  Charades R@0.5 is 54.4 in TempSamp-R1 and 42.9 in VideoChat-R1.5. **Do not build a comparison
  table by citing across papers — re-run.**
- **Known coverage gaps.** The WebSearch budget was exhausted (200/200) partway through, so the
  second sweep ran on arXiv + HuggingFace + CVF/ACL/NeurIPS proceedings + GitHub + OpenReview. The
  arXiv API rate-limited repeatedly, so a tail of very recent (2026-07/08) preprints may be missed —
  note also that `http://export.arxiv.org` returns an **empty body** from this environment (301 with
  no redirect follow); `https://` plus `curl -L` is required. `dblp.org` was intermittently
  unreachable from this IP; the `dblp.uni-trier.de` mirror works and surfaced five in-domain papers
  the arXiv/HF route could not see. The CVPR 2026 paginating program API (`/api/miniconf/events`)
  returns 403 and is a dead end — use the CVF listings.
- **CSAD, ComplexVAD, OpenDef-Bench, OV-AVEBench, CoMET-Bench** are new benchmarks introduced by
  papers in this sweep. None was evaluated; they are recorded only because their numbers appear in
  tables quoted above.
