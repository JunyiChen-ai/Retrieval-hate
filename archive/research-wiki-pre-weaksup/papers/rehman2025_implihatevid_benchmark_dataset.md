---
type: paper
node_id: paper:rehman2025_implihatevid_benchmark_dataset
title: "ImpliHateVid: A Benchmark Dataset and Two-stage Contrastive Learning Framework for Implicit Hate Speech Detection in Videos"
authors: ["Mohammad Zia Ur Rehman", "Anukriti Bhatnagar", "Omkar Kabde", "Shubhi Bansal", "Nagendra Kumar"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2508.06570"
  doi: null
  s2: null
tags: ["hateful-video", "implicit-hate", "contrastive-learning", "supervised-contrastive", "multimodal", "benchmark-dataset", "audio", "ASR-transcript", "ImageBind", "Wav2Vec2", "BERT", "sentiment", "emotion", "caption", "HateMM", "ACL2025", "English-only", "SOTA", "is_core_hateful_video"]
added: 2026-07-01T09:39:44Z
---

# ImpliHateVid: A Benchmark Dataset and Two-stage Contrastive Learning Framework for Implicit Hate Speech Detection in Videos

## One-line thesis
Introduces ImpliHateVid, the first video benchmark dedicated to implicit (no-slur) hate speech, together with a two-stage supervised-contrastive multimodal framework (audio + transcript + frames, plus sentiment/emotion/caption features) that sets new SOTA on both ImpliHateVid and HateMM.

## Problem / Gap
Video hate detection underexplored; no video dataset targets IMPLICIT (no-slur) hate, which is harder than explicit. No benchmark or method purpose-built for implicit video hate.

## Method
Two-stage supervised contrastive. Encoders: Wav2Vec2 (audio), BERT (ASR transcript, FFmpeg), ImageBind (frames). Complementary implicit-signal features: NRCLex emotion + VADER sentiment (fES), OFA captions -> BERT (fC). Stage 1: SupCon on the three modality encoders over concatenated tri-modal features. Stage 2: cross-encoder SupCon to refine fused rep. L_total = L_stage1+L_stage2+L_supES+L_supCP. **Contrastive (SupCon by class label) but NO retrieval/kNN.**

## Key Results
ImpliHateVid **F1 87.73%** (vs Wav2Vec2 77.24%, MulT 83.52%). HateMM (binary) F1 **97.58%** — but their re-run HateMM baselines (MulT 52.12%, CSID 71.40%) are far below the field norm (~0.79), so the +26pt margin is vs weak re-implementations; treat with caution.

## Limitations / Failure Modes
Depends on cross-modality alignment; noisy ASR degrades embeddings; frozen encoders hurt under shift; SupCon temperature-sensitive. **English-only, NO retrieval/kNN.** HateMM baseline numbers suspiciously low.

## Reusable Ingredients
(1) **Two-stage SupCon recipe** (per-modality then cross-encoder) — our contrastive axis; combinable with retrieval it LACKS. (2) Implicit complementary features (VADER + NRCLex + OFA captions). (3) ImageBind unified backbone. (4) **ImpliHateVid dataset** (2,009 videos: 509 implicit/500 explicit/1,000 non-hate, English, BitChute+Odysee) — implicit-axis benchmark. (5) Mental-health-aware annotation protocol.

## Relevance to This Project
Core for implicit AND contrastive axes: only implicit-video dataset + canonical contrastive method for hateful video. Defines ImpliHateVid benchmark. Uses contrastive but NOT retrieval/kNN — retrieval-guided contrastive (our RGCL adaptation) is a clear unclaimed extension. English-only leaves crosslingual/Chinese open.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
- Closest contrastive prior in hateful video; RGCL adds retrieval/kNN on top.

## Abstract (original)

> The existing research has primarily focused on text and image-based hate speech detection, video-based approaches remain underexplored. In this work, we introduce a novel dataset, ImpliHateVid, specifically curated for implicit hate speech detection in videos. ImpliHateVid consists of 2,009 videos comprising 509 implicit hate videos, 500 explicit hate videos, and 1,000 non-hate videos, making it one of the first large-scale video datasets dedicated to implicit hate detection. We also propose a novel two-stage contrastive learning framework for hate speech detection in videos. In the first stage, we train modality-specific encoders for audio, text, and image using contrastive loss by concatenating features from the three encoders. In the second stage, we train cross-encoders using contrastive learning to refine multimodal representations. Additionally, we incorporate sentiment, emotion, and caption-based features to enhance implicit hate detection. We evaluate our method on two datasets, ImpliHateVid for implicit hate speech detection and another dataset for general hate speech detection in videos, HateMM dataset, demonstrating the effectiveness of the proposed multimodal contrastive learning for hateful content detection in videos and the significance of our dataset.

