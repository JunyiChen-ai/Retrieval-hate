---
type: paper
node_id: paper:koushik2025_towards_robust_framework
title: "Towards a Robust Framework for Multimodal Hate Detection: A Study on Video vs. Image-based Content"
authors: ["Girish A. Koushik", "Diptesh Kanojia", "Helen Treharne"]
year: 2025
venue: "Companion Proceedings of the ACM Web Conference 2025 (WWW Companion '25), April 28-May 2, 2025, Sydney, NSW, Australia"
external_ids:
  arxiv: "2502.07138"
  doi: null
  s2: null
tags: ["hateful-video", "HateMM", "audio", "CLAP", "HateXplain", "CLIP", "embedding-fusion", "MO-Hate", "BART", "ASR-Whisper", "hateful-memes", "HMC", "modality-ablation", "temporal-LSTM", "implicit-hate", "benign-confounders", "WWW2025", "video-vs-meme", "RGCL-baseline", "is_core_hateful_video"]
added: 2026-07-01T09:39:48Z
---

# Towards a Robust Framework for Multimodal Hate Detection: A Study on Video vs. Image-based Content

## One-line thesis
A simple concatenation-based embedding fusion of hate-tuned text, image, and audio-text encoders (HateXplain + CLIP + CLAP, HCC1) achieves state-of-the-art on video hate detection (HateMM) but fails on image-text memes, showing that hate detection needs modality-specific fusion architectures.

## Problem / Gap
_TODO._

## Method
_TODO._

## Key Results
_TODO._

## Assumptions
_TODO._

## Limitations / Failure Modes
_TODO._

## Reusable Ingredients
_TODO._

## Open Questions
_TODO._

## Claims
_TODO._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
_TODO._

## Abstract (original)

> Social media platforms enable the propagation of hateful content across different modalities such as textual, auditory, and visual, necessitating effective detection methods. While recent approaches have shown promise in handling individual modalities, their effectiveness across different modality combinations remains unexplored. This paper presents a systematic analysis of fusion-based approaches for multimodal hate detection, focusing on their performance across video and image-based content. Our comprehensive evaluation reveals significant modality-specific limitations: while simple embedding fusion achieves state-of-the-art performance on video content (HateMM dataset) with a 9.9% points F1-score improvement, it struggles with complex image-text relationships in memes (Hateful Memes dataset). Through detailed ablation studies and error analysis, we demonstrate how current fusion approaches fail to capture nuanced cross-modal interactions, particularly in cases involving benign confounders. Our findings provide crucial insights for developing more robust hate detection systems and highlight the need for modality-specific architectural considerations. The code is available at https://github.com/gak97/Video-vs-Meme-Hate.

