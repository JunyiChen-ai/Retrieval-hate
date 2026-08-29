---
type: paper
node_id: paper:cspedessarrias2025_mmhsd_multimodal_hate
title: "MM-HSD: Multi-Modal Hate Speech Detection in Videos"
authors: ["Berta Céspedes-Sarrias", "Carlos Collado-Capell", "Pablo Rodenas-Ruiz", "Olena Hrynenko", "Andrea Cavallaro"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2508.20546"
  doi: null
  s2: null
tags: ["hateful-video-detection", "HateMM", "cross-modal-attention", "multimodal-fusion", "OCR-on-screen-text", "wav2vec2", "ViT", "LSTM", "Whisper", "Detoxify", "PaddleOCR", "ACM-MM-2025", "video-level-SOTA", "early-fusion", "weighted-cross-entropy", "is_core_hateful_video"]
added: 2026-07-01T09:39:45Z
---

# MM-HSD: Multi-Modal Hate Speech Detection in Videos

## One-line thesis
MM-HSD is a tetra-modal video hate-speech detector that uses Cross-Modal Attention (CMA) as an early feature extractor to fuse transcript, audio, video frames, and on-screen (OCR) text, achieving state-of-the-art macro-F1 (0.874) on HateMM by letting on-screen text query the other modalities.

## Problem / Gap
Video hate-speech uses simple concatenation fusion that misses inter-modal dependencies and omits on-screen/OCR text + audio waveform. No prior work systematically compared CMA query/key configs or used CMA as an early fusion extractor for video HSD.

## Method
Tetra-modal: transcript (T), audio waveform (A), frames (V), on-screen OCR text (O). Frozen encoders (small dataset): Detoxify for T+O, wav2vec2-xlsr-53 audio, ViT+LSTM frames, Whisper ASR, PaddleOCR. Core = **Cross-Modal Attention** CMA(K,Q,V) with K=V, query varied; applied to raw embeddings to make an extra feature concatenated with per-modality encoders before the head. Best config: **O (OCR) as query, T+A+V as key/value**. 5-fold CV, weighted CE, elastic-net. **No retrieval, no contrastive.**

## Key Results
HateMM: **M-F1 0.874** (std .009), ACC 0.878, hate-F1 0.853 — **current published video-level SOTA on HateMM**. Baselines: HateMM-orig 0.790, HXP+CLAP+CLIP 0.848, LLaMA-3.2-11B 0.820, TCE-DBF 0.840 (but higher hate-F1 0.876). Best single modality = transcript T 0.816; OCR alone weakest 0.594. CMA lifts 0.846->0.874.

## Limitations / Failure Modes
**Only HateMM** (single English BitChute set); no cross-dataset, no crosslingual/Chinese, no moment-level localization, no implicit-hate focus, **no retrieval/contrastive**. Loses to TCE-DBF on hate-specific F1. Large offline encoders.

## Reusable Ingredients
(1) **Exact HateMM SOTA target M-F1 0.874, hate-F1 0.853** + baseline table. (2) Frozen-encoder + light-head recipe (Detoxify, wav2vec2-xlsr-53, ViT+LSTM, PaddleOCR). (3) OCR as a first-class modality, best used as query. (4) CMA as cheap early-fusion module. (5) 5-fold CV + weighted CE. (6) Open repo github.com/idiap/mm-hsd.

## Relevance to This Project
The **current published video-level SOTA on HateMM (M-F1 0.874)** — primary numeric baseline to beat. Complementary: strong encoders + CMA but NO retrieval/NO contrastive (our novelty axes). Adopt its encoder stack as feature front-end; RGCL retrieval-guided contrastive is the differentiator. Gaps (English-only, video-level, no implicit focus) map onto our open directions.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
- Video-level HateMM SOTA; our method must beat M-F1 0.874.

## Abstract (original)

> While hate speech detection (HSD) has been extensively studied in text, existing multi-modal approaches remain limited, particularly in videos. As modalities are not always individually informative, simple fusion methods fail to fully capture inter-modal dependencies. Moreover, previous work often omits relevant modalities such as on-screen text and audio, which may contain subtle hateful content and thus provide essential cues, both individually and in combination with others. In this paper, we present MM-HSD, a multi-modal model for HSD in videos that integrates video frames, audio, and text derived from speech transcripts and from frames (i.e.~on-screen text) together with features extracted by Cross-Modal Attention (CMA). We are the first to use CMA as an early feature extractor for HSD in videos, to systematically compare query/key configurations, and to evaluate the interactions between different modalities in the CMA block. Our approach leads to improved performance when on-screen text is used as a query and the rest of the modalities serve as a key. Experiments on the HateMM dataset show that MM-HSD outperforms state-of-the-art methods on M-F1 score (0.874), using concatenation of transcript, audio, video, on-screen text, and CMA for feature extraction on raw embeddings of the modalities. The code is available at https://github.com/idiap/mm-hsd

