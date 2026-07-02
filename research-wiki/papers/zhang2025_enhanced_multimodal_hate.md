---
type: paper
node_id: paper:zhang2025_enhanced_multimodal_hate
title: "Enhanced Multimodal Hate Video Detection via Channel-wise and Modality-wise Fusion"
authors: ["Yinghui Zhang", "Tailin Chen", "Yuchen Zhang", "Zeyu Fu"]
year: 2025
venue: "2024 IEEE International Conference on Data Mining Workshops (ICDMW), Abu Dhabi, United Arab Emirates, 2024, pp. 183-190"
external_ids:
  arxiv: "2505.12051"
  doi: null
  s2: null
tags: ["hateful-video", "multimodal-fusion", "audio", "temporal-cross-attention", "HateMM", "supervised-baseline", "channel-wise-fusion", "modality-gating", "ViT", "BERT", "MFCC", "ICDMW-2024", "is_core_hateful_video"]
added: 2026-07-01T09:39:46Z
---

# Enhanced Multimodal Hate Video Detection via Channel-wise and Modality-wise Fusion

## One-line thesis
CMFusion detects hate videos by fusing text, audio, and video via a temporal cross-attention between video and audio plus channel-wise and modality-wise fusion modules, beating unimodal and prior multimodal baselines on HateMM.

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

> The rapid rise of video content on platforms such as TikTok and YouTube has transformed information dissemination, but it has also facilitated the spread of harmful content, particularly hate videos. Despite significant efforts to combat hate speech, detecting these videos remains challenging due to their often implicit nature. Current detection methods primarily rely on unimodal approaches, which inadequately capture the complementary features across different modalities. While multimodal techniques offer a broader perspective, many fail to effectively integrate temporal dynamics and modality-wise interactions essential for identifying nuanced hate content. In this paper, we present CMFusion, an enhanced multimodal hate video detection model utilizing a novel Channel-wise and Modality-wise Fusion Mechanism. CMFusion first extracts features from text, audio, and video modalities using pre-trained models and then incorporates a temporal cross-attention mechanism to capture dependencies between video and audio streams. The learned features are then processed by channel-wise and modality-wise fusion modules to obtain informative representations of videos. Our extensive experiments on a real-world dataset demonstrate that CMFusion significantly outperforms five widely used baselines in terms of accuracy, precision, recall, and F1 score. Comprehensive ablation studies and parameter analyses further validate our design choices, highlighting the model's effectiveness in detecting hate videos. The source codes will be made publicly available at https://github.com/EvelynZ10/cmfusion.

