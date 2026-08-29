---
type: paper
node_id: paper:yue2025_multimodal_hate_detection
title: "Multimodal Hate Detection Using Dual-Stream Graph Neural Networks"
authors: ["Jiangbei Yue", "Shuonan Yang", "Tailin Chen", "Jianbo Jiao", "Zeyu Fu"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2509.13515"
  doi: null
  s2: null
tags: ["hateful-video", "graph-neural-network", "dual-stream", "instance-importance-weighting", "multimodal-fusion", "explainability", "weak-localization", "HateMM", "MultiHateClip", "BMVC-2025", "ViT", "MFCC", "Whisper", "BERT", "temporal", "audio", "SOTA-baseline", "is_core_hateful_video"]
added: 2026-07-01T09:39:45Z
---

# Multimodal Hate Detection Using Dual-Stream Graph Neural Networks

## One-line thesis
A dual-stream graph neural network for hateful-video detection that separates a video into temporal instances, uses a complementary weight graph to assign per-instance importance so even minimal hateful content is emphasized, and combines instance features with these weights to produce an explainable video-level hate label.

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

> Hateful videos present serious risks to online safety and real-world well-being, necessitating effective detection methods. Although multimodal classification approaches integrating information from several modalities outperform unimodal ones, they typically neglect that even minimal hateful content defines a video's category. Specifically, they generally treat all content uniformly, instead of emphasizing the hateful components. Additionally, existing multimodal methods cannot systematically capture structured information in videos, limiting the effectiveness of multimodal fusion. To address these limitations, we propose a novel multimodal dual-stream graph neural network model. It constructs an instance graph by separating the given video into several instances to extract instance-level features. Then, a complementary weight graph assigns importance weights to these features, highlighting hateful instances. Importance weights and instance features are combined to generate video labels. Our model employs a graph-based framework to systematically model structured relationships within and across modalities. Extensive experiments on public datasets show that our model is state-of-the-art in hateful video classification and has strong explainability. Code is available: https://github.com/Multimodal-Intelligence-Lab-MIL/MultiHateGNN.

