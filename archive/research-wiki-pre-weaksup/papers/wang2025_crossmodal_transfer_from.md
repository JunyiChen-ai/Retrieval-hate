---
type: paper
node_id: paper:wang2025_crossmodal_transfer_from
title: "Cross-Modal Transfer from Memes to Videos: Addressing Data Scarcity in Hateful Video Detection"
authors: ["Han Wang", "Rui Yang Tan", "Roy Ka-Wei Lee"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2501.15438"
  doi: null
  s2: null
tags: ["hateful-video-detection", "cross-modal-transfer", "meme-to-video", "data-scarcity", "VLM-finetuning", "LoRA", "few-shot-prompting", "label-reannotation", "human-in-the-loop", "MultiHateClip", "HateMM", "FHM", "MAMI", "LLaMA-3.2-11B", "LLaVA-NeXT-Video", "WWW2025", "data-augmentation", "SOTA-benchmark", "is_core_hateful_video"]
added: 2026-07-01T09:39:46Z
---

# Cross-Modal Transfer from Memes to Videos: Addressing Data Scarcity in Hateful Video Detection

## One-line thesis
Abundant hateful-meme datasets, once re-annotated to align labels with video-task definitions, can substitute for or augment scarce hateful-video data to fine-tune VLMs, matching or beating video-only training on MHC and HateMM.

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

> Detecting hate speech in online content is essential to ensuring safer digital spaces. While significant progress has been made in text and meme modalities, video-based hate speech detection remains under-explored, hindered by a lack of annotated datasets and the high cost of video annotation. This gap is particularly problematic given the growing reliance on large models, which demand substantial amounts of training data. To address this challenge, we leverage meme datasets as both a substitution and an augmentation strategy for training hateful video detection models. Our approach introduces a human-assisted reannotation pipeline to align meme dataset labels with video datasets, ensuring consistency with minimal labeling effort. Using two state-of-the-art vision-language models, we demonstrate that meme data can substitute for video data in resource-scarce scenarios and augment video datasets to achieve further performance gains. Our results consistently outperform state-of-the-art benchmarks, showcasing the potential of cross-modal transfer learning for advancing hateful video detection. Dataset and code are available at https://github.com/Social-AI-Studio/CrossModalTransferLearning.

