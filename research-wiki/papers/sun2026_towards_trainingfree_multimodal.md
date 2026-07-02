---
type: paper
node_id: paper:sun2026_towards_trainingfree_multimodal
title: "Towards Training-free Multimodal Hate Localisation with Large Language Models"
authors: ["Yueming Sun", "Long Yang", "Jianbo Jiao", "Zeyu Fu"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2602.09637"
  doi: null
  s2: null
tags: ["hateful-video-detection", "temporal-localization", "frame-level", "training-free", "LLM-prompting", "multimodal", "chain-of-thought", "HateMM", "MultiHateClip", "zero-shot", "audio-music", "OCR", "speech", "implicit-hate", "video-anomaly-detection-transfer", "GPT-4o-mini", "is_core_hateful_video"]
added: 2026-07-01T09:39:48Z
---

# Towards Training-free Multimodal Hate Localisation with Large Language Models

## One-line thesis
LELA is the first training-free, LLM-based framework for frame-level hateful-video localization: it decomposes a video into five caption modalities and uses multi-stage prompting plus composition matching to assign per-frame hate scores, beating all training-free baselines and approaching supervised methods on HateMM and MultiHateClip.

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

> The proliferation of hateful content in online videos poses severe threats to individual well-being and societal harmony. However, existing solutions for video hate detection either rely heavily on large-scale human annotations or lack fine-grained temporal precision. In this work, we propose LELA, the first training-free Large Language Model (LLM) based framework for hate video localization. Distinct from state-of-the-art models that depend on supervised pipelines, LELA leverages LLMs and modality-specific captioning to detect and temporally localize hateful content in a training-free manner. Our method decomposes a video into five modalities, including image, speech, OCR, music, and video context, and uses a multi-stage prompting scheme to compute fine-grained hateful scores for each frame. We further introduce a composition matching mechanism to enhance cross-modal reasoning. Experiments on two challenging benchmarks, HateMM and MultiHateClip, demonstrate that LELA outperforms all existing training-free baselines by a large margin. We also provide extensive ablations and qualitative visualizations, establishing LELA as a strong foundation for scalable and interpretable hate video localization.

