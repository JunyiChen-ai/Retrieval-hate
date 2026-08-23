---
type: paper
node_id: paper:singh2025_labels_input_rethinking
title: "Labels or Input? Rethinking Augmentation in Multimodal Hate Detection"
authors: ["Sahajpreet Singh", "Kokil Jaidka", "Subhayan Mukerjee"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2508.11808"
  doi: null
  s2: null
tags: ["hateful-meme", "augmentation", "label-noise", "inspiration", "NOT-core-hateful-video"]
added: 2026-08-07T13:00:21Z
---

# Labels or Input? Rethinking Augmentation in Multimodal Hate Detection

## One-line thesis
Labels or Input? rethinks augmentation in multimodal hate detection, separating label-space from input-space interventions.

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

> Online hate remains a significant societal challenge, especially as multimodal content enables subtle, culturally grounded, and implicit forms of harm. Hateful memes embed hostility through text-image interactions and humor, making them difficult for automated systems to interpret. Although recent Vision-Language Models (VLMs) perform well on explicit cases, their deployment is limited by high inference costs and persistent failures on nuanced content. This work examines how far small models can be improved through prompt optimization, fine-tuning, and automated data augmentation. We introduce an end-to-end pipeline that varies prompt structure, label granularity, and training modality, showing that structured prompts and scaled supervision significantly strengthen compact VLMs. We also develop a multimodal augmentation framework that generates counterfactually neutral memes via a coordinated LLM-VLM setup, reducing spurious correlations and improving the detection of implicit hate. Ablation studies quantify the contribution of each component, demonstrating that prompt design, granular labels, and targeted augmentation collectively narrow the gap between small and large models. The results offer a practical path toward more robust and deployable multimodal hate-detection systems without relying on costly large-model inference.

