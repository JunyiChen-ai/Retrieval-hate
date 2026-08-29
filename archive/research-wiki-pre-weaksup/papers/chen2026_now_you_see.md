---
type: paper
node_id: paper:chen2026_now_you_see
title: "Now You See the Hate: Adaptive View Retrieval for Hidden Hateful Illusions"
authors: ["Qianpu Chen", "Derya Soydaner"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2607.19061"
  doi: null
  s2: null
tags: ["hateful-content", "retrieval", "view-selection", "evidence-calibration", "image-domain", "inspiration", "NOT-core-hateful-video"]
added: 2026-08-07T13:00:13Z
---

# Now You See the Hate: Adaptive View Retrieval for Hidden Hateful Illusions

## One-line thesis
Adaptive View Retrieval reframes hidden hateful-illusion detection as perceptual retrieval over a complementary view bank, adaptively selecting which views to trust and calibrating recovered evidence.

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

> Hateful optical illusions expose a serious gap in current multimodal safety systems. On original-view hateful illusions, previous work shows that six moderation classifiers achieve at most 20.9 to 24.5% accuracy and nine state-of-the-art VLMs remain at or below 10.2% with illusion-aware prompting, leaving most hidden hate undetected. We formulate hidden hateful illusion detection as a perceptual retrieval problem and propose Adaptive View Retrieval. This retrieve-and-calibrate framework assembles a complementary view bank for the image and hidden-message templates, adaptively selects which views to trust, retrieves hidden-message identities, and calibrates whether the recovered evidence is harmful. On HatefulIllusion with a frozen CLIP encoder, Adaptive View Retrieval reaches 93.2% balanced accuracy on the held-out test split. It substantially outperforms original-view baselines and fixed single-transform filters across hate slangs, hate symbols, and visibility levels. The same design also surpasses official fine-tuned CLIP baselines, matches or exceeds human performance on IllusionMNIST, IllusionFashionMNIST, and IllusionAnimals, and outperforms zoom-out preprocessing on HC-Bench under the SemVink protocol. Together, these results show that robust multimodal moderation requires recovering hidden meaning before deciding whether it is harmful.

