---
type: paper
node_id: paper:wang2025_hateclipseg_segmentlevel_annotated
title: "HateClipSeg: A Segment-Level Annotated Dataset for Fine-Grained Hate Video Detection"
authors: ["Han Wang", "Zhuoran Wang", "Roy Ka-Wei Lee"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2508.01712"
  doi: null
  s2: null
tags: ["hateful-video", "segment-level", "temporal-localization", "dataset", "benchmark", "online-detection", "multimodal", "audio", "implicit-hate", "ACM-MM-2025", "English", "ActionFormer", "LLaMA", "LSTR", "offensive-type-taxonomy", "target-victim", "is_core_hateful_video"]
added: 2026-07-01T09:39:44Z
---

# HateClipSeg: A Segment-Level Annotated Dataset for Fine-Grained Hate Video Detection

## One-line thesis
HateClipSeg is the first large-scale multimodal hate-video dataset with both video-level AND fine-grained segment-level offensive-type + target-victim annotations, exposing that current models handle trimmed classification only moderately and collapse on temporal localization, motivating temporally-aware multimodal moderation.

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

> Detecting hate speech in videos remains challenging due to the complexity of multimodal content and the lack of fine-grained annotations in existing datasets. We present HateClipSeg, a large-scale multimodal dataset with both video-level and segment-level annotations, comprising over 11,714 segments labeled as Normal or across five Offensive categories: Hateful, Insulting, Sexual, Violence, Self-Harm, along with explicit target victim labels. Our three-stage annotation process yields high inter-annotator agreement (Krippendorff's alpha = 0.817). We propose three tasks to benchmark performance: (1) Trimmed Hateful Video Classification, (2) Temporal Hateful Video Localization, and (3) Online Hateful Video Classification. Results highlight substantial gaps in current models, emphasizing the need for more sophisticated multimodal and temporally aware approaches. The HateClipSeg dataset are publicly available at https://github.com/Social-AI-Studio/HateClipSeg.git.

