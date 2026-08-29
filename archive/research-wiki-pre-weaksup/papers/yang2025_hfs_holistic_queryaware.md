---
type: paper
node_id: paper:yang2025_hfs_holistic_queryaware
title: "HFS: Holistic Query-Aware Frame Selection for Efficient Video Reasoning"
authors: ["Yiqing Yang", "Kin-Man Lam"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2512.11534"
  doi: null
  s2: null
tags: ["frame-selection", "top-k", "differentiable-selection", "MLLM-pseudo-labels", "selector-distillation", "mechanism-inspiration", "NOT-core-hateful-video"]
added: 2026-08-07T12:59:58Z
---

# HFS: Holistic Query-Aware Frame Selection for Efficient Video Reasoning

## One-line thesis
HFS: end-to-end trainable holistic query-aware frame selector that replaces independent top-K scoring and offline MLLM pseudo-labels with a task-adaptive selector.

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

> Key frame selection in video understanding presents significant challenges. Traditional top-K selection methods, which score frames independently, often fail to optimize the selection as a whole. This independent scoring frequently results in selecting frames that are temporally clustered and visually redundant. Additionally, training lightweight selectors using pseudo labels generated offline by Multimodal Large Language Models (MLLMs) prevents the supervisory signal from dynamically adapting to task objectives. To address these limitations, we propose an end-to-end trainable, task-adaptive framework for frame selection. A Chain-of-Thought approach guides a Small Language Model (SLM) to generate task-specific implicit query vectors, which are combined with multimodal features to enable dynamic frame scoring. We further define a continuous set-level objective function that incorporates relevance, coverage, and redundancy, enabling differentiable optimization via Gumbel-Softmax to select optimal frame combinations at the set level. Finally, student-teacher mutual learning is employed, where the student selector (SLM) and teacher reasoner (MLLM) are trained to align their frame importance distributions via KL divergence. Combined with cross-entropy loss, this enables end-to-end optimization, eliminating reliance on static pseudo labels. Experiments across various benchmarks, including Video-MME, LongVideoBench, MLVU, and NExT-QA, demonstrate that our method significantly outperforms existing approaches.

