---
type: paper
node_id: paper:li2025_select_less_reason
title: "Select Less, Reason More: Prioritizing Evidence Purity for Video Reasoning"
authors: ["Xuchen Li", "Xuzhao Li", "Shiyu Hu", "Kaiqi Huang"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2510.15440"
  doi: null
  s2: null
tags: ["video-LLM", "frame-selection", "evidence-purity", "reinforcement-learning", "information-dilution", "mechanism-inspiration", "NOT-core-hateful-video"]
added: 2026-08-07T12:59:57Z
---

# Select Less, Reason More: Prioritizing Evidence Purity for Video Reasoning

## One-line thesis
Select Less, Reason More: evidence-purity-rewarded adaptive frame selection for long-video reasoning, arguing uniform sampling dilutes critical evidence.

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

> Long-form video reasoning remains a major challenge for Video Large Language Models (Video LLMs), as static uniform frame sampling leads to information dilution and obscures critical evidence. Furthermore, existing pixel-space video reasoning agents, which are designed to actively interact with the video to acquire new visual information, remain suboptimal due to their lack of rigorous reward mechanisms to enforce evidence purity and their inability to perform temporal information supplementation beyond pre-sampled frames. To address this critical gap, we propose a novel evidence-prioritized adaptive framework built upon our core philosophy: "Select Less, Reason More." Our core contribution is the evidence-aware reinforcement learning (EARL) framework, which transforms the model into an active interrogator of evidence. EARL is precisely engineered to dynamically select the most relevant frames and, crucially, to perform localized re-sampling around the selected key frames to access fine-grained temporal detail. Extensive experiments on five demanding video reasoning benchmarks demonstrate that our EARL-trained model achieves new state-of-the-art among open-source Video LLMs, simultaneously learning an effective and high-purity visual evidence selection policy. Impressively, our 7B model achieves 59.8% on LongVideoBench, 69.0% on MVBench and 64.9% on VideoMME. These results highlight the importance of prioritizing evidence purity and the effectiveness of our framework.

