---
type: paper
node_id: paper:mei2025_expohm_learning_explainthendetect
title: "ExPO-HM: Learning to Explain-then-Detect for Hateful Meme Detection"
authors: ["Jingbiao Mei", "Mingsheng Sun", "Jinghong Chen", "Pengda Qin", "Yuhong Li", "Da Chen", "Bill Byrne"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2510.08630"
  doi: null
  s2: null
tags: ["hateful-meme", "explain-then-detect", "policy-optimization", "rationale", "inspiration", "NOT-core-hateful-video"]
added: 2026-08-07T13:00:18Z
---

# ExPO-HM: Learning to Explain-then-Detect for Hateful Meme Detection

## One-line thesis
ExPO-HM learns to explain-then-detect for hateful memes, using policy optimization over explanation-then-decision traces.

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

> Hateful memes have emerged as a particularly challenging form of online abuse, motivating the development of automated detection systems. Most prior approaches rely on direct detection, producing only binary predictions. Such models fail to provide the context and explanations that real-world moderation requires. Recent Explain-then-Detect approaches, using Chain-of-Thought prompting or LMM agents, perform worse than simple SFT baselines, and even advanced post-training methods such as GRPO fail to close the gap. Our analysis identifies two key issues of such systems: important policy-relevant cues such as targets and attack types are not hypothesized by the model as a likely explanation; and the binary reward signal is insufficient to guide reasoning. To address these challenges, we propose ExPO-HM (Explain-then-Detect Policy Optimization for Hateful Memes), inspired by the training and evaluation process of human annotators. ExPO-HM combines SFT warmup, GRPO with curriculum learning, and Conditional Decision Entropy (CDE) as both metric and reward for reasoning quality. Across three hateful meme benchmarks, ExPO-HM achieves state-of-the-art performance on binary detection, fine-grained classification, and reasoning quality, with up to 15\% and 17\% F1 improvement over the GRPO and DPO baselines, respectively. By moving hateful meme detection from simple binary alarms to explanation-driven detection, ExPO-HM provides accurate, interpretable, and actionable moderation support. Code available at https://github.com/JingbiaoMei/ExPO-HM

