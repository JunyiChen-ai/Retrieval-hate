---
type: paper
node_id: paper:koushik2026_tandem_temporalaware_neural
title: "TANDEM: Temporal-Aware Neural Detection for Multimodal Hate Speech"
authors: ["Girish A. Koushik", "Helen Treharne", "Diptesh Kanojia"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2601.11178"
  doi: null
  s2: null
tags: ["hateful-video", "temporal-grounding", "target-identification", "audio-visual", "reinforcement-learning", "GRPO", "GSPO", "VLM", "audio-LM", "Qwen2.5-VL", "Qwen2-Audio", "HateMM", "MultiHateClip", "ImpliHateVid", "LoRA", "interpretability", "long-video", "structured-reasoning", "moderation", "is_core_hateful_video"]
added: 2026-07-01T09:39:47Z
---

# TANDEM: Temporal-Aware Neural Detection for Multimodal Hate Speech

## One-line thesis
TANDEM reframes audio-visual hateful-video detection from binary classification into a structured reasoning task, using a tandem reinforcement-learning scheme where a vision-language model and an audio-language model mutually optimize each other via self-constrained cross-modal context, producing timestamps and target identities without dense frame-level supervision.

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

> Social media platforms are increasingly dominated by long-form multimodal content, where harmful narratives are constructed through a complex interplay of audio, visual, and textual cues. While automated systems can flag hate speech with high accuracy, they often function as "black boxes" that fail to provide the granular, interpretable evidence, such as precise timestamps and target identities, required for effective human-in-the-loop moderation. In this work, we introduce TANDEM, a unified framework that transforms audio-visual hate detection from a binary classification task into a structured reasoning problem. Our approach employs a novel tandem reinforcement learning strategy where vision-language and audio-language models optimize each other through self-constrained cross-modal context, stabilizing reasoning over extended temporal sequences without requiring dense frame-level supervision. Experiments across three benchmark datasets demonstrate that TANDEM significantly outperforms zero-shot and context-augmented baselines, achieving 0.73 F1 in target identification on HateMM (a 30% improvement over state-of-the-art) while maintaining precise temporal grounding. We further observe that while binary detection is robust, differentiating between offensive and hateful content remains challenging in multi-class settings due to inherent label ambiguity and dataset imbalance. More broadly, our findings suggest that structured, interpretable alignment is achievable even in complex multimodal settings, offering a blueprint for the next generation of transparent and actionable online safety moderation tools.

