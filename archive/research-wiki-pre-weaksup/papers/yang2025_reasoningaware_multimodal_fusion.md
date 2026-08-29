---
type: paper
node_id: paper:yang2025_reasoningaware_multimodal_fusion
title: "Reasoning-Aware Multimodal Fusion for Hateful Video Detection"
authors: ["Shuonan Yang", "Tailin Chen", "Jiangbei Yue", "Guangliang Cheng", "Jianbo Jiao", "Zeyu Fu"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2512.02743"
  doi: null
  s2: null
tags: ["hateful-video", "multimodal-fusion", "VLM-reasoning", "implicit-hate", "cross-lingual", "chinese", "temporal", "audio", "HateMM", "MultiHateClip", "cross-attention", "baseline-to-beat", "Qwen2.5-VL", "is_core_hateful_video"]
added: 2026-07-01T09:39:49Z
---

# Reasoning-Aware Multimodal Fusion for Hateful Video Detection

## One-line thesis
RAMF is a trainable multimodal fusion framework combining Local-Global Context Fusion + Semantic Cross Attention over frame/audio/transcript features with three-perspective adversarial-reasoning text (objective / hate-assumed / non-hate-assumed) from a frozen VLM, achieving SOTA on HateMM and both MultiHateClip (English + Chinese) subsets.

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

> Hate speech in online videos is posing an increasingly serious threat to digital platforms, especially as video content becomes increasingly multimodal and context-dependent. Existing methods often struggle to effectively fuse the complex semantic relationships between modalities and lack the ability to understand nuanced hateful content. To address these issues, we propose an innovative Reasoning-Aware Multimodal Fusion (RAMF) framework. To tackle the first challenge, we design Local-Global Context Fusion (LGCF) to capture both local salient cues and global temporal structures, and propose Semantic Cross Attention (SCA) to enable fine-grained multimodal semantic interaction. To tackle the second challenge, we introduce adversarial reasoning-a structured three-stage process where a vision-language model generates (i) objective descriptions, (ii) hate-assumed inferences, and (iii) non-hate-assumed inferences-providing complementary semantic perspectives that enrich the model's contextual understanding of nuanced hateful intent. Evaluations on two real-world hateful video datasets demonstrate that our method achieves robust generalisation performance, improving upon state-of-the-art methods by 3% and 7% in Macro-F1 and hate class recall, respectively. The source codes and data required to reproduce our results are available at https://github.com/Multimodal-Intelligence-Lab-MIL/RAMF.

