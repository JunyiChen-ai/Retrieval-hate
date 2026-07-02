---
type: paper
node_id: paper:yang2026_trainingfree_interpretable_hateful
title: "Training-Free and Interpretable Hateful Video Detection via Multi-stage Adversarial Reasoning"
authors: ["Shuonan Yang", "Yuchen Zhang", "Zeyu Fu"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2601.15115"
  doi: null
  s2: null
tags: ["hateful-video", "training-free", "VLM", "LLM-reasoning", "adversarial-reasoning", "interpretability", "false-positive-reduction", "chain-of-thought-alternative", "HateMM", "MultiHateClip", "Chinese", "Bilibili", "crosslingual", "precision-oriented", "prompt-engineering", "content-moderation", "EU-AI-Act", "is_core_hateful_video"]
added: 2026-07-01T09:39:50Z
---

# Training-Free and Interpretable Hateful Video Detection via Multi-stage Adversarial Reasoning

## One-line thesis
MARS is a training-free, prompt-only VLM framework that detects hateful videos by having the model adversarially construct the strongest evidence for BOTH a hateful and a non-hateful hypothesis and then synthesize a final decision, yielding interpretable, false-positive-reducing detection competitive with trained SOTA.

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

> Hateful videos pose serious risks by amplifying discrimination, inciting violence, and undermining online safety. Existing training-based hateful video detection methods are constrained by limited training data and lack of interpretability, while directly prompting large vision-language models often struggle to deliver reliable hate detection. To address these challenges, this paper introduces MARS, a training-free Multi-stage Adversarial ReaSoning framework that enables reliable and interpretable hateful content detection. MARS begins with the objective description of video content, establishing a neutral foundation for subsequent analysis. Building on this, it develops evidence-based reasoning that supports potential hateful interpretations, while in parallel incorporating counter-evidence reasoning to capture plausible non-hateful perspectives. Finally, these perspectives are synthesized into a conclusive and explainable decision. Extensive evaluation on two real-world datasets shows that MARS achieves up to 10% improvement under certain backbones and settings compared to other training-free approaches and outperforms state-of-the-art training-based methods on one dataset. In addition, MARS produces human-understandable justifications, thereby supporting compliance oversight and enhancing the transparency of content moderation workflows. The code is available at https://github.com/Multimodal-Intelligence-Lab-MIL/MARS.

