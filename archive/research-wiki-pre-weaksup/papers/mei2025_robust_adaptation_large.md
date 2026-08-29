---
type: paper
node_id: paper:mei2025_robust_adaptation_large
title: "Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection"
authors: ["Jingbiao Mei", "Jinghong Chen", "Guangyu Yang", "Weizhe Lin", "Bill Byrne"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2502.13061"
  doi: null
  s2: null
tags: ["hateful-memes", "retrieval-augmented", "retrieval-guided-contrastive-learning", "kNN", "FAISS", "large-multimodal-model", "Qwen2-VL", "LLaVA", "two-stage-finetuning", "contrastive-learning", "hard-negative-mining", "adversarial-robustness", "cross-domain-generalization", "EMNLP2025", "base-method", "inspiration", "NOT-core-hateful-video"]
added: 2026-07-01T09:39:51Z
---

# Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection

## One-line thesis
A two-stage robust adaptation framework (LMM-RGCL / RA-HMD) fine-tunes a Large Multimodal Model for hateful-meme detection by combining language-modeling supervision with retrieval-guided contrastive learning over the LMM's own embeddings, yielding SOTA in-domain accuracy, better cross-domain generalization, and adversarial robustness. INSPIRATION/BASE-METHOD, meme domain, NOT hateful-video.

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

> Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL

