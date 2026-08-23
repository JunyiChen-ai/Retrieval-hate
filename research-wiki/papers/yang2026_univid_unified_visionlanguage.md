---
type: paper
node_id: paper:yang2026_univid_unified_visionlanguage
title: "UNIVID: Unified Vision-Language Model for Video Moderation"
authors: ["Kejuan Yang", "Yizhuo Zhang", "Mingyuan Du", "Yue Zhang", "Dixin Zheng", "Kaili Zhao", "Yang Xiao", "Hanzhong Liang", "Kenan Xiao"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2606.05748"
  doi: null
  s2: null
tags: ["video-moderation", "VLM", "policy-aware-captioning", "interpretability", "industrial", "adjacent", "NOT-core-hateful-video"]
added: 2026-08-07T12:59:56Z
---

# UNIVID: Unified Vision-Language Model for Video Moderation

## One-line thesis
UNIVID: unified VLM for video moderation that emits policy-aware captions as an interpretable intermediate representation instead of black-box classifier scores.

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

> Global-scale video moderation faces a dual challenge: the need for fine-grained multi-modal reasoning and the demand for interpretable outputs to support downstream enforcement. Traditional moderation systems often rely on fragmented black-box classifiers that are difficult to maintain and lack transparency. In this paper, we present UNIVID, a UNIfied VIsion-language model for video moDeration. Unlike standard classification models, UNIVID generates policy-aware captions that serve as an interpretable intermediate representation, enabling human-verifiable decisions and multi-task reusability. While existing open-source and commercial VLMs often suffer from safety-guardrail refusals and lack fine-grained policy alignment, we develop a specialized training data recipe that combines expert human-refined labels with synthetic data to align the model with our safety guidelines. By integrating UNIVID as the core captioner, we design a novel end-to-end video moderation system that reduces violation leakage by 42.7% and overkill rate by 37.0% relatively. Meanwhile, by replacing over 1,000 policy-specific models with a single UNIVID backbone, we recycled extensive computation resources while reducing engineering maintenance overhead. To our knowledge, this is one of the first reports of a high-efficiency captioning VLM successfully supporting industrial-scale moderation and cross-functional business.

