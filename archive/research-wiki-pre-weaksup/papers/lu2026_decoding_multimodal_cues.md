---
type: paper
node_id: paper:lu2026_decoding_multimodal_cues
title: "Decoding Multimodal Cues: Unveiling the Implicit Meaning Behind Hateful Videos"
authors: ["Junyu Lu", "Deyi Ji", "Liqun Liu", "Xiaokun Zhang", "Youlin Wu", "Roy Ka-Wei Lee", "Peng Shu", "Huan Yu", "Jie Jiang", "Bo Xu", "Liang Yang", "Hongfei Lin"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2606.11953"
  doi: null
  s2: null
tags: ["hateful-video-detection", "implicit-hate", "explainable", "rationale-generation", "multimodal-CoT", "DPO", "MLLM", "Qwen2.5-VL", "HateMM", "ImpliHateVid", "Ex-HateMM", "Ex-ImpliHateVid", "ASR", "OCR", "SIGIR2026", "dataset-release", "preference-optimization", "is_core_hateful_video"]
added: 2026-07-01T09:39:51Z
---

# Decoding Multimodal Cues: Unveiling the Implicit Meaning Behind Hateful Videos

## One-line thesis
IARE recasts hateful video detection as explainable rationale generation, and proposes a two-stage MLLM framework combining multimodal chain-of-thought information augmentation (SFT) with DPO-based reasoning enhancement to jointly predict labels and produce faithful rationales, reaching SOTA on newly released Ex-HateMM and Ex-ImpliHateVid.

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

> Hateful videos have become prevalent on online platforms, highlighting an urgent need for effective detection. However, existing studies primarily focus on binary classification and fail to provide contextual rationales that reveal the implicit meanings behind these judgments, significantly undermining model explainability. To fill this gap, we aim to achieve explainable hateful video detection, enabling models to provide contextual rationales that integrate relevant evidence and logical reasoning alongside decisions. This approach can comprehensively enhance the understanding of video content and the explainability of the decision-making process. We first introduce two datasets, Ex-HateMM and Ex-ImpliHateVid, for explainable hateful video detection. Each dataset provides fine-grained annotations of multimodal harmful elements, along with contextual rationales. We then propose an Information Augmentation and Reasoning Enhancement (IARE) framework designed for explainable detection. The framework employs an information augmentation phase that leverages the multimodal chain-of-thought to integrate harmful elements, thereby enriching rationale evidence. Additionally, IARE incorporates a reasoning enhancement phase, in which Direct Preference Optimization guides the model toward correct reasoning paths and away from incorrect ones, thereby improving the logical coherence of its justifications. We conduct extensive experiments on the two datasets, comparing multiple baselines with our proposed IARE framework. The results demonstrate that IARE achieves state-of-the-art performance while also generating accurate rationales.

