---
type: paper
node_id: paper:chen2026_structured_evidence_selection
title: "Structured Evidence Selection for Weakly Supervised Video Anomaly Detection"
authors: ["Chenglizhao Chen", "Tianxiang Nan", "Wen Li", "Xinyu Liu", "Guisheng Zhang", "Mengke Song", "Xiaomin Yu"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2607.10298"
  doi: null
  s2: null
tags: ["weakly-supervised", "video-anomaly-detection", "evidence-selection", "MIL", "structured-reasoning", "mechanism-inspiration", "NOT-core-hateful-video"]
added: 2026-08-07T12:59:57Z
---

# Structured Evidence Selection for Weakly Supervised Video Anomaly Detection

## One-line thesis
SESAD reformulates weakly-supervised video anomaly detection as structured reasoning over selected clip-level visual evidence, to avoid scene-statistics shortcuts.

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

> Weakly supervised video anomaly detection relies solely on video-level labels for training, making it difficult to accurately localize anomalous events in complex scenes. In real-world videos, anomalous behaviors exhibit large variations in appearance and temporal duration, while scene appearance and action dynamics are often tightly entangled. Consequently, existing models tend to rely on scene-related statistical cues rather than true behavioral deviations, resulting in unstable detection performance. To address this challenge, we propose a Structured Evidence Selection framework (SESAD) that reformulates anomaly detection as a structured reasoning process over clip-level visual evidence. Instead of directly mapping aggregated features to anomaly scores, SESAD reorganizes clip representations into semantically structured candidate evidence and performs context-conditioned selection under scene and action constraints. This mechanism adaptively emphasizes anomaly-relevant semantics while suppressing scene interference, thereby alleviating semantic entanglement under weak supervision. Furthermore, we introduce a lightweight geometric discrimination module that constructs a dual-prototype structure in the embedding space, enabling anomaly decisions through relative geometric relations. Extensive experiments on UBnormal, ShanghaiTech, and UCF-Crime show that SESAD achieves 67.92, 97.99, and 88.46 AUC, respectively, while maintaining high computational efficiency and overall consistently stable anomaly discrimination.

