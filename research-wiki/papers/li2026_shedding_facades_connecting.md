---
type: paper
node_id: paper:li2026_shedding_facades_connecting
title: "Shedding the Facades, Connecting the Domains: Detecting Shifting Multimodal Hate Video with Test-Time Adaptation"
authors: ["Jiao Li", "Jian Lang", "Xikai Tang", "Wenzheng Shu", "Ting Zhong", "Qiang Gao", "Yong Wang", "Leiting Chen", "Fan Zhou"]
year: 2026
venue: "arXiv"
external_ids:
  arxiv: "2602.00132"
  doi: null
  s2: null
tags: ["hateful-video-detection", "test-time-adaptation", "cross-domain", "crosslingual", "chinese", "multimodal", "prototype-alignment", "centroid-clustering", "source-free", "implicit-hate", "HateMM", "MultiHateClip", "AAAI2026", "entropy-minimization", "semantic-drift", "is_core_hateful_video"]
added: 2026-07-01T09:39:49Z
---

# Shedding the Facades, Connecting the Domains: Detecting Shifting Multimodal Hate Video with Test-Time Adaptation

## One-line thesis
SCANNER is the first test-time adaptation framework for hate video detection, aligning unlabeled target-domain videos to clustered invariant hateful cores (demographic/target categories that stay constant even as surface manifestations evolve) to bridge severe cross-domain semantic drift under a source-free, target-label-free setting.

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

> Hate Video Detection (HVD) is crucial for online ecosystems. Existing methods assume identical distributions between training (source) and inference (target) data. However, hateful content often evolves into irregular and ambiguous forms to evade censorship, resulting in substantial semantic drift and rendering previously trained models ineffective. Test-Time Adaptation (TTA) offers a solution by adapting models during inference to narrow the cross-domain gap, while conventional TTA methods target mild distribution shifts and struggle with the severe semantic drift in HVD. To tackle these challenges, we propose SCANNER, the first TTA framework tailored for HVD. Motivated by the insight that, despite the evolving nature of hateful manifestations, their underlying cores remain largely invariant (i.e., targeting is still based on characteristics like gender, race, etc), we leverage these stable cores as a bridge to connect the source and target domains. Specifically, SCANNER initially reveals the stable cores from the ambiguous layout in evolving hateful content via a principled centroid-guided alignment mechanism. To alleviate the impact of outlier-like samples that are weakly correlated with centroids during the alignment process, SCANNER enhances the prior by incorporating a sample-level adaptive centroid alignment strategy, promoting more stable adaptation. Furthermore, to mitigate semantic collapse from overly uniform outputs within clusters, SCANNER introduces an intra-cluster diversity regularization that encourages the cluster-wise semantic richness. Experiments show that SCANNER outperforms all baselines, with an average gain of 4.69% in Macro-F1 over the best.

