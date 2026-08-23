---
type: idea
node_id: idea:r3-rank-copula-pooling
title: "Rank-copula multistream pooling"
stage: archived
outcome: pending
added: 2026-08-09T08:14:37Z
based_on: []
target_gaps: []
tags: ["round3", "hateful-video"]
---

# Rank-copula multistream pooling

**stage:** `archived`  ·  **outcome:** `pending`

## Thesis
Within-video soft empirical ranks of visual/transcript/audio segment features feed a differentiable copula tensor recording which marginal quantiles co-occur across synchronised streams.

## Key risks
Jury 4.5/10. Likely an ornate covariance pool versus Set Transformer / bilinear pooling. Scope loss: CLIP subclip caches are K=4+K=30 for HateMM but K=4 only for MHC/MHC_zh, so its sampling-density-stability arm is single-dataset and its 'no dataset exceeds 0.05 drift' clause is unfalsifiable.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

