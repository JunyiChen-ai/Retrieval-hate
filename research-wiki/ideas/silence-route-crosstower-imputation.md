---
type: idea
node_id: idea:silence-route-crosstower-imputation
title: "Deterministic silence route + cross-tower text imputation"
stage: archived
outcome: negative
added: 2026-08-07T13:36:11Z
based_on: []
target_gaps: []
tags: ["killed-by-review", "modality"]
---

# Deterministic silence route + cross-tower text imputation

**stage:** `archived`  ·  **outcome:** `negative`

KILLED at review: a ridge map image->text adds no information the image embedding does not already contain.

## Thesis
Hard-partition videos on ASR emptiness and give the speech-absent route an imputed text vector instead of a degenerate one.

## Key risks
Killed by the cross-model jury: deterministic image-to-text regression cannot recover information absent from the image embedding, so any gain is reparameterisation rather than modality repair; a sufficiently expressive image head learns the same transform. Prior art: SMIL (AAAI 2021), ActionMAE (CVPR 2023).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

