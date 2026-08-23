---
type: idea
node_id: idea:r3-mobius-interaction-distillation
title: "Mobius interaction distillation"
stage: archived
outcome: negative
added: 2026-08-09T08:14:37Z
based_on: []
target_gaps: []
tags: ["round3", "hateful-video"]
---

# Mobius interaction distillation

**stage:** `archived`  ·  **outcome:** `negative`

## Thesis
Mobius-invert teacher scores over modality coalitions into an interaction spectrum; the student predicts the coefficients and must reconstruct the teacher score from their sum.

## Key risks
Self-killed on its own frozen precondition: no coalition sweep exists in data/MLLM_scores (only per-segment scores under different prompts and model sizes). Shapley/Mobius explanation distillation is also a likely occupant.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

