---
type: idea
node_id: idea:r4-policy-cone-discriminant-head
title: "Policy-Cone Discriminant Head (PCD)"
stage: proposed
outcome: pending
added: 2026-08-09T17:52:42Z
based_on: []
target_gaps: []
tags: ["round4", "2026-08-10"]
---

# Policy-Cone Discriminant Head (PCD)

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
Represent the binary policy not by one class-name prompt but by a convex cone of paired violation/safe-use clause directions, and constrain a covariance-whitened frozen-feature head to lie in that cone.

## Key risks
Best reserve of round 4 (4.8/10). Cannot be frozen until the cone is mathematically attached to the deployed nonlinear Hadamard-fusion head (policy directions live in raw text space; the classifier decides after learned projections and an MLP), and an unconstrained visual residual can bypass the cone. Next action is a paper-and-pencil spec plus a novelty check against LP++/CLAP prompt-anchor work, NOT a pilot.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

