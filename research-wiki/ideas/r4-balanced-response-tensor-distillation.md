---
type: idea
node_id: idea:r4-balanced-response-tensor-distillation
title: "Balanced Semantic Response-Tensor Distillation (B-SRTD, C4 revived)"
stage: proposed
outcome: pending
added: 2026-08-09T17:52:42Z
based_on: []
target_gaps: []
tags: ["round4", "2026-08-10"]
---

# Balanced Semantic Response-Tensor Distillation (B-SRTD, C4 revived)

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
Distil a teacher's finite-difference Jacobian and mixed partial over NAMED semantic interventions (target substitution, endorsement/condemnation reversal) rather than its logits or explanations, into the bare head on frozen features.

## Key risks
Highest-scoring candidate of rounds 3 and 4 (7.0/10) and the only one never killed by a mechanism failure, an occupant or a null. Blocker is purely an asset build: data/Counterfactual/*/train_twins.jsonl is 348 records, ALL label=1, one intervention axis. Needs a balanced two-axis lattice (>=200 train + 80 val) plus human verification. Prior art bounds the claim to the named-intervention response tensor (1803.00443 Jacobian matching, DISCO 2212.10534).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

