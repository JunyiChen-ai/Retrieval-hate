---
type: idea
node_id: idea:r3-response-tensor-distillation
title: "Semantic response-tensor distillation"
stage: archived
outcome: pending
added: 2026-08-09T08:14:37Z
based_on: []
target_gaps: []
tags: ["round3", "hateful-video"]
---

# Semantic response-tensor distillation

**stage:** `archived`  ·  **outcome:** `pending`

## Thesis
Distil a video LMM teacher's finite-difference Jacobian/Hessian over named semantic interventions (target substitution, stance reversal, obfuscation, modality removal) rather than its logits or its explanations.

## Key risks
UNFUNDED, not disproven: the cached 348 counterfactual twins are all label=1 with a single intervention type (toxicity-sanitising rewrite), 132 of which flip, so the factorial lattice does not exist on disk. Revival prerequisite: build and human-verify a lattice with >=2 intervention axes and both classes. Nearest prior art: Jacobian matching 1803.00443, DISCO 2212.10534, 2510.21631.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

