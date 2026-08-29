---
type: idea
node_id: idea:r3-prosody-operator-binding
title: "Prosody-as-operator binding"
stage: proposed
outcome: pending
added: 2026-08-09T08:14:37Z
based_on: []
target_gaps: []
tags: ["round3", "hateful-video"]
---

# Prosody-as-operator binding

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
Audio parameterises a constrained low-rank operator applied to transcript states and is structurally forbidden from emitting a hate logit, so prosody matters only through its interaction with what is said.

## Key risks
Jury 4.3/10 on prior-art risk (FiLM, hypernetworks, bilinear fusion). But the estimand argument is strong: Phase 1 measured marginal audio utility, which is compatible with a strong conditional interaction. Cheapest untested idea left (cached CLAP + MPNet, CPU-minutes) and carries a clean shuffled-audio falsification. First thing to run in a round 4.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

