---
type: idea
node_id: idea:r4-executable-accountability-path-distillation
title: "Executable Accountability-Path Distillation (EAPD, C6 revived)"
stage: proposed
outcome: pending
added: 2026-08-09T17:52:42Z
based_on: []
target_gaps: []
tags: ["round4", "2026-08-10"]
---

# Executable Accountability-Path Distillation (EAPD, C6 revived)

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
Distil a typed agency graph into separately supervised proposition/quotation/endorsement/condemnation/accountable-speaker edges, and count hate only when an attack path reaches an accountable endorsing agent.

## Key risks
Revived from 3.0 unscored to 6.4/10 because round 4 measured a matching failure mode: ImpliHateVid's binary error budget is ~2/3 false positives on hate-adjacent non-hate, which is exactly use-vs-mention, reporting, counterspeech and quoted hate. Blocked on a 330-video Claude annotation build; standing risk is explanation distillation in disguise, and C1 adjacency.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

