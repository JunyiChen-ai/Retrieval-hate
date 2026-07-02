---
type: experiment
node_id: exp:exp-cross-dataset-transfer
title: "Cross-dataset kNN memory-bank swap transfer (POSITIVE)"
idea_id: "idea:cross-dataset-knn-memory"
verdict: yes
confidence: high
date: "2026-07-01"
hardware: ""
duration: ""
provenance: "Phase-3b, jobs 12136/12137, src/eval_cross_dataset.py"
added: 2026-07-02T05:30:53Z
tags: ["hateful-video", "cross-dataset", "kNN", "memory-bank-swap", "transfer", "POSITIVE", "validated", "vs-MoRE", "headline"]
---

# Cross-dataset kNN memory-bank swap transfer (POSITIVE)

**verdict:** `yes`  ·  **confidence:** `high`  ·  tests `idea:cross-dataset-knn-memory`

## Metrics
Transfers above majority on 5/6 informative cross cells. Lags in-domain by ~0.04-0.09 macro-F1. Test-time memory SWAP with zero retrain — a capability MoRE's trained MoE head structurally lacks.

## Reasoning
Verdict=yes: validates the headline novelty vs the closest SOTA (MoRE). Above-majority transfer on 5/6 informative cross cells demonstrates updatable cross-dataset kNN memory works without retraining.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

