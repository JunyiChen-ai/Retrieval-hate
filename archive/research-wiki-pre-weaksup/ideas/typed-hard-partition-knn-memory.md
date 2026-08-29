---
type: idea
node_id: idea:typed-hard-partition-knn-memory
title: "Typed hard partition of the kNN retrieval memory"
stage: archived
outcome: negative
added: 2026-08-07T13:36:11Z
based_on: []
target_gaps: ["gap:G-C", "gap:G-B"]
tags: ["G-C", "retrieval", "killed-by-pilot"]
---

# Typed hard partition of the kNN retrieval memory

**stage:** `archived`  ·  **outcome:** `negative`

KILLED BY PILOT: restricting retrieval to same-predicted-evidence-type entries did not improve neighbour purity.

## Thesis
Use the predicted evidence type as a hard partition key so cross-type retrieval is structurally impossible, removing the topic confound from the cross-dataset kNN memory.

## Key risks
KILLED by pilot P3(b) (2026-08-08): typed retrieval improved neighbour label purity by >=0.05 in 0 of 5 folds (per-fold deltas +0.018, +0.011, +0.010 and two smaller). The evidence TYPE itself is strongly learnable (OOF AUROC 0.842) and survives as an acquisition gate - see idea:pay-for-evidence-typed-acquisition - but routing the memory by it is dead.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

