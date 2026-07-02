---
type: idea
node_id: idea:cross-dataset-knn-memory
title: "Updatable cross-dataset kNN memory (test-time memory-bank swap)"
stage: piloted
outcome: positive
added: 2026-07-02T05:29:38Z
based_on: ["paper:mei2023_improving_hateful_meme", "paper:mei2025_robust_adaptation_large", "paper:lang2025_biting_off_more"]
target_gaps: ["gap:G2"]
tags: ["hateful-video", "kNN", "updatable-memory", "cross-dataset", "memory-bank-swap", "retrieval-guided-contrastive", "POSITIVE", "validated", "headline", "vs-MoRE", "iteration-1"]
---

# Updatable cross-dataset kNN memory (test-time memory-bank swap)

**stage:** `piloted`  ·  **outcome:** `positive`

Test-time exemplar ADD without retrain + memory-bank SWAP for cross-dataset transfer; VALIDATED headline novelty vs MoRE (whose trained MoE head structurally cannot swap memory).

## Thesis
Update-stable / cross-dataset kNN memory: at test time, ADD exemplars without retraining and SWAP the memory bank to transfer across datasets. This is a capability MoRE's trained MoE head structurally lacks. HEADLINE validated novelty vs the closest SOTA (MoRE).

## Key risks
Cross-paper baselines untrusted -> re-run on our split (done). Lags in-domain by ~0.04-0.09 macro-F1; net POSITIVE because it transfers above majority on 5/6 informative cross cells with zero retrain.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

