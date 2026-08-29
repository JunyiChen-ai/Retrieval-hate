---
type: experiment
node_id: exp:exp-baseline-reproduction
title: "Baseline reproduction (frozen-CLIP + frozen-Qwen RGCL/kNN on 4 datasets)"
idea_id: "idea:cross-dataset-knn-memory"
verdict: yes
confidence: high
date: "2026-07-01"
hardware: ""
duration: ""
provenance: ""
added: 2026-07-02T05:30:38Z
tags: ["hateful-video", "baseline", "reproduction", "frozen-CLIP", "frozen-Qwen", "RGCL", "kNN"]
---

# Baseline reproduction (frozen-CLIP + frozen-Qwen RGCL/kNN on 4 datasets)

**verdict:** `yes`  ·  **confidence:** `high`  ·  tests `idea:cross-dataset-knn-memory`

## Metrics
acc>=0.85 met on HateMM (frozen Qwen 0.870) and ImpliHateVid (~0.90). MHClip EN frozen CLIP acc 0.783 / frozen Qwen 0.789 (EN below 0.85). MHClip ZH near field ceiling on tiny test split. These are the clean apples-to-apples floors used to judge later levers.

## Reasoning
Reproduces RGCL/RA-HMD recipe on the 4 video datasets to establish trustworthy on-our-split floors (cross-paper baselines were untrusted). Verdict=yes: baseline runs completed and give stable reference numbers.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

