---
type: experiment
node_id: exp:exp-seg-mode-ablation
title: "Segment-mode / multi-granularity temporal retrieval ablation (NEGATIVE)"
idea_id: "idea:multigranularity-temporal-retrieval"
verdict: no
confidence: high
date: "2026-07-01"
hardware: ""
duration: ""
provenance: ""
added: 2026-07-02T05:30:38Z
tags: ["hateful-video", "ablation", "segment-level", "multi-granularity", "MIL", "NEGATIVE", "anti-repeat"]
---

# Segment-mode / multi-granularity temporal retrieval ablation (NEGATIVE)

**verdict:** `no`  ·  **confidence:** `high`  ·  tests `idea:multigranularity-temporal-retrieval`

## Metrics
Sign-flips by language. full seg: MHC-EN +0.015 F1 / MHC_zh -0.066 F1. milmax rescues ZH but collapses EN. driftneg near-no-op on EN, below-baseline on ZH. No seg_mode beats whole-video baseline on BOTH languages; no config crosses acc 0.85.

## Reasoning
Verdict=no: diagnosed as noisy MIL pseudo-positives without gold segment labels. Demoted from headline to honest ablation. Highest anti-repeat value — do not re-attempt segment-level temporal retrieval on these datasets without gold spans.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

