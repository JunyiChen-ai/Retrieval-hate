---
type: idea
node_id: idea:multigranularity-temporal-retrieval
title: "Multi-granularity / segment-level temporal retrieval (AUTO sub-clip FAISS + MIL drifting hard-negative)"
stage: archived
outcome: negative
added: 2026-07-02T05:29:27Z
based_on: ["paper:sun2025_multihateloc_towards_temporal", "paper:yang2025_revealing_temporal_label", "paper:mei2023_improving_hateful_meme"]
target_gaps: ["gap:G1", "gap:G4"]
tags: ["hateful-video", "multi-granularity", "temporal-retrieval", "segment-level", "MIL", "annotation-free", "FAISS", "drifting-negative", "NEGATIVE", "anti-repeat", "iteration-1"]
---

# Multi-granularity / segment-level temporal retrieval (AUTO sub-clip FAISS + MIL drifting hard-negative)

**stage:** `archived`  ·  **outcome:** `negative`

Segment-level temporal retrieval over auto sub-clips as a second FAISS index; FAILED — noisy MIL pseudo-positives, language sign-flips, no config beats baseline on both langs.

## Thesis
A second FAISS index over AUTO uniform-window sub-clips (no gold spans), mining within-video benign sub-clips as drifting hard negatives (MIL/dissimilarity), to add a temporal granularity that is meme-structurally impossible. NEGATIVE outcome: no seg_mode beats whole-video baseline on BOTH MHC-EN and MHC_zh, and no config crosses acc 0.85.

## Key risks
Sign-flips by language (full seg: MHC-EN +0.015 F1 / MHC_zh -0.066 F1; milmax rescues ZH but collapses EN; driftneg near-no-op on EN, below-baseline on ZH). Diagnosed as noisy MIL pseudo-positives without gold segment labels. Demoted from headline to honest ablation.

**Lesson:** Segment/multi-granularity retrieval is a tested NEGATIVE — language sign-flips (EN vs ZH) and noisy MIL pseudo-positives (no gold spans) mean no seg_mode beats the whole-video baseline on both MHClip splits or crosses acc 0.85. Demoted to honest ablation; do not re-propose as headline novelty.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

