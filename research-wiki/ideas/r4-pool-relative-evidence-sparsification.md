---
type: idea
node_id: idea:r4-pool-relative-evidence-sparsification
title: "Pool-Relative Evidence Sparsification (PRES)"
stage: archived
outcome: pending
added: 2026-08-09T17:52:42Z
based_on: []
target_gaps: []
tags: ["round4", "2026-08-10"]
---

# Pool-Relative Evidence Sparsification (PRES)

**stage:** `archived`  ·  **outcome:** `pending`

## Thesis
Use the unlabelled deployment pool only to estimate a background distribution over the 30 cached OCR windows, then pool each video from the few windows with the highest conditional surprisal relative to that background.

## Key risks
REMOVED by the objective feasibility gate: OCR window vectors exist for HateMM ONLY; HateClipSeg has windows but no train/test split at all and a 395/395 constant text channel; MHC/MHC_zh/ImpliHateVid have no OCR cache. Its decisive test-background-vs-train-background comparison is structurally single-dataset, on the contaminated dataset. Jury: a diagnostic wearing a method's clothes (6.6 -> 2.0).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

