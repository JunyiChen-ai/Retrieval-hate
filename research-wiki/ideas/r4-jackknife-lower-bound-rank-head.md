---
type: idea
node_id: idea:r4-jackknife-lower-bound-rank-head
title: "Jackknife Lower-Bound Rank Head (JLR)"
stage: archived
outcome: negative
added: 2026-08-09T18:06:50Z
based_on: []
target_gaps: []
tags: ["round4", "2026-08-10"]
---

# Jackknife Lower-Bound Rank Head (JLR)

**stage:** `archived`  ·  **outcome:** `negative`

## Thesis
Replace pointwise BCE with a pairwise objective on the leave-one-block-out lower confidence bound of each hate/non-hate margin, so orderings supported by few training items are discounted.

## Key risks
PILOTED R4-2 2026-08-10 -> KILL, 1 of 4 frozen clauses. Decisive: the identical five-head pairwise ensemble with the sd coefficient set to ZERO beats JLR on test ROC in 4 of 4 cells (-0.0012/-0.0025/-0.0026/-0.0005), so the stability discount is a consistent drag and all the gain comes from the pairwise objective + ensemble. Answers the underlying hypothesis negatively: unstable train-pair ordering is not what limits the frozen-feature head, ordinary model variance is.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

