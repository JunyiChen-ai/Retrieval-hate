---
type: idea
node_id: idea:r4-monotone-disagreement-lattice
title: "Monotone Disagreement Lattice (MDL)"
stage: archived
outcome: negative
added: 2026-08-09T18:06:50Z
based_on: []
target_gaps: []
tags: ["round4", "2026-08-10"]
---

# Monotone Disagreement Lattice (MDL)

**stage:** `archived`  ·  **outcome:** `negative`

## Thesis
Monotone lattice over per-encoder OOF logits, pinned to the validation-best encoder in the concordant region, free to learn non-additive corrections only where encoders disagree.

## Key risks
PILOTED R4-1 2026-08-10 -> KILL, 0 of 4 frozen clauses (MeanDeltaROC -0.0000, paired bootstrap LCB95 -0.00253). Instrument validated on a planted non-additive interaction (0.9922 vs mean-logit 0.9519), so the finding is that the measured cross-encoder complementarity is essentially ADDITIVE: a monotone non-additive surface extracts nothing a plain average has not. Deviation D1: the originally frozen within-hard-label permutation null was a false-KILL generator and was replaced by jury ruling with a paired stratified joint-row bootstrap.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

