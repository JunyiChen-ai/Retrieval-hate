---
type: idea
node_id: idea:r4-safe-covariate-shift-rank-adaptation
title: "Safe Covariate-Shift Rank Adaptation (SCRA)"
stage: proposed
outcome: pending
added: 2026-08-09T17:52:42Z
based_on: []
target_gaps: []
tags: ["round4", "2026-08-10"]
---

# Safe Covariate-Shift Rank Adaptation (SCRA)

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
Refit the head for the unlabelled target pool by maximising worst-case target-weighted pairwise AUC subject to a formal certificate that its worst-case rank risk cannot exceed the deployed bare head's.

## Key risks
6.1/10. The cleanest available answer to the TTA counter-literature's convergent demand (StatA CVPR 2025, Pitfalls ICML 2023, Illusion of Progress NeurIPS 2025 D&B): prove your adaptation cannot damage the un-adapted model. Weeks of theory; the ambiguity set may make the solution equal the bare head everywhere, and the moderation-specific gains may be too small for a general theory paper.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

