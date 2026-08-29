---
type: idea
node_id: idea:hatemm-annotation-noise-ceiling
title: "HateMM annotation-noise ceiling via a fresh uniform-random census"
stage: archived
outcome: negative
added: 2026-08-07T13:36:11Z
based_on: []
target_gaps: ["gap:G-D"]
tags: ["G-D", "killed-by-review", "label-noise"]
---

# HateMM annotation-noise ceiling via a fresh uniform-random census

**stage:** `archived`  ·  **outcome:** `negative`

KILLED as a standalone: a train-only audit cannot establish a test-set performance ceiling.

## Thesis
Audit 120 uniformly sampled HateMM-train videos blinded, estimate class-asymmetric flip rates, invert the Natarajan noisy-label risk into a macro-F1 ceiling band.

## Key risks
Killed as a standalone submission by the cross-model jury: train-to-test extrapolation undermines the central claim, Natarajan correction assumes a far simpler noise process than subjective value-dependent disagreement, and 120 videos give class-specific CIs too wide for the leaderboard argument. Retained only as supporting evidence for G-D.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

