---
type: idea
node_id: idea:r3-transduction-closure
title: "Cross-channel evasion transduction closure"
stage: piloted
outcome: negative
added: 2026-08-09T08:14:37Z
based_on: []
target_gaps: []
tags: ["round3", "hateful-video"]
---

# Cross-channel evasion transduction closure

**stage:** `piloted`  ·  **outcome:** `negative`

## Thesis
Evasion modelled as a typed transduction graph across overlay text, speech and metadata, trained with worst-path loss plus path closure: two attack sequences reaching the same semantic endpoint must produce the same latent state.

## Key risks
KILLED by frozen pilot R3-3 (2026-08-09), and inverted: P_obs 0.2856 falls BELOW the entire label-permuted null distribution (null range 0.447-0.596, N95 0.5816) on all 5 seeds; A_obs 0.1984 vs 3xN95 1.0919. Clean-margin retention 1.0039 means single-edge augmentation is free and already absorbs length-2/3 compositions. Path-closure would constrain a quantity already at or below noise.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

