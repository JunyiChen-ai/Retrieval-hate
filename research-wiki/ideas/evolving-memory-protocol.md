---
type: idea
node_id: idea:evolving-memory-protocol
title: "Evolving-memory protocol on temporal splits (drift tracking via updatable kNN memory)"
stage: piloted
outcome: positive
added: 2026-07-04T00:00:00Z
based_on: ["paper:mei2023_improving_hateful_meme", "paper:li2026_shedding_facades_connecting"]
target_gaps: []
tags: ["hateful-video", "temporal-split", "evolving-memory", "calibration-drift", "recalibration", "validated-as-calibration", "iteration-3", "W4"]
---

# Evolving-memory protocol on temporal splits (drift tracking via updatable kNN memory)

**stage:** `piloted`  ·  **outcome:** `positive` — **validated-as-calibration** (reframed)

DESIGN_iter3 Method B. Original form ("add new-period samples to the memory to track drift")
NOT supported; the protocol instead isolated the true failure mode (calibration drift) and the
correct O(1) fix, which the retrieval architecture uniquely exposes.

## Thesis (final, reframed)
Hate evolves measurably on MHClip-EN within its window (temporal split costs −0.084 macro-F1),
but the dominant component is **score/prior calibration drift, not lost separability**
(temporal ROC 0.8484 > random-split ref 0.7175; only 8.7% of test scores clear t=0.5 vs 24.2%
true pos-rate). The correct k-shot adaptation is **threshold recalibration: k=20 labelled
new-period samples fully recover the drop (0.7336 ≥ random floor 0.7113), zero retrain,
O(1), reversible** — an operating-point knob the retrieval architecture exposes first-class,
while a trained MoE/classifier head hides it inside the weights (adaptation requires
fine-tuning). Memory augmentation (add k to the bank) is flat-to-negative for all k ≤ 80,
all 3 selection strategies, both languages.

## Key risks / scope
- Claim scoped to EN; ZH shows no temporal drop (+0.014) and is the negative control:
  recalibration on tiny k with no drift signal is pure noise (k=5: −0.067) — deploy behind a
  drift monitor.
- Small val pool (17–19 positives); survivor bias compresses measurable drift; temporal-val
  doubles as model-selection val (standard; test untouched).
- Full numbers, jobs (12197/12214/12253), artifacts and leak-discipline asserts:
  `EVAL_temporal_memory_W4.md`; JSON results `logging/temporal_memory/*.json`.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
