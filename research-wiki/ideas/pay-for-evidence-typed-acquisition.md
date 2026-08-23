---
type: idea
node_id: idea:pay-for-evidence-typed-acquisition
title: "Pay-for-Evidence: distilled evidence-type gate driving budgeted discrete modality acquisition"
stage: archived
outcome: negative
added: 2026-08-07T13:36:11Z
based_on: ["paper:yang2025_hfs_holistic_queryaware", "paper:vivierardisson2026_differentiable_knapsack_topk", "paper:cspedessarrias2025_mmhsd_multimodal_hate"]
target_gaps: ["gap:G-C", "gap:G-A"]
tags: ["G-C", "G-A", "OCR", "knapsack", "distillation", "recommended-1", "killed-by-user"]
---

# Pay-for-Evidence: distilled evidence-type gate driving budgeted discrete modality acquisition

**stage:** `archived`  ·  **outcome:** `negative`

Predict which evidence type a video needs, then pay for that modality on only a few segments under a hard cost budget.

**KILLED 2026-08-09 (user ruling).** Direction closed at DESIGN ONLY stage — zero results, zero test-set contact, confirmation set unspent, C6 never produced a cost number. The full OCR cache (`data/OCR/`, 1246 videos) is now built, so OCR is a pre-computed input and the acquisition-cost premise this idea rests on no longer holds. Record: [`EXP_cvoi_acquisition_KILL_2026-08-09.md`](../EXP_cvoi_acquisition_KILL_2026-08-09.md); pre-registration: [`EXP_cvoi_acquisition_prereg.md`](../EXP_cvoi_acquisition_prereg.md). The OCR cache, frozen grouping logic and `scripts/cvoi_acq/` are retained.

## Thesis
The Gate-C census shows our failure population is modality-structured, not time-structured (on_screen_text required 53.4% of FN vs 33.3% of TP, Fisher OR 2.29 p=0.083; 30.1% of FN need on-screen text with no usable speech). A small head on frozen CLIP features predicts the required evidence type at OOF AUROC 0.842 (bootstrap LB 0.773, stable across FN/TP/FP strata: 0.768/0.880/0.898) - so a gate CAN decide where OCR is needed before paying for it. A differentiable knapsack then acquires the expensive modality on <=3 of 30 segments. MLLM supervises the TYPE at training time only; no MLLM at inference.

## Key risks
Predicting THAT on-screen text matters is not reading it; a cheap heuristic OCR-presence detector is a strong baseline that must be ablated head-to-head. Requires a HateMM OCR cache that does not yet exist (1-3 GPU-h). The originally proposed sibling mechanism - hard-partitioning the kNN memory by predicted type - was piloted and FAILED (0/5 folds reached +0.05 neighbour purity) and is excluded.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

