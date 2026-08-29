---
type: idea
node_id: idea:ccef-chance-corrected-evidence-faithfulness
title: "CCEF: chance-corrected evidence faithfulness, and the HateMM span-coverage result"
stage: proposed
outcome: pending
added: 2026-08-07T13:36:11Z
based_on: ["paper:yang2025_revealing_temporal_label", "paper:wang2025_hateclipseg_segmentlevel_annotated", "paper:sun2025_multihateloc_towards_temporal"]
target_gaps: ["gap:G-F"]
tags: ["G-F", "evaluation", "metric", "span-coverage", "recommended-2"]
---

# CCEF: chance-corrected evidence faithfulness, and the HateMM span-coverage result

**stage:** `proposed`  ·  **outcome:** `pending`

Chance-correct localisation against each video's own span coverage, because HateMM gold spans cover a median 82.9% of the video.

## Thesis
Measured on 298 hateful HateMM-train videos, official gold hateful spans cover mean 0.717 / median 0.829 of the runtime (34.6% of hateful videos annotated >=90% hateful), while the blinded coders' minimal sufficient evidence intervals cover median 0.131 - a 2.0x paired gap. So the chance top-1 segment hit rate is 0.762 and uncorrected localisation metrics in this field are near-vacuous. CCEF kappa-normalises the hit rate against the video's own coverage; a faithfulness gap (macro-F1 minus evidence-faithful macro-F1 at tIoU>=0.5) accompanies it. The controlled trim decomposition (full / length-matched random window / gold-span window) measured generic-trim -0.41 pt and oracle-alignment +0.48 pt CI [-0.79,+1.76] - i.e. essentially no headroom, against the +19.34/+30.45 macro-F1 that arXiv:2508.04900 attributes to the same trimming operation with no random-window control.

## Key risks
Metric-only papers are weak standalone submissions - ship as the companion instrument to the headline method and lead with the dataset result, not the definition. The faithfulness-gap half is substantially anticipated by NExT-GQA Acc@GQA (CVPR 2024) and EG-VQA EG-F1; only the per-video chance correction is unoccupied. Must be reproduced on HateClipSeg before claiming it generalises beyond HateMM.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

