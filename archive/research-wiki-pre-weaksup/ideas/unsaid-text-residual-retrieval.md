---
type: idea
node_id: idea:unsaid-text-residual-retrieval
title: "Unsaid-text retrieval: an OCR-minus-ASR residual as the retrieval key"
stage: proposed
outcome: pending
added: 2026-08-07T13:36:11Z
based_on: ["paper:lang2025_biting_off_more", "paper:cspedessarrias2025_mmhsd_multimodal_hate", "paper:chen2026_now_you_see"]
target_gaps: ["gap:G-A", "gap:G-B"]
tags: ["G-A", "G-B", "retrieval", "OCR", "recommended-3"]
---

# Unsaid-text retrieval: an OCR-minus-ASR residual as the retrieval key

**stage:** `proposed`  ·  **outcome:** `pending`

Key retrieval on the on-screen claim that was never spoken, instead of on the whole video.

## Thesis
MM-HSD reaches 0.874 with OCR and no retrieval; MoRE reaches 0.8235 with retrieval and no OCR; nobody has asked whether they recover the same videos. Whole-video keys are dominated by the benign remainder and by topic, so an OCR-carried video retrieves speech-carried rants. Hard-select the segment with the largest OCR-minus-ASR semantic residual and use that residual as the key. Gated first by a CPU-minutes OCR-free redundancy test: neighbour label purity conditioned on on_screen_text-required, and the flip rate on the 22 OCR-required-no-speech census false negatives.

## Key risks
Once OCR exists the residual may be dominated by OCR/ASR errors rather than genuinely unsaid meaning - a noise-floor control on OCR/ASR-agreeing videos is mandatory. Not independently novelty-searched to the depth of the top two. Shares the unbuilt OCR cache with the headline idea.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

