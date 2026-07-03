---
type: idea
node_id: idea:retrieval-consensus-denoising
title: "Retrieval-consensus segment denoising (memory washes its own sub-clip labels)"
stage: piloted
outcome: mixed
added: 2026-07-02T21:19:52Z
based_on: ["paper:mei2023_improving_hateful_meme", "paper:yang2025_revealing_temporal_label", "paper:wang2025_hateclipseg_segmentlevel_annotated"]
target_gaps: []
tags: ["hateful-video", "segment-denoising", "retrieval-consensus", "EM", "pseudo-label", "span-free", "MIXED", "language-inconsistent", "iteration-3"]
---

# Retrieval-consensus segment denoising (memory washes its own sub-clip labels)

**stage:** `piloted`  ·  **outcome:** `mixed`

DESIGN_iter3 Method A: sub-clip pseudo-label = agreement(self video label x kNN-neighbor video-label vote), EM rounds, drift demotion; status partial — ZH validated / EN failed, attribution running

## Thesis
Inherited video-level labels are noisy at sub-clip granularity (a hateful video contains benign sub-clips). Let the retrieval memory itself denoise: each sub-clip's pseudo-label = agreement between its own video label and a kNN vote over neighbouring sub-clips' video labels (topk=10, tau=0.2); confident sub-clips train the contrastive embedding, demoted 'drift' sub-clips of positive videos become mined hard negatives; 2 EM rounds re-derive roles in the learned fused space. Span-free (no gold segments). Kill ablation = consensus vs selfscore (MIST/C2FPL-style self-scoring) vs full (inherit labels, Phase-3 repro), gate = both languages >= lambda=0 floor.

## Key risks
STATUS partial (2026-07-03, jobs 12176-12181): ZH VALIDATED — consensus 0.7864 M-F1 / 0.8188 acc wins the kill ablation, repairs the Phase-3 full-mode hole (0.7050/0.7383) and beats the floor (0.7706/0.8054); Phase-3 milmax (0.7875/0.8255) stays numerically top ZH-CLIP overall but destroys EN, so consensus is the best principled/denoising ZH config. EN FAILED HARD — consensus 0.5948/0.7329 vs floor 0.7113/0.7826; gate (both languages same-direction >= floor) NOT passed. Attribution RUNNING (night W2): hypothesis = EN hate is speech-carried, so visual sub-clip keys make the kNN vote noisy; consensus demoted 300/720 ZH vs 161/672 EN positive-video sub-clips. Risk: if EN attribution fails, claim must be scoped to ZH/visual-carried hate.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

