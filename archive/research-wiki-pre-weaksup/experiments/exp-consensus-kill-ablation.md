---
type: experiment
node_id: exp:exp-consensus-kill-ablation
title: "Consensus kill ablation: seg_mode consensus vs selfscore vs full (PARTIAL — ZH validated, EN hard fail)"
idea_id: "idea:retrieval-consensus-denoising"
verdict: partial
confidence: high
date: "2026-07-03"
hardware: ""
duration: ""
provenance: "slurm/logs/mhc_train_cons_{12176..12181}.out; ckpt group logging/Retrieval/{MHC,MHC_zh}/RAC_video_consensus/"
added: 2026-07-02T21:20:20Z
tags: ["hateful-video", "consensus-denoising", "kill-ablation", "seg_mode", "EM", "MIXED", "language-inconsistent", "iteration-3", "wave-1"]
---

# Consensus kill ablation: seg_mode consensus vs selfscore vs full (PARTIAL — ZH validated, EN hard fail)

**verdict:** `partial`  ·  **confidence:** `high`  ·  tests `idea:retrieval-consensus-denoising`

## Metrics

**Protocol.** Frozen CLIP ViT-L/14-336, identical RGCL head, λ_seg=0.5, K=4 sub-clips,
consensus topk=10 / τ=0.2 / EM=2 rounds / drift kept / conflict=ignore; seed 0, 30 epochs;
selection = warmup-floored (epoch ≥5) val-selected: max `Val_Retrieval acc`, tie-break roc;
report that epoch's Test macro. Floors (λ=0) carried from Phase-3 jobs 12128/12130.

| Dataset | seg_mode | job | selEp | Test M-F1 | acc | Δ vs floor (F1 / acc) |
|---|---|---|---|---|---|---|
| MHC_zh | floor (λ=0) | 12130 | 29 | 0.7706 | 0.8054 | — |
| MHC_zh | full (inherit labels) | 12181 | 21 | 0.7050 | 0.7383 | −0.0656 / −0.0671 |
| MHC_zh | selfscore | 12180 | 29 | 0.7746 | 0.8188 | +0.0040 / +0.0134 |
| MHC_zh | **consensus** | 12179 | 23 | **0.7864** | **0.8188** | **+0.0158 / +0.0134** |
| MHC (EN) | floor (λ=0) | 12128 | 26 | 0.7113 | 0.7826 | — |
| MHC (EN) | **full** (inherit labels) | 12178 | 25 | **0.7262** | **0.7888** | +0.0149 / +0.0062 |
| MHC (EN) | selfscore | 12177 | 24 | 0.6394 | 0.7329 | −0.0719 / −0.0497 |
| MHC (EN) | consensus | 12176 | 20 | 0.5948 | 0.7329 | **−0.1165 / −0.0497** |

- **ZH: consensus wins this ablation** — repairs the Phase-3 full-mode hole (0.7050→0.7864 F1)
  and beats the floor on both F1 and acc. Note: Phase-3's milmax (job 12135, 0.7875/0.8255)
  remains numerically the top ZH CLIP config overall, but milmax destroys EN (−0.102 F1);
  consensus is the best *principled/denoising* ZH config and the one with a mechanism story.
- **EN: consensus is a hard fail** — worst EN config in the ablation, −0.117 F1 vs floor.
- **Gate (both languages same-direction ≥ floor): NOT PASSED.**
- `full` reproduces Phase-3 jobs 12129/12131 exactly (same numbers) — harness verified.
- λ=0 through the consensus code path was verified **bit-for-bit** against baseline before launch.

**Diagnostic (round-1 raw-CLIP consensus roles):**

| Corpus | sub-clips | ignore | neg | pos | drift | conflict | drift among positive-video sub-clips |
|---|---|---|---|---|---|---|---|
| MHC_zh | 2316 | 517 | 1172 | 219 | **300** | 108 | **300/720 = 41.7%** |
| MHC (EN) | 2196 | 594 | 1046 | 236 | **161** | 159 | **161/672 = 24.0%** |

ZH consensus aggressively demotes "toxic positives" (sub-clips of hateful videos judged benign
by neighbours) — 41.7% of positive-video sub-clips — and wins; EN demotes far fewer yet loses.

## Reasoning
Verdict=partial: the consensus mechanism is validated on ZH (fixes the Phase-3 inherited-label
regression AND beats the λ=0 floor) but fails hard on EN, so the pre-registered bilingual
same-direction gate fails. Working hypothesis for the EN failure (W2 attribution, running):
EN MHClip hate is predominantly SPEECH-carried, so visual-only sub-clip keys retrieve
semantically unrelated neighbours and the kNN vote is noise — consensus then demotes the wrong
sub-clips. ZH (Bilibili) hate is more visually/on-screen-text-carried, so visual keys vote
meaningfully. Implementation: `src/utils/consensus.py`, `seg_mode=consensus/selfscore` hooks in
the existing seg pipeline; ckpt group `RAC_video_consensus`.

## Caveats
- Tiny dev sets (MHC 80 / MHC_zh 78): sub-0.01 differences are noise; ZH consensus vs selfscore
  F1 gap (+0.012) is inside that band — the robust ZH claim is "consensus ≥ selfscore > full,
  both denoisers ≥ floor", not a consensus-vs-selfscore ranking.
- ZH consensus acc ties selfscore (0.8188); the F1 edge decides "best".
- Single seed (0), single run per cell — consistent with all prior iters but unreplicated.
- CLIP backbone only; interaction with the LoRA-ZH encoder (best ZH floor 0.8322 acc) untested
  — that stack is the W3 sprint.
- EN failure is confounded: selfscore ALSO fails on EN (−0.072 F1), so part of the EN problem
  may be sub-clip supervision per se on speech-carried hate, not the consensus vote specifically
  — exactly what the W2 three-way attribution must separate.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

