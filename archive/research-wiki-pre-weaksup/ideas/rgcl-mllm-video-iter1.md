---
type: idea
node_id: idea:rgcl-mllm-video-iter1
title: "Multi-granularity annotation-free temporal retrieval + updatable kNN memory for hateful video (MLLM-encoded)"
stage: archived
outcome: mixed
added: 2026-07-01T13:47:28Z
based_on: ["paper:mei2025_robust_adaptation_large", "paper:mei2023_improving_hateful_meme", "paper:lang2025_biting_off_more"]
target_gaps: ["gap:G1", "gap:G2", "gap:G4"]
tags: ["hateful-video", "retrieval-guided-contrastive", "kNN", "multi-granularity", "annotation-free", "temporal-retrieval", "updatable-memory", "cross-dataset", "MLLM-encoder", "frozen", "iteration-1"]
---

# Multi-granularity annotation-free temporal retrieval + updatable kNN memory for hateful video (MLLM-encoded)

**stage:** `archived`  ·  **outcome:** `mixed`

> **ARCHIVED / RESOLVED (2026-07-02):** This umbrella node bundled multiple mechanisms into one `proposed/pending` node. It has been resolved into three split idea nodes carrying honest per-mechanism outcomes:
> - `idea:multigranularity-temporal-retrieval` — **outcome: negative** (segment-level temporal retrieval; noisy MIL pseudo-positives; language sign-flips; no config beats baseline on both langs or crosses acc 0.85). This is the highest-value anti-repeat memory.
> - `idea:cross-dataset-knn-memory` — **outcome: positive** (test-time memory-bank swap; HEADLINE validated novelty vs MoRE; transfers above majority on 5/6 informative cross cells).
> - `idea:lora-mllm-encoder-lever` — **outcome: mixed** (LoRA-SFT of the Qwen2.5-VL encoder; best-ever ZH but regresses EN; crosses acc 0.85 on neither MHClip split).
>
> The umbrella `outcome: mixed` reflects the union of these split results. See the split nodes for verified numbers.

## Thesis
Two mechanistic deltas vs plain RGCL, both annotation-free and general across all 4 datasets: (1) multi-granularity temporal retrieval — a second FAISS index over AUTO sub-clips (uniform windows; no gold spans), mining the within-video benign sub-clip as a drifting hard negative (MIL/dissimilarity) — meme-structurally-impossible; (2) update-stable / cross-dataset kNN memory — test-time exemplar ADD without retrain + memory SWAP for transfer, which MoRE's trained MoE head cannot do. MLLM = frozen encoder LEVER (validated: HateMM 0.817->0.861 M-F1, crosses 0.85 and beats MoRE; MHC-EN 0.711->0.738), not the novelty; no CoT/DPO/LoRA.

## Key risks
Does multi-granularity sub-clip retrieval actually beat whole-video (the make-or-break ablation, esp. the owed MHC-EN gap); auto-segmentation quality without gold spans; MHClip-EN still <0.85 even after the MLLM lever (0.789); drifting-negative miner may be noisy without labels; cross-paper baselines untrusted -> re-run on our split.

**Lesson:** This umbrella node over-bundled three mechanisms; resolved into split nodes with honest outcomes — multi-granularity temporal retrieval = NEGATIVE, cross-dataset kNN memory = POSITIVE headline novelty vs MoRE, LoRA encoder = MIXED lever. Do not re-propose the bundle; cite the split nodes for verified per-mechanism numbers.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

