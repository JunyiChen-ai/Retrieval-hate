---
type: idea
node_id: idea:lora-mllm-encoder-lever
title: "LoRA-SFT of the Qwen2.5-VL encoder (prediction still via RGCL contrastive + kNN head)"
stage: piloted
outcome: mixed
added: 2026-07-02T05:29:44Z
based_on: ["paper:mei2025_robust_adaptation_large", "paper:mei2023_improving_hateful_meme"]
target_gaps: []
tags: ["hateful-video", "LoRA", "SFT", "MLLM-encoder", "Qwen2.5-VL", "lever", "MIXED", "language-inconsistent", "retrieval-guided-contrastive", "kNN", "iteration-1"]
---

# LoRA-SFT of the Qwen2.5-VL encoder (prediction still via RGCL contrastive + kNN head)

**stage:** `piloted`  ·  **outcome:** `mixed`

LoRA fine-tune the MLLM encoder; MIXED — best-ever ZH (+0.032 F1 vs frozen-CLIP floor) but REGRESSES EN below frozen CLIP/Qwen; crosses acc 0.85 on neither MHClip split.

## Thesis
LoRA-SFT the Qwen2.5-VL encoder while keeping final prediction via RGCL contrastive + kNN head. Encoder is a LEVER, not the novelty. MIXED outcome: language-inconsistent lever.

## Key risks
Val-selected, warmup-floored, adversarially verified (2026-07-02): MHC_zh 0.8023 M-F1 / 0.8322 acc = best-ever ZH (+0.032 F1 / +0.027 acc vs frozen-CLIP floor, clean apples-to-apples), but MHC-EN 0.6916 M-F1 / 0.7516 acc REGRESSES below frozen CLIP (0.783 acc) and frozen Qwen (0.789 acc). Neither MHClip split crosses acc 0.85 (ZH gap 0.018, EN gap 0.098).

**Lesson:** LoRA-SFT of the Qwen2.5-VL encoder is a MIXED performance lever, not novelty — best-ever ZH (0.8322 acc / 0.8023 macroF1, +0.027 acc vs frozen-CLIP floor) but REGRESSES EN below both frozen floors (0.7516 acc / 0.6916 macroF1). Crosses acc 0.85 on neither MHClip split; treat as a lever, not a contribution.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

