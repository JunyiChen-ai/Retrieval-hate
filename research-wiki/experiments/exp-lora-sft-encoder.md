---
type: experiment
node_id: exp:exp-lora-sft-encoder
title: "LoRA-SFT of Qwen2.5-VL encoder, prediction via RGCL contrastive + kNN head (MIXED)"
idea_id: "idea:lora-mllm-encoder-lever"
verdict: partial
confidence: high
date: "2026-07-02"
hardware: ""
duration: ""
provenance: ""
added: 2026-07-02T05:30:54Z
tags: ["hateful-video", "LoRA", "SFT", "Qwen2.5-VL", "encoder-lever", "MIXED", "language-inconsistent"]
---

# LoRA-SFT of Qwen2.5-VL encoder, prediction via RGCL contrastive + kNN head (MIXED)

**verdict:** `partial`  ·  **confidence:** `high`  ·  tests `idea:lora-mllm-encoder-lever`

## Metrics
Val-selected, warmup-floored, adversarially verified. MHC_zh 0.8023 M-F1 / 0.8322 acc = best-ever ZH (+0.032 F1 / +0.027 acc vs frozen-CLIP floor). MHC-EN 0.6916 M-F1 / 0.7516 acc = REGRESSES below frozen CLIP (0.783 acc) and frozen Qwen (0.789 acc). Neither MHClip split crosses acc 0.85 (ZH gap 0.018, EN gap 0.098).

## Reasoning
Verdict=partial (mixed): language-inconsistent lever — helps ZH (best-ever), hurts EN (regresses below both frozen floors), and crosses acc 0.85 on neither MHClip split. Kept as an honest lever result, not a headline.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

