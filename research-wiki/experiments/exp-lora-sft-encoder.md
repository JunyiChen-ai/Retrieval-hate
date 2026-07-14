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

## Cross-reference changelog (does NOT alter the `partial` verdict or any metric above)

- **2026-07-14 — this node's EN arm is the banking evidence for B4's pre-GPU closure.** The
  MHC-EN regression recorded here (0.6916 M-F1 / 0.7516 acc, below both frozen floors, §Metrics)
  is exactly the seed0 anchor that let the EN LoRA-encoder cell be closed **pre-GPU** as the
  campaign's 22nd negative — no new GPU spent. B4 re-read the primary log and re-derived the
  paired seed0 deltas vs the frozen-CLIP enc3s control (12850): val-selected **−0.0310 acc /
  −0.0197 F1** (regress), final-epoch **+0.0062 acc / +0.0157 F1** (≪ +0.030 bar); honest prior =
  FAIL both protocols (<5% falsification). Pointer: `refine-logs/B4_FORENSIC_RECON.md`
  (§(i) has-it-been-measured, §(v) honest prior). Also surfaced in
  `research-wiki/PAPER_MASTER_TABLES.md` PUR-2 and `TERMINUS_round2_mllm_plus3.md` §7.
- **2026-07-14 — this node's ZH-relative context updated by B3's paired measurement.** The
  ZH "best-ever" (0.8023 M-F1 / 0.8322 acc, §Metrics) was measured here vs the frozen-CLIP floor
  under a single-configuration val-selected lens. **B3** now supplies the clean same-code
  same-seed 3-head-seed paired verdict for the LoRA-encoder-vs-frozen-CLIP ZH comparison
  (`final-epoch: PASS (MARGINAL); val-selected: FAIL`; final-ep mean Δacc +0.0313 / ΔmF1 +0.0453),
  and the decomposition attributing the entire ZH gain to LoRA task/language adaptation
  (frozen-Qwen ZH −0.0112 vs LoRA-Qwen +0.0313) — consistent with this node's "helps ZH, hurts
  EN, language-inconsistent" reasoning. Pointers: `refine-logs/B3_VERDICT_REVIEW.md`,
  `research-wiki/experiments/exp-lora-zh-b3.md`. Novelty status of the LoRA lever = pending user
  ruling (unchanged from this node's `partial`/"not a headline" stance).

