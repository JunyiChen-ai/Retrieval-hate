---
type: idea
node_id: idea:role3-selective-reasoning
title: "Role 3 — confidence-gated selective reasoning (kNN margin gate + MLLM arbitration)"
stage: piloted
outcome: closed
added: 2026-07-05T00:00:00Z
based_on: ["idea:rgcl-mllm-video-iter1", "paper:jing2025_hvguard_utilizing_multimodal", "paper:yang2026_trainingfree_interpretable_hateful"]
target_gaps: []
tags: ["hateful-video", "selective-reasoning", "margin-gate", "MLLM-arbiter", "deferral", "CLOSED", "gate-positive", "arbiter-negative-7B", "wave-final"]
---

# Role 3 — confidence-gated selective reasoning (kNN margin gate + MLLM arbitration)

**stage:** `piloted`  ·  **outcome:** `closed` (7B arbiter line terminated; gate itself is a
positive result; revival conditions quantified and parked in TODO)

## Thesis

Route only low-confidence kNN decisions (vote-margin gate, thresholds chosen on val at
10/20/30% deferral) to a frozen/LoRA Qwen2.5-VL arbiter fed with frames, title/transcript, the
video's own archive and top-5 neighbour evidence cards; replace only deferred verdicts.

## Outcome (2026-07-05, jobs 12279/12288/12305; `EVAL_role3_selective_reasoning.md`)

- **Gate: WORKS.** EN test@30%: 24% of samples capture 42% of kNN errors (slice error rate 33%
  vs 15% outside); deferred slice skews to the Hateful/Offensive boundary. Oracle arbitration
  reaches EN acc 0.857–0.888 — the gate leaves room to cross 0.85.
- **Arbiters: ALL THREE GENERATIONS FAIL the val gate** (v1 generic prompt / v2
  rubric-calibrated / v3 task-LoRA + same JSON contract): every (prompt × rate) candidate is
  below before-acc on val in BOTH languages (EN best 0.7750 < 0.7875; ZH best 0.8590 < 0.8718)
  → val-selected config = "do not arbitrate"; EN stays 0.8075 (memory-clean 0.8199), ZH 0.8523.
- Arbiter quality is monotone v1→v2→v3 (EN deferred-acc 0.462→0.487→0.615) but the 7B ceiling
  sits below the 0.667 break-even line (0.846 needed to cross 0.85). v3 breaks the v1/v2
  0→1-only ratchet (4 good 1→0 flips) — direction right, magnitude insufficient.
- Honest footnotes: ZH v3 has UNSELECTED test-side gains (+0.0135/+0.0202 @10/20%) with val
  negative — small-n flip, reported, not claimed. Frames contribute ≈0 to arbitration
  (text-only identical or better, 2.6× cheaper). Selective calls save 76–90% of MLLM calls vs
  always-on pipelines (cost paragraph material, independent of arbiter quality).

## Anti-repeat / revival conditions

Do not iterate further prompts/LoRA at 7B on this boundary slice. Revive ONLY with an arbiter
that clears EN deferred@30% ≥ 0.667 (break-even) / ≥ 0.846 (crosses 0.85) — i.e. ≥72B or
API-class; "task calibration > prompt engineering" is the transferable lesson. Parked in
MORNING_REPORT §8 TODO #4.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
- tested_by → `EVAL_role3_selective_reasoning.md` (raw: scripts/role3/out/)
- blocked_by → 7B arbiter quality on boundary slices
