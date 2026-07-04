---
type: idea
node_id: idea:archive-as-retrieval-key
title: "MLLM structured archive as retrieval key (kNN-key augmentation / third stream)"
stage: archived
outcome: negative
added: 2026-07-04T00:00:00Z
based_on: ["paper:mei2023_improving_hateful_meme", "paper:mei2025_robust_adaptation_large"]
target_gaps: []
tags: ["hateful-video", "MLLM-archive", "retrieval-key", "knn-memory", "NEGATIVE", "refuted", "anti-repeat", "iteration-3"]
---

# MLLM structured archive as retrieval key (kNN-key augmentation / third stream)

**stage:** `archived`  ·  **outcome:** `negative` (REFUTED as an accuracy contribution)

DESIGN_iter3 Role-1 downstream claim: augmenting the kNN retrieval keys with CLIP-text-encoded
MLLM structured archives (alpha-weighted concat) improves detection. REFUTED by multi-seed
paired analysis; archives remain valuable as the auditability/editability substrate (pillar 4),
not as a performance lever.

## Thesis
Structured harmfulness archives (target_groups / mechanism / modality_cues / explicitness /
neutral_summary; English pivot) encoded into the retrieval-key space give the kNN memory
"eyes" on speech/on-screen-text semantics, improving accuracy over visual/fused keys alone.

## Why refuted
- Seed-0 wins (ZH 0.8523 vs floor 0.8322; EN 0.8075 vs 0.7888) were selection luck: 5-seed
  paired dAcc = **−0.0014 ± 0.0313**; final-epoch checkpoints byte-identical (sha1) and the
  alpha=0.25 key flips ZERO test votes on all 5 ZH seeds (`experiments/exp-archive-knn-seeds.md`).
- Three-arm ablation kills the "structured distillation beats transcript" ordering too:
  arc−trs ΔF1 +0.0001±0.0388 (ZH); the truncation-repair alternative also fails on ZH
  (`ABLATION_transcript_vs_archive.md`).
- EN alpha grid 0.15–0.35 and archive_mode=both: all noise-level or harmful (jobs 12247–12251).
- W5 consensus-space transplant (voting in archive/blend space) worse in BOTH languages
  (jobs 12243–12246) — the archive key space rescues nothing.
- Surviving residue (analysis-only): ZH val-selected dROC +0.009 on 4/5 seeds — a weak,
  selection-dependent ranking signal that does NOT survive the final-epoch protocol.

**Lesson (anti-repeat):** do not re-propose archive-derived keys as an accuracy lever on these
datasets/scales; the archive's validated value is auditability + targeted memory editing
(`AUDIT_archive_faithfulness.md`, `DEMO_memory_editing.md`), with prompt-v2 fixing target
recall (1.6%→49.4% on harmful ZH train) for editability, not detection.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
