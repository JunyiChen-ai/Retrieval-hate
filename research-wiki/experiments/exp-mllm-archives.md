---
type: experiment
node_id: exp:exp-mllm-archives
title: "MLLM structured harmfulness archives for MHClip EN+ZH (Qwen2.5-VL-7B, English pivot) (INFRA)"
idea_id: ""
verdict: yes
confidence: high
date: "2026-07-03"
hardware: ""
duration: ""
provenance: "slurm/logs/gen_archive_{12172,12173,12174,12184,12186}.out; data/Archive/{MHC,MHC_zh}/; data/CLIP_Embedding/{MHC,MHC_zh}/*_archive_*.pt"
added: 2026-07-02T21:20:20Z
tags: ["hateful-video", "MLLM-archive", "retrieval-key", "Qwen2.5-VL", "english-pivot", "infrastructure", "iteration-3", "wave-1"]
---

# MLLM structured harmfulness archives for MHClip EN+ZH (Qwen2.5-VL-7B, English pivot) (INFRA)

**verdict:** `yes`  ·  **confidence:** `high`

## Metrics

**Generator.** Frozen Qwen2.5-VL-7B-Instruct, greedy, 8-frame video + title/transcript (capped
6000 chars) → one structured JSON per video: `target_groups / mechanism /
modality_cues{visual,speech,on_screen_text} / explicitness / neutral_summary`. **English pivot
enforced by prompt** ("Respond in English only") — ZH videos archived in English.
Script: `src/utils/generate_video_archive_HF.py` (+ `scripts/slurm/gen_archive.sbatch`);
resume-by-id, append-mode, brace-repair parser, decord→PyAV frame fallback.

**Final per-id quality (deduped, retries folded in):**

| Corpus | split | N | parse_ok | schema_ok | refusal |
|---|---|---|---|---|---|
| MHC (EN) | train | 549 | 547 | 540 | 0 |
| MHC (EN) | val | 80 | 80 | 80 | 0 |
| MHC (EN) | test | 161 | 161 | 160 | 0 |
| MHC_zh | train | 579 | 575 | 571 | 0 |
| MHC_zh | val | 78 | 78 | 77 | 0 |
| MHC_zh | test | 149 | 149 | 145 | 0 |

Totals: parse_ok EN **788/790 (99.7%)**, ZH **802/806 (99.5%)**; schema_ok 780/790 / 793/806;
**zero refusals** across all 1596 videos.

**SLURM jobs.**

| Job | Role |
|---|---|
| 12172 | smoke (5+5 items EN/ZH, end-to-end incl. CLIP-text encode) |
| 12173 | MHC (EN) full generation + encode + B2 push |
| 12174 | MHC_zh full generation + encode + B2 push |
| 12184 | re-encode all 6 caches + B2 push (fill/refresh pass) |
| 12186 | RETRY=1 fill of parse-failed ids (EN 2 / ZH 4 retried) + re-encode + push |

**Products.**
- Archives: `data/Archive/{MHC,MHC_zh}/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_archive.jsonl`
- CLIP text-encoded keys: `data/CLIP_Embedding/{MHC,MHC_zh}/{train,dev_seen,test_seen}_archive_openai_clip-vit-large-patch14-336_HF.pt`
  (N = 549/80/161 and 579/78/149, Dt=768, zero-vector=0, missing-archive=0)
- Mirrored to `b2:junyi-data/RGCL_video/embeddings/{MHC,MHC_zh}/`.

## Reasoning
Verdict=yes: infrastructure goal met — DESIGN_iter3 Role-1 structured archives exist for every
MHClip EN/ZH sample with near-perfect parse rate and no refusals, and are already rendered into
CLIP-text embedding caches shaped exactly like the existing feature caches, so the W3
archive-as-retrieval-key wiring is a loader/fusion change only. The 6 residual parse-fails
(EN 2 / ZH 4; retried once, still unparseable JSON) are encoded from their `raw_output` text
fallback — no zero vectors in any cache.

## Caveats
- Archive text is CLIP-text-encoded, which truncates at 77 tokens — long `neutral_summary`
  fields are clipped (encoder warns "240 > 77"); if W3 shows signal, a longer-context text
  encoder is an obvious upgrade lever.
- schema_ok < parse_ok (~1–3%): those entries parsed after brace repair but missed strict
  schema; fields may be partially empty.
- Zero refusals is a *generation* property; no human audit yet of archive faithfulness
  (W2 includes spot-checking archives of consensus-demoted sub-clips).
- English pivot on ZH content risks losing ZH-specific coded language — acceptable for
  retrieval keys (shared space), but must be stated in the paper.
- Downstream utility completely untested — this node records infrastructure, not a
  performance claim (that is W3).

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

