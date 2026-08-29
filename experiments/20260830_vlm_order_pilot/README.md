# VLM order-distillation pilot (C1) — killed at Stage T, 2026-08-30

Plan and gates: `PILOT_PLAN.md` (frozen pre-run; independent review PASS).
Artifacts: `runs/20260830_vlm_order_pilot/` (teacher_*.jsonl, raw logs,
stage_t_eval.md/json).

## Stage T result (Qwen2.5-VL-7B, 16-s windows, frames+ASR, TEST hate videos)

Within-hate macro ROC (mean pooling, primary): hatemm .5780 (85),
mhclip_en .5137 (44), mhclip_zh .5759 (8), hateclipseg .5502 (67).
Unparseable generations: 0 across all corpora; coverage complete
(85/46/43/69 videos).

**Kill gate fired**: < .60 on both hatemm and mhclip_en. The VLM cannot order
windows inside hate videos better than the weak-MIL control (.578 hatemm), so
order distillation has no signal source. Stage D was not run.

## Salvage value

- These rows double as a LELA-style training-free MLLM baseline under our
  protocol (LELA itself is not reproducible: no code, no split/grid spec —
  see COMPETITOR_PROTOCOLS.md). Keep for the paper's baseline table.
- Combined with the C3 kill and the text-classifier probe
  (../20260830_xneg_mil_pilot/KILL_RECORD.md): pointwise/window-level semantic
  scoring of ANY kind (VLM, appearance kNN, external text classifier) fails to
  order seconds within hate videos, while span-supervised temporal models reach
  .75-.78. The ordering signal is temporal-contextual and requires span-level
  supervision somewhere in the pipeline — motivating the cross-corpus span
  transfer candidate (probe_cross_corpus).
