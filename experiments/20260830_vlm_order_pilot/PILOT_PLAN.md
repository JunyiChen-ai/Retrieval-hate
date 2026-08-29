# Pilot: dense VLM window scoring with order-only distillation (iteration step 8-9)

Date: 2026-08-30. Candidate C1 from `../20260830_powa_within_diagnosis/NOVELTY_SCOUT.md`
(verdict: open-with-differentiation). Core mechanism this round — nothing else changes.

## Mechanism hypothesis (frozen before results)

- Diagnosis established an objective gap: frozen CLIP+VGGish+BERT features reach
  within-ROC .7495 (HateMM) / .7692 (MHC-EN) under full supervision, while the best
  weak method extracts only .6315 / .6004. Video-level MIL never supervises the
  ordering of seconds inside a positive video.
- A generative VLM (Qwen2.5-VL-7B, local) reading 16-s windows (frames + ASR text)
  can judge per-window hate evidence. Its absolute scores may be miscalibrated, but
  its **within-video ordering** of windows is exactly the missing supervision.
- Method: score every window of every train video with the VLM once (offline);
  train the diagnosis TemporalConv head with (a) video-label top-k MIL plus (b) a
  pairwise within-video ranking loss on window pairs whose teacher scores differ
  by a margin. VLM absent at inference.

## Pilot stages and kill gates (frozen)

- **Stage T (teacher signal check, runs first).** Score test-split hate videos'
  windows with the VLM; broadcast window scores to seconds; evaluate within-hate
  macro ROC with the standard evaluator.
  - Kill: teacher within-ROC < .60 on BOTH HateMM and MHC-EN → the teacher has no
    ordering signal worth distilling; candidate dies, return to step 5.
  - Note: this arm doubles as a LELA-style training-free baseline row.
- **Stage D (distillation), only if Stage T passes.** Train head on train-split
  windows' teacher ordering + MIL; 3 seeds; test eval.
  - Success: within-ROC macro beats both the weak-MIL control (same arch/features)
    and the best reproduced baseline on ≥2 of {HateMM, MHC-EN, MHC-ZH}; high-pos
    stratum (pos_frac>0.6) must not degrade vs control.
  - Mechanism attribution: shuffle-teacher control (same loss, teacher scores
    permuted within video) must NOT reproduce the gain.

## Fixed choices

- Windows: 16 s, stride 8 s; 4 frames per window (uniform), ASR text of the window
  appended (empty string when no ASR). One VLM call per window, integer score 0-10
  ("hate evidence in THIS window only"), deterministic decoding (temperature 0),
  raw generations logged; unparseable → score 0 and counted.
- Teacher: local Qwen2.5-VL-7B-Instruct (already cached). No API calls.
- Metric priority: within-video macro ROC/AP (test) > pooled frame AP/ROC > video.
- Window->second pooling for the kill gate: **mean of covering windows** (max is a
  secondary column only). Frozen pre-run; independent review 2026-08-30 PASS with
  two pre-gate checks: unparse rate from the raw log must be low, and the eval's
  n-videos column must equal the corpus hate-test count.
- Evaluator: `scripts/reproduction_baselines/eval_baseline_scores.py` only.
- Test usage: test is used for evaluation at every stage (user ruling 2026-08-30);
  test labels never enter training or the teacher prompt.

## Deliverables

- `runs/20260830_vlm_order_pilot/teacher_scores_<corpus>.jsonl` (+ raw log)
- `runs/20260830_vlm_order_pilot/stage_t_eval.md`
- Stage D: `distill_<corpus>_seed*.jsonl`, `stage_d_eval.md`, shuffle-control row.
