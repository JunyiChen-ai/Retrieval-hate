# REJECTED — POWA negative-bag-certified benign insertion pilot

Rejected 2026-08-31: Stage P failed the frozen all-metric SOTA gate. HateMM
within-video ROC improved from `.5911` to `.6382` and crossed its `.6315` bar,
but pooled AP/ROC fell by `.0529/.0543`; HateClipSeg improved only
`.0059/.0041/.0061` and remained below all three VERA bars. Per the frozen
plan, no controls or weight tuning were run. Authoritative verdict:
`runs/20260831_powa_benign_insertion_pilot/stage_p_summary.json`.

Date: 2026-08-31. The independent pre-run review passed before training.

Mechanism hypothesis, frozen settings, strict SOTA gates, controls, and kill
rules are in `PILOT_PLAN.md`. Independent prior-art analysis is in
`NOVELTY_SCOUT.md`.

The candidate inserts a continuous multimodal window from a same-corpus
negative train video into a positive train sequence. Only the inserted donor
interior receives a dense benign target; unchanged recipient positions remain
latent and receive a consistency constraint. The original POWA loss and the
composite positive-bag MIL loss remain active.

Stage P command used after review PASS:

```bash
setsid bash experiments/20260831_powa_benign_insertion_pilot/run_stage_p.sh \
  > runs/20260831_powa_benign_insertion_pilot/stage_p_supervisor.log 2>&1 \
  < /dev/null &
```

Outputs: `runs/20260831_powa_benign_insertion_pilot/`. Each run records the
config/history, checkpoint, augmentation manifest, dense test scores, log,
PID, code commit, plan hash, and shared-evaluator `metrics.json`.

Stage M controls were not run because Stage P failed its frozen gate.
