# Code review: qmixE1 stopping rule and evaluator script (2026-10-31)

One independent reviewer agent (Claude Fable 5.1), rule 6 scope (bugs that change numbers or conclusions only).
Change: commit deb3b38. Files: `src/qtl/stop_rules.py` (QMIX entry `qmixE1`),
`experiments/20261031_fewer_calls/{evaluate_stop.py, launch/run_r5.sh, README.md}`.
Checks were read-through plus CPU-only Python on the stored policy runs of
`runs/20261031_two_goal_diagnostics/policy_lab/{hatemm,hateclipseg}/r5/seed234_*_eig/runs.pkl` (16-call dumps, same
`cpolicy.run_batch` call as `evaluate_stop.py`) and on the r5 trial and stop_check score files present on this
machine. No file other than this review was edited.

**Verdict: PASS.** No bug that changes a number the script will write. Two things the run must be read with
(section "Report with the numbers") and one README correction.

## Checked and found correct

1. qmixE1 components (`stop_rules.QMIX["qmixE1"] = ("dlsum", "eig")`).
   - `dlsum[k]` = sum over seconds of |logit p_t after call k - logit p_t after call k-1|, `inf` at k = 0; verified
     against the stored posteriors (`scores[k]`, `scores[k-1]`) for k = 1..3 on both corpora, identical.
   - `eig[k]` = `cpolicy.run_batch` records `float(e[j])`, the argmax EIG of the chosen question before call k+1,
     for both code paths (`Asker.eig` before the first nested answer, `qtree.eig` with the copy mixture after);
     both return bits, so the component is on one scale across calls.
   - Index alignment: for every val/test video `len(scores) == len(eig) + 1`, `len(p_G) == len(scores)`,
     `len(eig_sum) == len(voi) == len(eig)`. `rule_values` yields n = len(eig) values, value k read before call k+1.
     `policy.stop_calls` returns the first k with value < c, i.e. the number of calls made; `scores[k]` is the
     posterior after k answers. `add_quantile_rules` keeps `inf` as `inf`, so `qmixE1[0] = inf` for every video
     (first call always made) and `eig[0]` is finite; a video with no queryable node gets 0 calls and `scores[0]`.
     No stop index exceeded `len(scores) - 1` on any video at target 5 (both rules, both corpora).
2. No leakage. `add_quantile_rules` builds every reference distribution from `vals["val"]` only and maps test values
   onto it; `policy.calibrate` sees only validation rule values and the target mean (no labels); the only label use
   is `estimate_pi_neg` on training negatives, as in the complete method.
3. Policy equals the complete method's policy. The `run_batch` call (EIG order, `copy_pi` from `estimate_pi_neg` on
   train negatives, `copy_mode="both"`, `record_voi`, `soft_both` 8 levels asserted against the trial cfg,
   `eval_chunk` batching sorted by T, `make_model` -> `qtl.model.PriorNet` for backbone `macil`, the same class the
   r5 scripts load) is character-for-character the `policy_lab.py` call. Its fixed-8 test scores equal the complete
   method's reference `runs/20260929_query_paradigm_r5/stop_check/<corpus>/seed234_trial*_r5rt/scores_test_fixed8.jsonl`
   per second to 5e-7 (the jsonl rounding) on all 214 HateMM and 79 HateClipSeg test videos. They differ from the r5
   trial's own `scores_test_fixed8.jsonl` (no copy likelihood; max abs diff .70), so the fixed-8 row written by this
   script is the right comparison point for goal 1 (same policy, 8 calls), not the no-copy r5 table.
4. Fixed 8 = `scores[min(8, len(eig))]` = min(8, number of queryable nodes) since len(eig) = min(max_calls, n_q).
   Evaluator: `hc.write_scores` + `hc.run_evaluator` -> `scripts/reproduction_baselines/eval_baseline_scores.py` ->
   `frame_eval_common.evaluate`, the same chain `src/qtl/train.py` and `stop_check.py` use for every table number;
   `pr_auc`, `roc_auc`, `per_video.macro_auc` are read from `results.score_av`.
5. Launch: trial dirs are the r5 test-selected trials used elsewhere (HateMM 9/4/5, HCS 13/11/19, DeHate 8/4/4;
   STATUS line on the r5 reference). Logs and pid in `runs/20261031_fewer_calls/`.

## Report with the numbers (not bugs in the computation)

A. The 32-call cap is a constant of the evaluated rule and it binds. `--max-calls 32` truncates every video at 32
   calls; a video whose quantile value never drops below c is counted as 32 calls and scored with `scores[32]`. This
   enters the validation calibration (capped videos consume budget, so c rises and other videos stop earlier) and the
   quantile reference (calls 17..32 add values). On the 16-call dumps at target 5, qmixE1 left HateMM 14/109 val and
   25/214 test videos at the cap with nodes still unasked (HateClipSeg 4 val, 1 test); truncating the same runs to
   caps 8/10/12/14/16 moved the calibrated threshold .56 -> .62 -> .68 -> .72 -> .75 and changed the stop index of
   89..114 of 214 HateMM test videos. 118/214 HateMM test videos have >= 32 queryable nodes, so some will still sit at
   32 in the evaluator run. How it is recorded: per row `videos_at_max_calls` (test only), `calls_quantiles` (max),
   and `calls_test_<rule>_5.json`; the validation side records only `validation_mean_calls`.
   Consequences: (i) the README section-2 sentence "there is no floor, cap or schedule constant" is not true for the
   evaluated rule; state the cap (32, inherited from stop_check) or raise `--max-calls` until `videos_at_max_calls`
   is 0 on both splits. (ii) The README section-3 table was computed from 16-call dumps (`policy_lab.py` default
   `--max-calls 16`), a different effective rule from the 32-call evaluator run; expect the evaluator numbers to
   differ from that table, most on HateMM. Suggested one-line fix in `evaluate_stop.py`: also write
   `validation_videos_at_max_calls = sum(stop_calls(vals["val"][v][rule], thr) >= a.max_calls for v in ids["val"])`.
B. Selection record: `qmixE1 = ("dlsum", "eig")` is numerically identical to `rule_scan.py`'s `qmixS1 = ("dlsum",
   "sel")` (verified equal on both corpora), and `rule_scan_targets.json` holds its test rows at targets 4..6; the
   README section-3 "qmixE1" rows are those rows. This is consistent with the README's own statement that the rule
   was chosen on test; it just means the HateMM/HCS r5 (3) rows are the selection set, not a check.

## Noted, not bugs

- `qdata.Store(..., ["bert"])` is hard-coded; all six r5 trials have `text_sources == ["bert"]`. A future backbone
  with other text sources would fail loudly at the input projection, not silently.
- `summary_<host>.json` is written only after all trials; per-trial `metrics_test_*.json` are written as they finish.
