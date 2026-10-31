# Scale-aware cross-modal attention: independent rule-6 code review

Date: 2026-10-31. Reviewer: independent agent (Claude Fable 5.1), the one rule-6 review of
`experiments/20261031_scale_aware_cma/`. Scope per rule 6: only bugs that would change experimental observations or
conclusions (mechanism in forward/loss/score, leakage, alignment, hyperparameter and checkpoint chain, unified
evaluator). Style, refactoring, robustness and theory were not reviewed.

Read: `backbone.py`, `train.py`, `search.py`, `launch/run_search.py`, `launch/sbatch_search.sh`, the README;
`src/qtl/train.py` (commit 14033de diff against f2fca2c), `src/qtl/model.py`, `src/qtl/search.py`,
`src/qtl/run_owner.py`, `src/qtl/checkpoint.py`, `src/qtl/data.py`, `src/qtl/cpolicy.py`, `src/qtl/policy.py`;
r5 reference `runs/20260929_query_paradigm_r5/hatemm/seed234/study_summary.json` (`extra`) and
`.../trial9/summary.json` (`cfg`); the r5 entry points `experiments/20260925_query_paradigm/{train,search}.py`
(now wrappers around `src/qtl`). CPU checks were run with `~/miniconda3/envs/HateVideo/bin/python` (script kept in
the session scratchpad, not in the repo; no training, no GPU).

## Verdict: PASS

No bug was found that would change a number. The mechanism enters the forward pass and therefore the training
loss, the validation checkpoint criterion and the test scores; the zero-bias network is numerically the r5 network;
the configuration, search, checkpoint and evaluator chain are the same as r5's apart from the two intended keys.

## Checks and results

1. **Bias placement, indexing, broadcasting, padding.** `ScaleBiasAttention.forward` adds
   `scale_bias[:, buckets(T)]` (shape `(1, nhead, T, T)`) to `q k^T / sqrt(d_k)` before the key mask and before
   the softmax; the mask then fills padded keys with -1e9 as in r5. Verified: for every head h,
   `bias[h, i, j] == table[h, k(|i - j|)]`. With a random non-zero table, a padded batch of lengths (37, 20, 5)
   gives the same `s`, `g`, `v_out` on the real positions as each video run alone unpadded (max abs diff < 1e-5),
   so padding does not corrupt real rows. Validation and test call the model per video with a full mask
   (`policy.video_prior_nodes`), so the table is indexed at the video's true length there. Training feeds whole
   videos (`long_T` only shrinks batches; the "crop" in `TrainSet` is the I3D spatial crop, not a temporal crop), so
   `|t - t'|` is the true distance in seconds in both phases.
2. **Bucket function.** `scale_buckets(T)` equals `0` at d = 0 and `min(1 + floor(log2 d), 10)` for d >= 1,
   checked exhaustively against the README formula for T = 1, 2, 3, 600, 1100. Row 0 at d = 0, 1, 2, 3, 4, 7, 8,
   255, 256, 511, 512, 599 gives 0, 1, 2, 2, 3, 3, 4, 8, 9, 9, 10, 10.
3. **Zero bias equals r5; `scale_bias` switch.** `make_model(cfg)` loaded with a `PriorNet` state dict (only extra
   key `cma.attn.scale_bias`) gives identical `s`, `g`, `a_log`, `v_log`, `v_out`, `a_out` on real positions and
   identical `node_logits` (atol 1e-6, eval mode). The attention dropout stays at the r5 default 0.1 (the searched
   `dropout` is the layer's residual dropout, as in r5). With a non-zero table the per-second logits change (max
   diff .07 on random inputs), so the mechanism reaches the score. `scale_bias=true`: the table receives non-zero
   gradients (28 of 44 entries for T = 37, exactly the 7 buckets that occur at that length) and
   `small_parameters()` returns it. `scale_bias=false`: `requires_grad` False, table 0, `small_parameters()` empty.
4. **Optimizer hook (`src/qtl/train.py` 266-271).** Rebuilding the groups as the trainer does: with the bias
   learned, the table is in the `lr_answer` group exactly once and in the `lr` group zero times; every trainable
   parameter is in exactly one group. With the bias frozen, or with r5's `PriorNet`, the groups are the ones the
   previous code produced (all model parameters in the `lr` group, chain in `lr_answer`). The `const` backbone path
   is unchanged (`model_small` is empty, first group still omitted). `hasattr(model, "small_parameters")` is true
   only for `ScaleAwarePrior`; no other backbone in the repository defines the name. The cosine scheduler covers all
   groups as before.
5. **Effective configuration vs r5.** `DEFAULTS` + `EXTRA_DEFAULTS` + r5 trial 9's four sampled scalars, with
   `train.train`'s soft-source rule applied, differs from r5 trial 9's saved `cfg` only in `backbone`
   (`macil` -> `scale_cma`) and `scale_bias` (new, true), plus two keys that did not exist in r5's trainer at the
   time (`soft5_edges` "fixed", `state1_table` None) and are read only by the `soft5_*` and `anchored_ext` paths,
   so they are inert here. `categories` becomes `[0]`, `qtree.N_CAT, N_LEV` = 1, 8, `answer_source` soft_both,
   `node_prior` true, anchored answer model, learned closed zero-inflated chain. The Optuna space in
   `qtl.search.sample` is the four r5 scalars with r5's ranges; the objective and the validation-selected record
   are the shared ones; `extra` passes the `known` key check in `qtl.search.main` and the unknown-key check in
   `qtl.train.main`.
6. **Checkpoint chain.** `model.pth` carries `cma.attn.scale_bias` (a Parameter whether or not it trains) and no
   bucket cache; `summary["cfg"]` carries `backbone` and `scale_bias`, so `qtl.checkpoint.load_trial(trial, ...,
   model_factory=backbone.make_model)` rebuilds the right class and `load_state_dict(strict=True)` succeeds (checked
   both ways: a learned table into a frozen factory and back). `make_model` refuses an r5 `cfg`, so a later
   evaluation script cannot silently load this checkpoint into `PriorNet` or the reverse. Any later evaluation
   script must pass this experiment's `make_model` as the factory.
7. **Unified evaluator.** Unchanged: test numbers come from `hc.run_evaluator` through `qtl.train.evaluate`, and
   `run_owner` verifies `metrics_test_fixed8.json` for every trial.
8. **Output path and launch.** `run_owner` resolves `runs/20261031_scale_aware_cma/<corpus>/seed<seed>/` from the
   repository root; trials go to `trial<k>/` under it. Imports of `train.py` and `search.py` succeed from their own
   directory (as `sbatch` -> `run_search.py` -> `search.py` -> `train.py` subprocesses run them). The sbatch
   resources (1 GPU, 8 CPUs, 48G, 23 h) fit the lab partitions.

## Notes (not bugs, no number changes; for the launcher)

- The submit command writes Slurm stdout to `runs/20261031_scale_aware_cma/slurm_<corpus>_<seed>_%j.out`. Slurm
  does not create that directory and fails the job at start when it is missing. Checked on 2026-10-31: neither
  `uoa-lab1` nor `uoa-lab3` has `~/Retrieval-hate/runs/20261031_scale_aware_cma/`, and neither has commit 14033de
  yet (lab1 at 2101775 with one modified file, lab3 at 42ee17e). Before submission: `git pull` on the host and
  `mkdir -p runs/20261031_scale_aware_cma`.
- The rule-14(g) ablation (`scale_bias=false`) is r5's architecture, but `ScaleAwarePrior.__init__` replaces the
  attention module after `PriorNet.__init__`, so with the same seed its initial weights are a different random
  draw than an r5 run's. It is "exactly r5" as a function class, not a bit-identical rerun; a 3-seed comparison
  absorbs this. There is no dedicated entry for the locked single-trial ablation; `train.py --config` with the
  locked trial's `hparams.json` plus `"scale_bias": false` is accepted by `train.main`. Running it through
  `run_search.py --extra-config '{"scale_bias": false}'` needs a different `--out-root`, since `run_owner` refuses
  a directory that already has `identity.json`.
