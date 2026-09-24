# Rule-6 code review — 20260925_query_paradigm (QTL)

**Verdict: PASS with must-fix list** (2 must-fix items; item 1 before the search starts, item 2 before the ablation runs).

Reviewed 2026-09-25 against README sections 2 and 4 as on disk at 02:33 (the version that fixes the primary operating point to fixed B = 8 and documents the `/ n_answers` normalisation). Files: `qtree.py`, `policy.py`, `data.py`, `model.py`, `train.py`, `search.py`, `launch/run_search.sh`; the answer cache format was read from `data/vlm_tree/HateClipSeg/answers_qwen7b_mod5.shard{0,1}of2.jsonl`. Scope per rule 6: mechanism enters forward/loss/score, leakage, alignment, hparam/checkpoint chain, shared evaluator. No training was run; CPU checks on synthetic data only (scripts in the session scratchpad, not kept).

## Must-fix

### 1. Primary operating point in code is the adaptive rule; README (§2.4, §4, §5) fixes it to fixed B = 8 — the Optuna objective and the gate number are taken from the wrong operating point

- `train.py:252` `primary = res["adaptive"][str(int(cfg["primary_budget"]))]`
- `train.py:256` silent-group `ap_primary` uses `at_threshold(test_runs, primary["c_bits"])`
- `train.py:261-265` `metrics.json["test"]`, `res["test"]`, `res["test_mean_calls"]` come from `primary` (adaptive)
- `train.py:9-12` docstring: "summary["test"] = adaptive rule at primary_budget"
- `search.py:89-91` objective reads `s["test"]` (adaptive) and the validation-only pick reads `s["adaptive"][pb]["val"]`; `search.py:6-7` docstring says the same

What goes wrong: README §2.4 (line 65) now states "主比较点：每视频固定问 B = 8 次（按 EIG 顺序），不含任何在 validation 上校准的量；Optuna 目标与 checkpoint 选择都用它", §4 line 92 "test 在主比较点（每视频固定按 EIG 问 8 次）", §5 P1 "固定 B = 8". The code was last modified 02:20/02:22, before that README change, and still ranks trials by the adaptive mean-8 number (whose threshold `c` is calibrated on the same validation set used for checkpoint selection) and writes that number to `metrics.json` / `summary["test"]`. The best trial, the number compared against the rule-8 gate, and the "validation-selected trial" record would all be computed at a different operating point from the pre-registered one. The checkpoint criterion (`train.py:191-193`, fixed 8 EIG calls on validation) already matches the README; only the test-side selection is stale.

Fix (all the fixed-8 numbers are already computed through the shared evaluator at `train.py:229-236`, so this is a re-pointing):
```python
# train.py:252
primary = res["fixed"][str(int(cfg["primary_budget"]))]
# train.py:256
st = at_budget(test_runs, int(cfg["primary_budget"]))
# train.py:261-265: keep as is (they now read the fixed entry); primary["test_mean_calls"] exists for fixed too
# search.py:91
v = s["fixed"][pb]["val"]
```
Update the two docstrings (`train.py:9-12`, `search.py:6-7`) to "summary["test"] = fixed primary_budget EIG calls; adaptive rule at mean budgets reported alongside (P2/P2b)". Keep `res["adaptive"]` as it is — P2 (mean 4) and P2b (adaptive 8 vs fixed 8) read from it.

### 2. Ablation (b) as defined in README §5 has no arm in `train.py`

- `train.py:14-16` arms list: `categories`, `n_state`, `length_term`, `objective`, `order` — no fusion arm
- `policy.py:52-86` `run_video` only produces the tree posterior `p`

What goes wrong: README §5 (line 101) defines (b) as "不用树结构的逐秒相加：logit p_t = logit(g·π_t) + Σ_{已问节点 n 覆盖 t} [log P(o_n | s=2) − log P(o_n | s=0)] / (覆盖 t 的已问节点数)，同一模型、同样的已问节点". Rule 14(g) requires this ablation for the final claim; without an arm it cannot be produced, and the "OR-tree fusion is useful" conclusion has no control. Not blocking for the main search.

Fix: add cfg key `"fusion": "tree"` to `DEFAULTS` (`train.py:51-58`) and to the arms docstring; in `policy.run_video` also accumulate a flat score after every call when `fusion == "flat"`:
```python
# after logA[node] = ... and before vt.infer:
prior_logit = g + s - np.logaddexp(0.0, g) - np.logaddexp(0.0, s)   # logit(sigmoid(g) * sigmoid(s)) per second
llr = np.zeros(T); cnt = np.zeros(T)
for n in asked_with_answer: a_, b_ = tr["a"][n], tr["b"][n]; llr[a_:b_] += logA[n, 2] - logA[n, 0]; cnt[a_:b_] += 1
p_flat = 1 / (1 + np.exp(-(prior_logit + llr / np.maximum(cnt, 1))))
```
Question selection stays EIG on the tree posterior (same asked nodes); `Evaluator.run` / `at_budget` / `at_threshold` read `scores_flat` instead of `scores` when `fusion == "flat"`; the validation checkpoint criterion in `train.py:191-193` must read the same score list so the arm changes only its component. Training is unchanged in this arm.

## Checked and found correct (no action)

- **Mechanisms enter forward/loss/score.** `TreeBatch.log_evidence` (`qtree.py:182-224`) is used in the loss at `train.py:166-168` with `torch.where(label > 0.5, lp1, lp0)`: positives train π, θ[:,1:], ω through the tree; negatives train θ[:,0]. Verified on synthetic batches that the batched pass fed as train.py feeds it (padded `s` with junk in padded cells, bool `mask`, `Tm = f_v.shape[1]`, `leaf_row = vid*Tmax + a`) equals single-video float64 inference and brute-force enumeration for n_state 3/2 and categories all/[0]; `tb.n_obs` equals the number of non-None answers per video. Float32 drift at T = 5809 with 2047 answers: 5e-4 absolute on lp1 ≈ −17,000 (≈1e-6 per answer) — negligible. `VideoTree.infer` (`qtree.py:239-287`) posterior `p_G · P(y_t=1 | G=1, o)` is the final score (`policy.py:56,80-81`); `p_G`, node marginals `m` and per-second `p` match enumeration over all (G, y) on T = 16 after two asked nodes. `Asker.eig` (`policy.py:46-49`) equals the exact mutual information I(o_n; (G, y) | asked) computed by enumeration (5 candidates, diff < 1e-7), and the state weights at `policy.py:65` are the correct three-state probabilities. `stop_calls`/`calibrate` (`policy.py:89-110`) implement "ask while best EIG ≥ c" and the smallest c with validation mean calls ≤ B, monotone in c.
- **Arms enter.** `categories` → `AnswerModel.cats` (`qtree.py:103`) and `Asker` (`policy.py:36,44`); `n_state 2` → `init_theta`, `loglik` state sharing (`qtree.py:104-105`), `Asker` (`policy.py:37-38`); `length_term` → `omega=None`; `objective "mil"` → `s.detach()` for the tree + top-k BCE (`train.py:165,171-177`); `order "bfs"` → `Evaluator.run` (`train.py:96`), also used for the validation checkpoint criterion (same component change only).
- **Leakage.** The network's forward (`model.py:74-85`) receives `f_a, f_v, mask` only; no answer tensor reaches it. `init_theta`, `loglen` (`train.py:136-140`) use train ids only. Test labels appear only in `evaluate` for the silent-group statistic (analysis). Checkpoint selection is validation-only (`train.py:191-201`); the adaptive threshold `c` is calibrated on validation EIGs only (`train.py:238`) and applied to test. Val/test answers are read only as the simulated oracle (`Evaluator`), as README §2.1 states.
- **Alignment.** `data.Store`: audio native seconds (`align.aligned_audio(..., "second")`), BERT rows resampled 1 s → T, visual `W @ x` with `resample_matrix` verified equal to `align.resample_intervals` (20 random grids incl. audio outliving visual). Tree T = VGGish rows (`hc.video_duration`) and `train.py:131-132` asserts it equals the cached `T`; `fec.evaluate` raises on any score/GT length mismatch, so misalignment cannot pass silently. Node ids: `build_tree_manifest.tree_nodes` and `qtree.tree` use the same split `a + (b-a)//2`; queryable sets verified identical for T ∈ {4, 5, 7, 13, 100, 268, 5809}; `(a, b)` keys are int tuples on both sides (`data.py:88`, `policy.py:77`). Splits/labels via `hc.load_fixed_cohort`. Padding: leaves only exist for a < T so padded rows are never indexed; `sum_lv` masked; attention pooling masks padded keys/rows; CMAL slices by `seq`. Five-crop: `policy.video_prior` averages the per-second logit `s` and `g` over the five crops before inference (logit-mean rather than probability-mean; a valid protocol, README does not specify).
- **Hparam / checkpoint chain.** `search.sample` → `hparams.json` → `train.py --config` → `cfg.update` with unknown-key assert; `lr`, `lamda_cma`, `dropout` reach the optimiser, the CMAL warm-up and `CrossModalLayer.drop`. TPE sampler seed = training seed; one study per (corpus, seed) in `optuna.db`; best-epoch state for both `model` and `am` restored (`train.py:202-203`) before `evaluate`.
- **Shared evaluator.** Every reported test number (fixed and adaptive, all budgets) goes through `hc.run_evaluator` → `scripts/reproduction_baselines/eval_baseline_scores.py` (`train.py:222-227`), reading `results.score_av.{pr_auc, roc_auc, per_video.macro_auc}`; the raw evaluator outputs are kept as `metrics_test_<name>.json`.
- **README/code objective.** `/ n_obs` normalisation (`train.py:167-168`) is now stated in README §2.3; `DEFAULTS` match §4 (hid 128, ffn 128, 4 heads, batch 32 with length-based shrink, 50 epochs, cosine T_max 60, warm-up min(λ, .05·epoch), crop_repeat 5).

## Minor (not conclusion-changing, no action required)

- `search.py:102-104` counts FAIL trials toward the fixed trial budget, so a crashed trial silently reduces the number of completed trials below the declared 20/5. If desired, count only COMPLETE trials in `n_done()`.
- CMAL is called with `(audio_rep, visual_rep) = (v_out, a_out)` (`train.py:180-181`), reproducing upstream's default (`fix_rep_swap False`) as the comment says; consistent with the it5 baseline.
