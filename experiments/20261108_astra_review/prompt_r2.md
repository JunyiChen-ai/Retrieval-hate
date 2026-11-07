[Round 2 of at most 3 of the same review loop]

Same role, same rules as round 1: you are a hostile senior reviewer; read the files yourself; the author's notes are not evidence. Only read: do not create, modify or delete any file, and do not start training or GPU jobs (short Python one-liners that print to stdout are fine, python: `~/miniconda3/envs/HateVideo/bin/python`). The out-of-scope list from round 1 still holds (test-set trial selection and the test-selected vs validation-selected gap; the choice of datasets and metrics; not retraining the baselines; the budget being user-set).

## What changed since round 1

All four experiments you listed (E1–E4) were run. Code and documents changed in git commits `f989b00..e9cfb92` (read the diff: `git diff f989b00..e9cfb92`). The method itself (revision 6) is unchanged: no retraining of the method, no change to its code path (the new config key `soft_fixed_edges` is absent in the method and only used by two new ablation arms).

- Log of this round, with the results tables and the issues found in our own code review: `experiments/20261108_astra_review/README.md`. Our internal code review of the new code: `experiments/20261108_astra_review/REVIEW_RULE6_R1.md`.
- New code: `experiments/20261108_astra_review/{baseline_infer.py, e1_fusion.py, policy_replay.py, e3_level_matched.py, e4_budget_audit.py, launch/run_round1_cpu.sh}`, `src/qtl/replay.py`, `src/qtl/data.py` / `src/qtl/train.py` (`soft_fixed_edges`), arms `hard2` / `hard2q` in `experiments/20261101_r6_length_table/launch/sbatch_arms.sh`.
- Raw results (local, not in git):
  - E1: `runs/20261108_astra_review/e1/<corpus>/<baseline>/{seed_<s>.json, summary.json}`; baseline re-scoring checks `runs/20261108_astra_review/baseline_scores/<method>/<corpus>/seed_<s>/check.json`.
  - E2: `runs/20261101_r6_length_table/arms/<corpus>/seed<s>/{hard2,hard2q}/summary.json` (table: `experiments/20261101_r6_length_table/arms_table.py`); the paired differences use the existing `no_node`, `c_label`, `no_length`, `f_joint` arms.
  - E3: `runs/20261108_astra_review/e3/<corpus>/summary.json`.
  - E4: `runs/20261108_astra_review/e4/<test_selected|val_selected>/<corpus>/summary.json`.
  - Policy replays used by E1/E3/E4: `runs/20261108_astra_review/replay/<corpus>/seed<s>_trial<k>/runs_L<level>.pkl`.
- Wording (your point 5): `experiments/20261101_r6_length_table/README.md`, `research-wiki/STATUS.md` and `research-wiki/method_framework.html` now say "better overall, with local drops; gains get smaller after 8 calls", with the drops listed.

## Claims as we would now state them

- C1: unchanged.
- C2: more VLM calls improve the results overall, with local drops (largest: HCS 3→4 calls AP / ROC −.009) and smaller gains after 8 calls.
- C3: removing or replacing each component lowers AP or ROC by ≥ .01 on all three corpora (14 arms) and, isolated: soft vs hard answers (E2), fixed anchored table vs joint learning (E2), tree vs single level at the same checkpoint and calls per video (E3; except HCS 4–8 s windows). Training on the answers has no independent pooled effect on HCS once the node prior is held fixed. The backbone helps AP / ROC; on HCS it lowers within.
- C4: with the same 8 answers, our method has higher AP than the strongest baseline with either fusion (fixed or learned) on all three corpora, and higher ROC on HateMM and HCS; on DeHate ROC it ties the learned fusion (+.001) and is .007 below the fixed fusion.
- C5: the budget is calibrated on validation; curves are reported against the realized mean calls; the randomized two-threshold calibration makes the expected validation calls equal B, with metrics within .002 of the deterministic rule.

## Your task

1. For each of your round-1 points: is it settled by what was run? Check the code and the raw files, not our tables. If an experiment was done wrongly (bug, leakage, an unfair setting of the comparison), say exactly where.
2. Any remaining or new concern about the method and its claims, ranked by severity, typed EXPERIMENT or WRITING as before. Do not repeat concerns that were settled; do not raise items in the out-of-scope list.

## What to output

Write in Chinese; keep technical terms in English. Use exactly these sections:

1. **Score**: 1–10 for a top venue, plus a verdict: ready / almost / not ready.
2. **Round-1 points**: one line each (1–5): settled / partly settled / not settled, with the file or number you checked.
3. **Weaknesses**, ranked by severity. For each: (a) the problem, with the file or number you checked; (b) which claim it hurts; (c) type: EXPERIMENT or WRITING; (d) for EXPERIMENT: the minimum experiment, precise enough to implement (what to change, which corpora and seeds, what result settles the concern either way).
4. **Experiment list**: the EXPERIMENT items only, in priority order. If no remaining concern can be settled by an experiment, write the single line `NO_MORE_EXPERIMENTS`.
