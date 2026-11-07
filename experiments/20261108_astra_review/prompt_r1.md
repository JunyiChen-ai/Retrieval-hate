[Round 1 of at most 3 of a review loop]

You are a senior reviewer for a top ML / CV venue (NeurIPS, ICML, CVPR). Assume the work is broken somewhere and find where. Read the files yourself. The author's notes are not evidence: check claims against the code and the raw result files.

You have shell access to the repository. Only read: do not create, modify or delete any file, and do not start training or GPU jobs. If you want to recompute a number, use short Python one-liners that print to stdout (python: `~/miniconda3/envs/HateVideo/bin/python`).

## What is under review

A method paper on weakly supervised hateful video localization: per-second hate scores, trained with video-level labels only.

The current method is "revision 6". Its parts:
- A cross-modal backbone gives a per-second prior, a video prior g, and a prior φ_n for each node of a binary tree over the video.
- A VLM (Qwen2.5-VL-7B) is asked questions about segments. Each question is a node of the binary tree, chosen by expected information gain.
- Answers are soft, in 8 levels, read through a fixed two-state answer table. The table has one length slope shared by both states.
- Answers and priors are combined by a zero-inflated chain posterior.
- A stopping rule (qmixE1) spends a user-set average call budget.

Claims of the paper:
- C1: at 8 calls it beats the strongest trained baselines on all three corpora (HateMM, HateClipSeg, DeHate) in pooled frame AP and pooled frame ROC.
- C2: more VLM calls give better results (fixed 0–32 calls; budget 1–8).
- C3: every component is needed (14 ablation and control arms, 3 seeds, all three corpora).
- C4: the gain is not only "more information from the VLM": the strongest baseline given the same 8 answers is worse.
- C5: with the stopping rule, the average number of calls is a budget the user sets, with a reported 1–7 call curve.

## Files

- Method and its history: `research-wiki/method_framework.html` (single page), `experiments/20260925_query_paradigm/README.md` (design of revisions r0–r5), `experiments/20261101_r5_completion/README.md` (diagnosis that led to revision 6), `experiments/20261101_r6_length_table/README.md` (revision 6 results, ablations, baselines + VLM, call curves).
- Code: `src/qtl/` (model, tree, answer table `qtree.py`, chain `ctree.py`, policy, stop rules, training `train.py`), `experiments/20261101_r6_length_table/` (search, launch scripts, `arms_table.py`), `experiments/20261101_r5_completion/` (`final_curves.py`, `sim_monotone.py`, `baseline_plus_vlm.py`).
- Raw results (local, not in git):
  - `runs/20261101_r6_length_table/<corpus>/seed<s>/`: Optuna studies, every trial's `summary.json` and `run.log`.
  - `runs/20261101_r6_length_table/arms/table.json` and `arms/<corpus>/seed<s>/<arm>/`.
  - `runs/20261101_r6_length_table/final_curves/val_selected/<corpus>/summary.json`.
  - `runs/20261101_r6_length_table/baseline_plus_vlm/`.
  - `runs/20261101_r6_length_table/sim_monotone/`.
- Project rules and fixed protocol: `RESEARCH_ITERATION_RULES.md` (rules 1–3, 7, 8, 13, 14). Baseline table: `docs/duplex/OFFICIAL_VAL_RESULTS.md`. DeHate baselines: `experiments/20260926_dehate_external/README.md` section 5.1.
- Evaluator (single copy, do not question unless you find a bug): `scripts/reproduction_baselines/eval_baseline_scores.py`, core `scripts/duplex/frame_eval_common.py`.

## Out of scope (fixed by the PI; do not raise)

- Selecting Optuna trials by a test-set objective, "test-selected" numbers, and the gap between test-selected and validation-selected numbers. Take the protocol as given.
- The choice of the three datasets and three metrics.
- Not retraining the baselines (their numbers come from the fixed table).
- The per-video call budget being a user-set value rather than chosen automatically.

## Constraints on what can be run

- Compute: about 3 RTX 5090 GPUs through Slurm. One full ablation arm is about 40 minutes per corpus with 3 seeds in parallel on one GPU. One full Optuna search is about 20 trials × 3 seeds, 2–18 hours per corpus.
- VLM answers are cached for every tree node of every video (train, validation, test), so any policy or answer-model change over existing nodes is cheap. New questions (new prompts, new segment shapes, a different VLM) need new VLM inference on a separate cluster that needs admin approval: possible, but slow.
- No ensembles, no inference-time post-processing or calibration (rule 3). One method for all corpora (rule 13).

## What to output

Write in Chinese; keep technical terms in English. Use exactly these sections:

1. **Score**: 1–10 for a top venue, plus a verdict: ready / almost / not ready.
2. **Weaknesses**, ranked by severity. For each:
   - (a) the problem, with the file or number you checked;
   - (b) which claim it hurts;
   - (c) type: EXPERIMENT (can be settled by running an experiment or analysis on this project's data) or WRITING (needs a change in framing or wording, or is not settleable by experiment);
   - (d) for EXPERIMENT: the minimum experiment, precise enough to implement: what to change, which corpora and seeds, what result would settle the concern either way.
3. **Experiment list**: the EXPERIMENT items only, in priority order. If no remaining concern can be settled by an experiment, write the single line `NO_MORE_EXPERIMENTS`.
