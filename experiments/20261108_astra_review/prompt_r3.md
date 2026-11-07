[Round 3 of at most 3 of the same review loop — the last round]

Same role and rules as before: hostile senior reviewer; read the files yourself; the author's notes are not evidence. Only read: do not create, modify or delete any file, and do not start training or GPU jobs (short Python one-liners that print to stdout are fine, python: `~/miniconda3/envs/HateVideo/bin/python`). The out-of-scope list from round 1 still holds.

## What changed since round 2

Git commits `e9cfb92..6ebbbf7` (read `git diff e9cfb92..6ebbbf7`). The method (revision 6) is unchanged.

- Your weakness 1 (E1 did not learn baseline and answers jointly): new control E1b. `experiments/20261108_astra_review/baseline_infer.py --hidden` extracts each frozen official baseline's per-second hidden features (the input of its final per-second classification layer: MACIL-SD [a_out, v_out] crop mean; Fed-WSVAD temporal encoder output; DSANet classifier input), in the same forward pass that recomputes its score (check files `runs/20261108_astra_review/baseline_scores/<method>/<corpus>/seed_<s>/hidden_check.json`). `e1_fusion.py` variant `hid`: these features (standardized with training-split statistics, linear map to 32 d) + the baseline logit + all answer tokens into one head whose output is the fused score; MIL loss on that output; same node sampling, learning rates, epochs and validation selection as E1. Variant `hid_val`: the same head trained on the validation videos (where the baseline is out-of-sample, as on test), learning rate and epoch by 5-fold cross-validation on validation, one refit, test once. Results: `runs/20261108_astra_review/e1b/` and `runs/20261108_astra_review/e1b_val/` (`<corpus>/<baseline>/{seed_<s>.json, summary.json}`).
- Your weakness 2 (E4 ".002"): corrected in `experiments/20261108_astra_review/README.md` (E4 paragraph) and `research-wiki/STATUS.md`, with the per-trial maxima.
- Your weakness 3 (DSANet branch in the old fixed fusion): `experiments/20261101_r5_completion/baseline_plus_vlm.py` now reads `score_mlp` for DSANet; rerun `runs/20261108_astra_review/fixed_fusion_mlp/hateclipseg_r6_mlp_k8.json`; table corrected in `experiments/20261101_r6_length_table/README.md`. Our rerun gives .595 / .572 at its best alpha .3, not your .597 / .583 at alpha .2; please check which is right.
- Our code reviews: `experiments/20261108_astra_review/REVIEW_RULE6_R2.md`. Log with all tables: `experiments/20261108_astra_review/README.md` (Round 2 sections).

## Claims as we would now state them

C1, C2, C3, C5 as in round 2 (with the E4 correction). C4: with the same 8 answers, our method has higher AP than the strongest baseline under every fusion we tried (fixed weight chosen on test; learned answers-only head with a validation-chosen baseline weight; joint head on the baseline's hidden features, trained on training or on validation videos), on all three corpora, and higher ROC on HateMM and HCS; on DeHate ROC the fixed fusion is higher (.777 vs .770) and the learned answers-only head ties (.769).

## Your task

1. For each of your round-2 points: settled / partly settled / not settled, checked against code and raw files. If E1b was done wrongly (bug, leakage, an unfair setting), say exactly where.
2. Any remaining or new concern, ranked, typed EXPERIMENT or WRITING. Do not repeat settled concerns or raise out-of-scope items. Since this is the last round, list as EXPERIMENT only what is necessary for the claims as stated above, not what would merely be nice.

## What to output

Chinese, technical terms in English. Exactly these sections:

1. **Score**: 1–10 for a top venue, plus a verdict: ready / almost / not ready.
2. **Round-2 points**: one line each (1–3): settled / partly settled / not settled, with the file or number you checked.
3. **Weaknesses**, ranked by severity. For each: (a) the problem, with the file or number you checked; (b) which claim it hurts; (c) EXPERIMENT or WRITING; (d) for EXPERIMENT: the minimum experiment, precise enough to implement.
4. **Experiment list**: the EXPERIMENT items only, in priority order. If no remaining concern can be settled by an experiment, write the single line `NO_MORE_EXPERIMENTS`.
