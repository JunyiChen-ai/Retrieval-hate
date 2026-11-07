# 20261108_astra_review: external review loop with GPT 6 Astra

User request, 2026-11-08: have GPT 6 Astra review the current method acting as a hostile reviewer, list the experiments still needed, run them, and repeat. The loop ends when GPT 6 Astra can no longer name a concern that an experiment can settle; at most 3 rounds. Test-selected trial selection (and the test-selected vs validation-selected gap) is out of the review's scope (user instruction).

- Method under review: revision 6 (`research-wiki/method_framework.html`, `experiments/20261101_r6_length_table/README.md`).
- Reviewer: `codex exec -m gpt-6-astra -c model_reasoning_effort="xhigh" -s read-only` run from the repository root (codex-cli 0.159.3). The reviewer reads the repository and `runs/` itself. Later rounds resume the same codex thread (`codex exec resume <thread id>`), so the reviewer keeps its memory of earlier rounds.
- Files: `prompt_r<N>.md` holds the exact prompt sent; `review_r<N>.md` holds the reviewer's final message, verbatim. The event stream is in `runs/20261108_astra_review/events_r<N>.jsonl`. `REVIEW_STATE.json` records the round, thread id, and pending experiments.

## Rounds

### Round 1 (2026-11-08)

The first attempt (codex thread `01a1b783-c4a2-…`) could not read any file: codex's read-only sandbox needs unprivileged user namespaces, which this machine blocks (`kernel.apparmor_restrict_unprivileged_userns = 1`; `bwrap: loopback: Failed RTM_NEWADDR`). Rerun with codex's default sandbox mode from the user's own `~/.codex/config.toml` (`danger-full-access`), with "only read" in the prompt; afterwards `git status` and a search for files modified during the run showed no change outside this directory. Thread `01a1b785-6cb6-72f2-88a8-f906db6e7f02`; prompt `prompt_r1.md`; response `review_r1.md` (verbatim).

**Assessment: 5/10, not ready.** C1 is supported. Its points, by severity:

| # | Point | Claim | Type | Experiment |
|---|---|---|---|---|
| 1 | The 14(f) baseline + VLM control keeps only the shortest asked node per second and one mixing weight; on DeHate it already has a higher ROC than ours (.777 vs .770, all 3 seeds) | C4 | experiment | E1: frozen official baseline + a fusion head trained with video labels on all 8 answers (interval length and position, optionally our input features); validation and test use exactly the answers our method used |
| 2 | Several ablation arms change two things at once (`c_label`, `f_joint`); `words` changes the prompt and answer format, so it does not isolate "soft answers" | C3 | experiment | E2: the same `soft_both` answers cut into 2 levels (at p = .5 = the hard yes / no answer; also at the training median), full training, 3 corpora × 3 seeds; plus the paired differences of the composite arms against `no_node` / `no_length` |
| 3 | The single-level arms change both the training answers and the number of test calls (lvl16: 5.0 / 4.0 calls on HateMM / DeHate vs 7.6 / 7.3) | C3 | experiment | E3: same checkpoint, per video the same number of calls k_v = min(8, askable nodes at that level), tree policy vs single-level policy |
| 4 | The stopping rule does not spend the budget it is given (HateMM seed 3407, validation-selected, B = 7: 4.72 calls on validation) | C5 | experiment | E4: per seed target vs validation / test calls for B = 1–7; a randomized choice between the two neighbouring thresholds (probability from validation call counts only) so the expected validation calls equal B |
| 5 | "More calls are always better" is contradicted by local dips (HCS fixed 3 → 4 calls AP −.009; HateMM 24 → 32 ROC −.005) | C2 | writing | reword to "better overall, with local dips and saturation"; show the full curves |

Code for round 1 (all in this directory unless noted):
- E2: `src/qtl/data.py` / `src/qtl/train.py` new config key `soft_fixed_edges` (fixed p_yes edges instead of train quantiles; absent = the method, unchanged); arms `hard2`, `hard2q` in `experiments/20261101_r6_length_table/launch/sbatch_arms.sh`.
- Shared replay `src/qtl/replay.py` (moved out of `experiments/20261101_r5_completion/sim_monotone.py`, which and `final_curves.py` now import it; adds `splits` and `level`).
- `policy_replay.py` (saves the policy runs of a trial), `e3_level_matched.py`, `e4_budget_audit.py`, `baseline_infer.py` (re-scores the frozen official Fed-WSVAD / DSANet checkpoints on train / val / test; MACIL-SD HateMM already had them), `e1_fusion.py`; launcher `launch/run_round1_cpu.sh <corpus>` (CPU, no GPU).
- Inputs copied for E1: HCS CLIP features and the official HCS Fed-WSVAD / DSANet final checkpoints from `~/Hate-follow-up/runs/legacy_lab2_results_pre20261004/reproduction/` to uoa-lab3 (`runs/20261108_astra_review/baseline_models/`); the DeHate Fed-WSVAD checkpoints from uoa-lab3 to uoa-lab1; MACIL-SD HateMM train / val / test scores converted unchanged (`score_av`) to `runs/20261108_astra_review/baseline_scores/macilsd/hatemm/`.
