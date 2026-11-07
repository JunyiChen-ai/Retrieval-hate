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

Rule-6 code review of the round-1 code (`REVIEW_RULE6_R1.md`, one agent): FAIL, two findings, both fixed and confirmed (PASS):
- F1 (E1, would change the conclusion): the head took the frozen baseline's score on the baseline's own training videos, where it is far more confident than on held-out videos (MACIL-SD HateMM video AP .98 train vs .71–.77 test), so the head would learn to over-trust it. Fix: the learned part h sees only the answers (and optionally features); f = alpha · baseline logit + h, alpha chosen on validation.
- F2: DSANet must be read from `score_mlp` (the branch of the official table, `scripts/reproduction_baselines/aggregate_official_val.py`), not `score_align`. The existing 14(f) control (`experiments/20261101_r5_completion/baseline_plus_vlm.py`) read `score_align` for DSANet, so its HCS DSANet row (.561 / .528 / .555 baseline alone) is not the official branch; the official branch gives .5535 / .5280.

Runs (started 2026-11-08 09:10): CPU on uoa-lab3 (HateMM, HCS) and uoa-lab1 (DeHate), `launch/run_round1_cpu.sh`, logs `runs/20261108_astra_review/`. Two false starts the same morning, fixed before any result was produced: the trial summaries (`final_curves/*/summary.json`) were only on uoa-lab2 (copied to both hosts); `src/utils` shadowed Fed-WSVAD's upstream `utils` package (`baseline_infer.py` now binds the name to the upstream package). On uoa-lab3 one HateMM replay (seed 2025) was killed for lack of memory (other users' jobs leave about 12 GB) and is rerun. E2 GPU jobs: Slurm 1966 / 1967 / 1968 (HateMM / HCS / DeHate, arms `hard2 hard2q`, partitions uoa-lab1 or uoa-lab3).

#### Round 1 results (3-seed means, test, AP / ROC / within; revision-6 test-selected trials unless noted; unified evaluator)

**E1, frozen baseline + learned fusion of the same 8 answers** (`e1_fusion.py`; `runs/20261108_astra_review/e1/<corpus>/<baseline>/summary.json`; uoa-lab3 HateMM / HCS 09:14–09:24, uoa-lab1 DeHate 09:29–09:38). Baseline re-scoring reproduces the official test scores (largest absolute difference 5e-5, `baseline_scores/*/check.json`).

| Corpus (baseline) | Ours, 8 calls | Fusion, answers only + alpha·baseline | Fusion + our input features | Ours − fusion (answers only) |
|---|---|---|---|---|
| HateMM (MACIL-SD) | .702 / .889 / .783 | .662 / .876 / .645 | .659 / .857 / .643 | .040 / .013 / .138 |
| HCS (Fed-WSVAD 3) | .691 / .680 / .563 | .564 / .546 / .497 | .582 / .553 / .529 | .128 / .134 / .066 |
| HCS (DSANet, official branch) | same | .567 / .551 / .492 | .609 / .583 / .537 | .125 / .129 / .071 |
| DeHate (Fed-WSVAD 3) | .279 / .770 / .681 | .231 / .769 / .577 | .216 / .751 / .600 | .047 / .001 / .104 |

- The learned fusion is weaker than the fixed shortest-node fusion of the old 14(f) control on HateMM and DeHate (old: .643 / .861 and .248 / .777, mixing weight chosen on test). The old control's DeHate ROC .777 > .770 therefore stands as the strongest baseline + VLM number on DeHate ROC.
- C4 holds on AP on all three corpora (≥ .040) and on ROC on HateMM and HCS; on DeHate ROC our method ties the learned fusion (+.001; per seed +.008 / +.005 / −.011) and is .007 below the old fixed fusion.

**E2, paired differences of the composite arms** (existing runs, `runs/20261101_r6_length_table/arms/`):

| Comparison | HateMM | HCS | DeHate |
|---|---|---|---|
| answer training, both without node prior (`no_node` − `c_label`) | .025 / .016 / .061 (AP per seed .106 / .002 / −.033) | −.001 / −.004 / −.031 | .038 / .028 / .064 |
| fixed anchored table vs joint learning, both without length term (`no_length` − `f_joint`) | .014 / .006 / −.006 | .016 / .022 / .006 | .016 / .014 / −.002 |

Training the network on the answers has no measurable pooled effect on HCS once the node prior is held fixed; `c_label`'s ≥ .01 drop on HCS came from removing the node prior at the same time. The fixed table beats joint learning by ≥ .01 (AP or ROC) on all three corpora.

**E2, hard vs soft answers** (same `soft_both` answers, same prompt, full training; only the answer levels change; Slurm 1966–1968 on uoa-lab3, 11-08 09:4x–11:45; `runs/20261101_r6_length_table/arms/<corpus>/seed*/hard2*/summary.json`, table by `arms_table.py`). Drop of the method (8 levels at training quantiles) to each arm:

| Arm | HateMM | HCS | DeHate |
|---|---|---|---|
| `hard2`: 2 levels cut at p_yes = .5 (= the hard yes / no answer) | .050 / .024 / .034 | .016 / .050 / .018 | .059 / .032 / .032 |
| `hard2q`: 2 levels cut at the training median | .059 / .024 / .032 | .035 / .039 / .001 | .071 / .049 / .025 |

Soft answers are useful (AP and ROC drop ≥ .016 on every corpus, both cuts); every seed of both arms is below the method's mean on AP and ROC except HCS `hard2` AP seed 234 (.684 vs .691).

**E3, tree vs one tree level, same checkpoint and same calls per video** (`e3_level_matched.py`; `runs/20261108_astra_review/e3/<corpus>/summary.json`). Method − single level (standard deviation over seeds); mean calls k_v:

| Level (window length) | HateMM | HCS | DeHate |
|---|---|---|---|
| 4–8 s | .008 / .022 / .005 (7.1 calls) | .006 / −.002 / −.004 (8.0) | .018 / .033 / .003 (6.6) |
| 8–16 s | .013 / .018 / −.020 (6.3) | .022 / .008 / −.006 (8.0) | .025 / .024 / −.001 (5.3) |
| 16–32 s | .020 / .016 / −.028 (5.0) | .029 / .016 / −.004 (8.0) | −.002 / .014 / −.010 (4.0) |

With training and calls held equal, choosing among all tree levels beats every single level by ≥ .01 in AP or ROC except HCS 4–8 s windows (AP +.006, ROC −.002). Within-video ranking is not better with the tree: single coarse levels are higher on HateMM (−.020 / −.028).

**E4, budget actually spent** (`e4_budget_audit.py`; `runs/20261108_astra_review/e4/<test_selected|val_selected>/<corpus>/summary.json`). Deterministic calibration (the method as reported) reaches the target on validation within .1 call in most cases, but not always: HateMM seed 3407 stays at 5.46 (test-selected) / 4.72 (validation-selected) calls for B = 6 and 7, DeHate seed 2025 6.64 at B = 7. On test, the mean calls differ from B by up to .9 (HCS seed 3407, B = 5: 5.94; 63 validation videos). The randomized two-threshold rule makes the expected validation calls equal B by construction; on test it brings HateMM seed 3407 at B = 7 to 7.06 / 7.08 calls. Randomized minus deterministic at the same B (mean of 5 draws; corrected in round 2, the earlier text said ≤ .002 everywhere): test-selected trials ≤ .001 AP, ≤ .0021 ROC, ≤ .002 within per trial; validation-selected trials ≤ .0013 except HateMM seed 3407, where the deterministic rule underspends most: B = 7 AP +.0051, ROC +.0075, within +.0016 (standard deviation over draws .0010 / .0017 / .0003); 3-seed means ≤ .0026 AP / .0037 ROC (HateMM, validation-selected). The exact expected validation budget is a property of the randomized variant; the method as reported (deterministic threshold) can underspend, so the budget axis should be the realized mean calls.

### Round 2 (2026-11-08)

Prompt `prompt_r2.md` (resumed thread `01a1b785-…`; the reviewer read `git diff f989b00..e9cfb92` and the raw files); response `review_r2.md` (verbatim). No file changed during the run besides the response (git status and a search for files modified after the start).

**Assessment: 6/10, not ready.** Round-1 points 2 (hard vs soft), 3 (tree vs single level) and 5 (wording) settled; 1 and 4 partly settled. New points:

| # | Point | Claim | Type | Action |
|---|---|---|---|---|
| 1 | E1 did not learn the baseline and the answers jointly: h never sees the baseline, alpha is chosen after training (on HCS DSANet alpha = 0 for every seed and variant, so the baseline is dropped); the "feat" variant uses our input features, not the baseline's learned features. The over-trust argument of our code review (F1) is an inference, not a measurement | C4 | experiment | E1b: the frozen baseline's per-second hidden features (input of its final classification layer) and all answer tokens into one MIL head, loss on the fused output; same node sampling, learning rates, epochs and validation selection; 4 baseline–corpus pairs × 3 seeds |
| 2 | "Metrics change by ≤ .002" (E4) is false for HateMM validation-selected seed 3407, B = 7: AP +.0051, ROC +.0075 | C5 | writing | corrected above (E4 paragraph) and in STATUS |
| 3 | The old fixed-fusion HCS DSANet row used `score_align`; the reviewer's read-only recomputation with `score_mlp`: alpha .2, AP / ROC .597 / .583 | C4 | writing | recomputed, below |

**Old fixed fusion, HCS DSANet on the official branch** (`experiments/20261101_r5_completion/baseline_plus_vlm.py` now reads `score_mlp` for DSANet; rerun on uoa-lab1 11-08, same inputs as the r6 row, `runs/20261108_astra_review/fixed_fusion_mlp/hateclipseg_r6_mlp_k8.json`): baseline alone .5535 / .5280 (the official table); best alpha on test (by AP + ROC) .3: .595 / .572 / .539; alpha .2: .591 / .573. The reviewer's .597 / .583 at alpha .2 is not reproduced exactly (difference .006–.010 ROC; its computation is not in a file); both are .09–.11 below our .691 / .680. The Fed-WSVAD row is unchanged (.587 / .575 at its best alpha 0).

Code for round 2: `baseline_infer.py --hidden` (hidden features by forward hooks on the same pass that recomputes the score; MACIL-SD scorer added: [a_out, v_out], crop mean, index map), `e1_fusion.py --variants hid` (load_hidden: rows aligned like the scores, standardized with the training split; Head with z_t / 5 as an extra input; alpha fixed at 0), launcher `launch/run_round2_cpu.sh <corpus>`; outputs `runs/20261108_astra_review/baseline_scores/<method>/<corpus>/seed_<s>/<split>_hidden.npz`, `runs/20261108_astra_review/e1b/`. MACIL-SD HateMM official checkpoints copied to uoa-lab3 `runs/20261108_astra_review/baseline_models/macilsd/hatemm/`.

#### Round 2 results (3-seed means, test, AP / ROC / within; same test-selected trials as round 1)

**E1b, frozen baseline hidden features + baseline logit + all 8 answers in one head, loss on the fused output** (`e1_fusion.py --variants hid`; `runs/20261108_astra_review/e1b/<corpus>/<baseline>/summary.json`; uoa-lab3 HateMM / HCS 12:11–12:14 (two HateMM extractions were killed for lack of memory and rerun), uoa-lab1 DeHate 12:11–12:28). Hidden features: the extraction pass recomputes each baseline's score and matches the written scores to ≤ 3.4e-6 on every split, every video scored (`baseline_scores/<method>/<corpus>/seed_<s>/hidden_check.json`; dimensions 256 MACIL-SD, 512 Fed-WSVAD and DSANet).

| Corpus (baseline) | Ours, 8 calls | E1 answers only + alpha·baseline (round 1) | E1b joint head | Ours − E1b (per seed AP) |
|---|---|---|---|---|
| HateMM (MACIL-SD) | .702 / .889 / .783 | .662 / .876 / .645 | .612 / .826 / .624 | .090 / .063 / .158 (.081 / .087 / .101) |
| HCS (Fed-WSVAD 3) | .691 / .680 / .563 | .564 / .546 / .497 | .555 / .536 / .503 | .137 / .144 / .060 (.130 / .153 / .126) |
| HCS (DSANet, official branch) | same | .567 / .551 / .492 | .553 / .536 / .539 | .139 / .144 / .024 (.110 / .140 / .165) |
| DeHate (Fed-WSVAD 3) | .279 / .770 / .681 | .231 / .769 / .577 | .188 / .711 / .547 | .091 / .059 / .134 (.051 / .105 / .116) |

The joint head is below the answers-only head of round 1 on all four pairs and only slightly above the baseline alone on DeHate (.174 / .701) and HateMM (.573 / .807). Validation picks epoch 1 on HateMM for all three seeds and epochs 1–2 on DeHate for two of three: on its own training videos the frozen baseline is almost always right (MACIL-SD HateMM video AP .98), so the head learns to rely on the baseline features and this does not carry over to held-out videos. This is now a measurement of the round-1 F1 concern rather than an inference. **E1b held-out** (`--variants hid_val`: the same head trained on the validation videos, where the frozen baseline is out-of-sample as on test; learning rate and epoch by 5-fold cross-validation on validation; one refit on all validation videos; test once; `runs/20261108_astra_review/e1b_val/<corpus>/<baseline>/summary.json`; uoa-lab3 / uoa-lab1 12:29–12:35):

| Corpus (baseline) | E1b held-out | Ours − E1b held-out (per seed AP; per seed ROC) |
|---|---|---|
| HateMM (MACIL-SD) | .603 / .836 / .618 | .098 / .053 / .165 (.111 / .097 / .087; .062 / .052 / .044) |
| HCS (Fed-WSVAD 3) | .578 / .561 / .522 | .113 / .118 / .041 (.101 / .140 / .098; .108 / .127 / .120) |
| HCS (DSANet, official branch) | .527 / .492 / .518 | .165 / .188 / .045 (.158 / .180 / .157; .168 / .217 / .179) |
| DeHate (Fed-WSVAD 3) | .183 / .742 / .561 | .095 / .028 / .120 (.061 / .079 / .146; .010 / .036 / .039) |

Against E1b, training on out-of-sample baseline features raises ROC on HateMM, HCS Fed-WSVAD and DeHate (+.010 / +.025 / +.031); AP is higher only on HCS Fed-WSVAD (+.023, also above the round-1 answers-only head .564); HCS DSANet is lower in both. The validation sets are small (109 / 63 / 559 videos). Across the three learned fusions (E1, E1b, E1b held-out) and the fixed fusion, our method is higher in AP on all three corpora (smallest margins: HateMM .040 to E1, HCS .082 to E1 with features on DSANet, DeHate .030 to the fixed fusion with its weight chosen on test) and in ROC on HateMM and HCS by ≥ .013; on DeHate ROC the best baseline + answers number stays the fixed fusion (.777 vs .770) and the learned answers-only head ties (.769).
