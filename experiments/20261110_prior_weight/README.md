# 20261110_prior_weight: label-free weight of the location prior (revision 7 candidate)

**Candidate = revision 6 + one scalar: the weight c of the network's location prior (per-second logits and node
potentials), estimated without labels on the validation answers and used at validation and test. Question policy,
answer table, network and training loss as revision 6.** Proposal and search protocol: section 8. (The name
"revision 7" was used on 11-03 for a withdrawn candidate, "prior plus label term"; this is a different change.)

User 2026-11-10: autonomous iteration, no report until the goals are met or shown impossible; the prior's weight may
be made part of the method when it is estimated **without labels** from the marginal likelihood of the validation
videos' answers (ruling on rule 3). Goals in force: (a) the model must not use the answers worse than a plain fusion
of the same inputs, pooled AP / ROC ahead of "strongest baseline + equal-window answers"; (b) same checkpoint, same
calls: the tree policy's within not below a single tree level, AP or ROC advantage of at least .01; (c) AP / ROC not
below revision 6.

Everything in sections 1-6 is on the revision-6 checkpoints (nothing retrained; developmental test analysis, rule 10).
Hosts: uoa-lab3 (sc474398; CPU and Slurm GPU), uoa-lab1 (sc474397; Slurm GPU); outputs copied to uoa-lab2
`runs/20261110_prior_weight/`. AP / ROC / within, order HateMM / HCS / DeHate, scores evaluated at 6 decimals.

## 1. Estimator (`src/qtl/prior_weight.py`)

Weights inside the model: P(G = 1 | x) = sigmoid(a g + b), per-second logits c s_t, node potentials c phi_n. Estimated
on the validation videos by maximum marginal likelihood of their cached answers,
P(o | x) = P(G = 1 | x) P(o | G = 1, x) + P(G = 0 | x) P(o | G = 0), answer table and chain fixed. No video label, no
per-second label. Modes: `root` ((a, b) from the root answers), `root_all` (then c from all answers), `joint` (the
three from all answers), `loc` (c only, a = 1, b = 0), `trust` / `trust_len` (c plus two answer scalars, section 6).
The weights apply to the posterior and to the choice of questions (`cpolicy.run_batch(prior_weight=)`).

## 2. What the answers can and cannot identify

`stage_a_weights.py` (`stage_a/<corpus>/summary.json`), the three revision-6 test-selected trials per corpus:

- **(a, b) of the video logit are not identified.** Root-only estimates of a: HateMM 2.0 / .51 / .22, HCS 2.3 / 1.75 /
  74.7, DeHate 1.8 / .48 / 1.3; the root likelihood changes by .002-.08 nats per video between a = 1 and the optimum.
  The labelled regression of the video label on g (not used by the method) gives a = 1.43 / .59 / .25 on HateMM
  validation, .31 / .40 / .73 on HCS, .37 on DeHate. Reason: the network is trained to explain the training answers,
  so g and the answers agree beyond what the label explains; the answers cannot calibrate g against the label.
  The over-confidence of g found in `20261110_answer_likelihood` section 2 therefore stays unaddressed here.
- **c is identified** (all cached answers of the validation videos, tens of answers per video). Over all 60 trials of
  each revision-6 search: median c .16 / .20 / .23, 10-90% range .03-.49 / .04-.56 / .02-.55, positive in all but one trial (HCS, -.002). The
  per-second prior and node potentials of the network predict the held-out answers far worse than the training
  answers they were fitted on.
- Dose diagnostic (`diag_dose.py`, `diag/dose_<corpus>.json`; validation + test nodes grouped by the per-second GT):
  inside hateful nodes the answer does not grow with the hateful fraction on HateMM and DeHate (Spearman -.05 to
  +.10, fully hateful 4-8 s nodes answer lower than partly hateful ones), it does on HCS for long nodes (64+ s:
  Spearman .29; root: .39). A count-dependent answer model is not supported as a common mechanism. Side facts: the
  answer separates hateful from clean nodes inside positive videos with AUC .67 / .68 / .61 at 4-8 s and
  .84 / .74 / .62 at 32-64 s; clean nodes of positive videos answer higher than nodes of negative videos by .9-2.4
  logits at every length.

## 3. Why the three selected trials are the wrong place to compare (and what replaces them)

On the three test-selected trials, `loc` with the unchanged EIG policy gives HateMM .679 / .887 / .798 (revision 6:
.702 / .889 / .783): AP -.023. Those trials were selected for revision 6's own inference (best of 20 per seed), so
every other inference is compared against a maximum. `stage_a_all_trials.py` replays every variant on all 20 trials
of each seed (GPU, Slurm; `all_trials/<corpus>/seed<s>/trial<k>.json`, table `all_trials_table.py`) and reports
(i) the paired difference to revision 6 over the 60 trials, (ii) each variant's own best trial per seed by
(test AP + test ROC) / 2 (rule 7's selection). `r6|none|eig` reproduces the revision-6 search numbers exactly
(.702 / .889 / .783, .691 / .680 / .563, .278 / .770 / .681).

## 4. Result: the location weight

Paired difference to revision 6 over all trials (standard error .001-.003), 8 calls:

| variant | HateMM | HCS | DeHate |
|---|---|---|---|
| `loc`, EIG (greedy, as revision 6) | +.002 / +.004 / +.018 | +.011 / +.011 / +.029 | +.019 / +.022 / +.013 |
| `loc`, budget-aware plan, EIG value | +.008 / +.002 / +.022 | +.012 / +.012 / +.036 | +.018 / +.018 / +.019 |
| `loc`, budget-aware plan, eer value | +.021 / +.004 / +.032 | -.004 / -.006 / +.034 | +.021 / +.018 / +.017 |
| `loc`, greedy eer | +.016 / +.005 / +.023 | -.005 / -.007 / +.027 | +.026 / +.020 / +.007 |
| `loc`, one level (8 equal windows) | +.023 / +.006 / +.047 | -.010 / -.012 / +.032 | +.020 / +.017 / +.020 |
| no weight, budget-aware plan, EIG value | -.001 / -.005 / +.005 | +.004 / +.006 / +.003 | .000 / -.009 / .000 |
| validation table, `loc`, plan (eer) | +.013 / +.003 / +.028 | +.004 / -.001 / +.046 | +.018 / +.023 / +.004 |
| validation table, `loc`, plan (EIG) | +.011 / +.003 / +.026 | +.003 / .000 / +.048 | +.016 / +.023 / +.005 |
| validation table, `loc`, one level | +.013 / +.004 / +.024 | +.001 / -.001 / +.046 | +.018 / +.023 / +.006 |
| validation table, no weight, EIG | -.013 / -.001 / -.012 | -.018 / -.013 / -.002 | +.003 / +.015 / -.004 |

Own best trial per seed (3-seed mean; rule 7's selection applied to each variant):

| variant | HateMM | HCS | DeHate |
|---|---|---|---|
| revision 6 | .702 / .889 / .783 | .691 / .680 / .563 | .278 / .770 / .681 |
| `loc`, EIG | .698 / .889 / .774 | .708 / .692 / .617 | .298 / .793 / .693 |
| `loc`, plan (EIG) | .704 / .893 / .778 | .704 / .696 / .610 | .280 / .781 / .702 |
| `loc`, plan (eer) | .721 / .896 / .819 | .687 / .681 / .603 | .286 / .783 / .702 |
| `loc`, one level | .722 / .899 / .829 | .677 / .667 / .603 | .292 / .784 / .700 |
| naive control (baseline + 8 equal windows) | .707 / .881 / .692 | .693 / .651 / .577 | .243 / .773 / .601 |

- **The label-free location weight improves revision 6 on every corpus and metric with the policy unchanged**
  (paired, 60 trials): AP +.002 / +.011 / +.019, ROC +.004 / +.011 / +.022, within +.018 / +.029 / +.013.
- The weight is what matters: the plan without the weight changes nothing (row 6).
- The validation-fitted three-state table (`answer_calib.py`, needs per-second validation labels) makes the three
  policies agree (rows 7-9) but is not better than the anchored table with the weight; without the weight it is
  worse than revision 6. Not adopted.

## 5. Question policy under the weighted prior

Policies (`cpolicy.run_batch`): `eig` greedy (revision 6); budget-aware plan (`plan="disjoint"`,
`cpolicy._plan_disjoint`: before every call the remaining calls are planned as the pairwise disjoint unasked nodes
with the largest total value, exact max-plus dynamic programme on the tree; the best planned node is asked; nested
nodes only when no disjoint node is left) with value EIG or eer (expected per-second squared-error reduction); `level`
(the 2^d equal windows of one tree depth).

- No policy is best on all corpora. HateMM: plan (eer) and one level are ahead of the EIG policies by .013-.021 AP;
  HCS: the EIG policies are ahead of plan (eer) and one level by .015-.022 AP and .017-.024 ROC; DeHate: all within
  .008.
- What separates the corpora (section 2 side facts and `stage_a_policies.py`): on HateMM the answers of short nodes
  are unreliable and the 8 equal windows are about 18 s long, on HCS (videos about 240 s) the 8 equal windows are
  30 s and the EIG policies ask 11-15 s nodes on part of the video.
- With the validation-fitted table the three policies give the same numbers on every corpus (section 4 rows 7-9):
  with a table that knows how answers behave inside hateful videos, the adaptive tree policy has no advantage over 8
  equal windows at 8 calls.

Per budget (`budget_table.py`, `all_trials/budget_table_loc.json`; `loc` weight, paired over all trials; the single
level at budget B = the 2^d equal windows with 2^d <= B, so it uses 4 calls at B = 6 and 8 at B = 12):

| corpus, B | plan (EIG) vs greedy EIG | plan (EIG) vs level | plan (eer) vs level | plan (EIG) vs revision 6 | plan (eer) vs revision 6 |
|---|---|---|---|---|---|
| HateMM 4 | -.003 / -.006 / +.008 | -.018 / -.009 / -.010 | +.002 / -.002 / -.006 | +.002 / .000 / +.034 | +.022 / +.007 / +.039 |
| HateMM 6 | +.003 / -.003 / +.005 | -.005 / .000 / -.018 | +.011 / +.002 / -.009 | +.006 / +.002 / +.025 | +.021 / +.005 / +.035 |
| HateMM 8 | +.006 / -.002 / +.004 | -.015 / -.004 / -.025 | -.002 / -.002 / -.015 | +.008 / +.002 / +.022 | +.021 / +.004 / +.032 |
| HateMM 12 | +.011 / +.001 / .000 | -.004 / .000 / -.029 | +.003 / +.002 / -.021 | +.019 / +.005 / +.011 | +.026 / +.007 / +.019 |
| HateMM 16 | +.010 / +.002 / .000 | -.006 / -.002 / -.013 | -.001 / -.001 / -.012 | +.021 / +.006 / +.013 | +.026 / +.007 / +.014 |
| HCS 4 | +.004 / +.002 / +.003 | +.021 / +.014 / +.009 | +.007 / +.003 / +.012 | +.015 / +.014 / +.032 | +.001 / +.003 / +.035 |
| HCS 6 | .000 / -.002 / +.004 | +.024 / +.017 / +.009 | +.011 / +.002 / +.007 | +.011 / +.009 / +.029 | -.002 / -.005 / +.027 |
| HCS 8 | +.001 / +.001 / +.007 | +.022 / +.024 / +.004 | +.006 / +.006 / +.003 | +.012 / +.012 / +.036 | -.004 / -.006 / +.034 |
| HCS 12 | +.004 / +.006 / +.010 | +.029 / +.031 / +.012 | +.012 / +.012 / +.011 | +.013 / +.011 / +.039 | -.003 / -.008 / +.038 |
| HCS 16 | +.006 / +.011 / +.014 | +.012 / +.018 / -.007 | .000 / +.004 / -.002 | +.017 / +.016 / +.046 | +.005 / +.001 / +.051 |
| DeHate 4 | +.008 / .000 / +.010 | -.002 / .000 / +.001 | +.002 / -.001 / -.002 | +.020 / +.020 / +.018 | +.024 / +.019 / +.015 |
| DeHate 8 | -.001 / -.004 / +.007 | -.002 / +.001 / -.001 | +.001 / +.001 / -.004 | +.018 / +.017 / +.017 | +.021 / +.017 / +.015 |
| DeHate 16 | -.003 / -.008 / -.003 | -.006 / +.005 / -.002 | -.004 / +.005 / -.003 | +.008 / +.009 / +.017 | +.010 / +.009 / +.016 |

(DeHate at B = 16: the tree policies use 13.0 calls on average, the single level 10.6, because short videos have
fewer than 16 queryable windows of one depth.)

- The two plan values trade the corpora at every budget: the eer plan equals the single level on HateMM (AP / ROC
  within .003, .011 ahead at B = 6 where the level cannot use 2 calls; within .006-.021 lower) and stays at revision 6
  on HCS; the EIG plan is .011-.017 AP ahead of revision 6 on HCS and .015-.018 AP behind the single level on HateMM
  at 4 and 8 calls.
- The EIG plan against the greedy EIG policy (same value, same weight): within .011 at every budget, mostly positive.

Greedy EIG with the stopping rule (qmixE1, threshold set on the validation videos without labels so that their mean
number of calls is the budget; at most 16 calls per video) against the single level at the same mean number of calls
(`stopE1c16`, paired over all trials, `loc` weight):

| corpus | mean 4 calls | mean 8 calls | calls at mean 8 (hateful / other videos) |
|---|---|---|---|
| HateMM | -.010 / +.004 / -.013 | -.017 / -.003 / -.029 | 11.0 / 5.8 |
| HCS | +.015 / +.009 / +.013 | +.017 / +.019 / +.009 | 8.2 / 5.8 |

Spending the calls unevenly over the videos does not change the picture: behind the single level on HateMM, ahead
of it on HCS (there also on within).

## 7. Decision for the candidate

1. **Adopt: the location weight (mode `loc`) with the question policy of revision 6 (greedy EIG).** It is the one
   change with a positive paired difference on every corpus and metric at every tested budget.
2. **Not adopted: the video-logit weights (a, b)** - not identified without labels (section 2).
3. **Not adopted: the budget-aware plan.** With the EIG value its paired gain over the greedy policy at 8 calls is
   +.006 / +.001 / -.001 AP (below the .01 of rule 14(g)), and every budget becomes its own run, which the
   stopping rule (one trajectory per video) does not support. With the eer value it gains .019 AP on HateMM and loses
   .015 AP / .017 ROC to the EIG policies on HCS. The code stays (`cpolicy.run_batch(plan=)`, `train.py` `plan`).
4. **Not adopted: the validation-fitted table and the leak / miss scalars** (sections 4 and 6).

Open after this decision (goal b): under the weighted prior a single tree level (8 equal windows) is ahead of the
tree policies on HateMM (greedy EIG: -.021 AP, -.029 within; EIG plan -.015 / -.025; eer plan -.002 / -.015) and
behind them on HCS (-.021 AP / -.023 ROC against greedy EIG); DeHate within .008.

## 8. Proposal (rule 4) and search protocol (rule 7)

**Mechanism.** The prior network's per-second logits s_t and node potentials phi_n are fitted on the training
videos' answers (loss term -log P(answers | G = Y, x) / n) and predict the answers of unseen videos much worse; the
tree inference nevertheless uses them at full strength, so the answers are under-weighted where they disagree with
the network. Revision 7 multiplies s_t and phi_n by one scalar c inside the model, P(y | G = 1, x) proportional to
exp(c s . y) chain(y) prod_n exp(c phi_n z_n), and estimates c as a hyper-parameter of the prior by maximum marginal
likelihood of the validation videos' cached answers (type-II maximum likelihood; `prior_weight.fit(mode="loc")`):
no video label and no per-second label. The same c enters the posterior (exact tree inference), the EIG of every
candidate question and every statistic of the stopping rule. During training c is re-estimated after every epoch
for that epoch's network, and the validation criterion that picks the checkpoint is computed under it; the selected
epoch's c is stored with the checkpoint and used on test. The training loss is unchanged (c = 1 on the training
videos, where the network's outputs are in-sample).

**Source.** Own mechanism from the test error analysis of 11-10 (sections 2-4 and
`experiments/20261110_answer_likelihood/README.md` sections 2 and 4). Related practice outside the task: evidence
maximisation of prior hyper-parameters (empirical Bayes), tempered priors / power priors.

**Inputs.** None new: the cached answers of the validation videos (already extracted for all nodes) and the
revision-6 inputs. No extra VLM call at test: c is a constant of the deployed model.

**User ruling 2026-11-10.** "规则 3 禁止把 calibration 当后处理。估计方式可以不用标签(用 validation 视频的回答的边际似然)" -
"我同意这里": estimating the prior's weight without labels from the validation answers' marginal likelihood is not the
post-processing rule 3 forbids.

**Expected (from section 4, revision-6 checkpoints).** AP / ROC not below revision 6 on HateMM, higher on HCS and
DeHate (paired +.011 / +.011 and +.019 / +.022), within higher on all three (+.018 / +.029 / +.013).

**Search (rule 7), declared before the search.** `search.py`: the revision-6 configuration (`soft_both`,
`node_prior`, `anchored_shared`, `length_term`) plus `prior_weight: "loc"`; the same four scalars and ranges as
revisions 5 / 6 (`src/qtl/search.py:sample`: lr 1e-4..1e-3 log, lamda_cma .5..2, dropout .1..5, lr_answer 1e-4..1e-1
log), Optuna TPE, sampler seed = training seed, seeds 234 / 2025 / 3407, three corpora, objective (test AP + test
ROC) / 2 at fixed 8 calls; checkpoint by validation (AP + ROC) / 2 at 8 calls under that epoch's c. Trial budget by
rule 7 from each seed's measured first trial (20 when at most 1 hour, else 5; `budget.json`); DeHate runs at most
two seeds per GPU, because three processes on one GPU took 3308-3407 s per trial in revision 6 and the fit of c adds
about 2-4 s per epoch. Launch `launch/sbatch_search.sh <corpus> ["<seeds>"]`, output
`runs/20261110_prior_weight/search/<corpus>/seed<s>/`. The first 10 trials of a seed draw the same hyper-parameters
as revision 6's (seeded TPE start-up), so they can also be compared pairwise.

"Without labels" means: no video label and no per-second label enters the estimate of c. The DeHate validation
cohort itself excludes hateful videos without a usable span (frame protocol), as everywhere in this project.

**Pre-registered reading (3-seed means of the test-selected trials).**
- Q1 (goal c): AP and ROC not below revision 6 by more than .005 on any corpus (revision 6: .702 / .889,
  .691 / .680, .278 / .770).
- Q2 (rule 8): the main gate on all six numbers.
- Q3: within above revision 6 on all three corpora (.783 / .563 / .681).
- Reported besides: validation-selected trials, the mean over all trials, fixed 0-32 calls, the fitted c per trial
  (median, minimum, number of trials with c < .05) and its marginal-likelihood gain (`nll_all_before / after`), the
  comparison with the naive control (.707 / .881, .693 / .651, .243 / .773; by the one rule of section 10
  .707 / .881, .696 / .653, .243 / .777 - both are given) and with the single level (goal b).

**Fixed before the search (review remarks, `REVIEW_RULE4_R7.md` 5.1, `REVIEW_RULE6_R7.md`).**
- c is used as estimated: no bound, no clipping (also when negative or above 1).
- Trial budget: whatever `src/qtl/search.py` writes from the first trial of each seed (20 when at most 3600 s, else
  5); the measured seconds are archived in `budget.json` and reported. No change after seeing the time.
- Ablation of the weight (rule 14(g)), all on the test-selected trials of this search: (i) the same checkpoints with
  c = 1 (the weight removed at inference; the checkpoint was still chosen under c); (ii) revision 6's own search
  (weight removed from inference and from checkpoint choice); (iii) the 2 x 2 split "questions chosen under c or 1"
  x "posterior under c or 1" (`run_batch(prior_weight=(1, 0, c), policy_prior_scale=1 / c)` = revision-6 questions
  with the weighted posterior; `run_batch(prior_weight=None, policy_prior_scale=c)` = weighted questions with the
  revision-6 posterior).
- If the weight does not reach rule 14(g) (AP or ROC drop of at least .01 when removed, on every corpus) it stays in
  the method and is not claimed as a novelty mechanism.
- Diagnostic references (not the method): c estimated on the training answers (in-sample), a hand-set c = .5, and c
  fitted with the validation labels.
- Trials 0-9 of each seed retrain the networks of revision 6's trials 0-9 (same seeded hyper-parameters, same loss);
  only the selected epoch and c differ. The search is therefore not evidence independent of section 4.
- Wording: c concerns the location prior only; the video logit's over-confidence is not addressed. "Without labels"
  refers to the estimate of c; the checkpoint choice still uses the validation per-second labels, the answer table
  the training video labels. Estimating c needs the full-tree answers of the validation videos (offline, no extra
  call on test videos). The estimator is standard empirical Bayes, of the same form as a temperature on a
  sub-network's logits; what differs from output calibration is where it acts (the prior inside the inference, hence
  the choice of questions).

## 6. Label-free answer scalars (`trust`, `trust_len`): no effect

Leak rho (a clean node of a hateful video answers like a hateful node with probability rho) and miss eta (a hateful
node answers like a clean one with probability eta, optional log-length slope), estimated with c by the same marginal
likelihood (`prior_weight.TrustTable`). Fitted rho and eta are 0 (median over 60 trials, HateMM and HCS); with the
length slope the paired differences to `loc` are within .004. The answers alone do not reveal that short hateful
nodes are missed: the model explains a "No" by "the node is clean". Not adopted.

## 9. Goal (b) on HateMM: no tree policy is ahead of 8 equal windows, and why

Same checkpoint, same weight (`loc`), paired over the 60 trials, policy minus single level at the same budget
(AP / ROC / within; `budget_table.py`, `all_trials_table.py`):

| policy | HateMM 4 | HateMM 8 | HateMM 16 | HCS 8 | DeHate 8 |
|---|---|---|---|---|---|
| greedy EIG (candidate of section 7) | -.015 / -.003 / -.019 | -.021 / -.002 / -.029 | -.016 / -.005 / -.013 | +.021 / +.023 / -.003 | -.001 / +.005 / -.008 |
| plan, EIG value | -.018 / -.009 / -.010 | -.015 / -.004 / -.025 | -.006 / -.002 / -.013 | +.022 / +.024 / +.004 | -.002 / +.001 / -.001 |
| plan, eer value | +.002 / -.002 / -.006 | -.002 / -.002 / -.015 | -.001 / -.001 / -.012 | +.006 / +.006 / +.003 | +.001 / +.001 / -.004 |
| greedy eer | | -.007 / .000 / -.024 | | +.005 / +.004 / -.005 | +.005 / +.003 / -.012 |
| greedy EIG + stopping rule, same mean calls | -.010 / +.004 / -.013 | -.017 / -.003 / -.029 | | +.017 / +.019 / +.009 | |
| validation table: plan (EIG) vs level, both with that table | | -.004 / -.001 / +.001 | | +.002 / +.001 / +.004 | -.004 / .000 / -.001 |

On the three selected trials of section 5 the cover policies (first a disjoint cover chosen by EIG, with or without
the root) are also behind the single level on HateMM (-.010 / -.004 / -.021 and -.009 / .000 / -.019).

- **HateMM: 8 equal windows (about 16 s each) are the best use of 8 calls among everything tested** (the 15
  policies of `20261109_tree_within` and those above). The tree policies that come closest are those that reduce to an equal
  partition (plan with the eer value: AP / ROC within .002, within .006-.015 lower).
- **More calls do not help either** (all-trial means): single level with 8 calls .693 / .883 / .811; the best tree
  policy with 12 calls .696 / .885 / .790; with 16 calls .699 / .886 / .785 (16 equal windows .700 / .887 / .797). No
  policy with up to twice the calls is .01 AP or ROC above the 8 equal windows, and within falls when shorter nodes
  are added. An AP or ROC advantage of .01 over the single level at equal calls is therefore not available on
  HateMM from these answers.
- Why (data facts, `gt` of the test videos; sections 2 and 5): (1) hate is dense in HateMM's hateful videos: median
  .70 of the seconds (mean .61, 71 % of the videos above .38, 2.1 hateful runs per video), and the answer to a node
  does not grow with its hateful fraction, so a Yes on a long node does not tell which half holds the hate -
  asking coarse nodes first buys no localisation; (2) the network's location prior carries little information on
  held-out videos (c median .16), so there is no basis for giving some parts of the video finer questions than
  others; (3) nodes shorter than about 16 s are answered unreliably on HateMM (hateful vs clean nodes inside hateful
  videos: AUC .67 at 4-8 s, .84 at 32-64 s), so going below the 8 windows somewhere costs more than it returns.
- **HCS: the tree policies are ahead of the single level** (AP / ROC +.012 to +.031 at every budget; with the stopping
  rule also within +.009). HCS is as dense (median .65) but its videos are 240 s: the 8 equal windows are 30 s long
  and long HCS nodes answer lower when little of them is hateful (section 2), while the EIG policies ask 11-15 s
  nodes where the prior points.
- DeHate: tree and level within .01 at 4 and 8 calls; ROC +.013 for the greedy tree at 16 (with 13.0 against 10.6
  calls).
- With the validation-fitted table (last row) tree and level are equal on all three corpora.

Goal (b) reads: tree within not below any single level and a tree AP or ROC advantage of at least .01. It holds on
HCS, not on HateMM (neither half), and DeHate is a tie. On HateMM the second half is not reachable with the cached
answers (second bullet); the first half is reachable only by policies that coincide with the level.

## 10. First level of the question policy, chosen on validation (stage A, then proposal)

Sections 5 and 9: which way of spending the calls is better is a property of the corpus's answers, and the two ways
are the ends of one family of policies on the same tree and the same posterior: **the 2^d equal windows of tree
depth d are asked first (in EIG order), then the greedy EIG policy continues over all nodes** (`cpolicy.run_batch`
`first=`, `qtree.depth_nodes`). d = 0 is the greedy policy of revisions 2-6 (no forced first level; it does not
mean "the root first"); with a budget of exactly 2^d calls the policy is the single level; d in between starts with
the halves or the quarters. On a video too short for all windows of depth d to be queryable (nodes of at least 4 s)
the first level falls back to the deepest fully queryable level and the remaining calls are spent greedily, so on
short videos "d = 3" is not strictly "8 equal windows": HateMM 7.63 calls against 7.13 for the level-only policy of
sections 5 and 9, DeHate 7.26 against 6.56, per-trial within differences up to .006; HCS has no such video. d (how
coarse the first pass over the video is) is one integer hyper-parameter of the policy at a budget and can be chosen
like the checkpoint: on validation.

Stage A (`stage_a_all_trials.py` policies `first<d>` and variants `...@val` = the same replay on the validation
videos; CPU, uoa-lab1, from the branch checkout `~/Retrieval-hate-r7-first-level`; `policy_select.py`,
`all_trials/policy_select_b8_first.json`): for every revision-6 trial, under the `loc` weight, the d in 0..3 with
the largest validation (AP + ROC) / 2 at 8 calls; test numbers of the chosen d.

| 8 calls | HateMM | HCS | DeHate |
|---|---|---|---|
| chosen d = 0 / 1 / 2 / 3 (share of the 60 trials) | 5 / 3 / 2 / 90 % | 50 / 32 / 10 / 8 % | 27 / 32 / 32 / 10 % |
| d with the best test value = 0 / 1 / 2 / 3 (share) | 13 / 3 / 5 / 78 % | 57 / 35 / 8 / 0 % | 15 / 37 / 32 / 17 % |
| chosen, all-trial mean | .693 / .884 / .809 | .681 / .664 / .585 | .247 / .774 / .680 |
| chosen vs revision 6, paired | +.023 / +.006 / +.045 | +.007 / +.006 / +.031 | +.020 / +.021 / +.011 |
| chosen vs d = 0 (greedy, section 7), paired | +.021 / +.002 / +.027 | -.004 / -.005 / +.002 | +.002 / .000 / +.001 |
| chosen vs d = 3 (single level), paired | .000 / +.001 / -.002 | +.017 / +.018 / -.001 | +.001 / +.005 / -.007 |
| chosen, best trial per seed | .722 / .898 / .831 | .703 / .684 / .617 | .304 / .792 / .694 |
| d = 0, best trial per seed | .698 / .889 / .774 | .708 / .692 / .617 | .298 / .793 / .693 |
| d = 3, best trial per seed | .722 / .898 / .831 | .677 / .667 / .603 | .291 / .784 / .701 |
| revision 6, best trial per seed | .702 / .889 / .783 | .691 / .680 / .563 | .278 / .770 / .681 |
| naive control (rule below) | .707 / .881 / .692 | .696 / .653 / .577 | .243 / .777 / .601 |

Naive control, one rule for the three corpora (`runs/20261109_tree_within/naive_questions/<corpus>.json`,
`methods.<baseline>.uniform8`, the row `best_alpha_on_test`, 3-seed mean): the strongest baseline plus the answers
of 8 equal windows, fusion weight alpha taken as the best on test, and where two baselines were run the one with
the higher (AP + ROC) / 2. HateMM MACIL-SD alpha .5 (seed std .004 / .005, 7.13 calls); HCS DSANet alpha .1
(.001 / .001, 8.00 calls; Fed-WSVAD3 alpha 0 gives .693 / .651); DeHate Fed-WSVAD3 alpha .4 (.016 / .003, 6.56
calls; alpha .5 gives .243 / .773). Earlier sections quote .693 / .651 for HCS and .243 / .773 for DeHate.

- Validation returns the level (d = 3) on HateMM and a shallow start (d = 0 or 1, equivalent on test) on HCS; on
  DeHate the four depths are within .007 of each other and the choice is immaterial.
- On the revision-6 checkpoints the chosen depth is ahead of the naive control on all six AP / ROC numbers
  (+.015 / +.017, +.007 / +.031, +.062 / +.016) and not below revision 6 (+.020 / +.009, +.011 / +.004,
  +.026 / +.023; the HCS ROC difference is inside the .005 noise band of rule 8); within +.049 / +.054 / +.013
  against revision 6.
- The cost of choosing on validation instead of knowing the better depth: HCS -.004 / -.005 in the all-trial mean
  against d = 0 and -.005 / -.008 on the best trial per seed (.703 / .684 against .708 / .692); nothing on HateMM
  and DeHate. The chosen d is the d with the best test value in 77 % / 38 % / 18 % of the trials (4 candidates).
- A binary choice {d = 0, single level} gives the same picture (`policy_select_b8_eig_level.json`: .722 / .899 /
  .829, .704 / .686 / .611, .298 / .793 / .693); adding the plan policies as candidates does not add
  (`policy_select_b8_3.json`, `policy_select_b8_5.json`).
- Continuing greedily after the level does not help on HateMM: level of 8 then 8 greedy calls .688 / .884 / .791
  (all-trial mean, 16 calls) against 16 equal windows .700 / .887 / .797 and against the 8 windows alone
  .693 / .883 / .811. So d is chosen per budget (candidates d <= floor(log2 B)).
- The same choice at 4 and 16 calls (`policy_select_b4_first.json`, `policy_select_b16_first.json`; 60 trials):

  | | HateMM | HCS | DeHate |
  |---|---|---|---|
  | 4 calls, chosen d = 0 / 1 / 2 | 10 / 18 / 72 % | 42 / 50 / 8 % | 48 / 52 / 0 % |
  | 4 calls, chosen vs d = 0, paired | +.011 / +.003 / +.015 | -.003 / -.001 / -.002 | +.005 / +.001 / .000 |
  | 4 calls, best trial per seed: chosen (d = 0) | .706 / .892 / .823 (.692 / .889 / .801) | .696 / .692 / .597 (.701 / .693 / .604) | .292 / .779 / .684 (.280 / .779 / .685) |
  | 16 calls, chosen d = 0 / 1 / 2 / 3 / 4 | 3 / 3 / 0 / 0 / 93 % | 27 / 28 / 13 / 3 / 28 % | 33 / 13 / 20 / 32 / 2 % |
  | 16 calls, chosen vs d = 0, paired | +.015 / +.005 / +.013 | -.002 / -.004 / +.006 | +.001 / .000 / .000 |
  | 16 calls, best trial per seed: chosen (d = 0) | .717 / .897 / .793 (.708 / .892 / .802) | .704 / .694 / .605 (.710 / .700 / .608) | .300 / .797 / .713 (.300 / .798 / .713) |

  HateMM takes the level at every budget (d = log2 B) and gains .011-.021 AP over the greedy policy; HCS loses
  .002-.004 AP / ROC in the all-trial mean (.005-.008 on the best trial per seed) to the noise of the choice among
  depths that differ little there; DeHate is neutral.

### Proposal (rule 4): revision 7 with a validated first level

**Mechanism.** Revision 7 (section 8: label-free location weight c; one tree, one answer table, one posterior,
training loss and checkpoint choice unchanged) with one more hyper-parameter of the question policy, the depth d of
its breadth-first start. After training, for every reported budget B, the trained model is run on the validation
videos with every d in 0..floor(log2 B) under its stored c, and the d with the largest validation (AP + ROC) / 2 at
B calls is used on test (ties: the smaller d). `train.py` config `first_level: "validated"`
(`Evaluator.run(first_depth=)`, `evaluate`; `summary["fixed"][B]["first_level"]` and `["depths"]` = every candidate
depth's validation and test numbers). The rule and the candidate set are the same on every corpus; the returned d
is an outcome and is reported per corpus and budget.

**Why a validated hyper-parameter and not one better policy.** Section 9: the best first pass differs by corpus for
reasons the model cannot see without per-second labels (how short nodes and diluted long nodes are answered inside
hateful videos); the label-free leak / miss scalars (section 6), the validation-fitted table (section 4) and the
budget-aware plans (section 5) did not make one policy best everywhere. The validation per-second labels are already
used to choose the checkpoint; the 09-29 ruling allows them where they do not enter a gradient.

**Rule 13: the main agent's reading, pending the user's ruling.** Rule 13 allows scalar hyper-parameters to differ
between corpora when one pre-declared validation rule selects them; d is one integer with the same candidates and
the same rule everywhere, closest to the "时间窗长度" the rule lists. Against this reading: at 8 calls d = 0 and
d = 3 differ by whether EIG chooses the nodes at all (at d = 3 it only orders them), which has the effect of the
"模块开关不同" the rule forbids, the switch being set by the validation rule instead of by hand
(`REVIEW_RULE4_R7_FIRST_LEVEL.md` section 3). Until the user rules, the candidate is not reported as meeting rules
14(c) / 14(e). d is not in the Optuna search space and is not chosen on test: it is a second validation selection
inside the trial, after the checkpoint (rule 7 names only the checkpoint; reported as such under rule 14(d)).

**Source.** Own analysis (sections 5, 9). Related practice outside the task, to be cited and delimited in a paper:
FV-Action (arXiv 2608.08315, section 5.5: uniform probes beat content-chosen ones at a fixed budget when no reliable
prior says where the event is); group testing (individual / non-adaptive designs are optimal once positives are
dense; two-stage designs with a non-adaptive first stage); noisy twenty questions (Jedynak, Frazier, Sznitman 2012:
the non-adaptive dyadic policy is as good as the adaptive bisection under entropy loss); static against dynamic
policies in active feature acquisition. Inside the task the closest item is TANDEM (arXiv 2601.11178), which
switches temporal segmentation on or off per dataset by hand; here one pre-declared validation rule selects d.

**Inputs.** None new. No extra training cost; after training, floor(log2 max_calls) = 5 more policy runs on the
validation and test videos.

**What is and is not claimed.** Claimed: one posterior model on a temporal tree with a label-free prior weight; the
depth of the policy's first pass is selected on validation; on HCS the selected policy is the adaptive one and is
ahead of equal windows. Not claimed: that the adaptive policy beats equal windows on every corpus - on HateMM the
selected depth is expected to be the single level, and section 9 shows that no tree policy does better there. The
depth is not claimed as a novelty mechanism (it is a hyper-parameter; rule 14(g) is read for the weight c).
- The HateMM gain over revision 6 comes from d, not from c (stage A: c at d = 0, paired AP +.002 / ROC +.004; the
  chosen d over d = 0, paired AP +.021), and is reported so.
- Where the chosen d equals log2 B, EIG does not choose the nodes (only their order). In this candidate the EIG
  question choice therefore cannot show a rule-14(g) drop on HateMM and cannot be listed as a novelty mechanism that
  holds on the three corpora. The proposal does not cause this (section 9: at d = 0 the 8 equal windows are .021 AP
  above the EIG questions on HateMM) but it fixes it into the method. The tree is the core novelty the user named on
  11-09; this goes to the user with the report.
- Stopping rule: the threshold rules (`adaptive`, `adaptive_voi`; one trajectory per video) are evaluated on the
  greedy trajectory (d = 0) as before. On a corpus whose chosen d is not 0, the fixed-budget configuration (main
  table: 8 calls of the chosen d) and the stopping rule (fewer-calls curve) are two different question policies. The
  cost of the stopping rule (STATUS goal 1: AP / ROC at most .01 below fixed 8 calls) is reported against both
  references, fixed 8 calls of d = 0 (the same trajectory, as in revisions 5 / 6) and fixed 8 calls of the chosen
  d; against the second it is expected not to hold on HateMM (section 5: the greedy policy with the stopping rule
  at a mean of 8 calls is .017 AP below the level). The stopping rule is not changed in this round.

**Expected (stage A above).** AP / ROC above revision 6 and above the naive control on the three corpora; the HateMM
gain comes from d = 3, the HCS / DeHate gains from the weight c.

**Search (rule 7), declared before the search.** `search_first_level.py`: the revision-7 configuration of section 8
(`prior_weight: "loc"`) plus `first_level: "validated"`; same four scalars and ranges, Optuna TPE, sampler seed =
training seed, seeds 234 / 2025 / 3407, three corpora, objective (test AP + test ROC) / 2 at 8 calls of the depth
chosen on validation at 8 calls; checkpoint as in section 8 (validation (AP + ROC) / 2 at 8 calls of the greedy
policy under that epoch's c). Trial budget from each seed's measured first trial; DeHate at most two seeds per GPU.
Launch `launch/sbatch_search_first_level.sh <corpus> ["<seeds>"]`, output
`runs/20261110_prior_weight/search_first_level/<corpus>/seed<s>/`.

**Relation to the section-8 search (after the reviews, `REVIEW_RULE4_R7_FIRST_LEVEL.md` 5.1 / 5.2, and the
correction of section 11).** Both searches start from trial 0 with the corrected search for c (section 11) and run
from the main checkout (the branch is merged). The section-8 search that was running with the old search for c was
cancelled on 2026-11-10 23:02 (HateMM 6 finished trials per seed, DeHate seeds 234 / 2025 2 trials each, HCS and
DeHate seed 3407 not started); its outputs are kept in `runs/20261110_prior_weight/search_before_fit_fix/` and are
not used. Order on each machine: this search first, then the section-8 search, which runs to its registered trial
counts and is read as declared in section 8. If the user rules d outside rule 13 the section-8 search is the
candidate; otherwise it is the d = 0 ablation whose hyper-parameters were proposed from its own objective.
Machines (Slurm jobs submitted 2026-11-10 23:05): this search on uoa-lab3 HateMM, HCS, DeHate seeds 234 / 2025
(jobs 2139-2141) and on uoa-lab1 DeHate seed 3407 (2135); the section-8 search on uoa-lab3 DeHate seeds 234 / 2025
(2142) and on uoa-lab1 HateMM, HCS, DeHate seed 3407 (2136-2138). uoa-lab1's GPU was taken by another user's job
(up to one day) at the moment the old jobs were cancelled, so HateMM and HCS of this search went to uoa-lab3.
DeHate never runs with the launcher's default of three seeds on one GPU (`REVIEW_RULE6_R7_FIRST_LEVEL.md`
estimates 2840-3160 s per trial for two seeds and more than 3600 s for three).
Every trial of this search also holds `depths["0"]` at every budget: this row is called **the fixed d = 0 ablation
row**, not the section-8 candidate's search result (trials 10-19 are proposed by TPE from the chosen-depth
objective). It is reported in three readings: its own best trial per seed, the paired difference over all trials,
and the paired difference over trials 0-9 only (hyper-parameters independent of either objective, as long as
neither search restarted before trial 10). Reproduction check: trials 0-9 of the section-8 search (same
hyper-parameters, same training) against `depths["0"]` at 8 calls of the same trial numbers here; differences are
reported.

**Fixed before the search (review remarks).**
- The checkpoint is chosen by the greedy policy's validation criterion (d = 0), then d; no joint choice of
  (epoch, d) after the launch. No minimum margin is added to the choice of d (ties: the smaller d).
- Trial budget: each trial runs 10 more 32-call policy passes (5 depths x validation and test; estimated 190-460 s
  instead of 75-90 s on DeHate), the first trial included, and the first trial sets the trial count (20 when at most
  3600 s, else 5). No change after seeing the time; if a seed gets 5 trials, the comparison with revision 6
  (20 trials) says so.
- `summary["test_runs"]` and `silent_group` hold the trajectory of the chosen depth at 8 calls (no VOI record when
  d is not 0); `metrics.json` does not name d, `summary["first_level"]` does.
- Per-budget choice: every budget takes one of at most 6 fixed trajectories (no rerun per budget), so neighbouring
  points of a budget curve can come from different d; curves mark each point's d.

**Pre-registered reading (3-seed means of the test-selected trials).**
- R1 (goal a): AP and ROC above the naive control on the three corpora, the control taken by the one rule above
  (.707 / .881, .696 / .653, .243 / .777). "Above" = a lead not smaller than the larger of the two seed standard
  deviations and at least .005 (rules 8 / 14(b)); smaller leads are written "not below". `test_mean_calls` is
  reported next to the control's 7.13 / 8.00 / 6.56 calls.
- R2 (goal c): AP and ROC not more than .005 below revision 6 on any corpus (.702 / .889, .691 / .680,
  .278 / .770), as Q1 of section 8.
- R2b (rule 8): the main gate on all six numbers against `docs/duplex/OFFICIAL_VAL_RESULTS.md`, read separately.
- R3 (goal b, first half): at 8 calls on the test-selected trials, within of the chosen depth not more than .005
  below `depths["3"]` and not more than .005 below `depths["0"]`. Two references for "the single level" are
  reported: `depths["3"]` (level first, greedy fill on short videos) and the level-only policy of sections 5 / 9
  (replay `level` on the same checkpoints with the stored weight). On a trial whose chosen d is 3 the first
  comparison is a policy against itself: it is reported as "the selected policy is the single level; goal (b) is
  not met on this corpus", not as "the first half holds". Likewise for a chosen d = 0 against the greedy policy.
- R4: the chosen d per corpus and budget (share of trials), and how often it is the d with the best test value;
  the test numbers when d is instead taken as the best on test per trial are reported as a reference, not used.
  The cost of the choice is given in both readings (all-trial paired difference and best trial per seed, against
  each fixed d).
- Ablation of the choice: on the same trials, every fixed d (`depths`), with each fixed d's own best trial per seed.
- Rule 14(g) for the weight c: (1) on the candidate's own configuration - the test-selected trials of this search,
  same checkpoint with c = 1, both with d chosen again on validation under c = 1 and with d kept (replays
  `r6|none|first<d>` and `...@val` with `--src runs/20261110_prior_weight/search_first_level`; in these replays the
  column "r6" is the revision-7 checkpoint without the weight, not revision 6); (2) the section-8 ablations
  (i)-(iii) on the section-8 search.
- Goal (b), second half (AP / ROC advantage of at least .01 over the single level): reported per corpus from
  `depths`; expected on HCS only (section 9).

**Reviews.** Rule 4: GO (`REVIEW_RULE4_R7_FIRST_LEVEL.md`; d passes only as an unclaimed hyper-parameter of the
reviewed candidate; rule 13 needs the user's ruling; its 8 required changes are in this section). Rule 6: PASS
(`REVIEW_RULE6_R7_FIRST_LEVEL.md`; no bug; remarks 1-5 and 7 are in this section and in the tools).

## 11. Correction: the search for c (2026-11-10, found by the code review)

`REVIEW_RULE6_R7_FIRST_LEVEL.md` remark 8: on 2 of the 180 revision-6 trials the fitted c was exactly 1 with an
unchanged likelihood. Scan of the validation negative log-likelihood per video over c (uoa-lab1 CPU, revision-6
checkpoints, `prior_weight._nll_all`; S = per-second logits of the validation videos):

| HCS seed 3407 | S mean / sd | gradient at c = 1 | c = 0 | .003 | .01 | .1 | .2 | 1 | 3 | old fit |
|---|---|---|---|---|---|---|---|---|---|---|
| trial 7 (ordinary) | .01 / .18 | 157 | 189.55 | 189.54 | 189.52 | 189.30 | 189.22 | 190.83 | 193.66 | c = .20 |
| trial 8 | 5.36 / .89 | -2e-6 | 189.47 | 189.41 | 190.59 | 197.93 | 198.04 | 198.05 | 198.05 | c = 1 (not moved) |
| trial 9 | 5.30 / .86 | -9e-7 | 189.40 | 189.36 | 190.45 | 197.93 | 198.05 | 198.05 | 198.05 | c = 1 (not moved) |
| trial 10 | 2.93 / .44 | 1e-3 | 189.52 | 189.43 | 189.74 | 197.09 | 198.19 | 198.28 | 198.28 | c = .003 |
| trial 14 | 4.08 / .77 | -3e-4 | 189.93 | 190.00 | 190.48 | 197.72 | 198.30 | 198.36 | 198.36 | c = 2.31 (flat part) |
| trial 17 | 4.21 / .80 | -3e-4 | 189.55 | 189.48 | 190.20 | 198.09 | 198.33 | 198.38 | 198.38 | c = 2.26 (flat part) |

- Cause: networks trained with a high learning rate (about 1e-3) output per-second logits that are all large; the
  weighted prior then puts its whole mass on "every second hateful" for every c above about .2, the likelihood is
  flat there, and L-BFGS started at c = 1 does not move or drifts along the flat part, although c near 0 (the
  location prior switched off) has a much higher likelihood. The estimator did not return the maximum-likelihood c
  on these trials. HateMM trials are not saturated (seed 234 trials 0-3: one minimum between c = .03 and .3,
  gradient at 1 of 90-170).
- Correction (`src/qtl/prior_weight.py`, `C_GRID`, `_fit_c`, `fit(start="grid")`): the likelihood is evaluated on 13
  grid points of c (0 to 3) and L-BFGS starts from the best one (ties: the point closest to 1). The estimator's
  definition (maximum marginal likelihood of the validation answers, no label, no bound on c) is unchanged.
  Confirmed by the reviewer (`REVIEW_RULE6_R7_FIRST_LEVEL.md` section 9): `start="one"` reproduces the old search
  exactly; on ordinary priors both starts give the same c to five decimals; cost 13 likelihood evaluations per
  epoch (estimated +60-130 s per DeHate trial).
- Scope: the fit runs every epoch, so every saturated epoch of every trial now gets c near 0, its validation
  criterion changes and the selected epoch can change. Both searches therefore restart from trial 0 (section 10).
- Stage A of sections 4, 5, 9 and 10 used the old search (weight mode `loc` in the trial files). Replays with the
  corrected search are the weight mode `locg` (`stage_a_all_trials.py`); `stage_a_policies.py` and
  `stage_a_weights.py` would use the corrected search if run again, their existing outputs come from the old one.

Effect on the revision-6 checkpoints (replays with weight mode `locg`, 60 trials per corpus, 8 calls, CPU on
uoa-lab1; `all_trials/policy_select_b8_first_locg.json`):

- Trials whose c changes by more than .02: HateMM 0, HCS 4 (the table above; within .51-.53 becomes .63-.65, trial
  8 becomes .716 / .688 / .648), DeHate 2 (seed 2025 trial 19: c 103.6 -> .013, .178 / .688 / .522 ->
  .232 / .783 / .673; seed 234 trial 6: c 4.80 -> .056, .190 / .719 / .615 -> .235 / .779 / .688). Every other
  trial keeps its c and its metrics to three decimals.
- c under the corrected search: median .16 / .17 / .21; range .006 to .83, -.001 to 1.53, .005 to 1.46; c below
  .05 in 13 / 12 / 16 of the 60 trials.
- Greedy EIG, all-trial mean, corrected minus old search: .000 / .000 / .000, .000 / +.001 / +.008,
  +.002 / +.003 / +.004. Paired to revision 6: +.002 / +.004 / +.018, +.012 / +.013 / +.037, +.020 / +.024 / +.014.
- First level chosen on validation (d in 0..3), corrected search:

  | 8 calls | HateMM | HCS | DeHate |
  |---|---|---|---|
  | chosen d = 0 / 1 / 2 / 3 | 5 / 3 / 2 / 90 % | 50 / 30 / 10 / 10 % | 27 / 33 / 30 / 10 % |
  | chosen, all-trial mean | .693 / .884 / .809 | .680 / .663 / .592 | .249 / .777 / .684 |
  | chosen vs revision 6, paired | +.023 / +.006 / +.045 | +.006 / +.006 / +.038 | +.022 / +.024 / +.015 |
  | chosen vs d = 0, paired | +.021 / +.002 / +.027 | -.006 / -.007 / +.001 | +.002 / .000 / +.001 |
  | chosen, best trial per seed | .722 / .898 / .831 | .703 / .684 / .617 | .304 / .792 / .694 |
  | d = 0, best trial per seed | .698 / .889 / .774 | .711 / .695 / .618 | .298 / .793 / .693 |
  | d = 3, best trial per seed | .722 / .898 / .831 | .677 / .667 / .603 | .291 / .784 / .701 |

- The readings of sections 4, 5, 9 and 10 stand (HateMM is unchanged trial by trial). On HCS the greedy policy's
  best trial per seed rises to .711 / .695, and the cost there of choosing d on validation is -.006 / -.007 in the
  all-trial mean and -.008 / -.011 on the best trial per seed.

First trial of the restarted first-level search (HateMM, uoa-lab3, job 2139): 905 s per trial with three seeds on
the GPU, so 20 trials per seed. Checks on trial 0: `fixed["8"]["depths"]["0"]["val"]` equals the validation
metrics of the selected epoch in `history` on the three seeds; seed 234 reproduces trial 0 of the cancelled
section-8 search exactly at d = 0 (selected epoch 32, c .531, the same c at every epoch, test .6811 / .8864 /
.8027), i.e. the correction does not change a HateMM trial without a saturated epoch. Chosen d at 8 calls is 3 on
the three seeds (at 16 calls 4, at 32 calls 5, at 4 calls 2 / 0 / 1); test at the chosen d .703 / .893 / .817,
.697 / .884 / .824, .686 / .880 / .819 against d = 0 .681 / .886 / .803, .682 / .882 / .790, .678 / .883 / .798.

## 12. Result of the first-level search (section 10)

Source: `runs/20261110_prior_weight/search_first_level/<corpus>/` (evaluator outputs per trial; copied back to the
local machine), read by `summarize_first_level.py` into `search_first_level/summary_<corpus>.json`. The objective
of the search is the test (AP + ROC) / 2 at 8 calls of the depth chosen on validation (developmental upper bound,
rule 10). AP / ROC / within.

### HateMM (uoa-lab3, job 2139, 3 seeds x 20 trials, 905 s per trial; finished 2026-11-11 04:07)

| 8 calls | AP / ROC / within |
|---|---|
| chosen d, test-selected trials (16, 11, 15; d = 3 on the three; c .17 / .12 / .11; 7.63 calls) | **.723 / .898 / .826** (seed std .001 / .001 / .006) |
| naive control (MACIL-SD + 8 equal windows, alpha .5, 7.13 calls) | .707 / .881 / .692 (std .004 / .005 / .003) |
| revision 6 (test-selected) | .702 / .889 / .783 |
| the same three trials with d = 0 / 1 / 2 | .693 / .889 / .806, .695 / .891 / .809, .705 / .896 / .812 |
| fixed d = 0 (greedy), its own best trial per seed | .707 / .890 / .760 (std .003 / .002 / .035) |
| fixed d = 1, own best | .702 / .888 / .779 |
| fixed d = 2, own best | .707 / .897 / .806 |
| fixed d = 3, own best | .723 / .898 / .826 |
| chosen d, validation-selected trials (14, 19, 13) | .677 / .868 / .760 |
| chosen d, mean of the 60 trials | .691 / .882 / .798 |

- **R1 holds on HateMM**: +.016 / +.017 over the naive control, more than both seed standard deviations and than
  .005.
- **R2 holds on HateMM**: +.021 / +.009 over revision 6 (within +.043). R2b (rule 8) is read when the three corpora
  are finished.
- **R3**: the chosen d is 3 on the three test-selected trials. As pre-registered this reads "the selected policy is
  the single level; goal (b) is not met on HateMM" (section 9). Against d = 0 on the same trials the chosen depth is
  +.030 / +.008 / +.020. Second reference (level-only replay, below): .723 / .898 / .827, the same policy.
- **R4**: chosen d at 8 calls d = 3 in 82 % of the 60 trials (d = 0 13 %, d = 1 5 %); at 2 calls d = 1 in 83 %, at
  4 calls d = 2 in 67 %, at 16 calls d = 4 in 95 %, at 32 calls d = 5 in 92 %: the level at every budget. The chosen
  d is the d with the best test value in 63 % of the trials (best-on-test shares 17 / 3 / 15 / 65 %). Cost of the
  choice against a fixed d = 3: -.001 / .000 / -.005 over all trials, nothing on the best trial per seed.
- **Ablation of the choice** (chosen minus fixed d = 0): all trials +.016 / +.001 / +.022; trials 0-9 only
  +.012 / +.001 / +.021; best trial per seed +.016 / +.008 / +.066. The fixed d = 0 row (the section-8 candidate's
  policy, hyper-parameters proposed from the chosen-depth objective in trials 10-19) equals the naive control on AP
  (.707) and is +.009 on ROC.
- c over the 60 trials: median .26 (-.21 to .75), below .05 in 8 trials.
- Other budgets on the test-selected trials, chosen d against d = 0: 0 calls .640 / .836 / .758; 1 call
  .670 / .869 / .769; 2 calls .690 / .881 / .793 (d = 1) against .685 / .877 / .797; 4 calls .713 / .893 / .824
  (d = 2) against .694 / .886 / .808; 16 calls .718 / .895 / .806 (d = 4) against .707 / .891 / .807; 32 calls
  .723 / .890 / .809 (d = 5) against .699 / .886 / .799. Four equal windows (4 calls) are already above revision
  6's 8 calls and above the control's 8 windows; AP / ROC do not rise from 8 to 16 calls (-.005 / -.003), within
  falls by .020.
- The trial chosen by validation is far below the trial chosen by test (.677 / .868 against .723 / .898): the gap
  between validation and test trial choice of the earlier revisions is unchanged.

### HCS (uoa-lab3, job 2140, 3 seeds x 20 trials, 387-389 s per trial; finished 2026-11-11 06:14)

| 8 calls | AP / ROC / within |
|---|---|
| chosen d, test-selected trials (18, 18, 12; d = 0 on the three; c .22 / .02 / .08; 8.00 calls) | **.704 / .688 / .605** (seed std .003 / .008 / .008) |
| naive control (DSANet + 8 equal windows, alpha .1, 8.00 calls) | .696 / .653 / .577 (std .001 / .001 / .001) |
| revision 6 (test-selected) | .691 / .680 / .563 |
| the same three trials with d = 1 / 2 / 3 | .689 / .672 / .588, .689 / .677 / .614, .680 / .662 / .610 |
| fixed d = 0 (greedy), its own best trial per seed | .704 / .688 / .605 (the same trials) |
| fixed d = 1, own best (trials 5, 15, 4) | .706 / .688 / .607 (std .005 / .005 / .008) |
| fixed d = 2, own best | .692 / .682 / .613 |
| fixed d = 3, own best | .680 / .666 / .607 |
| chosen d, validation-selected trials (2, 6, 13) | .681 / .657 / .596 |
| chosen d, mean of the 60 trials | .685 / .665 / .600 |

- **R1 holds on HCS**: +.008 / +.035 over the naive control (within +.028); the AP lead is larger than both seed
  standard deviations (.003, .001) and than .005.
- **R2 holds on HCS**: +.012 / +.008 over revision 6 (within +.042).
- **R3**: the chosen d is 0 on the three test-selected trials (and on 59 of the 60 trials): the selected policy is
  the greedy tree policy, so the comparison with `depths["0"]` is a policy against itself (pre-registered reading).
  Against `depths["3"]` (the 8 equal windows first) on the same trials: AP / ROC +.024 / +.026, within -.004
  (.605 against .610; inside the .005 tolerance, seed std .008). The second half of goal (b) (AP or ROC at least
  .01 above the single level) holds on HCS in every reading: same trials +.024 / +.026, all trials +.017 / +.016
  (within -.005), trials 0-9 +.015 / +.016 (within -.003), each one's own best trial +.024 / +.022 (within -.002).
  Not covered by R3: d = 2 (4 windows first, then greedy) has the highest within on the same trials (.614, +.008
  over the chosen d) at -.015 / -.011 AP / ROC. Second reference (level-only replay with the stored weight, below):
  .680 / .662 / .610 on the same trials, the same as `depths["3"]` to three decimals.
- **R4**: chosen d at 8 calls d = 0 in 98 % of the 60 trials (one trial d = 2); at 2 calls d = 0 in 73 % (d = 1
  27 %); at 4 calls 80 / 13 / 7 % (d = 0 / 1 / 2); at 16 calls d = 0 in 50 %, d = 4 in 27 %, d = 1 / 2 / 3 in
  12 / 10 / 2 %; at 32 calls d = 5 in 100 %. The chosen d is the d with the best test value in 57 % of the trials
  (best-on-test shares 58 / 33 / 7 / 2 %). Cost of the choice at 8 calls: none against d = 0 (the same policy);
  against a fixed d = 1 +.004 / +.004 / +.007 over all trials and -.003 / -.001 / -.002 on the best trial per seed
  (inside the seed std). In stage A on the revision-6 checkpoints the choice took d = 0 in 50 % of the trials and
  cost .006 / .007 against d = 0; in this search it takes d = 0 in 98 % and costs nothing at 8 calls.
- Other budgets, chosen d minus d = 0 over the 60 trials: 2 calls -.001 / -.001 / -.001; 4 calls -.001 / .000 / .000;
  16 calls -.001 / -.001 / +.004; 32 calls **+.017 / +.018 / +.017** (d = 5, the 32 equal windows first). On the
  test-selected trials: 0 calls .651 / .638 / .602; 1 call .663 / .649 / .604; 2 calls .677 / .666 / .610 against
  .679 / .668 / .614; 4 calls .686 / .678 / .615 against .688 / .680 / .621; 16 calls .709 / .686 / .619 (d = 2, 0,
  1) against .718 / .697 / .621 (the choice costs .009 / .011 on these three trials, .001 over all trials); 32
  calls .733 / .709 / .662 (d = 5) against .718 / .691 / .644.
- c over the 60 trials: median .09 (-.007 to .51), below .05 in 21 trials. Mean test numbers of the trials by c:
  c < .05 (21 trials) .688 / .659 / .621; .05 to .15 (23) .683 / .667 / .593; above .15 (16) .683 / .669 / .584.
  Within is .03 to .04 higher when c is near 0, AP / ROC differ by at most .010 (the observation of section 11 on
  seed 3407; not followed up).
- The trial chosen by validation is below the trial chosen by test (.681 / .657 against .704 / .688), as on HateMM.

**HateMM and HCS together.** R1 and R2 hold on both. The depth chosen on validation is the single level on HateMM
(d = 3 in 82 % of the trials) and the greedy tree on HCS (d = 0 in 98 %): the two corpora take the two ends of
the candidate set, as stage A predicted. On HCS the candidate is therefore the section-8 candidate (weight c,
greedy tree) at 8 calls.

### Weight c on the candidate's own configuration (rule 14(g), reading 1) and the level-only reference (HateMM, HCS)

Replays of the 60 checkpoints per corpus of this search (`stage_a_all_trials.py --src
runs/20261110_prior_weight/search_first_level --out-root runs/20261110_prior_weight/r7_trials`, uoa-lab1 CPU,
`launch/run_all_trials_cpu.sh` with the search directory and output root; 18 s per trial; the checkpoints were
copied from the local machine to uoa-lab1; results copied back), read by `c_ablation_first_level.py` into
`r7_trials/c_ablation_<corpus>.json`. Variants: c = 1 (`r6|none|<policy>`, test and validation) and the stored c
(`r6|saved|<policy>`) for the four first levels, and the level-only policy with the stored c. 8 calls.

| HateMM | test-selected trials (16, 11, 15) | each row's own best trial per seed | mean of the 60 trials |
|---|---|---|---|
| candidate (stored c, d chosen on validation) | .723 / .898 / .826 | the same | .691 / .882 / .798 |
| c = 1, d kept | .698 / .887 / .808 | .701 / .889 / .797 (trials 16, 4, 15) | .669 / .875 / .779 |
| c = 1, d chosen again under c = 1 | .689 / .882 / .794 (d = 3, 1, 0) | .701 / .885 / .795 (16, 10, 18) | .670 / .877 / .773 |
| candidate minus "d kept" | +.025 / +.011 / +.018 | +.022 / +.008 / +.029 | +.021 / +.008 / +.019 (trials 0-9: +.017 / +.005 / +.017) |
| candidate minus "d chosen again" | +.033 / +.016 / +.032 | +.022 / +.013 / +.031 | +.020 / +.005 / +.025 (trials 0-9: +.016 / +.003 / +.021) |

| HCS | test-selected trials (18, 18, 12) | each row's own best trial per seed | mean of the 60 trials |
|---|---|---|---|
| candidate (stored c, d chosen on validation) | .704 / .688 / .605 | the same | .685 / .665 / .600 |
| c = 1, d kept | .661 / .646 / .570 | .682 / .675 / .560 (trials 1, 1, 6) | .650 / .635 / .553 |
| c = 1, d chosen again under c = 1 | .662 / .652 / .562 (d = 1, 3, 1) | .682 / .675 / .560 (1, 1, 6) | .650 / .638 / .555 |
| candidate minus "d kept" | +.043 / +.042 / +.036 | +.022 / +.013 / +.045 | +.034 / +.030 / +.048 (trials 0-9: +.024 / +.022 / +.044) |
| candidate minus "d chosen again" | +.042 / +.036 / +.044 | +.022 / +.013 / +.045 | +.035 / +.027 / +.046 (trials 0-9: +.026 / +.020 / +.042) |

- **Reading 1 of rule 14(g) for c holds on the two finished corpora**: setting c = 1 on the candidate's
  checkpoints lowers AP by .016 to .033 on HateMM and by .022 to .043 on HCS in every reading (ROC .003 to .016 and
  .013 to .042). On HateMM the weight matters once the policy is the level: section 10 read "the HateMM gain comes
  from d" from the greedy policy (c alone +.002 AP there); at the chosen depth both parts contribute (the choice of
  d +.016 / +.001 / +.022 over all trials, the weight +.021 / +.008 / +.019).
- Limit of this reading: the epoch of every checkpoint and the hyper-parameters proposed by the search were chosen
  with c in place, so c = 1 on these checkpoints is not "the method trained and selected without c". The reference
  selected without c is the revision-6 search with d chosen on validation (replays below) and the section-8
  ablations.
- Without c the choice of d is unstable: under c = 1 validation takes d = 0 / 1 / 2 / 3 in 28 / 17 / 5 / 50 % of the
  HateMM trials (82 % d = 3 with c) and in 27 / 25 / 15 / 33 % of the HCS trials (98 % d = 0 with c); it differs
  from the candidate's d in 55 % and 73 % of the trials.
- **Reference selected without c** (revision-6 search: trained, epoch and trial chosen without c; replays of its 60
  checkpoints per corpus with c = 1 for every first level, test and validation, uoa-lab1 CPU;
  `all_trials/policy_select_b8_first_none.json`, `policy_select.py --mode none --candidates eig first1 first2
  first3`). Each row's own best trial per seed, 8 calls:

  | | HateMM | HCS | DeHate |
  |---|---|---|---|
  | no c, d = 0 (revision 6) | .702 / .889 / .783 | .691 / .680 / .563 | .278 / .770 / .681 |
  | no c, d chosen on validation | .702 / .889 / .783 | .687 / .683 / .562 | .279 / .771 / .682 |
  | no c, d = 3 | .702 / .888 / .781 | .668 / .670 / .553 | .279 / .756 / .687 |
  | c, d = 0 (replay on revision-6 checkpoints; this search) | .698 / .889 / .774; .707 / .890 / .760 | .711 / .695 / .618; .704 / .688 / .605 | .298 / .793 / .693; running |
  | c, d chosen on validation (replay; this search) | .722 / .898 / .831; **.723 / .898 / .826** | .703 / .684 / .617; **.704 / .688 / .605** | .304 / .792 / .694; running |

  The candidate minus "no c, d chosen on validation": HateMM +.021 / +.009 / +.043, HCS +.017 / +.005 / +.043
  (DeHate, replays only: +.025 / +.021 / +.012). Without c the choice of d changes nothing against revision 6
  (paired over all trials -.001 / .000 / +.007, .000 / +.002 / .000, .000 / .000 / .000; chosen d = 0 / 1 / 2 / 3 in
  47 / 8 / 12 / 33 %, 75 / 15 / 10 / 0 %, 38 / 40 / 22 / 0 % of the trials).
  **On HateMM neither part alone raises AP: d without c .702, c without d .698 (replay) / .707 (this search), both
  .722 / .723.** The 8 equal windows are ahead of the greedy policy only when the per-second prior is weighted
  down (at weight 1 that prior lowers the equal-window scores, `20261109_tree_within/README.md` section 12). On
  HCS and DeHate the gain is the weight's; d adds nothing at 8 calls on HCS and +.006 AP on DeHate in the replays.
- **Level-only reference of R3** (stored c): HateMM .723 / .898 / .827 on the test-selected trials, the candidate
  (d = 3) minus it .000 / .000 / -.001 - the selected policy is the level. HCS .680 / .662 / .610, the candidate
  minus it +.024 / +.026 / -.004 on the test-selected trials, +.017 / +.016 / -.005 over all trials, +.024 / +.022 /
  -.002 with each one's own best trial: the same as against `depths["3"]`.
- **Reproduction.** The test replays with the stored weight equal the search's own numbers for every trial and
  depth (largest difference .0000). The validation numbers do not: `train.py` computes the validation metrics on
  unrounded scores, the replay (and every test evaluation, through the score files) on scores rounded to 6
  decimals. HateMM, unrounded minus rounded, mean over the trials: AP / ROC +.001, within +.003 (d = 0, 1) and
  +.006 (d = 2, 3), largest within difference .034; 41 % of the (trial, depth) pairs differ by more than .001. HCS:
  at most .003, means below .0005. With the rounded validation numbers the chosen d changes in 3 of the 60 HateMM
  trials (3 to 0, 3 to 1, 3 to 0; none of the test-selected trials) and in no HCS trial; the mean over all trials
  moves by at most .001. Cause: posteriors below 1e-6 in hateful videos keep their order unrounded and become ties
  at 6 decimals. The candidate's rule is the one of `train.py` (unrounded), unchanged.
- **What the 6-decimal score files cost at test (diagnostic; replay variants `<variant>@raw` = the same test replay
  with the metrics on unrounded scores, same evaluator; no reported number is changed).** HateMM, stored c, 60
  trials, unrounded minus 6 decimals: AP / ROC +.002 / +.002 (largest .018 / .024), within **+.010 to +.011** at
  every depth (largest .045; more than .005 in 62 to 68 % of the trials). On the test-selected trials the chosen
  depth reads .723 / .898 / .833 unrounded against .723 / .898 / .826; the mean over all trials .692 / .884 / .809
  against .691 / .882 / .798. The difference between d = 0 and d = 3 is not affected (-.002 within). HCS: no
  effect (largest .002). So on HateMM the evaluator's within of this model is about .01 lower than the within of
  the scores the model produces, because per-second probabilities below 5e-7 are written as 0. Writing more
  decimals (or a monotone transform of the score) would remove it; that changes the score files every reported
  number comes from, so it is the user's decision. On the revision-6 checkpoints (`all_trials/`, 60 trials per
  corpus) the effect is smaller: HateMM, revision 6 as reported (no weight, greedy) within +.004 over all trials
  (more than .005 in 37 % of them) and .783 to .788 on its three reported trials, AP / ROC +.001; with the weight
  on these checkpoints +.004 over all trials and nothing on the best trials. HCS and DeHate: +.001 to +.002 within
  over all trials for revision 6, nothing on the reported trials, nothing with the weight.

### DeHate

Running: seeds 234 / 2025 on uoa-lab3 (job 2141, started 2026-11-11 06:14); seed 3407 on uoa-lab1 (2135) runs
since 2026-11-11 06:48 (the other user's job ended early). First trial on uoa-lab3 2868-2869 s, so 20 trials per
seed (expected end about 22:10). uoa-lab3 stays at the commit of the search start while its jobs run; uoa-lab1
was pulled before job 2135 started and once during its first trial (06:57); the commits after the search start
change only this README, STATUS, the readers and the CPU replay tools (no file under `src/`, not `train.py`, not
the search scripts).

