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

- The two plan values trade the corpora at every budget: the eer plan equals the single level on HateMM (AP / ROC
  within .003, .011 ahead at B = 6 where the level cannot use 2 calls; within .006-.021 lower) and stays at revision 6
  on HCS; the EIG plan is .011-.017 AP ahead of revision 6 on HCS and .015-.018 AP behind the single level on HateMM
  at 4 and 8 calls.
- The EIG plan against the greedy EIG policy (same value, same weight): within .011 at every budget, mostly positive.

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
  comparison with the naive control (.707 / .881, .693 / .651, .243 / .773) and with the single level (goal b).

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
