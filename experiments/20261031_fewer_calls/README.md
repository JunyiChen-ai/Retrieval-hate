# Goal 1: fewer VLM calls with a two-sided stopping rule

Status (2026-10-31): candidate chosen from the diagnostics below; rule-6 code review PASS
(`docs/reviews/20261031_fewer_calls_code.md`), maximum calls set to 8 after its point A; rule-4 proposal review GO
(`docs/reviews/20261031_fewer_calls_proposal.md`); unified-evaluator results on the r5 checkpoints meet goal 1
(section 4); to be repeated on the final goal-2 backbone. Part of the user's two-goal autonomous iteration (2026-10-31).

Goal 1: on HateMM and HateClipSeg the final method uses clearly fewer than 8 VLM calls per video on average (about
4-5). Against the same method at fixed 8 calls (3-seed means): AP and ROC at most .01 lower, within not lower
(read as: not lower by more than .005, the single-seed noise level). DeHate is an external check only.

## 1. Evidence (rule-10 developmental, `experiments/20261031_two_goal_diagnostics/`)

- Allocating calls by P(G) or other video-level rules cannot reach 4-5 calls at the required quality
  (`call_allocation.py`; HateClipSeg is 87% positive).
- With the complete method's policy (EIG order, copy likelihood in EIG and posterior), the trade-off at 4-5 calls is
  between two kinds of videos (`order_anatomy.py`, `rule_scan.py`):
  - Pooled AP / ROC need calls on negative videos: rules driven by video-level uncertainty and the last change
    (qmixSG, qmixS, qmixG, qmixD) keep AP / ROC but lose .005-.008 within on HateClipSeg.
  - Within needs calls on positive videos: rules driven by the information still left in the video (eigsum,
    eigtop3) give positives 7-9 calls and negatives 1-3, and lose .012-.025 AP / ROC.
  - The current rule (qmixSG_rt10) at a mean of 5 calls loses .014 within on HateClipSeg.
- The expected-error-reduction question order and a validation-fitted answer model did not help on both corpora
  (`policy_lab.py`, README of the diagnostics, finding 4).

## 2. Mechanism

Stopping rule qmixE1 (`src/qtl/stop_rules.py`): ask the next question while either of two statistics is still large
relative to its distribution on validation (no labels):
- backward: dlsum, the total change in per-second log-odds that the last answer caused (the pooled-metric change);
- forward: the EIG of the best next question (the selection order's own value).
Each is replaced by its quantile among all finite validation (video, call) values, the rule value is the larger
quantile, and one threshold c is set so that the validation mean number of calls is 5 (policy.calibrate). The first
call is always made (no change has been observed yet); there is no floor or schedule constant.
The maximum number of calls is the fixed budget the rule is compared with (8): the rule only decides when to stop
earlier, so no video gets more calls than in the fixed-8 method and no new constant is added. The maximum does change
the rule (the code review found a 32-call cap binding on HateMM and moving c), so results at a 16-call maximum are
kept in section 3 for reference.
It is the qmixSG family with the video-entropy and rank-change components replaced by the next question's EIG. Read
together, a video stops when the last answer no longer moves the scores and no remaining question is informative.

Order, answer model, copy likelihood and network are unchanged.

## 3. Selection record (rule 10)

The rule was chosen on test, from the stopping-rule families of README 17.4 of `20260925_query_paradigm`:
- scan: about 190 rules x 2 orders x 2 targets on the r5 test-selected trials, policy run to 16 calls
  (`rule_scan.json`);
- qmixE3 (dlsum and the sum of the top-3 EIGs) passed both corpora; qmixE1 replaces the top-3 sum by the top-1 value to
  remove the constant 3, decided before its result was seen. qmixE1 is the scan's `qmixS1`, so the HateMM and
  HateClipSeg r5 (3) rows below are the selection set, not a check;
- checks: 6 more r5 trials per corpus (2nd and 3rd best per seed), DeHate r5 trials, the inside-outside trials.

Differences to fixed 8 of the same trials (AP / ROC / within, mean test calls), target 5, EIG order, shared core
evaluator inside the scan (`hc.frame_metrics`):

| Setting | qmixE1, max 8 | qmixE3, max 8 | qmixE1, max 16 |
|---|---|---|---|
| HateMM r5 (3) | -.002 / -.003 / -.003 (4.9) | -.003 / -.002 / -.001 (4.9) | -.003 / -.005 / +.004 (4.9) |
| HateMM r5 more (6) | -.000 / -.002 / -.002 (4.9) | -.001 / -.002 / -.004 (4.9) | +.004 / -.001 / -.004 (4.8) |
| HateClipSeg r5 (3) | -.009 / -.009 / +.004 (5.0) | -.009 / -.007 / +.008 (4.9) | -.007 / -.007 / +.010 (5.0) |
| HateClipSeg r5 more (6) | -.006 / -.006 / +.003 (4.5) | -.006 / -.006 / +.003 (4.5) | -.004 / -.007 / +.005 (4.6) |
| DeHate r5 (3) | -.002 / -.003 / -.004 (4.9) | -.002 / -.003 / -.004 (4.9) | -.002 / +.001 / +.001 (4.7) |
| HateMM inside-outside (3) | | | -.004 / -.005 / -.005 (4.5) |
| HateClipSeg inside-outside (3) | | | -.009 / -.000 / -.004 (5.1) |

Max 8: the stored 16-call runs truncated to 8 calls (`rule_scan.py --cap 8`, `rule_scan_cap8.json`). At max 8,
qmixE1 at target 4.5 misses on HateClipSeg r5 (AP -.010); at target 5 the HateClipSeg r5 margin to the .01 bound is
.001. The claim rows are produced by `evaluate_stop.py` through the unified evaluator.

## 4. Results on the r5 checkpoints (unified evaluator)

`evaluate_stop.py`, maximum 8 calls, target 5, 3-seed means over the r5 test-selected trials. Sources:
`runs/20261031_fewer_calls/{hatemm,hateclipseg}/r5/summary_sc474397.json`, `dehate/r5/summary_sc474398.json`.
AP / ROC / within, mean test calls:

| Corpus | fixed 8 | qmixE1 | qmixE1 - fixed 8 | fixed 5 - fixed 8 |
|---|---|---|---|---|
| HateMM | .6857 / .8822 / .7665 (7.63) | .6834 / .8798 / .7677 (4.92) | -.0023 / -.0024 / +.0012 | -.0087 / -.0082 / -.0087 (4.90) |
| HateClipSeg | .6831 / .6718 / .5590 (8.00) | .6736 / .6629 / .5636 (4.96) | -.0095 / -.0089 / +.0046 | -.0071 / -.0031 / -.0070 (5.00) |
| DeHate (external) | .2582 / .7702 / .6673 (7.26) | .2559 / .7680 / .6649 (4.95) | -.0023 / -.0022 / -.0024 | +.0012 / -.0052 / -.0039 (4.77) |

- Goal 1 is met on both main corpora: about 4.9-5.0 calls instead of 8 (HateMM: 7.63, because some videos have fewer
  than 8 queryable nodes); AP and ROC at most .0095 lower; within not lower.
- The HateClipSeg AP margin to the .01 bound is .0005. Per seed it ranges from -.003 to -.019 (seed 2025).
- The calls are bimodal. Many videos stop after 1-2 calls, and many use all 8: HateMM 63-109 of 214 test videos at 8,
  HateClipSeg 22-32 of 79. Positive videos get more calls than negative ones: HateMM 6.3-7.3 vs 3.6-3.8,
  HateClipSeg 4.7-5.7 vs 2.9-4.0.
- Uniform 5 calls fails the within bound on both corpora (-.0087, -.0070). On HateClipSeg it keeps AP / ROC better
  than qmixE1 (by .0024 / .0058).

## 5. Ablation (each component alone, same validation mean calls)

`launch/run_r5.sh ablation`; source `runs/20261031_fewer_calls/{hatemm,hateclipseg}/r5_ablation/summary_sc474397.json`.
Differences to fixed 8 (AP / ROC / within, mean test calls):

| Corpus | qmixE1 | dlsum alone (no next-question EIG) | EIG alone (no last change) | uniform 5 |
|---|---|---|---|---|
| HateMM | -.002 / -.002 / +.001 (4.92) | +.001 / -.001 / +.002 (5.25) | -.005 / -.009 / -.006 (4.84) | -.009 / -.008 / -.009 (4.90) |
| HateClipSeg | -.010 / -.009 / +.005 (4.96) | -.009 / -.009 / -.006 (4.55) | -.021 / -.021 / -.001 (4.86) | -.007 / -.003 / -.007 (5.00) |

- Only the combined rule meets goal 1 on both corpora. dlsum alone fails the within bound on HateClipSeg (-.006);
  EIG alone fails AP / ROC on HateClipSeg (-.021); uniform 5 fails the within bound on both.
- Rule 14(g) (pooled AP or ROC drop >= .01 on both corpora when a component is removed) is not met by either
  component: removing dlsum drops HateClipSeg AP / ROC by .012 but HateMM by .003 / .006; removing the EIG changes
  pooled AP / ROC by less than .001 on HateClipSeg (its effect is within, -.011). The stopping rule is an efficiency
  result (same quality at fewer calls), not a pooled-accuracy mechanism.

## 6. The number of calls is a user budget (user ruling 2026-11-01)

The mean of 5 calls is hand-set. Letting validation choose it does not work: the validation objective's bootstrap
standard error (.025-.053) is far above the .005-.01 differences that matter, and the chosen budget swings from 1 to 7
calls across seeds (`experiments/20261031_two_goal_diagnostics/README.md`, finding 11). The user ruled (2026-11-01)
that the mean number of calls is a budget the user sets for their resources, like k of fixed-k; the paper reports the
curve below, with 5 as one point.

Curve (`experiments/20261031_two_goal_diagnostics/stop_budget_curve.py`, uoa-lab1, stored r5 policy runs with copy
likelihood truncated to 8 calls, unified evaluator, 3-seed means; output
`runs/20261031_two_goal_diagnostics/stop_budget_curve/<corpus>/summary.json`). Each cell: test mean calls; AP / ROC /
within minus fixed 8 of the same trials; then minus fixed k at the nearest whole number of calls.

| Budget | HateMM | HateClipSeg | DeHate |
|---|---|---|---|
| 1 | 1.00; -.026 / -.026 / +.007; = fixed 1 | 1.02; -.020 / -.016 / -.014; +.000 / +.001 / +.000 | 1.00; -.015 / -.024 / -.013; = fixed 1 |
| 2 | 2.17; -.007 / -.008 / +.007; +.011 / +.011 / +.002 | 1.92; -.014 / -.009 / -.012; +.005 / +.004 / -.000 | 1.94; -.004 / -.012 / -.008; +.005 / +.005 / -.001 |
| 3 | 3.01; -.008 / -.008 / +.009; +.010 / +.006 / +.017 | 2.89; -.014 / -.013 / -.006; -.001 / -.008 / +.006 | 2.85; -.008 / -.009 / -.004; -.003 / +.005 / +.004 |
| 4 | 3.96; -.007 / -.005 / +.012; +.009 / +.007 / +.017 | 3.81; -.010 / -.008 / +.000; -.002 / -.006 / +.010 | 3.84; -.003 / -.003 / -.005; +.000 / +.006 / +.002 |
| 5 | 4.92; -.002 / -.002 / +.001; +.006 / +.006 / +.010 | 4.96; -.010 / -.009 / +.005; -.002 / -.006 / +.012 | 4.95; -.002 / -.002 / -.002; -.004 / +.003 / +.002 |
| 6 | 5.88; +.000 / -.000 / -.006; +.007 / +.005 / -.004 | 5.70; -.009 / -.008 / +.004; -.006 / -.006 / +.018 | 5.98; -.001 / -.000 / -.004; -.003 / +.003 / -.004 |
| 7 | 6.48; +.000 / +.000 / -.001; +.007 / +.005 / +.001 | 6.84; -.009 / -.008 / +.002; -.009 / -.007 / +.002 | 6.42; -.000 / -.000 / -.001; -.002 / +.003 / -.001 |

Fixed 8 of these trials (with copy likelihood, the reference): HateMM .686 / .882 / .767 (7.63 calls: short videos
have fewer than 8 nodes), HateClipSeg .683 / .672 / .559 (8.00), DeHate .258 / .770 / .667 (7.26).

- HateMM: the rule is above uniform fixed k at the same calls at every budget from 2 to 7 (AP +.006 to +.011) and
  within .003 of fixed 8 from budget 5 up.
- HateClipSeg: AP / ROC stay about .009 below fixed 8 from budget 4 to 7, while fixed 7 is within .001; the rule
  gives fewer calls to negative videos (5.9 of 7 at budget 7), and on HateClipSeg negatives need calls for pooled
  AP / ROC (finding 2). Within is above fixed k by .010-.018 at budgets 4-6.
- DeHate: within .004 of fixed 8 from budget 4 up; against fixed k, ROC higher (+.003 to +.006), AP equal or lower
  (up to -.004).
- Goal 1 at budget 5 holds on all three main corpora (HateClipSeg AP -.0095, the closest to the .01 bound).

## 7. How to run

`evaluate_stop.py --corpus <c> --tag <tag> --trial-dirs <trials>`: policy up to 8 calls on validation and test,
qmixE1 and qmixE3 calibrated to a validation mean of 5, unified evaluator for the rules, fixed 8 and fixed 5.
Launch: `launch/run_r5.sh main` (uoa-lab1), `launch/run_r5.sh dehate` (uoa-lab3).
Output: `runs/20261031_fewer_calls/<corpus>/<tag>/<seed_trial>/` and `summary_<host>.json`.

The final claim must be made on the final backbone (goal 2); the r5 run is the reference for the current backbone.
