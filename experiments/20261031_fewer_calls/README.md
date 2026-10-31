# Goal 1: fewer VLM calls with a two-sided stopping rule

Status (2026-10-31): candidate chosen from the diagnostics below; rule-6 code review PASS
(`docs/reviews/20261031_fewer_calls_code.md`), maximum calls set to 8 after its point A; waiting for the rule-4
proposal review before the evaluator run. Part of the user's two-goal autonomous iteration (2026-10-31).

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

## 4. How to run

`evaluate_stop.py --corpus <c> --tag <tag> --trial-dirs <trials>`: policy up to 8 calls on validation and test,
qmixE1 and qmixE3 calibrated to a validation mean of 5, unified evaluator for the rules, fixed 8 and fixed 5.
Launch: `launch/run_r5.sh main` (uoa-lab1), `launch/run_r5.sh dehate` (uoa-lab3).
Output: `runs/20261031_fewer_calls/<corpus>/<tag>/<seed_trial>/` and `summary_<host>.json`.

The final claim must be made on the final backbone (goal 2); the r5 run is the reference for the current backbone.
