# R7-1 — annotator-vote soft-label training: result

Run 2026-08-17. Design, arms, targets and decision rule frozen and committed at `9f2f6a5`
(`idea-stage/R7_SOFTVOTE_FREEZE.md`) **before any seed in the range 100-129 was executed**. Single
submission: one background job, 300/300 runs complete, 0 failures, 2767 s wall; the analyzer was
run exactly once on the complete grid, with no edit after the freeze commit and no re-run.

- Targets: `idea-stage/r7_softvote/build_targets.py` → `targets_{MHC_zh,MHC}_{SOFT10,SOFT05}.json`,
  `build_meta.json` (SHA-256 of every file recorded).
- Grid: `idea-stage/r7_softvote/run_arms.sh` → `logging/runs/r7_softvote/` (`run.log`, `run.pid`,
  `logs/`).
- Read-out: `idea-stage/r7_softvote/analyze.py` → `idea-stage/r7_softvote/results.json`.
- Cost: **¥0**. Zero API calls, zero cloud. Local RTX 5090, shared with another user's job.

# VERDICT: **KILL**

Neither dataset passes. Every soft-target arm **loses** macro-F1 against the hard-label baseline
under the primary protocol, and every soft-target arm also loses against its own entropy-matched
label-smoothing control. No arm reaches the TRICK branch either, because the smoothing controls are
themselves flat.

## 1. Arm means — test macro-F1, 30 seeds (100-129), P1 primary

| arm | MHC_zh mean | std | MHC (EN) mean | std |
|---|---|---|---|---|
| `A0` hard label | **0.8054** | 0.0122 | **0.7149** | 0.0131 |
| `SOFT10` votes, Offensive = hate | 0.7748 | 0.0181 | 0.7075 | 0.0100 |
| `SOFT05` votes, Offensive = half | 0.7781 | 0.0338 | 0.6905 | 0.0127 |
| `LS10` smoothing matched to SOFT10 | 0.8075 | 0.0065 | 0.7185 | 0.0113 |
| `LS05` smoothing matched to SOFT05 | 0.8060 | 0.0072 | 0.7110 | 0.0094 |

MHC_zh `A0` = 0.8054 against the ledger's 0.8014 (seeds 30-89): same quantity on a disjoint seed
range, differing by 0.0040 ≈ 1.8 se. Consistent. MHC (EN) has no `ro_` cache and therefore no ledger
entry; its `A0` is defined inside this grid and is never compared outside it.

## 2. Paired deltas — P1, paired bootstrap 95 % CI over 30 seeds

| dataset | contrast | mean | 95 % CI | seeds positive |
|---|---|---|---|---|
| MHC_zh | **SOFT10 − A0** | **−0.0306** | [−0.0383, −0.0226] | 4/30 |
| MHC_zh | **SOFT05 − A0** | **−0.0274** | [−0.0392, −0.0147] | 8/30 |
| MHC_zh | LS10 − A0 | +0.0021 | [−0.0018, +0.0059] | 17/30 |
| MHC_zh | LS05 − A0 | +0.0006 | [−0.0050, +0.0059] | 15/30 |
| MHC_zh | SOFT10 − LS10 | −0.0327 | [−0.0393, −0.0256] | 4/30 |
| MHC_zh | SOFT05 − LS05 | −0.0279 | [−0.0403, −0.0147] | 8/30 |
| MHC | **SOFT10 − A0** | **−0.0074** | [−0.0132, −0.0010] | 9/30 |
| MHC | **SOFT05 − A0** | **−0.0244** | [−0.0312, −0.0173] | 4/30 |
| MHC | LS10 − A0 | +0.0036 | [−0.0022, +0.0095] | 12/30 |
| MHC | LS05 − A0 | −0.0038 | [−0.0094, +0.0019] | 11/30 |
| MHC | SOFT10 − LS10 | −0.0110 | [−0.0153, −0.0064] | 5/30 |
| MHC | SOFT05 − LS05 | −0.0206 | [−0.0262, −0.0145] | 3/30 |

Against the frozen rule, per dataset: the best soft arm is `SOFT05` on MHC_zh (−0.0274) and
`SOFT10` on MHC (−0.0074). Condition 1 (`mean ≥ +0.005`) fails on both. Condition 3 (`SOFT` beats
its matched `LS`) fails on both. → **KILL**, and not TRICK, since no soft arm cleared the bar at all.

**The label-smoothing controls are also flat.** The largest is `LS10 − A0` = +0.0036 on EN, CI
[−0.0022, +0.0095], 12/30 seeds positive. Softening the target *per se* buys nothing here either;
this is not a case of "the regularisation helped and the votes were incidental".

## 3. P2 corroboration (final epoch)

| dataset | contrast | P1 | P2 |
|---|---|---|---|
| MHC_zh | SOFT10 − A0 | −0.0306 | −0.0016 [−0.0090, +0.0050] |
| MHC_zh | SOFT05 − A0 | −0.0274 | −0.0242 [−0.0356, −0.0128] |
| MHC | SOFT10 − A0 | −0.0074 | +0.0040 [−0.0025, +0.0100] |
| MHC | SOFT05 − A0 | −0.0244 | −0.0426 [−0.0484, −0.0370] |

`SOFT05` is negative under both protocols on both datasets — unambiguous. `SOFT10` is negative
under P1 and statistically indistinguishable from zero under P2 on both datasets. Under the frozen
rule P2 is corroboration only and cannot rescue an arm that fails the primary bar by a wide margin;
`SOFT10`'s P2 point estimates (−0.0016 and +0.0040) are far below +0.005 and their CIs contain 0,
so P2 does not even weakly support a gain. The protocol disagreement is real, is explained in §4,
and is recorded rather than smoothed over.

## 4. Why it lost — two distinct mechanisms, both arithmetic

Mean predicted-positive count on test at the read-out epoch (test positives: MHC_zh 45 of 149,
MHC 49 of 161):

| dataset | arm | P1 pred-pos | P2 pred-pos | mean selected epoch (P1) |
|---|---|---|---|---|
| MHC_zh | A0 | 43.9 | 46.1 | 21.4 |
| MHC_zh | SOFT10 | 41.3 | 47.3 | 18.0 |
| MHC_zh | SOFT05 | **34.5** | **35.5** | 24.4 |
| MHC_zh | LS10 | 45.1 | 45.4 | 21.1 |
| MHC | A0 | 44.6 | 46.5 | 26.4 |
| MHC | SOFT10 | 43.1 | 46.9 | 23.6 |
| MHC | SOFT05 | **38.0** | **35.9** | 26.9 |
| MHC | LS10 | 45.2 | 45.7 | 26.9 |

**(a) `SOFT05` fails by construction, and the size was predictable from the target.** Its mean
training target is 0.2191 (ZH) / 0.1846 (EN) against a hard positive rate of 0.311 / 0.306. Halving
the weight of `Offensive` removes roughly a third of the positive mass from the objective, the head
learns a correspondingly lower output level, and at the fixed 0.5 threshold it predicts 34.5 / 38.0
positives where 45 / 49 exist. The loss is recall paid for nothing. This reproduces, at the level
of an actual trained head rather than an oracle, what
`refine-logs/C05PLUS_FORENSIC_RECON_2026-07-31.md` §4.2 recorded from the error analysis:
down-weighting `Offensive` is *monotonically harmful*, and on ZH there is no Offensive-specific
error mass to reallocate.

**(b) `SOFT10` fails through the epoch-selection step, not through the final model.** Its target
mean (0.3199 / 0.2940) sits close to the base rate, and at the last epoch it predicts 47.3 / 46.9
positives — normal. But its P1-selected epoch is systematically earlier (18.0 vs A0's 21.4 on ZH)
and at that epoch it under-predicts (41.3). Soft targets compress the head's output range, so the
dev macro-F1 curve at a fixed 0.5 threshold becomes a noisier and less transferable selection
signal: `SOFT10`'s P1 seed std is 0.0181 against A0's 0.0122. The soft target does not damage the
converged model; it damages the instrument that picks which epoch to keep.

Neither mechanism is about the votes carrying wrong information. Both are about the *scale* of the
target interacting with a fixed 0.5 decision threshold. That distinction matters for what this does
and does not close (§6).

## 5. How much signal was on the table

Measured before any arm ran, and recorded in the freeze:

| dataset | arm | fraction of train targets strictly in (0,1) | distinct target values | corr. with hard label |
|---|---|---|---|---|
| MHC_zh | SOFT10 | 0.181 | 6 | 0.949 |
| MHC_zh | SOFT05 | 0.370 | 9 | 0.898 |
| MHC | SOFT10 | 0.098 | 5 | 0.975 |
| MHC | SOFT05 | 0.308 | 7 | 0.918 |

With 2 annotators on 422/461 of the videos, most items admit only `{0, 0.5, 1}`. On `SOFT10` fewer
than one train row in five (ZH) or one in ten (EN) carries any information the hard label does not
already carry. The asset is real — MultiHateClip does release per-annotator votes, which HateMM,
HateClipSeg and ImpliHateVid do not — but at 2 annotators per item it is thin.

## 6. What this closes and what it does not

**Closed:** replacing the BCE training target with a vote-derived soft target, at this annotator
depth (2-4 raters, no rater IDs), on this substrate, at the current +0.005 bar. Two weightings, two
datasets, two protocols, 30 seeds each, with an entropy-matched smoothing control. Nothing is near
the bar; the nearest point estimate is +0.0040 and it belongs to a *control* arm, not a soft arm.

**Also closed as a by-product:** plain label smoothing on this head. Both matched epsilons are flat
on both datasets, CIs containing zero.

**Not closed:** (i) anything that uses the votes as something other than a scalar regression target
— rater-level modelling is impossible here anyway (the release carries no annotator IDs,
`refine-logs/C05PLUS_FORENSIC_RECON_2026-07-31.md` §0.1); (ii) the possibility that a soft target
combined with a *validation-selected threshold* rather than a fixed 0.5 would not lose, since §4
locates both failure mechanisms at the threshold/scale interaction. That is a real loose end and it
is stated as such — but it is a repair to the read-out, not evidence that the votes carry usable
information, and pursuing it would be optimising a channel whose best observed point estimate is
still negative.

## 7. Prior-record consistency

The freeze's step-0 check argued that the 2026-07-31 "graded 3-class Offensive soft-label" pre-gate
(oracle ceiling EN +0.0250 / ZH +0.0256, killed against a retired +0.030 bar) did not transfer,
because it tested a different object against a different bar. That argument was correct as a
procedural matter — the pilot was legitimate to run — and the outcome now **agrees with the old
record's direction** and is worse than its oracle: where the oracle bounded a monotone reweighting
of `Offensive` rows at +0.025, the realised vote-derived targets deliver −0.007 to −0.031. The
oracle was an upper bound on a related family, and the realisable value sits far below it.

## 8. Expectation check

The freeze predicted KILL or TRICK, on the grounds that `SOFT10`'s targets barely move and
`SOFT05` down-weights `Offensive` in a way already recorded as harmful. Both halves were correct in
direction. Two things were not anticipated: the magnitude on ZH `SOFT10` (−0.0306, expected roughly
null), and that the damage runs through epoch selection rather than through the converged model.

## 9. Data discipline

- Soft targets were built from `mhc_{Chinese,English}_train.tsv` only; `build_targets.py` halts on
  any vote file not named `*_train.tsv`. **The test-split votes were never used for anything.**
- Epoch selection used the val split's hard macro-F1. Test labels were read only for the final
  metric. No `w`, no `eps`, no epoch rule and no threshold was selected on test.
- One upstream inconsistency was logged and not fixed: ZH train row `BV1jW4y1n7fP` has votes
  `['Offensive','Offensive']` (p = 1.0) while the release's `Majority_Voting` says `Normal` and the
  project label is 0. 1 of 579 rows; the project label was left untouched.
- Code: `--soft_target_json`, `--label_smoothing`, `--dump_head_scores` all default off, and with
  them off `MHC_zh / A0 / seed 30` reproduces `logging/runs/r6_confirm/logs/MHC_zh_A0_s30.trainlog`
  line-for-line on all 60 dev/test epoch lines.

## 10. Novelty search

Not run. The frozen rule makes it conditional on GO; the verdict is KILL.

## 11. Reproduction

```
python idea-stage/r7_softvote/build_targets.py     # CPU, seconds
bash   idea-stage/r7_softvote/run_arms.sh          # 300 head runs, 2767 s
python idea-stage/r7_softvote/analyze.py           # CPU, seconds
```
Raw: `idea-stage/r7_softvote/results.json` (per-seed values for every arm, dataset and protocol),
`logging/runs/r7_softvote/logs/*.trainlog`.
