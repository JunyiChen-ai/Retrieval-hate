# R8-D — where the head-level gains are, and where they are lost

**Date** 2026-08-17. **Cost** ¥0 API, ~15 min local RTX 5090 (shared), 0 test-set contact.
**Purpose** premise checks for round-8 idea discovery, whose search axis is *downstream head
architecture + training objective / optimisation dynamics + view-combination mechanism*.

**Test discipline.** No test split file is opened by any script in `idea-stage/r8_decomp/`. Every
number below is out-of-fold on a stratified 5-fold split of **train + val**, i.e. the pool the
project is already free to use. The four datasets keep their own pools (HateMM 851, MHC-EN 629,
MHC-ZH 657, ImpliHateVid 1608 items).

| script | question | raw |
|---|---|---|
| `decomp.py` | is the recorded ensemble gain member complementarity or head estimation variance? | `results.json` |
| `decomp2.py` | does the trajectory carry a gain the deployed P1 read-out throws away? | `results2.json` |
| `decomp3.py` | does a ranking gain convert into macro-F1, and at which operating point is it lost? | `results3.json` |

Head, optimiser and hyper-parameters are `idea-stage/r4_harness.py` (the deployed
`classifier_hateClipper` geometry: map 1024, proj 1024, 3 layers, align fusion, dropout
0.2/0.4/0.1, AdamW 1e-4, batch 64, 30 epochs, warm-up 5).

---

## 1. The ensemble gain is not cross-encoder complementarity

`decomp.py`, 5 folds x 3 seeds x {2,3} encoders, final-epoch read-out at threshold 0.5, so no
epoch or threshold selection enters. "single" = mean over seeds of one head; "seed-ens" = the
probability average of 3 seeds of the **same** encoder; "cross-ens" = the probability average of
one head per encoder.

| dataset | best single | seed-ens (same encoder) | cross-ens (all encoders) |
|---|---|---|---|
| HateMM | 0.8690 | **0.8844** (+0.0153) | 0.8622 (−0.0068) |
| MHC-EN | 0.8217 | 0.8232 (+0.0016) | 0.7790 (−0.0427) |
| MHC-ZH | 0.8494 | 0.8542 (+0.0048) | 0.8304 (−0.0191) |
| ImpliHateVid | 0.9121 | 0.9154 (+0.0033) | 0.9224 (+0.0103) |

Same contrast on ROC: seed-ens − single = +0.0041 / +0.0054 / +0.0080 / +0.0021, **positive 4/4**.

Two readings, both load-bearing for round 8:

1. **Averaging three seeds of one encoder is positive on macro-F1 in 4/4 and on ROC in 4/4.** The
   head's own estimation variance is a real, recoverable error term.
2. **Equal-weight averaging across encoders loses to the best single encoder on 3 of 4 datasets.**
   The recorded "+1.3 to +5.3 three-encoder ensemble gain" (`IDEA_REPORT` §8.7) is measured against
   the *validation-selected* single encoder and uses weighted / logistic / MLP stackers. Once the
   best encoder is known and the members are averaged with equal weight, the cross-encoder gain is
   negative except on ImpliHateVid, the one dataset whose two encoders are closest in strength
   (0.9106 vs 0.9121). The ensemble line is largely **encoder selection plus variance reduction**,
   not an exploitable complementarity. This is the same conclusion the R4-1 lattice reached from
   the other direction (post-hoc non-additive combination extracts ΔROC = −0.0000).

---

## 2. The trajectory carries a ranking gain that the deployed read-out throws away

`decomp2.py` adds an inner dev split (1/8 of each outer training portion) so the deployed protocol
is one of the arms. 5 folds x 5 seeds, best encoder per dataset. `SEL` = P1, the epoch that
maximises inner-dev macro-F1; `FINAL` = last epoch; `TRAJ` = mean probability over epochs 20-29;
`TRAJW` = mean probability over the 5-epoch window centred on the selected epoch.

Deltas against `SEL`:

| dataset | FINAL ΔF1 / ΔROC | TRAJ ΔF1 / ΔROC | TRAJW ΔF1 / ΔROC |
|---|---|---|---|
| HateMM | −0.0009 / **+0.0134** | +0.0005 / **+0.0171** | −0.0019 / +0.0055 |
| MHC-EN | −0.0001 / −0.0011 | +0.0024 / +0.0059 | −0.0025 / +0.0075 |
| MHC-ZH | −0.0086 / **+0.0098** | −0.0040 / **+0.0150** | −0.0051 / +0.0087 |
| ImpliHateVid | −0.0006 / −0.0006 | +0.0010 / +0.0032 | +0.0031 / +0.0020 |

**Trajectory averaging is positive on ROC in 4/4 (+0.003 to +0.017) and worth essentially nothing
on macro-F1 (+0.0005, +0.0024, −0.0040, +0.0010).** On MHC-ZH it is ROC-better and macro-F1-worse
in the same cell. The selected epoch has mean 15.8-23.8 with a **seed std of 2.9-7.0 epochs**, so
P1 selection is itself a noisy operation on 60-130 inner-dev items.

This is the *second* independent instance of the same signature. The first is already banked:
`IDEA_REPORT` §8.8 records that a pairwise / AUC objective beats BCE on **test ROC in 4 of 4 cells**
(+0.0080 / +0.0167 / +0.0115 / +0.0020) — and no macro-F1 gain was ever recorded for it.

Two mechanisms, arrived at independently, both buy ~1 point of ROC and 0 points of the reported
metric.

---

## 3. The ranking gain does not convert, and the operating point is not where it is lost

`decomp3.py` re-runs the same grid and evaluates every arm at four operating points: fixed 0.5,
a threshold fitted per fold on the inner dev split, a quantile threshold matched to the training
positive rate, and a global threshold fitted on the pooled out-of-fold scores (an upper bound; it
uses no test data, only the train+val pool the project already owns).

Test macro-F1 is never involved. Absolute out-of-fold macro-F1:

| dataset | arm | ROC | @0.5 | dev-fitted | prior-matched | **oracle** |
|---|---|---|---|---|---|---|
| HateMM | SEL | 0.9193 | 0.8744 | 0.8715 | 0.8772 | 0.8769 |
| | TRAJ | 0.9363 | 0.8750 | 0.8673 | 0.8786 | **0.8821** |
| MHC-EN | SEL | 0.9008 | 0.8282 | 0.8259 | 0.8181 | 0.8389 |
| | TRAJ | 0.9067 | 0.8306 | 0.8298 | 0.8316 | **0.8422** |
| MHC-ZH | SEL | 0.9183 | 0.8661 | 0.8617 | 0.8648 | 0.8703 |
| | TRAJ | 0.9334 | 0.8621 | 0.8592 | 0.8725 | **0.8711** |
| ImpliHateVid | SEL | 0.9677 | 0.9113 | 0.9097 | 0.9132 | 0.9145 |
| | TRAJ | 0.9710 | 0.9123 | 0.9084 | 0.9129 | **0.9147** |

Three facts, in the order that matters.

**(a) The operating-point headroom is +0.0025 to +0.0119, not the +0.012 to +0.046 on record.**
`IDEA_REPORT` §8.2 measured the threshold oracle on the 149-215-item **test** splits and obtained
+1.2 to +4.6 points. On the 629-1608-item train+val pool the same oracle is worth **+0.25 to +1.2
points**. The difference is the oracle overfitting a small evaluation set: a threshold fitted on 149
items is worth far more on those same 149 items than any threshold rule can be worth in general.
The realistic rules are worse than the oracle and mostly worse than 0.5 — a dev-fitted threshold
is **negative on 3 of 4 datasets** (−0.0029, −0.0008, −0.0029, −0.0039 for the TRAJ arm) and
prior-matching ranges from −0.0002 to +0.0104. **There is no meaningful prize at the operating
point, and the standing "+1.2 to +4.6 calibration cap" overstates it by roughly 4x.**

**(b) The ranking gain is still not converted even at the oracle threshold.** TRAJ − SEL, evaluated
at each arm's own oracle threshold, is **+0.0051 / +0.0033 / +0.0008 / +0.0001**. So the +0.003 to
+0.017 ROC advantage of trajectory averaging survives as at most half a macro-F1 point, and as
nothing at all on two datasets, *after* the operating point has been removed as a confound.

**(c) Therefore the ROC gain lives in the wrong part of the curve.** macro-F1 at any threshold is a
function of the ordering **local to that threshold**. A mechanism that improves global AUC by
re-ordering pairs that are already far apart moves ROC and cannot move macro-F1. That is exactly
what trajectory averaging does here, and — by the identical signature — it is the most economical
explanation for the banked but never-converted pairwise-objective result (+0.008 to +0.017 test ROC
in 4/4 cells, `IDEA_REPORT` §8.8, macro-F1 never reported).

---

## 4. What this means for round-8 candidate selection

**Closed by measurement, no pilot needed:**

1. **Operating-point / calibration / thresholding mechanisms.** Capped at +0.25 to +1.2 on a
   properly sized pool, and every realistic rule tested (dev-fitted, prior-matched) already sits at
   or below the fixed 0.5 baseline on most cells. Includes quantile-anchored training, learned
   thresholds, prior correction, and Saerens-EM / BBSE style rules.
2. **Global-AUC / global-ranking objectives as a route to macro-F1.** Two independent mechanisms
   now show the same +1-ROC-point / +0-macro-F1 signature. Any candidate justified by "it should
   improve the ranking" must state why its improvement is *local to the operating point*.
3. **Post-hoc combination of encoder views.** Equal-weight cross-encoder averaging is negative on
   3 of 4 datasets against the best single encoder (§1), which agrees with the R4-1 lattice null
   and with the published finding that plain concatenation is as good as any learned mixer
   (`2408.15998` Eagle; `2503.06063` CVPR 2025).
4. **Learned multi-layer / multi-view combination heads.** Occupied by `2601.09322` (attentive
   multilayer fusion over frozen ViTs, 20 datasets, Jan 2026), `2405.13800` (Dense Connector,
   NeurIPS 2024) and `2606.26379` (differentiable fusion search).
5. **Joint diversity-regularised ensemble training.** `2301.11323` (NeurIPS 2023) shows jointly
   optimising a collective objective makes base learners collude to inflate apparent diversity,
   with the failure worst at small n. Our n is 549-1283.
6. **Implicit-ensemble architectures** (BatchEnsemble / MIMO / Packed). `2601.16936` (Jan 2026)
   finds BatchEnsemble tracks a single model on accuracy, calibration and OOD, with members near
   identical in function and parameter space.

**Left standing on this axis:**

- A ranking objective whose pressure is **localised to the decision boundary** rather than spread
  over all positive-negative pairs. This is the only shape of objective consistent with (b) and (c).
  The intersection "ranking objective x frozen-feature probe" is empty in the literature, and
  ranking / AUC / margin objectives are unoccupied in hateful meme and hateful video detection
  altogether (the occupied territory is contrastive representation learning, fusion architecture
  and RL/explanation).
- Cross-view agreement regularisation on unlabelled inputs across **heterogeneous frozen encoders**,
  deployed single-view. Dormant since the 2005-2011 kernel era; the deep era replaced it with
  single-encoder consistency and single-encoder transduction. Its premise is damaged here by §1 —
  the views differ in strength by 5 to 21 macro-F1 points on 3 of 4 datasets — and it must be
  positioned against `2602.00132` (SCANNER, AAAI 2026, test-time adaptation on hate video,
  +4.69 % macro-F1) and `2501.01709` (MoVE-KD, CVPR 2025, mixture of encoders distilled into one).

**Mandatory control arms for anything in the first bullet**, both flagged by the objective sweep as
free mechanisms that could explain a pairwise gain away: logit adjustment (`2007.07314`) and
balanced-softmax classifier retraining (`2607.09832`). A pairwise loss over sampled positive x
negative pairs is an implicit class balancer, and `2512.01766` (May 2026) attributes last-layer
retraining's benefit to exactly that.
