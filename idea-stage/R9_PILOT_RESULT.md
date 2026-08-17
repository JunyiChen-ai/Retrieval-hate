# R9-1 ANCHOR-INT — **KILL, 0 of 3 datasets**

**Frozen design**: `idea-stage/R9_PILOT_FREEZE.md`, commit `20ab02b`, committed before the pilot
code existed. Raw grid `idea-stage/r9_anchor/results.json` (525 runs, 0 failures, 0 NaN),
verdict `idea-stage/r9_anchor/verdict.json`, log `logging/runs/r9_anchor/run.log`.
Analyzer run exactly once on the complete grid. **¥0 API, ~9 min GPU.**

## 1. The grid

Mean test macro-F1 by α (α = 0 frozen Qwen2.5-VL-7B, α = 1 its LoRA-adapted version; the arm is
the L2-renormalised convex combination of both pooled streams):

| dataset | seeds | α=0.0 | 0.2 | 0.4 | 0.5 | 0.6 | 0.8 | α=1.0 |
|---|---|---|---|---|---|---|---|---|
| HateMM | 15 | 0.8547 | 0.8636 | 0.8639 | 0.8668 | 0.8697 | **0.8749** | 0.8698 |
| MHC-EN | 30 | 0.7292 | 0.7365 | **0.7450** | 0.7326 | 0.7027 | 0.7079 | 0.7235 |
| MHC-ZH | 30 | 0.7662 | 0.7724 | 0.7797 | 0.7780 | 0.7930 | 0.8004 | **0.8017** |

Mean **validation** macro-F1 by α — the quantity the frozen rule uses to select α:

| dataset | α=0.0 | 0.2 | 0.4 | 0.5 | 0.6 | 0.8 | α=1.0 |
|---|---|---|---|---|---|---|---|
| HateMM | **0.8665** | 0.8660 | 0.8567 | 0.8539 | 0.8560 | 0.8535 | 0.8505 |
| MHC-EN | **0.7750** | 0.7732 | 0.7587 | 0.7519 | 0.7545 | 0.7653 | 0.7691 |
| MHC-ZH | 0.8492 | 0.8660 | 0.8677 | **0.8705** | 0.8652 | 0.8606 | 0.8488 |

## 2. Verdict against the frozen rule

| dataset | α* (val) | endpoint | Δ vs endpoint | 95 % paired CI | interior | size | CI | mechanism | result |
|---|---|---|---|---|---|---|---|---|---|
| HateMM | 0.0 | 1.0 | **−0.0151** | [−0.0381, +0.0086] | no | no | no | no | FAIL |
| MHC-EN | 0.0 | 0.0 | 0.0000 | [0, 0] | no | no | no | no | FAIL |
| MHC-ZH | 0.5 | 1.0 | **−0.0237** | [−0.0523, +0.0025] | yes | no | no | no | FAIL |

**0 of 3 → KILL.** No interior mixture beats the better endpoint on any dataset, and the one
dataset where the selection rule did choose an interior α (MHC-ZH, α*=0.5) lands 2.4 macro-F1
points *below* the adapted endpoint it was supposed to improve on.

Clause 4 (mechanism) is informative on its own. On MHC-ZH the α*=0.5 arm does cut the break rate
P(wrong | frozen correct) from 0.0333 to 0.0167 — a 50 % reduction, clearing the reviewer's 25 %
threshold — but it simultaneously drops the repair rate P(correct | frozen wrong) from 0.2759 to
0.1034, i.e. it keeps only **37 %** of the repairs against a required 80 %. This is exactly the
failure mode the reviewer predicted when he insisted clause 4 be two-sided: interpolation does not
separate the repairs from the breaks, it shrinks both toward the frozen model in proportion.

## 3. The second, unplanned finding: validation cannot select α on the two English splits

On HateMM the val curve is monotonically **decreasing** in α while the test curve is monotonically
**increasing** over the same grid (Spearman −1 vs +0.86); on MHC-EN val prefers α=0 while test
prefers α=0.4. Validation splits here are 107 and 80 items, where one item is worth ~1 macro-F1
point, and val is drawn from the same pool the LoRA was trained on — so the adapted representation
looks worse on val than it is on test. Any future mechanism on this axis that needs a
validation-selected hyper-parameter inherits this defect, and it is not fixable by adding seeds:
it is a property of the split sizes and of the LoRA's exposure to the train pool.

## 4. What the KILL does and does not establish

**Establishes**: within the frozen-feature substrate, there is no point between the frozen and the
adapted representation that keeps the adaptation's repairs and drops its breaks. The break/repair
trade-off is a single scalar dial, not two separable populations.

**Does not establish**: that the weight-space version `W_α = W_0 + α·s·B·A` behaves the same way.
Feature-space interpolation is not weight-space interpolation for a transformer
(`f_{(1−α)θ0+αθ1}(x) ≠ (1−α)f_{θ0}(x) + α f_{θ1}(x)`), and this limitation was declared in §7 of
the freeze before the run. The pilot was designed as a screen that is informative when it fails:
the more expressive of the two operations found nothing, so the less expressive one is not worth
the extraction passes it would cost.
