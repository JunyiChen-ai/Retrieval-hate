# R7-1 — annotator-vote soft-label training: design and decision rule, frozen 2026-08-17

Frozen **before any seed in the run range 100-129 is executed**. Zero API cost, zero cloud cost.
Local RTX 5090; one head run measured at 8.9-9.3 s.

## 0. Prior-record check (done before this design was written)

Searched `TARGET_FINDINGS.md`, `idea-stage/IDEA_REPORT.md`, `refine-logs/*.md`, `research-wiki/log.md`.

| prior record | what it actually tested | does it kill this pilot? |
|---|---|---|
| `refine-logs/LITSWEEP5_HATEMM_EN.md` §Cand-1 + `refine-logs/C05PLUS_FORENSIC_RECON_2026-07-31.md` §4.2 — "graded 3-class Offensive soft-label" | A **label-oracle upper bound** on re-weighting rows whose **majority** class is `Offensive` to a constant softer target. Result: oracle ceiling EN **+0.0250** / ZH **+0.0256**, judged against a **+0.030** bar and killed. | **No.** (a) Different object: that target is a function of the *majority* 3-class label and is constant across all Offensive videos; this pilot's target is a function of the *per-annotator vote multiset* and varies per video (6 / 9 distinct values on ZH, see §2). (b) The bar it failed was **+0.030**, from the era before the user's "incremental gains are acceptable" ruling; the current bar is **+0.005**, and an oracle ceiling of +0.025 is 5x the current bar. A negative against a retired bar is not a kill under the current one. |
| `idea-stage/PILOT_A_RESULT.md` (P-A, GO) | Whether *contestedness* is retrievable from CLIP-neighbourhood geometry. Endpoint = AUROC of a retrieval score against disagreement. | No — different endpoint (retrievability of disagreement), no training-target change. |
| `idea-stage/LEG2_KILL_RESULT.md` | Vote-distribution **similarity as a contrastive pair-weighting function** (GenSCL cosine on vote histograms). Killed. | No — that is the contrastive pair topology; here the contrastive term is switched off entirely (`--contrast_mode none`) and only the BCE target moves. |
| `idea-stage/CN_VOTE_RECON.md` | Whether the `Counter Narrative` dissent value is usable stance supervision. NO at gate 3. | No — different use of the same file. Here CN is simply counted as non-positive (denominator only). |

**Conclusion: not previously tested, not previously vetoed. Proceed.**

## 1. Hypothesis

The training signal thrown away by majority-voting the 2-4 annotator votes into a binary label
carries usable information. Replacing the hard BCE target by the vote-derived soft target improves
test macro-F1 by at least the current bar, and does so by **more than an entropy-matched label
smoothing control** — i.e. the gain comes from what the votes say, not from the mere softness of
the target.

## 2. The target

For video *i* with annotator vote multiset `V_i` over `{Normal, Offensive, Hateful, Counter Narrative}`:

```
p_i(w) = ( #Hateful + w * #Offensive ) / |V_i|,     w in {1.0, 0.5}
```

`Counter Narrative` is counted as **non-positive** (it enters the denominator only). `w` is the
frozen choice of whether `Offensive` counts as hate (the project's deployed binary label is
`Hateful ∪ Offensive`, so `w = 1.0` is the reading that matches deployment; `w = 0.5` is the
alternative reading). Both are run; neither is tuned.

Built by `idea-stage/r7_softvote/build_targets.py` (train split only; an explicit guard aborts on
any vote file whose name is not `*_train.tsv`). Realised values, computed **before** any arm was
run and independent of any model:

| dataset | arm | mean target | frac. targets strictly in (0,1) | distinct values | corr with hard label | mean binary entropy (nats) | entropy-matched LS eps |
|---|---|---|---|---|---|---|---|
| MHC_zh (n=579) | SOFT10 | 0.3199 | 0.181 | 6 | 0.949 | 0.1149 | **0.02445** |
| MHC_zh | SOFT05 | 0.2191 | 0.370 | 9 | 0.898 | 0.2221 | **0.05823** |
| MHC (EN, n=549) | SOFT10 | 0.2940 | 0.098 | 5 | 0.975 | 0.0625 | **0.01143** |
| MHC | SOFT05 | 0.1846 | 0.308 | 7 | 0.918 | 0.1929 | **0.04810** |

Annotators per video: ZH 2/3/4 for 422/151/6 videos; EN 2/3/4 for 461/87/1.

**One upstream inconsistency logged, not fixed:** ZH train video `BV1jW4y1n7fP` carries votes
`['Offensive','Offensive']` (so `p = 1.0`) while the release's `Majority_Voting` column says
`Normal` and the project binary label is `0`. The project's binary label is left untouched
(it is what evaluation scores against); only the training target for this one row disagrees with it.
This is 1 of 579 rows.

## 3. Arms and grid

Only the BCE **training target** changes. Features, head, optimiser, epochs, batch size, dropout,
fusion, seed streams and **all evaluation** are identical to `idea-stage/r6_confirm/run_confirm.sh`.
Evaluation always scores against the hard cached labels.

| arm | training target |
|---|---|
| `A0` | hard binary label (the current baseline) |
| `SOFT10` | `p_i(1.0)` |
| `SOFT05` | `p_i(0.5)` |
| `LS10` | hard label smoothed by `eps` matched to SOFT10's mean entropy |
| `LS05` | hard label smoothed by `eps` matched to SOFT05's mean entropy |

Binary label smoothing is `y' = (1-2*eps)*y + eps`, i.e. positives -> `1-eps`, negatives -> `eps`.

- **Datasets**: `MHC_zh` (cache `R6RO-A0`, = generic-LoRA `ro_L28`, the R6 baseline substrate) and
  `MHC` (cache `Qwen2.5-VL-7B-Instruct-LoRA_HF`; no `ro_` cache exists for EN, so EN's A0 is that
  cache and EN numbers are **never** compared against the ZH/HateMM `ro_L28` ledger — only against
  the EN A0 run inside this same grid).
- **Seeds**: 100-129, 30 per arm per dataset. Disjoint from the R6 audit (0-29) and the R6
  confirmation (30-89).
- **Grid**: 2 x 5 x 30 = **300 head runs**, ~48 min.
- **Read-out**: **P1 (primary)** = epoch selected on the val split by validation macro-F1, ties to
  the earliest epoch, epochs >= warmup 5; test macro-F1 at threshold 0.5. **P2 (corroboration)** =
  final epoch (29), threshold 0.5. The `(dev acc, dev roc)` key is **not** used.
  Parser, confusion-matrix reconstruction and epoch selector are imported verbatim from
  `idea-stage/r6_audit/analyze_audit.py`. Split sizes: `MHC_zh` test N=149 P=45; `MHC` test N=161 P=49.

## 4. Data discipline

- Soft targets are built **only** from `mhc_{Chinese,English}_train.tsv`. The `valid` and `test`
  vote files are never opened; `build_targets.py` halts if asked.
- Epoch selection uses the val split's **hard** macro-F1.
- Test labels are read only for the final metric. No threshold, no epoch rule, no arm definition,
  no `w`, and no `eps` is selected on test. **The test-split votes are never used for anything.**

## 5. Frozen decision rule

Per dataset, paired seed-wise over the 30 seeds, paired bootstrap over seeds (20 000 resamples,
`default_rng(20260817)`), primary protocol **P1**.

Let `w*` be the arm in `{SOFT10, SOFT05}` with the larger `mean(SOFT_w - A0)`. Because `w*` is a
max over two arms, its confidence interval is taken at the **Bonferroni-corrected 97.5 % level**
(percentiles 1.25 / 98.75). Both arms' uncorrected 95 % CIs are reported as well.

A dataset **PASSES** iff:

1. `mean(SOFT_{w*} - A0) >= +0.005`, **and**
2. that arm's Bonferroni-corrected CI excludes 0, **and**
3. `mean(SOFT_{w*} - LS_{w*}) > 0` (the soft target beats its own entropy-matched smoothing control).

Verdicts:

- **GO-2DS** — both datasets pass under P1 and P2 agrees in sign on both.
- **GO-1DS** — exactly one dataset passes under P1, P2 agrees in sign on it, and the other dataset
  has `mean(SOFT_{w*} - A0) >= -0.002` under P1 (no material harm).
- **TRICK** — conditions 1 and 2 hold on at least one dataset but condition 3 fails there. Recorded
  as a regularisation trick, **not** as a direction; no novelty search is run.
- **KILL** — anything else.

**Sanity quantity (not gating, reported):** `mean(LS_w - A0)` for both `w`, on both datasets. If a
plain smoothing control alone clears +0.005 while SOFT does not, that is the TRICK reading above.

## 6. If GO

Immediately run a novelty search (`annotator disagreement soft label training hate speech video`,
`learning from disagreement multimodal`, `LeWiDi video`). The text domain is known to be occupied
(`refine-logs/LITSWEEP5_HATEMM_EN.md` §1: LeWiDi-2025, DiADEM `2604.08425`, EDO `2607.08493`,
soft-label `2511.14117`, Socio-Contrastive `2604.18069`, RGPO `2607.20515`, Fornaciari et al.
NAACL 2021); `idea-stage/IDEA_REPORT.md` §Confirmed-clear records **zero** hateful-**video** work
using annotator vote distributions. Novelty is judged under the user's "occupancy is not an
automatic kill; done better + with insight is legitimate novelty" ruling.

## 7. Expected outcome, stated before the run

KILL or TRICK. Reasoning: on `SOFT10` only 18.1 % (ZH) / 9.8 % (EN) of train rows carry a target
strictly between 0 and 1, and correlation with the hard label is 0.949 / 0.975 — the target barely
moves. `SOFT05` moves more but changes the decision boundary's meaning (it down-weights the
`Offensive` half of the positive class, which `refine-logs/C05PLUS_FORENSIC_RECON_2026-07-31.md`
§4.2 records as *monotonically harmful* on ZH). This expectation is recorded so that a positive
result cannot be presented as having been anticipated.

## 8. Code touched

`src/model/loss.py` (`bce_target`, applied in the `contrast_mode == 'none'` branch),
`src/run_rac.py` (`--soft_target_json`, `--label_smoothing`, `--dump_head_scores`),
`src/utils/metrics.py` (flag-gated per-item logit dump, used by R7-2 not by this pilot).
All three flags default to off. Verified: with the flags off, `MHC_zh / A0 / seed 30` reproduces
`logging/runs/r6_confirm/logs/MHC_zh_A0_s30.trainlog` **line-for-line identically** on all 60
`dev`/`test` epoch lines.
