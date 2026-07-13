# C3 oracle-ceiling probe — INDEPENDENT VERDICT REVIEW

Date: 2026-07-14
Reviewer = **Claude Opus 4.8** (`claude-opus-4-8`, 1M ctx), fresh **independent verdict reviewer**,
zero prior context, CPU-only, no GPU/SLURM, no commits (archiver handles commits).

**Under review:** verdict `TARGET_CONTENT_CAPPED` in `refine-logs/C3_G0COND_ORACLE_PROBE.md`
(script `scripts/analysis/c3_g0cond_oracle_probe.py`, results `refine-logs/C3_G0COND_ORACLE_PROBE_OUT.json`).

**Trigger:** the probe's diagnostic **label-oracle** arm — feeding the GOLD LABEL itself as the
auxiliary feature A at coverage 1.0 — reported direct-probe Δacc of only **+0.0473** (CLIP) /
**+0.0140** (Qwen), where a perfectly separating feature should drive the probe to ~1.0 accuracy
(Δacc ≈ +0.176 / +0.162). If the machinery crushes auxiliary features, the target arm (+0.0145 /
+0.0035) is also understated and the CAPPED verdict may overreach.

## VERDICT (one line)

**OVERTURNED.** The label-oracle gap is a real machinery bug — a shared heavy L2 penalty
(C=0.001) applied to StandardScaler-standardized `[Z, A]` columns crushes the low-dimensional
auxiliary column so a *known-perfect* feature is credited only 27% (CLIP) / 9% (Qwen) of its true
value. Under corrected machinery that restores the label oracle to its true ceiling (accZA =
**1.0000**, Δacc = full headroom +0.1761 / +0.1616 on both encoders), the **oracle-target Δacc rises
from +0.0145/+0.0035 to +0.0487/+0.0487** — point estimate **above** the +0.040 bar on both
encoders, 95% CI **[+0.022, +0.075] (CLIP) / [+0.030, +0.069] (Qwen)**. The doc's central claim
("even a perfect target channel is ≤ +0.0204, roughly half the bar") is a machinery artifact. The
oracle **ceiling clears the bar**, so the kill-switch precondition ("even the perfect oracle < +0.040")
is FALSE and `TARGET_CONTENT_CAPPED` cannot stand as written. The C3 target-content decision must be
**re-made** on the merits of a real (imperfect) predictor — see §6 for the important nuance that the
overturn does NOT mean a real target C3 will work.

**A-line PAUSE impact: NO** — the A-line kill's binding constraints are machinery-independent (§7).

---

## 1. Anomaly diagnosis — MECHANICAL CAUSE (adjudicated)

The gap is **over-regularization that crushes the auxiliary columns**, exactly as hypothesized. It is
NOT the arm feeding the wrong feature (the label one-hot `A_lab[arange(n), y] = 1` is verified correct),
NOT non-convergence (lbfgs converges in 20–35 iters, well under `max_iter=2000`), NOT a bootstrap/CI
issue.

**Coefficient evidence (in-sample fit, Qwen `[Z, one-hot(label)]`):**

| C | penalty 1/(2C) | train acc | \|w_label\| (mean of 2 cols) | mean \|w_Z\| | logit from label |
|---|---|---|---|---|---|
| 0.001 | 500 | 0.9637 | **0.103** | 0.00279 | ≈ 0.10·2.04 ≈ **0.21** (tiny) |
| 100   | 0.005 | 1.0000 | **1.786** | 0.04376 | ≈ 1.79·2.04 ≈ **3.6** (saturating) |

The standardized label column separates the classes by ≈2.04 units (base rate 0.40 ⇒ standardized
values {+1.22 for y=1, −0.816 for y=0}). At C=0.001 the shared L2 budget (penalty 500·‖w‖²) shrinks
the label coefficient to 0.10, so its logit contribution (~0.21) is far too small to override the
1792-/7168-dim Z contribution → held-out predictions are Z-dominated, not label-dominated → held-out
accuracy stays ≈0.82–0.85 instead of ≈1.0. StandardScaler then *equalises* the per-column scale of
the 2-dim label one-hot with the thousands of Z columns, so under one global C the auxiliary column
has no scale advantage and competes on equal quadratic-penalty footing with noise directions in Z.

**Regularization sweep (StandardScaler-over-all, shared C; the doc's own machinery):**

| C | CLIP accZ | CLIP label Δacc | CLIP target Δacc | Qwen accZ | Qwen label Δacc | Qwen target Δacc |
|---|---|---|---|---|---|---|
| **0.001** (doc) | 0.8266 | +0.0475 | +0.0121 | 0.8392 | +0.0148 | +0.0040 |
| 0.01 | 0.8168 | +0.0914 | +0.0188 | 0.8302 | +0.0367 | +0.0063 |
| 0.1 | 0.7997 | +0.1272 | +0.0296 | 0.8266 | +0.0439 | +0.0094 |
| 1.0 | 0.7966 | +0.1470 | +0.0282 | 0.8239 | +0.0694 | +0.0103 |
| 10 | 0.7867 | +0.1478 | +0.0332 | 0.8096 | +0.0627 | +0.0031 |
| 100 | 0.7800 | +0.1492 | +0.0358 | 0.8091 | +0.0390 | +0.0049 |
| 1000 | 0.7755 | +0.1532 | +0.0399 | 0.8082 | +0.0399 | +0.0045 |

Smoking gun: on CLIP the label Δacc climbs monotonically from +0.0475 → +0.1532 as the penalty is
relaxed, tracking toward the full headroom +0.176, and the target Δacc climbs in lock-step
(+0.0121 → +0.0399). At C=0.001 both are crushed. On Qwen the *symmetric* sweep cannot fully un-crush
the label because relaxing C also over-fits the 7168-dim Z (accZ collapses), which is precisely why a
Z-preserving asymmetric fix is required (§3).

## 2. Methodology diff vs the A-line probe

The C3 script (`scripts/analysis/c3_g0cond_oracle_probe.py`) reuses the A-line machinery
(`refine-logs/lb_scgp_global/M1_G0COND_PROBE.py`) essentially **verbatim** on the axes that matter to
the anomaly. Identical on both: A enters as `np.concatenate([Z, A], axis=1)`; `Pipeline(StandardScaler,
LogisticRegression(C, max_iter=2000))` standardizes **all** columns incl. the A one-hot; **same C for
g(Z) and g'([Z,A])**; C picked by inner 5-fold CV on **Z-only** over {1e-3,1e-2,1e-1,1} (both C3
encoders → C=0.001); `RepeatedStratifiedKFold(5×5)`; MDL bits `−log2 p`; per-video (example-clustered)
bootstrap B=5000. The C-grid never exceeds 1.0, so neither script can escape the crush. **The bug is
inherited, not introduced by C3.** Confirmation: the A-line's own `oracle_full` arm (gold LABEL at
coverage 1.0) also fails to reach ~1.0 — MHC/CLIP accZ 0.7621→0.8805 (+0.1184, only ~50% of the +0.238
headroom); MHC/Qwen +0.0397 (~20% of +0.195). Same understatement, same cause.

## 3. Corrected machinery (fixes the label oracle to behave sanely)

The corrected probe changes **only the treatment of A**, leaving Z's treatment byte-identical to the
original: standardize **Z alone** (fit on the train fold, exactly as StandardScaler does column-wise
inside `[Z,A]`), keep Z regularized at its CV-optimal **C=0.001**, and append A as **raw one-hot × s**
(s large ⇒ A effectively unpenalized: a logit of order 10 needs w≈10/s, whose L2 cost 500·(10/s)² → 0).
This isolates the crush without the confound of over-fitting Z. Verification that Z is untouched: the
corrected Z-only baseline reproduces the original exactly (CLIP 0.8239, Qwen 0.8384).

**Label-oracle under corrected machinery — now SANE (calibration check passes):**

| encoder | orig label Δacc | corrected label accZA | corrected label Δacc | full headroom (1−accZ) |
|---|---|---|---|---|
| CLIP | +0.0473 | **1.0000** | **+0.1761** | 0.1761 ✔ exact |
| Qwen | +0.0140 | **1.0000** | **+0.1616** | 0.1616 ✔ exact |

The perfect separator now yields perfect held-out accuracy and Δacc equal to the analytic headroom to
the last digit → the corrected machinery is a faithful ceiling-measurement. (Robust for all s ≥ 50;
Δbits label ≈ +0.53–0.59, CI excludes 0.)

## 4. Corrected TARGET-oracle numbers (decision arm)

Primary (s=50, where both label oracles hit exactly 1.0), 5×5 CV, per-video bootstrap B=5000:

| encoder | Z-only accZ | corrected target accZA | **corrected target Δacc [95% CI]** | corrected Δbits/vid [95% CI] |
|---|---|---|---|---|
| CLIP (0.8239) | 0.8239 | 0.8726 | **+0.0487 [+0.0220, +0.0750]** | +0.1648 [+0.1241, +0.2046] |
| Qwen (0.8384) | 0.8384 | 0.8871 | **+0.0487 [+0.0298, +0.0685]** | +0.1391 [+0.0976, +0.1805] |

vs the doc's original oracle-target: CLIP +0.0145 [+0.0091,+0.0204], Qwen +0.0035 [+0.0013,+0.0059].
**The corrected mean is ~3.4× (CLIP) / ~14× (Qwen) the doc's, and above the +0.040 bar on both.**

**Scale-robustness (target Δacc is not an s-artifact):**

| s | CLIP label accZA | CLIP target Δacc [CI] | Qwen label accZA | Qwen target Δacc [CI] |
|---|---|---|---|---|
| 10  | 1.0000 | +0.0460 [+0.026,+0.066] | 0.9809 | +0.0368 [+0.023,+0.050] |
| 50  | 1.0000 | +0.0487 [+0.022,+0.076] | 1.0000 | +0.0487 [+0.030,+0.069] |
| 100 | 1.0000 | +0.0457 [+0.020,+0.072] | 1.0000 | +0.0457 [+0.026,+0.065] |
| 200 | 1.0000 | +0.0457 [+0.019,+0.073] | 1.0000 | +0.0446 [+0.025,+0.064] |

Corrected target Δacc is stable at **+0.045–0.049** on both encoders wherever the label oracle is fully
un-crushed (s ≥ 50). Cross-check with a symmetric fix: in the StandardScaler-all sweep (§1), pushing
C to 1000 (closest the symmetric machinery gets to a sane label on CLIP, +0.1532) brings CLIP target
to **+0.0399** ≈ the bar — the same conclusion by an independent route.

## 5. Why target is genuinely worth ~+0.049 (not spurious)

The corrected A is a 9-column one-hot: it can add only a **per-category logit offset** (9 d.o.f.,
very low variance, cannot memorise individual videos), so the +0.049 is not overfitting. Its source is
the strongly label-skewed dominant categories (`target_facts`, this session):

target-alone majority-vote Bayes acc = **0.7903** — per-cat [non-hate, hate]:
`-1:[32,0] · Blacks:[107,225] · Jews:[27,45] · Whites:[8,10] · Others:[264,6] · LGBTQ:[3,3] ·
Muslims:[2,5] · Sexits:[2,3] · Asian:[1,1]`. Categories "Blacks" (68% hate) and "Others" (2% hate)
are 602/744 = 81% of videos and near-deterministic in opposite directions, so a per-category prior
corrects many of Z's residual errors. Note target-alone (0.790) is **below** Z-alone (0.824/0.838):
target's value is purely *complementary* to Z, which is exactly why its marginal-over-Z gain is real
but modest and sensitive to how the probe apportions capacity — the crush drove it to near-zero, a
faithful probe puts it at ~+0.049.

## 6. VERDICT — OVERTURNED (with nuance the re-decision must carry)

- **CONFIRMED?** No. It requires corrected CI upper < +0.030; the corrected CI upper is +0.075 (CLIP) /
  +0.069 (Qwen).
- **OVERTURNED?** **Yes.** The doc's decision rule is a kill-switch: "C3 target-content is CAPPED unless
  the *oracle* target gain projects ≥ +0.040." The oracle is meant to be the **ceiling**. The doc
  measured that ceiling with machinery that discounts a *known-perfect* feature to 9–27% of truth, so
  its "ceiling" (≤ +0.0204) was a heavy under-estimate, not a ceiling. The faithful ceiling (validated
  by the label oracle reaching exactly 1.0) is **+0.0487 on both encoders — above the bar**. The
  kill-switch precondition is therefore false; `TARGET_CONTENT_CAPPED` does not hold as written and the
  C3 target-content decision must be re-made.

- **Nuance the re-decision MUST keep (this overturn is narrow):** the oracle is a *perfect-predictor,
  full-coverage* upper bound. (i) A real C3 MLLM predicts target **imperfectly**; (ii) it is delivered
  through the weaker **top-20 kNN** channel, not a direct linear head; (iii) the corrected ceiling
  (+0.0487, CI down to +0.022/+0.030) sits **at/just above** the bar, not comfortably above; (iv) the
  project's C2 rule demands +0.030 on multiple metrics × seeds × Holm. So a *real* target-content C3
  remains a **marginal** bet and could well fail the full protocol. The correct statement is **"the
  oracle ceiling clears the bar → you cannot foreclose the target channel on the ceiling; decide it on
  a real predictor,"** not "the target channel is promising." The doc's practical caution survives; its
  specific quantitative claim (ceiling ≤ +0.0204, "cannot carry C3," "pivot away from target") does not.

**Corrected headline numbers for the record:**
`oracle_target Δacc = +0.0487 [+0.0220,+0.0750] (CLIP) / +0.0487 [+0.0298,+0.0685] (Qwen);
Δbits/vid = +0.1648 [+0.124,+0.205] / +0.1391 [+0.098,+0.181]; label-oracle Δacc = +0.1761 / +0.1616
(accZA = 1.0000, faithful ceiling).`

## 7. A-line PAUSE impact assessment — NO

The label-oracle crush affects the A-line script identically, **but it does not touch the A-line kill's
binding constraints**, which are machinery-independent:

1. **Measured certificate (real-A) is flat/anti-informative.** `real_full` Δbits CI is entirely
   negative (MHC/CLIP [−0.0151,−0.0029], MHC/Qwen [−0.0090,−0.0008]) or straddles 0 (both MHC_zh
   cells); `real_covered` (the cert on videos where it *did* parse) has Δbits CI including 0 in all four
   cells. These are direct measurements of the real signal's quality — no oracle regularization involved.
2. **Analytic coverage ceiling `c·(1−a_cov)` ≤ +0.0277.** At coverage 8.7%/6.9%, a *perfect* oracle
   revealed only on covered videos can flip at most (coverage)×(error-rate-on-covered) of the set:
   MHC/CLIP +0.0240, MHC/Qwen +0.0277, MHC_zh/CLIP +0.0162, MHC_zh/Qwen +0.0183. This is arithmetic on
   the coverage and the covered-set accuracy — a logistic probe (crushed or not) cannot exceed it. Max
   +0.0277 < +0.040. **This is the A-line's binding kill and it is fully machinery-independent.**
3. The only A-line arm the crush distorts is **`oracle_full`** (gold label at coverage 1.0), which is
   the *non-binding* "v3-viability" reference, not a decision arm. Un-crushing it would push it **up**
   toward the ~+0.19–0.24 headroom — which, if anything, *strengthens* the observation "a gold-quality
   full-coverage cert could matter," but that ceiling is explicitly ruled unreachable by (1): the real
   cert is noise-quality where it parses, so a coverage-repair (v3) would propagate a zero-information
   signal, not the gold ceiling.

**Conclusion:** the A-line `A_LINE_PAUSE` verdict stands. Its kill rests on the measured cert being
flat and on the machinery-independent coverage cap +0.0277, neither of which the label-oracle bug can
move. (The bug is worth flagging as a **general machinery caveat** for any future full-coverage oracle
arm, but it changes no A-line decision.)

## 8. Provenance / reproduction

- Original artefacts reviewed: `refine-logs/C3_G0COND_ORACLE_PROBE.md`,
  `scripts/analysis/c3_g0cond_oracle_probe.py`, `refine-logs/C3_G0COND_ORACLE_PROBE_OUT.json`;
  A-line: `refine-logs/lb_scgp_global/M1_G0COND_PROBE.py`, `..._OUT.json`, `..._RECORD.md`.
- This review's diagnostics (persisted next to this doc, conda `HateVideo`, CPU, ~2 min):
  `refine-logs/C3_PROBE_VERDICT_REVIEW_diag.py` → `refine-logs/C3_PROBE_VERDICT_REVIEW_OUT.json`
  (repro C=0.001, C-sweep, corrected A-free s=50, per-arm-best-C, target-alone Bayes);
  `refine-logs/C3_PROBE_VERDICT_REVIEW_scalerobust.py` (s ∈ {10,50,100,200}).
- Data (read-only, same caches as the probe): `data/CLIP_Embedding/HateMM/train_{openai_clip-vit-large-patch14-336_HF,Qwen2.5-VL-7B-Instruct_HF}.pt`,
  `data/gt/HateMM/target_map.json`. Gold used probe-only (features/targets), never in-method. No
  validation/test content, no GPU/SLURM/network. Not committed (archiver handles commits).

## Required statements
- No performance/accuracy claim on any held-out benchmark; all accuracy/codelength numbers are
  train-only cross-validation used solely to measure conditional information / audit the probe.
- Gold read = `primary` target + train `labels`, probe-only. Write scope = this file +
  `C3_PROBE_VERDICT_REVIEW_diag.py` + `C3_PROBE_VERDICT_REVIEW_scalerobust.py` +
  `C3_PROBE_VERDICT_REVIEW_OUT.json` under `refine-logs/`. Not committed.
