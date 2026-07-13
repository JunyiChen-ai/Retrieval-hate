# SAV (C2) F-G1 — INDEPENDENT VERDICT REVIEW

Date: 2026-07-14
Reviewer = **Claude Opus 4.8** (`claude-opus-4-8`, 1M ctx), fresh **independent verdict reviewer**,
zero prior context, CPU-only, no GPU/SLURM, no commits (archiver handles commits).

**Under review:** verdict `KILL` in `artifacts/sav_f0/probe/verdict.json` (job 13099), produced by
`scripts/analysis/sav_f0_probe.py` + `sav_f0_common.py` under the pre-registration
`research-wiki/experiments/exp-sav-f0.md` (Rev-2a) F-G1. Raw transcription:
`refine-logs/SAV_F0_EXECUTION_RECORD.md`.

**Mandatory background:** `refine-logs/C3_PROBE_VERDICT_REVIEW.md` (the L2-over-regularization
"crush" bug found in a sibling probe 2026-07-14).

---

## VERDICT (one line)

**KILL_CONFIRMED.** The F-G1 machinery IS defective — the L2 λ grid tops out at the strongest
regularization it offers (λ=100 ⇒ sklearn `C`=0.01), and inner-CV saturates there on **every arm ×
dataset**, so the probe is pinned below the regularization its features actually want. This is the
C3-class over-regularization crush, confirmed by the calibration mandate: a **known-perfect gold
one-hot appended to the pooled feature reaches only acc 0.85 / Fano 0.84 on MHC** (full headroom
would be ~1.0), and the probe+Fano pipeline reaches **exactly 1.0000 / 0.9999 when the same gold
column is un-crushed** — so the pathology is regularization, not the pipeline. **But the defect
inflated SAV's apparent advantage, not suppressed it:** the crush handicaps the higher-dimensional
pooled *baseline* (which falls to acc 0.6375 — *below* the 0.6875 majority-class baseline — with
codelength 1.11 bits/ex > 1.0, Fano-flooring its projection to 0.50). Under **corrected machinery**
(λ grid widened upward; per-arm CV now lands interior at λ≈1000, the pooled baseline un-floored to
acc 0.66 / 0.889 bits): (1) the MHC carrying cell **fails** the pre-declared rule — Δacc drops
+0.0875 → **+0.0525 with CI now including 0**, ΔL CI now includes 0, projected gain +0.046 with
CI-low **below** 0; (2) HateMM no-harm **fails more decisively** — SAV significantly *increases*
codelength and *lowers* projected accuracy (real harm on the encoder-swap's banked-win dataset);
(3) U-1 recovers from a crush artifact to ≈pooled, confirming no head subset carries information
beyond the pooled feature. Every corrected number moves **against** SAV. The pre-declared rules were
**correctly applied** by the machine, and they still return KILL under valid machinery. **The MHC
"+0.0875" positive cell is a crushed-baseline + Fano-floor artifact, not a stable signal** (§4).

---

## 1. λ-grid finding (Red Flag #1) — direction and consequence

`sav_f0_probe.fit_logreg_probe`: `Cs = (1.0 / C.LAMBDAS)`, `LAMBDAS = np.logspace(-4, 2, 7) =
[1e-4 … 1e2]`, and `chosen_lambda = 1.0/clf.C_[0]`. sklearn `C` is the **inverse** regularization
strength, so:

| grid λ | sklearn C = 1/λ | meaning |
|---|---|---|
| 1e-4 | 1e4 | weakest reg on grid |
| **1e2 (grid MAX, chosen everywhere)** | **1e-2** | **STRONGEST reg on grid** |

**So λ=100 = the grid's strongest available regularization, and inner-CV saturates at that edge on
all 13 arms × 3 datasets.** Direction: the CV wants **more** shrinkage than the grid allows. This is
the same mis-specified-grid pathology as C3 (there the CV pinned at `C=0.001`, the strong-reg edge of
*its* grid). A fixed-λ sweep (single seed, MHC) shows the true optima sit **past the grid ceiling**:

```
        lambda | pooled_acc pooled_ell pool_fano |  sav_acc  sav_ell sav_fano | projgain
         1e+02 |     0.6500     1.0900    0.5000 |   0.7125   0.8154   0.7474 |  +0.2474   <- deployed edge
         1e+03 |     0.7000     0.8793    0.7016 |   0.7375   0.8092   0.7513 |  +0.0497
         1e+04 |     0.6875     0.8546    0.7206 |   0.6875   0.8648   0.7130 |  -0.0076
[ref] constant base-rate predictor: ell=0.8962b; majority-class acc=0.6875
```

The pooled baseline's optimum is λ≈1e3–1e4 (STRONGER than the ceiling). At the deployed λ=100 the
3584-d pooled probe is **under-regularized → overfits noise → val codelength 1.09 > 1.0 bit/ex** (worse
than a constant base-rate predictor) and **acc 0.65 < the 0.6875 majority baseline**. Because the
lower-dimensional SAV arm (1280-d) sits near its own optimum already at λ=100, the single grid-capped
λ is fair to SAV but too weak for pooled — **the ceiling biases Δacc/ΔL/projGain in SAV's FAVOUR**
(the opposite sign to C3, where the crush hit an appended auxiliary column). Edge-saturation here is
NOT benign, but its bias direction protects the KILL.

## 2. Calibration mandate (Red Flag #2) — machinery FAILS, corrected re-run required

Per the 2026-07-14 mandate I ran a label-oracle under the SAME machinery (gold one-hot appended to
the pooled baseline and to SAV), plus a standalone gold arm and a C3-style un-crush.

| calibration arm (MHC / HateMM, seed 0) | acc | ell (bits) | Fano | λ chosen | headroom? |
|---|---|---|---|---|---|
| standalone gold one-hot (2-d) | 1.0000 / 1.0000 | 0.0001 | 1.0000 | ~0 | ✔ (perfect *standalone* signal registers) |
| **[pooled ⊕ gold] — DEPLOYED CV** | **0.8500 / 0.9252** | 0.629 / 0.212 | **0.842 / 0.966** | 100 / 1 | ✘ **MHC fails; crushed** |
| [SAV@10 ⊕ gold] — DEPLOYED CV | 0.9875 / 0.9907 | 0.023 / 0.056 | 0.998 / 0.994 | 0.1 / 1 | ≈ (lower-dim ⇒ less crush) |
| **[pooled ⊕ raw gold×50] — un-crushed (Z@λ100)** | **1.0000 / 1.0000** | 0.0018 / 0.0014 | **0.9999** | — | ✔ **pipeline is sound** |

**Reading:** appending a *known-perfect* feature to the 3584-d pooled feature under the deployed
machinery reaches only **Fano 0.84 on MHC** — the machinery discounts a perfect signal, the C3 crush.
The un-crush arm (standardize Z only, append raw gold×50 so it is effectively unpenalized) reaches
**0.9999**, proving the probe + holdout-log-loss + Fano *pipeline* is faithful; the **only** pathology
is the shared L2 penalty. Per the mandate ("if it does not [reach headroom], the machinery is invalid
… re-run with corrected machinery"), the deployed numbers are unreliable and the corrected re-run below
is authoritative. (Note the two crush faces need opposite fixes: the appended-oracle crush wants
*weaker* reg on the aux column; the decision-arm handicap wants a *wider grid upward*. No SAV **decision**
arm is a `[Z,A]` concat — all are standalone — so the correct fix for the decision is the wider grid;
the appended-oracle arm is a diagnostic only.)

## 3. Corrected machinery re-run (all decision arms, both gating datasets, same pre-declared rules)

**Correction (minimal, principled):** widen the L2 grid upward `LAMBDAS = logspace(-4, 5, 10)` so
per-arm inner-CV is not pinned at the edge; **everything else byte-identical** (StandardScaler,
`LogisticRegressionCV` inner 5-fold, holdout-log-loss bits, Fano projection, 10k example-level
clustered bootstrap, the pre-declared decision rules). 5 seeds. CV now lands interior (λ_mean≈1000).

**MHC (carrying; corrected pooled: λ̄=1000, acc 0.6600, ell 0.8886 — un-floored, proj 0.694):**

| arm | Δacc [95% CI] (x0) | ΔL bits [95% CI] (x0) | projGain [95% CI] (x0) | deployed→corrected |
|---|---|---|---|---|
| SAV@10 | **+0.0525 [−0.0125,+0.1225] (0)** | +0.0622 [−0.0157,+0.1446] (0) | +0.0463 [−0.0097,+0.1807] (0) | Δacc +0.0875(x0=1)→+0.0525(**x0=0**) |
| SAV@20 | +0.0550 [−0.0050,+0.1200] (0) | +0.0524 [−0.0207,+0.1284] (0) | +0.0398 [−0.0149,+0.1594] (0) | projG +0.16→+0.040 (<bar) |
| U-1 | +0.0175 [−0.0500,+0.0851] (0) | +0.0146 [−0.0894,+0.1128] (0) | +0.0120 [−0.1118,+0.1077] (0) | ΔL **−0.288 → +0.015** (recovered) |
| C-pos | +0.0075 [−0.0650,+0.0800] (0) | +0.0220 [−0.0591,+0.1053] (0) | +0.0177 [−0.0522,+0.1353] (0) | position control ≈ null |

**HateMM (no-harm; corrected pooled: λ̄=1000, acc 0.7888, ell 0.6728):**

| arm | Δacc [95% CI] (x0) | ΔL bits [95% CI] (x0) | projGain [95% CI] (x0) |
|---|---|---|---|
| SAV@10 | −0.0318 [−0.0785,+0.0150] (0) | **−0.0585 [−0.1085,−0.0063] (1, sig MORE bits)** | **−0.0280 [−0.0596,−0.0032] (1, sig neg)** |
| SAV@20 | −0.0243 [−0.0654,+0.0168] (0) | −0.0400 [−0.1017,+0.0173] (0) | −0.0188 [−0.0587,+0.0075] (0) |
| U-1 | −0.0224 [−0.0729,+0.0280] (0) | −0.0579 [−0.1257,+0.0036] (0) | −0.0277 [−0.0749,+0.0016] (0) |

**Pre-declared rule application under corrected machinery:**
- MHC SAV@10: `pass_deltaL`=False (ΔL CI includes 0) ∧ `pass_projgain`=False (mean +0.046 > 0.04 but
  ci_low −0.0097 < 0) ⇒ **mhc_pass=False.**
- MHC SAV@20: `pass_deltaL`=False, `pass_projgain`=False (mean +0.040 not > 0.04) ⇒ **mhc_pass=False.**
- HateMM SAV@10 no-harm: `ok_deltaL`=False (CI entirely > 0 bits) ∧ `ok_dacc`(ci_low ≥ −0.010)=False
  (ci_low −0.0785) ⇒ **noharm=False.** SAV@20 noharm=False.

Combined pre-registered rule (`proceed = mhc_pass ∧ hatemm_noharm`) ⇒ **KILL, on both counts,
independently.** The corrected numbers move uniformly against SAV.

## 4. The MHC "+0.0875" positive cell (Red Flag #3) — adjudicated: artifact, not signal

**(a) Is the projection-rule failure a conservatism artifact or substantive?** In the *deployed* run
the projGain CI-low was pinned at exactly **0** by a **Fano-floor conservatism artifact**: the crushed
pooled baseline's codelength (1.11 > 1.0 bit/ex) floors its Fano projection at 0.50, and any bootstrap
draw where the SAV arm's mean also crosses 1.0 bit yields gain 0. So the deployed failure *was* partly
a floor artifact — but it does not rescue the cell, because under **corrected** machinery the pooled
projection un-floors (0.694) and the projGain becomes **+0.0463 with CI-low −0.0097 < 0**: the rule now
fails for a **substantive** reason (the real projected-gain CI straddles 0), not the floor.

**(b) Stable signal or seed-luck?** Neither — it is a **dimensionality/regularization artifact against
a crushed baseline.** Per-seed MHC Δacc is sign-stable (all 5 positive: +.0625,+.0500,+.1875,+.0750,
+.0625), but (i) the pooled baseline is *below* the 0.6875 majority baseline in all 5 seeds; (ii) the
top-10 head sets are **near-disjoint across seeds** (5-seed Jaccard 0.02/0.07/0.11 for k=10/20/40;
intersection 0) yet every disjoint set scores ~0.70–0.78 — the fingerprint of "any low-dim slice
regularizes better than the crushed 3584-d pooled feature," not a localized sparse hate subspace;
(iii) **machinery-independent:** SAV's own native majority-vote read-out (cosine nearest-centroid, no
logreg, no λ) is **0.59 / 0.6125 / 0.6325 on MHC — below the 0.6875 majority baseline.** SAV's native
classifier is worse-than-trivial on the exact dilution-target dataset. Correcting the baseline shrinks
Δacc to +0.0525 (CI includes 0). The +0.0875 is a crushed-baseline artifact.

**(c) Does the instability invalidate the F-G2 carry-forward design?** The design's pinning of the
deployed head set to the **deterministic full-train** selection *pre-F-G1* is methodologically
legitimate (not back-fit to results). But the near-zero Jaccard shows the head-selection landscape is
flat / noise-dominated, so the carried-forward top-k is an **arbitrary draw with no reproducible signal
behind it** — the same "selection artifact" failure mode as the campaign's withdrawn archive-as-key
claim. This is moot: F-G1 fails under valid machinery, so F-G2 is never reached. The instability does
not void the design's provenance discipline; it does confirm there is nothing stable to carry.

## 5. HateMM no-harm violation (Red Flag #4) — REAL harm, strengthened by correction

Deployed SAV@10 Δacc −0.0187 [−0.0729,+0.0355] (CI included 0; failed no-harm only via the wide CI-low).
Under corrected machinery the harm is **larger and now significant in codelength/projection**: Δacc
−0.0318 [−0.0785,+0.0150], **ΔL −0.0585 [−0.1085,−0.0063] (CI entirely > 0 bits saved by pooled)**,
**projGain −0.0280 [−0.0596,−0.0032] (CI entirely negative).** The fixed-λ sweep shows the deficit
*grows* as the pooled baseline is properly regularized (λ=1e3: pooled 0.785 vs SAV 0.748). This is a
**genuine degradation** from swapping pooled→SAV features on HateMM — the dataset where the
encoder-swap positive lives — not a machinery artifact. It fails the pre-declared HateMM no-harm kill
number under both deployed and corrected machinery.

## 6. U-1 worse-than-pooled (Red Flag #5) — crush artifact that resolves to a clean null

Deployed U-1 (full 784-head concat, 100,352-d) was worse than pooled everywhere (MHC ΔL −0.288, HateMM
−0.405). This was **over-regularization crush**, not genuine dilution: 100,352 dims need λ≫100, so at
the ceiling U-1 is massively under-regularized and overfits. Under the corrected wide grid, U-1 CV
lands at λ≈1e4–8e4 and **recovers to ≈pooled**: MHC ΔL +0.0146 [−0.089,+0.113], Δacc +0.0175 (CI
includes 0); HateMM ΔL −0.0579 (CI includes 0). So the *deployed* U-1 number was a crush artifact, but
the *corrected* reading still supports the pre-declared tie-breaker interpretation as a clean **null**:
the full head-space concatenation carries no label information beyond the pooled feature (U-1 ≈ pooled ⇒
nothing to mine), reached honestly only after fixing the regularization. It is neither a source of gain
nor of harm.

## 7. Were the pre-declared rules correctly applied by the machine?

**Yes.** `decide()` faithfully implements the pre-registered logic: `_k_pass_mhc` = (ΔL mean>0 ∧ ci_low>0)
∧ (projGain mean>0.040 ∧ ci_low>0); `_noharm_hatemm` = (ΔL ci_high≥0) ∧ (Δacc ci_low ≥ −0.010); combined
`proceed = mhc_pass ∧ hatemm_noharm`. Applied to the deployed numbers it correctly returns KILL, and
applied to the corrected numbers it *still* returns KILL. I do **not** relitigate the rules (they were
pre-registered); I confirm only that (i) the machine applied them correctly and (ii) the machinery that
produced the intermediate numbers is defective in a direction that, once corrected, leaves the KILL
intact and strengthened.

## 8. Adjudication + justification

**KILL_CONFIRMED.** The F-G1 probe machinery is genuinely defective — its L2 λ grid ceiling (λ=100 =
strongest available) sits below the arms' true regularization optima, inner-CV saturates at that edge
on every arm, and the mandated calibration confirms the C3-class crush (a perfect gold feature appended
to the pooled baseline reaches only Fano 0.84 on MHC; the un-crushed pipeline reaches 0.9999). But the
defect **inflates SAV's apparent advantage** by handicapping the higher-dimensional pooled baseline
(driven below its own majority-class baseline and Fano-floored), so correcting it can only hurt SAV. It
does: under machinery whose calibration is restored and whose every arm receives its own regularization
optimum, the MHC carrying cell fails the pre-declared rule (Δacc +0.0525 and ΔL both with CIs including
0; projected-gain CI-low below 0), HateMM no-harm fails more decisively (SAV significantly increases
codelength and lowers projected accuracy), and U-1 resolves to a clean pooled-equivalent null. Three
machinery-independent facts seal it: SAV's native majority-vote is below the majority baseline on MHC;
the discriminative head sets are near-disjoint across seeds yet equi-accurate (a dimensionality effect,
not a sparse subspace); and the HateMM harm grows as the baseline is un-crushed. The MHC "+0.0875"
positive is a crushed-baseline + Fano-floor artifact. SAV is dead on the exact terms it pre-registered
to satisfy, and the machine reached the right verdict — the machinery flaw did not change the outcome,
only some intermediate numbers, all of which move against SAV once fixed. The dilution hypothesis is
falsified: MHC-EN is data/label-limited at this frozen-encoder read-out capacity (H0), not
pooling-limited.

**F-G2/F-G3 implication:** do not proceed. No corrected cell clears the pinned F-G1 rules on the
carrying dataset, and the no-harm control is violated. No cheap extra evidence would rescue it — the
only lever (regularization) has been exhausted in SAV's favour and it still fails; SAV's native
read-out is sub-trivial on MHC; and open-weights 7B is the ceiling (per prior campaign walls).

## 9. Provenance / reproduction

- Artefacts reviewed: `artifacts/sav_f0/probe/verdict.json`, `refine-logs/SAV_F0_EXECUTION_RECORD.md`,
  `research-wiki/experiments/exp-sav-f0.md`, `scripts/analysis/sav_f0_{probe,common}.py`,
  `artifacts/sav_f0/guard/*/guard.json` (all guard pass=True, min_cosine=1.0 — extraction faithful,
  feature caches sound, so this CPU re-analysis on them is valid).
- This review's diagnostics (persisted next to this doc, conda `HateVideo`, CPU only, no GPU/SLURM/net):
  `refine-logs/sav_f1_review_diag.py` (reproduction — matches verdict to all digits),
  `refine-logs/sav_f1_review_calib.py` (label-oracle calibration + fixed-λ sweep),
  `refine-logs/sav_f1_review_corrected.py` → `refine-logs/sav_f1_review_corrected_out.json`
  (C3-style un-crush + wide-grid 5-seed corrected re-run). Imports the ACTUAL probe machinery
  (`sav_f0_probe`, `sav_f0_common`) so reproduction is byte-faithful.
- Data (read-only, same per-video extract caches the probe consumed): `artifacts/sav_f0/extract/`;
  gold train+val labels used for probe/calibration only (never in-method); no TEST labels touched.
  Not committed (archiver handles commits).

## Required statements
- No performance/accuracy claim on any held-out test benchmark. All accuracy/codelength numbers are
  train/val cross-validation used solely to measure conditional information and audit probe validity.
- Gold read = train + val `labels` (and a diagnostic label one-hot), probe/calibration only.
- Write scope = this file + `sav_f1_review_{diag,calib,corrected}.py` + `sav_f1_review_corrected_out.json`
  under `refine-logs/`. Not committed. No prereg/config/CLAUDE.md/settings mutated.
