# B5 VERDICT REVIEW — operating-point conversion of the frozen-Qwen MHC-ZH ranking advantage

**Reviewer:** fresh zero-prior-context independent VERDICT REVIEWER (did not design or run B5).
Read-only forensic adjudication; **NO GPU, NO SLURM**; CPU-seconds numpy/sklearn hand-checks only.
Zero interaction. This document renders the BINDING verdict against the pre-registered rules.
**Date:** 2026-07-15.

**Files read (primary, verified — not the executor's summary):**
`research-wiki/experiments/exp-conv-zh-b5.md` (prereg r0/r1/r2, amendments A1–A11),
`refine-logs/B5_PROBE_DESIGN.md` (executable spec + G-repro anchor tables),
`refine-logs/B5_PREREG_REVIEW.md` (binding review terms, A1/A2 blocking),
`refine-logs/B5_GATE_AMENDMENT_RULING.md` (A11 ruling of record, commit 5295076 + a08deed re-check),
`refine-logs/B5_PROBE_RECORD.md` (jobs 13156 CPU / 13158 cuda / 13170 CPU (b)–(e)),
`slurm/logs/b5probe_13170.out` (primary log, exit 0), `refine-logs/b5_probe_out/` (12 npz + results.json).

---

## VERDICT (one line)

**B5 is DEAD** — pre-registered category **K1 (§12): neither protocol is eligible at the oracle
ceiling** (per-protocol AND-eligibility fails under BOTH final-epoch and val-selected). The
label-oracle upper bound itself fails the +0.03/+0.03 AND-bar, so B5 is DEAD **regardless of the honest
preview** (§6.4), which independently also fails. The frozen-Qwen MHC-ZH ranking advantage (roc +0.050,
3/3 seeds) is **not convertible into the goal's decision metrics (acc + macro-F1) at any operating
point.** This is a **performance/diagnosis-only** death, **D7-irrelevant** (no novelty implication).

---

## A. A7 HAND-CHECK (mandated) — independent recomputation from the npz dumps: **CONFIRMS to 4 dp**

I authored an independent numpy/sklearn recomputation (my own grid + lower-median tie-break + sklearn
`f1_score(average='macro', zero_division=0)`, NOT the probe script) directly from the dumped
`votes_*`/`labels_*` arrays.

**Cell picked: `Qwen_s0_final`** (deliberately the highest-calibration-tax, most non-trivial slot:
frozen τ = −0.533151, well away from the deployed vote=0 cut). Paired control `CLIP_s0_final` also
checked.

| cell | frozen τ (record) | my τ (indep.) | honest test acc (mine / record) | honest test mF1 (mine / record) |
|---|---|---|---|---|
| **Qwen_s0_final** | −0.533151 | −0.533151 | **0.7517 / 0.7517** ✓ | **0.7380 / 0.7380** ✓ |
| CLIP_s0_final | +0.061628 | +0.061628 | 0.8121 / 0.8121 ✓ | 0.7771 / 0.7771 ✓ |

Both cells reproduce τ*, honest_acc, and honest_mF1 exactly to 4 dp from the raw votes. The
"votes-gated but calibration-ungated" gap A7 was created to close is **closed**: the grid /
tie-break / macro-F1-at-τ arithmetic in the probe machine is validated. **A7 = PASS.**

## B. A2 DEV ANCHOR CHECK (mandated) — frozen τ maximizes dev macro-F1: **CONFIRMED (6/6 slots)**

Design requires ≥2 slots; I verified **6** (spanning both encoders and both protocols). For each, I
rebuilt the dev unique-vote-midpoint grid independently and confirmed (i) the frozen τ equals my
independently re-derived argmax-dev-macroF1 τ to 1e-6, and (ii) the dev macro-F1 achieved at the frozen
τ equals the grid maximum.

| slot | frozen τ | my τ | dev mF1 @ frozen τ | dev mF1 max | is argmax? |
|---|---|---|---|---|---|
| CLIP_s0_final | +0.06163 | +0.06163 | 0.8106 | 0.8106 | ✓ |
| Qwen_s1_valsel | +0.11897 | +0.11897 | 0.8756 | 0.8756 | ✓ |
| Qwen_s0_final | −0.53315 | −0.53315 | 0.8022 | 0.8022 | ✓ |
| CLIP_s1_final | −0.66502 | −0.66502 | 0.7956 | 0.7956 | ✓ |
| Qwen_s2_valsel | +0.01900 | +0.01900 | 0.8301 | 0.8301 | ✓ |
| CLIP_s2_final | +0.22164 | +0.22164 | 0.7970 | 0.7970 | ✓ |

The dev-side selection anchoring holds: every frozen threshold is genuinely the dev-macroF1-optimal cut
under the pre-registered grid/tie-break. **A2 = PASS.** (Split balance independently confirmed from the
dumps: dev n=78 pos=28 = 0.3590; test n=149 pos=45 = 0.3020 — matches the 30/70 imbalance premise.)

## C. G-REPRO STATUS under A11 — **12/12 PASS** (governing amendment: A11 ruling, commit 5295076)

Job 13170 (CPU, script v4) reports G-repro **12/12 PASS** (log lines 35–48; `grepro_all_pass=true` in
results.json). Under the governing **amendment A11** (`B5_GATE_AMENDMENT_RULING.md`, AMEND-APPROVED
2026-07-15, commit 5295076; re-check a08deed → `CLEARED-FOR-CPU-CONTINUATION`) the gate is: test+dev
`acc` AND `macroF1` exact-4dp, AND test+dev `roc` within |Δ| ≤ 1e-3.

Independent re-derivation from results.json: all 12 slots' deployed test/dev acc and macroF1 match the
13115 anchors exactly; **max |roc Δ| across all 12 slots × {test,dev} = 0.0007 ≤ 1e-3.** The A11
amendment is properly adjudicated (roc is a non-redrawable cuBLAS rank statistic, provably unused
downstream — code-verified in the ruling §B.2: `select_tau`/oracle/honest/D3 consume only acc/mF1), so
the amendment relaxes nothing the conclusion rests on. **G-repro = PASS under A11**, confirmed on the
existing evidence. (The deployed vote SIGN — the only quantity the calibration consumes — is reproduced
exactly everywhere.)

## D. KILL-SWITCH RULING (pre-declared A1, ruled FIRST) — **KILL FIRES; B5 DEAD**

The oracle is the each-arm-own-test-optimal-τ ceiling (fair pairing). I recomputed the paired
Qwen−CLIP oracle deltas independently from the 12 npz dumps (my own oracle = max over the test
unique-vote grid, separately for acc and macro-F1):

| protocol | mean ΔAcc_oracle (mine / rec) | mean ΔmF1_oracle (mine / rec) | ELIGIBLE (AND ≥ +0.03)? |
|---|---|---|---|
| **final-epoch** | **+0.0022 / +0.0022** (2/3 +) | **+0.0213 / +0.0213** (3/3 +) | **NO** (both clauses < +0.03) |
| **val-selected** | **−0.0000 / −0.0000** (1/3 +) | **+0.0134 / +0.0134** (2/3 +) | **NO** (both clauses < +0.03) |

Per-seed oracle deltas independently reproduced (final ΔAcc [+0.0201,+0.0067,−0.0201], ΔmF1
[+0.0228,+0.0370,+0.0040]; valsel ΔAcc [−0.0134,−0.0067,+0.0201], ΔmF1 [−0.0009,+0.0182,+0.0230]) —
match the record and log 13170 §(c) to 4 dp.

**Ruling: neither protocol is eligible ⇒ the pre-registered KILL fires (A1 §6.4, K1 §12). B5 = DEAD.**
Even at a *perfect, per-encoder, per-metric, label-oracle* threshold the paired accuracy advantage is
essentially zero (+0.0022 final, −0.0000 valsel) and the paired macro-F1 advantage (+0.0213, +0.0134)
falls well short of the +0.03 bar. The ceiling fails ⇒ dead **regardless of the honest preview**, and
no formal GPU stage is authorized or warranted. Exhaustion is re-confirmed for this cell. Oracle
numbers are an upper bound and are NOT claimed as a result.

## E. CORROBORATING READS (recorded verbatim; honest preview independently re-verified)

**Honest val-calibrated preview** (frozen dev-τ → test; the deployable number; independently recomputed
from npz, matches to 4 dp):
- **final-epoch:** mean paired ΔAcc = −0.0157 (1/3 +), ΔmF1 = +0.0023 (2/3 +) → clears +0.03/+0.03 &
  3/3 = **False**.
- **val-selected:** mean paired ΔAcc = −0.0022 (1/3 +), ΔmF1 = −0.0118 (0/3 +) → clears = **False**.

The honest arm fails on both protocols and both metrics — so B5 dies even setting the oracle aside; the
oracle kill (D) is the binding, stronger statement (dead even at the ceiling).

**D3 bootstrap quantiles (1000 paired resamples, common index A6):**
- final: ΔAcc 5/50/95 = −0.0291 / +0.0022 / +0.0604 (5th ≤ 0 → **D3-fragile**); ΔmF1 = −0.0108 /
  +0.0144 / +0.0606 (**D3-fragile**).
- valsel: ΔAcc 5/50/95 = −0.0201 / +0.0067 / +0.0403 (**D3-fragile**); ΔmF1 = −0.0282 / +0.0006 /
  +0.0358 (**D3-fragile**). Both protocols, both metrics D3-fragile — the median paired Δ sits near
  zero with the 5th percentile below it.

**τ cross-seed stability (calibration-transfer diagnosis):**
Qwen τ std = 0.176 (final) / 0.104 (valsel); CLIP τ std = 0.386 (final) / 0.387 (valsel). CLIP's
dev-macroF1-optimal cut is **~2–4× less stable** across seeds than Qwen's (CLIP final τ =
[+0.062, −0.665, +0.222]). Read against the diagnosis: the higher-AUC Qwen scores yield a *more
reproducible* optimal threshold (consistent with genuinely better rank ordering), while CLIP's honest
calibration is itself noisy — yet **even Qwen's more stable, better-ranked scores do not separate any
better at the operating point** (D). The AUC edge lives in the ordering of far-from-boundary examples,
not in decision-boundary separability that a threshold can exploit; combined with the ~0.386 CLIP τ
instability, the calibration-transfer route is both non-convertible (oracle) and fragile (D3).

## F. BINDING VERDICT — **DEAD** (K1: neither protocol eligible at the oracle ceiling)

**Epitaph — what is NOW CLOSED.** ZH **per-encoder decision-threshold calibration as a conversion lever
for the frozen-Qwen AUC advantage** is closed. The B1-left-untouched sub-cell (calibrated-threshold
accuracy of the frozen-Qwen ZH representation) is now measured and negative at the ceiling. B1's
"unconverted AUC advantage" mystery is **answered**: the advantage is genuinely **non-convertible** — it
does **not** survive at *any* operating point (not the deployed vote≥0 cut, and not the label-oracle
best cut). The roc−acc "mis-calibration signature" (Qwen gap 0.086 vs CLIP 0.025) was **necessary but
not sufficient**: moving Qwen's threshold recovers ~2 acc points (calibration tax), but CLIP's own best
cut recovers a comparable amount, leaving the paired Qwen−CLIP gap at ≈0 on acc and <+0.03 on mF1. The
+0.050 AUC edge is an easy-example-ordering effect, not better near-boundary separability.

**What is NOT closed.** B5 was **performance/diagnosis only** and is **D7-irrelevant**: threshold
calibration per se is generic (prereg §9), and the novelty clause was a deferred D7-class user ruling
that is **never reached** because the performance clause fails first. Therefore this death opens and
closes **no** novelty-line question — the 4-pillar story, the encoder role on HateMM, the localizer, and
the guard-rail role are all untouched. This closes exactly **one** exhaustion-audit cell; it does not
re-open or foreclose any other axis. No headline/family claim is created (the ≥2-dataset bar is not
reached: HateMM remains the only formally passing dataset).

**What the diagnosis DID establish.** The advantage does **not survive at any operating point.** Oracle
ceilings: paired ΔAcc_oracle +0.0022 (final) / −0.0000 (valsel); paired ΔmF1_oracle +0.0213 / +0.0134.
A perfect threshold cannot convert the ranking advantage into a +0.03 decision-metric advantage — the
strongest possible negative on this cell. This is a decision-useful result: the exhaustion audit's lead
open cell is now a measured, ceiling-level negative, not an unexamined gap.

## G. HYGIENE — no provenance gaps; executor no-interpretation discipline HELD

- **Transcription (numeric-provenance discipline):** The record's §(a)–(e) tables match primary log
  `b5probe_13170.out` **verbatim** — I spot-checked the two kill-switch means (log L75/L82:
  +0.0022/+0.0213 and −0.0000/+0.0134), the DEAD flag (L84), and both full honest-preview tables
  (L90–102) against the record and results.json; all consistent. No 0.8732-class transcription drift.
- **Executor no-interpretation:** HELD. The (b)–(e) flags (ELIGIBLE / DEAD / clears / D3-fragile) are
  the probe script's **pre-declared mechanical rule outputs**, explicitly labelled as such in the record
  (§ handoff) and the log footer ("Executor applies NO pass/fail interpretation"); no scientific
  judgment was inserted by the executor. The binding verdict is rendered here, independently.
- **A11 governance:** proper — ruling commit 5295076, executor application a08deed re-checked to
  `CLEARED-FOR-CPU-CONTINUATION`; script v4 sha256 `3d075345…` is gate-tolerance-only (roc widened to
  1e-3; vote/select_tau/oracle/D3/strict-order untouched, ruling §D re-verified). Scope guard §E:
  A11 is B5-only, not a precedent.
- **Minor, non-defect note:** log §(a) prints the CPU-replay's *recomputed* deployed roc (e.g. Qwen
  s0-valsel test 0.8840 vs 13115 anchor 0.8838), not the anchor — the record is transparent about this
  drift (≤7e-4), and under A11 it is within tolerance while deployed acc/mF1 (which the calibration
  consumes) are exact. Not a provenance gap.
- **Commit state:** repo HEAD at execution be30d87; amended prereg/design/script/record batch-committed
  after verdict per ceremony — consistent, no anomaly. G-repro/4dp discipline applied throughout (all
  reported numbers are 4 dp and independently reproduced).

**A7 + A2 processed; calibration machine fully validated; KILL confirmed. B5 = DEAD.**
