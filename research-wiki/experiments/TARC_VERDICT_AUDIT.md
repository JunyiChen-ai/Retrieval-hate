# TARC B-line — Independent Verdict Audit (post-hoc integrity check)

**Auditor:** fresh, zero-context reviewer. **Date:** 2026-07-13.
**Mandate:** re-derive every TARC B-line gate table (G1 / G2-retention / G3) **from the raw
trainlogs only**, computing my own numbers *before* reading the recorded conclusions, then
diff cell-by-cell against `exp-tarc-t0.md` §10 / §11 / §12. B-line was already ruled KILL; this
audit either confirms or overturns it.

**Method (independent):** I re-implemented the enc3seed selection logic exactly from
`scripts/slurm/enc3seed.sbatch` (protocol A = warmup≥5, argmax val-acc with **roc tie-break**,
first-epoch on a full tie; protocol B = final epoch 29), parsing the `..._Retrieval Epoch N
macroF1: … acc: … roc: …` lines. The sbatch parser prints only TEST; I additionally extracted
VAL acc/F1 at the same selected/final epochs (needed for the G1/G2 val gates). Parser +
gate scripts:
`scratchpad/parse_tarc.py`, `scratchpad/gates.py` (run against the 21+6+6 trainlogs listed
in §10/§11/§12). I read §6/§7 (rules) first and did **not** look at §10–§12 until my own
tables were computed.

---

## BOTTOM LINE: **VERDICT CONFIRMED. Zero discrepancies. B-line KILL stands.**

Every trainlog-derivable number in §10, §11.2, and §12 reproduces to 4 dp, and every gate
verdict re-derives identically from first principles. All spot-checked source line-number
citations resolve to the exact cited lines. No number is off; no verdict flips.

---

## 1. OFF no-op reproduction gate

**HateMM (G1 job 12975 OFF ×3 vs enc3seed job 12850):** every A/B, val/test, acc/F1 cell is
**byte-identical**.

| seed | valsel TEST acc/F1 | final TEST acc/F1 | matches 12850? |
|---|---|---|---|
| 0 | 0.8279 / 0.8172 | 0.8186 / 0.7997 | ✓ |
| 1 | 0.8279 / 0.8163 | 0.8047 / 0.7822 | ✓ |
| 2 | 0.8047 / 0.7920 | 0.8140 / 0.7988 | ✓ |

Matches §10.1 exactly, and the §6/§7 baseline (`valsel 0.8279/0.8279/0.8047`, `final
0.8186/0.8047/0.8140`; seed0 no-op `0.8172 F1 / 0.8279 acc`, `0.7997 / 0.8186`). **PASS.**

**MHC (G3 job 12997 OFF ×3 vs enc3seed 12850 MHC-CLIP):** reproduces to 4 dp:
s0 0.7826/0.7113 & 0.7640/0.7145; s1 0.7329/0.6034 & 0.7826/0.7159; s2 0.7702/0.6997 &
0.7888/0.7303. Matches §12.3. **PASS.** ⇒ TARC code is a byte-identical no-op on both datasets;
both batches valid/decidable.

---

## 2. G1 gate (val Δacc, variant − OFF, paired within seed; gate: mean ≥ +0.015 ∧ sign ≥ 2/3)

My re-derivation (identical to §10.3):

**Protocol A (val-selected):** every variant fails.
v1prefer −0.0063 (0/3) · v1require −0.0125 (0/3) · v3lt0.1 −0.0094 (1/3) · v3lt0.5 −0.0499
(0/3) · v2vg0.5 +0.0031 (1/3) · v2vg1.0 +0.0094 (2/3, mean<0.015). All ✗.

**Protocol B (final epoch 29):**
| variant | Δ s0/s1/s2 | mean | pos | verdict |
|---|---|---|---|---|
| **v1prefer** | +0.0093/+0.0280/+0.0187 | **+0.0187** | 3/3 | **PASS** |
| v1require | +0.0187/+0.0093/+0.0000 | +0.0093 | 2/3 | ✗ (mean<0.015) |
| **v3lt0.1** | +0.0280/+0.0280/+0.0094 | **+0.0218** | 3/3 | **PASS** |
| v3lt0.5 | +0.0093/−0.0467/−0.0187 | −0.0187 | 1/3 | ✗ |
| v2vg0.5 | +0.0000/+0.0000/+0.0094 | +0.0031 | 1/3 | ✗ |
| v2vg1.0 | +0.0000/+0.0000/+0.0094 | +0.0031 | 1/3 | ✗ |

**G1 verdict re-derived:** v1prefer PASS (B), v3lt0.1 PASS (B), other 4 KILL. B-line NOT
killed at G1. **Identical to §10.3/§10.4.** (The pass is protocol-B-only; both A means are
negative — matches the §10.4 caveat.)

---

## 3. G2 retention gate (mllm_pred 12992 vs OFF 12975; protocol-B val Δacc ≥ 0.6 × GT G1 Δacc)

| variant | GT G1 Δacc(B) | thr = 0.6× | mllm_pred Δ s0/s1/s2 | mean | verdict |
|---|---|---|---|---|---|
| **v1prefer** | +0.0187 | **+0.0112** | +0.0374/+0.0093/+0.0187 | **+0.0218** (117 %) | **PASS** |
| v3lt0.1 | +0.0218 | +0.0131 | +0.0000/+0.0093/+0.0094 | +0.0062 (29 %) | **KILL** |

**Identical to §11.2.** Retrain ep29 VAL acc (v1prefer 0.8318/0.8224/0.8411; v3lt0.1
0.7944/0.8224/0.8318) reproduced exactly. v1prefer's MLLM-predicted gain matches/exceeds its
GT-oracle gain; v3lt0.1 retains only 29 % → killed. v1prefer is the sole variant to G3.

*Scope note:* §11.1 sub-gate A (MLLM prediction macro-F1 0.6137 / 0.6760) is **not**
trainlog-derivable — it comes from `target_pred_qwen7b.json` scored against `target_map.json`,
outside this log-only re-derivation. I neither confirm nor dispute it here; it was not part of
the pairing tables in scope. Everything trainlog-derivable in §11 (sub-gate B) is confirmed.

---

## 4. G3 final gate (TEST Δacc AND ΔF1 ≥ +0.030, sign 3/3, per protocol, per dataset)

**HateMM (12992 v1prefer mllm_pred ×3 vs 12975 OFF):**
| protocol | Δacc s0/s1/s2 → mean (pos) | ΔF1 s0/s1/s2 → mean (pos) |
|---|---|---|
| A (val-sel) | −0.0093/−0.0046/+0.0139 → **+0.0000** (1/3) | −0.0051/−0.0027/+0.0134 → **+0.0019** (1/3) |
| B (final) | +0.0000/+0.0000/−0.0093 → **−0.0031** (0/3) | +0.0057/+0.0053/−0.0101 → **+0.0003** (2/3) |

**MHC-EN (12997 v1prefer ×3 vs off ×3):**
| protocol | Δacc s0/s1/s2 → mean (pos) | ΔF1 s0/s1/s2 → mean (pos) |
|---|---|---|
| A (val-sel) | +0.0000/+0.0373/+0.0000 → **+0.0124** (1/3) | +0.0090/+0.1052/−0.0100 → **+0.0347** (2/3) |
| B (final) | +0.0186/−0.0124/−0.0124 → **−0.0021** (1/3) | −0.0187/+0.0007/−0.0159 → **−0.0113** (1/3) |

**Identical to §12.2 / §12.3 / §12.4** in all 16 cells and both means-of-means.
No protocol on either dataset clears +0.030 acc **and** +0.030 F1 at 3/3. **G3 FAIL → B-line
KILLED.** Confirmed.

### 4a. Independent verification of the flagged MHC-A ΔF1 +0.0347 cell

The single above-bar-looking cell (MHC protocol-A ΔF1 mean +0.0347) is, as §12.3 claims, a
**val-selection artifact carried entirely by seed 1**. Verified directly from
`tarc_g3mhc_off_seed1_12997.trainlog`:

- The val-selection (warmup≥5) ties at **val acc 0.7625** between **ep16 (roc 0.7884)** and
  **ep25 (roc 0.7636)**; the roc tie-break picks **ep16**.
- ep16's TEST macro-F1 is **0.6034** (line 167, confirmed) — anomalously low vs the other
  val-max ep25's 0.7166 and the ep17–29 band of 0.64–0.72.
- v1prefer seed1 selects ep26 (TEST F1 0.7086, line 249). ΔF1 = 0.7086 − 0.6034 = **+0.1052**,
  which alone lifts the 3-seed mean to +0.0347.
- The paired Δacc for the same cells is only +0.0124 (1/3 pos) and protocol B is negative on
  both metrics, so the conjunctive gate fails regardless. **The artifact claim is correct;**
  it does not represent a real effect.

---

## 5. Provenance spot-checks (source line citations)

All checked citations resolve to the exact cited line:
`tarc_g1_off_seed0_12975.trainlog:303` = Val ep29 acc 0.7944 ✓ ·
`tarc_g2rt_v1prefer_seed0_12992.trainlog:302` = Val ep29 acc 0.8318 ✓ ·
`tarc_g3mhc_off_seed1_12997.trainlog:167` = Test ep16 F1 0.6034 acc 0.7329 ✓ ·
`tarc_g3mhc_v1prefer_seed1_12997.trainlog:249` = Test ep26 F1 0.7086 acc 0.7702 ✓.

---

## 6. DISCREPANCIES

**None.** Every trainlog-derivable cell in §10, §11.2, and §12 reproduces to 4 dp; every gate
verdict (G1 2/6 pass → G2 v1prefer-only → G3 both-datasets-fail → KILL) re-derives identically
from first principles. No number is off by any amount; no verdict flips. The only §11 item not
re-derivable from logs (sub-gate A MLLM macro-F1) is explicitly out of scope for a log audit
and is flagged, not disputed.

**VERDICT CONFIRMED — the TARC B-line KILL is sound.**
