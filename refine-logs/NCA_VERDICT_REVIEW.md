# NCA / soft-kNN HEAD-LOSS family — VERDICT (independent 0-context reviewer)

**Role:** independent 0-context verdict reviewer. CPU only; NO GPU/SLURM/Modal; NO `state/` mutation; NO push.
I trust only the frozen documents and the primary trainlogs; every ruling below is applied against
`refine-logs/NCA_PREREG.md` VERBATIM (I quote each binding clause before applying it).
**Date:** 2026-07-25 NZST.
**Frozen object:** `refine-logs/NCA_PREREG.md`, FROZEN sha256
`7607863c15bef1d40c1f1f0a5b980123dd84aba260ced6db1fa993f277db5591` (original freeze), REFREEZE-1 amended
sha A `2ae7a73f6df4008186e5200f851e16902f567ec93f2c3681d03743c909dd0c9b` (`src/model/loss.py` only; B/C
byte-identical). Authorization chain confirmed intact at submit: `NCA_FREEZE.md` (APPROVED-WITH-NOTES) +
REFREEZE-1, `NCA_REFREEZE_REVIEW.md` (R1–R7 all PASS), `NCA_SUBMIT_RECORD.md` (submit-instant sha re-verify
ALL MATCH; codex re-gate CLEARED; smoke PASS; family job **13482** COMPLETED exit 0:0, 24/24 trainlogs).
**Raw source:** `NCA_SUBMIT_RECORD.md §6` (executor transcription; no gates/deltas applied at that stage).

---

## 0. Binding clauses quoted VERBATIM (before application)

**KS-arm-dead — the KILL bar (§3.3, SIGN-based, per arm × dataset):**
> Per the **frame16 DEV-1 sign discipline** (house n=3 = **no bootstrap**; the kill uses SIGN, not a
> CI-straddles-0 test): an arm×dataset cell is **KILLED** iff, on **BOTH protocols**, `mean paired Δacc ≤ 0`
> **OR** the acc sign is not 3/3 positive — i.e. **neither** protocol produces a clean positive-mean-and-3/3-sign
> result (a tie-or-regress on both protocols = no net improvement over the floor).

⇒ Applied as: a cell **SURVIVES** iff **at least one** protocol has (mean paired Δacc > 0 **AND** acc sign 3/3
positive); otherwise **KILLED**.

**FORMAL promote bar (§3.1 rule 4 + §3.2, per arm × dataset):**
> (4) **pass = mean paired Δacc ≥ +0.030 AND mean paired Δmacro-F1 ≥ +0.030 AND sign 3/3 positive**; (5) headline
> claim requires pass on ≥ 2 datasets under a stated protocol; both protocols judged separately; verdict written
> exactly "final-epoch: pass/fail; val-selected: pass/fail".

> House **+0.030 acc AND +0.030 mF1** conjunct, **3/3 seeds positive**, under **BOTH** protocols vs the banked
> floor (§2). Below the conjunct under a protocol → **NEGATIVE** on that protocol. **D7-DEAD (F0.3): even a formal
> PASS is an engineering/ablation row, NEVER a novelty win.**

**KS-regression note (§3.4):**
> If arm − floor **mean Δacc ≤ −0.014** on a leg (below the full head-seed spread, §2.3), the objective
> **degraded** the head.

**Family = one multiplicity bite (§3.6):**
> **ONE sbatch = ONE pre-registered family = ONE multiplicity bite** whether one or all four arms survive. The
> four arms **share** the single "litsweep2 wave-3 NCA head-loss" bite. … **Family verdict is per-arm × per-dataset**
> (each judged only vs its own floor).

**Head-seed noise band (§2.3):** ±**0.014** (for the KS-regression leg and the MARGINAL/within-noise note).

---

## 1. Floors — verified against the prereg (prereg wins) + primary-log spot-check

Prereg §2 floor means (authoritative) vs the task-provided floors — **identical, no discrepancy**:

| dataset | protocol | prereg §2 mean acc/mF1 | task-provided | match |
|---|---|---|---|---|
| ZH (13150) | val-sel | 0.8322 / 0.8015 | 0.8322 / 0.8015 | ✓ |
| ZH (13150) | final | 0.8456 / 0.8173 | 0.8456 / 0.8173 | ✓ |
| HateMM (13241) | val-sel | 0.8775 / 0.8711 | 0.8775 / 0.8711 | ✓ |
| HateMM (13241) | final | 0.8791 / 0.8726 | 0.8791 / 0.8726 | ✓ |

Per-seed floors used for pairing (prereg §2.1/§2.2):
- **ZH val-sel:** s0 0.8322/0.8023, s1 0.8255/0.7956, s2 0.8389/0.8065.
- **ZH final:** s0 0.8456/0.8181, s1 0.8389/0.8113, s2 0.8523/0.8226.
- **HateMM val-sel:** s0 0.8791/0.8730, s1 0.8744/0.8678, s2 0.8791/0.8724.
- **HateMM final:** s0 0.8791/0.8730, s1 0.8791/0.8724, s2 0.8791/0.8724.

Primary-log floor spot-check (own Read of the cited trainlogs):
- ZH 13150 seed0 val-sel ep20 `Test_Retrieval` acc `0.8322` at `enc3s_MHC_zh_…-LoRA_HF_seed0_13150.trainlog:197`; final ep29 acc `0.8456` at `:270`. ✓
- HateMM 13241 seed2 val-sel ep10 `Test_Retrieval` acc `0.8791` at `enc3s_HateMM_…-LoRA-curric_HF_seed2_13241.trainlog:127`. ✓

---

## 2. D1 — Provenance spot-check (numeric-provenance discipline)

5 of 24 runs re-read from PRIMARY trainlogs (spread across ALL 4 arms, BOTH datasets, seeds 0/1/2) at the exact
line numbers cited in `NCA_SUBMIT_RECORD.md §6`. Format note: each eval emits a binary-f1 `Test_Retrieval acc:`
line then a `Test_Retrieval … macroF1: … acc:` line; the cited `(:N)` is the **macroF1** line, and the
transcribed pair is (that line's `acc`, that line's `macroF1`). All match **bit-for-bit**:

| # | arm | dataset | seed | protocol | file : line | primary log (acc / macroF1) | transcribed §6 | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | A1a nca_tau0.1 | MHC_zh | 2 | val-sel (ep15) | `nca_nca_tau0.1_MHC_zh_…LoRA_HF_seed2_13482.trainlog:157` | 0.8591 / 0.8295 | 0.8591 / 0.8295 | ✓ MATCH |
| 1 | A1a nca_tau0.1 | MHC_zh | 2 | final (ep29) | same file `:270` | 0.8389 / 0.8090 | 0.8389 / 0.8090 | ✓ MATCH |
| 2 | A1b nca_tau0.2 | HateMM | 2 | val-sel (ep8) | `nca_nca_tau0.2_HateMM_…curric_HF_seed2_13482.trainlog:108` | 0.8837 / 0.8776 | 0.8837 / 0.8776 | ✓ MATCH |
| 2 | A1b nca_tau0.2 | HateMM | 2 | final (ep29) | same file `:298` | 0.8791 / 0.8730 | 0.8791 / 0.8730 | ✓ MATCH |
| 3 | A2 supcon_tau0.1 | MHC_zh | 0 | val-sel (ep9) | `nca_supcon_tau0.1_MHC_zh_…LoRA_HF_seed0_13482.trainlog:109` | 0.8255 / 0.7956 | 0.8255 / 0.7956 | ✓ MATCH |
| 3 | A2 supcon_tau0.1 | MHC_zh | 0 | final (ep29) | same file `:270` | 0.8725 / 0.8436 | 0.8725 / 0.8436 | ✓ MATCH |
| 4 | A3 mixup_a2.0 | HateMM | 1 | val-sel (ep24) | `nca_mixup_a2.0_HateMM_…curric_HF_seed1_13482.trainlog:256` | 0.8837 / 0.8781 | 0.8837 / 0.8781 | ✓ MATCH |
| 4 | A3 mixup_a2.0 | HateMM | 1 | final (ep29) | same file `:302` | 0.8791 / 0.8730 | 0.8791 / 0.8730 | ✓ MATCH |
| 5 | A3 mixup_a2.0 | MHC_zh | 0 | val-sel (ep20) | `nca_mixup_a2.0_MHC_zh_…LoRA_HF_seed0_13482.trainlog:199` | 0.8322 / 0.8023 | 0.8322 / 0.8023 | ✓ MATCH |
| 5 | A3 mixup_a2.0 | MHC_zh | 0 | final (ep29) | same file `:272` | 0.8725 / 0.8458 | 0.8725 / 0.8458 | ✓ MATCH |

**D1 RESULT: 5/5 runs (10/10 cited lines) match the executor's transcription bit-for-bit — 0 discrepancies.**
The §6 transcription is trustworthy; no cell is re-valued.

---

## 3. D5 — Per-arm × dataset delta tables (paired δ = arm seed s − floor seed s; both protocols; acc & mF1)

Sign count = number of the 3 seeds with paired **Δacc > 0**. Mean deltas shown = arithmetic mean of the 3
per-seed paired δ (equal to arm-mean − floor-mean; shown to 4dp).

### 3.1 A1a — NCA τ=0.1

| dataset | protocol | Δacc s0 | Δacc s1 | Δacc s2 | **mean Δacc** | acc sign | mean ΔmF1 |
|---|---|---|---|---|---|---|---|
| ZH | val-sel | +0.0067 | +0.0067 | +0.0202 | **+0.0112** | **3/3** | +0.0113 |
| ZH | final | −0.0134 | +0.0067 | −0.0134 | **−0.0067** | 1/3 | −0.0083 |
| HateMM | val-sel | +0.0093 | 0.0000 | −0.0186 | **−0.0031** | 1/3 | −0.0028 |
| HateMM | final | +0.0046 | +0.0046 | −0.0047 | **+0.0015** | 2/3 | +0.0014 |

### 3.2 A1b — NCA τ=0.2

| dataset | protocol | Δacc s0 | Δacc s1 | Δacc s2 | **mean Δacc** | acc sign | mean ΔmF1 |
|---|---|---|---|---|---|---|---|
| ZH | val-sel | 0.0000 | 0.0000 | +0.0134 | **+0.0045** | 1/3 | +0.0045 |
| ZH | final | −0.0134 | 0.0000 | −0.0134 | **−0.0089** | 0/3 | −0.0098 |
| HateMM | val-sel | −0.0047 | 0.0000 | +0.0046 | **−0.0000** | 1/3 | −0.0002 |
| HateMM | final | −0.0047 | −0.0047 | 0.0000 | **−0.0031** | 0/3 | −0.0035 |

### 3.3 A2 — neighborhood-SupCon τ=0.1

| dataset | protocol | Δacc s0 | Δacc s1 | Δacc s2 | **mean Δacc** | acc sign | mean ΔmF1 |
|---|---|---|---|---|---|---|---|
| ZH | val-sel | −0.0067 | +0.0067 | +0.0134 | **+0.0045** | 2/3 | +0.0053 |
| ZH | final | +0.0269 | 0.0000 | −0.0067 | **+0.0067** | 1/3 | +0.0063 |
| HateMM | val-sel | 0.0000 | +0.0047 | 0.0000 | **+0.0016** | 1/3 | +0.0015 |
| HateMM | final | +0.0046 | 0.0000 | −0.0047 | **−0.0000** | 1/3 | −0.0006 |

### 3.4 A3 — manifold mixup α=2.0

| dataset | protocol | Δacc s0 | Δacc s1 | Δacc s2 | **mean Δacc** | acc sign | mean ΔmF1 |
|---|---|---|---|---|---|---|---|
| ZH | val-sel | 0.0000 | 0.0000 | +0.0134 | **+0.0045** | 1/3 | +0.0037 |
| ZH | final | +0.0269 | +0.0134 | 0.0000 | **+0.0134** | 2/3 | +0.0122 |
| HateMM | val-sel | −0.0140 | +0.0093 | −0.0047 | **−0.0031** | 1/3 | −0.0031 |
| HateMM | final | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0/3 | +0.0004 |

**Arithmetic worked example (A1a ZH val-sel):** arm §6 s0/s1/s2 acc = 0.8389/0.8322/0.8591; floor s0/s1/s2 =
0.8322/0.8255/0.8389 ⇒ δ = +0.0067/+0.0067/+0.0202 ⇒ mean +0.0336/3 = **+0.0112**, sign **3/3+**. mF1 δ =
(0.8090−0.8023)/(0.7997−0.7956)/(0.8295−0.8065) = +0.0067/+0.0041/+0.0230 ⇒ mean +0.0338/3 = **+0.0113**.
**This is the ONLY cell with 3/3 positive acc sign on any protocol.**

**Cross-check note:** A1b HateMM val-sel and A2 HateMM final each have per-seed acc δ = (±0.0046/0.0/∓0.0047),
per-seed mean = −0.0001/3 = **−0.00003 (≤ 0)** — shown as "−0.0000" above; the mean-of-rounded-means reads 0.0000.
Either reading gives `mean Δacc ≤ 0`, so the KS ruling is unaffected.

---

## 4. D2 — KS-arm-dead ruling (per arm × dataset), applied VERBATIM

Survive iff ≥1 protocol has (mean Δacc > 0 AND acc sign 3/3+); else KILLED.

| arm | dataset | val-sel (mean Δacc / sign) | final (mean Δacc / sign) | any clean pos-mean & 3/3? | **KS-arm-dead** |
|---|---|---|---|---|---|
| A1a nca_tau0.1 | ZH | +0.0112 / **3/3** ✔ | −0.0067 / 1/3 | **YES (val-sel)** | **SURVIVES** |
| A1a nca_tau0.1 | HateMM | −0.0031 / 1/3 | +0.0015 / 2/3 | no | **KILLED** |
| A1b nca_tau0.2 | ZH | +0.0045 / 1/3 | −0.0089 / 0/3 | no | **KILLED** |
| A1b nca_tau0.2 | HateMM | −0.0000 / 1/3 | −0.0031 / 0/3 | no | **KILLED** |
| A2 supcon_tau0.1 | ZH | +0.0045 / 2/3 | +0.0067 / 1/3 | no (sign never 3/3) | **KILLED** |
| A2 supcon_tau0.1 | HateMM | +0.0016 / 1/3 | −0.0000 / 1/3 | no | **KILLED** |
| A3 mixup_a2.0 | ZH | +0.0045 / 1/3 | +0.0134 / 2/3 | no (sign never 3/3) | **KILLED** |
| A3 mixup_a2.0 | HateMM | −0.0031 / 1/3 | 0.0000 / 0/3 | no | **KILLED** |

**KS-arm-dead: 7 of 8 cells KILLED. Only A1a NCA τ=0.1 × ZH survives** (its val-sel leg is the sole
positive-mean-AND-3/3-sign result in the whole family).

## 5. D3 — FORMAL promote bar (per arm × dataset), applied VERBATIM

Need mean Δacc ≥ +0.030 AND mean ΔmF1 ≥ +0.030 AND 3/3 acc-sign, on a protocol. **No cell reaches +0.030 on
either metric on either protocol** (family-max mean Δacc = A3 ZH final +0.0134; family-max val-sel mean Δacc =
A1a ZH +0.0112). Every leg is FAIL.

```
A1a NCA τ=0.1:   MHC_zh:  final-epoch: fail; val-selected: fail  [FORMAL §3.2]. KS-arm-dead: survives.
                 HateMM:  final-epoch: fail; val-selected: fail. KS-arm-dead: KILLED.
A1b NCA τ=0.2:   MHC_zh:  final-epoch: fail; val-selected: fail. KS-arm-dead: KILLED.
                 HateMM:  final-epoch: fail; val-selected: fail. KS-arm-dead: KILLED.
A2  n-SupCon:    MHC_zh:  final-epoch: fail; val-selected: fail. KS-arm-dead: KILLED.
                 HateMM:  final-epoch: fail; val-selected: fail. KS-arm-dead: KILLED.
A3  mixup α=2.0: MHC_zh:  final-epoch: fail; val-selected: fail. KS-arm-dead: KILLED.
                 HateMM:  final-epoch: fail; val-selected: fail. KS-arm-dead: KILLED.
```

**MARGINAL note (§7.2, B3 §2.2 precedent):** A1a NCA τ=0.1 × ZH val-sel is a clean 3/3-positive acc result
(mean **+0.0112** acc / **+0.0113** mF1) that survives KS-arm-dead but sits **below** the +0.030 FORMAL bar AND
**below** the ±0.014 head-seed noise band (§2.3) — a within-noise clean-sign positive. This is exactly the
F0.5-predicted "**ZH val-sel hardening**" outcome (measured-not-promoted limbo, prereg §8 bullet 2); still
D7-DEAD (F0.3).

**KS-regression (§3.4):** NO leg fires — no cell's mean Δacc ≤ −0.014 (most-negative mean Δacc = A1b ZH final
**−0.0089**). SupCon (A2) did not regress either (F0.5(b) headwind not realized at the mean level).

## 6. D4 — Family one-bite verdict (§3.6)

The four arms share ONE multiplicity bite. Any arm×dataset FORMAL-PASS ⇒ family alive; else family dead.
**No arm×dataset cell cleared the FORMAL bar** ⇒ the family yields no promotable/headline row.

**FAMILY VERDICT LINE (binding):**
> **The NCA / soft-kNN head-loss family (one multiplicity bite, 4 arms × 2 datasets) is DEAD: no arm×dataset cell
> cleared the FORMAL bar (mean Δacc ≥ +0.030 AND mean ΔmF1 ≥ +0.030, 3/3 seeds positive, BOTH protocols) on any
> protocol. 7 of 8 cells are KS-arm-dead KILLED; the sole survivor, A1a NCA τ=0.1 × ZH, survives KS-arm-dead on a
> within-noise clean-sign val-sel positive (+0.0112 acc / +0.0113 mF1) but sits below the FORMAL bar and below the
> ±0.014 head-seed noise band — measured-not-promoted limbo, D7-DEAD. The loss↔inference-mismatch axis is CLOSED
> at ~0.33 GPU-h; law-I is upheld (the one vote-matched training objective did not convert the frozen-LoRA-Qwen
> feature ceiling into a promotable gain on either dataset).**

Headline claim (§3.1 rule 5: pass on ≥2 datasets): **NOT met** — zero datasets pass under any protocol.

---

## 7. D6 — NON-BINDING observations (NOT claims)

- **A1a ZH val-sel is the family's best cell** and the only clean 3/3-positive-sign leg; its +0.0112 acc lift is
  real-in-sign but **within** the ±0.014 head-seed noise band — a hardening signal, not a promotion. Consistent
  with the prereg's honest prior (F0.5: realistic deliverable = ZH val-sel hardening, P(≥+3)=2–4%).
- **A3 (mixup) ZH final** posts the family-max mean Δacc **+0.0134** (acc sign 2/3, seed2 tie) and mF1 +0.0122 —
  positive-but-under-bar and NOT 3/3, so KS-killed; a variance-reduction signal on the ZH final-epoch leg only,
  vanishing under val-selection.
- **A2 (SupCon) ZH final seed0** is the single largest per-seed positive in the family (Δacc **+0.0269**,
  ΔmF1 +0.0255) but is isolated (seed1 tie, seed2 −0.0067) — seed noise, not an effect.
- **HateMM is inert as predicted (F0.5(c)):** every HateMM cell hovers at the 0.8775–0.8806 project-best band;
  most seed deltas are 0.0000 (identical `Test_Retrieval` lines vs floor), consistent with a near-ceiling
  hold-the-line dataset. A3 HateMM final is a perfect 3×0.0000 tie vs floor.
- **No arm globally regressed:** the SupCon batch-64 collapse headwind (F0.5(b)) and any mixup destabilization did
  not materialize at the 3-seed mean (KS-regression clean everywhere).
- **Direction of the whole family** matches the prereg's expected outcome (§8 bullet 1, "the honest expected
  outcome"): a near-total KS sweep with a single within-noise ZH val-sel survivor — a genuinely un-enumerated axis
  converted to a measured door-closer in one bite.

---

## 8. Review statements

ZERO GPU/SLURM/Modal spent (CPU-only: Read of frozen docs + primary trainlogs, arithmetic, `sha`-free reasoning
over already-verified freeze ledger). No held-out test job run — I re-read only the existing 13482/13150/13241
trainlogs. Rulings applied against `NCA_PREREG.md` VERBATIM. Prereg NOT modified. `state/` and
`autoresearch/…/state/` NOT touched. No `research-wiki/` mutation. Single file written
(`refine-logs/NCA_VERDICT_REVIEW.md`). Committed locally; NOT pushed.
