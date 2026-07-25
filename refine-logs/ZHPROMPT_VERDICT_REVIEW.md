# ZHPROMPT — Chinese-Instruction Re-Extraction — INDEPENDENT 0-CONTEXT VERDICT

**Reviewer:** independent 0-context verdict reviewer (no prior context on this probe).
**Date:** 2026-07-25 NZST.
**Frozen object judged VERBATIM:** `refine-logs/ZHPROMPT_PREREG.md`
(FROZEN sha `07df7c7135e115f41f12be5ee95a06d29fa7360255b0995c5503bf6d3c841aab`).
**On-disk re-hash this review:** `07df7c7135e115f41f12be5ee95a06d29fa7360255b0995c5503bf6d3c841aab` — **MATCH** (frozen bytes intact).
**Authorization chain:** `ZHPROMPT_PREREG_REVIEW.md` (`APPROVED-WITH-NOTES`) → `ZHPROMPT_FREEZE.md` → `ZHPROMPT_SUBMIT_RECORD.md` (S0–S6; job 13487 COMPLETED 0:0).
**Mandate:** CPU-only; no GPU/SLURM; no `state/` mutation; no push. Every number re-parsed from PRIMARY trainlogs
with my own parser (`scratchpad/verdict_parse.py`); the submit-record S5 table taken on **nothing** but its word.

---

## 0. Binding clauses quoted VERBATIM from the frozen prereg (applied below, unaltered)

**KS-parity (§3.3, machinery guard):**
> KS-parity (machinery guard, pre-science; task req 3). BEFORE any Chinese-prompt judging, an **English-DEFAULT**
> re-extraction of ONE stream must reproduce the banked cache **BIT-EXACT**. … **Threshold =
> `img max|Δ| == 0.0 AND text max|Δ| == 0.0`** (bit-exact) … **Fail ⇒ HALT (plumbing bug), not a result.**

**KS-dead (§3.3, per-arm kill; SIGN-based; NO auto-defund):**
> KS-dead (per-arm screen kill; recon §7 "≤0 on EITHER protocol" gate). A treatment arm's **3-seed mean paired
> Δacc ≤ 0 vs its own floor on EITHER protocol ⇒ that arm KILLED** (banked as the prompt-language null). **Secondary
> read:** mean paired Δacc `< +0.015` on **BOTH** protocols (inside the ±0.014 ZH seed-noise band, §2.3) ⇒ also
> KILL. **Per-arm, arms INDEPENDENT, NO auto-defund** (F0.6 …): a KS-dead **frozen** arm does **NOT** kill the LoRA
> arm; each is judged only vs its own floor. State each killed arm explicitly at verdict time.

**FORMAL promote bar (§3.2, goal-facing; per arm):**
> House **+0.030 acc AND +0.030 mF1** conjunct, **3/3 seeds positive**, under **BOTH** protocols vs the arm's banked
> floor (§2). Below the conjunct under a protocol → **NEGATIVE** on that protocol. **D7-DEAD (F0.3): even a formal
> PASS is a performance/robustness row … NEVER a novelty win.**

**Decision rule (§3.1 rule 4, verbatim from `exp-encoder-3seed.md:73-85`):**
> **pass = mean paired Δacc ≥ +0.030 AND mean paired Δmacro-F1 ≥ +0.030 AND sign 3/3 positive** … both protocols
> judged separately; verdict written exactly "final-epoch: pass/fail; val-selected: pass/fail".

**KS-regression note (§3.4):**
> If arm − floor **mean Δacc ≤ −0.014** on a leg (below the full head-seed spread, §2.3), the Chinese prompt
> **degraded** the stream → bank "Chinese extraction prompt hurts on ZH <arm>." A note within the KS-dead frame …

**D7-DEAD (F0.3, novelty scope):**
> F0.3 — Novelty = D7-DEAD, say it plainly. The injected-instruction LANGUAGE is an extraction knob … **Novelty-nil
> / D7-DEAD:** even a formal PASS is a performance/robustness row ("Chinese vs English extraction prompt on ZH"),
> same D7 class as frame-budget (F67) / head-recipe / readout — **never** a novelty contribution.

---

## D1 — Provenance spot-check: all 6 primary trainlogs re-parsed independently (PASS — bit-for-bit)

Protocol as pinned (§2/§3.1/V1): **val-sel** = epoch ≥ warmup 5 with **max `Val_Retrieval` acc, roc tie-break**;
**final** = max epoch (29). Metrics = TEST acc / TEST macroF1 (mF1) at the selected epoch. Parser reads the
per-epoch `Val_Retrieval Epoch N macroF1: … acc: A roc: R` and `Test_Retrieval Epoch N macroF1: M … acc: A …`
lines. My re-derivation reproduces the S5 RAW table **exactly, every seed, both protocols, acc/mF1/roc AND the
cited line numbers**, and the floors match prereg §2.1/§2.2 exactly.

**Arm-L (LoRA Chinese-prompt) 13487 — `enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF-zhp_seed{0,1,2}_13487.trainlog`:**

| seed | val-sel ep | val-sel acc/mF1 | TEST/VAL line | final ep | final acc/mF1 | TEST line |
|---|---|---|---|---|---|---|
| 0 | 7 | 0.7852 / 0.7541 | L100 / L99 | 29 | 0.8389 / 0.8065 | L299 |
| 1 | 8 | 0.8255 / 0.8002 | L109 / L108 | 29 | 0.8255 / 0.7904 | L299 |
| 2 | 5 | 0.7785 / 0.7450 | L80 / L79 | 29 | 0.8389 / 0.8065 | L297 |
| **mean** | | **0.7964 / 0.7664** | | | **0.8344 / 0.8011** | |

**Arm-F (frozen Chinese-prompt) 13487 — `enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct_HF-zhp_seed{0,1,2}_13487.trainlog`:**

| seed | val-sel ep | val-sel acc/mF1 | TEST/VAL line | final ep | final acc/mF1 | TEST line |
|---|---|---|---|---|---|---|
| 0 | 25 | 0.7785 / 0.7203 | L262 / L261 | 29 | 0.8121 / 0.7608 | L299 |
| 1 | 7 | 0.7718 / 0.7327 | L100 / L99 | 29 | 0.8054 / 0.7613 | L299 |
| 2 | 5 | 0.7584 / 0.7058 | L80 / L79 | 29 | 0.7785 / 0.7158 | L297 |
| **mean** | | **0.7696 / 0.7196** | | | **0.7987 / 0.7460** | |

Per-seed roc (re-derived, matches S5): Arm-L val-sel 0.8594/0.8981/0.8712, final 0.9083/0.8818/0.9028; Arm-F
val-sel 0.8929/0.8417/0.8494, final 0.8915/0.8880/0.8675. **[P1]==[P2]==this review, all 6 runs, all metrics.**

**Floors re-verified vs prereg §2 (PASS):**
- Arm-L floor 13150: s0 ep20 0.8322/0.8023 → final 0.8456/0.8181; s1 ep26 0.8255/0.7956 → 0.8389/0.8113; s2 ep19
  0.8389/0.8065 → 0.8523/0.8226. **mean val-sel 0.8322/0.8015 (0.80147), final 0.8456/0.8173 (0.81733)** — matches §2.1.
- Arm-F floor 13115: s0 ep22 0.7919/0.7412 → 0.8188/0.7864; s1 ep25 0.8121/0.7871 → 0.8054/0.7759; s2 ep28
  0.8054/0.7759 → 0.7852/0.7514. **mean val-sel 0.8031/0.7681 (0.76807), final 0.8031/0.7712 (0.77123)** — matches §2.2.

**D1 result: PASS.** The §S5 transcription is confirmed bit-for-bit against the six primary trainlogs; both floors
re-derive to the prereg §2 values at 4dp. No transcription drift.

## D2 — KS-parity evidence stands (PASS)

Recorded max|Δ| lines re-read from the smoke job's primary `.out` (`slurm/logs/zhpsmoke_13486.out`, job 13486
COMPLETED 0:0):
```
L17: [KS-PARITY frozen] n=8 id_order_match=True img_max|Δ|=0.0 text_max|Δ|=0.0 -> PASS
L18: [KS-PARITY LoRA]   n=8 id_order_match=True img_max|Δ|=0.0 text_max|Δ|=0.0 -> PASS
L19: KS_PARITY_OVERALL_PASS
```
Both extractors' English-default re-extraction reproduce the banked cache **bit-exact** (`img max|Δ|==0.0 AND text
max|Δ|==0.0`, n=8, id order matched) — the §3.3 threshold is met, **no HALT**. The mandatory N1 head-repro (L298-301)
reproduces 13150 seed0 to 4dp (`N1_REPRO_PASS_4DP_MATCH`), closing the run_rac.py/loss.py additive-drift confound
directly. **KS-parity: PASS — the machinery guard holds; the arms may be judged.**

---

## D3–D5 — Deltas per seed + means, KS-dead + FORMAL applied per arm

Arithmetic on the D1 numbers (arm − own floor); every value re-computed this review.

### Arm-L (LoRA, PRIMARY) vs floor 13150 (§2.1)

| seed | protocol | Arm-L acc/mF1 | floor acc/mF1 | Δacc | ΔmF1 |
|---|---|---|---|---|---|
| 0 | val-sel | 0.7852 / 0.7541 | 0.8322 / 0.8023 | −0.0470 | −0.0482 |
| 1 | val-sel | 0.8255 / 0.8002 | 0.8255 / 0.7956 | +0.0000 | +0.0046 |
| 2 | val-sel | 0.7785 / 0.7450 | 0.8389 / 0.8065 | −0.0604 | −0.0615 |
| **mean** | **val-sel** | **0.7964 / 0.7664** | **0.8322 / 0.8015** | **−0.03580** | **−0.03503** |
| 0 | final-ep | 0.8389 / 0.8065 | 0.8456 / 0.8181 | −0.0067 | −0.0116 |
| 1 | final-ep | 0.8255 / 0.7904 | 0.8389 / 0.8113 | −0.0134 | −0.0209 |
| 2 | final-ep | 0.8389 / 0.8065 | 0.8523 / 0.8226 | −0.0134 | −0.0161 |
| **mean** | **final-ep** | **0.8344 / 0.8011** | **0.8456 / 0.8173** | **−0.01117** | **−0.01620** |

- Δacc sign: val-sel **0/3 positive**, final **0/3 positive**. ΔmF1 sign: val-sel 1/3, final 0/3.
- **KS-dead:** mean Δacc ≤ 0 on **BOTH** protocols (val-sel −0.03580 ≤ 0; final −0.01117 ≤ 0) ⇒ the "≤ 0 on EITHER
  protocol" gate is met (twice over) ⇒ **Arm-L KILLED.** (Secondary read also fires: mean Δacc < +0.015 on both.)
- **FORMAL:** thresholds §2.3 = val-sel acc ≥ 0.8622 & mF1 ≥ 0.8315, final acc ≥ 0.8756 & mF1 ≥ 0.8473. Measured
  means (0.7964/0.7664 val-sel; 0.8344/0.8011 final) are **all below**, no leg reaches +0.030, sign not 3/3 ⇒
  **final-epoch: fail; val-selected: fail.**

### Arm-F (frozen, control) vs floor 13115 (§2.2)

| seed | protocol | Arm-F acc/mF1 | floor acc/mF1 | Δacc | ΔmF1 |
|---|---|---|---|---|---|
| 0 | val-sel | 0.7785 / 0.7203 | 0.7919 / 0.7412 | −0.0134 | −0.0209 |
| 1 | val-sel | 0.7718 / 0.7327 | 0.8121 / 0.7871 | −0.0403 | −0.0544 |
| 2 | val-sel | 0.7584 / 0.7058 | 0.8054 / 0.7759 | −0.0470 | −0.0701 |
| **mean** | **val-sel** | **0.7696 / 0.7196** | **0.8031 / 0.7681** | **−0.03357** | **−0.04847** |
| 0 | final-ep | 0.8121 / 0.7608 | 0.8188 / 0.7864 | −0.0067 | −0.0256 |
| 1 | final-ep | 0.8054 / 0.7613 | 0.8054 / 0.7759 | +0.0000 | −0.0146 |
| 2 | final-ep | 0.7785 / 0.7158 | 0.7852 / 0.7514 | −0.0067 | −0.0356 |
| **mean** | **final-ep** | **0.7987 / 0.7460** | **0.8031 / 0.7712** | **−0.00447** | **−0.02527** |

- Δacc sign: val-sel **0/3 positive**, final **0/3 positive**.
- **KS-dead:** mean Δacc ≤ 0 on **BOTH** protocols (val-sel −0.03357 ≤ 0; final −0.00447 ≤ 0) ⇒ **Arm-F KILLED.**
  (Secondary read also fires.)
- **FORMAL:** thresholds §2.3 = val-sel acc ≥ 0.8331 & mF1 ≥ 0.7981, final acc ≥ 0.8331 & mF1 ≥ 0.8012. Measured
  means (0.7696/0.7196 val-sel; 0.7987/0.7460 final) are **all below** ⇒ **final-epoch: fail; val-selected: fail.**

### KS-regression note (§3.4) — Chinese prompt DEGRADED both streams on val-sel
- **Arm-L:** val-sel mean Δacc **−0.0358 ≤ −0.014** ⇒ bank "Chinese extraction prompt hurts on ZH LoRA (val-sel)."
  (final-ep mean Δacc −0.0112 is negative but > −0.014 = inside the ±0.014 band.)
- **Arm-F:** val-sel mean Δacc **−0.0336 ≤ −0.014** ⇒ bank "Chinese extraction prompt hurts on ZH frozen (val-sel)."
  (final-ep mean Δacc −0.0045 inside the band.)
- mF1 corroborates: every mF1 mean Δ is negative; Arm-F val-sel ΔmF1 −0.0485 and final ΔmF1 −0.0253, Arm-L
  final ΔmF1 −0.0162 all clear the −0.014 magnitude too.

### Arms are independent — NO auto-defund (F0.6) applied
Both arms are killed **each on its own evidence vs its own floor**; neither kill is inherited. (Here both die
regardless, so independence is not load-bearing for the outcome, but it is respected: Arm-L is killed by Arm-L's
own −0.0358/−0.0112 Δacc, not by Arm-F's death.)

### Probe-level outcome
**Both arms KS-dead** ⇒ the extraction-instruction-**LANGUAGE** axis is **CLOSED** at ~1.1 GPU-h (recon §8's honest
expected PV1 outcome). The injected-prompt language carries **no net positive vote signal** on ZH — and on the
harder val-selected protocol it measurably **regresses** both streams (beyond the ±0.014 noise band). The live
reviewer question ("why English prompts for Chinese inputs?") is answered with a **measured null-to-negative**:
a faithful Chinese translation of the deployed English instruction/scaffolding does **not** help — English
prompting is not a liability on ZH.

**D7-DEAD restated (F0.3):** the injected-instruction language is an extraction knob; even a formal PASS would have
been only a performance/robustness row ("Chinese vs English extraction prompt on ZH"), the same D7 class as
frame-budget / head-recipe / readout — **never a novelty contribution.** With both arms KILLED, this is doubly
moot: nothing to promote, and nothing that could have been a novelty even had it promoted.

---

## VERDICT (binding language, §7.2 format)

```
KS-parity: PASS bit-exact  (job 13486; img/text max|Δ|=0.0 both extractors — machinery guard holds).
Arm-L (LoRA, PRIMARY):   final-epoch: fail; val-selected: fail  [FORMAL §3.2].  KS-dead: KILLED.
Arm-F (frozen, control): final-epoch: fail; val-selected: fail                .  KS-dead: KILLED.
+ KS-regression note (§3.4): Chinese extraction prompt HURTS on ZH — Arm-L val-sel mean Δacc −0.0358,
  Arm-F val-sel mean Δacc −0.0336 (both ≤ −0.014); final-epoch legs negative but inside the ±0.014 band.
PROBE VERDICT: BOTH ARMS KS-DEAD (KILLED) — extraction-instruction-language axis CLOSED; measured
null-to-negative; D7-DEAD regardless.
```

---

## D6 — Non-binding observations (clearly labeled; NOT part of the ruling)

1. **Direction/magnitude of the language effect.** The Chinese re-extraction did not merely fail to help — it
   moved the wrong way. On val-sel both arms regress by ~−0.034 to −0.036 acc (Arm-L −0.0358, Arm-F −0.0336),
   materially beyond the ±0.014 head-seed band; on final-epoch both are still negative but small (Arm-L −0.0112,
   Arm-F −0.0045 acc) and within the band. Macro-F1 tells the same story more strongly on the frozen arm
   (Arm-F val-sel ΔmF1 −0.0485). So the honest LOW prior (F0.5: PV1 ~60-65%) landed, on the mildly-negative edge.

2. **Asymmetry between arms — the PRIMARY hypothesis is refuted, not merely unconfirmed.** The whole reason Arm-L
   was PRIMARY (F0.6/§4): the deployed ZH LoRA was SFT-trained with a Chinese instruction but the floor extracts
   with English, so a Chinese prompt should *remove* a train/inference language mismatch and could lift Arm-L above
   its floor (prior revised up to 8-12%). The data show **no such advantage**: Arm-L and Arm-F regress by nearly
   identical margins on val-sel (−0.0358 vs −0.0336), and Arm-L is actually *slightly worse* than Arm-F on
   final-epoch Δacc (−0.0112 vs −0.0045). The SFT-language-mismatch mechanism did **not** convert to a measured
   lift; walls (a) native-bilingual encoder and (d) 78-dev selection noise dominated, as F0.5 warned.

3. **Does it answer the recon's paper-value question?** Yes, decisively. "Does English-vs-Chinese instruction
   matter, and why feed English prompts to Chinese inputs?" is answered: switching the deployed extraction
   instruction/scaffolding to faithful Chinese does **not** improve ZH detection for either the frozen or the
   deployed-LoRA encoder, and slightly degrades it under val-selection. This is a clean, citable **door-closer
   sentence** ("we measured it — Chinese-instruction extraction gives no lift, a small regression under
   val-selection; English prompting is not the ZH bottleneck") but, per F0.3, a paper *sentence*, not a novelty
   *mechanism*.

4. **Selection-noise fingerprint (wall (d)).** The treatment arms' val-sel epochs are strikingly early/scattered
   (Arm-L ep 7/8/5; Arm-F ep 25/7/5) versus the floors' later, steadier picks (13150 ep 20/26/19; 13115 ep
   22/25/28). The 78-item dev set is choosing early, unstable epochs on the Chinese-prompt features — exactly the
   F45/F63/F66 selection-lock pathology the prereg flagged as ZH's real wall — which is where the val-sel
   regression concentrates. This is consistent with a representation that is no worse *per epoch* but harder for
   the tiny dev set to select on; it does not change the ruling (final-epoch, the selection-free leg, is also
   negative, so the axis is closed on both protocols).

5. **Cheapness of the closure.** ~1.1 GPU-h converted a genuinely un-enumerated axis (extraction-instruction
   language on the deployed path — distinct from B1 encoder-language, P8c summary-language, F70 prompt-structure)
   into a measured, banked null-to-negative. Per §3.6 this is the ONLY ZH-prompt-language bite; the family is spent.

---

**Reviewer statements:** ZERO GPU/SLURM/Modal spent (CPU-only: sha256 re-hash, independent trainlog re-parse via
`scratchpad/verdict_parse.py`, arithmetic, smoke-log re-read). Judged the frozen `ZHPROMPT_PREREG.md`
(`07df7c7…`, on-disk == frozen) VERBATIM. No `state/` or `research-wiki/` mutation. Committed locally, NOT pushed.
