# HEAD-RECIPE family (SAM + modality-dropout on the align head; ZH + HateMM) — INDEPENDENT 0-CONTEXT VERDICT REVIEW

**Reviewer role:** independent 0-context verdict reviewer. No prior project context; trusts ONLY primary
artifacts. Renders the binding verdict strictly against the frozen pre-registration
`refine-logs/HEADRECIPE_PREREG.md` VERBATIM. Zero user interaction. CPU-only (no GPU/SLURM/Modal). Modified
nothing except this file; `autoresearch/goal_mllm_plus3/state/` untouched; nothing pushed.
**Out of scope (F0.3 / §8):** the D7 novelty boundary (SAM + modality-dropout are D7-DEAD generic training knobs)
and goal-level satisfaction — this review decides the **PERFORMANCE clause only.**
**Date:** 2026-07-25 NZST.

---

## 0. Hash-freeze verification (done FIRST, before any metric was read)

```
on-disk sha256(refine-logs/HEADRECIPE_PREREG.md)
  = 68be61ac5ad77d2cfc66f3b355b574855bb426ed2c2a6e6b89f8050347ba232d
expected (task + refine-logs/HEADRECIPE_FREEZE.md frozen block)
  = 68be61ac5ad77d2cfc66f3b355b574855bb426ed2c2a6e6b89f8050347ba232d
```
**MATCH.** The prereg on disk is the frozen binding text. NOT VOID. Proceeding.

**Frozen artifacts + reused machinery re-verified on disk at verdict time (no drift since submit):**
```
A 1012c9e378905e5c10a0447475560de4a32904af691e457bf4ce77a3d36cc20d  src/run_rac.py                            [MATCH]
B e7b61df485b97eb683279398746090c2d4b3d446fc4c53b5c85e14d366c23378  src/model/classifier.py                   [MATCH]
C c88f685f68f83611fde3f91751f330d30b6be278693a405f4b9fb80f53ebb009  scripts/slurm/headrecipe_family.sbatch    [MATCH]
loss.py       48796638fdd60fcfb313e97e7f89d73226d96f23369f8c8ebb61ca5814f9cd64  src/model/loss.py       [MATCH]
retrieval.py  d43e3bc417f775175021283c4bd4aa25c0df98aa4c4b34a90f8c696e195bcf57  src/utils/retrieval.py  [MATCH]
```

**Measurement provenance (raw logs only, job IDs per `HEADRECIPE_SUBMIT_RECORD.md`):** family = job **13478**
(2 arms × 2 datasets × 3 seeds = 12 head-only runs on cached LoRA features), trainlogs
`slurm/logs/hr_{SAM_rho0.05,MODDROP_p0.3}_{MHC_zh,HateMM}_<MODEL>_seed{0,1,2}_13478.trainlog` (12/12 present).
Comparison floors = **ZH job 13150** (generic-LoRA / B3) raw trainlogs
`slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_seed{0,1,2}_13150.trainlog` and **HateMM job 13241**
(curriculum-LoRA) raw trainlogs
`slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_13241.trainlog` — **re-parsed, NOT
re-run**. Every number below re-derived with the **byte-identical `enc3seed_lora_hatemm.sbatch` embedded parser**
(val-sel = epoch ≥ warmup 5 maximising `(Val_Retrieval acc, roc)`, report that epoch's TEST metrics; final =
max-epoch TEST), independently re-implemented and hand-verified against raw lines.

---

## 1. Comparison floors — re-derived vs the prereg §2.1/§2.2 pinned tables (numeric-provenance discipline)

Independent re-parse of the raw 13150 / 13241 trainlogs. **Every per-seed value, every selected epoch, and both
3-seed means reproduce the prereg pinned tables EXACTLY to 4dp** — no discrepancy, no blocking flag.

### 1.1 ZH floor — job 13150 (generic-LoRA / B3; goal-relevant, marginal)

| seed | val-sel ep (Test line) | val-sel acc/mF1 | final ep (Test line) | final acc/mF1 |
|---|---|---|---|---|
| 0 | 20 (:220) | 0.8322 / 0.8023 | 29 (:302) | 0.8456 / 0.8181 |
| 1 | 26 (:275) | 0.8255 / 0.7956 | 29 (:303) | 0.8389 / 0.8113 |
| 2 | 19 (:207) | 0.8389 / 0.8065 | 29 (:298) | 0.8523 / 0.8226 |
| **mean** | | **0.8322 / 0.8015** | | **0.8456 / 0.8173** |

Prereg §2.1 pins val-sel **0.8322 / 0.8015**, final **0.8456 / 0.8173** — reproduces to 4dp. ✔

### 1.2 HateMM floor — job 13241 (curriculum-LoRA; near-ceiling, project-best)

| seed | val-sel ep (Test line) | val-sel acc/mF1 | final ep (Test line) | final acc/mF1 |
|---|---|---|---|---|
| 0 | 29 (:331) | 0.8791 / 0.8730 | 29 (:331) | 0.8791 / 0.8730 |
| 1 | 14 (:178) | 0.8744 / 0.8678 | 29 (:329) | 0.8791 / 0.8724 |
| 2 | 10 (:140) | 0.8791 / 0.8724 | 29 (:331) | 0.8791 / 0.8724 |
| **mean** | | **0.8775 / 0.8711** | | **0.8791 / 0.8726** |

Prereg §2.2 pins val-sel **0.8775 / 0.8711**, final **0.8791 / 0.8726** — reproduces to 4dp. ✔ (Seed0 val-sel
selects ep29 = final, so its two rows coincide — exactly as the prereg states.)

**The banked floors ARE the comparators (F0.7 / §4.1b): they were NOT re-run.** Both floor trainlogs pre-date the
family submit (13150 = 2026-07-14, 13241 = 2026-07-18; family 13478 = 2026-07-25) and their runtime Namespaces
carry **no `sam=`/`mod_dropout=` keys** (0 occurrences) — i.e. they were produced by the pre-patch code, confirming
the additive-flag no-re-run claim.

---

## 2. Family arms — raw measured numbers (job 13478), re-parsed + line-verified

Val-selection argmax hand-verified from the raw `Val_Retrieval` epochs (warmup ≥ 5, max acc, roc tie-break); TEST
provenance = the `Test_Retrieval … macroF1` line at the cited epoch.

### 2.1 ARM A — SAM (`--sam True --sam_rho 0.05`)

| dataset | seed | val-sel ep (Test line) | val-sel acc/mF1 | final ep (Test line) | final acc/mF1 |
|---|---|---|---|---|---|
| MHC_zh | 0 |  7 (:100) | 0.7852 / 0.7385 | 29 (:299) | 0.7987 / 0.7612 |
| MHC_zh | 1 |  8 (:109) | 0.8255 / 0.8002 | 29 (:299) | 0.8054 / 0.7646 |
| MHC_zh | 2 |  5 (:80)  | 0.8121 / 0.7893 | 29 (:297) | 0.8054 / 0.7784 |
| **MHC_zh mean** | | | **0.8076 / 0.7760** | | **0.8032 / 0.7681** |
| HateMM | 0 | 25 (:288) | 0.8791 / 0.8735 | 29 (:329) | 0.8884 / 0.8828 |
| HateMM | 1 | 28 (:323) | 0.8884 / 0.8828 | 29 (:334) | 0.8837 / 0.8776 |
| HateMM | 2 | 29 (:330) | 0.8791 / 0.8724 | 29 (:330) | 0.8791 / 0.8724 |
| **HateMM mean** | | | **0.8822 / 0.8762** | | **0.8837 / 0.8776** |

### 2.2 ARM B — modality-dropout (`--mod_dropout True --mod_dropout_p 0.3`)

| dataset | seed | val-sel ep (Test line) | val-sel acc/mF1 | final ep (Test line) | final acc/mF1 |
|---|---|---|---|---|---|
| MHC_zh | 0 | 24 (:254) | 0.8322 / 0.8023 | 29 (:300) | 0.7919 / 0.7577 |
| MHC_zh | 1 |  6 (:90)  | 0.8523 / 0.8202 | 29 (:298) | 0.8389 / 0.8090 |
| MHC_zh | 2 | 29 (:305) | 0.8121 / 0.7771 | 29 (:305) | 0.8121 / 0.7771 |
| **MHC_zh mean** | | | **0.8322 / 0.7999** | | **0.8143 / 0.7813** |
| HateMM | 0 |  5 (:86)  | 0.8512 / 0.8450 | 29 (:327) | 0.8651 / 0.8567 |
| HateMM | 1 |  5 (:86)  | 0.8512 / 0.8476 | 29 (:327) | 0.8837 / 0.8776 |
| HateMM | 2 | 29 (:329) | 0.8698 / 0.8620 | 29 (:329) | 0.8698 / 0.8620 |
| **HateMM mean** | | | **0.8574 / 0.8515** | | **0.8729 / 0.8654** |

*(Discrete grids: HateMM test = 215 samples, 1/215 = 0.00465; ZH test = 149 samples, 1/149 = 0.00671. Every
per-seed value sits on its grid, so ties are exact — e.g. SAM-HateMM s2 and MOD-ZH s0/s2 val-sel select ep29 and
coincide with final; SAM-HateMM s2 exactly equals the 13241 floor. Not copy artifacts — the selected epochs differ
and the F1 legs move independently.)*

---

## 3. Outcome tables — paired within head-seed (Δ = arm − the arm's OWN floor, §2), both protocols

### 3.1 ARM A (SAM) — MHC_zh vs ZH floor (§1.1)

| seed | protocol | SAM acc/mF1 | floor acc/mF1 | Δ(SAM−floor) acc/mF1 |
|---|---|---|---|---|
| 0 | val-sel | 0.7852/0.7385 | 0.8322/0.8023 | **−0.0470/−0.0638** |
| 1 | val-sel | 0.8255/0.8002 | 0.8255/0.7956 | **+0.0000/+0.0046** |
| 2 | val-sel | 0.8121/0.7893 | 0.8389/0.8065 | **−0.0268/−0.0172** |
| **mean** | **val-sel** | 0.8076/0.7760 | 0.8322/0.8015 | **−0.0246/−0.0255** (acc sign **0/3**) |
| 0 | final-ep | 0.7987/0.7612 | 0.8456/0.8181 | **−0.0469/−0.0569** |
| 1 | final-ep | 0.8054/0.7646 | 0.8389/0.8113 | **−0.0335/−0.0467** |
| 2 | final-ep | 0.8054/0.7784 | 0.8523/0.8226 | **−0.0469/−0.0442** |
| **mean** | **final-ep** | 0.8032/0.7681 | 0.8456/0.8173 | **−0.0424/−0.0493** (acc sign **0/3**) |

### 3.2 ARM A (SAM) — HateMM vs HateMM floor (§1.2)

| seed | protocol | SAM acc/mF1 | floor acc/mF1 | Δ(SAM−floor) acc/mF1 |
|---|---|---|---|---|
| 0 | val-sel | 0.8791/0.8735 | 0.8791/0.8730 | **+0.0000/+0.0005** |
| 1 | val-sel | 0.8884/0.8828 | 0.8744/0.8678 | **+0.0140/+0.0150** |
| 2 | val-sel | 0.8791/0.8724 | 0.8791/0.8724 | **+0.0000/+0.0000** |
| **mean** | **val-sel** | 0.8822/0.8762 | 0.8775/0.8711 | **+0.0047/+0.0052** (acc sign **1/3**) |
| 0 | final-ep | 0.8884/0.8828 | 0.8791/0.8730 | **+0.0093/+0.0098** |
| 1 | final-ep | 0.8837/0.8776 | 0.8791/0.8724 | **+0.0046/+0.0052** |
| 2 | final-ep | 0.8791/0.8724 | 0.8791/0.8724 | **+0.0000/+0.0000** |
| **mean** | **final-ep** | 0.8837/0.8776 | 0.8791/0.8726 | **+0.0046/+0.0050** (acc sign **2/3**) |

### 3.3 ARM B (mod-dropout) — MHC_zh vs ZH floor (§1.1)

| seed | protocol | mod-drop acc/mF1 | floor acc/mF1 | Δ(mod−floor) acc/mF1 |
|---|---|---|---|---|
| 0 | val-sel | 0.8322/0.8023 | 0.8322/0.8023 | **+0.0000/+0.0000** |
| 1 | val-sel | 0.8523/0.8202 | 0.8255/0.7956 | **+0.0268/+0.0246** |
| 2 | val-sel | 0.8121/0.7771 | 0.8389/0.8065 | **−0.0268/−0.0294** |
| **mean** | **val-sel** | 0.8322/0.7999 | 0.8322/0.8015 | **+0.0000/−0.0016** (acc sign **1/3**) |
| 0 | final-ep | 0.7919/0.7577 | 0.8456/0.8181 | **−0.0537/−0.0604** |
| 1 | final-ep | 0.8389/0.8090 | 0.8389/0.8113 | **+0.0000/−0.0023** |
| 2 | final-ep | 0.8121/0.7771 | 0.8523/0.8226 | **−0.0402/−0.0455** |
| **mean** | **final-ep** | 0.8143/0.7813 | 0.8456/0.8173 | **−0.0313/−0.0361** (acc sign **0/3**) |

### 3.4 ARM B (mod-dropout) — HateMM vs HateMM floor (§1.2)

| seed | protocol | mod-drop acc/mF1 | floor acc/mF1 | Δ(mod−floor) acc/mF1 |
|---|---|---|---|---|
| 0 | val-sel | 0.8512/0.8450 | 0.8791/0.8730 | **−0.0279/−0.0280** |
| 1 | val-sel | 0.8512/0.8476 | 0.8744/0.8678 | **−0.0232/−0.0202** |
| 2 | val-sel | 0.8698/0.8620 | 0.8791/0.8724 | **−0.0093/−0.0104** |
| **mean** | **val-sel** | 0.8574/0.8515 | 0.8775/0.8711 | **−0.0201/−0.0195** (acc sign **0/3**) |
| 0 | final-ep | 0.8651/0.8567 | 0.8791/0.8730 | **−0.0140/−0.0163** |
| 1 | final-ep | 0.8837/0.8776 | 0.8791/0.8724 | **+0.0046/+0.0052** |
| 2 | final-ep | 0.8698/0.8620 | 0.8791/0.8724 | **−0.0093/−0.0104** |
| **mean** | **final-ep** | 0.8729/0.8654 | 0.8791/0.8726 | **−0.0062/−0.0072** (acc sign **1/3**) |

**Paired delta means, per combo per protocol (Δacc / ΔmF1, acc sign):**
- **SAM × ZH:**     val-sel **−0.0246 / −0.0255** (0/3) · final **−0.0424 / −0.0493** (0/3)
- **SAM × HateMM:** val-sel **+0.0047 / +0.0052** (1/3) · final **+0.0046 / +0.0050** (2/3)
- **mod × ZH:**     val-sel **+0.0000 / −0.0016** (1/3) · final **−0.0313 / −0.0361** (0/3)
- **mod × HateMM:** val-sel **−0.0201 / −0.0195** (0/3) · final **−0.0062 / −0.0072** (1/3)

---

## 4. Per-switch rulings (frozen text VERBATIM; each combo ruled exactly as worded)

**KS-arm-dead — the KILL bar (§3.3, DEV-2 sign formalism), verbatim:**
> an arm×dataset cell is **KILLED** iff, on **BOTH protocols**, `mean paired Δacc ≤ 0` **OR** the acc sign is not
> 3/3 positive — i.e. **neither** protocol produces a clean positive-mean-and-3/3-sign result.

**KS-regression note (§3.4), verbatim:** if `mean Δacc ≤ −0.014` on a leg → "SAM / mod-dropout hurts on <dataset>".

**FORMAL promote bar (§3.2), verbatim:** +0.030 acc AND +0.030 mF1, 3/3 sign, under BOTH protocols, judged
independently per protocol.

### 4.1 ARM A (SAM) × MHC_zh → **KILLED; FORMAL FAIL both**
- **KS-arm-dead:** val-sel mean Δacc **−0.0246 ≤ 0** (sign 0/3) → regresses; final mean Δacc **−0.0424 ≤ 0**
  (sign 0/3) → regresses. Neither protocol clean ⇒ **KILLED.**
- **KS-regression:** val-sel −0.0246 ≤ −0.014 **AND** final −0.0424 ≤ −0.014 ⇒ note **"SAM hurts on ZH"** (both
  protocols; below the ±0.014 head-seed spread).
- **FORMAL:** val-sel −0.0246/−0.0255 (0/3), final −0.0424/−0.0493 (0/3) — both far below +0.030/+0.030 conjunct.
  **FAIL both protocols.**

### 4.2 ARM A (SAM) × HateMM → **KILLED; FORMAL FAIL both**
- **KS-arm-dead:** val-sel mean Δacc +0.0047 > 0 **BUT acc sign 1/3** (not 3/3) → OR-clause fires; final mean Δacc
  +0.0046 > 0 **BUT acc sign 2/3** (not 3/3) → OR-clause fires. Neither protocol produces a clean
  positive-mean-and-3/3 result ⇒ **KILLED.** (Directionally a small positive nudge at the near-ceiling hold, but
  not clean-signed and nowhere near the bar.)
- **KS-regression:** neither leg ≤ −0.014 (val-sel +0.0047, final +0.0046) ⇒ **no regression note** (SAM did not
  hurt HateMM).
- **FORMAL:** val-sel +0.0047/+0.0052 (1/3), final +0.0046/+0.0050 (2/3) — both far below +0.030/+0.030, sign not
  3/3. **FAIL both protocols.**

### 4.3 ARM B (mod-dropout) × MHC_zh → **KILLED; FORMAL FAIL both**
- **KS-arm-dead:** val-sel mean Δacc **+0.0000 ≤ 0** (a tie; also sign 1/3) → ties-or-regresses; final mean Δacc
  **−0.0313 ≤ 0** (sign 0/3) → regresses. Neither protocol clean ⇒ **KILLED.**
- **KS-regression:** final −0.0313 ≤ −0.014 ⇒ note **"mod-dropout hurts on ZH"** (final-epoch). (val-sel +0.0000 is
  not ≤ −0.014 → no note on that leg.) This is the pre-declared ARM-B downside-skew on text-carried ZH (F0.5(b)),
  confirmed on the final-epoch leg.
- **FORMAL:** val-sel +0.0000/−0.0016 (1/3), final −0.0313/−0.0361 (0/3) — both below the conjunct. **FAIL both
  protocols.**

### 4.4 ARM B (mod-dropout) × HateMM → **KILLED; FORMAL FAIL both**
- **KS-arm-dead:** val-sel mean Δacc **−0.0201 ≤ 0** (sign 0/3) → regresses; final mean Δacc **−0.0062 ≤ 0**
  (sign 1/3) → regresses. Neither protocol clean ⇒ **KILLED.**
- **KS-regression:** val-sel −0.0201 ≤ −0.014 ⇒ note **"mod-dropout hurts on HateMM"** (val-selected). (final
  −0.0062 not ≤ −0.014 → no note on that leg.) Again the pre-declared ARM-B downside-skew on a text-carried target
  (F0.5(b)).
- **FORMAL:** val-sel −0.0201/−0.0195 (0/3), final −0.0062/−0.0072 (1/3) — both below the conjunct. **FAIL both
  protocols.**

**All four arm×dataset cells KILLED (KS-arm-dead); all four FORMAL-FAIL on both protocols.** D7-DEAD regardless
(F0.3): even a formal PASS would have been an engineering/ablation row, never a novelty win.

---

## 5. Fixed write-up lines (prereg §7.3 format)

```
ARM A (SAM):        MHC_zh:  final-epoch: FAIL; val-selected: FAIL  [FORMAL §3.2]. KS-arm-dead: KILLED.
                    HateMM:  final-epoch: FAIL; val-selected: FAIL              . KS-arm-dead: KILLED.
ARM B (mod-drop):   MHC_zh:  final-epoch: FAIL; val-selected: FAIL              . KS-arm-dead: KILLED.
                    HateMM:  final-epoch: FAIL; val-selected: FAIL              . KS-arm-dead: KILLED.
(+ KS-regression: SAM hurts on ZH (both protocols, Δacc −0.0246 / −0.0424);
   mod-dropout hurts on ZH (final-epoch, Δacc −0.0313); mod-dropout hurts on HateMM (val-selected, Δacc −0.0201).
   No MARGINAL note — no within-noise pass on any cell.)
```

Task-format lines (one per combo, all bars):
- `SAM × MHC_zh:   final-epoch: FAIL (Δacc −0.0424 / ΔmF1 −0.0493, sign 0/3); val-selected: FAIL (Δacc −0.0246 / ΔmF1 −0.0255, sign 0/3); KS-arm-dead: KILLED (both protocols regress); KS-regression: SAM hurts on ZH (both legs ≤ −0.014).`
- `SAM × HateMM:   final-epoch: FAIL (Δacc +0.0046 / ΔmF1 +0.0050, sign 2/3, < +0.030); val-selected: FAIL (Δacc +0.0047 / ΔmF1 +0.0052, sign 1/3, < +0.030); KS-arm-dead: KILLED (positive mean but neither protocol 3/3-signed); no regression note.`
- `mod × MHC_zh:   final-epoch: FAIL (Δacc −0.0313 / ΔmF1 −0.0361, sign 0/3); val-selected: FAIL (Δacc +0.0000 / ΔmF1 −0.0016, sign 1/3); KS-arm-dead: KILLED (both protocols tie-or-regress); KS-regression: mod-dropout hurts on ZH (final-epoch).`
- `mod × HateMM:   final-epoch: FAIL (Δacc −0.0062 / ΔmF1 −0.0072, sign 1/3); val-selected: FAIL (Δacc −0.0201 / ΔmF1 −0.0195, sign 0/3); KS-arm-dead: KILLED (both protocols regress); KS-regression: mod-dropout hurts on HateMM (val-selected).`

---

## 6. Compliance clauses (prereg binds; checked)

- **No-flag byte-identity / floors NOT re-run (F0.7 / §4.1b) — COMPLIANT.** The banked floors (ZH 13150, HateMM
  13241) are the comparators and were **not** re-run: both trainlogs pre-date the family submit and carry **no
  `sam=`/`mod_dropout=` keys** (0 occurrences) — i.e. produced by the pre-patch code. This review re-derived both
  floors from those raw logs and they match the prereg to 4dp (§1). The 4 additive argparse keys land inert on the
  OFF-arm (SAM run: `mod_dropout=False`; mod-dropout run: `sam=False`; both carry `sam_rho=0.05`, `mod_dropout_p=0.3`
  as defaults) — the established additive-flag pattern.
- **Same-code discipline (Namespace diff = the two new flags + derived only) — COMPLIANT.** Runtime Namespaces
  confirm every config pin landed identically to the floor: `fusion_mode='align'`, `topk=20`, `metric='cos'`,
  `loss='triplet'`, `hybrid_loss=True`, `proj_dim=1024`, `map_dim=1024`, `dropout=[0.2,0.4,0.1]`, `batch_norm=False`,
  `epochs=30`, `batch_size=64`, `lr=0.0001`, `grad_clip=0.1`, `no_hard_negatives=1`, `hard_negatives_loss=True`,
  `no_pseudo_gold_positives=1`, `reindex_every_step=False`, `warmup=5`, `lambda_seg=0.0`, `seg_mode='full'`,
  `archive_feats=None`, `lambda_aux=0.0`, `tarc_target_source='off'`, `oracle_probe=False`, `lambda_tarc=0.0`. A
  treatment run's Namespace differs from its floor ONLY in `output_path`/`exp_comment`/`group_name` (all
  derived-inert; group `RAC_video_headrecipe`) + the arm's ≤2 active flags + the 4 inert new keys — **exactly the
  §4.1b/F0.7 pinned diff.** (For the ZH cell the `model` string is even identical to the floor — same
  `…-LoRA_HF` cache; for HateMM both use `…-LoRA-curric_HF`.)
- **Knobs frozen (rho 0.05 / p 0.3) — COMPLIANT.** Every ARM-A run shows `sam=True, sam_rho=0.05`; every ARM-B run
  shows `mod_dropout=True, mod_dropout_p=0.3`. No post-hoc knob tuning; the family is one bite (§3.6).
- **Single test-touch accounting (F0.1) — COMPLIANT.** The 12 job-13478 head reads (4 arm×dataset cells × 3 seeds)
  are the ONLY budgeted head-recipe test evaluations = exactly ONE family test-touch; zero test-touch before this
  verdict (the submit record §5.1 transcribed no gates/deltas). Prior ZH/HateMM test exposures under the identical
  `enc3s` protocol are pre-declared (F0.1) and are re-measurements, not first exposures.
- **Collision safety (§4.3) — COMPLIANT.** Exactly 12 `hr_*_13478.trainlog` present; family group
  `RAC_video_headrecipe` exists under both `logging/Retrieval/{MHC_zh,HateMM}/` (the real run's own group, distinct
  from the floor groups); banked floor caches + trainlogs are read-only inputs, untouched (floor logs pre-date the
  family). No smoke residue: `logging/Retrieval/*/_smoke_hr` and `slurm/logs/hr_smoke_*` both **ABSENT (0)** at
  verdict time (cleaned per submit record §4.2).
- **Smoke + codex-gate evidence recorded — COMPLIANT.** Submit record §3 documents the mandatory codex gate
  (`gpt-5.4` xhigh, `--full-auto`, session `019f954e-…`): **NO P1 findings**, re-mine-reuse invariant (F0.6)
  SATISFIED under the deployed dense CPU-FAISS path (`retrieval.py:341` gate stays closed at w+ε, assert placed
  before perturbation), SAM ordering + global grad-norm correct, no-flag byte-identity preserved, mod-dropout
  ones-fill/at-most-one-stream/eval-gate correct; two P2s both inactive under the deployed config (`aux_pack` off,
  assert satisfied since dense-FAISS always mines) ⇒ no code fix ⇒ shas UNCHANGED (freeze intact). Submit record §4
  documents smoke PASS: $0-CPU mask-rate (drop 0.2965 / img 0.1487 / text 0.1478 / both 0, eval-gate off) +
  no-flag Namespace equivalence, and GPU smoke (job 13477, exit 0:0) — loss finite for all 3 arms, SAM re-mine-reuse
  assert did NOT trip, SAM double-step VISIBLE (≈1.42× baseline wall). Smoke artifacts deleted.
- **Two disclosed headwinds carried (F0.5) — COMPLIANT + BORNE OUT.** (a) ARM-A SAM F69 grad-norm↔acc wrong-sign
  headwind: SAM produced no clean gain on either dataset (regresses ZH materially; near-ceiling HateMM nudge not
  3/3). (b) ARM-B mod-dropout downside-skew on text-carried datasets (F45/F58): mod-dropout regresses ZH
  (final-epoch −0.0313) and HateMM (val-sel −0.0201) — the pre-declared "most likely hurts" direction is
  measured. (c) HateMM near-ceiling: bar ~0.909 untouched. (d) family = one bite, knobs frozen — honored.
- **Freeze integrity — COMPLIANT.** Prereg self-sha + A/B/C + reused-machinery (loss.py, retrieval.py) shas all
  MATCH on disk at verdict time (§0); no drift since freeze/submit.

**No compliance violations found.**

**Four review notes that travel (from the submit record; non-blocking, decision-inert):** (1) mod-dropout perturbs
the retrieval *query* too (within-mechanism; the mining INDEX stays clean, encoded under `model.eval()`); (2) SAM
clips/steps on the perturbed gradient (standard SAM); (3) the §4.4.2 mask-rate reference numbers are one seed's
draw, not a fixed target; (4) the SAM re-mine assert is conservatively over-broad (can only ever block, never
fabricate a pass). None affect any ruling above.

---

## 7. FINAL VERDICT BLOCK (performance clause only)

**Prereg §7.3 fixed write-up:**
```
ARM A (SAM):        MHC_zh:  final-epoch: FAIL; val-selected: FAIL  [FORMAL §3.2]. KS-arm-dead: KILLED.
                    HateMM:  final-epoch: FAIL; val-selected: FAIL              . KS-arm-dead: KILLED.
ARM B (mod-drop):   MHC_zh:  final-epoch: FAIL; val-selected: FAIL              . KS-arm-dead: KILLED.
                    HateMM:  final-epoch: FAIL; val-selected: FAIL              . KS-arm-dead: KILLED.
KS-regression: SAM hurts on ZH (both protocols); mod-dropout hurts on ZH (final-epoch) and HateMM (val-selected).
```

**Per-switch (verbatim rulings), all four combos:**
- **SAM × MHC_zh — KILLED; FORMAL FAIL both.** val-sel Δacc −0.0246/ΔmF1 −0.0255 (0/3); final Δacc −0.0424/ΔmF1
  −0.0493 (0/3). Both protocols regress ⇒ KILLED; both legs ≤ −0.014 ⇒ **SAM hurts on ZH**.
- **SAM × HateMM — KILLED; FORMAL FAIL both.** val-sel Δacc +0.0047/ΔmF1 +0.0052 (1/3); final Δacc +0.0046/ΔmF1
  +0.0050 (2/3). Positive means but neither protocol is 3/3-signed ⇒ no clean result ⇒ KILLED; no regression
  (SAM did not hurt HateMM). Far below the +0.030 bar.
- **mod × MHC_zh — KILLED; FORMAL FAIL both.** val-sel Δacc +0.0000/ΔmF1 −0.0016 (1/3); final Δacc −0.0313/ΔmF1
  −0.0361 (0/3). Tie-or-regress on both ⇒ KILLED; final ≤ −0.014 ⇒ **mod-dropout hurts on ZH (final-epoch)**.
- **mod × HateMM — KILLED; FORMAL FAIL both.** val-sel Δacc −0.0201/ΔmF1 −0.0195 (0/3); final Δacc −0.0062/ΔmF1
  −0.0072 (1/3). Both regress ⇒ KILLED; val-sel ≤ −0.014 ⇒ **mod-dropout hurts on HateMM (val-selected)**.

**Composite (performance clause only):** the two remaining head-training-dynamics escape hatches — **SAM**
(flat-minima optimizer) and **modality-dropout** (identity-fill stream-dropout regularizer) — applied to the
deployed RGCL align-fusion head over cached LoRA features, 3-seed paired dual-protocol vs each dataset's banked
floor, produce **no net head-level gain on either dataset under either protocol**. All **four** arm×dataset cells
are **KS-arm-dead (KILLED)** and **FORMAL-FAIL on both protocols**. On the realistic target (marginal ZH), SAM is a
material regression (val-sel −0.0246, final −0.0424, both 0/3) and mod-dropout is a wash-to-regression (val-sel
±0.0000, final −0.0313); the pre-declared ARM-B downside-skew on text-carried ZH/HateMM (F0.5(b)) is measured on
three legs. On the near-ceiling hold (HateMM), SAM nudges the mean up ~+0.005 on both protocols but never
3/3-signed and nowhere near the ~0.909 bar, and mod-dropout regresses it. This is the prereg's pre-declared honest
most-likely outcome (F0.5: priors ~8–12%, downside-skewed; F69 SAM headwind; text-carried mod-dropout headwind) —
**both head-recipe doors are CLOSED at < 0.15 GPU-h**: two prose-argued escape hatches converted to measured
door-closers in one multiplicity bite. No formal pass, no surviving cell, no novelty (SAM + modality-dropout are
D7-DEAD generic training knobs by construction, F0.3). No compliance violation; the banked floors are untouched and
not re-run; the head is byte-identical same-code modulo the two additive flags.

**Out of scope for this reviewer (F0.3 / §8):** SAM and modality-dropout carry no novelty weight regardless of sign
(D7-DEAD by construction), and goal-level satisfaction is a USER ruling. This review renders the **performance
clause only**, as the frozen prereg mandates.

---

*Reviewer statements: hash verified before any metric was read; both floors (ZH 13150, HateMM 13241) re-derived
from raw trainlogs with the byte-identical enc3seed parser and match the prereg §2.1/§2.2 to 4dp; the 12 family
runs (job 13478) re-parsed from raw and line-verified (per-seed both protocols, selected-epoch argmax, TEST-line
line numbers); the runtime Namespaces confirm same-code + frozen knobs vs the floors; banked floors confirmed not
re-run (no `sam=`/`mod_dropout=` keys, pre-dating the family); artifact + reused-machinery shas confirmed unchanged
at verdict time; no GPU/SLURM/Modal spent; no `state/` mutated; nothing pushed; no goal/novelty claim made.*
