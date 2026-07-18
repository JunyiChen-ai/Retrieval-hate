# CAND-2 Curriculum LoRA-SFT — INDEPENDENT 0-CONTEXT VERDICT REVIEW

**Reviewer role:** independent 0-context verdict reviewer. No prior project context. Renders the binding
verdict strictly against the frozen pre-registration `refine-logs/CAND2_CURRICULUM_PREREG.md` VERBATIM.
Zero user interaction. Modified nothing except this file. **Out of scope (USER's rulings, per prereg F0.3/§8):
the D7 novelty sub-ruling and goal-level satisfaction — this review decides the PERFORMANCE clause only.**
**Date:** 2026-07-18.

---

## 0. Hash-freeze verification (done FIRST)

```
on-disk sha256(refine-logs/CAND2_CURRICULUM_PREREG.md)
  = e5a689d9ede0a79e89eb041f028228e70cdc821029349387e7ec9fdff939790e
expected (task + CAND2_FREEZE.md + CAND2_SUBMIT_RECORD.md)
  = e5a689d9ede0a79e89eb041f028228e70cdc821029349387e7ec9fdff939790e
```
**MATCH.** The prereg on disk is the frozen binding text (commit `76ef0e2`). Proceeding.

**Measurement provenance (raw logs only, job IDs per CAND2_SUBMIT_RECORD.md):** curriculum head reads =
job **13241**, six trainlogs `slurm/logs/enc3s_{MHC_zh,HateMM}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_seed{0,1,2}_13241.trainlog`
(SFT 13237/13238 → extract 13239/13240 → head 13241). Comparison arms re-parsed from raw:
ZH generic-LoRA **13150**, ZH frozen-CLIP **13115**, HateMM generic-LoRA **13235**, HateMM frozen-CLIP **12850**.
Every number below re-derived with the **byte-identical `enc3seed.sbatch` readout parser**
(val-sel = epoch ≥ warmup 5 maximising `(Val_Retrieval acc, roc)`, report that epoch's TEST metrics;
final = max-epoch TEST metrics). Parser faithfulness hand-verified against raw lines (e.g. ZH-curric s0 val-sel
tie at val-acc 0.8846 across ep13/20/21, roc tie-break → ep13; HateMM-curric final ep29 lines).

### 0.1 Comparison arms — re-derived vs the prereg's §2.1/§2.2 pinned tables (numeric-provenance discipline)

Every generic/floor mean below matches the prereg's §2.1/§2.2 to **4dp** (independent re-parse):

| arm | protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | mean acc/F1 | prereg §2 match |
|---|---|---|---|---|---|---|
| ZH frozen-CLIP (13115) | val-sel | 0.8054/0.7706 | 0.8054/0.7579 | 0.8121/0.7742 | 0.8076/0.7676 | ✔ |
| ZH frozen-CLIP (13115) | final-ep | 0.8054/0.7706 | 0.8054/0.7542 | 0.8322/0.7913 | 0.8143/0.7720 | ✔ |
| ZH generic-LoRA (13150) | val-sel | 0.8322/0.8023 | 0.8255/0.7956 | 0.8389/0.8065 | 0.8322/0.8015 | ✔ |
| ZH generic-LoRA (13150) | final-ep | 0.8456/0.8181 | 0.8389/0.8113 | 0.8523/0.8226 | 0.8456/0.8173 | ✔ |
| HateMM frozen-CLIP (12850) | val-sel | 0.8279/0.8172 | 0.8279/0.8163 | 0.8047/0.7920 | 0.8202/0.8085 | ✔ |
| HateMM frozen-CLIP (12850) | final-ep | 0.8186/0.7997 | 0.8047/0.7822 | 0.8140/0.7988 | 0.8124/0.7936 | ✔ |
| HateMM generic-LoRA (13235) | val-sel | 0.8605/0.8521 | 0.8698/0.8620 | 0.8558/0.8495 | 0.8620/0.8545 | ✔ |
| HateMM generic-LoRA (13235) | final-ep | 0.8651/0.8580 | 0.8744/0.8660 | 0.8698/0.8613 | 0.8698/0.8618 | ✔ |

The prereg's floor/generic transcriptions are trustworthy; all comparisons below use these re-derived numbers.

---

## 1. Curriculum arm — raw measured numbers (job 13241, re-parsed + spot-checked vs raw lines)

| arm | protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | mean acc/F1 |
|---|---|---|---|---|---|
| **ZH-curric** (13241) | val-sel | 0.8188/0.7837 (ep13) | 0.8523/0.8270 (ep18) | 0.8054/0.7734 (ep5) | **0.8255/0.7947** |
| **ZH-curric** (13241) | final-ep | 0.8591/0.8339 | 0.8523/0.8249 | 0.8456/0.8158 | **0.8523/0.8249** |
| **HateMM-curric** (13241) | val-sel | 0.8791/0.8730 (ep29) | 0.8744/0.8678 (ep14) | 0.8791/0.8724 (ep10) | **0.8775/0.8711** |
| **HateMM-curric** (13241) | final-ep | 0.8791/0.8730 | 0.8791/0.8724 | 0.8791/0.8724 | **0.8791/0.8726** |

Raw-line spot-checks (verbatim from trainlogs): HateMM-curric s0 final `Test_Retrieval Epoch 29 macroF1: 0.8730
… acc: 0.8791`; s1 val-sel `Epoch 14 macroF1: 0.8678 … acc: 0.8744`; s2 val-sel `Epoch 10 macroF1: 0.8724 …
acc: 0.8791`; ZH-curric s1 val-sel `Epoch 18 macroF1: 0.8270 … acc: 0.8523`; s2 val-sel `Epoch 5 macroF1: 0.7734
… acc: 0.8054`. All match the parser.

---

## 2. Outcome tables — filled cell-by-cell (prereg §7.1 / §7.2)

### 2.1 ZH — curriculum-LoRA vs frozen-CLIP (K-C2-1) AND vs generic-LoRA (K-C2-2)

| seed | protocol | curric acc/F1 | CLIP acc/F1 | Δ(curric−CLIP) acc/F1 | generic acc/F1 | Δ(curric−generic) acc/F1 |
|---|---|---|---|---|---|---|
| 0 | val-sel | 0.8188/0.7837 | 0.8054/0.7706 | **+0.0134/+0.0131** | 0.8322/0.8023 | **−0.0134/−0.0186** |
| 1 | val-sel | 0.8523/0.8270 | 0.8054/0.7579 | **+0.0469/+0.0691** | 0.8255/0.7956 | **+0.0268/+0.0314** |
| 2 | val-sel | 0.8054/0.7734 | 0.8121/0.7742 | **−0.0067/−0.0008** | 0.8389/0.8065 | **−0.0335/−0.0331** |
| **mean** | **val-sel** | **0.8255/0.7947** | 0.8076/0.7676 | **+0.0179/+0.0271** | 0.8322/0.8015 | **−0.0067/−0.0068** |
| 0 | final-ep | 0.8591/0.8339 | 0.8054/0.7706 | **+0.0537/+0.0633** | 0.8456/0.8181 | **+0.0135/+0.0158** |
| 1 | final-ep | 0.8523/0.8249 | 0.8054/0.7542 | **+0.0469/+0.0707** | 0.8389/0.8113 | **+0.0134/+0.0136** |
| 2 | final-ep | 0.8456/0.8158 | 0.8322/0.7913 | **+0.0134/+0.0245** | 0.8523/0.8226 | **−0.0067/−0.0068** |
| **mean** | **final-ep** | **0.8523/0.8249** | 0.8143/0.7720 | **+0.0380/+0.0529** | 0.8456/0.8173 | **+0.0067/+0.0076** |

Sign vectors — curric−CLIP: val-sel acc `[+,+,−]` (2/3); final acc `[+,+,+]` (3/3). curric−generic: val-sel acc
`[−,+,−]` (1/3); final acc `[+,+,−]` (2/3).

### 2.2 HateMM — curriculum-LoRA vs frozen-CLIP (K-C2-1) AND vs generic-LoRA (K-C2-2)

| seed | protocol | curric acc/F1 | CLIP acc/F1 | Δ(curric−CLIP) acc/F1 | generic acc/F1 | Δ(curric−generic) acc/F1 |
|---|---|---|---|---|---|---|
| 0 | val-sel | 0.8791/0.8730 | 0.8279/0.8172 | **+0.0512/+0.0558** | 0.8605/0.8521 | **+0.0186/+0.0209** |
| 1 | val-sel | 0.8744/0.8678 | 0.8279/0.8163 | **+0.0465/+0.0515** | 0.8698/0.8620 | **+0.0046/+0.0058** |
| 2 | val-sel | 0.8791/0.8724 | 0.8047/0.7920 | **+0.0744/+0.0804** | 0.8558/0.8495 | **+0.0233/+0.0229** |
| **mean** | **val-sel** | **0.8775/0.8711** | 0.8202/0.8085 | **+0.0573/+0.0626** | 0.8620/0.8545 | **+0.0155/+0.0166** |
| 0 | final-ep | 0.8791/0.8730 | 0.8186/0.7997 | **+0.0605/+0.0733** | 0.8651/0.8580 | **+0.0140/+0.0150** |
| 1 | final-ep | 0.8791/0.8724 | 0.8047/0.7822 | **+0.0744/+0.0902** | 0.8744/0.8660 | **+0.0047/+0.0064** |
| 2 | final-ep | 0.8791/0.8724 | 0.8140/0.7988 | **+0.0651/+0.0736** | 0.8698/0.8613 | **+0.0093/+0.0111** |
| **mean** | **final-ep** | **0.8791/0.8726** | 0.8124/0.7936 | **+0.0667/+0.0790** | 0.8698/0.8618 | **+0.0093/+0.0108** |

Sign vectors — curric−CLIP: val-sel acc `[+,+,+]` (3/3); final acc `[+,+,+]` (3/3). curric−generic: val-sel acc
`[+,+,+]` (3/3); final acc `[+,+,+]` (3/3).

---

## 3. Per-switch rulings (frozen text VERBATIM; each ruled exactly as worded)

### K-C2-0 — mining validity ($0 CPU pre-GPU gate) — **PASS both (banked result STANDS)**

Re-read the banked diagnostics `refine-logs/CAND2_KC20_{MHC_zh,HateMM}.json`; both carry `"KC20_PASS": true`
and the frozen `train_curric_json_sha256` (`c8260dd3…` ZH / `73307ef2…` HateMM) equal to freeze-block F/G, i.e.
the mining that graded these gates is the mining that built the trained curriculum.

| check | criterion | ZH | HateMM |
|---|---|---|---|
| (a) non-degenerate boundary | frozen LOO kNN error ∈ [0.15,0.35] | 0.2073 ✔ | 0.1935 ✔ |
| (b) concentration | confusion-weight Gini ≥ 0.30 | 0.5634 ✔ | 0.6497 ✔ |
| (c) differs from uniform | curric unique coverage < 0.90·N | 0.6667 ✔ | 0.6756 ✔ |
| descriptor | top-30% confusable mass vs uniform | 2.11× | 2.08× |

Neither dataset landed at the ≈0 memorization auto-KILL. **K-C2-0 = PASS both; the banked $0 result stands.**

### K-C2-1 — performance, must HOLD the inherited passes (curric − CLIP; §3.3)

Rule = per dataset × protocol: mean Δacc ≥ +0.030 AND mean ΔmF1 ≥ +0.030 AND sign 3/3 AND ≥ (generic − 0.014).

- **ZH final-ep:** mean +0.0380 acc / +0.0529 F1, sign 3/3, non-regression 0.8523 ≥ (0.8456−0.014=0.8316). **PASS
  — MARGINAL** (per B3 §2.2 precedent: seed2 Δacc = +0.0134 sits below the +0.030 per-seed bar; mean clears
  +0.030 but not +0.040).
- **ZH val-sel:** mean +0.0179 acc (< +0.030), sign 2/3 (seed2 −0.0067). **FAIL.**
- **HateMM final-ep:** mean +0.0667 acc / +0.0790 F1, sign 3/3, all seeds ≥ +0.030, non-regression
  0.8791 ≥ 0.8558. **PASS (non-marginal) — HOLD confirmed.**
- **HateMM val-sel:** mean +0.0573 acc / +0.0626 F1, sign 3/3, all seeds ≥ +0.030, non-regression
  0.8775 ≥ 0.8480. **PASS (non-marginal) — HOLD confirmed.**

ZH replicates B3's protocol-dependent pattern (final-ep marginal PASS, val-sel FAIL). HateMM holds its inherited
both-protocol pass. **K-C2-1: ZH = final-ep PASS (marginal) / val-sel FAIL; HateMM = PASS both (held).**

### K-C2-2 — ADD-OVER-GENERIC, the novelty-earning bar (curric − generic, head-seed paired; §3.4)

Rule PASS (per dataset, ≥1 protocol) = mean Δacc ≥ +0.010 AND sign 3/3 positive AND mean ΔmF1 ≥ 0.
TIE (= NO NOVELTY, the F0.7 outcome) = mean |Δacc| < +0.010 **OR** sign not 3/3.

- **ZH val-sel:** mean −0.0067 acc, sign 1/3. |Δacc| < +0.010 AND sign not 3/3 ⇒ **TIE.**
- **ZH final-ep:** mean +0.0067 acc (< +0.010), sign 2/3 (seed2 −0.0067). ⇒ **TIE.**
- **HateMM val-sel:** mean **+0.0155** acc (≥ +0.010), sign **3/3** positive, mean ΔmF1 **+0.0166** (≥ 0). ⇒ **PASS.**
- **HateMM final-ep:** mean +0.0093 acc (**< +0.010** by 0.0007), sign 3/3 positive. First clause of TIE
  (`mean |Δacc| < +0.010`) trips ⇒ **TIE.**

Per-dataset (≥1-protocol rule): **ZH K-C2-2 = TIE both protocols (NO NOVELTY on ZH);
HateMM K-C2-2 = PASS (via val-sel; final-ep TIE).** The prereg's novelty signal ("K-C2-2 PASS on ≥1 dataset")
is therefore **MET — but on HateMM, not on ZH** (the a-priori-most-likely leg, §3.6). **F0.2 caveat travels
with this PASS (mandatory):** it is a **single curriculum draw vs a single generic draw**, read by 3 head-seeds —
head-seed-paired, cannot separate the curriculum effect from SFT-draw luck; a stability claim would need ≥3 fresh
curriculum retrains (out of scope, pre-declared). The HateMM PASS is also **protocol-split** (val-sel only;
final-ep is a +0.0093 near-miss tie) and lands on the **hold/image-inherited leg** (F0.4), not the primary ZH leg.

### KS-regression — below-generic KILL (mean Δacc(curric−generic) ≤ −0.014 on a held leg; §3.5)

Most-negative leg-mean = ZH val-sel **−0.0067** (> −0.014); all other legs ≥ 0. **NOT triggered** (no dataset/
protocol degraded beyond the head-seed spread). No KILL.

### KS-below-floor — regime sanity (curric below CLIP floor on ZH; §3.6)

ZH-curric means are above the CLIP floor on both protocols (val-sel 0.8255 > 0.8076; final 0.8523 > 0.8143);
per-seed only the single val-sel seed2 dips −0.0067 below its CLIP pair, arm-level clearly above.
**NOT triggered.** The curriculum did not break the ZH mechanism.

### ZH-robustness clause — pre-declared "ZH leg strengthened" pattern (§3.7)

Declared strengthened iff EITHER (curric − CLIP):
- **(a) val-sel conjunct now PASSES** (mean Δacc ≥ +0.030 AND ΔmF1 ≥ +0.030, 3/3): measured +0.0179 acc, sign
  2/3 → **FAIL (a).**
- **(b) final-ep becomes NON-marginal** (mean Δacc ≥ +0.040 AND 3/3 per-seed Δacc ≥ +0.030): measured mean
  +0.0380 (< +0.040) and seed2 +0.0134 (< +0.030) → **FAIL (b).**

Neither pattern met ⇒ **ZH-robustness = NOT strengthened.** ZH final-ep remains a marginal pass, essentially
B3's status (B3 final-ep +0.0313; curric +0.0380 — higher but still sub-+0.040 with seed2 below the per-seed
bar). The primary declared performance goal of cand-2 — strengthen the marginal ZH leg — was **not achieved.**

---

## 4. Compliance clauses (prereg binds; checked)

- **Same-code pairing (§4.1c/§4.2):** Namespace diff, HateMM-curric(13241) vs HateMM-generic(13235) same seed =
  **only** `model`, `group_name`, `exp_comment`, and the derived `output_path` differ; **76/80 fields identical**
  (fusion_mode align, topk 20, proj/map 1024, dropout [0.2,0.4,0.1], bz 64, lr 1e-4, epochs 30, triplet,
  hybrid_loss, warmup 5, hard_neg 1, cos, lambda_seg 0, archive OFF). The code-version confound is retired.
  **COMPLIANT.**
- **Single-curriculum-draw (F0.2):** one SFT draw per dataset (13237 ZH, 13238 HateMM), one extraction each
  (13239/13240), 3 head-seeds (13241). Honored; the caveat travels with the HateMM K-C2-2 PASS (§3 above).
  **COMPLIANT.**
- **Single test-touch:** the job-13241 curriculum-LoRA head reads are the ONLY budgeted curriculum-encoder test
  evaluations (ZH + HateMM); no earlier curriculum test exposure. **COMPLIANT.**
- **Class-balance disclosure (F0.8):** the confusion-weighting class-balance shift was pre-declared
  (ZH 31.1%→41.1%, HateMM 40.1%→37.7% hateful); non-blocking 0.1pt rounding slip noted by the prereg review.
  **DISCLOSED.**
- **Freeze integrity (carried):** prereg sha matches; freeze-block A–H + reused-machinery shas matched at freeze
  (CAND2_FREEZE.md); builder bit-exact idempotent; KC20 JSON `train_curric` shas equal frozen F/G. Non-blocking
  echoed note: HateMM KC20 `n_train_cache 744` vs `n_train_sft 743`, `n_anchor_missing_from_cache 0` — all 743
  SFT anchors present in the cache; one cache-only train video is a potential LOO neighbour only. Train-only, no
  leakage, predates cand-2. **BENIGN.**

**No compliance violations found.**

---

## 5. FINAL VERDICT BLOCK

**Prereg §7.3 fixed write-up format:**

```
ZH:     final-epoch: PASS (K-C2-1, MARGINAL) · K-C2-2: tie · ZH-robustness: not strengthened.
        val-selected: FAIL (K-C2-1)          · K-C2-2: tie.
HateMM: final-epoch: PASS (K-C2-1, hold)     · K-C2-2: tie.
        val-selected: PASS (K-C2-1, hold)     · K-C2-2: pass (single-draw caveat, F0.2).
```

**Per-switch (verbatim rulings):**
- **K-C2-0 (mining validity):** PASS both — banked $0 result STANDS.
- **K-C2-1 (hold inherited pass):** ZH final-ep PASS (MARGINAL) / val-sel FAIL; HateMM PASS both protocols (held).
- **K-C2-2 (add-over-generic, novelty bar):** **ZH = TIE both protocols (NO NOVELTY on ZH — the F0.7 outcome
  on the primary leg);** **HateMM = PASS (val-sel, +0.0155 acc, 3/3 sign, ΔmF1 +0.0166; final-ep TIE at +0.0093).**
  Novelty signal ("K-C2-2 PASS on ≥1 dataset") **MET on HateMM only,** carrying the F0.2 single-curriculum-draw
  caveat and a protocol-split (val-sel-only) caveat, on the hold/image-inherited leg (F0.4) — not on ZH.
- **KS-regression:** NOT triggered (no leg ≤ −0.014). **KS-below-floor:** NOT triggered (ZH-curric above CLIP floor).
- **ZH-robustness:** NOT strengthened (neither §3.7(a) nor (b) met).

**Composite (performance clause only):** The curriculum **held** both inherited K-C2-1 passes (HateMM both
protocols; ZH final-ep, still marginal). It did **not** deliver the primary declared performance upgrade — the ZH
leg is **not** strengthened and **ties** generic on both protocols (K-C2-2 TIE = "generic LoRA with reshuffled
data" on ZH, the prereg's own pre-declared wording). The add-over-generic bar (K-C2-2) is met on **exactly one
dataset — HateMM, val-selected protocol only, single-draw** — an outcome the prereg admits under its "≥1 dataset,
≥1 protocol" wording but which lands off the a-priori-favoured leg and does not coincide with a ZH-robustness
upgrade (§8's first bullet requires BOTH; only the K-C2-2-on-one-dataset half is satisfied). No kill-switch fired;
no regression; no floor breach; no compliance violation.

**Explicitly out of scope for this reviewer (USER's rulings, prereg F0.3/§8):** whether the single-draw,
protocol-split, HateMM-only K-C2-2 pass suffices for the **D7 memory→adaptation-coupling novelty sub-ruling**, and
whether the overall result satisfies the **goal**. This review renders the performance clause only, as the frozen
prereg mandates.

---

*Reviewer statements: hash verified before reading any metric; every floor/generic/curriculum number re-derived
from raw trainlogs with the byte-identical enc3seed parser and spot-checked against raw lines; no GPU/SLURM/Modal
spent; no state/ mutated; nothing pushed; no goal/novelty claim made.*
