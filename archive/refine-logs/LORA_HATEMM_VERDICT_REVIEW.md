# LoRA-HateMM — INDEPENDENT 0-CONTEXT VERDICT REVIEW

**Reviewer:** independent 0-context verdict reviewer (no prior context; zero user interaction).
**Date:** 2026-07-18.
**Scope:** render the binding verdict for `refine-logs/LORA_HATEMM_PREREG.md` STRICTLY against the frozen
pre-registration text. Performance clause only. **No goal-level / novelty (D7) judgment is made here — that is
the user's ruling, per prereg F0.3 / §9.**
**Measurement:** job chain 13233 (`lora_sft` HateMM) → 13234 (`gen_embed_lora`) → 13235 (`enc3seed`, GROUP
`RAC_video_lora_hm`; 3 HateMM-LoRA + 3 MHC-EN-LoRA head rows). All numbers below RE-PARSED by me from the raw
trainlogs with the EXACT `enc3seed.sbatch` embedded parser (val-selected = epoch ≥ warmup 5 with max
`Val_Retrieval` acc, roc tie-break → that epoch's `Test_Retrieval` acc/macroF1; final-epoch = max epoch's
`Test_Retrieval` acc/macroF1).

---

## 0. Hash-freeze verification (gate — MISMATCH would have stopped the review)

| artifact | expected sha256 | measured | status |
|---|---|---|---|
| `refine-logs/LORA_HATEMM_PREREG.md` | `da1759a4b18fd7689c9086d5343a0e6805dedd24b76c3f2933bdc83a1941cd4b` | `da1759a4b18fd7689c9086d5343a0e6805dedd24b76c3f2933bdc83a1941cd4b` | **MATCH** |
| A `…/hatemm_qwen25vl_lora_sft.yaml` | `d2f415cd93fa6f7b439fd4b4573a536baf48ad42186dc8bd50f3fab20553e36a` | identical | **MATCH** |
| B `scripts/slurm/lora_sft.sbatch` | `e767eba0ca6ff40679857e5efb759d72aa985629a9ece6584ea424ac2baba62f` | identical | **MATCH** |
| C `scripts/slurm/enc3seed_lora_hatemm.sbatch` | `19c76b177f7dc883a9e03524450ad2e6cb302cdd0a6704d69da68a62188a06fc` | identical | **MATCH** |

Frozen prereg hash matches to the byte. Authorization intact (prereg §6.4: "any mismatch = authorization VOID" — no mismatch).

---

## 1. Comparison floors — INDEPENDENTLY RE-DERIVED from primary 12850 trainlogs

Re-parsed by me from `slurm/logs/enc3s_*_12850.trainlog` (+ EN-Qwen `arcbase_MHC_*_1227{5,6}.trainlog` for
seeds 1/2). **Every floor number matches prereg §2 to 4dp** — the prereg's floors are honest.

### 1.1 HateMM floors (re-derived)

| floor | protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | 3-seed mean acc/F1 |
|---|---|---|---|---|---|
| frozen-CLIP (KS-1 pairs vs this) | val-sel | 0.8279/0.8172 | 0.8279/0.8163 | 0.8047/0.7920 | **0.8202 / 0.8085** |
| frozen-CLIP | final-ep | 0.8186/0.7997 | 0.8047/0.7822 | 0.8140/0.7988 | **0.8124 / 0.7936** |
| frozen-Qwen (KS-2 pairs vs this) | val-sel | 0.8698/0.8606 | 0.8651/0.8586 | 0.8837/0.8753 | **0.8729 / 0.8648** |
| frozen-Qwen | final-ep | 0.8605/0.8507 | 0.8605/0.8514 | 0.8837/0.8753 | **0.8682 / 0.8591** |

### 1.2 MHC-EN floors (re-derived; for the bundled B4-EN arm)

| floor | protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | 3-seed mean acc/F1 |
|---|---|---|---|---|---|
| frozen-CLIP (EN KS-1 pairs vs this) | val-sel | 0.7826/0.7113 | 0.7329/0.6034 | 0.7702/0.6997 | **0.7619 / 0.6715** |
| frozen-CLIP | final-ep | 0.7640/0.7145 | 0.7826/0.7159 | 0.7888/0.7303 | **0.7785 / 0.7202** |
| frozen-Qwen (EN, context) | val-sel | 0.7888/0.7378 | 0.7826/0.7283 | 0.7702/0.6997 | **0.7805 / 0.7219** |
| frozen-Qwen (EN, context) | final-ep | 0.8012/0.7596 | 0.7702/0.7203 | 0.7826/0.7475 | **0.7847 / 0.7425** |

---

## 2. Outcome tables — filled from raw 13235 trainlogs

### 2.1 HateMM — LoRA-Qwen vs frozen-CLIP (`enc3s_HateMM_…-LoRA_HF_seed{0,1,2}_13235.trainlog`)

| seed | protocol | LoRA acc/F1 (sel epoch) | CLIP acc/F1 | Δacc | ΔF1 |
|---|---|---|---|---|---|
| 0 | val-sel | 0.8605/0.8521 (ep19) | 0.8279/0.8172 | +0.0326 | +0.0349 |
| 1 | val-sel | 0.8698/0.8620 (ep14) | 0.8279/0.8163 | +0.0419 | +0.0457 |
| 2 | val-sel | 0.8558/0.8495 (ep22) | 0.8047/0.7920 | +0.0511 | +0.0575 |
| **mean** | **val-sel** | **0.8620/0.8545** | **0.8202/0.8085** | **+0.0419** | **+0.0460** |
| 0 | final-ep | 0.8651/0.8580 (ep29) | 0.8186/0.7997 | +0.0465 | +0.0583 |
| 1 | final-ep | 0.8744/0.8660 (ep29) | 0.8047/0.7822 | +0.0697 | +0.0838 |
| 2 | final-ep | 0.8698/0.8613 (ep29) | 0.8140/0.7988 | +0.0558 | +0.0625 |
| **mean** | **final-ep** | **0.8698/0.8618** | **0.8124/0.7936** | **+0.0573** | **+0.0682** |

Sign consistency: **val-sel 3/3 positive (acc and mF1); final-ep 3/3 positive (acc and mF1).**
Effect-size descriptors (n=3, NO significance claim per rule §3.1.3): val-sel paired-t acc +7.84 / mF1 +7.05;
final-ep paired-t acc +8.51 / mF1 +8.64.

### 2.2 MHC-EN — LoRA-Qwen vs frozen-CLIP (bundled B4 closure; `enc3s_MHC_…-LoRA_HF_seed{0,1,2}_13235.trainlog`)

| seed | protocol | LoRA acc/F1 (sel epoch) | CLIP acc/F1 | Δacc | ΔF1 |
|---|---|---|---|---|---|
| 0 | val-sel | 0.7516/0.6916 (ep26) | 0.7826/0.7113 | −0.0310 | −0.0197 |
| 1 | val-sel | 0.7391/0.6920 (ep5) | 0.7329/0.6034 | +0.0062 | +0.0886 |
| 2 | val-sel | 0.7888/0.7506 (ep29) | 0.7702/0.6997 | +0.0186 | +0.0509 |
| **mean** | **val-sel** | **0.7598/0.7114** | **0.7619/0.6715** | **−0.0021** | **+0.0399** |
| 0 | final-ep | 0.7702/0.7302 (ep29) | 0.7640/0.7145 | +0.0062 | +0.0157 |
| 1 | final-ep | 0.7764/0.7360 (ep29) | 0.7826/0.7159 | −0.0062 | +0.0201 |
| 2 | final-ep | 0.7888/0.7506 (ep29) | 0.7888/0.7303 | +0.0000 | +0.0203 |
| **mean** | **final-ep** | **0.7785/0.7389** | **0.7785/0.7202** | **+0.0000** | **+0.0187** |

Sign consistency: val-sel acc 2/3, mF1 2/3; final-ep acc 1/3 (seed2 exactly 0), mF1 3/3.
(Prereg §3.5 pre-declared "B4 seed0 anchor: val-sel −0.031 acc, final +0.006 acc" — my re-derivation matches
exactly: val-sel seed0 Δacc −0.0310, final seed0 Δacc +0.0062. The honest expected-FAIL prior is confirmed.)

---

## 3. Kill-switch rulings (prereg wording applied verbatim)

### KS-1 — PERFORMANCE CONJUNCT (primary kill; §3.2): mean Δacc ≥ +0.030 AND mean ΔmF1 ≥ +0.030 AND sign 3/3, each protocol judged independently, LoRA−CLIP.

**HateMM:**
- **val-selected:** mean Δacc **+0.0419** ≥ +0.030 ✔ · mean ΔmF1 **+0.0460** ≥ +0.030 ✔ · sign **3/3** ✔ → **PASS.**
  (Cushion over the bar: +0.0119 acc / +0.0160 mF1. Smallest per-seed gain +0.0326 acc; not a within-noise pass —
  the val-sel cushion is tighter than final's but 3/3 with tight spread, paired-t +7.8. NOT flagged MARGINAL:
  ~9× the B3 marginal-pass margin (+0.0013).)
- **final-epoch:** mean Δacc **+0.0573** ≥ +0.030 ✔ · mean ΔmF1 **+0.0682** ≥ +0.030 ✔ · sign **3/3** ✔ → **PASS.**
  (Comfortable: cushion +0.0273 acc / +0.0382 mF1.)

**MHC-EN (bundled B4 closure):**
- **val-selected:** mean Δacc **−0.0021** < +0.030 ✘ (and sign 2/3) → **FAIL.**
- **final-epoch:** mean Δacc **+0.0000** < +0.030 ✘ (and acc sign 1/3) → **FAIL.**
  (Expected-FAIL prior confirmed; the 22nd pre-registered negative / formal EN-LoRA-encoder closure.)

### KS-2 — FAMILY-COHERENCE HONESTY FLAG (NOT a performance kill; §3.3): trips iff LoRA < frozen-Qwen − 0.014.

- **final-epoch:** LoRA 0.8698/0.8618 vs frozen-Qwen 0.8682/0.8591 → LoRA − Qwen **+0.0015 acc / +0.0026 mF1**;
  **LoRA ≥ frozen-Qwen → flag NOT tripped; STRENGTHENS** the single-lever narrative (LoRA = best HateMM encoder on final-epoch).
- **val-selected:** LoRA 0.8620/0.8545 vs frozen-Qwen 0.8729/0.8648 → LoRA − Qwen −0.0108 acc; threshold
  frozen-Qwen − 0.014 = 0.8589; **LoRA 0.8620 ≥ 0.8589 → within seed band → flag NOT tripped** (LoRA sits marginally
  below frozen-Qwen but not beyond the 0.014 band).
- **Net:** the honesty flag does **not** trip on either protocol. The data are nonetheless consistent with the
  pre-declared F0.4 framing — LoRA ≈ frozen-Qwen (adds ≈0 over the frozen encoder), so the HateMM gain over CLIP is
  substantially the image-modality frozen-Qwen conversion (image-inherited), distinct from B3's text-borne
  LoRA-specific ZH gain. This nuance is **material to D7 and travels with any claim** (as pre-declared); it does
  **not** change the KS-1 pass/fail.

### KS-3 — REGIME SANITY / P9 CROSS-CHECK (§3.4): fires iff LoRA lands below the CLIP floor.

- HateMM LoRA (val-sel 0.8620, final 0.8698) is **far ABOVE** the CLIP floor (0.8202 / 0.8124). **KS-3 does NOT
  fire — no P9-echo.** The encoder-level regime converted on HateMM (opposite of P9's decision-level C3-knn −4.7),
  re-confirming the two-regime disambiguation (prereg F0.6).

---

## 4. Compliance clauses (prereg makes these binding)

| clause | finding | status |
|---|---|---|
| **Same-code pairing (§4.2)** | `run_rac.py` invocation block in `enc3seed_lora_hatemm.sbatch` vs `enc3seed.sbatch` = **BYTE-IDENTICAL** (diff). Only manipulated CLI vars: `--model` (CLIP→LoRA), `--group_name` (fresh). CONFIGS = 3 HateMM-LoRA + 3 MHC-LoRA; `GROUP_NAME=RAC_video_lora_hm`; `WARMUP=5`. | **PASS** |
| **Namespace-diff (§4.1b)** | LoRA-head vs 12850-CLIP Namespace differs in the 4 allowed fields (`model`, `exp_comment`, `group_name`, `output_path`) **plus 7 additional fields** (`lambda_tarc=0.0`, `oracle_probe=False`, `tarc_hn_mode='off'`, `tarc_mllm=''`, `tarc_multitarget='primary'`, `tarc_target_source='off'`, `tarc_vote_gamma=0.0`) that are **absent (None)** in the older 12850 code. **ALL computation-relevant hyperparameters (fusion_mode, topk, dropout, proj/map_dim, lr, bs, epochs, loss, triplet_margin, hybrid_loss, warmup, archive_feats=None, majority_voting, hard-neg config) are bit-identical.** | **PASS (non-material deviation, see note)** |
| **G-repro SFT-loss sanity (§4.1a)** | `logging/lora/HateMM/all_results.json`: eval_loss **0.1084**, train_loss 0.1048, epoch 2.97, train_runtime 10254.7 s. Loss finite (no NaN), decreased (from ~0.9 init), checkpoint written. eval_loss is slightly **below** the soft ~0.12–0.18 expectation band (MHC anchor 0.1620). | **PASS (benign — see note)** |
| **frozen-CLIP control re-paired from 12850, not re-run (§4.1c)** | Floors re-derived from the banked 12850 logs; no re-run. | **PASS** |
| **Single-encoder-draw (F0.2)** | All 3 HateMM head seeds read ONE LoRA encoder draw (`--model Qwen2.5-VL-7B-Instruct-LoRA_HF`); reported ±band is head-seed variance, not SFT-draw variance. Pre-declared; symmetric with the single-draw frozen-CLIP control. | **COMPLIANT (as declared)** |
| **Test-touch (§7 / F0.1)** | Only the six `…-LoRA_HF_seed*_13235.trainlog` LoRA head logs exist (HateMM + MHC-EN); no earlier LoRA-encoder head reads. **Exactly ONE budgeted LoRA-encoder test evaluation per dataset; zero test-touch before this verdict.** | **PASS** |

**§4.1b materiality note.** The 7 extra fields are TARC/oracle argparse defaults added to `run_rac.py` *after* the
12850 runs; the LoRA runs used a newer interpreter but every one of these knobs is set to its **OFF/inert** value.
`run_rac.py` proves the no-op: L411 comment "Every knob defaults to a full no-op"; L914 gates the entire TARC path
behind `tarc_target_source != "off"`; L1220–1225 "Inert when --tarc_target_source off (target_pack None)". These
branches are never entered here. Moreover the **already-accepted B3 verdict (job 13150) ran under the identical
TARC-off Namespace condition against the same 12850-era floors** — this newer-code-vs-12850 pairing is the
established, blessed protocol, not a fresh confound. The code-version confound §4.1b was written to retire is NOT
reintroduced: the variables that affect computation reduce to `--model` + derived-inert fields ONLY. Deviation is
**documentation-completeness (the literal "…ONLY" wording is exceeded by inert defaults), not a validity break.**

**§4.1a note.** A lower eval_loss (0.1084 < 0.12) indicates a *tighter* generative fit than the MHC anchor, not a
pathology; the binding gate conditions (finite / decreasing / checkpoint written) are all met and the downstream
head produces a healthy, non-degenerate signal. Benign; not a gate failure.

---

## 5. FINAL VERDICT BLOCK (prereg §8.3 fixed format; binding language verbatim)

```
HateMM:  final-epoch: PASS; val-selected: PASS.
MHC-EN:  final-epoch: FAIL; val-selected: FAIL.
```

- **HateMM KS-1: PASS on BOTH protocols** — mean paired LoRA−CLIP: val-sel +0.0419 acc / +0.0460 mF1 (3/3);
  final-ep +0.0573 acc / +0.0682 mF1 (3/3). Both conjuncts (acc AND mF1 ≥ +0.030) met, 3/3 sign both protocols.
- **HateMM KS-2 family flag: NOT tripped** (final LoRA ≥ frozen-Qwen → strengthens; val-sel within the 0.014 seed
  band). Pre-declared F0.4 image-inheritance framing is data-consistent (LoRA ≈ frozen-Qwen) and travels to D7.
- **HateMM KS-3: NOT fired** (LoRA far above CLIP floor; no P9-echo; encoder-level regime converts on HateMM).
- **MHC-EN KS-1: FAIL on BOTH protocols** — val-sel mean Δacc −0.0021, final mean Δacc +0.0000; the expected-FAIL
  formal closure (22nd pre-registered negative). EN-LoRA-encoder cell now formally closed.
- **Compliance:** hash-freeze MATCH (prereg + A/B/C); same-code byte-identical; single test-touch; single-draw as
  declared; one non-material Namespace-diff deviation (inert TARC-off defaults, provably no-op, B3-precedented) and
  one benign SFT-loss note (eval_loss 0.1084) — neither affects any KS ruling.

**Out of scope of this verdict (per prereg F0.3 / §9):** whether this performance PASS counts toward the goal's
"novel" clause — the **D7 (encoder-class novelty boundary) ruling is the user's**, NOT rendered here. This review
decides the **performance clause only**: encoder-level LoRA-Qwen **passes on HateMM** (both protocols) and **fails
on MHC-EN** (both protocols), against the frozen-CLIP floors, under the frozen pre-registration.
```
HATEMM_KS1 = PASS/PASS (val-sel / final-ep)   [primary]
MHC_EN_KS1 = FAIL/FAIL (val-sel / final-ep)   [bundled closure]
KS2_family_flag = NOT TRIPPED (final ≥ frozen-Qwen; val-sel within band)
KS3_P9_echo     = NOT FIRED (above CLIP floor)
D7_novelty      = DEFERRED TO USER (not decided here)
```
