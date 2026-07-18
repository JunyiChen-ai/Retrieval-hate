---
type: experiment
node_id: exp:exp-lora-hatemm
title: "LoRA-HateMM (round-4 line-A) — encoder-level LoRA-Qwen vs frozen-CLIP on HateMM: 3-seed paired encoder-swap test, dual protocol, archive OFF (+ bundled B4-EN formal closure)"
idea_id: "idea:lora-mllm-encoder-lever"
status: CLOSED
verdict: partial
confidence: high
date: "2026-07-18"
hardware: "1x A100 (SLURM); fresh LoRA-SFT ~3.1 h (job 13233) + extraction ~0.4 h (13234) + 3-seed head ~2 min (13235); one budgeted HateMM + one MHC-EN test evaluation"
duration: "job chain 13233 (lora_sft) -> 13234 (gen_embed_lora) -> 13235 (enc3seed head; 3 HateMM-LoRA + 3 MHC-EN-LoRA rows)"
novelty_clause: "PENDING USER D7 RULING. This cell measures the PERFORMANCE clause of the goal ONLY (+0.03 acc AND +0.03 macro-F1, 3/3 sign, per protocol). The lever is encoder-level LoRA-SFT adaptation of the Qwen2.5-VL-7B encoder — an Axis-B / RA-HMD-family technique the project classifies as an encoder-class lever (D7-encoder-class-novelty-dead by ruling). Whether an encoder-level LoRA performance pass counts toward the goal's 'novel' clause is the user's D7 ruling, NOT decided here. The cell is not folded into any main table (PAPER_MASTER_TABLES.md PUR-banner)."
provenance: "CLOSED under full single-submit ceremony. Recon (GO) refine-logs/LORA_HATEMM_FORENSIC_RECON.md (edeaedc); prereg refine-logs/LORA_HATEMM_PREREG.md (3ebd880); independent 0-context prereg review refine-logs/LORA_HATEMM_PREREG_REVIEW.md (2e41332, APPROVED-WITH-NOTES); hash-freeze refine-logs/LORA_HATEMM_FREEZE.md (8de0991); single-submit record refine-logs/LORA_HATEMM_SUBMIT_RECORD.md (56a732a, job chain 13233->13234->13235); independent 0-context verdict refine-logs/LORA_HATEMM_VERDICT_REVIEW.md (6b8f634). Floors re-parsed from the banked 12850 frozen-CLIP/Qwen enc3s trainlogs (same parser as exp-encoder-3seed / exp-lora-zh-b3). frozen-CLIP floor per ERRATUM 66012e9 (0.8279/0.8172, not the withdrawn 0.8732)."
added: 2026-07-18T00:00:00Z
tags: ["hateful-video", "MLLM-encoder", "LoRA-SFT", "encoder-adaptation", "RA-HMD", "frozen-CLIP", "multi-seed", "paired-test", "HateMM", "MHC-EN", "dual-protocol", "pre-registered", "CLOSED", "line-A", "F53", "novelty-pending", "D7"]
---

# LoRA-HateMM (round-4 line-A) — encoder-level LoRA-Qwen vs frozen-CLIP on HateMM

> **STATUS: `CLOSED` 2026-07-18 — VERDICT (`refine-logs/LORA_HATEMM_VERDICT_REVIEW.md`, commit
> `6b8f634`; job chain 13233→13234→13235, independent 0-context reviewer, hash-verified vs the
> frozen prereg `3ebd880`/`8de0991`):**
>
> ```
> HateMM:  final-epoch: PASS; val-selected: PASS.
> MHC-EN:  final-epoch: FAIL; val-selected: FAIL.
> ```
>
> **Novelty = PENDING USER D7 RULING** (encoder-class lever). Not folded into any main table.

**verdict:** `partial` (HateMM performance PASS both protocols; MHC-EN closed FAIL both; novelty pending D7)
· **confidence:** `high`

## 0. What this cell is (and is NOT) — read first

The **round-4 line-A** measurement completes the encoder-level LoRA performance matrix. It replaces the
frozen-CLIP video/text front-end with an **encoder-level LoRA-SFT-adapted Qwen2.5-VL-7B encoder** on
HateMM — features fed to the **unchanged** archive-OFF RGCL align-fusion head + top-20 kNN (the
`enc3s`/12850 protocol) — paired 3 head-seeds vs the banked frozen-CLIP floor, dual-protocol. Bundled
arm: the **B4-EN** LoRA-encoder cell (`dataset=MHC`), an expected-FAIL formal closure.

- **PERFORMANCE clause only.** Whether the pass counts toward the goal's "novel" clause is the user's
  **D7 ruling**, not decided here (prereg F0.3).
- **Two regimes, not one (why P9 does not pre-kill).** This is the **encoder-level** regime (`stage: sft`
  generative word-label SFT, r16/α32, features → a freshly-trained RGCL head). P9's banked HateMM
  negative (C3-knn −4.7 below floor) is the **decision-level** regime (`sft_classifier`, r128/α256,
  raw-kNN read-out, no trained head). Proven non-isomorphic by opposite ZH behaviour (encoder-level ZH
  kNN **+0.031** B3 vs decision-level ZH C3-knn **−2.2** P9). P9's datum is a tempering yellow flag
  (KS-3), not a pre-kill (prereg F0.6).
- **Single-encoder-draw limitation (pre-declared, F0.2).** All 3 head-seeds read ONE HateMM LoRA-SFT
  encoder draw; the ±band is **head-seed variance, NOT LoRA-SFT-draw variance** — symmetric with the
  single-draw frozen-CLIP control, so this is a legitimate head-level paired test, not an encoder-draw
  paired test.

## 1. Design (pre-registered, frozen)

- **Stage 1 — LoRA-SFT** (`sbatch scripts/slurm/lora_sft.sbatch HateMM`, job 13233): base
  Qwen2.5-VL-7B-Instruct, `stage: sft` (pure CAUSAL_LM, word-label hateful/normal), `lora_rank 16`,
  `lora_alpha 32`, dropout 0.0, target q/k/v/o + gate/up/down proj, **`freeze_vision_tower: true` +
  `freeze_multi_modal_projector: true`** (only the LLM backbone moves — the F0.4 mechanism basis), lr
  1e-4, 3 epochs, eff-bs 8, bf16, 8-frame. Train on **HateMM own train split only** (743 records:
  297 hateful / 446 normal). SFT-loss sanity: eval_loss **0.1084** (finite, decreasing; benign, slightly
  tighter than the MHC anchor 0.1620).
- **Stage 2 — extraction** (`gen_embed_lora.sbatch HateMM logging/lora/HateMM`, job 13234): merge_and_unload
  the adapter, extract 8-frame dual-stream 3584-d img/text embeddings → distinct `..._LoRA_HF.pt` cache.
- **Stage 3 — 3-seed RGCL head** (`enc3seed_lora_hatemm.sbatch`, job 13235): 6 head-only runs (HateMM-LoRA
  seeds 0/1/2 + MHC-EN-LoRA seeds 0/1/2), `--model Qwen2.5-VL-7B-Instruct-LoRA_HF`, archive OFF,
  topk 20, lr 1e-4, 30 epochs, bz 64, proj=map 1024, hybrid triplet+BCE, warmup 5. The `run_rac.py`
  argv is **byte-identical** to the 12850 CLIP control except `--model` + a fresh `--group_name`.
- **Control (NOT re-run):** banked frozen-CLIP 12850 enc3s logs; frozen-Qwen 12850 is the KS-2 secondary
  floor.

## 2. Protocols (both reported, judged independently — NO protocol/metric shopping)

- **(A) val-selected:** epoch ≥ warmup 5 with max `Val_Retrieval` acc (roc tie-break) → that epoch's Test acc / macro-F1.
- **(B) final-epoch:** Test acc / macro-F1 at the last trained epoch (29).

## 3. Comparison floors (re-parsed from the banked 12850 trainlogs; ERRATUM-corrected)

| HateMM floor | protocol | s0 acc/F1 | s1 acc/F1 | s2 acc/F1 | 3-seed mean acc / F1 |
|---|---|---|---|---|---|
| frozen-CLIP (PRIMARY — KS-1 pairs vs this) | val-sel | 0.8279/0.8172 | 0.8279/0.8163 | 0.8047/0.7920 | **0.8202 / 0.8085** |
| frozen-CLIP (PRIMARY) | final-ep | 0.8186/0.7997 | 0.8047/0.7822 | 0.8140/0.7988 | **0.8124 / 0.7936** |
| frozen-Qwen (SECONDARY — KS-2 pairs vs this) | val-sel | 0.8698/0.8606 | 0.8651/0.8586 | 0.8837/0.8753 | **0.8729 / 0.8648** |
| frozen-Qwen (SECONDARY) | final-ep | 0.8605/0.8507 | 0.8605/0.8514 | 0.8837/0.8753 | **0.8682 / 0.8591** |

## 4. Results — HateMM, LoRA-Qwen vs frozen-CLIP (job 13235; per-seed, both protocols)

| seed | protocol | LoRA acc/F1 (sel ep) | CLIP acc/F1 | Δacc | ΔF1 |
|---|---|---|---|---|---|
| 0 | val-sel | 0.8605/0.8521 (e19) | 0.8279/0.8172 | +0.0326 | +0.0349 |
| 1 | val-sel | 0.8698/0.8620 (e14) | 0.8279/0.8163 | +0.0419 | +0.0457 |
| 2 | val-sel | 0.8558/0.8495 (e22) | 0.8047/0.7920 | +0.0511 | +0.0575 |
| **mean** | **val-sel** | **0.8620/0.8545** | **0.8202/0.8085** | **+0.0419** | **+0.0460** |
| 0 | final-ep | 0.8651/0.8580 (e29) | 0.8186/0.7997 | +0.0465 | +0.0583 |
| 1 | final-ep | 0.8744/0.8660 (e29) | 0.8047/0.7822 | +0.0697 | +0.0838 |
| 2 | final-ep | 0.8698/0.8613 (e29) | 0.8140/0.7988 | +0.0558 | +0.0625 |
| **mean** | **final-ep** | **0.8698/0.8618** | **0.8124/0.7936** | **+0.0573** | **+0.0682** |

Sign consistency: **val-sel 3/3 positive (acc and mF1); final-ep 3/3 positive (acc and mF1).**
Effect-size descriptors only (n=3, no significance claim): val-sel paired-t acc +7.84 / mF1 +7.05;
final-ep acc +8.51 / mF1 +8.64.

### 4b. Bundled B4-EN closure — MHC-EN, LoRA-Qwen vs frozen-CLIP (expected-FAIL formal close, 22nd negative)

| seed | protocol | LoRA acc/F1 (sel ep) | CLIP acc/F1 | Δacc | ΔF1 |
|---|---|---|---|---|---|
| 0 | val-sel | 0.7516/0.6916 (e26) | 0.7826/0.7113 | −0.0310 | −0.0197 |
| 1 | val-sel | 0.7391/0.6920 (e5) | 0.7329/0.6034 | +0.0062 | +0.0886 |
| 2 | val-sel | 0.7888/0.7506 (e29) | 0.7702/0.6997 | +0.0186 | +0.0509 |
| **mean** | **val-sel** | **0.7598/0.7114** | **0.7619/0.6715** | **−0.0021** | **+0.0399** |
| 0 | final-ep | 0.7702/0.7302 (e29) | 0.7640/0.7145 | +0.0062 | +0.0157 |
| 1 | final-ep | 0.7764/0.7360 (e29) | 0.7826/0.7159 | −0.0062 | +0.0201 |
| 2 | final-ep | 0.7888/0.7506 (e29) | 0.7888/0.7303 | +0.0000 | +0.0203 |
| **mean** | **final-ep** | **0.7785/0.7389** | **0.7785/0.7202** | **+0.0000** | **+0.0187** |

The seed-0 anchor reproduces the pre-GPU forensic value exactly (val-sel −0.0310 acc; final +0.0062 acc),
confirming the honest expected-FAIL prior. The EN LoRA-encoder cell is **formally closed**.

## 5. Kill-switch rulings (prereg wording applied verbatim)

- **KS-1 (primary performance conjunct):** mean Δacc ≥ +0.030 AND mean ΔmF1 ≥ +0.030 AND sign 3/3, each
  protocol independently.
  - **HateMM val-selected: PASS** (+0.0419 / +0.0460, 3/3; cushion +0.0119 acc / +0.0160 mF1 — **NOT
    marginal**, ≈ 9× the B3 +0.0013 margin).
  - **HateMM final-epoch: PASS** (+0.0573 / +0.0682, 3/3; cushion +0.0273 / +0.0382).
  - **MHC-EN both protocols: FAIL** (val-sel −0.0021 acc; final +0.0000 acc).
- **KS-2 (family-coherence honesty flag; trips iff LoRA < frozen-Qwen − 0.014): NOT tripped.** final-ep
  LoRA 0.8698/0.8618 ≥ frozen-Qwen 0.8682/0.8591 (LoRA − Qwen **+0.0015 acc / +0.0026 mF1** →
  STRENGTHENS the single-lever narrative); val-sel LoRA 0.8620 ≥ frozen-Qwen 0.8729 − 0.014 = 0.8589
  (within the seed band). **LoRA ≈ frozen-Qwen** (adds ≈ 0 over the frozen encoder), so the HateMM gain
  over CLIP is substantially the frozen-Qwen conversion inherited — but the F58 per-stream decomposition
  (`refine-logs/HATEMM_LORA_STREAM_DECOMP.md`, `51eb95b`) corrects the pre-declared F0.4
  "image-inheritance" gloss: the decisive single stream on HateMM is **text** (text-only kNN AUC ≥
  image-only for all three encoders, both footings), and the pass is **text-carried on a swap-neutral
  image base and frozen-swap-sufficient** — the frozen swap already converts HateMM's text signal to a
  Pareto (frozen−CLIP +0.0558 acc), so LoRA's further text-sharpening (train-LOO 0.888→0.920) adds ≈ 0.
  It is inherited (LoRA ≈ frozen-Qwen), distinct from B3's text-borne **LoRA-specific** ZH gain (where
  frozen-Qwen fails, −0.0112). This nuance travels to D7 (does not change KS-1).
- **KS-3 (P9 regime echo; fires iff LoRA below CLIP floor): NOT fired.** LoRA far above the CLIP floor →
  the encoder-level regime converts on HateMM (opposite P9's decision-level C3-knn −4.7), re-confirming
  the two-regime disambiguation.

## 6. Compliance (prereg §4, binding)

- **Hash-freeze:** prereg + configs A/B/C matched byte-for-byte at submit time (re-verified after the
  repo HEAD advanced past `8de0991`).
- **Same-code pairing:** the `run_rac.py` argv block is byte-identical to the 12850 CLIP control; only
  `--model` (CLIP→LoRA) + a fresh `--group_name` are manipulated.
- **Single test-touch:** exactly one budgeted LoRA-encoder test evaluation per dataset (HateMM + MHC-EN);
  zero test-touch before the verdict.
- **Non-material Namespace-diff deviation (flagged honestly):** the LoRA head ran under a newer
  `run_rac.py` carrying 7 additional TARC/oracle argparse fields absent in the 12850 code, all at their
  inert OFF values (provably no-op via `run_rac.py` L411/L914/L1220–1225; the identical condition under
  which the already-accepted B3 verdict was rendered). Every computation-relevant hyperparameter is
  bit-identical. Documentation-completeness note only, not a validity break.

## 7. What the pass means for the goal (D7 boundary — this cell does NOT decide)

Performance-conjunct ledger, with the protocol qualifier:
- **Under the final-epoch protocol, one lever — encoder-level LoRA — clears +0.03/+0.03 on two datasets**
  (HateMM +0.0573/+0.0682 solid; MHC-ZH B3 +0.0313/+0.0453 marginal).
- **Under the val-selected protocol, the same lever clears HateMM only** (ZH val-selected FAILs; the
  78-dev selection tax).

This is the first single encoder lever to clear ≥ 2 datasets — but it is **one lever with two
mechanisms**, not one mechanism: ZH's gain is text-borne and LoRA-specific (frozen-Qwen −0.0112 on ZH),
HateMM's is text-carried on a swap-neutral image base and inherited from the frozen swap (KS-2 not
tripped, LoRA ≈ frozen-Qwen; the frozen swap already converts HateMM's text signal, so LoRA's further
text-sharpening adds ≈ 0 — F58, `refine-logs/HATEMM_LORA_STREAM_DECOMP.md`, `51eb95b`). EN stays closed
(label-limited, image-collapsed,
F44). Whether an encoder-class adaptation lever satisfies the goal's "novel" clause is the pending user
**D7 ruling**; this cell is an encoder-class lever regardless of outcome and is not folded into any main
table.

## 8. Connections

- completes → `exp:exp-lora-zh-b3` (ZH encoder-level LoRA marginal pass) and `exp:exp-lora-sft-encoder` (the EN/ZH single-config precedent) — this cell adds the HateMM leg + the formal EN 3-seed closure
- contrasts-with → `exp:exp-encoder-3seed` (HateMM frozen-Qwen swap PASS both protocols; here LoRA ≈ frozen-Qwen on HateMM, KS-2 not tripped)
- distinct-regime-from → `EXP_p9_lmm_rgcl_video` (decision-level LoRA-SFT; C3-knn HateMM −4.7 below floor — non-isomorphic to this encoder-level cell)
- reuses-control-arm-of → `exp:exp-encoder-3seed` (12850 frozen-CLIP seeds 0/1/2 = the paired floor, not re-run)
- mechanism → `refine-logs/ENCODER_SWAP_DIAGNOSIS.md` (F44, HateMM Pareto conversion on a swap-neutral image base) + `refine-logs/B3_ZH_LORA_DECOMPOSITION.md` (F45, text-borne ZH LoRA gain) + `refine-logs/HATEMM_LORA_STREAM_DECOMP.md` (F58, HateMM pass measured text-carried / frozen-swap-sufficient / LoRA-inherited); analysed as the adaptation law in `DRAFT_analysis_chapter.md` §3.9
- scoped-by → `refine-logs/LORA_HATEMM_FORENSIC_RECON.md` (GO recon, regime disambiguation), `refine-logs/LORA_HATEMM_PREREG.md` (frozen prereg), `refine-logs/LORA_HATEMM_VERDICT_REVIEW.md` (binding verdict)
- novelty-clause → PENDING USER D7 RULING (`PAPER_MASTER_TABLES.md` PUR-3 / PUR-banner)

## 9. Revision history

| rev | date | status | change | authority |
|---|---|---|---|---|
| r0 | 2026-07-18 | CLOSED | Initial per-experiment note, authored at paper-integration time from the committed ceremony chain (recon `edeaedc` → prereg `3ebd880` → review `2e41332` → freeze `8de0991` → submit `56a732a` → verdict `6b8f634`, job chain 13233→13234→13235). HateMM PASS both protocols (val-sel +0.0419/+0.0460, final +0.0573/+0.0682, 3/3); MHC-EN FAIL both (formal B4 closure, 22nd negative); KS-2 not tripped (LoRA ≥ frozen-Qwen), KS-3 not fired; one non-material inert-defaults Namespace deviation. Novelty = PENDING USER D7 RULING; not folded into any main table. | paper integrator |
