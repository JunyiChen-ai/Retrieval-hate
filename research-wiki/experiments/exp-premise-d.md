---
type: experiment
node_id: exp:exp-premise-d
title: "Premise-(d) gate (round-4 closing) — CLIP-img (+) LoRA-EN-Qwen-text composition: does the F50 adaptation carve-out convert MHC-EN? KILL (6th better-signal/no-conversion datum; even ADAPTED text does not convert EN)."
idea_id: ""
status: CLOSED — KILL (non-binding executor label; B5/K-D-1 kill-switch fires; machinery valid, calibrated)
verdict: kill
confidence: n/a
date: "2026-07-18"
hardware: "ZERO GPU / ZERO Modal / ZERO test-touch. CPU-only (OMP_NUM_THREADS=4), ~seconds. Raw-feature kNN proxy (no trained head, deterministic RNG=20260717); bootstrap/permutation over dev items for CIs. Banked train + dev_seen caches only; test_seen never opened."
duration: "~seconds CPU."
provenance: "Design source: refine-logs/TIE_BRANCH_RECON.md §2 (commit 6b9985a, the TIE-branch recon LEAD candidate = premise-(d)). Raw-only $0 gate record refine-logs/PREMISE_D_GATE_RECORD.md + PREMISE_D_GATE_OUT.json (commit 6e6061b; script scripts/analysis/premise_d_gate.py sha256 909f9d1a...db9a931d). FA machinery reused verbatim from scripts/analysis/fa_fusion_gate.py (FA sha256 9e2fcbf3...); reproduction anchor refine-logs/FA_GATE_OUT.json (23e5bfda...). Judged block = LoRA-EN Qwen text cache data/CLIP_Embedding/MHC/{train,dev_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt (mtime 2026-07-02, stable B4-era; adapter = EN own-train-split 549 records, r16/a32, per B4_FORENSIC_RECON.md:71-93). Non-binding executor label; NOT pushed. Binding close = orchestrator's."
added: 2026-07-18T00:00:00Z
tags: ["hateful-video", "MLLM-encoder", "modality-fusion", "cross-encoder", "LoRA-text", "adaptation-carveout", "Hadamard", "align-fusion", "Pareto-vs-rotation", "oracle-threshold", "B5-killswitch", "selection-null", "AUC-unconvertible", "MHC-EN", "HateMM-positive-control", "$0-gate", "round-4", "F55", "KILL", "6th-no-conversion"]
---

# Premise-(d) gate (round-4 closing) — CLIP-img (+) LoRA-EN-Qwen-text composition (F55)

> **STATUS: CLOSED — KILL.** $0 CPU gate (`PREMISE_D_GATE_RECORD.md`, `6e6061b`), non-binding executor label =
> KILL: the pre-declared K-D-1 (B5-ported) oracle kill-switch fires; machinery valid, detectors calibrated
> (planted Pareto+rotation fire; HateMM positive control +0.0467 passes; FA-A2 reproduced bit-exact). Zero GPU,
> zero test-touch, no Modal. Pointer note; all numbers transcribed from the record. Paper integration: analysis
> §3.6 (premise-(d) = 6th better-signal/no-conversion instance, completes the F50 EN-composition story);
> experiments §7 (round-4 closing prose row); master tables T5.4.

## Direction — the F50 adaptation carve-out
FA (F50) closed EN's *frozen* cross-encoder composition (CLIP-img 0.734 (+) frozen-Qwen-text 0.851 -> composite
best-ever EN AUC **0.898**, `d_oracle +0.025 < +0.03` KILL) but its own ban carved out one untested cell:
*"do not re-propose fixed compositions ... over banked FROZEN features; conversion requires ADAPTATION (F45)."*
Premise-(d) swaps the **frozen** Qwen-text block for the **LoRA-EN-adapted** Qwen-text block (the B4-arm cache) —
the adaptation the ban itself names — keeps the healthy CLIP image block, and re-runs the FA oracle machinery.
**Question: does the LoRA text swap close the +0.005 oracle gap and convert as a PARETO move, or not?** This is
the single genuinely-uncovered, $0, goal-relevant cell the TIE-branch recon (F54) identified.

## Result (transcribed from PREMISE_D_GATE_RECORD.md; MHC-EN primary, n_train=549, n_dev=80/25 hate)
- **K-D-0b machinery VALID (bit-exact):** the frozen-text control arm (A2F) reproduces FA-A2 across all 21 grid
  configs, **max absolute difference 0.000000**; peak dev AUC **0.8982** (FA "0.898 best-ever EN"), ceiling
  `d_oracle +0.0250` (FA +0.0250). Substrate check: MHC-EN concat proxy Qwen-CLIP dev acc **-0.0125** vs F44's
  -0.012 (sign-faithful). Calibration live (planted Pareto/rotation both fire). **HateMM positive control:**
  Δacc **+0.0467**, Δhate +0.1163, Δnon-hate +0.000 (clean Pareto), `d_oracle +0.0467 >= +0.03` -> the K-D-1
  switch fires on MHC-EN (+0.025) and NOT on HateMM's genuine win (+0.0467) => the kill is **calibrated**.
- **A2L judged arm (CLIP-img (+) LoRA-EN-Qwen-text), ceiling w=0.20:** dev acc **0.8125** (Δacc +0.0500),
  Δhate-rec +0.2800, Δnon-hate-rec **-0.0545**, dev AUC **0.8698**, oracle acc 0.8250, **`d_oracle +0.0250`**.
  **MAX `d_oracle` anywhere on the entire A2L grid = +0.0250 < +0.03.** Baseline A0 CLIP-concat dev acc 0.7625
  (oracle 0.800).
- **The two decisive comparisons — the swap HURTS, it does not help:**

  | quantity | FROZEN text (A2F = FA-A2) | LoRA-EN text (A2L) | swap effect |
  |---|---|---|---|
  | peak dev AUC | 0.8982 | 0.8698 | **-0.0284 (WORSE)** |
  | max `d_oracle` on grid | +0.0250 | +0.0250 | **+0.0000 (gap NOT closed)** |
  | any Pareto point-bar config? | one (w0.15) | **NONE** | worse shape |

  The adaptation the ban names as the conversion mechanism (F45: ZH text AUC 0.847 -> 0.925) does **not** occur
  on EN: the LoRA-EN text stream *degrades* the composite ranking (peak AUC -0.0284) and leaves the binding
  oracle threshold pinned at +0.025, identical to frozen. Mirror image of ZH.
- **Inferential guards on the candidate (A2L w=0.20) — all fail:** not Pareto (Δnon-hate -0.0545 << -0.01);
  bootstrap CI-low **[-0.0503, +0.1625]** crosses 0; K-D-1 oracle edge **+0.0250 < +0.03** (KILL fires);
  selection-null observed +0.050 vs null p95 +0.1375, null-mean +0.0745, **p=0.7532** (not survived — +0.05 sits
  at the 25th percentile of the noise floor). No memorization pathology (raw-feature kNN, train-LOO healthy).

## Verdict / paper reading
The premise-(d) F50 carve-out — **CLIP-img (+) LoRA-EN-Qwen-text** — is **measured and closed on MHC-EN.** The
frozen->LoRA text swap the ban itself names as the conversion mechanism does **not** convert: the +0.005 oracle
gap does not close (max `d_oracle` +0.0250, identical to frozen, below +0.03; K-D-1 B5 kill-switch fires), and
the LoRA adaptation actively **degrades** EN (peak AUC 0.8982 -> 0.8698). This is the **sixth "better-signal /
no-conversion" datum** on EN (after P3, S2S-F37, W2-A-F42, router-F47, FA-A2-F50) and completes the F50 story:
with premise-(d) dead, EN is closed at the **frozen (F50), collapsed-adapted (B4/F53), AND healthy-image (+)
adapted-text (premise-(d))** composition levels simultaneously. No PASS => the moot relaxation-(f)
D7-composition sub-ruling stays moot; no ceremony/headline/family claim is owed. Per recon §4, the round-3
terminus is complete for the adaptation family and the live decision reverts to the D7 ruling on generic LoRA
(a user ruling, zero further GPU). **Binding close remains the orchestrator's.**

## Connections
- carve-out-of -> `exp:exp-fa-r4` (FA / F50 closed EN's frozen composition; premise-(d) closes the "conversion
  requires adaptation" cell it left ajar — non-isomorphic: A2F frozen text vs A2L LoRA text)
- identified-by -> `refine-logs/TIE_BRANCH_RECON.md` (F54, TIE-branch recon LEAD candidate; premise-correction
  that "LoRA moves text only" is empirical not architectural)
- non-isomorphic-to -> `exp:exp-lora-hatemm` §4b (B4/F53: deployed LoRA pipeline = collapsed LoRA-img (+)
  LoRA-text, never the healthy CLIP-img); `exp:exp-cand2-curriculum` (Qwen-only curriculum, not a composition)
- mirror-image-of -> `refine-logs/B3_ZH_LORA_DECOMPOSITION.md` (F45: LoRA LIFTS ZH text 0.847->0.925; here it
  DEGRADES EN text composite 0.8982->0.8698)
- machinery-reused-from -> `scripts/analysis/fa_fusion_gate.py` (FA); reproduction anchor `FA_GATE_OUT.json`
- law -> `DRAFT_analysis_chapter.md` §3.6 (structural law I, 6th instance) + §3.9 (EN closed at every
  composition level in the adaptation phase diagram)
