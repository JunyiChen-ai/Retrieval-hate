---
type: experiment
node_id: exp:exp-fa-r4
title: "FA (round-4 wave-4 #1) — fusion/composition $0 probe: does the F44-cancelled Qwen-text gain convert on MHC-EN? KILL (5th better-signal/no-conversion datum; corrects F44 concat→align Hadamard erratum)."
idea_id: ""
status: CLOSED — KILL (non-binding executor label; B5 kill-switch fires; machinery valid, calibrated)
verdict: kill
confidence: n/a
date: "2026-07-17"
hardware: "ZERO GPU / ZERO test-touch / no Modal. CPU-only (OMP_NUM_THREADS=4), ~seconds. Raw-feature kNN proxy (no trained head, deterministic); bootstrap/permutation over dev items for CIs."
duration: "~seconds CPU."
provenance: "Raw-only $0 gate record refine-logs/FA_GATE_RECORD.md + FA_GATE_OUT.json (commit e0877c9; script scripts/analysis/fa_fusion_gate.py sha256 9e2fcbf39966cf85f6f5184eb29cf31cd6c577db1cfa5717ee70739a010b8b04, RNG=20260717). Banked caches data/CLIP_Embedding/{MHC,HateMM}/{train,dev_seen}_{openai_clip-vit-large-patch14-336_HF,Qwen2.5-VL-7B-Instruct_HF}.pt (N≈549/80 MHC-EN). kNN vote machinery reused from scripts/analysis/encoder_swap_geometry.py (cosine top-20 rank/sim-weighted signed vote, score>0). Align-Hadamard architecture claim re-verified vs src/model/classifier.py:110-122. Direction + F44 concat→align correction: refine-logs/WAVE4_CANDIDATES.md §0.3/§2 (commit 6032d32, F48). Findings: state/findings.jsonl F48 (correction+ideation), F50 (23rd pre-reg negative per that ledger — see PAPER_MASTER_TABLES tension #9). Non-binding executor label; NOT pushed."
added: 2026-07-17T00:00:00Z
tags: ["hateful-video", "MLLM-encoder", "modality-fusion", "cross-encoder", "Hadamard", "align-fusion", "concat-erratum", "F44-correction", "Pareto-vs-rotation", "oracle-threshold", "B5-killswitch", "selection-null", "AUC-unconvertible", "MHC-EN", "HateMM-positive-control", "$0-gate", "round-4", "F48", "F50", "KILL"]
---

# FA (round-4 wave-4 #1) — fusion/composition $0 probe (F50); F44 concat→align Hadamard erratum (F48)

> **STATUS: CLOSED — KILL.** $0 CPU gate (`FA_GATE_RECORD.md`, `e0877c9`), non-binding executor label =
> KILL on three independent locked rules; machinery valid, detectors calibrated (planted Pareto+rotation
> fire; HateMM positive control fires). Zero GPU, zero test-touch, no Modal. Pointer note; all numbers
> transcribed from the record. Paper integration: analysis §3.6 (FA = 5th and sharpest better-signal
> instance; the align/Hadamard erratum in the F44 mechanism leg); experiments §7 Table 6 + §2 erratum;
> master tables T5.3.

## Direction + the F44 correction (F48, `6032d32`)
The deployed head fuses via `fusion_mode='align'` = a parameter-free **element-wise Hadamard product of
two L2-normed projections** (`src/model/classifier.py:110-122`, `x = torch.mul(imghat, texthat)`) —
**not** the equal-weight **concat** F44's prose described. In align mode a linear `img_proj` cannot map
varying inputs to a constant, so the head **structurally cannot down-weight** the collapsed Qwen image
factor; F44's dismissal ("the head already has attenuation capacity and still failed") rests on a premise
the align head does not satisfy. That cell was therefore **unmeasured, not closed**. FA measures it:
does recovering the F44-cancelled Qwen-text gain via a different modality composition convert to
*accuracy* (Pareto) or only re-rank (rotation, B5-dead)?

## Result (transcribed from FA_GATE_RECORD.md)
- **K-FA-3 machinery valid:** MHC-EN concat proxy dev acc CLIP **0.7625**, Qwen **0.7500**,
  Qwen − CLIP = **−0.0125** vs F44's −0.012 (|Δ|=0.0005, sign −). **F44's numbers stand** via the
  sign-faithful concat-kNN proxy.
- **Calibration live:** planted pure-Pareto → detector flags Pareto; planted symmetric trade → flags
  rotation. **HateMM positive control** (real conversion): Δacc **+0.0467**, Δhate +0.1163, Δnon-hate
  +0.000 (clean Pareto), oracle-threshold **d_oracle = +0.0467 ≥ +0.03** — K-FA-2 fires on HateMM's real
  win, so the MHC kill is **calibrated**.
- **A1 within-Qwen reweight** (the fusion the align head cannot do): **pure rotation at every w** —
  w=0.5 = F44-exact **+0.040 hate / −0.036 non-hate**; w→0 (Qwen-text-only) AUC **0.8575** but Δacc
  **−0.025** (ranking edge accuracy can't absorb); A3a Qwen-align Hadamard Δacc **−0.0375** (worst,
  consistent with multiplicative corruption). **No w converts.**
- **A2 cross-encoder `CLIP-imĝ ⊕ Qwen-text̂`** (the CC object): AUC peaks at w=0.15 = **0.8982 — the
  highest measured anywhere on MHC-EN** (CLIP-img 0.734 ⊕ Qwen-text 0.851). The sole grid config meeting
  the K-FA-1 *point* bars (w=0.15: Δacc +0.050 / Δhate +0.120 / Δnon-hate +0.018, Pareto-**shaped**)
  survives **none** of the guards:
  - **K-FA-1 bootstrap CI-low:** Δacc +0.050, CI **[−0.0625, +0.150]** (crosses 0) → fails.
  - **K-FA-2 oracle-threshold** (both arms @ label-oracle τ): candidate 0.8250 − baseline 0.8000 =
    **+0.025 < +0.03** → **KILL fires** (ported B5 kill-switch).
  - **selection-null** (shuffle dev y, max-over-w Δacc, 1000×): observed +0.050 vs null p95 **+0.1375**
    (null-mean +0.076), **p = 0.766** — below the noise median → not survived.
- **Deployable** (train-LOO-selected w): A2 w=0.10 / A1 w=0.15 → dev Δacc +0.025, non-Pareto, within
  noise. No memorization pathology (raw-feature kNN, train-LOO 0.72–0.81 healthy).

**Verdict:** the modality-fusion door is measured and closed. FA is the **5th "better-signal /
no-conversion" datum** (after P3 · S2S F37 · W2-A F42 · router F47) and the sharpest — it lifts the
exact quantity B5 proved unconvertible (AUC, to a campaign-max 0.898) yet buys no accuracy at the
label-oracle operating point. CC not promoted; the relaxation-(f) D7-composition sub-ruling is moot.
