---
type: experiment
node_id: exp:exp-router-r4
title: "Router (round-4 line B) — per-item cross-channel routing (CLIP-arm vs Qwen-arm) over decision-level meta-features; $0 gate KILL. Includes MJ (wave-4 #2) modality-reliability router-input arithmetic NO-GO."
idea_id: ""
status: CLOSED — KILL (non-binding executor label; binding close = orchestrator spot-check, findings F47)
verdict: kill
confidence: n/a
date: "2026-07-17"
hardware: "ZERO GPU / ZERO Modal / ZERO test-touch. CPU-only (OMP_NUM_THREADS=4, faiss-cpu), ~3 min. Reloaded 12 banked enc3s e29 heads (router_ckpt_snapshot, 335 MB uncommitted; per-file sha256 in ROUTER_GATE_OUT.json)."
duration: "~3 min CPU."
provenance: "Raw-only $0 gate record refine-logs/ROUTER_GATE_RECORD.md + ROUTER_GATE_OUT.json (commit 30d0ee1; script scripts/analysis/cross_channel_router_gate.py sha256 d4adf545…, RNG=20260717). Banked caches data/CLIP_Embedding/{HateMM,MHC}/{train,dev_seen}_{openai_clip-vit-large-patch14-336_HF,Qwen2.5-VL-7B-Instruct_HF}.pt; 12 e29 heads from enc3s job-12850 RAC_video_archive_seeds (align fusion, topk=20, arithmetic vote). Vote machinery = src/utils/metrics.py compute_metrics_retrieval + src/model/classifier.py align head. MJ arithmetic NO-GO = refine-logs/MJ_FORENSIC_RECON.md (commit d57d05d), banked archive modality_cues provenance d0f9e7b. Findings: state/findings.jsonl F47 (22nd pre-reg negative per that ledger — see PAPER_MASTER_TABLES tension #9 for the ordinal discrepancy), F49 (MJ). Non-binding executor label; NOT pushed."
added: 2026-07-17T00:00:00Z
tags: ["hateful-video", "MLLM-encoder", "per-item-routing", "cross-channel", "channel-selection", "meta-features", "HistGradientBoosting", "oracle-headroom", "train-memorization", "dev-CV-ceiling", "permutation-null", "MHC-EN", "HateMM", "G0-cond", "$0-gate", "round-4", "F47", "MJ", "F49", "modality-locus", "alignment-ceiling", "KILL"]
---

# Router (round-4 line B) — per-item cross-channel routing $0 gate (F47) + MJ modality-reliability router-input NO-GO (F49)

> **STATUS: CLOSED — KILL.** $0 CPU gate (`ROUTER_GATE_RECORD.md`, `30d0ee1`), non-binding executor
> label = KILL; the binding close is the orchestrator spot-check (CTF/APX kill-side precedent —
> `ROUTER_GATE_OUT.json` numbers verbatim vs the record). Zero GPU, zero test-touch, no Modal. This note
> is a pointer to the committed record; all numbers below are transcribed from it. Paper integration:
> analysis §3.6 (router = 4th better-signal instance) and §3.8 (its own three-source closure);
> experiments §7 Table 6; master tables T5.3.

## Direction
Per video, predict which prediction **channel** to trust — the CLIP-encoder RGCL arm vs the
Qwen2.5-VL-7B-encoder RGCL arm — from decision-level meta-features (per-item kNN vote margins, neighbour
label purity, rank/similarity components, per-modality sub-votes `vimg`/`vtxt`, channel-disagreement
indicators), to convert the F44 MHC-EN encoder **rotation** into a Pareto gain. Non-isomorphic to K9
feature-space zeros / B5 global-threshold / P1 / P2 (verified in record §0).

## Result (transcribed from ROUTER_GATE_RECORD.md)
- **Machinery valid:** 12/12 regenerated e29 dev accs bit-exact vs 12850 deployed anchors; **K-R2**
  label-oracle calibration accZA(MHC) = **1.000** (machinery VALID, not MACHINERY_INVALID).
- **Oracle headroom (ceiling, ruled first):** perfect per-item router gains **+0.1083 MHC-EN**
  (s0 +0.1125 / s1 +0.1250 / s2 +0.0875) and **+0.0498 HateMM** — 4th oracle-exists instance.
- **PRIMARY train→dev GBM router:** **+0.0000** all seeds, both datasets (routed acc = best-single acc
  exactly; boot CI [0.0, 0.0]). **K-R1 KILL fires.**
- **Mechanistic cause (load-bearing):** the CLIP head **memorises train** — LOO train acc **0.998** vs
  Qwen **0.800** — so on the *train* disagreement subset "Qwen correct" = **0/109, 0/102, 0/92**
  (degenerate, always-CLIP), the inverse of the *dev* base rate (0.55 / 0.565 / 0.65). The deployable
  train→dev router has no dev-transferable supervision.
- **Dev-CV realizable ceiling (most favorable, peeks dev labels):** MHC-EN GBM **−0.0458** (CI
  [−0.0875, 0]), linear −0.0333 — both NEGATIVE. **K-R3 not survived:** perm-null p95 +0.0042,
  observed −0.0458, **p = 0.97**.
- **HateMM sanity:** routed − best = +0.0000 (OK; router learns ~always-Qwen).

**Closure:** per-item channel-selection dead at (a) unsupervised/feature-conditional (K9 zeros
F42/F39/F43), (b) train-supervised (degenerate memorised target), (c) dev-supervised (negative at the
CV ceiling). Confirms F44 "no coherent subgroup" at the per-item predictability level.

## MJ (wave-4 #2) — modality-reliability router input, arithmetic NO-GO (F49, `d57d05d`)
The one door F47 left ajar ("a genuinely NEW information source not derivable from banked
features/votes"). **Dead pre-GPU on arithmetic:** on the 80-item MHC-EN dev split (disagreement sizes
20/23/20, always-Qwen prior **0.588**, oracle **+0.108**), clearing +0.020 needs which-arm-wins accuracy
**q ≥ 0.663**; but the modality-locus **alignment ceiling** `a` is **≤ 0.588** generous / ≈ 0.50 (F44
"net −1") / ≈ 0.41 (F47 dev-CV) — a **perfect judge (j=1 ⇒ q=a)** gives gain ≈ +0.0002 to −0.046 and
cannot reach +0.020 (the model reproduces F47's realizable −0.046 to the digit at a=0.413). Structural
twin of P2 (comparability ⊥ vote-correctness). **Cost correction:** the judgment is **already banked**
(`data/Archive/MHC/*_Qwen2.5-VL-7B-Instruct_archive.jsonl` `modality_cues`, provenance `d0f9e7b`, dev
80/80 parse_ok) — no Modal generation owed (and frames are hard-banned from Modal anyway). The
subordinate $0 GO-IF closure probe was **declined** per the ceiling-below-bar precedent (A-line /
G0-cond: the router gate F47 ran only because its oracle *exceeded* the bar). **Ban scope:** per-item
router inputs whose alignment with which-arm-wins ≤ q_required are dead **on arithmetic**; any future
carve-out candidate must first demonstrate alignment > 0.663 from banked evidence.
