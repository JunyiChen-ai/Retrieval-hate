# LITSWEEP2 — 2026-Fresh Multimodal Sweep (Round-2, Agent #3)

**Date:** 2026-07-25 · **Scope:** mid-2025 → Jul-2026, emphasis on last 6 months · **Mode:** CPU-only, WebSearch/WebFetch · **Goal being served:** substantial gains (≥+3 acc on ≥2 datasets); every candidate isomorphism-checked against `directions_tried.json` (P1–P11, TARC, W2-*, S2S, CTF, APX, GIR, router/MJ/FA, LP, SWA, LAUD, vision-LoRA, ISR, frame16, gradnorm, readout, MCR, bidir) and findings F44–F72.

**One-line bottom line:** No 2026 paper beats us on HateMM/MHC with a transplantable in-box lever; the single highest-prior *untested* move in the whole field is a **new-generation 7B-class encoder swap** (best pick = **Molmo2-8B**, runner-up Qwen3-VL-8B), but the F44 mechanism caps its realistic payoff at *HateMM-only* — it does **not** clear the ≥2-dataset goal, because MHC-EN/ZH errors are label-limited, not representation-limited.

---

## HUNT 1 — 2026 harmful-video papers on HateMM / MultiHateClip

| Paper | Venue/ID | Best number | Lever | Transplantable? |
|---|---|---|---|---|
| **MM-HSD** | arXiv 2508.20546 (Idiap) | HateMM **0.878 acc / 0.874 M-F1** | On-screen-text (PaddleOCR) as CMA *query* over T+A+V, early+late fusion | **NO** — OCR is user-vetoed; remove OCR → M-F1 falls to **0.845**; fusion = dead F50/FA family. Code: github.com/idiap/mm-hsd |
| **SCANNER** | AAAI 2026 (arXiv 2602.00132) | +4.69% M-F1 over best TTA baseline | Test-time adaptation: centroid-guided invariant-target alignment + sample-adaptive weighting + intra-cluster diversity | **NO** — cross-domain (source-train → adapt to a *different* platform's test); collides with single-dataset-train veto AND is test-time adaptation (test-touch). Evaluated on HateMM + MHC-YT + MHC-Bili but in the shift protocol, not our in-domain one |
| **CMFusion** | arXiv 2505.12051 | HateMM **0.823 acc** / 0.860 F1 | Channel-wise + modality-wise gated fusion | **NO** — isomorphic to dead FA/F50; absolute *below* our 0.877; weak encoders (ViT/MFCC/BERT). Code promised github.com/EvelynZ10/cmfusion |
| **ImpliHateVid** | arXiv 2508.06570 | own new dataset | Two-stage contrastive (ViT/wav2vec/BERT + ImageBind/OFA) | **NO** — own implicit-hate dataset, not HateMM/MHC primary; contrastive core already ours |
| HarmVideoBench / MultiHateLoc | arXiv 2606.27187 / 2512.10408 | — | LMM harmful-video *benchmark* / temporal *localization* | Not accuracy methods; MultiHateLoc aligns with our earned P6 localizer role only |

**Read:** The published SOTA on HateMM (MM-HSD 0.878) is *at* our cand-2 level (0.8775–0.8791) and only clears it via a channel we've vetoed (OCR) plus a fusion mechanism we've killed (F50). Independent confirmation of the box, not a target. **SCANNER** is the only genuinely fresh *idea* (invariant-target centroid alignment) but is governance-blocked on two counts (cross-dataset + test-adaptation) and its "target-invariance" premise is our banned target-as-structure axis.

---

## HUNT 2 — New open-weight VLM encoders (Jan–Jul 2026 + late-2025) — THE key hunt

The encoder swap CLIP→Qwen2.5-VL-7B is the **only** lever that ever converted +3 (+5.3–5.6 acc HateMM, 3/3, both protocols, F44). So a strictly-better 7B-class encoder is the highest-prior *representation-level* move available. Enumerated candidates, all locally runnable:

| Model | Released | Vision tower | Video posture vs Qwen2.5-VL-7B | Local? | Governance |
|---|---|---|---|---|---|
| **Molmo2-8B** (also 7B-Olmo, 4B) | 2026-01-09 (Ai2) | **SigLIP 2** (different family) | **Leads open-data models on SHORT video** (our exact regime), competitive long-video; built on Qwen3-8B | 8B, fits ~16GB | **Fully open** weights+data+code — cleanest |
| **Qwen3-VL-8B** (2B/4B/8B dense, 30B-A3B MoE) | 2025-10 (Alibaba) | Qwen ViT (same lineage) | **Mixed**: wins LVBench + CharadesSTA (long/grounding); **loses MVBench** 69.6 (short-clip multi-task = closest to our 8-frame hate clips) | 8B, ~8–16GB | Apache-2.0, HF/ModelScope |
| **InternVL3.5-8B** | 2025-08 (Shanghai AI Lab) | **InternViT** (different family) | Overall 59.9; ≥ Qwen2.5-VL on MMStar/MMVet | 8B | Apache/MIT |
| MiniCPM-V-4.5 (8B) | 2025 | SigLIP-ish | Doc/OCR strong; not a clear short-video win | 8B, ~6GB | open |
| GLM-4.5V (106B-A12B), Ovis2.5-34B | 2025–26 | — | strong but **too big** for our 2-GPU box | ✗ | — |

**Governance note (honest):** downloading any of these is a **user-relaxation decision**. It is arguably *in the spirit of the box* — Qwen2.5-VL-7B itself was a download, and these are the same 7B-class local tier. But the box was frozen around the models actually on disk, so it needs an explicit user "yes" before GPU. Molmo2's fully-open (weights+data+code) status is the cleanest relaxation to argue.

### The verdict on "new encoder swap" (the mission's headline question)

**Prior, decomposed by the F44 mechanism** ("swap converts iff hate is visually grounded AND errors are representation-limited"):

- **HateMM** — both conditions hold. A genuinely-better encoder has **modest-positive** prior here (≥+1 plausible). *But* (a) we're already at 0.877 with little headroom, and (b) **F67** proved sampling density is not the 7B bottleneck and **B2/32B** proved intra-Qwen scaling *regresses* — so the gain, if any, must come from a *better representation*, not more capacity. A **different-family vision tower** (Molmo2/SigLIP2 or InternVL3.5/InternViT) is therefore strictly more informative than same-lineage Qwen2.5→Qwen3.
- **MHC-EN / MHC-ZH** — errors are **label-limited** (F44: image stream collapses to near-chance, rotation-not-Pareto; B2-32B regressed on both). A better encoder does **not** fix label-limited errors. Prior ≈ **zero** for conversion. This is the wall.

**Therefore:** the new-encoder swap is the **#1 ranked untested lever**, but its honest ceiling is a *HateMM-only* nudge — it re-passes an already-passing dataset and leaves the ≥2-dataset goal blocked at the same label-limit wall that killed B1/B2/B4/F44. It is worth a **cloud-triage probe** (frozen-feature extract + head, our standard $-cheap encoder screen) *if* the user green-lights a download — best single bet **Molmo2-8B** (short-video leader + different vision tower + cleanest governance), and the probe should read HateMM (headroom test) *and* MHC-EN image-AUC (does SigLIP2 escape the near-chance collapse? — the only way EN/ZH could move). If MHC image-AUC stays near chance, F44 predicts no conversion and the direction closes with a $-cheap null, same shape as B2.

---

## HUNT 3 — Small-data / PEFT video classification 2026

Task-Adapter++ (arXiv 2505.06002, order-aware few-shot action recog), SkillMoV MoV-routing (2606.17615), Frames2LoRA "video-as-LoRA" (2606.04351), MoLE mixture-of-LoRA-experts (2506.04673). **All are richer PEFT recipes on the same encoder axis** we already banked (B3/F53 LoRA-on-encoder = the proven adaptation lever; F65 showed vision-LoRA adds nothing over LLM-LoRA). A fancier adapter is a *tactic on an already-measured axis*, not a new information source — low prior, isomorphic to the adaptation axis. Frames2LoRA is per-video internalization (QA/retrieval), doesn't map to train-split classification.

---

## HUNT 4 — Retrieval / memory-augmented classification, 2026-fresh only

Memory-Modular Classification (arXiv 2504.06021, "memory replacement") overlaps our **editable-archive pillar ④** — but that pillar's AUTO variant is already negative and only the human-in-loop deletion helped; nothing new here. DeepSeek **Engram** (Jan 2026, sparse in-forward-pass memory) is an *architecture-capacity* module, not a classification-head lever. **No fresh retrieval/memory result beats our kNN-memory core or offers a convertible lever** — round-1's retrieval conclusion holds through Jul-2026.

---

## HUNT 5 — Test-time compute / multi-view aggregation at inference

The live idea: extract embeddings from **multiple frame-subset "views" of the same video** (single model, single seed), average the **embeddings** pre-vote. This is *not* cross-seed (vetoed), *not* MLLM-decision (dead P1–P5), *not* test-touch (no label use) — it is the **unconsumed micro-ruling** territory the mission flagged. **Honest isomorphism verdict: predicted ~0.** Averaging over frame-subset draws = a smoother estimate of the *same pooled vector*, which sits squarely inside three converging kills — **F67** (denser frames tie/regress the floor), **F66/ISR** ("pooling effectively lossless"; symmetric aggregation ~0 on the third representation object), **F37/F39** (temporal structure carries zero conditional info). It is the one micro-ruling item never literally run, so it qualifies at most as a **$0 CPU companion probe**, not a GPU bet — and the mechanism findings predict a null.

---

## TOP-5 (ranked by honest prior for our goal)

| # | Candidate | Fresh source | In-box? | Prior ≥+1 / ≥+3 per dataset | Verdict |
|---|---|---|---|---|---|
| **1** | **New 7B-class encoder swap — Molmo2-8B** (SigLIP2 tower, short-video leader) | Ai2, 2026-01 | **download-gated user relaxation** | HateMM ≥+1 *plausible* / ≥+3 unlikely (0.877 ceiling); EN/ZH ~0 (F44 label-limit) | **Only untested lever with a non-trivial prior. Probe-worthy IF user green-lights a download; ceiling = HateMM-only, goal stays blocked** |
| 2 | Qwen3-VL-8B swap (same-lineage, wins long-video, loses MVBench) | Alibaba, 2025-10 | download-gated | same wall; weaker than #1 (same vision lineage as incumbent; MVBench regression in our regime) | Fallback encoder; strictly dominated by #1 for informativeness |
| 3 | Multi-view embedding-average at inference | TTA lit, 2024–26 | in-box (micro-ruling) | ~0 (isomorphic F66/F67/F37) | $0 CPU companion only; mechanism predicts null |
| 4 | SCANNER-style invariant-target centroid alignment | AAAI 2026 | **out-of-box** (cross-dataset + test-adaptation + target-as-structure) | n/a | Governance-blocked on 3 counts; do not propose as-posed |
| 5 | Richer PEFT adapters (MoLE / Task-Adapter++) on encoder | 2025–26 | in-box | ~0 new (adaptation axis already banked B3/F53; F65 null) | Tactic on measured axis; not a new information source |

---

## Explicit ranking: "new 7B-class encoder swap" vs everything else

**New encoder swap wins the field** — it is the *only* candidate touching the *only* family (representation-level) that has *ever* cleared +3, so its prior strictly dominates every decision-side, fusion, retrieval-object, aggregation, PEFT-tactic, and audio candidate below it (all of which are dead or isomorphic-to-dead). **But "wins the field" ≠ "clears the goal."** The F44 mechanism is decisive and unchanged by a better encoder: HateMM is already passing and label-limited datasets (EN/ZH) do not respond to representation quality (B1/B2/B4/F44/F65 all say so). So the correct framing for the user is:

> The best available move is a Molmo2-8B (or Qwen3-VL-8B) frozen-encoder probe, gated on a download ruling. Its realistic payoff is a *HateMM-only* re-confirmation/nudge, not a second passing dataset. If the user's bar is genuinely ≥+3 on ≥2 datasets, **no fresh 2026 result — encoder or otherwise — is predicted to clear it inside the current constraint box.** The lever that could is the same one the box already forbids: escaping the MHC label limit (more/cleaner labels, or cross-dataset signal), which is a *user-ruling*, not a literature find.

**Actionable recommendation:** if any GPU is to be spent from this sweep, spend it on a **single frozen-feature Molmo2-8B encoder probe** reading HateMM headroom + MHC-EN image-AUC (the F44 diagnostic), after an explicit user download green-light. Everything else in the 2026 fresh literature is either below us, banned, or isomorphic to an existing kill.
