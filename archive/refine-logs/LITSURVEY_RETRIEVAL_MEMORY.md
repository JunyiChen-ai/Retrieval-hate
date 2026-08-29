# LIT-SURVEY: Retrieval-augmented / memory-based / kNN classification — adjacent-work sweep for INSPIRATION

**Author:** litsurvey-retrieval-memory (CPU-only; WebSearch/WebFetch + kill-ledger cross-check; NO GPU/SLURM/Modal; `state/` untouched).
**Date:** 2026-07-24 NZST.
**Mission (user-ordered):** survey *adjacent* work — retrieval-augmented / memory-based / kNN classification in ANY modality and ANY task — for (a) work whose **paradigm** is close to ours and (b) genuinely **novel mechanisms** we could borrow. Rank by **(novelty-inspiration × paradigm-fit × feasibility)**, with an honest separate performance prior.
**Scope discipline:** recon only. No prereg, no promotion, no GPU authorization. Every candidate carries an isomorphism check against `state/directions_tried.json` (P1–P11, TARC, B1–B5, C-line, W2-wave, F47/F49/F50/F51/F55/F60, F62–F67) and `state/findings.jsonl` (F1–F67), plus an in-box legality verdict and an honest prior on **P(≥+1pt on ≥1 dataset)** given the regime (test n = 215/161/149; seed noise ≈ 1.4 pt; local G-repro discipline). External/cloud numbers are triage context only — never mixed with local numbers.

**Relationship to the prior external sweep** (`refine-logs/REDTEAM_EXTERNAL_FAMILIES.md`, 2026-07-20): that sweep attacked the *exhaustion claim* and surfaced 5 openings — its top-3 were all subsequently **measured and killed** (LP/graph-diffusion F63, vision-tower PEFT F65, single-trajectory SWA F62/F62b; also audio F64, ISR F66, frame16 F67). This survey has a **different objective** — inspiration and novelty for the D7 integration story, not a new accuracy lever. I do **not** re-propose any redteam family; I cite them as banked kills.

---

## 0. Verdict up front

The retrieval/memory/kNN literature is where our paradigm *lives*, so it is dense with **positioning anchors** — but the inspection is honest: **almost every borrowable performance operator maps onto a banked kill.** The accuracy axis of our box was closed by the campaign (Laws I–IV) and then re-closed empirically by the redteam wave (F62–F67). What the adjacent literature genuinely offers is not a new accuracy lever but three things of real value:

1. **Related-work / framing anchors** that place our contribution precisely (RAC · kNN-LM · DkNN · memory-modular classification · retrieval-enhanced contrastive) — improves the paper, +0 perf.
2. **A model-editing vocabulary** (SERAC / GRACE / WISE / MEMOIR) that gives our editable/auditable-archive pillar ④ a principled evaluation frame (reliability / generality / locality) and a hot citation neighborhood — strengthens D7 novelty at +0 perf.
3. **Two genuinely new *contribution axes*** that are $0, in-box, and do **not** fight the closed accuracy axis: (i) **conformal credibility / selective prediction** read off the existing kNN vote (DkNN + Neighborhood Conformal), and (ii) a **cross-domain memory-swap generalization protocol** (memory-modular classification) that is RGCL's original selling point and is *legal* here (inference-time bank swap ≠ cross-dataset train mixing).

**Mandatory vote-vs-logit ledger check (mission-required).** Our deployed read is the **top-20 rank-weighted signed-cosine kNN vote, NOT the logit** (P9: raw kNN below floor, MLP head ≈ floor). Every literature candidate of the form "interpolate / fuse the kNN vote with the classifier logit" (kNN-LM interpolation, RAC retrieval-branch⊕base-encoder fusion, Tip-Adapter cache⊕zero-shot blend) is **ISOMORPHIC-DEAD in our box**: killed by **P9b** (head↔memory combination = redistribution, 0/12 cells beat floor) and re-closed by **F47** (per-item logit+vote routing dead at all three supervision sources). Do not re-propose logit⊕vote fusion under any feature family — it is our single most thoroughly closed decision operator.

---

## 1. Paradigm-adjacency map (related-work anchors — verified to exist)

These are the neighbourhood our method sits in; each is a citation/positioning asset, and each borrowable *operator* is checked against the ledger.

| Anchor | Venue/year (verified) | What it does | Operator in our terms | Ledger status of the operator |
|---|---|---|---|---|
| **RAC — Retrieval-Augmented Classification** (Long et al.) | CVPR 2022, arXiv 2202.11233 | Base encoder + parallel retrieval branch over non-parametric memory of the *own train set*; retrieval branch specializes on tail classes, frees encoder for head. +14.5%/+6.7% on Places365-LT / iNat-18. | Our closest pre-RGCL classification ancestor: memory = own train, non-parametric read. Its **branch-logit fusion** = our logit⊕vote. | **DEAD** as an operator (P9b/F47). Keep as **positioning** (we are RAC's kNN-vote descendant on video). |
| **kNN-LM / RAVEN** (Khandelwal 2020; Xu et al. COLM 2024) | ICLR 2020 / COLM 2024 | Interpolate model prediction with retrieved-neighbor label distribution; RAVEN adds a retrieval-augmented encoder for ICL. | Our signed-cosine vote is the discriminative, non-interpolated analog. | Interpolation = **DEAD** (P9b/F47). **Positioning** anchor for "non-parametric label read." |
| **DkNN — Deep k-NN** (Papernot & McDaniel) | 2018, arXiv 1803.04765 | kNN over learned representations + **conformal credibility/conformity p-values** → confident, interpretable, robust. | Add a per-video **credibility** read on top of our vote. | **LIVE as a new axis** (see Cand-2); accuracy inherits B5 null but selective-prediction is un-touched. |
| **Neighborhood Conformal Prediction (NCP)** (Kumar et al.) | AAAI 2023, arXiv 2303.10694 | kNN calibration examples reweighted by distance → adaptive, tighter conformal sets. | Distance-weighted conformal wrapper on the vote. | **LIVE as a new axis** (Cand-2). |
| **Memory-Modular Classification** (memory replacement) | arXiv 2504.06021, 2025 | Non-parametric **swappable** memory; generalize to unseen classes/domains by **replacing memory contents, no retraining**. | This *is* our pillar ② (updatable memory) and RGCL's headline cross-domain capability. | **LIVE as a framing + legal cross-domain eval** (Cand-3). Its meta-learning-on-web-data part is out-of-box (external-data veto). |
| **Memory-based model editing: SERAC / GRACE / WISE / MEMOIR** | ICML 2022 / NeurIPS 2023 / NeurIPS 2024 / 2025 | External **edit cache** + **scope/range classifier** gating which edits apply; lifelong edits with **no weight change**; reliability/generality/**locality** triangle. | Our human-in-the-loop archive edits (validated 2-entry deletion, structured records) = a *discriminative* memory-editing procedure. | **LIVE as the novelty-story backbone for pillar ④** (Cand-1). An *inference-time learned* scope-gate would graze F47. |
| **NNCLR / Retrieval-Enhanced Contrastive** (Dwibedi ICCV 2021; ICLR 2024, arXiv 2306.07196) | ICCV 2021 / ICLR 2024 | Retrieved nearest-neighbor used as the **positive** (or hard-negative) during contrastive **training** → memory shapes the encoder. | This *is* the direction our **cand-2 curriculum** lives in (FAISS-mined hard negatives, per-epoch reindex). | **Positioning for cand-2**; curriculum *tactics* banned without a new structural premise (ledger). |

---

## 2. Surviving candidates (full check; ranked in §4)

### Cand-1 — Editable-archive-as-**memory-editing** (framing + optional locality gate) — *novelty-story lead*
- **(a) Citations:** SERAC (Mitchell et al., ICML 2022); GRACE (Hartvigsen et al., NeurIPS 2023); WISE (Wang et al., NeurIPS 2024, `VJMYOfJVC2`); MEMOIR (OpenReview `t94tALZvZE`, 2025). All verified.
- **(b) Mechanism:** treat the base model as fixed; keep an **external cache of edits**; a **scope/range classifier** decides, per query, whether a cached edit is in-scope; if so the retrieved edit overrides the base read. Lifelong, weight-free, evaluated on **reliability / generality / locality**.
- **(c) Paradigm-adjacency:** our pillar ④ *is* a weight-free external memory whose entries are edited by a human (the ZH-validated 2-entry deletion) and are structured/auditable. The editing literature is the exact vocabulary and evaluation frame our pillar currently lacks.
- **(d) Borrowable operator in our terms:** (i) **framing/eval, $0-perf:** report archive edits under the reliability/generality/locality triangle (does deleting a noisy entry fix the target queries *without* moving unrelated ones?) — a principled, citable characterization of the human-in-the-loop deletion we already validated. (ii) **optional mechanism:** a **locality gate** that masks an individual memory entry from voting when it is out-of-scope for the query — the SERAC scope-classifier idea restricted to *per-entry vote masking*.
- **(e) In-box legality:** framing use = fully in-box (writing only; no gold in deployed path). The *learned inference-time* locality gate **grazes F47** (per-item learned decision over frozen features) → any mechanism variant needs the $0 G0-cond conditional-info gate + label-oracle calibration arm, and its prior is low by F47.
- **(f) Cost:** framing = writing only; locality-gate probe = $0 CPU on cached keys.
- **(g) Prior / novelty:** performance prior **~5–10%** (mechanism ≈ F47 class). **Novelty value HIGH** — it upgrades pillar ④ from an anecdote (2-entry deletion) to a mechanism with borrowed evaluation metrics and a live citation neighborhood; directly strengthens the D7 integration story at +0 perf.
- **(h) Rank: #1 on (novelty × fit × feasibility).**

### Cand-2 — **Conformal credibility / selective prediction** on the kNN vote — *new contribution axis*
- **(a) Citations:** DkNN (Papernot & McDaniel 2018, `1803.04765`); Neighborhood Conformal Prediction (Kumar et al., AAAI 2023, `2303.10694`); "Conformal Prediction via Label Ranking" (`2310.06430`). Verified.
- **(b) Mechanism:** compute a conformal nonconformity score from neighbor-label agreement, yielding per-item **confidence** (correctness likelihood) + **credibility** (train-set conformity) and calibrated **prediction sets / abstention** with a coverage guarantee; NCP reweights the calibration neighbors by distance.
- **(c) Paradigm-adjacency:** DkNN *is* our read (kNN over a learned representation); we already have the neighbor set and signed-cosine similarities — the conformal wrapper is a few dozen LOC on cached votes.
- **(d) Borrowable operator in our terms:** wrap the deployed top-20 vote with a conformal score computed on a train/dev calibration split → emit a per-video credibility and support **human-in-the-loop abstention** (route low-credibility videos to a moderator) with a coverage guarantee. Ties pillar ④ (auditable) to a formal selective-prediction result.
- **(e) In-box legality:** **in-box, per-item, $0.** Uses only train labels for calibration; standard split-conformal is per-item and does not touch test beyond the legal single read. Not a channel selector (F47 is silent — it operates on one representation's own vote).
- **(f) Cost:** **$0 CPU** on banked keys/votes.
- **(g) Prior / novelty:** **accuracy prior LOW (~5–10%)** — a pure operating-point/credibility read inherits the **B5/F34 null** (the ZH AUC edge is easy-example ordering, unconvertible at any threshold incl. oracle). BUT accuracy is the wrong axis: this adds a **genuinely new result type** (calibrated selective-prediction / abstention curves) that the pipeline has never reported and that complements the auditable pillar. **Novelty value MODERATE-HIGH.**
- **(h) Rank: #2** — cheapest way to add a *new* contribution axis instead of re-fighting the closed one.

### Cand-3 — **Cross-domain memory-swap** generalization protocol (memory-modular) — *pillar-② story + a legal table*
- **(a) Citation:** Memory-Modular Classification / memory replacement (arXiv 2504.06021, 2025). Verified.
- **(b) Mechanism:** knowledge lives in a swappable non-parametric memory; at inference the model **adapts to new classes/domains by replacing the memory contents, without retraining.**
- **(c) Paradigm-adjacency:** this is exactly pillar ② (updatable memory) and the *original RGCL selling point* — the kNN-vote classifier extends to new domains by swapping the bank. Our pipeline already supports it.
- **(d) Borrowable operator in our terms:** train the head on dataset A, then at **inference swap in dataset-B's memory bank** and read the vote — a zero-retrain cross-domain generalization table (e.g., HateMM-trained head reading an MHC bank, and vice-versa). Foregrounds "editable/updatable memory" as a measured capability, not a claim.
- **(e) In-box legality:** **IN-BOX** — critically, this is an **inference-time bank swap, NOT cross-dataset train mixing**, so the 2026-07-14 single-dataset-train veto is **not** violated (the veto bans mixed *training* splits; test-time memory substitution is a different object). The meta-learning-on-external-web-data part of 2504.06021 *is* out-of-box (external-data veto) — borrow only the swap-eval protocol, not their training.
- **(f) Cost:** low — cached keys already exist; the swap read is CPU.
- **(g) Prior / novelty:** not an in-domain +pt lever on our three anchors (so the ≥+1pt prior is N/A / low). **Novelty value HIGH** for pillar ② — turns an asserted capability into a table and matches a 2025 anchor. Caveat: cross-domain kNN transfer numbers may be modest and must be reported as *generalization*, never as an anchor-accuracy claim.
- **(h) Rank: #3.**

### Cand-4 — **Error-/misjudgment-pattern memory content** (PatMD) — *novel content, perf-dead*
- **(a) Citation:** "Fall into a Pit, Gain in a Wit: Cognitive-Guided Harmful Meme Detection via Misjudgment Risk Pattern Retrieval" (PatMD), arXiv 2510.15946v3, 2025. Verified (5 harmful-meme tasks, 6,626 memes; reports avg **+8.30% F1 / +7.71% acc** over baselines — external number, triage-only).
- **(b) Mechanism:** memory stores **why an item might be misjudged** (deconstructed memes tagged with false-negative / false-positive *risk patterns*), retrieved to **guide an MLLM's reasoning** via prompt augmentation at inference.
- **(c) Paradigm-adjacency:** same retrieve-then-decide skeleton, but the *memory content is error patterns* rather than labeled examples — a genuinely different memory-content idea.
- **(d) Borrowable operator in our terms:** store structured **error-pattern cards** in the editable archive (novel content for pillar ④, human-auditable).
- **(e) In-box legality / isomorphism:** to *use* error patterns for prediction requires either **MLLM-reasoning-at-inference** (P1–P5 / MLLM-decision-side = **DEAD**) or **MLLM-scores-as-training-signal** (**banned constraint**). So the performance operator is **ISOMORPHIC-DEAD**. Only the *archive-content* use (human-facing error cards) is in-box, at +0 perf.
- **(f) Cost:** content generation is MLLM-cloud/offline; audit use is free.
- **(g) Prior / novelty:** performance prior **~0** (usable operator maps to dead classes). **Novelty value MODERATE** — a fresh, auditable memory-content type for pillar ④.
- **(h) Rank: #4** (content-novelty note only).

### Cand-5 — **Robust / evidential vote aggregation** (trimmed / Dirichlet-reliability) — *denoising-pillar framing, perf-dead*
- **(a) Citations:** RNNP robust prototypes (`2011.11067`); TraNSF robust aggregation; multi-view evidential K-NN; Dirichlet-based prediction calibration for noisy labels (`2401.07062`). Verified.
- **(b) Mechanism:** replace mean/rank-weighted aggregation with a label-noise-robust operator (trimmed/median, or evidential Dirichlet with per-neighbor reliability weights).
- **(c)–(e) Isomorphism:** a **non-selecting** robust aggregator over frozen keys is exactly the **ISR/F66 symmetric-aggregation ≈ 0** class (headroom proven **selection-locked**, law-I arithmetic); a per-neighbor **reliability weighting that selects** is **law-III / F49** (per-item selection dead). Pooling is lossless (Law II). **ISOMORPHIC-DEAD.**
- **(g) Prior / novelty:** performance prior **~0–5%**. Minor framing value for the **ZH-validated denoising pillar ③** (evidential reliability as a lens on consensus denoising), +0 perf.
- **(h) Rank: #5.**

---

## 3. Checked and marked ISOMORPHIC-DEAD (listed per instruction; one line each)

| Candidate (external anchor) | Maps onto banked kill | Why |
|---|---|---|
| kNN-vote ⊕ classifier-logit interpolation (kNN-LM; RAC branch fusion; Tip-Adapter blend) | **P9b + F47** | logit⊕vote combination = redistribution (0/12) and per-item routing dead at all 3 supervision sources. |
| Learned permutation-invariant aggregator over top-k neighbors (Set Transformer / Deep Sets / attention-over-neighbors, ICLR'25 set-function work) | **P9b + F47** | a trained readout over the retrieved set = the joint head↔memory object (redistribution) and a learned decision-level combiner. |
| Trained GNN / message-passing over the similarity graph (GraFN; diffusion-GCN) | **F63 + P9b** | training-free LP already killed all 3 datasets; the trained/transductive form adds test-graph (test-touch + pseudo-label graze) and ≈ P9b head-training. |
| Prototype/cluster memory, Tip-Adapter/Proto-Adapter cache-model | **W2-E (F28) + F50** | prototype-memory = lossy fn of pooled vector (killed W2-E); the zero-shot-prototype blend = killed decision-side fusion. |
| Multi-hop label propagation / ECALP over the kNN memory (redteam Family A) | **F63** | monotone-negative in α, all datasets at/below one-hop; EN positive inside perm-null. |
| Vision-tower / projector PEFT (VPT/AdaptFormer/SSF; redteam Family B) | **F65** | image AUC moved but K-V2 TIE at the head everywhere (8th law-I instance). |
| Single-trajectory SWA / weight-averaging of head checkpoints (redteam Family C) | **F62 / F62b** | HateMM loses dev pts on tax-bearing seeds; ZH dev-underpowered; Family C closed both datasets. |
| Learned-audio (Whisper / AST / wav2vec2) third stream | **F64 + APX/F41** | Whisper zero conditional info all 3 datasets; any audio must first beat a zero-info classical screen (transcript banks spoken content, F31). |
| Test-time frame augmentation / multi-view TTA (redteam Family E) | **F67 + Law I** | denser/alternate sampling adds nothing over 8f (F67); variance reduction ≈ 0 at n≈150. |
| Manifold/embedding mixup + consistency reg on head (redteam Family D) | *un-isomorphic but D7-dead* | generic head regularizer, perf-only, near seed-noise; novelty-null for D7. |
| Conformal/dev **threshold** co-tuning for accuracy | **B5 / F34** | label-oracle operating-point ceiling itself < +0.03 both protocols (distinct from Cand-2's *selective-prediction* use, which is not an accuracy claim). |
| Retrieval-augmented ICL / RAVEN as inference-time reasoning; PatMD's MLLM-reasoning read | **P1–P5 + external-API/generation constraints** | MLLM-decision-side reasoning is the dead P-line; closed-API generation is vetoed. |
| Retrieval-mined hard-negative **curriculum** variants (NNCLR/retrieval-contrastive tactics) | **cand-2 ledger note** | cand-2 already occupies this; rep2 binding-consumed, no new curriculum tactics without a new structural premise. |
| Self-influence / retrieval-guided **training-sample reweighting** (PRESENCE, LESS) | *grazes AUG/F60 + cand-2* | data-selection/reweighting over own train ≈ AUG's dominated object; no promoting cheap gate on our tiny train. |

---

## 4. TOP-5 — ranked by (novelty-inspiration × paradigm-fit × feasibility)

| # | Candidate | In-box? | Cost | Accuracy prior (≥+1pt/≥1 ds) | Novelty/story value | Why it ranks |
|---|---|---|---|---|---|---|
| **1** | **Editable-archive-as-memory-editing** (SERAC/GRACE/WISE/MEMOIR framing + optional locality gate) | YES (framing); mechanism grazes F47 | writing / $0 probe | ~5–10% | **HIGH** | Upgrades pillar ④ from anecdote to a mechanism with borrowed reliability/generality/locality metrics + a hot citation neighborhood; pure D7-story win at +0 perf. |
| **2** | **Conformal credibility / selective prediction on the vote** (DkNN + NCP) | YES, per-item, $0 | $0 CPU | ~5–10% (B5-capped) | **MOD–HIGH** | Adds a *new contribution axis* (calibrated abstention / credibility) instead of re-fighting the closed accuracy axis; complements the auditable pillar; cheapest new result the pipeline can produce. |
| **3** | **Cross-domain memory-swap protocol** (memory-modular classification) | YES (bank swap ≠ train mixing) | low, cached keys | N/A (generalization, not in-domain +pt) | **HIGH** | Turns pillar ②'s asserted "updatable memory" into a measured cross-domain table matching a 2025 anchor; legal because it swaps memory at *inference*, not training. |
| **4** | **Error-/misjudgment-pattern memory content** (PatMD) | content-only in-box; perf use DEAD (P1–P5 / banned) | offline gen | ~0 | **MODERATE** | A novel, auditable memory-*content* type for pillar ④; performance operator is isomorphic-dead. |
| **5** | **Robust / evidential vote aggregation** (RNNP/TraNSF/evidential-kNN) | DEAD (F66/F49) | $0 | ~0–5% | LOW | Minor framing lens for the ZH-validated denoising pillar ③; aggregation cannot convert selection-locked headroom. |

**Bottom line.** The adjacent retrieval/memory/kNN literature confirms — from the *inspiration* side, independent of the redteam's *exhaustion* side — that our pipeline sits in a well-populated, currently-hot neighborhood (RAC → RGCL/RA-HMD → memory-modular classification; kNN-LM → DkNN → NCP; SERAC → WISE → MEMOIR). But every borrowable **accuracy** operator maps onto a banked kill (logit⊕vote = P9b/F47; LP = F63; prototypes = W2-E; vision-PEFT = F65; SWA = F62; audio = F64; robust/non-selecting aggregation = F66). The honest, useful yield is **three +0-performance but high-novelty moves for the D7 integration story**: (1) reframe the editable archive as *discriminative memory-editing* with a borrowed reliability/generality/locality evaluation (Cand-1), (2) add a $0 in-box **selective-prediction / credibility** result on the existing vote (Cand-2), and (3) report the **cross-domain memory-swap** generalization that is RGCL's headline capability and legal here (Cand-3). None is an accuracy lever; all three strengthen the *story* the campaign's numbers already support.

---

## Provenance

- **Kill ledger:** `autoresearch/goal_mllm_plus3/state/directions_tried.json` (P1–P11, TARC, B1–B5, C-line, W2-wave, F47/F49/F50/F51/F55/F60, F62–F67, banned_constraints); `state/findings.jsonl` (F1–F67); `refine-logs/REDTEAM_EXTERNAL_FAMILIES.md` (redteam families A–E, LP/vision-PEFT/SWA/audio/ISR/frame16 all subsequently killed).
- **External anchors (verified to exist via search; 2018–2025):** RAC CVPR 2022 (arXiv 2202.11233); kNN-LM ICLR 2020; RAVEN COLM 2024; DkNN 2018 (arXiv 1803.04765); Neighborhood Conformal Prediction AAAI 2023 (arXiv 2303.10694); Conformal via Label Ranking (arXiv 2310.06430); Memory-Modular Classification 2025 (arXiv 2504.06021); SERAC ICML 2022; GRACE NeurIPS 2023; WISE NeurIPS 2024 (OpenReview VJMYOfJVC2); MEMOIR 2025 (OpenReview t94tALZvZE); NNCLR ICCV 2021; Retrieval-Enhanced Contrastive Vision-Text ICLR 2024 (arXiv 2306.07196); PatMD 2025 (arXiv 2510.15946, external numbers triage-only); RNNP (arXiv 2011.11067); Dirichlet prediction calibration for noisy labels (arXiv 2401.07062); PRESENCE / LESS self-influence reweighting (arXiv 2311.00913, rejected).
- **Discipline:** CPU-only; zero GPU/SLURM/Modal/downloads; `state/` unmodified; cloud/external numbers never mixed with local G-repro numbers.
