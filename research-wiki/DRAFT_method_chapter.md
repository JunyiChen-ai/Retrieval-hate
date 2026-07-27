# Method

*Draft chapter — framing per DECISION_MEMO D4 recommendation (Kit-A A.2); revisable upon user ruling.
All quantities are transcribed from committed project documents; internal provenance is given as
[DOC:file] and external methods / datasets as `\cite{}` placeholders. No number or claim in this
chapter is new — each is traceable to a committed result. This chapter is written for the
finalize-as-is option (a) and does not depend on any pending headline-protocol ruling (D2).*

---

## 1. Overview

We detect hateful video with a **retrieval-memory** detector rather than a trained classification
head. A short clip is turned into a small set of frozen multimodal features; a lightweight
alignment-fusion head is trained with a **retrieval-guided contrastive objective**; and, at
inference, the label is decided by a **k-nearest-neighbour (kNN) vote over a labelled memory bank**
in the learned fused space — not by the classifier logit. The pipeline has four stages:

1. **Frozen (or LoRA-adapted) MLLM/CLIP encoder.** Each video is encoded once, offline, into a
   two-stream `{image_feats, text_feats}` cache (8-frame visual pooling + a title/transcript text
   tower). The encoder is never back-propagated through in the core recipe; it is a swappable
   front-end (§7). We port this core, unchanged in spirit, from RGCL / RA-HMD for hateful memes
   \cite{rgcl,rahmd}.
2. **Alignment-fusion head (≈ a few million trainable parameters).** A `classifier_hateClipper`-style
   head projects each stream to a shared space, fuses them by elementwise product (`align`), and
   returns both a logit and an L2-normalised embedding used as the retrieval key
   [DOC:DESIGN_iter1.md §2.2].
3. **Retrieval-guided contrastive training.** Per epoch we (re)index the head's own embeddings in
   FAISS and mine, for every anchor, a **pseudo-gold positive** (nearest same-label exemplar) and a
   **hard negative** (nearest opposite-label exemplar), training a margin-contrastive objective over
   these mined pairs plus in-batch negatives (§7).
4. **kNN-vote inference over a labelled memory.** The prediction is a similarity-weighted vote of the
   *K* nearest labelled exemplars in the memory bank — a **non-parametric read-out**, not a fixed
   classification head. This single design choice is what makes the memory *updatable*,
   *swappable*, *auditable*, and *editable* (Pillars 2–4).

On this backbone we make four contributions, all scoped strictly to hateful **video**: (P1) the
retrieval-guided-contrastive + kNN core itself, ported to video and evaluated head-to-head against
the only published retrieval-augmented hateful-video method; (P2) a zero-retrain updatable memory
with an O(1) temporal-recalibration protocol; (P3) a consensus denoising mechanism for
label-inherited segment supervision, validated on Chinese with an honest cross-lingual boundary;
and (P4) an auditable and human-editable archive memory. A fifth thread characterises the **three
earned roles and one explicit non-role** of the MLLM (§6). Figure 1 (**architecture diagram —
TODO**) sketches the four stages and the four memory operations.

## 2. Pillar 1 — retrieval-guided contrastive embedding + kNN-vote inference

**The core.** Training mines, per anchor and per epoch, a FAISS-retrieved pseudo-gold positive and a
hard negative from the *learned* embedding space, and optimises a margin-contrastive loss so that
same-label exemplars cluster and confusable opposite-label near-duplicates are pushed apart; the
confusable hard negative is exactly the hateful-vs-offensive / benign-confounder case these
benchmarks document as hardest [DOC:gap_map.md G1/G2]. Inference then reads the label off a
kNN vote over the labelled memory.

**Delta vs the meme origin (RGCL / RA-HMD).** By project scope, RGCL/RA-HMD is *inspiration*, so the
bare port carries their novelty, not ours. Two adaptations are load-bearing for video: (i) a
video front-end (8-frame pooling + transcript/title text) feeding the identical head, and (ii) the
memory operations of Pillars 2–4, which the meme setting never exercises (a meme is static and its
memory is treated as fixed) [DOC:DESIGN_iter1.md §1].

**Delta vs the closest field method (MoRE, WWW 2025).** MoRE is the *only* published
retrieval-augmented hateful-video method \cite{more}, so the honest framing is a **precise
mechanism delta, not a "we bring retrieval to hateful video" claim** — retrieval already exists in
the field. The differences are architectural [DOC:gap_map.md G1, DOC:DESIGN_iter1.md §1]:

| Axis | MoRE | Ours |
|---|---|---|
| Retriever | **frozen** weighted-cosine, heuristic | **learned** fused space |
| Objective | BCE over attention experts (BHAN) + MoE router | retrieval-guided **contrastive** over FAISS-mined pseudo-gold-positive / hard-negative |
| Inference head | trained MoE router (weights bake in the decision) | **kNN vote** over a labelled memory (non-parametric) |
| Test-time memory update | structurally impossible | swap / append, **zero retrain** (Pillar 2) |

On the identical binary protocol — same split (line-for-line diff), same clean test subset, MoRE's
own released code re-run (as-released and bug-fixed, 5-seed) — our best configuration wins on all
three shared benchmarks: **+5.6 / +8.7 / +6.7 accuracy** (HateMM / MHClip-EN / MHClip-ZH) and
**+6.2 / +22.9 / +9.7 macro-F1** over the stronger MoRE variant [DOC:BASELINE_MoRE_rerun.md,
DOC:PAPER_MASTER_TABLES.md T1.2]. HateMM and ImpliHateVid already meet the field's acc ≥ 0.85 bar
(frozen-Qwen 0.870 / 0.861 on HateMM); MHClip-EN sits near a documented ceiling (§6 non-role, and the
analysis chapter's attribution) and MHClip-ZH is reported under both protocol calibrations (§7, D2).

## 3. Pillar 2 — updatable memory and a temporal-recalibration protocol

Because the decision is a kNN vote over a labelled bank, the classifier is *reconfigurable at
inference with no gradient step*. Two operations follow.

**Cross-dataset swap (capability).** A head trained on source A classifies target T by **swapping in
T's own labelled memory** — no retraining. The learned space transfers: it beats the target
majority baseline on **5 of 6** informative cross cells, lagging in-domain by only ≈ 0.04–0.09
macro-F1 on working cells [DOC:experiments/exp-cross-dataset-transfer.md, DOC:PAPER_MASTER_TABLES.md T3]. A
trained MoE head is **structurally incapable** of being re-pointed at a new support set; this is a
headline capability delta vs MoRE, reported honestly as a capability demonstration (cross never
beats in-domain), not an accuracy win.

**Temporal protocol (mechanism + fix).** On an MHClip-EN temporal split, performance drops −0.084
macro-F1 (0.7113 → 0.6273). We attribute the drop to **calibration drift, not a loss of
separability**: the temporal-split ROC (0.8484) *exceeds* the random-split reference (0.7175), so the
classes remain separable and only the operating point has moved [DOC:EVAL_temporal_memory_W4.md].
The correct lightweight adaptation is therefore **threshold recalibration**, not memory growth:
labelling k = 20 new-period examples and re-calibrating the operating point recovers the drift in
full (0.7336 ≥ the random-split floor 0.7113) with **zero retraining**, against an oracle ceiling of
0.7646. The retrieval architecture exposes the operating point as a **first-class, O(1), reversible
knob**; a trained MoE head hides it in its weights. Two honesty controls travel with the claim: the
naive "add the k = 20 samples to the memory" mechanism is flat-to-negative (0.6180), so the win is
recalibration specifically; and MHClip-ZH shows **no drift** (a negative control), so under a
no-drift regime small-k recalibration is pure noise and deployment should gate it behind a
drift monitor.

## 4. Pillar 3 — consensus denoising of label-inherited segment supervision

Sub-clips of a hateful video inherit the video label, but most are benign; this **poisons** a naive
segment-contrastive term. Our consensus E-step re-labels each sub-clip by the agreement between its
inherited video label and a similarity-weighted kNN vote of *video-level* labels over the labelled
memory, keeping only confident (margin ≥ τ) sub-clips and routing the disagreement cell
(hateful video, benign vote) to a within-video *drifting hard negative* rather than a positive; the
head is retrained across a short EM outer loop [DOC:src/utils/consensus.py, DOC:DESIGN_iter3.md SS2].

**Honest, language-conditioned boundary.** On MHClip-ZH the mechanism **repairs** the poisoning: the
inherited-label segment term costs −0.066 macro-F1, and consensus removes that hole, landing at or
weakly above the floor across 5 seeds and both protocol calibrations. It **does not beat the floor**
significantly (val-selected +0.0115, p ≈ 0.57; final-epoch +0.0247, p ≈ 0.11), so the claim is
precisely *"consensus de-poisons sub-clip supervision"*, not an accuracy win
[DOC:experiments/exp-consensus-zh-seeds.md]. On MHClip-EN the same repair does **not** transfer, and the
attribution chain is complete rather than hand-waved: swapping the vote space (archive / blend)
does not rescue it, so the vote *space* is not the culprit; an evidence-matched segment-speech key
(window-level ASR + CLIP-text) fully repairs the *annotator* (supervision supply, within-video vote
granularity, severity anti-correlation) yet training stays ≤ floor — pinning the residual failure on
**the segment-supervision channel itself having no gain for speech-carried hate**, not on a bad key
choice [DOC:EXP_mm_segment_keys.md]. This yields a reusable methodological by-product:
**evidence-matched segment keys + a probe-before-train gate**.

## 5. Pillar 4 — auditable and editable archive memory

Encoding each video into an MLLM-produced **structured archive record** (schema fields:
explicitness, modality, mechanism, target group) gives the memory two properties a weight-baked head
cannot offer [DOC:AUDIT_archive_faithfulness.md, DOC:DEMO_memory_editing.md].

**Auditable.** A stratified audit finds the archive faithful on 77% of sampled records, with three
recurring failure modes catalogued; the label-blind audit independently re-discovers human-flagged
noisy memory ids with correct reasons.

**Editable (human-in-the-loop).** Semantic addressing + surgical deletion is pure-CPU and seconds-fast.
Deleting **two** human-flagged noisy memory entries lifts MHClip-EN test accuracy 0.8075 → 0.8199 **at
seed 0** with **zero retraining**, exceeding all five random-seed floors. **This is a human-in-the-loop
capability demonstration, single-seed; it is not an accuracy claim.** The round-8 multi-seed replay (F88)
finds +0.0124 on seed 0 and **zero vote flips on seeds 1–3** (four-seed mean +0.0031), with the 14-id rule
list stronger but still sub-bar (+0.0093 acc / +0.0089 mF1, 3 of 4 seeds, 0 items broken), so the earlier
"project's best single EN point" wording is withdrawn and the property claimed here is *editability*, not
a gain [DOC:ERRPAT_MHC-EN_2026-07-26.md §6.5, commit `ad56a62`; experiments §5, analysis §4].

**Guard-rail / semantic veto (automatic, bounded).** A two-vote AND rule for *automatic* repair does
**not** reproduce the human gain (C − A = +0.0000, 0/4 EN seeds): it structurally cannot reach
memories that are semantically contradictory yet not embedding outliers. Its surviving value is a
**semantic veto**: it blocks an embedding-only (Cleanlab-style) rule from over-deleting
genuinely-hateful-but-embedding-hard entries (abuse testimony, assault reporting, slur-bearing text),
worth C − D = +0.47pt EN / +0.40pt ZH [DOC:EXP_auto_memory_repair.md]. The archive's payoff is
**integrity and controllability, not raw accuracy** — a framing defensible precisely because it is
not dressed as an accuracy claim.

## 6. The MLLM's three roles and one explicit non-role

We fixed, as a pre-registered mandate, that the MLLM must earn a *removable* role — one whose
deletion measurably costs performance beyond the ~150-video noise floor (1 acc pt ≈ 1.6 videos). It
earns exactly **three roles, none of them a main-table-accuracy role** [DOC:CAMPAIGN_mllm_method_role.md,
DOC:TERMINUS_mllm_campaign_DRAFT.md; see the analysis chapter for the full campaign].

**(i) Encoder (main-table lever).** Frozen Qwen2.5-VL features beat CLIP by +4.2 macro-F1 on HateMM
and cross the 0.85 bar (0.870 / 0.861). This is the *frozen-encoder identity*, not the new method
role the mandate sought, and we say so [DOC:PAPER_MASTER_TABLES.md T1.1].

**(ii) Localization scorer (P6 → P10-b, the one removable method role).** A per-window MLLM scorer
reads M = 120 frames binned into **K = 30 windows** (≤ 4 frames/window) plus that window's Whisper
ASR and emits an integer 0–3 hate-evidence density; span-free within-video AUC (wv-AUC) is the
primary metric [DOC:EXP_p6_mllm_localization.md]. The 7B scorer reaches **wv-AUC 0.5435** vs memory
0.5140 and random 0.5088 (paired vs memory Δ +0.0296, CI [+.009, +.050], p = 0.007). Amplification
uses **A-fuse**, a coarse×fine rank aggregation, `0.5·(K=30 fine) + 0.5·(K=4 coarse, mapped)`,
computed from the **same** scorer's outputs; its gain grows monotonically with scorer size (+0.0305
7B → +0.0437 32B → +0.0526 72B on calibration), the one lane where scale converts. The promoted 72B
A-fuse, on the single permitted HateClipSeg test touch, reaches **wv-AUC 0.5755** (CI [0.5581,
0.5933]; paired vs memory +0.0615, vs 7B +0.0319) — **modest**, not substantial (< 0.60)
[DOC:PAPER_MASTER_TABLES.md T2.1]. Crucially, the paper's **contrast for this role is the retrieval
memory, not a MIL head**: A-fuse − memory = +0.0996 (significant), whereas A-fuse − a same-operator
video-label MIL proxy is not significant [DOC:EXP_p11_weaksup_localization.md]. We contrast against
memory because a MIL head needs target-domain video labels that the zero-shot memory swap does not —
the two are not the same capability. Wording red line: we claim only **span-free** localization
(never first / annotation-free / dense-supervision-free).

**(iii) Guard-rail / audit.** The semantic-veto and human-audit roles of §5 — removal cost surfaces
as controllability, not accuracy.

**Explicit non-role (for falsifiability).** At 7B–72B scale, the MLLM earns **no main-table-accuracy
role** in this retrieval-memory pipeline. Eleven pre-registered routes (label-noise repair, prior
recalibration, neighbour reranking, evidence-density pooling, schema distillation, counterfactual
mining, score fusion, semantic speech compression, decision-level LoRA-SFT) are all honest kills or
within-noise, each
guard-backed by a reproduction / bit-for-bit / probe check [DOC:PAPER_MASTER_TABLES.md T4]. Two
mechanistic conclusions generalise: comparability ⊥ vote-correctness (scale improves the judge's
*calibration*, not its *selectivity*), and a passing no-head probe is *necessary but not sufficient*
(a learned alignment-fusion head absorbs input-space advantages). The quantified prize is real but out of
this campaign's reach: an oracle membership editor lifts the gated slice to 100% and overall accuracy
+7.5pt EN / +10.6pt ZH (both cross 0.85) — which a stronger comparability judge is shown *not* to
unlock. Details are deferred to the analysis chapter [DOC:DRAFT_analysis_chapter.md].

## 7. Training and inference details

**Encoder front-ends (swappable, run once offline).** (a) frozen CLIP ViT-L/14-336 — 8-frame
mean-pool visual (1024-d) + title/transcript text tower (768-d); (b) frozen Qwen2.5-VL-7B-Instruct
hidden-state pooling (Dv = Dt = 3584); (c) LoRA-adapted Qwen2.5-VL-7B, used for the ZH main stack
[DOC:DESIGN_iter1.md §2.1, DOC:PAPER_MASTER_TABLES.md T1.1]. `run_rac.py` reads feature dims from the
cache, so no head-code change is needed across front-ends.

**Head.** `classifier_hateClipper`: per-stream linear projection to `map_dim = 1024`, L2-norm,
**`align` fusion** (elementwise product), a 3-layer MLP (`proj_dim = 1024`, dropout [0.2, 0.4, 0.1]),
returning `(logit, L2-norm embedding)`; the embedding is the retrieval key.

**Objective.** `L = L_contrastive + ce_weight · L_BCE` (hybrid, `ce_weight = 0.5`). The
retrieval-guided term is a **triplet-margin** contrastive loss (cosine metric, margin 0.1) over one
FAISS-mined pseudo-gold positive and one hard negative (mined from a 12× candidate pool) plus in-batch
negatives, with a per-epoch FAISS reindex over the head's current embeddings [DOC:src/model/loss.py].
*(Note: the RGCL family is named for "retrieval-guided contrastive"; the shipped video objective is
the triplet-margin instance, not InfoNCE — see the consistency note below.)*

**Optimisation.** AdamW, lr 1e-4, weight decay 1e-4, grad-clip 0.1, batch size 64, 30 epochs,
warmup ≥ 5 for best-epoch eligibility.

**Inference head.** The prediction is the **kNN vote** (K = 20, arithmetic similarity-weighted) over
the labelled memory in the learned fused space — the primary metric — *not* the BCE logit. Memory
operations (swap / append / edit) act on this bank without any gradient step.

**Memory-augmentation knobs.** Archive-kNN memory-key augmentation uses α = 0.25 (effective archive
weight α²/(1+α²)); consensus uses vote-k = 10, margin τ = 0.2, EM rounds = 2 [DOC:src/run_rac.py].

**Reporting protocol (two calibrations, side by side — per D2 recommendation).** Every
classification number is reported under **both** the pre-registered protocol (warmup ≥ 5,
validation-selected on the retrieval metric) **and** a selection-free final-epoch protocol
[DOC:PAPER_MASTER_TABLES.md T1, DOC:MORNING_REPORT.md §6]. On ~150-video tests, 1 acc pt ≈ 1.6
videos and a 78-sample dev makes validation-selection itself cost ≈ 2 acc pts, so sub-point,
single-protocol "gains" are recorded as within-noise. There is **no cross-seed ensembling**; every
gain claim passes a sha1 / bit-for-bit identity audit before a statistical test.

**Localization protocol.** HateClipSeg gold spans are validation-only (no label enters any scoring
path); scoring is zero-training over all 395 alive-subset videos, with within-video mean-AUC (+ 10k
bootstrap CI + sign-test) as the primary, threshold-free metric [DOC:EXP_p6_mllm_localization.md].
The test split was contacted **exactly once**, by the single promoted amplifier.

**Hyperparameters (full table — TODO).** A consolidated hyperparameter table (encoder, head, loss,
optimiser, memory knobs, per-dataset seeds) is deferred to a table placeholder.

---

### Method-vs-code consistency notes (for internal review; remove before submission)

1. **Objective family vs instance.** The shipped video runs use `loss='triplet'` (triplet-margin,
   cosine, margin 0.1), hybridised with BCE at `ce_weight=0.5`, confirmed from a live run Namespace
   [DOC:slurm/logs/arc_MHC_zh_*_12215.trainlog]. An InfoNCE path (`loss='contrastive'`) exists in
   `src/model/loss.py` but is **not** the video recipe. Framing docs (DESIGN_iter1 §2.3) say
   "InfoNCE/triplet"; this draft states the triplet-margin instance to stay faithful to what runs.
   Recommend the paper say "retrieval-guided contrastive (triplet-margin)" rather than "InfoNCE".
2. **Fusion mode.** `parse_args` default is `fusion_mode='concat'`, but every training script and the
   live run use `--fusion_mode align` (elementwise product). The draft describes `align`, matching the
   runs; the argparse default is a non-load-bearing discrepancy.
3. **Head parameter count.** DESIGN_iter1 §2.2 states ≈ 6–8M trainable params; with the 3584-d Qwen
   streams the two input projections alone are ≈ 7M, so the true count is encoder-dim-dependent and
   larger for Qwen than for the CLIP front-end. The draft hedges to "a few million"; a precise count
   per front-end should be filled into the hyperparameter table.
4. **Archive-kNN α default.** `parse_args` default `archive_alpha=1.0`, but the ZH main stack uses
   α = 0.25 (confirmed in the run Namespace). This is a config choice, not a code inconsistency; the
   archive-kNN key contributes ≈ 0 accuracy at final-epoch (already documented in T1.1) — the archive's
   value is in Pillar 4 (audit/edit), not accuracy.
