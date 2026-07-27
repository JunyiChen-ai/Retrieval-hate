# DESIGN — Iteration 1: Cross-dataset updatable-kNN-memory RGCL for hateful video (MLLM-encoded); multi-granularity as analysis

_Original: 2026-07-01. **CONFIRMED REVISION: 2026-07-02** (user-approved reframe). **CONSOLIDATION: 2026-07-02** (Phase-3 results in — headline demoted/promoted). Status: proposed. Idea slug: `rgcl-mllm-video-iter1`._

> **2026-07-02 confirmed-revision note.** This document supersedes the 2026-07-01 "SR-RGCL-Vid"
> framing, whose headline novelty leaned on **gold segment spans**. That dependence is **DROPPED**:
> only ONE dataset (HateClipSeg) has open, downloadable spans, and the user rejected the resulting
> single-dataset trap (both HateMM and MultiHateClip spans turned out absent locally — see
> `ITERATION_LOG.md` Phase-1 STEP-0 blocks). Cross-lingual and gold-span-segment DEPENDENCE are
> dropped as novelty axes.
>
> **2026-07-02 CONSOLIDATION note (Phase-3 results in — headline demoted/promoted).** The 2026-07-02
> confirmed-revision proposed **TWO** annotation-free mechanistic deltas as co-headline novelty:
> (1) multi-granularity / segment-retrieval and (2) cross-dataset updatable kNN memory. Phase-3
> ablations (`ITERATION_LOG.md` Phase-3 iter1/iter2 + Phase-3b) now force an honest split:
> - **(1) Multi-granularity / segment-retrieval is DEMOTED from headline novelty to honest ANALYSIS.**
>   Across `full` / `driftneg` / `milmax` at λ=0.5 the segment term **sign-flips by language**
>   (MHC-EN +0.015 F1 for `full`, MHC_zh −0.066; `milmax` reverses it: MHC_zh +0.017, MHC-EN −0.102):
>   **no single seg_mode is ≥ baseline on both languages**, and no config crosses acc>0.85. Diagnosed
>   as noisy MIL positives on tiny dev sets. It is now reported as a **negative/analysis result**, not
>   a claimed win.
> - **(2) Cross-dataset updatable kNN memory is PROMOTED to the VALIDATED load-bearing novelty vs
>   MoRE.** Phase-3b showed the learned fused-embedding space is **swappable at inference** and beats
>   the majority baseline on **5/6 informative cross cells**, lagging in-domain by only ~0.04–0.09
>   macro-F1 on working cells (degrading, not collapsing, cross-lingually; failing only into MHC as a
>   target). **MoRE's trained MoE head structurally cannot re-point at a new support set** — this is
>   the delta that survived validation.
> - **The MLLM stays a validated performance LEVER, not the novelty** (beats CLIP on HateMM + MHC-EN;
>   loses on MHC_zh; ≈ on ImpliHateVid — a mixed 4-set picture, honestly the encoder lever not the
>   contribution).
> Prior useful content (novelty table structure, method wiring against `src/`, protocol) is preserved.

This document is the Iteration-1 method decision: chosen method, novelty table, method design,
MLLM-difference statement, performance protocol, and experiment roadmap.

---

## 0. Chosen method (confirmed 2026-07-02)

**The current method we already run = plain RGCL on video** (frozen encoder → learned
retrieval-guided contrastive InfoNCE over FAISS-mined pseudo-gold-positive / hard-negative →
kNN-vote inference head). That bare port carries RGCL's novelty, **not ours** (project rule: RGCL
is inspiration). After Phase-3 validation, our load-bearing contribution is **ONE validated
mechanistic difference vs plain RGCL** — an update-stable / cross-dataset kNN memory MoRE cannot
provide — plus a validated MLLM encoder lever, with multi-granularity retained as an **honest
analysis** (tried, not a robust win):

**HEADLINE NOVELTY (VALIDATED) — Update-stable / cross-dataset updatable kNN memory.**
Because inference is a **kNN vote over a labeled memory bank**, the classifier can be updated by
**ADDING new labeled exemplars at test time WITHOUT retraining** (evolving-hate adaptation), and the
memory can be **SWAPPED** for cross-dataset transfer. **MoRE's trained MoE head cannot do either**
(MoRE explicitly flags evolving hate as a need it does not deliver; its decision is baked into a
trained MoE head that cannot be re-pointed at a new support set at inference). Phase-3b **validated**
this: swapping the memory bank of a foreign head beats the target's majority baseline on **5/6
informative cross cells**, lagging in-domain by only **~0.04–0.09 macro-F1** on working cells,
degrading (not collapsing) cross-lingually and failing only into MHC as a target. This is the delta
that survived hostile validation — a genuine **capability** MoRE structurally lacks.

**ANALYSIS (tried, NOT a robust win) — Multi-granularity, annotation-free temporal retrieval.**
Plain RGCL retrieves and contrasts at a SINGLE **whole-instance** granularity: a whole video is
one mean-pooled vector, indexed once in one FAISS index. We added a **FINE** granularity: split each
video into **AUTO sub-clips** (uniform temporal windows; **NO gold spans**), embed each sub-clip,
build a **SECOND FAISS index**, and run retrieval-guided contrastive at **BOTH** whole-video AND
sub-clip level, with the within-video **benign sub-clip of a hateful video** mined **without gold
labels** (MIL / dissimilarity heuristic) as a **"drifting" hard negative**. **HONEST RESULT (Phase-3
iter1/iter2):** the segment term **sign-flips by language** — `full` λ=0.5 gives MHC-EN +0.015 F1 /
MHC_zh −0.066; `milmax` reverses it (MHC_zh +0.017 / MHC-EN −0.102); `driftneg` is a near-no-op on
EN and still below baseline on ZH. **No seg_mode is ≥ baseline on both languages, none crosses
acc>0.85**, and the effect is diagnosed as **noisy MIL positives** on tiny dev sets (MHC dev=80,
MHC_zh dev=78). It remains structurally meme-impossible (a meme has no time axis), so it is kept as a
**mechanistic-analysis / negative result** and a diagnostic axis — **not a headline novelty claim**.
HateClipSeg gold spans (open on the Social-AI-Studio github, 435 videos / 11,714 segments) would be a
validation slice only, not required.

**MLLM = performance LEVER, NOT the novelty (VALIDATED 2026-07-02).**
RAMF / HVGuard already do frozen-VLM-features → head, so the MLLM is not a field contribution. We
use a **FROZEN Qwen2.5-VL-7B** as a multimodal encoder (Config MLLM-A hidden-state pooling; optional
per-sub-clip neutral description as a fine retrieval key). On the **identical RGCL head**, the full
4-dataset picture is **mixed** (a lever, not a uniform win):
- **HateMM:** 0.817 M-F1 / 0.828 acc (CLIP) → **0.861 / 0.870** (MLLM) — crosses acc>0.85 and beats
  MoRE 0.8235 M-F1 (approaches MM-HSD 0.874).
- **MHC-EN (MHClip-Y):** 0.711 / 0.783 (CLIP) → **0.738 / 0.789** (MLLM) — modest, still <0.85.
- **MHC_zh (MHClip-B):** 0.771 / 0.805 (CLIP) → **0.741 / 0.792** (MLLM) — **MLLM LOSES** (−0.029 F1);
  the earlier "MLLM edges CLIP on Chinese" read was WRONG (not apples-to-apples) and is owned/corrected.
- **ImpliHateVid:** 0.910 / 0.910 (CLIP) → **0.900 / 0.900** (MLLM) — ≈ tie (−0.010 F1).

So MLLM wins on 2/4 (HateMM clearly, MHC-EN modestly), loses on ZH, ties on ImpliHateVid — exactly a
**performance lever to ablate, not a contribution**. Frozen, ~7M-param head, **no LoRA / no
generation** this iteration. Difference vs **RA-HMD** (LoRA-SFT, memes) = frozen + video; vs
**reasoning-VLMs** (MARS / HVGuard / IARE, prompt-to-reason) = pure encoder + kNN over exemplars (no
verdict generation).

**DROPPED / DEMOTED (confirmed):**
- **Gold-span-segment DEPENDENCE** — a single-dataset trap (only HateClipSeg has open spans).
  Auto-segmentation replaced it, but see below.
- **Multi-granularity / segment-retrieval as HEADLINE NOVELTY (DEMOTED 2026-07-02 consolidation).**
  Delivered annotation-free via auto-segmentation, but Phase-3 ablations show it sign-flips by
  language and never crosses 0.85 → kept as **honest analysis / negative result**, not a claimed win.
- **Cross-lingual as a novelty axis.** Temporal / audio / implicit remain **evaluation slices**, not
  novelty.

**What we do NOT claim:** MLLM use, M-RoPE, or audio-prosody as standalone novelty (supporting
ablations); **multi-granularity as a headline win** (analysis only). We do not claim >85% on 3-class.

---

## 1. Novelty vs closest SOTA

Field = hateful/harmful VIDEO detection only (memes/text = inspiration, per project scoping).
"OUR" = frozen-MLLM/CLIP encoder → learned retrieval-guided contrastive embedding (InfoNCE over
FAISS-mined pseudo-gold-pos/hard-neg at whole-video granularity) → **update-stable kNN-vote** memory
(test-time-updatable, cross-dataset-swappable) — **the validated headline delta**. Auto sub-clip /
multi-granularity retrieval is an **analysis axis** carried in the table but no longer claimed as a
win (Phase-3: sign-flips by language, no consistent gain).

| Closest SOTA | What they do | Uses MLLM? | Retrieval? | Contrastive? | kNN inference? | OUR DELTA vs them |
|---|---|---|---|---|---|---|
| **plain RGCL (video)** — the "current method" we run (mei2023) | frozen encoder → retrieval-guided contrastive + kNN, **SINGLE whole-instance granularity**, static memory | No (CLIP) / lever (MLLM) | Yes — whole-instance only | Yes | Yes, but memory used as fixed | **VALIDATED delta:** **update-stable / cross-dataset kNN memory** — test-time exemplar ADD without retrain + memory SWAP for transfer, which plain RGCL treats its memory as fixed and never exploits (Phase-3b: beats majority on 5/6 cross cells, lags in-domain only ~0.04–0.09 F1). **Analysis axis (NOT a claimed win):** multi-granularity annotation-free sub-clip retrieval — a SECOND FAISS index over AUTO sub-clips with a within-video drifting benign sub-clip as hard negative (structurally meme-impossible), tried but Phase-3 shows it sign-flips by language with no consistent gain. |
| **MoRE** (lang2025, WWW25) | retrieval-FOR-experts: frozen weighted-cosine video retriever → attention experts (BHAN) + MoE soft-router; **all BCE** | No (benchmarks Qwen2-VL/LLaVA-OV) | Yes — **frozen** cosine, whole-video only | No — BHAN is attention, no InfoNCE | **No** — trained MoE head | **VALIDATED delta:** **kNN-vote head that is test-time-updatable + memory-swappable** — the evolving-hate + cross-dataset adaptation MoRE's trained MoE head *structurally cannot* do (its decision is baked into the MoE head; it flags evolving hate as an unmet need). Phase-3b validated the memory swap (5/6 cross cells above majority). Plus **learned** retrieval space (InfoNCE) not frozen cosine and **real contrastive** loss not BCE-over-attention. (Multi-granularity sub-clip retrieval = analysis axis, no consistent win — not part of the claim.) Head-to-head on MoRE's exact binary protocol. |
| **RA-HMD / LMM-RGCL** (mei2025) | LoRA-SFT an LMM, then retrieval-guided contrastive over the LMM's OWN embeddings + kNN — **MEMES** | Yes — **LoRA-tuned**, at inference | Yes — learned | **Yes** — the recipe we adapt | **Yes** | **Video** (first RGCL-in-video) with a **cross-dataset / test-time-updatable kNN memory** RA-HMD does not exercise; **frozen** MLLM (no LoRA-SFT) → ~100× fewer trainable params, no next-token SFT stage (RA-HMD's own analysis shows SFT can hurt robustness). (Analysis axis: multi-granularity sub-clip retrieval — a within-video drifting benign sub-clip as hard negative is **structurally impossible in a static meme** — tried but no consistent win.) |
| **MultiHateLoc** (sun2025, WWW26) | frame-level cross-modal contrastive, **same-video/same-timestamp** positives; MIL temporal localization | No | No (temporal alignment, not exemplar retrieval) | **Yes** (same-timestamp) | No | Positives chosen by *timestamp co-occurrence*; ours are **retrieval-mined across the corpus** (nearest labeled sub-clip = pseudo-gold positive; within-video benign sub-clip = drifting hard negative) — a retrieval + updatable-kNN head they lack, and our sub-clips are **auto-segmented, annotation-free** (no timestamp supervision). |
| **RGCL** (mei2023) — the meme origin | frozen-CLIP retrieval-guided contrastive + kNN — **MEMES** | No (frozen CLIP) | Yes | Yes | Yes | Same as "plain RGCL (video)" row: by project rule RGCL is *inspiration*, so the bare port carries RGCL's novelty. Our validated win = **update-stable / cross-dataset kNN memory** + optional MLLM encoder ablation; multi-granularity sub-clip retrieval (meme-impossible) is carried as analysis, not a claimed win. |
| **MM-HSD** (cspedessarrias2025) | multimodal fusion (wav2vec2-xlsr audio + OCR-as-query cross-modal attn); HateMM video-level SOTA **0.874 M-F1** | No | No | No | No | Performance *bar*, not a novelty threat: no retrieval, no contrastive, no kNN. 0.874 is the number to approach on HateMM, not a rival. |
| **ImpliHateVid / TCL** (rehman2025) | two-stage **SupCon** (positives/negatives by CLASS LABEL), per-modality → cross-encoder | No | No | **Yes** (SupCon) | No | Positives/negatives by *class label*; ours by **retrieved nearest labeled exemplars/sub-clips** (confusable near-duplicate + within-video drifting benign sub-clip). Distinct contrastive signal + an updatable-kNN head they lack. |
| **MARS / HVGuard / IARE** (yang2026 / jing2025 / lu2026) | **PROMPT the (M)LLM to REASON** (CoT / dual-hypothesis / DPO rationale) and classify from generated text | Yes — **reasoner/generator** at inference | HVGuard fuses rationale via MoE; none retrieve labeled exemplars | IARE = DPO preference contrast (not InfoNCE) | No | MLLM as a **pure encoder** — no generation, no CoT, no DPO. "Explanation" = the retrieved kNN neighbors (faithful-by-construction), not a possibly-hallucinated rationale; one forward + FAISS lookup vs multi-stage generation. |

**Defensible one-line novelty (post-Phase-3 consolidation):** *first hateful-VIDEO method to give the
retrieval-guided-contrastive + kNN head an **update-stable, cross-dataset-swappable kNN memory** that
MoRE's trained MoE head structurally cannot provide (test-time exemplar adds + memory swap for
evolving-hate and cross-dataset transfer, validated Phase-3b: above majority on 5/6 cross cells), with
the MLLM as a validated frozen-encoder performance lever. Multi-granularity annotation-free sub-clip
retrieval — a second FAISS index over auto sub-clips with the within-video benign sub-clip mined (no
gold labels) as a drifting hard negative, structurally impossible for the closest meme twin
(RA-HMD/RGCL) — is retained as an honest analysis axis (Phase-3: sign-flips by language, no consistent
win), not a headline claim.*

---

## 2. Method design (implementable against /data/jehc223/RGCL/src)

### 2.1 Encoder front-end (frozen; run once offline, cached like current CLIP/MLLM tensors)

Both configs produce the two-stream `{ids, img_feats, text_feats, labels}` .pt cache that
`src/data_loader/dataset.py:load_feats_split` already consumes (RGCL core untouched):

- **Config CLIP (baseline / lower bound):** frozen CLIP ViT-L/14-336, 8-frame mean-pool video
  (`img_feats`, 1024) + title+transcript text tower (`text_feats`, 768). Unchanged — the ablation
  floor for the multi-granularity delta.
- **Config MLLM-A (hidden-state; VALIDATED lever):** frozen **Qwen2.5-VL-7B-Instruct**, bf16,
  `torch.no_grad`, `output_hidden_states=True`. Extractor
  `src/utils/generate_VideoMLLM_embedding_HF.py` (already built, Phase 2), reusing the decord/PyAV
  8-frame sampler.
  - `img_feats` (Dv=3584): 8 frames + fixed neutral instruction → mean of last-layer hidden states
    over the vision+instruction span, L2-norm.
  - `text_feats` (Dt=3584): same frames + title + transcript + fixed analytic instruction → mean
    over the assistant-header tail span, L2-norm.
- **Sub-clip keys (NEW, Delta 1):** each auto sub-clip embedded the same way (its own frame window),
  giving a fine-granularity vector. Optionally, per-sub-clip **neutral description** re-embedded as a
  fine retrieval key (frozen Qwen2.5-VL description, `do_sample=False`) — an *optional* key, never a
  novelty in itself.

Audio-prosody (ablation add-on, off in the core config): frozen wav2vec2-xlsr / CLAP pooled vector
as a 3rd stream (gap_map G5: audio is the weakest modality).

### 2.2 Head + fusion (reuse `classifier_hateClipper`, unchanged shape)

`classifier_hateClipper(image_dim=Dv, text_dim=Dt, num_layers=3, proj_dim=1024, map_dim=1024,
fusion_mode='align')`. `run_rac.py` reads dims from the cache, so Dv=Dt=3584 (MLLM) or 1024/768
(CLIP) needs **no code change** — proven in Phase 2. Trainable params ≈ 6-8M. Returns `(logit, L2-norm embed)`.

### 2.3 Multi-granularity retrieval-guided contrastive loss (reuse `src/model/loss.py`, add fine granularity)

- **Whole-video (unchanged, "current method"):** per-epoch FAISS refresh over the head's own
  projected embeddings; `dense_retrieve_hard_negatives_pseudo_positive` mines, per anchor, the
  same-label nearest neighbor (pseudo-gold positive) and opposite-label nearest neighbors (hard
  negatives). Objective = retrieval-guided InfoNCE/triplet + in-batch negatives, hybridized with BCE
  (`hybrid_loss=True`, `ce_weight≈0.3`).
- **Sub-clip granularity (NEW — Delta 1, annotation-free):**
  1. **Auto-segment** each video into sub-clips: **uniform temporal windows** first (e.g. K equal
     spans over the frame timeline — zero dependencies, general on all 4 datasets); **shot-boundary**
     (PySceneDetect, optional) later as a refinement. **No gold spans.**
  2. Embed each sub-clip → a **second FAISS index** over sub-clip embeddings.
  3. Mine, per sub-clip anchor: pseudo-gold positive = nearest same-label sub-clip across the corpus;
     hard negative = the **within-video benign sub-clip of a hateful video** (the drifting hard
     negative), identified **without gold labels** via a **MIL / dissimilarity heuristic** (the
     sub-clip least consistent with the video's hate label, or lowest predicted-hate score), plus the
     nearest opposite-label sub-clip.
  4. Additive loss: `L = L_video_RGCL + λ_sub · L_subclip_RGCL`. On any dataset λ_sub is always
     available (auto-seg), so it never degrades to 0; **HateClipSeg gold spans** are used *only* to
     VALIDATE that the auto-mined drifting negatives agree with annotated hate spans.

### 2.4 Update-stable kNN inference + memory operations (Delta 2; reuse `evaluate_rac.py:retrieve_evaluate_RAC`)

FAISS index over train (+val, leak-safe per MoRE) embeddings; K≈10-20 majority/similarity vote.
Report **both** the kNN-vote metric (primary) and the BCE-head metric; select the epoch by
Val_Retrieval acc/ROC (fixes the baseline's best-model bug).

Delta-2 mechanisms (what plain RGCL / MoRE do not exploit):
- **Test-time exemplar ADD (no retrain):** append new labeled exemplars to the memory bank at
  inference; measure the accuracy lift on an evolving-hate held-out slice.
- **Cross-dataset memory SWAP:** train the embedding on dataset A, replace the memory bank with
  dataset B's labeled exemplars, evaluate on B — no retraining. Demonstrated across all 4 datasets.
- **Update-stability:** ensure the embedding is stable enough that added/swapped exemplars remain
  well-placed (e.g. bounded drift across epochs / frozen-embedding memory ops).

### 2.5 Trained vs frozen

- **FROZEN:** Qwen2.5-VL (or CLIP), wav2vec2/CLAP, ASR (Whisper/FunASR), any sentence encoder. Zero
  MLLM trainable params. **No LoRA, no SFT this iteration.**
- **TRAINED:** only the ~6-8M HateClipper head (projections + align-fusion MLP + logit head +
  retrieval-embedding head). FAISS/kNN have zero trainable params. Auto-segmentation has zero
  trainable params.

### 2.6 Code deltas (localized, honest)

1. Encoder extractors — CLIP (existing) + MLLM (`generate_VideoMLLM_embedding_HF.py`, done Phase 2).
2. **Auto-segmenter** (uniform windows; PySceneDetect optional) + **sub-clip embedder** → cached
   sub-clip .pt (new).
3. One additive **sub-clip-RGCL** loss term in `compute_loss` + a **second FAISS index** in the
   per-epoch refresh, with the **annotation-free drifting-negative miner** (MIL/dissimilarity) —
   small edit to `loss.py` / `retrieval.py` (new).
4. **Memory-op harness** for test-time ADD + cross-dataset SWAP (new eval path around the kNN vote).
5. (If 3-class chosen for a slice) change the hardcoded 1-logit head to N-logit + N-way kNN — scoped
   only to the MHClip 3-class secondary.

Everything else (align-fusion, whole-video mining, kNN eval) is reused verbatim.

---

## 3. MLLM difference statement (validated 2026-07-02)

Our MLLM is a **pure multimodal encoder**: a frozen Qwen2.5-VL-7B maps (8-frame video / sub-clip +
title + transcript [+ optional audio]) to a single pooled hidden state (or an optional neutral cached
description re-embedded as a fine key), which our tiny RGCL head shapes into a retrieval embedding —
we **never sample a verdict, chain-of-thought, or rationale**. This differs from **RA-HMD / LMM-RGCL
(memes)** in that (a) we operate on **video** with a **cross-dataset / test-time-updatable kNN memory**
(the validated delta) — and can add an **auto sub-clip** retrieval unit a single static meme cannot
have (carried as analysis, not a claimed win after Phase-3), and (b) we keep the MLLM **frozen** this
iteration (no LoRA, no next-token SFT — which RA-HMD's own analysis shows can hurt robustness), training
only a ~6-8M head at ~100× fewer trainable parameters. (LoRA-SFT is under test in Iteration 2.)
It differs from **reasoning-VLMs (MARS / HVGuard / IARE / RAMF)** in that they **prompt the model to
GENERATE reasoning** (CoT / dual-hypothesis / DPO rationale) and classify from that text, whereas we
consume the model's internal representation and decide by **kNN vote over retrieved labeled
exemplars** — our explanation is a set of real neighbor videos/sub-clips (faithful-by-construction),
not generated text that can hallucinate, at one forward pass + a FAISS lookup.

**Validation (Phase 2, identical head/loss/split, val-selected epoch):** frozen Qwen2.5-VL-7B
hidden-state features beat frozen-CLIP features on the identical RGCL head on BOTH make-or-break
datasets — HateMM 0.8172 → **0.8606** M-F1 (0.8279 → **0.8698** acc; crosses 0.85, beats MoRE 0.8235),
MHC-EN 0.7113 → **0.7378** M-F1 (0.7826 → **0.7888** acc; modest, still <0.85). The review's central
risk ("MLLM may not beat CLIP") is **retired**.

---

## 4. Performance protocol

**Decision (unchanged): lead with BINARY harmful-vs-normal; report 3-class as an honest secondary.**

- **Primary — BINARY (offensive ∪ hateful = 1):** the protocol MoRE, MM-HSD, RAMF, HVGuard all
  report, and the one our baseline runs. Metrics: **macro-F1 / precision / recall + accuracy**
  (macro-F1 is the field headline). We reproduce **MoRE's exact binary merge + train+val
  memory-bank** for a fair head-to-head (segment/sub-clip memory indexes train/val only; no test
  leakage). **Target acc>0.85 on HateMM (MET with MLLM) + MHClip (still owed)** — the
  multi-granularity + updatable-memory levers target exactly this MHClip gap.
- **Secondary — 3-class (Hateful / Offensive / Normal) for MHClip:** reported **honestly** as a
  harder slice (field ceiling ~0.63 EN / 0.50 ZH M-F1). We do **NOT** claim 85% here; we claim
  **beating 0.63 EN / 0.50 ZH**. Requires the N-logit head change (§2.6).

**Targets (binary macro-F1 / acc), with the levers — real Phase-2 numbers where measured:**

| Dataset | CLIP baseline (ours) | **MLLM (Phase 2 FINAL, measured)** | SOTA to beat | Target | Remaining lever |
|---|---|---|---|---|---|
| **HateMM** (EN) | 0.8172 M-F1 / 0.8279 acc | **0.8606 / 0.8698** — crosses 0.85, beats MoRE 0.8235 | MoRE 0.8235; **MM-HSD 0.874** | close gap to MM-HSD | audio + OCR (multi-granularity = analysis only) |
| **MHC-EN / MHClip-Y** (EN) | 0.7113 / 0.7826 (**leak-fixed 549-row train**) | frozen Qwen **0.7378 / 0.7888** — modest, still <0.85. **LoRA (selEp26): 0.6916 / 0.7516 — REGRESSES** below both frozen CLIP (−0.020 F1 / −0.031 acc) and frozen Qwen (−0.046 F1 / −0.037 acc); gap to 0.85 = **0.098** | MoRE 0.7519 | acc>0.85, beat MoRE | **updatable memory** (LoRA-SFT did NOT help EN — regressed; multi-granularity also sign-flips) |
| **MHC-ZH / MHClip-B** (ZH) | 0.7706 / 0.8054 (already beats MoRE 0.7475) | frozen Qwen **0.7412 / 0.7919** — MLLM LOSES (−0.029 F1, epoch-0 caveat). **LoRA (selEp20): 0.8023 / 0.8322 — BEST-EVER ZH**, beats frozen-CLIP floor (+0.032 F1 / +0.027 acc, apples-to-apples); gap to 0.85 = **0.018** | MoRE 0.7475; HVGuard ZH-bin 0.822 | acc ~0.83-0.85 (aspirational) | LoRA-SFT is the ZH win (closest to 0.85 yet, still ~1.8 acc pts short) |
| **ImpliHateVid** (EN) | 0.9101 / 0.9102 (beats TCL 0.8773, clears 0.85) | **0.9002 / 0.9002** — ≈ tie (−0.010 F1), still clears 0.85 | TCL 0.8773 | hold/exceed | implicit-exemplar retrieval |

> **Baseline correction to record (2026-07-02):** the MHClip-Y (MHC-EN) CLIP floor is **0.711 M-F1 /
> 0.783 acc** on the **LEAK-FIXED 549-row train** — NOT the earlier-quoted **0.622 / 0.745**, which
> was a **leaked-550 unstable-epoch artifact** (id `k9OtaMbK0Ac` in both train and test; different
> unstable epoch on a tiny 80-sample val). All MHC-EN deltas are reported against the clean 549 floor.

**Honesty caveats:** ">85% on BOTH HateMM AND MHClip simultaneously" is **not guaranteed** — MHC-EN
still sits at 0.789 acc after the MLLM lever, and the multi-granularity delta did **NOT** close that
gap (Phase-3: sign-flips by language, no config crosses 0.85). MHClip-ZH >0.85 is aspirational
(~~ASR-bound~~ — see erratum below), and frozen Qwen actually *lost* to CLIP on ZH — so the **remaining performance lever for
the MHClip gap is LoRA-SFT of the encoder** (deferred to Iteration 2; in-flight as of 2026-07-02, see
below). The **defensible spine**: beat MoRE on the binary protocol + approach MM-HSD on HateMM at a
fraction of trainable/inference cost, resting on the **one validated mechanistic delta MoRE/RGCL
lack** — a **test-time-updatable + cross-dataset-swappable** kNN memory (Phase-3b) — with multi-
granularity reported as honest analysis, not a claimed win. Cross-paper numbers untrusted → re-run all
baselines on our split.

> **ERRATUM 2026-07-28 (propagated from `refine-logs/LITSWEEP3_ZH_SPECIFIC.md:18-37`, F77 / commit
> `d4af64b`).** "MHClip-ZH >0.85 is aspirational (**ASR-bound**)" is **withdrawn.** The deployed ZH text
> stream is **not** the Whisper ASR — it is the **Bilibili description/metadata** field (`gt["text"]`),
> median **106 Chinese characters** (train 106 / val 108.5 / test 105), with 42 % of train rows carrying
> literal `<em class="keyword">…</em>` markup. The Whisper ASR lives in a separate, **non-deployed** file
> (`data/ASR/MHC_zh/*_asrK4_whisper-large-v3.jsonl`). The related ledger figure "ZH transcripts median 4
> words" is a **whitespace-split artefact** (Chinese has no inter-word spaces, so `text.split()` is
> meaningless). Measured ZH wall as of round 6: **78-dev val-selection noise** (+0.0246 vs the +0.030 bar)
> plus representation saturation (LoRA-Qwen ZH text-AUC 0.925) — not a transcription ceiling. The
> subsequent LoRA-SFT conclusion in this paragraph is unaffected and was independently confirmed (B3/F45).

---

## 5. Experiment roadmap

Ordered; **[SLURM]** = one sbatch job (no `--time`), **[login]** = login-node prep, **[cpu]** = head
training (faiss-cpu, `--Faiss_GPU False`). All heavy work via subagent/workflow, never the main
conversation.

**Phase 0 — protocol & baseline alignment (DONE, 2026-07-01).** Fixed val-selection bug; re-ran the
honest frozen-CLIP-RGCL binary baseline on all 4 datasets; leakage audit (found + fixed MHC-EN leak).

**Phase 2 — MLLM encoder integration (DONE, 2026-07-02).** Built frozen Qwen2.5-VL-7B extractor;
full 4-dataset ablation CLIP-RGCL vs MLLM-RGCL — **MLLM wins on HateMM (crosses 0.85, beats MoRE) +
MHC-EN (modest); LOSES on MHC_zh (−0.029 F1); ≈ ties ImpliHateVid.** A performance lever, mixed
across the set, not a uniform win.

**Phase 3 — multi-granularity ablation (DONE, 2026-07-02) — HONEST NEGATIVE.** Auto-segmenter +
second FAISS index + additive `L_subclip_RGCL` with the annotation-free drifting-negative miner were
built and run. Make-or-break ablation whole-video vs +multi-granularity, plus `driftneg` / `milmax`
seg-modes, on MHC (EN) + MHC_zh: the segment term **sign-flips by language** (no seg_mode ≥ baseline
on both), never crosses 0.85, diagnosed as noisy MIL positives → **DEMOTED to analysis, not a claimed
novelty.**

**Phase 3b — cross-dataset kNN memory transfer (DONE, 2026-07-02) — VALIDATED.** Built
`src/eval_cross_dataset.py`; swapping a foreign head's memory bank beats the target majority on **5/6
informative cross cells**, lagging in-domain only ~0.04–0.09 macro-F1 (fails only into MHC as target;
degrades, not collapses, cross-lingually). **The headline capability MoRE structurally lacks —
validated.**

**Phase 3 — REMAINING owed work.**
3.4 [cpu] **Update-stable / cross-dataset memory demos (headline delta):** (a) test-time exemplar ADD
    (evolving-hate slice, no retrain) accuracy lift; (b) memory SWAP matrix done (Phase-3b) — extend
    to test-time ADD.
3.5 [cpu] **HateClipSeg validation slice:** confirm auto-mined drifting negatives agree with the open
    gold spans (analysis diagnostic, not required by the method).
3.6 [cpu] Ablation ladder (each a row, identical split): (a) MLLM vs CLIP (done); (b) +audio; (c)
    +OCR; (d) retrieval-contrastive vs BCE-only (MoRE-style); (e) SupCon-by-label vs retrieval-mined;
    (f) whole-video vs +sub-clip (done — negative); (g) kNN-vote vs BCE-head; (h) static vs updatable
    memory.
3.7 [cpu] 3-class MHClip secondary (N-logit head + N-way kNN); M-F1 vs 0.63/0.50.
3.8 Multi-seed / CI reporting; final head-to-head vs MoRE / MM-HSD.

**Iteration 2 — LoRA-SFT of the MLLM encoder (DONE 2026-07-02) — LANGUAGE-SPLIT RESULT: helps ZH,
hurts EN; neither MHClip split crosses 0.85.** Because frozen Qwen did not close the MHClip gap (and
lost on ZH), LoRA-SFT of the Qwen2.5-VL encoder was run as a performance lever — final predictions
still come from our RGCL contrastive + kNN head (keeps us distinct from generative reasoning-VLMs
MARS/HVGuard). These are the **FIRST LoRA-adapted RGCL metrics on disk** (trainlogs 2723309 MHC,
2794237 MHC_zh); they supersede the earlier "no LoRA-adapted RGCL metrics exist" status note. Same
val-selection rule as all other runs (warmup-floored epochs≥5, max Val_Retrieval acc, tie-break Val
roc, Test_Retrieval kNN head).

- **MHC_zh (ZH), LoRA selEp20 = 0.8023 M-F1 / 0.8322 acc:** LoRA **HELPS** — beats the warmup-consistent
  frozen-CLIP floor (0.7706 / 0.8054) by **+0.032 M-F1 / +0.027 acc** (clean apples-to-apples), and is
  the **best ZH number recorded** (above the prior ZH best, seg-mode milmax 0.7875 / 0.8255). Also
  above frozen-Qwen non-LoRA (0.7412 / 0.7919), though that ZH frozen-Qwen baseline is an epoch-0
  checkpoint (not warmup-consistent, provisional) so treat that magnitude cautiously; the direction
  matches the CLIP comparison. **Gap to 0.85 = 0.018 — closest any config has come on ZH, still short.**
- **MHC-EN (EN), LoRA selEp26 = 0.6916 M-F1 / 0.7516 acc:** LoRA **REGRESSES** — worse than both the
  frozen-CLIP floor (0.7113 / 0.7826; −0.020 M-F1 / −0.031 acc) AND frozen-Qwen non-LoRA (0.7378 /
  0.7888; −0.046 M-F1 / −0.037 acc) under identical head/selection. LoRA-SFT of the encoder moved EN
  **further** from 0.85, not closer. **Gap to 0.85 = 0.098** — EN remains the hardest, unsolved slice
  (hate/offensive confusion, tiny 161-sample test).

**Does MHClip cross acc>0.85? NO — neither language.** ZH (0.8322) and EN (0.7516) both fall short.
acc≥0.85 is met ONLY on HateMM (frozen Qwen, 0.870) and ImpliHateVid (~0.90); the MHClip 0.85 target
stays **OPEN on both languages after LoRA-SFT**. LoRA is a **language-inconsistent lever** (best-ever
ZH, but hurts EN — echoing the earlier seg-mode finding that EN and ZH respond oppositely), confirming
it as a **performance lever, not novelty**. The headline novelty is unchanged: the **cross-dataset /
test-time-updatable kNN memory** (validated Phase-3b) remains the load-bearing contribution; LoRA does
not touch that claim. Retrieval-augmented teacher distillation and rationale-as-retrieval-key remain
deferred.

---

## 6. Open decisions for the user (genuine forks)

1. **Auto-segmentation granularity:** uniform K-window (zero-dep, general — recommended first) vs
   shot-boundary (PySceneDetect; needs `scenedetect`+`opencv` install, more faithful, adds a
   heuristic). Ship uniform first, add shot-boundary as an ablation.
2. **Drifting-negative miner:** MIL (lowest predicted-hate sub-clip) vs pure feature-dissimilarity
   (farthest-from-video-centroid sub-clip) — validate both against HateClipSeg gold spans.
3. **MHClip-ZH / ImpliHateVid MLLM head-training:** run now (extractions done) to complete the 4-set
   MLLM table before Phase 3, or fold into Phase 3.
4. **Protocol emphasis:** binary-only headline vs binary + 3-class MHClip secondary (needs N-logit).
5. **Cross-dataset SWAP scope:** all 4×4 pairs vs the EN-anchor trio only, for the Delta-2 demo.
