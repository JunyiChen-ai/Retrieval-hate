# LIT SURVEY: MLLM-as-embedding / adapted-encoder paradigm — the family closest to ours

**Author:** lit-survey agent (CPU-only; WebSearch/WebFetch; ZERO GPU/SLURM/Modal; `state/` untouched).
**Date:** 2026-07-24 NZST.
**Mission (user-ordered):** survey the paradigm CLOSEST to ours — **using (M)LLMs/LVLMs as adapted feature encoders for discriminative/classification tasks** — hunting *novel borrowable mechanisms*, 2024–2026 emphasis. Every candidate carries a mandatory isomorphism check against `state/directions_tried.json`, `state/findings.jsonl` (F1–F67), and `refine-logs/REDTEAM_EXTERNAL_FAMILIES.md`, an in-box legality verdict, an honest performance prior, and a D7-novelty read.

---

## 0. Verdict up front

**The MLLM-embedding field's gains come from exactly four levers, and this project has already measured or arithmetic-capped three of them.** The SOTA recipe to turn a generative VLM into a discriminative embedding model (E5-V, VLM2Vec/-V2, GME, NV-Embed, LLM2Vec, UniME, mmE5) is always some subset of: **(i) a contrastive adaptation objective**, **(ii) hard-negative mining**, **(iii) a better readout/pooling than naive mean/last-token**, **(iv) synthetic/scaled training data**. Mapping onto our ledger:

- **(i) contrastive objective** — the *joint* form is P9b (KILLED, redistribution law); the *decoupled* form is genuinely un-run but is bordered by P9b and arithmetic-capped on EN (F44/F55).
- **(ii) hard negatives** — cand-2 measured TIE (F56); head-level global-hardest already deployed (C3geo/F25).
- **(iii) readout/pooling** — **never varied in this project.** Law I (F37/F66) says data-driven *pooling* is lossless on our reps and the headroom is selection-locked, but the *token-position / layer / prompt* the video-level vector is read from is an untouched axis.
- **(iv) synthetic data** — AUG (F60 dominated) + single-dataset-train veto.

So the **one genuinely un-enumerated axis inside this paradigm is the READOUT** (which token, which layer, which prompt, echo-repetition), plus **one genuinely un-tried architectural MECHANISM — bidirectional-attention surgery** (F35 *diagnosed* the causal-prefix limitation but no run ever *removed* the causal mask). Neither has a strong prior on a **new** dataset — EN is label-limited (proven at five levels: F44/F50/F55/F58/F65), HateMM already passes (F53), so the realistic target for every candidate here is **hardening the marginal ZH val-selected leg** (the 78-dev selection tax, F45) and a possible **D7-novelty upgrade**. The survey does **not** surface a mechanism that simultaneously clears D7 and carries a strong performance prior — consistent with the terminus. It surfaces two cheap-to-falsify ZH-hardening probes and one novel mechanism worth a user ruling.

An important **honest cross-check for search-space #5 (video):** the SOTA *video*-MLLM-embedding works (VLM2Vec-V2, VidVec) use **no special temporal operator** — they feed frame tokens jointly to the LLM and read a *single* embedding via one-word/intermediate-layer/last-token readout. This independently **corroborates** our F35/F37/F67 kills (temporal-pooling and frame-density are not the lever); the borrowable video insight is the *readout*, not the frame handling.

---

## 1. The paradigm map — how each 2024–2026 work converts a generative VLM into an embedding model

| Work | Verified cite | Backbone | Objective | Readout / pooling | Hard-neg / data |
|---|---|---|---|---|---|
| **PromptEOL** | Jiang et al., *Scaling Sentence Embeddings with LLMs*, Findings EMNLP 2024, [arXiv 2307.16645](https://arxiv.org/abs/2307.16645) | decoder LLM | none (zero-shot) or contrastive FT | **"…in one word:" prompt + last-token** — the one-word constraint forces the next-token to carry sentence meaning (else "The" wins) | — |
| **E5-V** | Jiang et al., 2024, [arXiv 2407.12580](https://arxiv.org/abs/2407.12580) | MLLM (LLaVA-NeXT) | **text-pair-only contrastive** (no image-text pairs → −95% cost) | **"…in one word:" prompt** unifies modalities into one text-space vector | text pairs |
| **VLM2Vec** | Jiang et al., ICLR 2025, [arXiv 2410.05160](https://arxiv.org/abs/2410.05160) | Phi-3.5-V / LLaVA-1.6 | **contrastive (InfoNCE), instruction-conditioned** | **last-token** | MMEB (20 train sets) |
| **VLM2Vec-V2** | Meng et al., 2025, [arXiv 2507.04590](https://arxiv.org/abs/2507.04590) | Qwen2-VL | InfoNCE, temp 0.02 | (feeds **8 uniform frames** jointly; single vector) | MMEB-V2 (video/doc) |
| **GME** | Zhang et al., 2024, [arXiv 2412.16855](https://arxiv.org/abs/2412.16855) | Qwen2-VL-2B/7B | contrastive | **last-token hidden state** | mined hard negatives |
| **NV-Embed** | Lee et al., ICLR 2025, [arXiv 2405.17428](https://arxiv.org/abs/2405.17428) | Mistral-7B | **two-stage contrastive instruction-tuning** (stage-2 drops in-batch negs) | **latent-attention pooling** (512 latents, 8 heads) + **bidirectional attention** (causal mask removed) | curated hard negs |
| **LLM2Vec** | BehnamGhader et al., 2024 (COLM), [arXiv 2404.05961](https://arxiv.org/abs/2404.05961) | decoder LLMs | **MNTP** (masked next-token pred) + unsupervised **SimCSE** | **bidirectional attention** + mean-pool | unsupervised |
| **VladVA** | Bica et al., CVPR 2025, [arXiv 2412.04378](https://arxiv.org/abs/2412.04378) | LLaVA-1.5 (LVLM) | **hybrid: contrastive (short caps) + autoregressive NTP (long descs)** | last-token | soft-prompt + LoRA PEFT |
| **UniME** | Gu et al., ACM MM 2025, [arXiv 2504.17432](https://arxiv.org/abs/2504.17432) | MLLM | **stage-1 textual-discriminative KD from an LLM text-embedder teacher** → stage-2 hard-neg instruction tuning | last-token | multi-hard-neg / false-neg filtering |
| **UniME-V2** | 2025, [arXiv 2510.13515](https://arxiv.org/abs/2510.13515) | MLLM | contrastive + **MLLM-as-a-Judge hard-neg mining** | last-token | judge-scored hard negs |
| **mmE5** | Chen et al., Findings ACL 2025, [arXiv 2502.08468](https://arxiv.org/abs/2502.08468) | Llama-3.2-11B-Vision | contrastive | last-token | **560K GPT-4o synthetic** samples |
| **Echo embeddings** | Springer et al., 2024, [arXiv 2402.15449](https://arxiv.org/abs/2402.15449); training-free variant [arXiv 2502.20726](https://arxiv.org/abs/2502.20726) | decoder LLM | none / contrastive | **repeat input twice, read 2nd occurrence** so early tokens see full context (fixes causal pooling *without* architecture change) | — |
| **VidVec** | Tzachor et al., Feb 2026, [arXiv 2602.08099](https://arxiv.org/abs/2602.08099) | VideoLLaMA3-7B | **Dual-Softmax loss** (bidirectional over the sim matrix) | **one-word `<emb>` prompt + INTERMEDIATE-layer (24) readout** (intermediate > final) | 2 FPS, 180-frame cap, no temporal pool |

**Pattern:** every gain is objective (contrastive/hybrid), data (hard-neg/synthetic), or readout (one-word/latent-attention/intermediate-layer/echo/bidirectional). No work uses a temporal frame operator — they all read a single vector off jointly-processed frame tokens.

---

## 2. Isomorphism adjudication (every candidate ruled against the ledger)

### A. READOUT axis — genuinely un-varied in this project

**A1. Prompt-based "one-word" readout (PromptEOL / E5-V / VidVec `<emb>`).**
- **(a) Cite:** [2307.16645](https://arxiv.org/abs/2307.16645), [2407.12580](https://arxiv.org/abs/2407.12580), [2602.08099](https://arxiv.org/abs/2602.08099).
- **(b) Mechanism:** append a summary-eliciting prompt ("…in one word:" / a dedicated `<emb>` token) and take the **last-token** hidden state instead of mean-pooling. The one-word constraint forces the final next-token prediction to compress the whole input's meaning into one position; PromptEOL shows this beats naive pooling *even zero-shot*.
- **(c) Adjacency:** identical paradigm (decoder→embedding readout). Directly on our extraction step.
- **(d) Borrowable operator (our terms):** at extraction, replace *fixed-neutral-prompt + mean-pool* with *one-word-readout prompt + last-token* for each of the two streams (frames, transcript). Needs a **new forward pass** (different prompt) → **not** reusable from the pooled `.pt` cache; head retrains on the new vectors.
- **(e) In-box:** YES (own train split, no gold, no OCR, local 7B, no ensemble). A *single* better readout prompt is clean; the **multi-prompt ENSEMBLE** variant (MetaEOL, [arXiv 2402.18458](https://arxiv.org/abs/2402.18458)) is the case the ledger flags as *"not literally covered"* by the `cross-seed ensembles` ban — reportable status: **flagged, not banned; needs a one-line user micro-ruling** before an ensemble readout.
- **(f) Cost:** re-extraction ~0.5 GPU-h/dataset + head-only (minutes).
- **(g) Prior / novelty:** LOW–MODEST (~15%). EN capped by F44 (label-limited, not readout-limited); target = ZH val-sel hardening / HateMM robustness. **D7 novelty LOW** but it is a *readout* not an *encoder-adaptation*, so it partially sidesteps the F24 encoder-class veto (nothing about the encoder changes).
- **ISOMORPHISM: SURVIVES.** Readout token-position/prompt was never varied; not F67 (frame density), not S2S/CTF/ISR (temporal frame-group *objects*), not any pooling kill.

**A2. Intermediate-layer readout (VidVec).**
- **(a) Cite:** [2602.08099](https://arxiv.org/abs/2602.08099) (Feb 2026, freshest).
- **(b) Mechanism:** read the embedding from an **intermediate transformer layer (≈24)**, not the final layer; VidVec finds intermediate layers give better video-text embeddings than the last layer.
- **(c) Adjacency:** same paradigm, video-MLLM-embedding, directly a readout choice.
- **(d) Borrowable operator:** at extraction, sweep the layer index we harvest the pooled dual-stream vector from (we currently read final/near-final). $0 if we had cached all layers — we did not, so **re-extraction** dumping a chosen intermediate layer.
- **(e) In-box:** YES (pure extraction change).
- **(f) Cost:** re-extraction ~0.5 GPU-h/dataset (one or two candidate layers) + head-only.
- **(g) Prior / novelty:** LOW–MODEST (~15–20%). Cheapest, freshest, most clearly *not* an encoder-adaptation. D7 novelty LOW (a readout choice), but combined with A1 it is a coherent "principled readout for hate-video MLLM embeddings" story.
- **ISOMORPHISM: SURVIVES.** Which layer we read was never varied.

**A3. Echo / repetition readout (training-free).**
- **(a) Cite:** [2402.15449](https://arxiv.org/abs/2402.15449); training-free "Retrieval Backward Attention" [arXiv 2502.20726](https://arxiv.org/abs/2502.20726).
- **(b) Mechanism:** present the input **twice**; read tokens from the **second** occurrence so each early token can attend to the full sequence (via the first copy) — a *training-free, architecture-free* fix for exactly the causal-attention pooling limit.
- **(c) Adjacency:** directly targets our F35 finding (Qwen group vectors are cumulative causal-prefix summaries; a mean-pool over them integrates the sequence but per-token vectors are context-blind).
- **(d) Borrowable operator:** at extraction, double the token context ([frames][transcript][frames][transcript]) and pool the second copy. Reuses the *same* frozen encoder, no training.
- **(e) In-box:** YES.
- **(f) Cost:** re-extraction with ~2× context ~0.5–1 GPU-h/dataset; head-only after.
- **(g) Prior / novelty:** LOW (~10%). Honest hazard: our video vector is already a **mean-pool** over causal prefixes, which already integrates the whole sequence at the *pooled* level — echo mainly helps *per-token* embeddings, so at the pooled dual-stream level the marginal gain may be small (this is the same reason F37/F66 found pooling "lossless"). Cheap enough to be a $0.5-GPU companion to A1/A2. D7 novelty LOW.
- **ISOMORPHISM: SURVIVES** (readout-time, no training); prior discounted by Law-I pooling-lossless evidence.

**A4. Latent-attention learned pooling (NV-Embed).**
- **(a) Cite:** [2405.17428](https://arxiv.org/abs/2405.17428).
- **(b) Mechanism:** a learned latent-attention layer (512 latent queries cross-attend the token sequence) replaces mean/last-token pooling; NV-Embed shows it consistently beats both.
- **(c) Adjacency:** same paradigm; a *learnable* readout over frozen encoder hidden states.
- **(d) Borrowable operator:** cache the **full token sequence** (not just the pooled vector) and train a small latent-attention pool as part of the (otherwise frozen-encoder) head.
- **(e) In-box:** YES (encoder frozen → not P9b; own train).
- **(f) Cost:** MEDIUM — full-sequence caches (bigger re-extraction) + head training.
- **(g) Prior / novelty:** LOW (~10%). Heavily discounted: F57 leans against "fancier head over frozen features (prior≈0)"; **Law I / F66** is the decisive hazard — a learned data-driven operator over the token sequence is exactly the class F66 proved cannot convert (91–98% of oracle headroom is reachable only by law-III-banned per-item selection; a latent pool is not a selector but it *is* a symmetric data-driven aggregation, the flat leg of F66). D7 novelty LOW (known pooling head).
- **ISOMORPHISM: SURVIVES narrowly** (learned pooling over the full sequence never tried) **but Law-I/F66/F57 cap it hard** — belongs below the training-free readouts.

### B. OBJECTIVE axis — bordered by P9b; check the ban scope exactly

**Ban scope recap (mission-critical):** P9b = *joint* LMM-RGCL training = the **RGCL retrieval loss (triplet+BCE on the fused HEAD key, jointly with the head)** coupled into the encoder LoRA → KILLED by the redistribution law (head↔memory ±1.8pt, 0/12; F51 codifies it as one of exactly two adapted objects). F51 explicitly rules "RGCL-loss-into-encoder is the P9b primary loss, not TARC." **The ban is specifically the RGCL head-key loss coupled jointly.** It does **not** literally name a *decoupled supervised-contrastive pretext on the pooled encoder embeddings* with the head trained *afterward* on frozen adapted features.

**B1. Decoupled supervised-contrastive encoder SFT (VLM2Vec / GME / SupCon).**
- **(a) Cite:** VLM2Vec [2410.05160](https://arxiv.org/abs/2410.05160), GME [2412.16855](https://arxiv.org/abs/2412.16855), NV-Embed [2405.17428](https://arxiv.org/abs/2405.17428); SupCon: Khosla et al., NeurIPS 2020 ([arXiv 2004.11362](https://arxiv.org/abs/2004.11362)).
- **(b) Mechanism:** adapt the encoder with a **supervised-contrastive loss over hate/non-hate labels** (InfoNCE/SupCon, in-batch + hard negatives) on the *pooled embeddings* — i.e. the standard "make matched/same-class pairs close" recipe — **as a decoupled pretext**, then freeze and run the existing triplet+BCE head + kNN.
- **(c) Adjacency:** this IS the mainstream MLLM-embedding paradigm; our word-label *generative* SFT is the odd-one-out (we adapt generatively then harvest, they adapt contrastively).
- **(d) Borrowable operator:** replace word-label generative SFT with a SupCon/InfoNCE SFT at similar cost (~3–5 GPU-h/dataset), decoupled from the RGCL head.
- **(e) In-box:** YES on data grounds. **D7 = encoder-class (F24 veto), user sub-ruling.**
- **(f) Cost:** re-SFT ~3–5 GPU-h/dataset (same order as current LoRA-SFT).
- **(g) Prior / novelty:** LOW (~10%). Two caps: (1) **F44/F55 arithmetic** — EN is label-limited, so a better *representation* objective cannot open it (proven at frozen, collapsed-adapted, healthy-image, and vision-unfreeze levels); HateMM already passes generatively (F53). (2) The redistribution-law *hazard* — decoupling is the escape from the letter of P9b, but nothing guarantees escape from the *mechanism* (contrastively sharpening the encoder may just move signal the head already extracts). Novelty: contrastive-adapting is arguably *more* on-paradigm than generic generative LoRA, but still encoder-class.
- **ISOMORPHISM: PARTIALLY-ISOMORPHIC / SURVIVES-CAVEATED.** Decoupled SupCon ≠ P9b's joint RGCL-loss and ≠ generic generative LoRA, so it is a *third objective* on the letter of F51 — but it shares P9b's spirit and inherits the EN arithmetic cap. Report to user with the exact ban-scope note above.

**B2. Hybrid contrastive + generative adaptation (VladVA).** *(strongest objective-level candidate)*
- **(a) Cite:** [2412.04378](https://arxiv.org/abs/2412.04378) (CVPR 2025).
- **(b) Mechanism:** VladVA adapts an LVLM for discrimination by **combining a contrastive loss (short captions) with the autoregressive next-token loss (long descriptions)**, via soft-prompt + LoRA — keeping generative ability while adding discriminative structure; +4.7–7.0 R@1 over CLIP-family, 61.7→85.6 R@1 Flickr30k on LLaVA-1.5-13B.
- **(c) Adjacency:** *our generative word-label SFT is exactly VladVA's autoregressive half with no contrastive half.* This is the single most direct "we already do half of this recipe" finding in the survey.
- **(d) Borrowable operator:** add a **decoupled contrastive term** (over pooled embeddings, label- or caption-supervised) *alongside* our existing generative word-label SFT — one adaptation, hybrid loss, then the existing frozen head.
- **(e) In-box:** YES on data grounds; **D7 encoder-class sub-ruling** (but the hybrid recipe is a *named, non-generic* adaptation, more novelty than "generic LoRA").
- **(f) Cost:** re-SFT ~3–5 GPU-h/dataset (adds a loss term, same forward).
- **(g) Prior / novelty:** LOW (~10%), EN-capped identically to B1; target = ZH val-sel robustness / HateMM hardening. **Best novelty-per-cost of the objective candidates** — it is the least-isomorphic objective change (the added contrastive term is decoupled from the RGCL head, so *not* P9b) and it is a documented, cited recipe our pipeline is already halfway to.
- **ISOMORPHISM: SURVIVES-CAVEATED** (same borders as B1; the *hybrid* framing is the non-isomorphic angle).

**B3. Bidirectional-attention surgery + MNTP (LLM2Vec / NV-Embed).** *(most novel, F35-motivated)*
- **(a) Cite:** LLM2Vec [2404.05961](https://arxiv.org/abs/2404.05961), NV-Embed [2405.17428](https://arxiv.org/abs/2405.17428).
- **(b) Mechanism:** **remove the causal attention mask** in the LLM backbone (bidirectional attention) and adapt with **masked-next-token-prediction (MNTP)** so every token encodes full past+future context, then extract. This is the field's standard decoder→encoder conversion.
- **(c) Adjacency:** same paradigm, and it **directly attacks our own F35 finding** — F35 *diagnosed* that Qwen2.5-VL group vectors are cumulative causal-prefix summaries (position dominates content) and explicitly noted this as a structural limitation; no run ever *removed* the mask. This candidate is the mechanistic answer F35 sets up.
- **(d) Borrowable operator:** enable bidirectional attention in the Qwen LLM stack + a short MNTP adaptation, then re-extract embeddings.
- **(e) In-box:** YES on data grounds. **D7:** still encoder-class, but an *architecture-level* named mechanism (not "generic LoRA") — the **strongest D7-novelty** in the survey; user sub-ruling.
- **(f) Cost:** MEDIUM — re-SFT (MNTP adaptation, few GPU-h) + re-extraction; some engineering to flip the mask in the VLM attention path (codex-review-gated: touching attention masks in `generate`/forward).
- **(g) Prior / novelty:** LOW–MODEST (~15%). Honest caveat: the **pooled** dual-stream vector is already a *mean-pool over causal prefixes*, which integrates the whole sequence — so bidirectional attention may add little at the *video-level pooled* readout even though it changes per-token vectors (same Law-I hazard as A3/A4). But it is the one candidate that changes *what information the representation carries* rather than how it is read/adapted, and it is the most defensible novelty claim.
- **ISOMORPHISM: SURVIVES.** Genuinely un-tried; F35 motivates rather than forecloses it; not covered by F51's two adapted objects (those are LoRA-weight-space; this is an attention-topology change).

### C. DATA / DISTILLATION / MATCHING axis — isomorphic-dead

**C1. Textual-discriminative KD from an LLM text-embedder teacher (UniME stage-1).** [2504.17432](https://arxiv.org/abs/2504.17432).
- **ISOMORPHIC-DEAD.** Feature/relational KD into the encoder = **C5 (F32, killed)**: "textbook feature-KD fails D7; distilled geometry *is* the D7-dead encoder-swap representation; teacher=student-scale → no gap." UniME's twist is an *external* strong text-embedder teacher, but (i) distilling into the encoder is still encoder-class (D7), (ii) the oracle ceiling for a text-discrimination teacher = the banked encoder-swap/B1 result (HateMM-only pass, EN fail, ZH −0.0112 — fails ≥2-dataset goal), and (iii) an external teacher grazes the `external model APIs / data-export` ban if API-based (open teachers exist but re-enter C5). *Cite the ban: F32 / `state` C5 entry.*

**C2. Synthetic data + MLLM-as-judge hard negatives (mmE5 / UniME-V2).** [2502.08468](https://arxiv.org/abs/2502.08468), [2510.13515](https://arxiv.org/abs/2510.13515).
- **ISOMORPHIC-DEAD (multiple).** Synthetic training data = **AUG (F60, dominated)** + the `TRAINING DATA = single-dataset train split ONLY` veto (no external/synthetic pool as a contribution). Hard-negative mining = **cand-2 (F56, TIE)** + head-level global-hardest already deployed (**C3geo/F25**). MLLM-as-judge hard-neg selection = grazes **P2/MJ (F49)**: which-pair-is-harder judgments hit the modality/comparability alignment ceiling (a ≤ 0.588 < q_req 0.663). *Cite: F60, F56, F25, F49, single-dataset veto.*

**C3. Generative matching / re-ranking supplement (ReMatch).** [arXiv 2511.19278](https://arxiv.org/abs/2511.19278) (Nov 2025).
- **ISOMORPHIC-DEAD.** "Boosting representation through matching" supplements contrastive with MLLM-scored query-candidate matching — i.e. an MLLM re-ranker over retrieved candidates = **P2/P2b (comparability ⊥ vote-correctness, dead)** + **P10/P10b (A-fuse, localization-only, no accuracy role)**. *Cite: P2/P2b, P10 `state` entries.*

### D. VIDEO frame-handling axis (search-space #5)

- **VLM2Vec-V2** (8 uniform frames, InfoNCE) and **VidVec** (2 FPS/180-cap, no explicit temporal pool — LLM processes the visual token sequence jointly, single-vector readout) are the SOTA video-MLLM-embedding works, and **neither uses a non-pooling temporal operator.** Their video gains come from the *objective* (contrastive/dual-softmax) and *readout* (one-word/intermediate-layer), folded into B/A above.
- **ISOMORPHIC-DEAD for any frame operator.** Denser sampling = **F67 (16f KILL, 8f saturates)**; temporal set/pool/order over frame groups = **S2S/CTF/ISR/W2-C (F37/F39/F66/F30, don't-pool family CLOSED)**; per-item frame/query selection = **law-III / F66** (selection-locked). The survey's honest video finding is a **corroboration** of these kills, not a new lever.

---

## 3. Cross-references to the mission's flagged items

- **P4 schema-distill / CoT-then-classify:** any "distill a reasoning chain into the embedding" variant (a live idea in the UniME/rationale line) grazes **P4 (redundant)** + **P11 (MLLM-scores-as-training-signal, banned)** + **C3-nontarget (F18, dense reasoning text DEAD at fusion)**; F54 already noted rationale-SFT "needs gold spans or grazes P11." **Dead.**
- **Multi-prompt ensembling exact status:** the `cross-seed ensembles` ban is on averaging across *independent seeds*; a **multi-PROMPT** ensemble (MetaEOL, [2402.18458](https://arxiv.org/abs/2402.18458)) is **flagged as "not literally covered"** and needs a one-line user micro-ruling. A **single** better readout prompt (A1/A2) is clean and unflagged.
- **Two-stage SFT→contrastive (mission item 6):** our current pipeline is *generative-SFT → frozen-head-contrastive*. The field's two-stage is *contrastive-SFT → head*. **Joint** = P9b (KILLED). **Decoupled/alternating** two-stage (B1/B2) is un-run and survives-caveated; **B3** (bidirectional) is a distinct third route. There is no un-tried *joint* variant that escapes the P9b redistribution law.

---

## 4. TOP-5 — ranked by (prior × cost), novelty noted separately

| # | Candidate | Cite | In-box? | Cost | Prior (≥+1pt / ≥1 dataset) | D7-novelty | Isomorphism | Cheapest next step |
|---|---|---|---|---|---|---|---|---|
| **1** | **Intermediate-layer + one-word `<emb>` readout** (VidVec / PromptEOL / E5-V) | 2602.08099 / 2307.16645 / 2407.12580 | YES (extraction-only) | **~0.5 GPU-h re-extract + head-min** | **~15–20%** (ZH val-sel hardening; EN capped) | LOW (but *not* encoder-adaptation → partial F24 sidestep) | **SURVIVES** — readout token/layer never varied | re-extract 1–2 intermediate layers with one-word readout; $0-style G0-cond conditional-info gate on cached dev first |
| **2** | **Bidirectional-attention surgery + MNTP** (LLM2Vec / NV-Embed) | 2404.05961 / 2405.17428 | YES (data); **D7 sub-ruling** | **MED** (re-SFT + re-extract) | **~15%** | **HIGHEST** — architecture-level named mechanism, F35-motivated, not "generic LoRA" | **SURVIVES** — F35 diagnosed, never removed the mask; outside F51 two-object (weight-space) | codex-gated mask flip + short MNTP; re-extract; conditional-info gate |
| **3** | **Hybrid contrastive + generative adaptation** (VladVA) | 2412.04378 | YES (data); D7 sub-ruling | **MED** (re-SFT, +loss term) | **~10%** (EN capped; ZH/HateMM robustness) | MED — named recipe, we already do the generative half; contrastive term decoupled → **not** P9b | **SURVIVES-CAVEATED** — bordered by P9b, decoupling is the escape | add decoupled SupCon term to existing word-label SFT |
| **4** | **Echo / repetition readout** (training-free) | 2402.15449 / 2502.20726 | YES (extraction-only) | **~0.5–1 GPU-h re-extract** | **~10%** (Law-I discounted) | LOW | **SURVIVES** — readout-time, targets F35 causal-prefix at $0 | double context, pool 2nd copy; companion to #1 |
| **5** | **Latent-attention learned pooling** (NV-Embed) | 2405.17428 | YES (encoder frozen → not P9b) | **MED** (full-seq caches + head) | **~10%** (F66/F57 discounted) | LOW | **SURVIVES narrowly** — learned pooling over full sequence untried, but Law-I/F66/F57 cap | cache full token seq; train small latent pool in head |

**Rejected as isomorphic-dead (with the exact ban):** SupCon-only encoder SFT decoupled (B1 — survives on the letter but dominated by #3's hybrid framing + EN cap; listed as a variant of #3); textual-discriminative KD / UniME stage-1 (= C5/F32); synthetic data + MLLM-judge hard-negs (= AUG/F60 + single-dataset veto + cand-2/F56 + MJ/F49); ReMatch generative matching (= P2/P10); CoT/rationale distillation (= P4/P11/F18); any temporal frame operator (= F67/S2S/CTF/ISR/F66).

**Bottom line.** This is the paradigm closest to ours, and the sweep confirms the field's four levers are (three of four) already measured-negative or arithmetic-capped in this project. The only genuinely un-enumerated axis inside the paradigm is the **READOUT** (never varied — #1/#4/#5) plus one **architectural mechanism** (bidirectional attention — #2, the highest-novelty item, and the one our own F35 sets up). None has a strong prior on a *new* dataset — EN is label-limited at five proven levels and HateMM already passes — so the honest framing for all five is **ZH-val-sel hardening (the F45 78-dev selection tax) + a possible D7-novelty upgrade**, best gated by a $0/cheap G0-cond conditional-info screen before any GPU. The single highest-value move for the D7 integration story is **#2 (bidirectional-attention surgery)**; the single cheapest-to-falsify is **#1 (intermediate-layer one-word readout)**.

---

## Provenance
- Ledger cross-check: `state/directions_tried.json`, `state/findings.jsonl` (F1–F67), `refine-logs/REDTEAM_EXTERNAL_FAMILIES.md`. Laws I–IV per `research-wiki/DRAFT_analysis_chapter.md`.
- External anchors (verified titles + arXiv IDs, 2024–2026): PromptEOL 2307.16645; E5-V 2407.12580; VLM2Vec 2410.05160; VLM2Vec-V2 2507.04590; GME 2412.16855; NV-Embed 2405.17428; LLM2Vec 2404.05961; VladVA 2412.04378; UniME 2504.17432; UniME-V2 2510.13515; mmE5 2502.08468; Echo 2402.15449 (+ training-free 2502.20726); MetaEOL 2402.18458; VidVec 2602.08099; ReMatch 2511.19278; SupCon (Khosla NeurIPS'20) 2004.11362.
- Discipline: CPU-only, zero GPU/SLURM/Modal/downloads; `state/` unmodified; cloud/external numbers are triage context only, never mixed with local G-repro numbers. Nothing here is a prereg or GPU authorization.
