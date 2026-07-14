# W2-A FORENSIC RECON — cross-modal GROUNDED retrieval key (transcript-conditioned Qwen vision representation)

**Agent:** W2-A design owner (round-3 wave-2 lead). **Date:** 2026-07-15. **ZERO GPU** (recon + prereg design only; no forward run — Qwen cannot be exercised without SLURM).
**Candidate source:** `research-wiki/ROUND3_CANDIDATES_WAVE2_2026-07-15.md` §W2-A (lines 43-113), scout prior **MODEST–FAIR**.
**Sibling recons mirrored for severity + format:** `refine-logs/B4_FORENSIC_RECON.md`, `refine-logs/C2MEM_FORENSIC_RECON.md`, `refine-logs/S2S_PROBE_DESIGN.md` §1/§4 (banked-cache correctness reference).
**Companion prereg:** `research-wiki/experiments/exp-w2a-grounded.md`.

---

## HEADLINE VERDICT (one paragraph)

**W2-A is LIVE but its prior must be REVISED DOWN to LOW–MODEST, and its primary mechanism must be RE-SPECIFIED, because the scout's premise is partly false and the naive realization is architecturally vacuous.** Two load-bearing facts, both verified this recon: **(1)** the banked pipeline does **not** encode vision and transcript independently — `img_feats` excludes the transcript (true), but **`text_feats` is already a JOINT forward over frames + title + transcript**, pooled at the response span, so the img×text interaction the scout says "the retrieval key never contains" **is in fact already banked** in `text_feats` and already enters the retrieval geometry (the head projects and fuses both `img_feats` and `text_feats`). **(2)** Qwen2.5-VL's LLM backbone is **fully causal** (`self.is_causal=True`, `modeling_qwen2_5_vl.py:723`; `_update_causal_mask` at `:1244`/`:1302`), so with the banked **video-first** token order the vision tokens are placed **before** the transcript and — by the causal mask — **cannot attend to it**. A "transcript-conditioned vision" key pooled from the vision-token span of a video-first forward is therefore **provably identical to the ungrounded vision pool (≡ `img_feats`) up to the instruction text — a NO-OP**. The only architecturally real realization of "a visual representation a dual encoder structurally cannot produce" is a **transcript-FIRST ordering** (transcript placed before the frames, so the vision tokens causally attend back to it) with a **vision-token-span pool** — this is the re-specified W2-A primary. Its honest D1 threat is severe and directly C3-nontarget-shaped: the joint interaction is **already partly banked in `text_feats`**, so the pre-declared **concat(img_feats, text_feats)-must-lose** arm is the whole ballgame. Recommendation: **proceed to a pre-registered, oracle-gated, zero-test-touch probe** (one new grounded forward per video, ~2–3 GPU-h, both datasets), with the concat-must-lose D1 arm binding; **novelty remains a D7-class user ruling** (this is the closest wave-2 candidate to the encoder line, and the cheap decoder-reordering realization is *more* generic than the cited iGVLM/TIE conditioning-pathway prior art, not less).

---

## 1. TRANSCRIPTS REALITY (load-bearing — where the "native transcript" actually lives)

**One-liner:** The dataset-native transcript is the **`text` field of `data/gt/<ds>/<split>.jsonl`**, produced upstream by **whisper-large-v3 ASR** (evidence: `data/ASR/<ds>/<split>_asrK{4,30,60}_whisper-large-v3.jsonl`). It is the **same transcript the banked `text_feats` forward already consumes** (`generate_VideoMLLM_embedding_HF.py:349-355`, `item["text"]`). It is **NOT MLLM-generated** — this is the non-isomorphism anchor vs C3-nontarget (§4).

### 1.1 Coverage stats (verified this recon, `data/gt/<ds>/<split>.jsonl`)

| dataset | split | N | transcript non-empty | empty | median len (chars, non-empty) | max len | music-marker 🎼 |
|---|---|---|---|---|---|---|---|
| **HateMM** | train | 744 | 705 (**94.8%**) | **39** | 694 | **80 731** | 348 |
| HateMM | val | 107 | 98 (91.6%) | 9 | 429 | 12 275 | 44 |
| HateMM | test | 215 | 189 (87.9%) | 26 | 743 | 13 055 | 80 |
| **MHC-EN** | train | 549 | 549 (**100.0%**) | 0 | 369 | 1 367 | 215 |
| MHC-EN | val | 80 | 80 (100.0%) | 0 | 440 | 1 422 | 32 |
| MHC-EN | test | 161 | 161 (100.0%) | 0 | 407 | 1 302 | 59 |
| MHC-ZH | train | 579 | 579 (100.0%) | 0 | 106 | 708 | 423 |

**Three consequences that drive the design:**
- **HateMM has a 5–12% empty-transcript tail.** On those videos W2-A's mechanism is **vacuous by construction** (no transcript to condition on → the grounded key degenerates to the ungrounded vision pool). They dilute the HateMM paired Δ by ~8% of rows; handled identically in both arms (paired contrast unbiased but attenuated), count logged. **MHC-EN is 100%-covered**, so it exercises the grounding mechanism on **every** row — for W2-A specifically, MHC-EN is the *cleaner* mechanism-existence dataset even though it is the historically weaker-prior binding-gap dataset.
- **HateMM transcripts are pathologically long (max 80 731 chars).** This is the same "long-input" surface that produced the A-line's swallowed-OOM parse failures. **But the banked `text_feats` forward already survived every one of these** (§2.2: 0 text-OOM zero-rows; the single HateMM-train zero-row is the undecodable-*video* guard, not a text failure) — Qwen's processor truncates to the model max, so the joint forward is proven end-to-end on this exact input distribution. W2-A's grounded forward inherits that same truncation behaviour (parity note in the prereg).
- The `text` field carries **only the ASR transcript**; there is **no `title` key** in any HateMM/MHC gt file (verified: keys = `['id','text','label']`), so the banked extraction's `item.get("title","")` is always `""` and `text_feats`'s prompt is effectively `TEXT_INSTRUCTION + "\nTitle: (none)\nTranscript: <text>"`. W2-A uses the identical `text` string.

### 1.2 Other transcript-derived caches on disk (for the isomorphism audit)

- `data/CLIP_Embedding/MHC{,_zh}/*_transcript_mpnet512_HF.pt` — a **separate MPNet** transcript embedding (different text encoder; not Qwen; not used by W2-A).
- `data/ASR/<ds>/*_asrK{4,30,60}_whisper-large-v3.jsonl` — **segment-level** ASR (the source W2-B's `_mm` sub-clip channel draws on). W2-A uses the **whole-video** `text` field, not the K-segment split.

---

## 2. BANKED CACHES REALITY — the scout's "independent pooling" premise is only HALF true

### 2.1 What the two banked streams actually are (`generate_VideoMLLM_embedding_HF.py`, read this recon)

| stream | forward inputs | pooled span | contains transcript? | carries img×text interaction? |
|---|---|---|---|---|
| **`img_feats`** (Dv 3584) | frames **+ fixed `IMG_INSTRUCTION`** (`:45-47`, `:345-348`) | **prefix** `[0:last-<\|im_start\|>]` = sys+userhdr+**all vision tokens**+instruction+im_end (`:290-303`) | **NO** | **NO** (vision + generic instruction only) |
| **`text_feats`** (Dt 3584) | **frames + `TEXT_INSTRUCTION` + title + transcript** (`:349-359`) | **response** span = trailing assistant-header tokens (`:304-318`) | **YES** | **YES** — response tokens are last, causally attend over *both* frames and transcript |

**This is the decisive recon finding.** `text_feats` is **not** a text-only pool: it is the causal-LM summary of a **joint frames+transcript forward** (the `_build_messages` call at `:263`/`:241-251` puts the video into the same message as the text). So the pipeline's retrieval key **already contains a cross-modal interaction term** — the scout's framing that "the banked pipeline encodes vision and transcript independently … so the retrieval key never contains the interaction" (`ROUND3…§W2-A:45-55`) is **false for `text_feats`**. What is genuinely missing is only a **different pooling** of that joint forward (a vision-conditioned pool, vs `text_feats`'s response-span pool). W2-A's real claim shrinks accordingly: not "add the missing interaction," but "**a different, vision-side pool of the joint forward carries convertible info the response-span pool (`text_feats`) does not.**" That is a strictly weaker, C3-nontarget-shaped claim.

### 2.2 Cache integrity (verified this recon via `torch.load`, HateVideo env)

`data/CLIP_Embedding/{HateMM,MHC}/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF.pt`: shapes match the banked contract (HateMM 744/107/215, MHC-EN 549/80/161, Dv=Dt=3584, L2-normed). **Zero-rows:** HateMM train has **exactly 1** img-zero AND 1 text-zero row — the **same** undecodable-video guard row (both streams zeroed together, `:360-363`); all other splits have **0** zero-rows. Crucially, the 39/9/26 HateMM **empty-transcript** videos have **non-zero** `text_feats` (norm 1.0) — the forward still ran with `"Transcript: (none)"`. **⇒ no text-length OOM anywhere; the joint forward is proven on the full HateMM long-transcript distribution.** These are the G-recon anchor files for W2-A's machinery gate (§5).

---

## 3. THE ARCHITECTURAL FINDING — causal masking kills the naive design; the key-pooling decision (SETTLED)

### 3.1 Qwen2.5-VL's LLM backbone is fully causal (verified this recon)

`modeling_qwen2_5_vl.py` (installed `transformers==4.49.0`): `Qwen2_5_VLAttention.is_causal=True` (`:723`); `is_causal` passed to SDPA (`:989`,`:997`,`:904`); `Qwen2_5_VLModel._update_causal_mask` builds a standard lower-triangular causal mask (`:1244`,`:1302`,`:1365-1383`). The merged vision tokens are **injected as ordinary sequence positions** (masked_scatter, in-place, no shift — the banked code's own preflight invariant `last_hidden.shape[0]==input_ids.numel()`, `:283`); there is **no bidirectional attention over vision tokens** in the LLM. (The ViT vision encoder has its own attention, but that is *pre-*LLM and sees no text.)

### 3.2 Consequence: token ORDER decides what "grounded" can mean

In a causal decoder, hidden state at position *i* depends only on inputs at positions ≤ *i*. The banked message order is **video-first**: `[sys][user-hdr][VISION_1..VISION_nv][instruction(+transcript)][im_end][asst-hdr]`. Therefore:

| pooled span | positions | attends to transcript? | what it is |
|---|---|---|---|
| **vision-token span** (video-first) | earliest | **NO** (transcript is downstream, masked out) | **≡ `img_feats` vision part — a NO-OP w.r.t. the transcript** |
| transcript-token span (video-first) | mid | attends *back* to vision | "transcript grounded in vision" (a real interaction, but ≈ `text_feats` content) |
| response/asst-header span (video-first) | latest | attends to both | **exactly `text_feats` — already banked** |

**So the ROUND3 pitch — "the vision-token pool *after* it has cross-attended to the transcript" (`§W2-A:52`) — is architecturally impossible with the banked video-first order.** Vision tokens never see a downstream transcript. A video-first vision-span "grounded key" would pass a naive extraction gate and then behave as `img_feats` in the probe (the classic silent no-op).

**The only realization that produces a genuinely transcript-conditioned VISUAL representation — the one thing "a dual encoder structurally cannot produce" — is to place the transcript BEFORE the frames** (`content=[{text:transcript},{video:frames},{text:instruction}]`), so the vision tokens causally attend back to the transcript, then pool the **vision-token span**. This is the standard decoder mechanism behind text-guided visual features (iGVLM 2603.02748, TIE 2511.20770; §6), realized here for free via reordering rather than a dedicated conditioning pathway.

### 3.3 SETTLED key-pooling design (ONE primary + ONE sensitivity)

- **PRIMARY grounded key `grd` (3584-d):** **transcript-first** joint forward `[transcript][frames][IMG_INSTRUCTION]`; pool the **mean of the vision-pad token hidden states** (last layer), L2-norm. This is "the visual representation, conditioned on what was said." Vision-span (not prefix-span) so the claim stays *visual* (a conditioned-vision key, not a generic joint pool). Justified against the C3-nontarget lesson: the info must be shown to live in the **interaction** the marginals cannot form, and a vision-side pool is the span most distinct from the already-banked response-span `text_feats`.
- **SENSITIVITY (one only) `grd_pfx`:** same transcript-first forward, **prefix-style** pool (vision **+** trailing instruction tokens, mirroring the `img_feats` recipe span but under the grounded order). Tests robustness of the pooling-span choice without a second forward. **Cannot rescue a failed primary** (S2S N3 rule).
- **Explicitly NOT primary:** the video-first vision-span pool (§3.2, vacuous) — it is instead repurposed as the **ungrounded reference** for the "grounding-live" positive control (§5). The video-first transcript-token-span and the response-span (`text_feats`) variants are **deferred** (the latter is already banked; probing it is just probing `text_feats`).

---

## 4. NON-ISOMORPHISM AUDIT (vs `state/directions_tried.json` — 24 dead + 9 bans)

### 4.1 vs C3-nontarget (19th, `DEAD_AT_FUSION`) — the closest and most dangerous sibling

- **Distinct on object:** C3-nontarget **generated** dense MLLM reasoning text (job 13101 `.npy` embeddings) and fused it as a **separate channel**. W2-A generates **nothing**, adds **no channel** — it re-pools the **native transcript** (whisper ASR, already in the pipeline) inside the encoder forward. **The DATASET-NATIVE-transcript-in-forward vs MLLM-GENERATED-text-channel distinction is clean and holds** (audited rigorously per the team-lead directive). ✓ non-isomorphic on injection point.
- **BUT it inherits C3-nontarget's epitaph risk almost exactly.** C3 died because "info banked in Qwen pathway" — the generated text added ≤+0.016 conditional acc over `concat(CLIP,Qwen)`, below the +0.040 bar, indistinguishable from the permutation null. W2-A's interaction term is **even more banked**: it comes from the **same joint forward** whose response-span pool **is** `text_feats`, a feature already in the retrieval key. So W2-A's honest question is identical in shape to C3's: *does a re-pool of the joint forward add conditional info beyond `concat(img_feats, text_feats)`?* — and C3 answered the analogous question **no**. This is why the prior is revised down and why the concat-must-lose arm is binding.

### 4.2 vs B1 (20th, frozen-Qwen ZH fail) and B2/B4 (encoder line)

- B1 **swapped the encoder** (frozen Qwen pooled) on MHC-ZH → FAIL both protocols (final −0.0112 acc, 1/3 sign). W2-A does **not** swap the encoder — same frozen Qwen2.5-VL-7B; it changes the **key-construction operation** (grounding via reordering). ✓ non-isomorphic on slot.
- **Shared risk:** W2-A's grounded key is built from the **same frozen Qwen-7B representation** whose *pooled* form already **failed to convert MHC-EN** (SAV #18 "dilution hypothesis FALSIFIED; MHC-EN data/label-limited"; B1 ZH fail). The grounding-adds-interaction escape is W2-A's **only** distinction from that dead pooled representation — the same structural position S2S is in, and it rides the same alignment-not-bits argument S2S owns.

### 4.3 vs encoder-swap (D7) — the honest "近 encoder-class" framing

W2-A is a **grounded ENCODING mechanism**: a skeptic legitimately reads it as "yet another frozen-Qwen feature with a different prompt order and pooling span." The rebuttal is the interaction-term argument (uncomputable from the two marginals) — **but §2.1 shows one of those marginals (`text_feats`) is already a joint pool**, so the rebuttal is *weaker* than the scout assumed. The defensible novelty is **composite only** ("first transcript-conditioned visual representation used as the retrieval key in hateful-video, so implicit visual–speech incongruity enters retrieval geometry rather than a decision-side conflict head"). **Whether that clears the novelty clause is a D7-class USER RULING**, identical in kind to S2S and B3-LoRA — **not decidable here.** State plainly: **final novelty = user ruling.**

### 4.4 Bans / vetoes / D-laws

Single-dataset own-train memory ✓; **no OCR** ✓ (native ASR ≠ OCR); **no gold in-method** ✓ (transcript/frames unlabeled; gold only as probe ceiling, REFLECTION §4 compliant); no cross-seed ensemble ✓; no external API ✓ (local Qwen-7B); **no MLLM-generated text / no generation** ✓; not a P1–P5 re-proposal ✓; no kNN-pool expansion ✓ (same memory videos, richer key). **D1** is the real threat (§4.1, exposed by the concat arm); **D2** representation-level (the winning class); **D3** paired zero-training LOO + bootstrap. **Legal to run** — legality is not the binding constraint; D1 redundancy and D7 novelty are.

---

## 5. EXTRACTION-CORRECTNESS GATES — the G-recon analog for a forward with NO banked twin

The grounded (transcript-first) forward has **no banked twin** to reproduce (novel ordering), so — exactly as the team lead directed — correctness is certified from **internal consistency + banked anchors on control forwards**:

1. **G-recon-IMG (banked machinery anchor, MANDATORY, HALT).** A control forward `[frames][IMG_INSTRUCTION]` (video-first, byte-identical to the banked `img_feats` recipe — the harness **imports the banked helpers verbatim**, S2S §3 precedent) must reproduce banked `img_feats[v]`: **cos ≥ 0.9999 AND max-abs ≤ 1e-3** on every non-guard video. This is precisely the team-lead-named "text-ablated forward reproduces banked img_feats" gate, run as a control. Proves model load + frame sampler + pooling code are the banked ones.
2. **Grid gate (MANDATORY, HALT).** `n_vis == grid_t·(grid_h//2)·(grid_w//2)` from the model's own `video_grid_thw` + `spatial_merge_size=2` (S2S A1) — pins the vision/text boundary so the vision-span pool is correctly located, in **both** the control and the grounded (reordered) forward.
3. **Grounding-LIVE positive control (MANDATORY, HALT — the analog of S2S's temporal control).** For transcript-present videos, `cos(grd_vision_pool, ungrounded_vision_pool) < τ_live` (materially different — the transcript actually flowed into the vision tokens); for **empty-transcript** videos, `cos(...) ≥ 0.999` (no transcript → no change, as causal masking predicts). If the "present" set does not move, the mechanism is a silent no-op and the probe is **VOID** (this is the exact trap §3.2 warns about; τ_live pre-declared in the prereg, calibrated on a smoke subset, conservative).
4. **Placebo-transcript control (MANDATORY, subset ≥50 videos).** The grounded key with a **shuffled/mismatched** transcript must differ from the true-transcript grounded key → proves content-sensitivity, not a position-shift artifact. The extraction-time analog of the permutation null.
5. **Length/parity invariant.** Assert `last_hidden.shape[0]==input_ids.numel()` (banked preflight); log the M-RoPE offset (transcript-before shifts vision position ids — correct, handled by `get_rope_index`, but verified non-crashing).

**Cost:** 2 forwards/video (grounded + img-control) over 1856 videos ≈ **~2–3 GPU-h**, 1×A100, single sbatch, same ceremony as S2S Stage E; placebo on a ~50-video subset (negligible). Storage: 4×3584-d fp16 vectors/video (grd, grd_pfx, ungrounded-vision, img-recon) ≈ **~55 MB** total, sub-GB.

---

## 6. PRIOR ART (WebSearch, 2026-07-15) — raw operation established; the composite unclaimed; the cheap realization is MORE generic

- **Text/instruction-conditioned visual representations in decoder VLMs are an active, named area:** iGVLM ("Dynamic Instruction-Guided Vision Encoding," arXiv 2603.02748) uses a *dedicated conditioning pathway*; TIE ("Text-Guided Semantic Image Encoder," 2511.20770) derives query/task embeddings to enrich visual features; "Text-Guided Layer Fusion" (2601.03100) forms a query-conditioned fused visual representation. **⇒ the raw operation (condition visual features on text) is NOT novel** — corroborates ROUND3's standalone-novelty-fails verdict. Note W2-A's realization is *cheaper and more generic* than these (it uses plain decoder causal attention + reordering, not a trained conditioning module), which **weakens**, not strengthens, the mechanism-novelty case.
- **In-domain hateful-video fusion is crowded but does NOT use a grounded key as retrieval geometry:** CMFusion (channel/modality fusion, 2505.12051), MM-HSD (2508.20546, A+V+T+OCR), MoRE (WWW 2025, **pooled** joint retriever → MoE), Reasoning-Aware fusion (2512.02743, decision-side), MultiHateLoc/GNN (2512.10408). One review line states plainly that "**concatenation fails to capture inter-dependencies between modalities**" — which is exactly the axis W2-A's concat-must-lose arm tests, and exactly why the in-domain field has moved to cross-modal-attention *decision heads*, not *retrieval keys*. The specific composite ("transcript-conditioned visual representation as the kNN retrieval key in hateful-video") is **absent** — the gap is real, but it is a composite/domain-transfer gap, i.e. a D7 user ruling.

---

## 7. HONEST PRIOR (revised) + falsifiable sentence

**Prior: LOW–MODEST** (below the scout's MODEST–FAIR). Revised down by two recon findings: (i) the interaction term W2-A adds is **already partly banked in `text_feats`** (§2.1), making this a C3-nontarget-shaped redundancy question that the analogous C3 probe answered *no*; (ii) the naive design is a causal-masking no-op (§3.2), so W2-A survives only in the reordered form whose novelty is *more* generic than the cited conditioning-pathway prior art (§6).

**One falsifiable sentence:** *If, on the frozen grounded key, the paired LOO kNN vote does not beat `concat(img_feats, text_feats)` — the capacity-matched marginals baseline that already contains one joint-forward pool — by an oracle-ceiling Δacc ≥ +0.04 on at least one dataset AND a raw Δacc AND Δmacro-F1 ≥ +0.05 on HateMM, then the transcript-conditioned visual pool carries no convertible information beyond the two banked marginals, and W2-A is DEAD (the interaction is redundant with the already-joint `text_feats`).*

---

## 8. RECOMMENDATION TO LEAD

1. **Proceed to the pre-registered probe** (`exp-w2a-grounded.md`), but bank the revised **LOW–MODEST** prior and the **re-specified transcript-FIRST vision-span** primary (the video-first version is a documented no-op — do not let an implementer build it).
2. **The concat(img,text)-must-lose D1 arm is the entire verdict.** Report the primary Δ as **GROUNDED − CONCAT** (not grounded − img): a grounded key that beats `img_feats` but not `concat(img,text)` is a **KILL**, because `concat` already contains the joint `text_feats` pool. This is the binding, pre-declared arm ordering.
3. **Queue the grounded extraction on the LOCAL GPU** (raw-video, license-sensitive — off cloud, per CLAUDE.md), ceremony-ready behind the S2S extractor in the queue; the Stage-P probe then runs CPU/cloud, zero test-touch.
4. **Novelty is a D7 user ruling** — present W2-A as a *grounded-encoding* mechanism whose composite claim needs the D1-interaction evidence AND the user's novelty call; do not present it as an independent novelty win.

---

## PROVENANCE

- Banked extraction (img_feats prefix / text_feats response, joint forward): `src/utils/generate_VideoMLLM_embedding_HF.py:45-52,241-251,254-323,349-359`.
- Causal backbone: `.../transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:723` (`is_causal=True`), `:989,:997,:904` (SDPA), `:1244,:1302,:1365-1383` (`_update_causal_mask`); vision-token layout (temporal-major, in-sequence) `:466-505,529-534,560-562` (per S2S §4). transformers 4.49.0, HateVideo env.
- Transcript reality: `data/gt/{HateMM,MHC,MHC_zh}/{train,val,test}.jsonl` (`text` field; coverage table §1.1 computed this recon); ASR provenance `data/ASR/<ds>/*_whisper-large-v3.jsonl`; MPNet transcript cache `data/CLIP_Embedding/MHC{,_zh}/*_transcript_mpnet512_HF.pt`.
- Banked cache integrity (norms, zero-rows): `data/CLIP_Embedding/{HateMM,MHC}/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF.pt` (torch.load this recon).
- Candidate spec: `research-wiki/ROUND3_CANDIDATES_WAVE2_2026-07-15.md` §W2-A (43-113), D1-probe caveat (90-92).
- House standard: `research-wiki/experiments/exp-s2s-r3.md` (§4 correctness gates, §6 oracle/Fano/raw bars, §6.6 dataset rule), `refine-logs/S2S_PROBE_DESIGN.md` (§2 parity table, §4 anchors, §5 probe plan).
- Graveyard + bans + D-laws + REFLECTION §4: `autoresearch/goal_mllm_plus3/state/directions_tried.json`, `research-wiki/REFLECTION_mllm_integration_failures.md`. C3-nontarget sibling: `refine-logs/C3_FUSION_PROBE_RECORD.md`.
- Prior art (WebSearch 2026-07-15): iGVLM arXiv 2603.02748; TIE 2511.20770; Text-Guided Layer Fusion 2601.03100; CMFusion 2505.12051; MM-HSD 2508.20546; MoRE (WWW 2025); Reasoning-Aware fusion 2512.02743; MultiHateLoc 2512.10408.
