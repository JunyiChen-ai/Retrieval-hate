# MLLM-FRONT RECON — putting a frozen MLLM *in front* of the small multimodal model

**Date:** 2026-08-11 · **Type:** pure recon + design. **Zero pilots, zero API batch calls, zero
training-code changes.** Deliverable is a design brief for a user ruling.

**Direction as stated by the user.** A (possibly closed-API) MLLM runs on the video *first* and
emits an intermediate artifact — structured analysis, stance description, rationale, attributes.
A normal small multimodal model (this project's frozen features + trained head) then *consumes*
that artifact and makes the final call. **The MLLM is never fine-tuned.** Cited precedent:
HVGuard (EMNLP 2025 Main), which the user characterised as "naive but accepted".

---

## §0 — Executive summary (read this first)

1. **HVGuard is real, is exactly this shape, and its published numbers are at parity with this
   project's *no-mechanism* baseline — and so is the second, closer occupant, RAMF.**
   On HateMM binary macro-F1 the ordering is **MM-HSD 0.874 (no MLLM at all) > our frozen
   three-encoder ensemble 0.8732 (no mechanism) > HVGuard 0.8597 > RAMF 0.837 > MoRE 0.8235**.
   Building this shape competently is not, by itself, a result. (HVGuard: GPT-4o writes a CoT
   rationale; XLM/ViT/Wav2Vec encode text/frames/audio; the rationale goes through the *same text
   encoder* and everything is concatenated into an 8-expert MoE. Full binary macro-F1
   HateMM/MHC-EN/MHC-ZH 0.8597 / 0.7714 / 0.8219 vs our 0.8732 / 0.7776 / 0.8183 — different
   splits, so read it as parity, not as a win either way.)
2. **This project has already run five variants of "MLLM front → frozen-feature downstream" and
   killed all five** (P3 evidence-density pooling, P4 schema-field auxiliary distillation, P7
   score-level fusion, P8/P8b/P8c summary-as-text-input, P9 decision-level). The campaign's
   cross-cutting verdict was *"MLLM semantic capability ⊥ or ⊆ the decision variable."* Any new
   proposal in this shape must say why it is not the sixth.
3. **There is exactly one documented exception to that verdict, and it is the whole reason this
   direction is not dead**: speaker **stance / use-vs-mention**. §9.2 prices it at **45.4 % of all
   108 test errors, mean +6.5 macro-F1**; §8.13 *measured* that the frozen text encoders cannot
   represent it (violation vs exemption prompts cosine 0.83–0.92; clause directions lose to random
   directions by −0.046 mean ROC). So stance is simultaneously (a) the largest error bucket,
   (b) provably absent from the features, and (c) the one family §9.6 marked **OPEN**.
   Every previous MLLM channel we tried was redundant with the label; this one is not — *if* the
   MLLM can produce it.
4. **The topology is occupied everywhere; only two narrow slots are open, and their conjunction is
   the method.** A fresh independent sweep plus the repo's own source-verified
   `research-wiki/MLLM_USAGE_LANDSCAPE.md` agree: MLLM-front→small-model is published and winning
   in memes (Pro-Cap, IntMeme, Tzelepi & Mezaris, M3Hop-CoT, MemeScouts) and now twice in video
   (HVGuard, **RAMF**, TMLR — frozen Qwen2.5-VL-32B, three-perspective text as a fourth modality,
   HateMM 0.837). Even the *typed-slot* wiring is occupied (MemeScouts: frozen Qwen3-VL-30B → 89
   constrained questions → Random Forest, macro-F1 0.85, 1st place). **Open:** (a) *stance /
   use-vs-mention as the artifact content* — nobody feeds speaker attitude to a trained head, in
   any modality; (b) *confidence-gated rather than always-on invocation* on the academic
   hateful-video benchmarks. **The recommended method is their conjunction: wake a frontier MLLM
   only where the memory is uncertain, and ask it for typed speaker stance, not a rationale.**
   (§3, §4.5)
5. **The user's earlier blocker does not apply here.** §9.8/§9.9 froze a 750-human-judgement audit
   *because the stance labels were to be used as gold supervision* (`L_verdict + 0.5·L_stance`).
   In this direction the MLLM output is an **input feature computed identically on train and
   test**, never a training target — the same leakage/validation status as CLIP features, ASR, or
   the OCR cache. Its quality is validated *by the downstream metric*, not by human agreement.
   This is a real and defensible distinction and it should be stated explicitly in any prereg.
6. **Cost is not the blocker.** One structured pass over all four datasets (4,671 videos) is
   **≈ $50–150 one-time** on the Batch API. The real cost item is that the front-end must also run
   at *inference* (~5¢/video on Opus 5), which changes what the method *is* and must be declared —
   though the Role-3 gate below cuts that by 76–90 %.
7. **There is a parked in-house line whose revival condition this proposal literally satisfies,
   and it costs about two dollars to test.** `EVAL_role3_selective_reasoning.md` (2026-07-05):
   the confidence gate is a **positive** result (24 % of EN test captures 42 % of errors; oracle
   arbitration on the slice reaches acc 0.857–0.888), and every 7B arbiter **failed** with a
   frozen anti-repeat clause reading *"revive ONLY with … ≥ 72B or API-class"* against bars of
   ≥ 0.667 (break-even) / ≥ 0.846 (crosses 0.85). One API-class pass over the 39-item deferred
   slice settles it. **Run this before anything else.** (§4.4)

---

## §1 — HVGuard teardown (paper read in full)

**Citation.** Jing, Zhang, Zhuang, Guo, Wang, Xu, Yi, Guo, Hu. *HVGuard: Utilizing Multimodal
Large Language Models for Hateful Video Detection.* EMNLP 2025 Main, pp. 8993–9006,
`10.18653/v1/2025.emnlp-main.456`. **No arXiv id — cite the Anthology/DOI, do not invent one.**
Code: `github.com/yihengjingWHU/HVGuard` (HVGuard.py, CoT.py, cached `.pth` embeddings, cleaned
`annotation(new).json`; the README does not state which MLLM generated the shipped rationales,
nor API cost). Confidence: **HIGH** — full PDF text extracted and read, including all appendices.

### 1.1 What the front-end generates

A **four-stage CoT prompt chain** over GPT-4o (also tested: Gemini-1.5-pro, Qwen-VL-7B):

| stage | input | output |
|---|---|---|
| Adaption prompt | — | role/domain framing ("You are a content moderation specialist…") |
| Visual meaning | 32 uniformly sampled frames | scene/character description, **explicitly instructed to *ignore* subtitles in frames** |
| Textual meaning | title + FunASR transcript | meaning analysis, with a specific instruction to resolve **puns and homophones** |
| Fusion meaning | title + transcript + frames + **voice emotion** + res1 + res2 | free-text rationale on implicit meaning and whether the video expresses hateful content |

Output is a **free-text rationale** `v_i^M`. There is no schema, no typed slots, no stance field.

### 1.2 How the downstream consumes it

`E^T = f_T(title+transcript)`, `E^A = f_A(audio)`, `E^F = f_F(frames)`, and — the whole
integration — **`E^M = f_T(v_i^M)`: the rationale is pushed through the *same* text encoder**.
Then `E = concat(E^T, E^A, E^F, E^M)` → `n=8` two-layer ReLU experts + a linear softmax gate with
dropout → weighted sum → linear → cross-entropy. **Encoders are frozen** (an embedding cache is
built once); only experts + gate are trained. lr 1e-4, batch 32, early stop at 100 epochs,
3×V100. Best config: GPT-4o + XLM + ViT + Wav2Vec.

### 1.3 Datasets and numbers

Two datasets, **filtered** (corrupted/blurry removed, transcripts re-run with FunASR):
HateMM 1,066 (427 H / 639 N); MultiHateClip EN 891 (72/218/601), ZH 897 (112/180/605).
**Random 7:2:1 train/test/val split** — not the source papers' splits, not ours.
Single run per cell; **no seeds, no variance, no significance test.**

| dataset (binary) | HVGuard Acc | HVGuard **M-F1** | best baseline M-F1 | **our round-4 ensemble M-F1** |
|---|---|---|---|---|
| HateMM | 0.8563 | **0.8597** | 0.7594 (MultiHateClip FC) | **0.8732** |
| MHC-EN | 0.8539 | **0.7714** | 0.6806 | **0.7776** |
| MHC-ZH | 0.8603 | **0.8219** | 0.6908 | **0.8183** |
| ImpliHateVid | — | — | — | 0.9276 |

M-F1 is genuinely macro over the two classes (verified: HateMM 0.8479/0.8715 → 0.8597;
MHC-EN 0.6308/0.9120 → 0.7714; MHC-ZH 0.7408/0.9031 → 0.8219). Baselines are five
self-reproduced systems: GPT-4o / Gemini-1.5-pro / Qwen-VL zero-shot, HateMM (Das 2023),
MultiHateClip (Wang 2024). Headline gain "+6.88–13.13 % Acc, +9.21–34.37 % M-F1" is against
*those* reproductions.

**Against our line: HVGuard is +/− nothing.** −1.35 on HateMM, −0.62 on MHC-EN, +0.36 on MHC-ZH;
mean over the three shared datasets 0.8177 vs our 0.8230. On different splits, so treat as
*parity*, not as a win either way. **MM-HSD (ACM MM 2025, `2508.20546`) reports macro-F1 0.874 on
HateMM** — above both, using PaddleOCR at 1 fps as a cross-modal attention query.

### 1.4 Ablations (Table 3, MHC-EN)

| arm | ternary Acc/M-F1 | binary Acc/M-F1 |
|---|---|---|
| w/o vision encoder | 0.7865 / 0.4760 | 0.8202 / 0.7397 |
| w/o text encoder | 0.7753 / 0.5633 | 0.8258 / 0.7090 |
| w/o audio encoder | 0.7697 / 0.5807 | 0.8258 / 0.7413 |
| **w/o modal features (rationale only)** | 0.7584 / 0.4816 | 0.8146 / 0.7126 |
| **w/o CoT (generic prompt only)** | 0.7416 / 0.4715 | 0.7921 / **0.5512** |
| MoE → MLP | 0.7809 / 0.5936 | 0.8371 / 0.7466 |
| MoE → cross-attention | 0.8034 / 0.6525 | 0.8427 / **0.8037** |
| **HVGuard (full)** | 0.8090 / 0.6646 | 0.8539 / 0.7714 |

The CoT ablation is the load-bearing one and it is large (binary M-F1 0.5512 → 0.7714). Everything
else is ≤ 3 points. Appendix E sweeps 16 MLLM×encoder combinations; **MLLM choice dominates**
(GPT-4o+XLM+ViT+Wav2Vec 0.6646 vs Qwen-VL+Bert+ViViT+MFCC 0.4835 ternary M-F1).

### 1.5 Where it is naive / weak — the differentiation raw material

1. **Two of its own ablation numbers contradict the MoE claim.** On MHC-EN *binary*, Table 7's
   `HVGuard(w/o gate)` scores **M-F1 0.8045 vs the full model's 0.7714** — i.e. as printed,
   removing the gating network *improves* macro-F1 by 3.3 points, while accuracy drops. And
   Table 3's cross-attention arm beats the full MoE on binary M-F1 (0.8037 vs 0.7714) too. Worse,
   Table 7's `w/o gate` row is **byte-identical (0.8315 / 0.8045) for the EN and ZH binary cells** —
   almost certainly a duplicated row. The "MoE is crucial" claim rests on accuracy and ternary
   M-F1 only, and is contradicted on the metric we report.
2. **The rationale is a 768-d free-text vector concatenated into a fusion net.** This is exactly
   the integration our A0 end-to-end OCR test measured as **−0.0246 macro-F1** on HateMM
   (`idea-stage/A0_OCR_E2E_RESULT.md`) and the shape P8 killed. Free text through a frozen encoder
   is lossy and dilutive; no typing, no low-dimensional structure.
3. **It explicitly throws away on-screen text**: the vision prompt says *"ignoring subtitles in the
   frames"*. Our §9.4 measurement is that on HateMM the decisive evidence is frequently
   burned-in — MEMRI-TV subtitles, Britain First title cards, and a slur present *only* in
   burned-in text on a video with an empty transcript. HVGuard is structurally blind to that.
4. **Its own residual error is precisely our stance bucket, and it does nothing about it.**
   Appendix C: **74.12 % of false positives involve sensitive terms or profanity**; 89.41 % on
   MHC-EN, with 22.35 % on LGBTQ+ topics; the worked example is a *Walking Dead* clip where
   characters swear under stress. That is "hate-associated surface, benign stance" — the exact
   failure mode §9.2 prices at +6.5 macro-F1. HVGuard names it in an appendix and leaves it open.
5. **Evaluation hygiene**: single run, no seeds, no CIs; a re-filtered corpus and a fresh random
   7:2:1 split, so its numbers are not comparable to the source papers' or to any other work;
   baselines are all self-reproduced; no 2025 comparator (no MM-HSD, no RA-HMD, no CRAVE);
   the "first reasoning-based hateful video detection framework" claim is a framing claim.
6. **No structure at all in the intermediate artifact.** No stance, no speaker attribution, no
   target slot, no evidence-modality typing. The MLLM writes prose; the head sees a vector.

> **The honest read of HVGuard for us:** it is a proof that the *shape* is publishable at a top
> venue, and simultaneously a proof that the *naive instantiation* of the shape lands at our
> current baseline. Its value to us is as the incumbent to differentiate against, and its own
> appendix hands us the differentiation axis.

### 1.6 The user named HVGuard, but **RAMF is the closer and more dangerous incumbent**

*Yang, Chen, Yue, Cheng, Jiao, Fu.* **RAMF: Reasoning-Aware Multimodal Fusion for Hateful Video
Detection.** **TMLR**, arXiv `2512.02743` (v2 May 2026). Already in the repo as
`research-wiki/papers/yang2025_reasoningaware_multimodal_fusion.md` (ingested 2026-07-01) and
dissected in `MLLM_USAGE_LANDSCAPE.md` §5 from the HTML full text and the official GitHub.

**Mechanism.** Offline, per sample: 16 frames + Whisper transcript → **frozen Qwen2.5-VL-32B**
emits three texts — `T_O` objective description, `T_H` **hate-assumed** inference, `T_N`
**non-hate-assumed** inference. Training side: the three original modalities are encoded
(BERT/HateXplain, MFCC+CLAP, ViT+CLIP), the three reasoning texts are encoded too, and fusion is
two-stage — LGCF + SCA over the grounded modalities plus `T_O` → `Y₁`, then `Y₁` refined against
`T_H`/`T_N` via SCA → `Y₂` — into a trained classification head (CE). **Only the fusion modules
train; the VLM never decides.** HateMM Acc 84.3 / **M-F1 83.7** (beats MoRE); MHC-ZH 72.4 / 69.3;
MHC-EN 68.5 / 64.1. Always-on, free text, no schema, nothing reused across samples.

**Why it matters more than HVGuard here.**
1. It is the **strongest published "frozen MLLM text as a feature" system in the video domain**,
   and it is still **below our mechanism-free ensemble** (0.837 vs 0.8732 on HateMM).
2. Its `T_H` / `T_N` pair is **the nearest existing neighbour to a stance idea** — and the gap is
   exact and defensible: RAMF performs **assumption-conditioned adversarial inference**
   (*"suppose this were hateful — argue it"*), whereas the proposal here performs
   **speaker-attitude extraction** (*"what stance does the speaker take toward this content —
   endorse, condemn, report, quote, or depict?"*). One conditions on a hypothesis about the
   *label*; the other reads a property of the *utterance*. Any prereg must state this in one
   sentence, because a reviewer will ask.
3. **Displacement risk is real and concentrated.** RAMF, MARS (ICASSP 2026) and LELA
   (`2602.09637`) are all from the same group (Zeyu Fu's lab), publishing steadily along this
   line. Speaker stance is a natural next prompt for them. This is the single largest
   time-sensitivity in the plan.

---

## §2 — The in-house prior nobody outside this repo has (the real bar)

`research-wiki/TERMINUS_mllm_campaign_DRAFT.md` (FINAL, 2026-07-09) records **11 pre-registered
routes** for giving an MLLM a method role. Five of them are this exact shape:

| route | mechanism | result |
|---|---|---|
| **P3** evidence-density pooling | MLLM scores each window 0–3 → softmax-reweighted pooling of frozen CLIP video embedding | FAIL on all 3 datasets. HateMM had the campaign's cleanest probe (+0.0108, k-consistent) → training within noise |
| **P4** schema-field auxiliary distillation | aux linear heads predict MLLM archive fields (`explicitness`/`modality`/`mechanism`/`target_group`) from the fused embedding, λ=0.1, heads dropped at eval | probe **PASS** (fields decodable AUC .62–.93, and field→label AUC .74–.78) → training **within noise**. Diagnosis: *the fields are redundant with the hate label the head is already supervised on* |
| **P7** score-level fusion | fuse kNN vote share with an MLLM semantic channel under two frozen rules | train-side **KILL**: corr(channel, vote share) = **+0.21…+0.51** (premise of decorrelated errors falsified); all 8 rule×channel cells net −0.10…−0.38 |
| **P8 / P8b / P8c** summary-as-text-input | MLLM writes ≤60-word evidence-dense summary → becomes the text channel, single-chunk encoded, head retrained | FAIL everywhere. EN probe was the campaign's strongest (+1.6 over floor, +4.6 over naive truncation) yet trained **−0.023 / −0.079**, *worse than blind first-70-token truncation*. ZH: no summary variant (text/vision, EN/CN) beats naive raw truncation; root cause is the **frozen English-centric CLIP text tower byte-fragmenting Chinese (97 % truncated at 75 tokens)** |
| **P9 / P9b** decision-level | LoRA-SFT the whole MLLM + read out its head / our kNN memory | EN +0.6, ZH +1.0 vs protocol-matched floor (noise); our kNN read-out **−2.7 / −2.2 / −4.7 below floor** |

**Four transferable constraints fall out, and they are the design spec for anything new:**

- **C1 — do not send MLLM text through the frozen CLIP text tower.** P8c is a complete
  three-arm attribution: the bottleneck is the encoder, not the summary.
- **C2 — do not add a fourth 768-d stream to the fusion MLP.** A0 arm B added +1.84 M params
  (+36.8 %) for a third stream and lost **−0.0246 ± 0.0114** macro-F1, 0/3 seeds positive.
- **C3 — a passing no-head probe is necessary but not sufficient.** Two sharpest instances
  (P3-HateMM, P8-EN). Any new probe gate must be treated as a *filter*, never as evidence.
- **C4 — the MLLM channel must not be decodable from the frozen features.** P4 failed precisely
  because it was. This is a *precommittable* test: measure decodability of the new channel from
  `concat(img, text)` before spending anything downstream, and kill if it is high.

**And one standing counter-fact that makes stance the exception.** §8.13 (PCD spec, closed at the
specification stage): matched violation/exemption policy clauses embed at cosine **0.920** in CLIP
joint space and **0.833 / 0.869** in multilingual mpnet (EN/ZH); the K pair-difference directions
are mutually near-orthogonal (0.035–0.067); with a trained readout the clause directions **lose to
dimension-matched random directions on 3 of 4 datasets (mean −0.046 ROC)**. Verbatim conclusion:
> *"separating hate from condemnation/quotation/reclaimed use must come from a model that reasons,
> not from an embedding direction."*

So: C4 is satisfiable *by construction* for a stance channel, and only for a stance channel. That
is the entire technical argument for this round.

**Existing assets that are already on disk** (relevant to cost and to what a pilot would reuse):
- `data/Summaries/{HateMM,MHC,MHC_zh}/{train,val,test}.jsonl` — 7B text-only ≤60-word summaries
- `data/Summaries_vision{,_zh}/MHC_zh/` — 7B vision-grounded summaries (EN and forced-CN)
- `data/Archive/{MHC,MHC_zh}/…_archive.jsonl` (+ `v2/`) — 7B structured schema:
  `{target_groups, mechanism, modality_cues{visual,speech,on_screen_text}, explicitness,
  neutral_summary}` — **note there is no stance field anywhere in this schema**
- `data/MLLM_scores/…` — per-window evidence-density scores (7B → 72B ladder)
- `data/lora_frames/{HateMM,MHC,MHC_zh}/` — 8 pre-extracted JPGs × 2,661 videos
- `data/OCR/{HateMM,MHC_test,MHC_zh_test}/` — PaddleOCR caches
- `idea-stage/r4_harness.py` — the round-4 head already accepts an `extra` feature block
  (`Head(..., extra_dim=…)`, concatenated post-Hadamard) — i.e. the *naive* wiring is one line,
  which is exactly why the design work below is about **not** using it.

---

## §3 — Placement map: who already owns "MLLM front → small model"

> Independent occupancy sweep, 2026-08-11. Verdicts per sub-family, then the item list.
> `[OPENED]` = PDF/abstract actually read; `[SNIPPET]` = seen in a result page only.

### 3.0 What this project's own prior recon already established (carried forward, not re-derived)

From `research-wiki/NOVELTY_RECON_2026-08-09.md` and IDEA_REPORT §9.6, the relevant standing
verdicts — these bound the sweep below and are the reason the new sweep targets sub-family 5
specifically:

| family (§9.6 naming) | verdict | strongest occupant |
|---|---|---|
| F1 rationale-then-verdict SFT / RLVR | **OCCUPIED** | IARE `2606.11953` (CoT-SFT + DPO on hateful *video*, Ex-HateMM 85.86→90.14); LEAF `2026.findings-acl.604`; ExPO-HM `2510.08630` (ICLR 2026) |
| F2 generative MLLM as classifier | ADJACENT | RA-HMD `2502.13061` (EMNLP 2025 oral); `2501.15438` (WWW 2025); HateClipSeg `2508.01712` |
| **F3 stance / use-vs-mention as SUPERVISION** | **OPEN** | `2404.01651` (NAACL 2024) is **prompting-only** and its Limitations leave fine-tuning unexplored; TANDEM `2601.11178` supervises *target*, not stance; ImpSH `2606.18852` contrasts *implied statement* |
| F5a/b OCR integration | **OCCUPIED** | MM-HSD `2508.20546` (ACM MM 2025, macro-F1 0.874 on HateMM); `2602.09637` (OCR in an LLM prompt on HateMM + MultiHateClip) |

Two things follow. (i) **ExPO-HM is the sharpest published warning against the naive version**:
on Qwen2.5-VL-7B it measured Direct-SFT **75.0** F1 > CoT-SFT 74.5 > GRPO 74.5 — explain-then-detect
*loses* to direct detection. (ii) `2404.01651` already banked an **82.6 % FPR reduction** from
stance-style *prompting*, which kills any inference-time-prompt framing and is precisely why D1–D3
are all *feature-channel* designs rather than prompt designs. The open question the sweep below
must settle is narrower than F3: **not** stance-as-supervision, but **stance as a typed feature
channel consumed by a trained classifier**.

### 3.0b There is already a source-verified in-house landscape for exactly this question

**`research-wiki/MLLM_USAGE_LANDSCAPE.md` (2026-07-02, 215 lines)** dissects every
"LLM/MLLM-for-detection" competitor in this domain from primary sources (arXiv HTML, Anthology
PDFs, GitHub). Its one-line verdict: *"every existing reasoning-VLM method in the video domain is
**always-on, ungated, with no retrieval/memory, and generates text that is used once and
discarded**."* Its method table:

| method | venue | MLLM role | invocation | artifact form | stored / reused |
|---|---|---|---|---|---|
| MARS | ICASSP 2026 | direct verdict (4-stage adversarial) | always-on | label + conf + free text | none |
| **HVGuard** | EMNLP 2025 | rationale as a 4th modality feature | always-on | free text | engineering cache |
| IARE | SIGIR 2026 | fine-tuned MLLM, verdict + rationale | always-on (dual MLLM) | free text | none |
| TANDEM | arXiv 2026 | RL-tuned MLLM, structured verdict | always-on | **XML schema** (label/timestamps/targets/summary) | transient |
| **RAMF** | TMLR | frozen VLM, 3-perspective text as feature | always-on | free text | none |
| LELA | arXiv 2026 | per-frame LLM scoring | always-on (per frame × modality) | free text + scalar | none |
| MoRE | WWW 2025 | **no LLM** | — | — | feature-triplet memory bank |
| RA-HMD | EMNLP 2025 | LMM = encoder | fixed by protocol, no gating | logits / kNN votes | FAISS embedding bank |

It also fixes a **standing writing red line** that constrains any design here: our MLLM role
*"must not be phrased as 'generate caption/description then classify' (Pro-Cap owns it), nor
'LLM rationale as feature/distillation' (HVGuard/RAMF/Mr.Harm own it), nor 'LLM debate/judge'
(ExplainHM/MARS own it)."*

### 3.1 Fresh sweep (2026-08-11) — independent, and it agrees

An independent sweep this session (arXiv listing enumeration + direct fetches, ~75 tool calls)
reached the same verdicts and extended them. Sub-family verdicts:

| # | sub-family | verdict |
|---|---|---|
| 1 | LLM/MLLM captions as an extra text channel (memes) | **OCCUPIED — saturated** |
| 2 | LLM rationales/explanations → smaller classifier | **OCCUPIED — dense, with published negatives** |
| 3 | MLLM knowledge distillation into a small model | **OCCUPIED** |
| 4 | MLLM structured attributes / slot tuples as features | **OCCUPIED** (open until M3Hop-CoT, 2024) |
| 5 | **stance / use-vs-mention as a FEATURE CHANNEL** | **OPEN** |
| 6 | agentic / multi-agent MLLM → trained classifier | ADJACENT (agents prompt-only; pseudo-label branch occupied) |
| 7 | video: MLLM front-end → trained fusion model | **OCCUPIED — two direct occupants** |

**(7) Video — the two direct occupants, and where our line actually sits.**
`[OPENED]` = read; the M-F1 column is *binary macro-F1 on HateMM*, the only number all of these
share.

| system | venue | HateMM M-F1 | note |
|---|---|---|---|
| **MM-HSD** `2508.20546` `[OPENED]` | ACM MM 2025 | **0.874** | no MLLM at all — PaddleOCR@1fps as cross-modal attention query |
| **our round-4 three-encoder pairwise ensemble** | — (no mechanism) | **0.8732** | IDEA_REPORT §8.10 |
| **HVGuard** `2025.emnlp-main.456` `[OPENED]` | EMNLP 2025 Main | 0.8597 | GPT-4o CoT rationale → MoE |
| **RAMF** `2512.02743` `[OPENED]` | TMLR | 0.837 | frozen Qwen2.5-VL-32B, 3-perspective text as a 4th modality |
| MoRE (DOI `10.1145/3696410.3714560`) | WWW 2025 | 0.8235 | retrieval memory, zero LLM |
| MARS `2601.15115` `[OPENED]` | ICASSP 2026 | 0.778 | training-free, the chain decides |
| LELA `2602.09637` `[OPENED]` | arXiv 2026 | (frame-level PR-AUC 72.6) | training-free localisation |
| IARE `2606.11953` `[OPENED]` | SIGIR 2026 | 0.9014 **on its own new Ex-HateMM** | MLLM is LoRA-SFT'd + DPO — *not* our shape |

> **The single most important line in this recon:** *both* published occupants of the exact shape
> — HVGuard (0.8597) and RAMF (0.837) — sit **below** this project's mechanism-free three-encoder
> ensemble (0.8732), and both sit below MM-HSD (0.874), which uses no MLLM at all. Building this
> shape competently is not, by itself, a result.

**RAMF is the closest thing to a stance idea that exists, and the gap is exact.** It asks a frozen
VLM for three texts: an objective description, a **hate-assumed** inference, and a
**non-hate-assumed** inference, then fuses them in two stages (LGCF + SCA) into a trained head.
That is **assumption-conditioned adversarial reasoning** — *"what if this were hateful?"* — not
**speaker-attitude extraction** — *"what attitude does the speaker take toward this content?"*
Same lab as MARS and LELA (Zeyu Fu's group), and they are publishing along this line steadily, so
this is also the **highest displacement risk** in the whole plan.

**(5) Stance / use-vs-mention as a feature channel — OPEN, and precisely so.**
Exhaustive enumeration of the `use-mention`, `counterspeech detection`, and `slur reclamation`
namespaces put every occupant into three non-competing buckets:

- **A — prompt-only mitigation, no trained head.** *Gligorić, Cheng, Zheng, Durmus, Jurafsky,*
  **"NLP Systems That Can't Tell Use from Mention Censor Counterspeech, but Teaching the
  Distinction Helps"**, NAACL 2024 Main, `2024.naacl-long.331` / arXiv `2404.01651` `[OPENED]` —
  names the failure mode, shows it propagates into hate and misinformation detectors, and fixes it
  **by prompting**: FPR on counterspeech −82.61 % (hate) / −59.06 % (misinfo). It explicitly
  leaves the architectural solution on the table. *Goldzycher & Schneider,* **"Hypothesis
  Engineering for Zero-Shot Hate Speech Detection"**, TRAC@COLING 2022, arXiv `2210.00910`
  `[OPENED]` — quotation identification → hate classification of the quoted span → **stance of the
  surrounding text toward the quote**; zero-shot NLI composition, no trained head, no LLM.
  HateCheck 79.4→87.3, ETHOS 69.6→79.6. **The closest conceptual ancestor by far.**
- **B — reclamation/stance as the *task label*, not a feature.** MultiPRIDE@EVALITA 2026 (slur
  reclamation shared task); *AIWizards*, arXiv `2602.12818` `[OPENED]` — LLM weak-annotates
  *community membership*, soft-labels a BERT, transfers its representation into the reclamation
  classifier; the artifact is speaker **identity**, not speaker **stance**, and it did not beat a
  strong BERT baseline. PrideMM/MemeCLIP `2409.14703` and DARC-CLIP `2604.23214` have stance as a
  *parallel output head*, never an input to the hate head.
- **C — the architecture exists, on a different task.** *Gatto, Sharif, Preum,* **"Chain-of-Thought
  Embeddings for Stance Detection on Social Media"**, Findings of EMNLP 2023,
  `2023.findings-emnlp.273` / arXiv `2310.19750` `[OPENED]` — frozen ChatGPT CoT text → encoded →
  fed as an **additional feature** into a RoBERTa pipeline. Biden subset 50.6 → **71.3 F1**.
  Our exact topology, executed on stance-*as-task*.

> **What does not exist anywhere the sweep could reach:** a frozen (M)LLM emitting an explicit
> *speaker-attitude / use-vs-mention / endorse-condemn-report-quote* judgment that a **trained**
> hate/toxicity/misinformation head consumes as an **input feature channel**. Not in text, not in
> memes, not in video.

**(4) Structured attributes — occupied, and this is the sharpest constraint on D2.**
*M3Hop-CoT*, EMNLP 2024 Main `2024.emnlp-main.1234` / arXiv `2410.09220` `[OPENED]`: three-hop
prompting emits three typed slots (emotion / target-awareness / social context), CLIP-encoded,
hierarchically cross-attended into the meme representation; MAMI macro-F1 91.75 dev / 80.28 test.
*MemeScouts@LT-EDI 2026*, arXiv `2604.24179` / `2026.ltedi-1.23` `[OPENED]`: **frozen Qwen3-VL-30B
answers 89 constrained questions → an integer feature vector → a 500-tree Random Forest**, macro-F1
0.85 EN (1st place). **This is D2's wiring, already published.** The sweep confirmed explicitly:
*none of the 89 questions concerns speaker stance, use-vs-mention, or endorsement-vs-condemnation.*
Also *Tzelepi & Mezaris*, CVPRW 2025 (MULA), arXiv `2504.09914` `[OPENED]`: frozen MiniGPT-4 → 10
descriptions + 10 emotions → **frozen CLIP** → concat → 3-layer trainable head; Harm-C acc 87.23.
**The exact target shape, published, in the image domain.**

**(2) Rationales → smaller classifier — the two results that most constrain the *wiring*.**
*ARG / ARG-D*, **AAAI 2024**, arXiv `2309.12247` `[OPENED, full PDF]` — the canonical citation and
the **mandatory gating baseline**: frozen GPT-3.5 rationales + a *Rationale Usefulness Evaluator*
supervised by whether the rationale led to a correct verdict, plus scalar reweighting. Weibo21 /
GossipCop macro-F1: BERT .753/.765 → ARG .784/.790, but **plain concatenation = .767/.777** —
i.e. the entire learned trust-gating apparatus buys **~1.3–1.7 points**. It also ships a cost
cascade (route 23 % of items to the LLM, recover full performance) — *published prior art for
D0's economics*. And *Brook & Markov*, arXiv `2510.15685` `[OPENED]` — compares four consumption
mechanisms (text concat / **embedding concat** / hierarchical transformer fusion / LLM-driven
enhancement) for LLM artifacts in hate detection: **embedding concatenation wins**. A second
(Idiap-group) negative reports MoE / adaptive fusion / learnable-query give no significant gain
over concatenation.

> **Consequence for §4, and it is a correction to my own first draft.** I designed D1/D2 to avoid
> concatenation because *our* A0 measured concat at −0.0246. The literature says the opposite for
> LLM-text artifacts. Both can be true — A0 concatenated a 768-d **OCR embedding** into a Hadamard
> fusion head, not a low-dimensional typed vector — but the honest position is: **the wiring is not
> the contribution and must not be claimed as one. The contribution is what the artifact says.**
> Plain concatenation therefore becomes a **mandatory arm** in S3, not a strawman.

**Also relevant and previously known in-house** (`MLLM_USAGE_LANDSCAPE.md` scan point 3, and
independently the sweep's family 6): **confidence-gated selective MLLM invocation is OPEN on the
academic hateful-video benchmarks.** No paper on HateMM / MultiHateClip / ImpliHateVid wakes an
MLLM only when a light model is uncertain; every reasoning-VLM method above is always-on (MARS
even states in text that its confidence score is *"intended solely for interpretability rather
than thresholding"*). Boundary citations that must be made: **Filter-And-Refine** (TikTok,
arXiv `2507.17204`) routes by *embedding similarity to a seed bank*, not by classifier
uncertainty, and cuts 97.5 % of traffic; **Google `2406.12800`** escalates to *humans*, not to an
MLLM; meme-side **LMM Agents** `2411.05383` retrieves labelled samples into the prompt but is
always-on; **ARG**'s cost cascade is the text-domain precedent.

### 3.2 The honest bottom line of the placement map

1. **The topology is occupied everywhere** — memes (Pro-Cap, PromptHate, IntMeme, Tzelepi &
   Mezaris, M3Hop-CoT, MemeScouts) and, since EMNLP 2025 / TMLR, video (HVGuard, RAMF).
   *Topology alone is worth nothing.*
2. **The typed-slot wiring is also occupied** — M3Hop-CoT and MemeScouts in memes, TANDEM's XML
   schema in video. *D2's architecture is not the novelty.*
3. **The fusion machinery is occupied *and* empirically discounted** — ARG's own ablation and
   Brook & Markov both say elaborate fusion buys ~1 point or less over concatenation.
   *D1's logit-level wiring is risk control, not a contribution.*
4. **Two things are open, on two independent verifications**, and they are the whole of what this
   direction has:
   - **(a) the artifact content**: speaker stance / use-vs-mention as a typed feature channel
     (fresh sweep; consistent with §9.6's F3 = OPEN and with MemeScouts' 89 questions containing
     none);
   - **(b) the invocation policy**: confidence-gated rather than always-on, on academic
     hateful-video benchmarks (`MLLM_USAGE_LANDSCAPE.md`, source-verified 2026-07-02).
5. **Their conjunction is the strongest available position**, and it is not a coincidence that
   this project already owns half of it: **a stance-typed front-end invoked only where the small
   model is uncertain.** Not "always-on rationale-as-feature" (occupied twice in video), not "89
   generic attribute questions" (occupied in memes) — but *the one attribute the frozen encoders
   provably cannot represent (§8.13), extracted only where the head is unsure (Role-3's gate,
   already validated in-house).*

---

## §4 — Designs (three new + one already parked in-house)

All three share a common front-end contract and three common downstream rules derived from §2.

### 4.0 The shared front-end contract (what the MLLM emits)

Not prose. A **typed proposition record**, one row per hate-relevant proposition found in the
video, emitted as strict JSON (`output_config.format`, `strict: true`):

```
{ "propositions": [ {
    "surface":        "<≤25-word quote of the hate-bearing material>",
    "carrier":        "speech" | "on_screen_text" | "visual" | "audio_nonverbal",
    "voice":          "uploader" | "on_screen_speaker" | "quoted_third_party"
                      | "archival_source" | "caption_overlay",
    "stance":         "endorses" | "condemns" | "reports" | "quotes_mentions"
                      | "depicts_without_comment",
    "target":         "<free text group, or null>",
    "reclaimed":      true | false,          // in-group reclaimed use
    "confidence":     0.0 – 1.0
  } ],
  "video_stance":     <the same 5-way label at video level>,
  "hate_surface_present": true | false,      // is there ANY hate-associated surface at all
  "decisive_carrier": "speech" | "on_screen_text" | "visual" | "none"
}
```

Design notes: (i) `hate_surface_present` × `video_stance` is the 2×5 cell structure that the
downstream actually consumes — it separates "no hateful surface" from "hateful surface, benign
stance", which is the S bucket; (ii) `carrier` recovers the on-screen-text channel HVGuard
discards and that §9.4 showed is complementary on HateMM; (iii) `voice` is the use-vs-mention
axis proper (who is speaking, not what is said); (iv) nothing here is free text that gets
embedded — every consumed field is categorical (C1).

Inputs to the call: 8 pre-extracted frames (already on disk for 3 of 4 datasets) + title +
transcript + OCR window text. **Identical prompt and identical procedure on train, val and test** —
this is unsupervised input processing, same leakage status as CLIP/ASR/OCR extraction, and must be
asserted in code (one function, one frozen prompt hash, all splits in one job).

---

### 4.1 **D1 — Stance-Conditional Logit Offset (SCLO)** · *recommended*

**Mechanism in one line.** The existing head is untouched; a stance-typed additive term shifts its
logit, so the model keeps one ranking function but gets a different decision boundary depending on
*who is speaking and in what stance*.

**Downstream wiring.** `logit_final = z(x) + Σ_s p_s · b_s + b_0 · 1[hate_surface_present]`, where
`z(x)` is the frozen round-4 head's logit, `p_s` the MLLM's 5-way stance posterior (or one-hot),
and `b_s` **five learned scalars** fit on train only. **Total added parameters: 6.** Ablation is
literally `b = 0`.

**Why this routes around every recorded failure.** C1: no text encoder touched. C2: zero capacity
added to the fusion MLP — the A0 failure cannot occur by construction. C4: the channel is a
5-way categorical the encoders provably cannot represent (§8.13). P7's failure (channel correlates
with, and is weaker than, the decision variable) is sidestepped because the channel is *not a
second classifier* — it never produces a score, only a per-cell threshold.

**Anticipated objection, and the answer.** §8.4 measured that a *global* test-label-oracle
threshold buys only +1.2 to +4.6 macro-F1, which looks like a ceiling. It is not the binding one:
that measurement is for **one** threshold over the whole score axis. SCLO partitions the score
space into 5 (or 10, with the surface flag) cells and gives each its own threshold — a strictly
larger hypothesis class. **This must nevertheless be measured before committing**: compute the
5-cell oracle-threshold ceiling on train/val first. If it is under ~+2, D1 is capped and should be
dropped for D2.

**Which errors it repairs.** Pure-S false positives with high `z` (protest song containing a slur,
segregationist archival footage, counter-speech exposé, the *Walking Dead* clip): these get a
per-cell threshold raised. Pure-S false negatives (news *report* labelled hate) get one lowered.
It does **not** repair X-bucket ranking errors (34.3 %).

**Differentiation.** vs HVGuard: no MoE, no rationale embedding, no added fusion capacity — a
6-parameter typed calibration. vs any calibration paper: the partition is a *semantic*
speaker-stance type produced by a frozen reasoner, not a confidence bin.

**Risks.** (a) The oracle-cell ceiling may be small — mitigated by measuring it first, free.
(b) Class-cell sparsity: `quotes_mentions` may have <20 train items on some datasets; needs a
shrinkage prior toward `b_0`, frozen in advance. (c) It is a decision-rule mechanism, and the jury
has previously ruled decision-rule mechanisms low-ceiling; the counter is that the *conditioning
variable* is new, not the rule.

---

### 4.2 **D2 — Typed Evidence Vector + logit-level gate (TEV)** · *the main arm; note §3.1(4) — this wiring is occupied (MemeScouts, M3Hop-CoT), so the novelty must live in the stance fields, not the vector*

**Mechanism in one line.** The proposition set is discretised into a ~40-dim multi-hot *typed
evidence vector*; a logistic regression over it produces a second logit that is combined with the
head's logit through a single validation-fit scalar — never concatenated into the fusion MLP.

**Downstream wiring.**
`v(x)` = [5 stance one-hot at video level] ⊕ [5 voice × 5 stance co-occurrence multi-hot]
⊕ [4 carrier bits] ⊕ [`hate_surface_present`, `reclaimed`, `decisive_carrier` one-hot]
⊕ [log(1+#propositions), max confidence] ≈ **40 dims, all categorical or counts**.
`u(x) = w·v(x)` (logistic regression, ~41 params, trained on train split).
`logit_final = z(x) + α·u(x)`, α a **single scalar** fit on validation. Ablation: α = 0.

**Why not concatenation.** Because §2/C2 measured concatenation losing. The logit-level
combination keeps the head's learned geometry intact and makes the contribution of the MLLM
channel exactly one interpretable number.

**Mandatory pre-registered gate (this is the P4 lesson turned into a kill switch).** Fit a probe
`concat(img_feats, text_feats) → v(x)` on train. **If mean per-field AUC ≥ 0.80, the channel is
redundant with the features and the design is DEAD before any head is trained.** P4 died of
exactly this and we did not measure it in advance. Symmetrically, measure `v(x) → label` AUC on
train: if it is below ~0.60 the channel carries nothing.

**Which errors it repairs.** Everything D1 repairs, plus O-bucket items (the `carrier` and
`decisive_carrier` fields expose on-screen-text-only evidence that the CLIP image tower cannot
read — the channel HVGuard explicitly discards) and part of M (silent videos, where
`decisive_carrier ∈ {on_screen_text, visual}` is informative and the transcript is empty).

**Differentiation.** The whole method is "an MLLM fills a *typed evidence slot-set*, and a
40-parameter model reads it at the logit". Against HVGuard: structured vs prose, 40 params vs an
8-expert MoE, logit-level vs feature-level, on-screen text kept vs explicitly discarded. Against
the LLM-caption / rationale-augmentation literature (§3): those append text to a text channel;
this never produces text the downstream reads.

**Risks.** (a) The redundancy probe may fire — that is a *feature* of the design (cheap, honest
kill) but it is a real ~40 % chance given P4. (b) `reclaimed`, `archival_source` will be rare;
the vector must be frozen in advance with an OTHER bucket, no post-hoc field selection.
(c) 40 dims × ~550–1,300 train rows is fine for logistic regression but the α fit must be on
validation, not test, and must be single-shot.

---

### 4.3 **D3 — Stance-Contrast Regularisation (SCR)** · *highest ceiling, highest risk*

**Mechanism in one line.** For each *training* item, the front-end also emits a **minimally
stance-flipped rewrite of the same material** (endorse ↔ condemn, holding target, slur, topic and
register fixed); the head is trained with an auxiliary paired-margin loss forcing its score to
move in the label direction between the pair — installing stance as a usable direction in a space
that §8.13 proved does not otherwise contain one.

**Downstream wiring.** Encode the rewritten text with the *same* frozen text encoder to get
`x̃ = (img, text̃)`. Loss `L = L_BCE(z(x), y) + λ · max(0, m − sign_flip·(z(x) − z(x̃)))`,
λ and m frozen in advance, image features **shared and identical** across the pair. Front-end runs
on **train + val only**; **test is untouched by the generator**, so unlike D1/D2 there is no
inference-time API cost and no method-declaration issue.

**Why this is not P5 / not B-SRTD.** P5 (counterfactual twin negatives) failed on two measured
grounds: a quality gate (7B self-verdict flip rate 0.503 EN / 0.337 ZH, against a 0.80 bar) and a
mechanism problem (twins used as *hard negatives inside the RGCL contrastive loss*, where the
shared visual anchor at cosine 0.73 pulled positives apart). D3 changes both: the generator is a
frontier closed model, not 7B (the flip-rate gate is re-run and is the kill switch); and the pair
enters as a **directional margin on the decision score**, never as a negative in a metric space,
so the shared-anchor pathology cannot occur. B-SRTD was blocked on *building* a balanced lattice
with human verification; D3 does not need a balanced lattice, only per-item pairs, and its
quality gate is machine-checkable (does an independent verifier model assign the flipped label?).

**Pre-registerable quality gate (frozen before generation).** A second, context-isolated instance
of the same model, shown *only* the rewritten material with no knowledge of the source label,
must assign the flipped stance on **≥ 0.80** of pairs, per dataset — the same bar P5 set and
missed. Below that, D3 is DEAD without a single training run. This gate is machine-checkable and
does not implicate the §9.9 human-audit trigger, because the rewrites are training *inputs* whose
value is measured by the downstream metric, not gold labels.

**Risks.** (a) The flip-quality gate can fail again — but it is cheap and pre-registerable.
(b) The rewrite is text-only while the *label* is video-level, so on visually-carried items the
pair is a no-op; expect the gain to concentrate on speech-carried items (71 % of EN hate evidence
is speech-borne, per the P8 grounding). (c) This is the one design where "MLLM-generated training
signal validated by machines" is a fair criticism — it is weaker than D1/D2 on that axis, and the
round-2 rejection reason ("the capability you advertise is an evaluation artefact") has some grip.
(d) Requires touching the training loop (D1/D2 do not).

---

### 4.4 **D0 — the already-parked design whose revival condition this proposal literally satisfies**

**This is the most decision-relevant thing in the recon and it was already on disk.**
`research-wiki/ideas/role3-selective-reasoning.md` + `EVAL_role3_selective_reasoning.md` (2026-07-05,
jobs 12279/12288/12305) — and note its `based_on` field already cites
`paper:jing2025_hvguard_utilizing_multimodal`, i.e. the project logged HVGuard on **2026-07-01** and
built a line off it.

**Role-3 = confidence-gated selective reasoning.** Route only low-margin kNN decisions to a frozen
MLLM arbiter (16 frames + title/transcript + the video's own archive + top-5 neighbour evidence
cards, strict JSON verdict), replace only the deferred verdicts.

Measured, on MHC-EN/ZH:

- **The gate is a positive result.** EN test @30 % deferral: **24 % of samples capture 42 % of the
  kNN errors** (slice error rate 33 % vs 15 % outside); the deferred slice skews to the
  Hateful/Offensive boundary. **Oracle arbitration on that slice reaches EN acc 0.857–0.888** —
  the gate leaves genuine headroom.
- **All three 7B arbiter generations FAIL the val gate** (v1 generic prompt, v2 rubric-calibrated,
  v3 task-LoRA): every (prompt × rate) candidate is below before-acc on val in both languages
  (EN best 0.7750 < 0.7875; ZH 0.8590 < 0.8718), so the val-selected config is literally
  *"do not arbitrate"*. Quality is monotone v1→v2→v3 (EN deferred-acc 0.462 → 0.487 → **0.615**)
  but the 7B ceiling sits **below the 0.667 break-even line**, and 0.846 would be needed to cross
  0.85.
- **Frames contributed ≈ 0** to arbitration (text-only identical or better, 2.6× cheaper).
- **Selective calls save 76–90 % of MLLM calls vs an always-on pipeline.**
- **Frozen anti-repeat clause, verbatim:** *"Do not iterate further prompts/LoRA at 7B on this
  boundary slice. Revive ONLY with an arbiter that clears EN deferred@30 % ≥ 0.667 (break-even) /
  ≥ 0.846 (crosses 0.85) — i.e. **≥ 72B or API-class**."*

**Two consequences.**

1. **The user's proposal supplies exactly the missing ingredient of an already-quantified,
   already-pre-registered, gate-positive line.** A single API-class arbitration run over the
   EN@30 % deferred slice — **39 videos, ~$2** — answers a question the project has already
   framed, with a bar frozen in advance (≥ 0.667 / ≥ 0.846). That is the cheapest high-information
   experiment available anywhere in this direction.
2. **It is also the cost lever for D1/D2.** §5's inference-time objection (~5¢/video on every test
   item) is dissolved by running the front-end **only on gated items**: 76–90 % of calls saved,
   and the gate is already implemented and validated.

**Caveats, stated honestly.** (a) Role-3 is *verdict replacement*, i.e. the F2 "generative MLLM as
classifier" family — ADJACENT per §9.6, nulled in-house by P9/K5, and selective prediction /
disagreement-driven abstention is occupied per `NOVELTY_RECON_2026-08-09.md` leg (iii). **D0 is a
probe and a cost lever, not the paper's mechanism.** (b) Its numbers are on the older archive-kNN
base (EN acc 0.8075), in *accuracy*, not against the round-4 ensemble in macro-F1 — any revival
must re-base. (c) It tests the frontier model's *verdict* quality, which is related to but not the
same as its *stance-labelling* quality, which is what D1/D2 actually need.

### 4.5 Ranking and recommendation

| | mechanism novelty vs §3 | risk of a §2-style repeat | code touched | test-time API | error buckets hit | cost to first verdict |
|---|---|---|---|---|---|---|
| **D0 Role-3 revival** | **low** (F2/selective-prediction, occupied) — a *probe*, not the paper | n/a (bar frozen 2026-07-05) | none (harness exists) | gated only | boundary slice | **≈ $2** |
| **D1 SCLO** | moderate | **lowest** (6 params, cannot dilute) | none (post-hoc on stored logits) | yes (gateable) | S | ≈ $15–30 |
| **D2 TEV** | **content high / wiring occupied** | medium (P4 redundancy is the real risk, and is gated) | head wrapper only | yes (gateable) | S, O, M | ≈ $15–30 (shared with D1) |
| **D3 SCR** | moderate-high | high (P5 precedent) | training loop | **none** | S (speech-carried) | ≈ $15–30 + train-time only |

**Recommendation — and §3 changes it. The recommendation is a conjunction, not a menu.**

§3.2 found exactly two open slots and this project already owns half of one of them. The defensible
method is their **conjunction**:

> **Gated Stance Typing** — a frozen frontier MLLM is woken **only where the small model's
> retrieval memory is uncertain** (Role-3's validated margin gate), and on those items it emits a
> **typed speaker-stance record** rather than a rationale; a trained head consumes the typed
> record. Against always-on rationale-as-feature (HVGuard, RAMF — both *below* our baseline) the
> difference is *what is extracted* and *when it is extracted*.

That framing satisfies `MLLM_USAGE_LANDSCAPE.md`'s writing red line (it is not
caption-then-classify, not rationale-as-feature, not debate/judge), it clears both
`2404.01651` (prompt-only) and `2310.19750` (stance-as-task), and it inherits Role-3's already-
measured gate positive.

Execution order:

1. **D0 first — ≈ $2, and it answers a question frozen eight weeks ago.** One API-class
   arbitration pass over the stored 39-item EN@30 % deferred slice against the frozen
   ≥ 0.667 / ≥ 0.846 bars. It is simultaneously (i) the cheapest evidence on whether a frontier
   model helps at all on *this project's* hardest slice, and (ii) the invocation-policy half of
   the conjunction above. If it cannot beat 0.667, that is strong prior evidence against D1–D3.
2. **Then D1 + D2 off a single shared front-end pass.** They consume the *same* JSON — D1 is a
   strict sub-model of D2 (α·u restricted to the stance one-hot), so D1 is a free nested ablation
   arm, not a separate expenditure. **Run the front-end through the Role-3 gate**, which converts
   the inference-cost objection into a *contribution* (76–90 % of calls saved) instead of a
   liability.
3. **Hold D3 in reserve.** Only design needing a training-loop change, only one exposed to the
   "machine-validated training signal" criticism, and its premise falls out of the D1/D2 pass free.

**Three arms are now mandatory in any prereg, because §3 says the wiring is not the story:**
(i) **plain concatenation** of the typed vector into the head — Brook & Markov's winner and the
honest strawman-killer; (ii) an **ARG-style learned trust gate**, whose own ablation says it buys
~1.3–1.7 points over concat, as the ceiling on wiring cleverness; (iii) **α = 0 / b = 0**, the
remove-the-MLLM ablation. If the typed-stance channel does not beat plain concatenation of a
*generic* attribute vector (the MemeScouts control), the claim is about the wiring and is dead.

---

## §5 — Costs, and the thing that must be declared

**Front-end generation, one pass, all four datasets (4,671 videos).** Assumes 8 frames/video
(already extracted for 2,662 of them, `data/lora_frames/`), ~750–800 image tokens/frame,
~600 prompt tokens (prompt-cacheable), ~400 output tokens. **Image tokens dominate** — measured
transcript lengths are small (median chars / mean chars: HateMM 673 / 1,290 · MHC-EN 387 / 454 ·
MHC-ZH 106 / 133 · ImpliHateVid 974 / 1,714, i.e. ~35–430 tokens), so ~7.4 K input tokens/video is
the working figure. ImpliHateVid has **no local raw video** (`data/video/ImpliHateVid/` holds only
an id→path TSV) so it is text-only unless re-downloaded, which also caps what any stance channel
can see there.

| model | 2,662 frame-bearing videos | + 2,009 text-only | **total, Batch API (−50 %)** |
|---|---|---|---|
| Claude Opus 5 ($5/$25 per MTok) | $136 | $35 | **≈ $86** |
| Claude Sonnet 5 ($3/$15; intro $2/$10) | $81 | $21 | **≈ $51** (≈ $34 at intro) |
| Claude Haiku 4.5 ($1/$5) | $27 | $7 | **≈ $17** |

**One-time generation cost is ~$50–150 and is not a decision-relevant constraint.** Frames may
already be enough at reduced resolution; the Batch API halves everything; prompt caching cuts the
fixed prompt.

**The cost that *is* decision-relevant: D1 and D2 need the front-end at inference time.**
≈ **$0.05/video on Opus 5** (~$0.02 on Sonnet 5). Consequences that must be written into any
prereg and any paper:
1. The method is no longer "frozen features + a head". It is "one frontier-API call per video +
   frozen features + a head". The efficiency comparison against MM-HSD (PaddleOCR + attention) and
   even against HVGuard (which also calls GPT-4o per video, so this is *parity* with the incumbent,
   not a new sin) is unfavourable and must be stated, not buried.
2. Reproducibility depends on a closed model version. Pin the model id and generation date; ship
   the generated JSON as an artifact so the downstream is reproducible without API access.
3. **D3 does not have this problem** — its front-end is train-time only.

---

### 5.1 Suggested pilot shape (if the user says go) — staged, cheapest-kill-first

Four stages, each with a pre-registered kill condition, each cheap enough that the *earlier* ones
can kill the direction before the expensive one runs. **Total ≈ $30–90 API + ≈ 1 GPU-hour.**

**Correction to an obvious-looking free check.** `idea-stage/r5_buckets.json` carries per-item S/O/M/…
codes **only for the 108 test errors**, not for correct items and not for train/val. So the
cell-conditional oracle ceiling cannot be computed for free: the *bucket* ceiling is already known
(§9.2, oracle-fix S = +3.47 / +10.48 / +8.90 / +3.00, mean **+6.46**), but the fraction of it a
5-cell threshold captures needs stance labels on *every* item, which only the front-end produces.
The D1 ceiling check therefore lands **after** S1, not before it — stated here rather than
discovered later.

| # | stage | cost | pre-registered kill |
|---|---|---|---|
| **S-0** | **D0 / Role-3 revival probe.** One API-class arbitration pass over the stored EN@30 % deferred slice (39 videos, text-only — frames contributed ≈0). Bars were frozen 2026-07-05. | **≈ $2** | deferred-acc **< 0.667** ⇒ the frontier model does not beat this project's hardest slice; treat as strong prior evidence against D1–D3 and require the user to re-authorise before spending further |
| **S1** | **Front-end on one dataset, all splits** (HateMM: 744+107+215 = 1,066 videos, frames on disk). Frozen prompt, strict JSON, one job, one prompt hash. Test inputs only — labels untouched. | ≈ **$15–30** batch (Opus 5) | JSON parse rate < 0.95, or `video_stance` degenerate (> 90 % one class) ⇒ the prompt is broken; stop and re-spec **before** spending on any other dataset |
| **S2a** | **D1 ceiling, on val.** Cell-conditional oracle-threshold gain over the global-threshold baseline, val split only. | $0, CPU-minutes | val cell-conditional gain **< +2.0** ⇒ **D1 DEAD**; D2/D3 may still proceed |
| **S2b** | **Redundancy + informativeness probe — the P4 gate we never ran.** Train split only, no head trained. (a) `concat(img,text) → v(x)` per-field AUC; (b) `v(x) → label` AUC. | $0, CPU-minutes | **(a) mean AUC ≥ 0.80 ⇒ DEAD** (channel already in the features — P4's exact failure mode). **(b) AUC < 0.60 ⇒ DEAD** (channel carries nothing). Both must pass |
| **S3** | **Downstream, single submission.** D2 with D1 nested, **plus the three mandatory arms of §4.5** (plain concatenation; ARG-style learned trust gate; α = 0). 3 seeds, α fit on val, **test touched once**. Comparator = the frozen §8.10 three-encoder pairwise ensemble. | ≈ 1 GPU-h | mean Δ macro-F1 **< +1.0** over the frozen comparator, or < 2/3 seeds positive, or the α = 0 ablation not clearly worse, **or the typed-stance vector not beating a generic-attribute vector under the same wiring** (the MemeScouts control) ⇒ honest kill |

Only if S3 passes on HateMM does the front-end run on MHC-EN / MHC-ZH (a further ≈ $25–50).
The ordering deliberately puts **both free gates before any GPU spend**, and puts the P4-style
redundancy gate first among them — that is the one process correction this recon takes from §2.

### 5.2 Why this would not simply be the sixth kill — stated plainly

| the five in-house kills | why D1/D2 is not that |
|---|---|
| P3 (density pooling) reweighted the *frozen visual embedding* | D1/D2 never touch the embedding; they act at the logit |
| P4 (schema distillation) used fields **decodable from the features** and redundant with the label | S2 makes exactly that a *pre-registered kill gate* rather than a post-hoc diagnosis; and §8.13 measured that *stance specifically* is not decodable |
| P7 (score fusion) assumed decorrelated error channels; measured corr +0.21…+0.51 | the stance channel is not a second classifier and produces no score to correlate; it supplies a conditioning variable |
| P8 (summary as text input) pushed MLLM prose through the frozen CLIP text tower | nothing consumed by D1/D2 is text; every field is categorical |
| P9 (decision-level SFT) fine-tuned the MLLM | the MLLM is frozen and closed by construction |

And the one that is *not* answered: **the campaign's cross-cutting verdict "MLLM semantics ⊥ or ⊆
the decision variable" could still be true for stance too.** The only evidence that it is not is
§8.13's measurement that the frozen encoders cannot represent stance — which establishes
*non-redundancy*, not *usefulness*. S2(b) is the test of usefulness, and it is honest that it
might fail.

---

## §6 — What is explicitly *not* being done here, and one disclosure

No pilot. No batch annotation call. No API call of any kind. No training-code change. No prereg
written yet. The next action is a **user ruling**, not an executor decision, because it commits
API spend and picks a direction.

**Disclosure.** Preparing this brief read `idea-stage/r5_error_dump.json` (test-split transcripts
and OCR — inputs) and `idea-stage/r5_buckets.json` (the per-item error coding of the 108 test
errors, which is derived from test labels), both under the user's 2026-08-09 protocol ruling and
both previously disclosed in §9.11(a). **No metric was computed, no threshold or design parameter
was fitted, and no candidate score was produced.** The material was used to check that the
proposed front-end schema (`carrier`, `voice`, `stance`) actually spans the observed failure
modes. The one artefact this changed is §5.1's correction that the bucket coding covers *errors
only*, so the D1 ceiling check is not free.

The only file this session modified outside `idea-stage/` is
`research-wiki/papers/jing2025_hvguard_utilizing_multimodal.md`, whose body was `_TODO._` and is
now filled from the full paper read in §1 (venue/DOI corrected, method/results/limitations/claims
written). No experiment, code, or configuration was touched.

## §7 — The decision the user is being asked to make

0. **Authorise the ≈ $2 D0 probe first?** It is the cheapest decision-relevant experiment in the
   project's backlog and its bar was frozen eight weeks ago. Recommended yes regardless of the
   answer to (1).
1. **Go / no-go on the shape at all**, given that (a) HVGuard *and* RAMF prove it publishes,
   (b) both sit **below** our existing mechanism-free ensemble on HateMM, and (c) §3 finds the
   topology, the typed-slot wiring and the fusion machinery all occupied. The only things open are
   the **artifact content** (speaker stance) and the **invocation policy** (gated, not always-on).
2. If go: the recommended object is their **conjunction — "Gated Stance Typing"** (§4.5), executed
   as D0 → D1+D2 off one shared front-end pass. D3 stays in reserve.
3. **Confirm the ruling in §0.5** — that MLLM output used as an *input feature computed identically
   on all splits* does not inherit the §9.9 750-human-judgement trigger, which was written for
   stance-as-supervision. If the user disagrees, the direction is blocked on the same funding
   question as before and should be closed.
4. **Front-end model and budget**: Opus 5 (~$86 batch) vs Sonnet 5 (~$51) vs Haiku 4.5 (~$17).
   HVGuard's Appendix E is direct evidence that **MLLM quality dominates every other factor** in
   this shape, which argues for Opus 5 despite the price.

---

## §8 — Reproducibility index for this recon

| artifact | path / id |
|---|---|
| HVGuard PDF (extracted text) | `aclanthology.org/2025.emnlp-main.456.pdf` → `pdftotext -layout` |
| HVGuard code | `github.com/yihengjingWHU/HVGuard` |
| in-house MLLM campaign, all 11 routes | `research-wiki/TERMINUS_mllm_campaign_DRAFT.md` |
| **D0 / Role-3: gate positive, 7B arbiter fails, revival bar frozen** | `research-wiki/ideas/role3-selective-reasoning.md` · `research-wiki/EVAL_role3_selective_reasoning.md` · raw `scripts/role3/out/` |
| HVGuard already in the wiki since 2026-07-01 | `research-wiki/papers/jing2025_hvguard_utilizing_multimodal.md` — **body filled this session from the §1 read** |
| **source-verified in-house occupancy map of this exact question** | `research-wiki/MLLM_USAGE_LANDSCAPE.md` (2026-07-02): method table, writing red lines, scan point 3 (confidence-gated invocation = OPEN on academic benchmarks) |
| RAMF node | `research-wiki/papers/yang2025_reasoningaware_multimodal_fusion.md` · arXiv `2512.02743` |
| fresh sweep, key ids | `2404.01651` (NAACL 2024, use-vs-mention, prompt-only) · `2210.00910` (Hypothesis Engineering) · `2310.19750` (CoT embeddings, stance-as-task) · `2410.09220` (M3Hop-CoT) · `2604.24179` (MemeScouts, 89 questions → RF) · `2504.09914` (Tzelepi & Mezaris) · `2309.12247` (ARG — mandatory gating baseline) · `2510.15685` (Brook & Markov — concat wins) · `2507.17204` (Filter-And-Refine — similarity routing) |
| P8 / P8b / P8c summary-as-input | `research-wiki/EXP_p8_semantic_compression.md`, `EXP_p8b_vision_summary.md` |
| P4 schema-field distillation | `research-wiki/EXP_p4_schema_distill.md` |
| P9 / P9b decision-level | `research-wiki/EXP_p9_lmm_rgcl_video.md` |
| A0 OCR end-to-end (the concat lesson) | `idea-stage/A0_OCR_E2E_RESULT.md`, `a0_ocr_e2e.json` |
| PCD spec (frozen encoders cannot hold stance) | `idea-stage/PCD_SPEC.md`, IDEA_REPORT §8.13 |
| error taxonomy + prices (S = 45.4 %, +6.5) | IDEA_REPORT §9.2; `idea-stage/r5_buckets.json`, `r5_error_dump.json` |
| comparator to beat | IDEA_REPORT §8.10 — 0.8732 / 0.7776 / 0.8183 / 0.9276 |
| stance-as-supervision prereg (the ruling this direction is distinguished from) | IDEA_REPORT §9.8, §9.9 |
| head with an `extra` block already wired | `idea-stage/r4_harness.py::Head` |
| assets on disk | `data/lora_frames/`, `data/Archive/`, `data/Summaries*/`, `data/MLLM_scores/`, `data/OCR/` |

**Verify before citing** (flagged by the sweep as unopened or unverified): PromptHate's arXiv id
(use Anthology `2022.emnlp-main.22`); Pro-Cap's own AUC table; ExplainHM / LoReHM / MIND /
"Beneath the Surface" headline numbers; EARAM's adaptivity mechanism; the Idiap four-fusion
negative's exact id. Items the sweep marked `[SNIPPET]` were never opened. Note also that
**HVGuard has no arXiv id** — cite Anthology/DOI, per the §9.6 citation-hygiene note.

**Stated blind spots of this recon.** (a) No pilot, no measurement — every number here is either
published, previously measured in this repo, or an arithmetic estimate. (b) The 5-cell
oracle-threshold ceiling for D1 is *not yet measured* and is the single cheapest thing that could
kill it. (c) The D2 redundancy probe is *not yet run*. (d) The D3 flip-quality rate for a frontier
model is unknown; only the 7B figure (0.503 EN / 0.337 ZH) exists. (e) ImpliHateVid's raw video is
absent locally, so any frame-bearing front-end is 3-of-4 datasets unless it is re-downloaded, and
that must be declared in any resulting table exactly as §9.5/K1 required. (f) The occupancy sweep
exhausted its web-search budget partway and finished via arXiv listing enumeration and direct
fetches — stricter in coverage, but non-arXiv venues (ACM DL, IEEE, CEUR) are under-sampled.
(g) **Time sensitivity:** RAMF/MARS/LELA are one lab publishing steadily along this line;
speaker stance is their obvious next prompt.
