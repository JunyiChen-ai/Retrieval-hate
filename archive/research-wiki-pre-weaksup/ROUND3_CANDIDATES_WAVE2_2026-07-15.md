# ROUND-3 PROBE-ABLE CANDIDATE POOL — WAVE 2 (2026-07-15)

**Scout:** round-3 second ideation sweep. ZERO GPU (design + literature only).
**Working mode (user directive 2026-07-14):** many parallel probes on Modal cloud, features-only, minutes,
~$0 → build a DEEP candidate pool. This is **wave 2**. It does NOT re-propose wave-1
(`ROUND3_NOVELTY_CANDIDATES_2026-07-14.md`: S2S set-matching = LEAD; C2 multi-view memory = rides S2S
extraction; C3-geo KILLED; C4 self-killed D1; C5 LOW) or anything isomorphic to them.
**Deliverable bars (all enforced below):** D7-TIGHTENED novelty (mechanism novel *within hateful-video*,
not encoder-class lever nor generic-trick transfer), graveyard non-isomorphism (`directions_tried.json`,
24 dead ids + 9 bans), D1/D2/D3 diagnosis laws, all vetoes, and **probe-ability** (prefer Modal
features-only on banked caches; mark frameset-blocked and new-extraction probes explicitly).

---

## 0. What wave-2 targets, and what the fresh 2026 literature already KILLS

Wave-1's structural read stands: every one of the 24 dead routes retrieves over **one pooled vector per
video** and only ever (a) swapped the encoder (the sole +3 class, HateMM-only, now **D7-dead** for
novelty) or (b) bolted a low-bandwidth decision-side signal (all dead by **D1**). Wave-1 opened the
**retrieval-object** hole with S2S (frame-set late interaction). Wave-2 asks: **what representation-level
levers remain that are NOT frame-set-matching and NOT encoder-swap** — the three untouched axes are
**cross-modal interaction** (the MLLM's defining capability, never used as a *representation*), **temporal
order** (S2S's MeanMaxSim is order-free by construction), and **memory/input structure** (organisation and
new signal sources). Every candidate below lives on one of these axes.

**A 2026 literature sweep this session pre-kills three tempting directions — recording them so the pool
does not waste probes:**
- **Test-time adaptation / test-time memory for hate video is DONE:** "Shedding the Facades, Connecting
  the Domains: Detecting Shifting Multimodal Hate Video with Test-Time Adaptation" (arXiv 2602.00132,
  2026). Any "test-time memory adaptation" mechanism is now in-domain prior art → **W2-K2 killed**.
- **LLM/VLM reasoning-fusion for hate video is DONE and is D1-class:** "Reasoning-Aware Multimodal Fusion
  for Hateful Video Detection" (2512.02743) and "Training-Free ... via Multi-stage Adversarial Reasoning"
  (2601.15115, fetched: CoT reasoning traces + fusion). Confirms the D1 wall; do not re-enter.
- **Cross-modal incongruity/conflict modelling is heavily prior-arted** (sarcasm: Sci. of incongruity
  aligning S1566253523004487; 2408.02595; ACL-2022 cross-modal graph; and *in-domain* "explicit
  conflict-aware feature interaction modelling" for hate video is emerging per the 2024-25 survey line) →
  a plain "model modality disagreement" mechanism cannot clear D7 standalone → **W2-K1 killed** as a
  decision signal; its only live residue (congruity carried *in the representation*, not as a decision
  scalar) is folded into **W2-A**.

---

## W2-A — [LEAD wave-2] Cross-modal *grounded* retrieval keys (transcript-conditioned Qwen vision representation)

**Mechanism (1 para).** The banked pipeline encodes vision and transcript **independently**: `img_feats`
= a Qwen forward over frames + a *fixed* instruction (the transcript is NOT in this forward, verified
`generate_VideoMLLM_embedding_HF.py` / S2S §4), and `text_feats` = a separate forward over the transcript.
So the retrieval key never contains the **interaction** between what is shown and what is said. W2-A runs
**one** frozen Qwen2.5-VL-7B forward with the **native transcript AND the frames in context together**, and
takes the resulting **transcript-grounded visual representation** (the vision-token pool *after* it has
cross-attended to the transcript) as the retrieval key. This is the one thing a dual encoder (CLIP, or two
independent Qwen pools) **structurally cannot produce**: the key now encodes visual–transcript
**(in)congruity** — benign-looking frames under a hateful voice-over, a reclaimed slur over friendly
footage — as *geometry*, so two videos with the same implicit visual↔speech contradiction retrieve each
other. No generated text, no score, no decision signal: only a richer key feeding the unchanged top-20 vote.

**Injection point + bandwidth class.** Retrieval **representation** construction (representation-geometry,
D2 — the only class that ever cleared +3). Bandwidth: one 3584-d grounded key per video that carries an
**interaction term uncomputable from the marginals** `img_feats`, `text_feats`. Not decision-side.

**Non-isomorphism vs specific dead ids.**
- **vs encoder-swap / B1 / B2 (D7-dead):** those swap *which* pretrained encoder produces the pool with the
  *same independent-pooling recipe*. W2-A does not change the encoder; it changes the **key-construction
  operation** to a joint cross-modal forward, adding an img×text interaction term no swap or concatenation
  can form. (This is also the honest D7-risk edge — see novelty below.)
- **vs C3-nontarget (19th, DEAD_AT_FUSION):** C3 *generated* dense MLLM reasoning text and fused it as a
  channel — redundant, encoder-banked. W2-A generates nothing and adds no channel; it conditions the
  **vision representation** on the *native, already-in-pipeline* transcript. Different object, no generation.
- **vs reasoning-fusion papers (2512.02743 / 2601.15115):** they produce a reasoning trace and fuse a
  decision. W2-A produces an internal *representation* and votes; no reasoning, no decision fusion.
- **vs S2S / C2 (wave-1):** S2S/C2 keep a **vision-only** frame set and change the *matching*. W2-A keeps a
  **single grounded** key and changes what the key *contains* (adds the text-conditioning). Orthogonal and
  **composable** (grounded frame-*sets* are the natural W2-A×S2S cross).

**D7 novelty (standalone AND composite).** *Standalone:* text-conditioned / cross-attention visual
representations are an established VLM mechanism (search: text-guided visual representation 2409.19961;
PMC12349264 VTG; autoregressive-vs-cross-attention fusion) — so the raw operation is **not** novel.
*Composite (the defensible claim):* **first use of an MLLM's cross-modally-grounded internal representation
as the retrieval key in hateful-video detection, so implicit visual–transcript incongruity enters the
retrieval geometry rather than a decision-side conflict head** (which is where all in-domain incongruity
work lives). Closest prior art: MoRE (WWW 2025, pooled joint retriever → MoE, no grounded key); the sarcasm
incongruity line (decision-side, not retrieval geometry, not hate-video). **Honest D7 risk (flagged
prominently):** this is the closest wave-2 candidate to the encoder-class line — a skeptic reads it as "yet
another frozen-Qwen feature." The rebuttal is the interaction-term argument (uncomputable from the two
banked marginals), but whether that clears the user's novelty clause is a **D7-class user ruling**, same as
S2S/B3 — not decidable here. Rank #1 on mechanism value *if accepted*.

**D1/D2/D3.** **D2:** representation-level, the winning class. **D1 is the real threat and the probe is
built to expose it:** if the grounded key's label information is already contained in `concat(img_feats,
text_feats)`, grounding is redundant (the classic "probe passes, training flat"). The G0-cond probe
therefore measures **conditional** info of the grounded key *beyond the concat*, oracle-gated. **D3:** the
paired zero-training LOO design cancels seed noise; a real effect must clear the P3-priced raw bar.

**Veto check.** Native transcript already in pipeline (no OCR ✓, no new channel, no generated text ✓); no
gold annotations (frames/transcript unlabeled ✓); single-dataset own-train memory ✓; no cross-seed
ensemble ✓; local Qwen-7B only ✓; no API ✓; not a P1–P5 re-proposal ✓.

**Probe design (needs NEW extraction — local GPU, small; NOT features-only-cloud).** Stage E: one frozen
grounded forward per video (transcript+frames) over train∪val of **HateMM primary + MHC-EN binding-gap**
(~1–2 GPU-h, mirrors S2S Stage E; test extracted-not-scored). Stage P (CPU, cloud-ok once keys exist,
zero test-touch): paired LOO kNN vote, **grounded-key vs POOLED `img_feats` vs `concat(img,text)`**, plus a
**Fano label-oracle calibration arm** (must reach ≥0.99, REFLECTION §4 mandate) and an **oracle ceiling**
(gold-guided key selection). **Pre-declared kill logic:** (K1) grounded-key paired Δacc over
`concat(img,text)` < +0.04 oracle-ceiling on **every** dataset → the interaction term carries no
convertible info → DEAD, no head GPU; (K2) raw HateMM paired Δacc AND ΔF1 < +0.05 vs concat → below P3
shrinkage → dead. Data dependency: **new grounded extraction** (raw video, local queue).

**Prior: MODEST–FAIR.** *Falsifiable:* if a transcript-grounded Qwen key does not beat `concat(img_feats,
text_feats)` kNN by a paired margin projecting to +3 on ≥1 dataset's oracle arm, the cross-modal
interaction is redundant with the two independent pools and W2-A is dead.

**Cost.** ~1–2 GPU-h grounded extraction (local) + CPU/cloud probe (minutes).

---

## W2-B — Multimodal *sub-clip* set-matching retrieval (retrieval-object axis; PROBE-ABLE TODAY on banked CLIP)

**Mechanism (1 para).** Represent a video not as one pooled vector nor as S2S's vision-only *frame-group*
set, but as a **set of K multimodal sub-clip embeddings** — each sub-clip is a short contiguous segment
Qwen (or CLIP) encodes as a coherent unit (its own frames, and in the `_mm` variant its own ASR segment).
Retrieval score = set-matching (MeanMaxSim) between sub-clip sets. This operationalises P6's proven "hate
is local" premise as the **retrieval object** at the semantic granularity of a *segment* (a dog-whistle
scene, a hateful chant) rather than a raw frame. **These sets are already banked as float caches** —
`data/CLIP_Embedding/{HateMM,MHC}/*_subclipK4_*` (K=4, parent-indexed: MHC train = 549 vids × 4 = 2196
vectors), `HateMM/train_subclipK30` (K=30), and a **multimodal** `MHC/train_subclipK4_mm` (frames+ASR) — so
a CLIP-encoder version is **runnable on Modal features-only, today, at ~$0**.

**Injection point + bandwidth class.** Retrieval object + metric (representation-geometry, D2). Bandwidth
K× a pooled vector; higher, not decision-side.

**Non-isomorphism.** **vs S2S (wave-1 LEAD):** S2S's set = **vision-only frame-group** vectors from one
Qwen forward (T=4 spatial-token means); W2-B's set = **independent multimodal sub-clip** encodings (each a
segment Qwen/CLIP reasons over holistically, `_mm` includes ASR). Different object and extraction. **vs P3
(dead):** P3 kept ONE pooled vector re-weighted by an MLLM segment SCORE; W2-B keeps the **segment set** and
changes the metric, no score. **vs P6 (positive, localization):** P6 *scored* segments for a localization
read-out; W2-B *matches* segment sets for classification retrieval.

**D7 novelty — HONEST, WEAK STANDALONE.** The *mechanism* (set-matching over temporal units) is the S2S
thesis; W2-B is a **granularity/modality variant**, so as a standalone contribution it is **near-isomorphic
to S2S and does not clear D7 on its own**. Its distinct, defensible sliver is the **multimodal sub-clip
unit** (`_mm`: segment-level frames+ASR jointly, vs S2S's vision-only). **Primary value is not novelty — it
is the cheapest possible prior-update:** it answers TODAY, on existing caches, whether "don't-pool /
set-matching beats pooling" holds at all on hate-video kNN, **de-risking the entire don't-pool family
(S2S, C2, W2-C) before the frameset extraction lands.** I therefore recommend it as a probe, and flag that
a *positive* result would need S2S's Qwen-token version (or the `_mm` multimodal angle) to carry the
novelty.

**D1/D2/D3.** D2 representation-level; D1 does not bite (no decision signal); D3 handled by paired LOO +
bootstrap. Caveat: CLIP sub-clips are a weaker encoder than Qwen (B-line shows CLIP<Qwen on HateMM), so a
CLIP-null does not fully close the Qwen-token version — this asymmetry is pre-declared.

**Veto check.** Banked own-train sub-clips ✓; `_mm` uses native ASR not OCR ✓; no gold ✓; single-dataset ✓;
no ensemble/API ✓.

**Probe design (TODAY, Modal features-only).** Stage P only (no extraction): build train∪val sub-clip-set
memory from banked `subclipK4` (and `subclipK4_mm` for MHC); paired LOO kNN vote **SET vs POOLED (mean of
sub-clips)** on identical sub-clip vectors; Fano calibration arm; permutation null (shuffle sub-clip sets
across videos); bootstrap. **Kill:** raw paired Δacc AND ΔF1 < +0.05 on HateMM (K=4 and K=30 arms) AND
< survival bar on MHC → set-matching does not beat pooling even at the segment granularity on banked
features → strong negative prior update for the whole family. Data dependency: **banked pooled/set caches
ONLY — cloud-runnable today.**

**Prior: MODEST** (CLIP encoder + near-dup self-retrieval on LOO are the risks). *Falsifiable:* if sub-clip
set-matching does not beat pooled-sub-clip-mean kNN by a paired +0.05 on HateMM, the "pooling destroys
alignment" thesis is weak on this data and S2S's prior should be revised down before spending its GPU.

**Cost.** **~$0**, CPU minutes on Modal. No GPU.

---

## W2-C — Temporal-order / escalation-aware alignment kernel (temporal-order axis; rides S2S frameset)

**Mechanism (1 para).** S2S's MeanMaxSim is **order-free** — it matches a shared hateful frame regardless of
*when* it occurs. But hate frequently has a **temporal grammar**: benign setup → escalation → hateful
payload ("reveal" structure). W2-C keeps S2S's frame-group set but replaces the order-free score with an
**order-constrained alignment**: soft-DTW / monotonic OTAM-style warping over `{g_1..g_T}`, and/or a
**transition representation** `{g_{t+1}−g_t}` that encodes narrative *turns* rather than frames. Two videos
now match when they share the same benign→hateful *trajectory*, not merely a frame.

**Injection point + bandwidth class.** Retrieval metric over the frame set (representation-geometry, D2).
Same bandwidth as S2S; adds an order constraint, no decision signal.

**Non-isomorphism.** **vs S2S:** S2S is explicitly order-free (MeanMaxSim) and *defers* OT/temporal
alignment as a later arm; W2-C makes **order the mechanism** and tests a **different hypothesis**
(escalation trajectory, not shared segment). It rides S2S's extraction but is a distinct kernel + distinct
falsifiable claim. **vs TARC (dead):** TARC conditioned the retrieval *graph* on a predicted target
(decision-side, ~3 bits); W2-C changes the pairwise *geometry* to be order-aware, no target, no graph
conditioning.

**D7 novelty.** *Standalone:* temporal alignment (OTAM CVPR-2020, soft-DTW, CMOT/TRX/HyRSM) is established
in few-shot action recognition, and CVPR-2025 "Temporal Alignment-Free Video Matching" even argues order is
sometimes unnecessary — so raw novelty is the **same transfer class as S2S** (weak). *Composite:* first
**order-/escalation-aware** retrieval for hateful-video, motivated by the reveal structure of hate;
"Revealing Temporal Label Noise in Multimodal Hateful Video" (2508.04900) confirms temporal structure is an
active in-domain concern but for *label noise*, not retrieval geometry. Honest: novelty is
transfer-composite, same D7-class ruling as S2S.

**D1/D2/D3.** D2 representation-level; D1 clean; **D3 is the binding weakness — at T=4 (8 frames) an
order kernel over 4 coarse groups is thin**; the 16-frame arm (T=8) is the meaningful test. Pre-declare
both, no shopping.

**Veto check.** All ✓ (frozen frames, own-train, no gold/OCR/API/ensemble).

**Probe design (BLOCKED on S2S frameset; then features-only).** Depends on the S2S Stage-E frameset cache
(job 13159→Stage-E). Once it exists: Stage P (CPU/cloud) paired LOO — **order-constrained (soft-DTW /
monotonic) AND transition-set vs S2S MeanMaxSim vs POOLED** on identical frames; Fano arm; permutation null
including a **frame-order-shuffle** null (isolates order from richer-key). **Kill:** order/transition arm
does not beat *both* POOLED and S2S-MeanMaxSim by a paired +0.05 on HateMM (16-frame arm) → order carries
no convertible structure beyond the order-free set. Run **as an added kernel arm inside the S2S probe**, not
a separate job. Data dependency: **S2S frameset (frameset-blocked)**.

**Prior: MODEST–LOW** (thin at T≤8; "alignment-free" caution). *Falsifiable:* if an order-constrained
kernel does not beat order-free MeanMaxSim by a paired margin on the 16-frame arm, hate's temporal grammar
adds nothing the shared-segment match already captures.

**Cost.** ~0 marginal (rides S2S extraction + S2S probe machinery).

---

## W2-D — Acoustic-channel retrieval (new-input axis; MHC; needs audio extraction)

**Mechanism (1 para).** The current pipeline is **frames + transcript-text only** — the **acoustic**
channel (aggressive prosody, tone, hateful chants/music, laughter over a slur) is entirely absent, yet
carries hate signal neither frames nor transcript text capture. W2-D adds an **audio-embedding retrieval
channel** (a frozen audio encoder over each video's waveform) and either (a) an audio set/pooled retrieval
arm fused at the representation level, or (b) audio-visual **temporal-correspondence** keys (does the
aggressive audio co-occur with specific visual moments). This is D2's *second* winning class per REFLECTION
§3.2 ("new high-bandwidth input channel") — the only candidate that adds a genuinely new **signal source**
rather than reshaping the same two channels.

**Injection point + bandwidth class.** New input channel + retrieval representation (high bandwidth, D2).

**Non-isomorphism.** No dead route uses audio (P1–P11/TARC are visual+text). **vs OCR ban:** audio ≠ OCR,
distinct channel. **vs encoder-swap:** adds a modality, not a swap of the existing one.

**D7 novelty — HONEST, THIN.** Audio is a *component add*, and multimodal hate detection already uses audio
(HateMM is A+V+T; MM-HSD 2508.20546 fuses audio). "Add audio" is **not** a novel mechanism → D7 risk is
real. The only defensible composite is **audio-visual temporal-correspondence as a retrieval structure** in
a contrastive kNN head (not plain fusion) — thin, and adjacent to the incongruity prior art (W2-K1). Include
as a lower-ranked *signal-source* option, honestly caveated.

**D1/D2/D3.** D2 new-channel class; D1: a fused audio *representation* is not a decision scalar, but a
naive audio-vote fusion would be (keep it representation-level); D3: MHC-EN is data-limited (SAV), the audio
channel must beat the concat baseline by the P3-priced bar.

**Veto check.** Own-train audio ✓; not OCR ✓; no gold ✓; single-dataset ✓; **but audio features are only
plausibly available for MHC** and the goal's binding gap is MHC-EN, so this at least targets the right
dataset. Local audio encoder only (no API).

**Probe design (NEEDS EXTRACTION — audio NOT banked).** `data/audio/MHC` exists but is **empty** (verified)
— so an audio encoder pass is required first (CPU/GPU-light; audio ⊂ the raw-video license question — keep
local). Then Stage P (cloud-ok): paired LOO kNN **audio-fused vs `concat(img,text)`**; Fano arm; oracle
ceiling. **Kill:** audio-fused Δacc over concat < +0.04 oracle on MHC-EN → the acoustic channel is
redundant given vision+transcript. Data dependency: **new audio extraction (local, MHC only)**.

**Prior: LOW–MODEST.** *Falsifiable:* if an audio-fused retrieval key does not beat vision+transcript concat
by a paired +0.03 on MHC-EN, hate's acoustic signal is already banked in the transcript/visual pathway.

**Cost.** Audio-encoder pass (light) + CPU probe. MHC only.

---

## W2-E — Unsupervised fine-grained hate-mode prototype memory (memory-organisation axis; PROBE-ABLE TODAY)

**Mechanism (1 para).** The kNN memory votes flatly over all train exemplars. Hate has distinct **modes**
(overt violence, slurs, dehumanising metaphor, dog-whistle). W2-E discovers **K prototypes per class** by
**unsupervised clustering of the Qwen representation** (no gold target labels — compliant), then restructures
retrieval as **prototype-anchored** (query → prototype assignment → mode-local kNN, or prototype centroids
as denoised keys). The memory geometry, not the encoder, changes.

**Injection point + bandwidth class.** Memory representation/organisation (representation-geometry, D2-ish).
No decision-side signal.

**Non-isomorphism.** **vs C2 multi-view memory (wave-1):** C2 *expands* each exemplar into multiple views;
W2-E *compresses across* exemplars into shared centroids — opposite operation. **vs archive-auto-repair
(banned):** that ban is on **MLLM-vote-based deletion** of noisy entries; W2-E does unsupervised
clustering, not MLLM-vote deletion — distinct, though "memory curation"-adjacent (flagged).

**D7 novelty — LOW.** Prototype/centroid memory is textbook few-shot (ProtoNet); the MLLM integration is
**thin** (prototypes over any encoder). Composite ("MLLM-representation-derived fine-grained hate-mode
prototypes in a video retrieval-contrastive head") is a weak novelty story. Included because the seed named
it and it is a **cheap TODAY probe**, but honestly ranked low.

**D1/D2/D3.** D2 memory-geometry; D1 clean; **D3 is the binding risk — K prototypes over ~600 exemplars is
noisy**, prototypes may just re-encode the flat kNN.

**Veto check.** Unsupervised (no gold) ✓; own-train ✓; single-dataset ✓; no ensemble/API/OCR ✓.

**Probe design (TODAY, Modal features-only).** Stage P on banked pooled Qwen features: cluster train
representation into K∈{2..8}/class; paired LOO **prototype-anchored kNN vs flat kNN**; Fano arm; bootstrap.
**Kill:** prototype-anchored Δacc AND ΔF1 < +0.03 vs flat kNN on all datasets → memory organisation is
redundant with flat retrieval. Data dependency: **banked pooled caches — cloud-runnable today.**

**Prior: LOW.** *Falsifiable:* if hate-mode prototypes do not beat flat kNN by a paired +0.03, the memory's
fine structure adds nothing the top-20 vote already uses.

**Cost.** **~$0**, CPU minutes on Modal.

---

## W2-K1 — [DOCUMENTED KILL] Cross-modal *disagreement* as a decision signal

**Idea:** two kNN memories (visual, transcript); flag videos where the two pathways' votes DISAGREE as
hateful (dog-whistle / reclaimed-slur incongruity). **Kill (self):** the disagreement is a **1-bit
decision-side scalar** → **D1 bites directly** (the "probe passes, training flat" pattern), *and* it is
prior-arted twice over — cross-modal incongruity is a mature sarcasm mechanism (S1566253523004487;
2408.02595; ACL-2022 graph) and **in-domain hate "conflict-aware feature interaction" is already emerging**
(2024-25 survey line) → fails D7 standalone. **Its one non-dead residue** — carrying congruity **in the
representation** rather than as a decision vote — is exactly **W2-A**; pursue it there. Do not spend a probe
on the decision-side version.

## W2-K2 — [DOCUMENTED KILL] Test-time memory adaptation / dynamic memory reweighting

**Idea:** adapt the memory geometry per query at test time (test-time training / memory reweighting for
distribution shift in evolving hate). **Kill:** **already in-domain prior art** — "Shedding the Facades …
Detecting Shifting Multimodal Hate Video with **Test-Time Adaptation**" (arXiv 2602.00132, 2026) → fails D7.
It is also **auto-memory-repair-adjacent** (banned: memory editing is human-in-loop only) and a per-query
reweighting collapses toward a decision-side signal (**D1**). Closed at ideation.

---

## RANKING (novelty-defensibility × prior × probe-cheapness)

| # | candidate | axis | injection point | in-domain novelty | prior | probe-cheapness / data dep | verdict |
|---|---|---|---|---|---|---|---|
| **1** | **W2-A cross-modal grounded key** | cross-modal interaction | representation construction | **MODERATE-composite** (D7-risk: encoder-edge) | **MODEST–FAIR** | LOW — needs grounded extraction (local) | pursue; **highest mechanism value if D7 accepted** |
| **2** | **W2-C temporal/escalation kernel** | temporal order | retrieval metric | MODERATE (transfer, S2S-class) | MODEST–LOW | rides S2S frameset (marginal ~0) | run as an **added arm inside S2S probe** |
| **3** | **W2-B multimodal sub-clip set-match** | retrieval object | object + metric | LOW standalone (S2S-iso); `_mm` sliver | MODEST | **HIGH — banked CLIP, TODAY, ~$0** | run TODAY as **family de-risker**, not a novelty bet |
| 4 | W2-D acoustic channel | new input | new channel + repr | LOW-thin (component add) | LOW–MODEST | LOW — needs audio extraction, MHC-only | queue only if A/B/C stall |
| 5 | W2-E hate-mode prototype memory | memory org | memory geometry | LOW (ProtoNet) | LOW | **HIGH — banked pooled, TODAY, ~$0** | cheap TODAY companion probe |
| — | W2-K1 disagreement-as-signal | (killed) | decision | — | — | — | **KILLED** (D1 + incongruity prior art; residue→W2-A) |
| — | W2-K2 test-time memory | (killed) | memory/decision | — | — | — | **KILLED** (in-domain TTA prior art 2602.00132) |

**Probe-ability partition:**
- **PROBE-ABLE TODAY (Modal features-only, banked caches, ~$0):** **W2-B** (banked `subclipK4` /
  `subclipK30` / `subclipK4_mm` CLIP sets), **W2-E** (banked pooled Qwen). No GPU, no queue.
- **BLOCKED on S2S frameset (job 13159→Stage-E):** **W2-C** — rides that exact cache; should be an added
  kernel arm in the S2S probe, near-zero marginal cost.
- **NEEDS NEW RAW-VIDEO GPU EXTRACTION (local queue, license-sensitive — keep off cloud):** **W2-A**
  (transcript+frames grounded forward, ~1–2 GPU-h), **W2-D** (audio encoder pass; audio dir is empty).

## RECOMMENDED FIRST PARALLEL-PROBE BATCH (cloud, today, ~$0)

**Batch = W2-B (primary) + W2-E (companion), both features-only on banked caches; and queue W2-A's grounded
extraction on the local GPU in parallel.**

- **W2-B is the single highest-value first probe.** For ~$0 CPU-minutes on banked CLIP sub-clip sets it
  answers **today** the question the whole don't-pool family rests on — *does set-matching over temporal
  units beat pooling on hate-video kNN?* — and thereby **de-risks the S2S LEAD and W2-C before the frameset
  extraction even lands.** A CLIP-null (with the pre-declared encoder-asymmetry caveat) revises the family's
  prior down cheaply; a CLIP-positive strongly corroborates S2S. It is not itself a D7-novel contribution,
  and I say so plainly — but as a probe it is the most information per dollar available right now.
- **W2-E runs in the same batch** at ~$0 as an independent cheap test of the memory-organisation axis; low
  prior, but a clean parallel kill.
- **In parallel, queue W2-A's grounded extraction locally** — it is the only wave-2 candidate with a
  genuinely novel *mechanism* claim and a MODEST–FAIR prior, and its extraction is the same ~1–2 GPU-h
  shape as S2S Stage E. Its D1-exposing probe (grounded key vs `concat(img,text)`) then runs on cloud.

**One-line each (ranked):** ① W2-A grounded cross-modal retrieval key (novel-if-accepted; needs extraction)
· ② W2-C temporal/escalation kernel (rides S2S frameset; distinct hypothesis) · ③ W2-B multimodal sub-clip
set-matching (TODAY, banked CLIP; family de-risker) · ④ W2-D acoustic channel (new signal source;
MHC-only; needs audio extraction) · ⑤ W2-E hate-mode prototype memory (TODAY, banked pooled; low prior).
**Killed at ideation:** W2-K1 disagreement-as-signal, W2-K2 test-time memory.

---

## PROVENANCE — citations (verified this session)

- **MoRE** — "Biting Off More Than You Can Detect: Retrieval-Augmented Multimodal Experts for Short Video
  Hate Detection," **WWW 2025** (dl.acm 10.1145/3696410.3714560; +6.91% avg macro-F1 on HateMM/MHClip-Y/-B).
  In-domain closest prior; **pooled** joint retriever → MoE. [search-confirmed]
- **Shedding the Facades … Test-Time Adaptation** — arXiv **2602.00132** (2026), shifting multimodal hate
  video via TTA. Kills W2-K2. [search-confirmed title/venue]
- **Reasoning-Aware Multimodal Fusion for Hateful Video Detection** — arXiv **2512.02743**. D1-class
  reasoning fusion. [search-confirmed]
- **Training-Free … Multi-stage Adversarial Reasoning** — arXiv **2601.15115** (Yang, Zhang, Fu). CoT
  reasoning traces + fusion over Qwen2.5-VL/Llama/Gemini/GPT. D1-class. [fetched — abstract/structure]
- **Revealing Temporal Label Noise in Multimodal Hateful Video Classification** — arXiv **2508.04900**.
  In-domain temporal structure (label noise). [search-confirmed]
- **MM-HSD** — "Multi-Modal Hate Speech Detection in Videos," arXiv **2508.20546**. Audio+visual+text (+OCR)
  fusion; context for W2-D. [search-confirmed]
- **Cross-modal incongruity (sarcasm)** — ScienceDirect **S1566253523004487**; arXiv **2408.02595**;
  ACL-2022 cross-modal graph (aclanthology 2022.acl-long.124). Prior art killing W2-K1 standalone.
  [search-confirmed]
- **Temporal alignment (few-shot video)** — OTAM (CVPR-2020, 1906.11415), soft-DTW, CMOT/TRX/HyRSM; and
  **"Temporal Alignment-Free Video Matching for Few-shot Action Recognition," CVPR 2025** (alignment-free
  caution for W2-C). [search-confirmed]
- **Text-conditioned visual representation (VLM)** — arXiv **2409.19961**; PMC12349264 (VTG); autoregressive
  vs cross-attention fusion. Establishes W2-A's raw operation is not novel (composite claim only).
  [search-confirmed]

**Internal provenance:** 24 dead ids + 9 bans + D1/D2/D3 — `state/directions_tried.json`,
`REFLECTION_mllm_integration_failures.md`. Wave-1 candidates + S2S mechanism — `ROUND3_NOVELTY_CANDIDATES_
2026-07-14.md`, `experiments/exp-s2s-r3.md`. Cloud/Modal features-only feasibility + G-repro hazard —
`CLOUD_GPU_FEASIBILITY_2026-07-14.md`. Cache verification this session (banked sub-clip sets
`subclipK4`/`subclipK30`/`subclipK4_mm`; `img_feats` forward excludes transcript; `data/audio/MHC` empty;
`lora_frames` = raw JPGs not features) — direct filesystem inspection, `data/CLIP_Embedding/`, `data/audio/`,
`data/lora_frames/`.
