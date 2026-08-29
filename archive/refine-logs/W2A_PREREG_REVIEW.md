# W2-A INDEPENDENT PRE-REGISTRATION REVIEW

**Reviewer:** fresh, zero-prior-context independent prereg reviewer. **Date:** 2026-07-15.
**Scope:** READ-ONLY except this deliverable. NO GPU, NO submission.
**Under review (commit e6be76c):** `refine-logs/W2A_FORENSIC_RECON.md` + `research-wiki/experiments/exp-w2a-grounded.md`.
**Verification performed this review:** source-read of the banked extractor, the installed Qwen2.5-VL
modeling file (HateVideo env, transformers 4.49.0), `metrics.py` retrieval vote, REFLECTION §4, the C3
fusion-probe record, `directions_tried.json` bans, and on-disk cache inventory.

---

## FINAL VERDICT: **APPROVED-WITH-AMENDMENTS** · **GO-WITH-AMENDMENTS** on the 2–3 GPU-h spend

Two amendments are **BLOCKING** (must land before code is authored); five are non-blocking. The design is
sound, honest about its LOW–MODEST prior, and its kill-switch is disciplined — but as written the probe can
**false-PASS** against a structurally-handicapped baseline, which is the precise failure mode the cited C3
probe exists to prevent. The two blocking amendments close that hole. With them, GO.

**One-paragraph GO justification.** The spend is 2–3 GPU-h local extraction plus a CPU/cloud probe, and an
oracle-ceiling kill-switch (§6.4) fires *before* any head GPU. The mechanism sits in the one regime that
ever cleared +3 in this project (representation-level, D2 — the encoder-swap lever), so it is the
highest-value wave-2 cell to falsify cheaply; MHC-EN at 100% transcript coverage is a genuine fair-coin for
the binding ≥2-dataset goal. The honest risk is C3-nontarget-shaped redundancy (the interaction is already
partly banked in `text_feats`), and C3 answered the analogous question *no* — but **with the CLIP-augmented
conditional-info baseline (Amendment 1) the probe can *detect* that redundancy instead of being fooled by
it**, so the run has real discriminating power rather than a rigged win. A negative closes the
cross-modal-grounding cell cheaply and definitively; a positive advances the binding goal even though
novelty remains a separate D7 user ruling. The decision value justifies the spend **conditional on
Amendments 1 and 2**.

---

## LOAD-BEARING CLAIM VERIFICATION (I re-derived both from source; both decide everything)

### Claim (a) — banked `text_feats` is a JOINT forward over frames+title+transcript, pooled at the response span. **VERIFIED TRUE.**

- `_build_messages(frames, instruction)` (`src/utils/generate_VideoMLLM_embedding_HF.py:241-251`) **always**
  emits `content=[{"type":"video","video":frames},{"type":"text","text":instruction}]` — every forward
  contains the frames.
- In `process_split` the text stream's prompt is built with the transcript inline
  (`:349-355`: `text_prompt = TEXT_INSTRUCTION + "\nTitle: " + title + "\nTranscript: " + transcript`) and
  then run through the **same joint forward** (`:356-359`: `text_vec = _encode(frames, text_prompt, …,
  span="response")`).
- The `"response"` span pools `last_hidden[start:]` from the last `<|im_start|>` (assistant header) to the
  end (`:304-318`) — trailing tokens that, under causal attention, have attended over **both** the frames
  and the transcript.

**⇒ The img×text interaction the scout claimed "the retrieval key never contains" is in fact already banked
in `text_feats` and already in the retrieval geometry.** The recon states this honestly (RECON §2.1, HEADLINE
finding 1) and the prereg prices it correctly: the D1 baseline is `concat(img_feats,text_feats)` and it
**must lose** (prereg §2 finding 1, §5, §6.4), and the prior is revised down to LOW–MODEST (prereg §13,
RECON §7). This makes W2-A a **C3-nontarget-shaped redundancy question**, correctly identified. Claim (a)
holds and is honestly represented.

### Claim (b) — Qwen2.5-VL LLM backbone is fully causal ⇒ a video-first vision-span "grounded" key is a provable no-op ⇒ transcript-first ordering is REQUIRED. **VERIFIED TRUE.**

- `Qwen2_5_VLAttention.is_causal = True` (`modeling_qwen2_5_vl.py:723`); `is_causal`/`causal_mask` flow into
  SDPA (`:989`, `:991-998`).
- `_prepare_4d_causal_attention_mask_with_cache_position` (`:1327-1381`) builds a standard lower-triangular
  mask: `diagonal_attend_mask = arange(target_length) > cache_position.reshape(-1,1)`, then
  `causal_mask *= diagonal_attend_mask` (`:1371,:1380`). There is **no bidirectional attention over vision
  tokens** in the LLM; vision embeddings are `masked_scatter`'d in place at ordinary positions
  (`:1809,:1827`), and the banked preflight `last_hidden.shape[0]==input_ids.numel()` confirms no shift
  (`generate_VideoMLLM_embedding_HF.py:283`).

**⇒ In the banked video-first order, vision tokens precede the transcript and cannot attend to it, so a
video-first vision-span pool ≡ the ungrounded vision pool (a silent no-op).** The only architecturally real
"transcript-conditioned vision" key requires the transcript BEFORE the frames. The prereg's SETTLED design
(transcript-first, vision-span pool; video-first pool used only as the ungrounded control) is the correct —
indeed the only — realization (prereg §2 finding 2, §2 mechanism, §4). Claim (b) holds.

### Flip-side scrutiny the team lead flagged — transcript-first ordering DOES break parity with every banked artifact via M-RoPE. **CONFIRMED, and it is a real (handled-but-under-gated) confound.**

`get_rope_index` (`:1546-1660`, docstring `:1582-1585`: "text start position_ids = max vision position_ids
plus 1"; symmetrically, vision position ids start at `max(preceding-text-position)+1`) confirms that placing
the transcript before the frames **shifts the vision tokens' M-RoPE position ids by the transcript token
count** — a *large* shift for HateMM (median 694 chars, max 80 731 chars → thousands of tokens). Two
consequences the review must rule on:

1. **The grounding signal is confounded (content-attention + position-shift), but this is correctly
   isolated by the placebo control** (content held-out vs position/length held-in) — see Amendment 3 to make
   that isolation match the probe's cross-video null.
2. **The gate-2 empty-transcript branch prediction (`cos(grd,ungrd_vis) ≥ 0.999`) is not guaranteed.** Even
   an empty transcript renders a non-empty `[{text:"(none)"}]` block before the frames, shifting vision
   positions by a few tokens and letting vision attend to `"(none)"`. Exact ≥0.999 may not hold from
   position-shift alone. As written (§4 gate 2, "must satisfy"), a benign position effect could trigger a
   **false HALT**. See Amendment 6.

### Is any gate validating the transcript-first forward *itself* (beyond grounding-live/placebo)? Ruling: **SUFFICIENT, with Amendments 6–7.**

The transcript-first forward has no banked twin (novel ordering) and *cannot* be validated by exact
reproduction — the correct house pattern (matches S2S) is internal-consistency + control-anchor. It is
validated by: (i) **grid gate** (§4 gate 0) runs on **both** forwards, so the vision span is correctly
located in the transcript-first forward; (ii) **G-recon-IMG** (gate 1) proves the machinery (model load,
frame sampler, hidden states) via the video-first control; (iii) **length/parity** (gate 4); (iv)
**grounding-live** empty-branch as a falsifiable causal-behavior prediction; (v) **placebo** content-check.
The IMG-control gate validates the *pipeline*, not the transcript-first path — but (i)+(iv)+(v) do cover the
transcript-first path behaviorally, and there is no stronger check available for a novel forward. I rule the
suite **sufficient for a features-only probe whose downstream is gated by an oracle kill-switch**, provided
Amendments 6 (empty-branch confound) and 7 (vision-pad-mask assertion) land.

---

## PER-ITEM RULINGS

### 1. Gate suite — complete, non-gameable, C3-template correctly specified?

- **K2 grounding-live / τ_live — PARTIALLY OK; τ_live is self-calibrated (Amendment 4).** The *binding*
  no-op VOID is a **fixed, pre-declared numeric** threshold — present-set median `cos ≥ 0.999` (§4 gate 2,
  K2). That is correct and non-gameable. But `τ_live` itself is a **formula calibrated on a same-data smoke
  subset** (`τ_live = median(present cos) − 0.5·(present−empty gap)`, §4 gate 2). Because it self-calibrates
  to whatever the model produces, the per-video "present must satisfy cos < τ_live" check **cannot fail if
  the effect is weak-but-nonzero** — it is not a real bar. **Ruling: the binding trigger IS pre-declared
  numerically (0.999); demote τ_live to a logged diagnostic (Amendment 4) so it is not read as a HALT gate.**
- **K3 placebo — OK but weaker than the probe null it mirrors (Amendment 3).** Within-video token-shuffle
  (§4 gate 3) tests *order*-sensitivity; the "content-sensitivity" claim and the probe's **cross-video**
  permutation null (§6.6) call for a **cross-video mismatched** transcript. RECON §5 item 4 itself says
  "shuffled/**mismatched**." Non-blocking strengthening.
- **K4 Fano ≥0.99 — OK, non-gameable.** Standard REFLECTION §4 machine-validity (§6.3); label-oracle ±1 key
  must vote ≥0.99 or the vote machine is VOID. Correct.
- **K5 oracle-ceiling — OK as a kill-switch, but inherits the CONCAT weighting handicap (see item 2).** The
  per-query gold-choice grd-vs-concat with tie→concat (§6.4) is a legitimate upper bound (oracle ≥ concat
  always). Non-gameable in the kill direction (a handicapped concat makes it *harder* to kill, i.e.
  conservative). Correct as specified.
- **K6 raw bars — pre-declared and well-derived, but the baseline geometry is handicapped (Amendment 2).**
  +0.05 acc AND +0.05 mF1 on HateMM, beat CONCAT-PCA in sign, bootstrap 5th-pct > 0, above 95th-pct
  permutation null (§6.5). The 1.7× P3 pessimism factor is correctly justified (mirrors S2S §6.5). The bar
  is fine; the *baseline it is measured against* is the problem (item 2).
- **C3-template conditional-info probe (§5, gate 8) — correctly specified per REFLECTION §4, EXCEPT the Z
  baseline is too weak (Amendment 1, BLOCKING).** It faithfully reproduces the C3 machinery: un-penalized
  aux block at Z's inner-CV-optimal C (the REFLECTION §4 "don't let shared L2 crush the aux columns" fix),
  5×5 RepeatedStratifiedKFold, per-video clustered bootstrap, **label-oracle calibration arm (accZA≈1.0 or
  MACHINERY_INVALID)** reaching full Fano headroom, ≥150-permutation null **as a distribution**, +0.040
  triple rule (C1 point ≥ bar ∧ C2 CI-lower > 0 ∧ C3 real > all perm maxima). I verified this against
  `refine-logs/C3_FUSION_PROBE_RECORD.md` §"PRE-DECLARED DESIGN" and REFLECTION §4 (`REFLECTION_mllm_
  integration_failures.md:37-45`) — the calibration-arm-reaches-headroom and null-as-distribution
  requirements are both met. **The one defect: W2-A sets `Z = concat(Qwen img+text)` (7168-d), whereas the
  C3 record's binding decision cell was `Z_best = concat(CLIP img+text, Qwen img+text)` (8960-d, the
  pipeline's *actual best banked config*).** C3's entire lesson was that a signal can beat Qwen-alone yet die
  against concat(CLIP,Qwen) — "CLIP-only gain = encoder redundancy, info banked in Qwen pathway"
  (`directions_tried.json` C3-nontarget epitaph). Testing only against Qwen-only concat re-opens exactly
  that false-pass. **BLOCKING — Amendment 1.**

### 2. CONCAT arm fairness — is the geometry fair, or does it structurally handicap CONCAT (false-PASS for GROUNDED)? **RULING: it IS handicapped; requires a fix (Amendment 2, BLOCKING).**

kNN cosine on the concat of two unit-norm halves is `cos(concat_i,concat_j) = ½·cos_img(i,j) +
½·cos_text(i,j)` — a **fixed 50/50 similarity weighting**. GROUNDED is a single 3584-d space whose kNN
cosine is an unconstrained blend of vision+transcript info. So the K5/K6 kNN comparison can hand GROUNDED a
win **not because it carries more information but because the fixed-α CONCAT is a poor retrieval geometry** —
a false-PASS. **CONCAT-PCA does NOT fix this:** PCA on `[img‖text]` picks unsupervised high-variance
directions, not a label-useful img-vs-text weighting, so the handicap survives dimensionality matching.
The linear conditional-info probe (§5) *is* weighting-invariant (the logistic head re-weights the halves
freely), and it is a binding co-gate (§6.1 gate 8) — but only if it uses the strongest Z (Amendment 1).
**Ruling: because K6 is currently a BINDING performance gate and can false-PASS, this must be fixed.**
Required (pick one, pre-declare): **(a)** add a memory-fit **weight-optimized CONCAT-α** baseline (α on a
small grid, chosen on train∪val only, leak-free) to the must-lose set, so GROUNDED must beat the *best-
weighted* concat; **OR (b)** demote the kNN raw bar K6 to advisory and make the weighting-invariant
conditional-info probe (with Amendment 1's CLIP-augmented Z) the sole binding performance adjudicator.

### 3. Empty-transcript tail — handled how? **RULING: pre-declared as included-as-noise; sound but add a covered-only sensitivity (Amendment 5).**

HateMM has a 5–12% empty-transcript tail (RECON §1.1). The prereg includes those rows in the paired
contrast (§4 zero-guard policy: not zero-guarded; forward runs with `"Transcript: (none)"`; "mechanism
vacuous on those rows, logged, paired-contrast unbiased") and pre-declares the resulting **~8% dilution**
(§6.6, §13). Including them is **conservative** (attenuates HateMM Δ toward 0, making the +0.05 bar harder,
which is the safe direction). This is honest and pre-declared. **Gap:** a real effect on the covered rows
could be masked by the dilution, and no covered-only view is pre-declared. **Amendment 5 (non-blocking):**
pre-declare a *secondary* (non-primary, non-shopping) covered-rows-only Δ. MHC-EN's 100% coverage (RECON
§1.1) means this only affects the HateMM row.

### 4. VETO / SCOPE — **RULING: fully compliant.**

- **whisper ASR = dataset-native pipeline artifact, NOT gold aux annotation and NOT MLLM-generated.**
  CONFIRMED. The transcript is the native `text` field of `data/gt/<ds>/<split>.jsonl`, produced by
  whisper-large-v3 ASR of the video's own audio, and is **already consumed by the banked `text_feats`
  forward** (`generate_VideoMLLM_embedding_HF.py:350`). It is neither OCR (on-screen text; user-vetoed —
  `directions_tried.json` banned_constraints "OCR channel") nor a gold label nor MLLM-generated reasoning
  text (which is what killed C3-nontarget). W2-A adds **no new channel** — it re-pools an existing modality.
  Correctly classified against the C3-nontarget closure and the no-gold rule (prereg §3, §4.4; RECON §4.1).
- **Single-dataset:** memory = own train∪val only (§5: HateMM 851, MHC-EN 629), no cross-dataset mixing —
  compliant with the 2026-07-14 user veto (`directions_tried.json`).
- **Zero test-touch:** §9 ledger + fail-closed no-test-touch guard (§5, S2S N4: `assert len(memory)==851/629`,
  never open `test_seen*`). Test grounded keys are *extracted and cached* for a later formal stage but never
  scored (§9). Compliant.
- **Videos stay local:** §4 "Local GPU only (raw video, license-sensitive — off cloud per CLAUDE.md)."
  Confirmed. Stage P' consumes only derived float vectors → features-only → **Modal-eligible** (prereg
  hardware line, §5). This matches the CLAUDE.md posture (probes on cloud, raw-video extraction/formal
  validation local).

### 5. Cost / sequencing sanity — **RULING: the spend is justified; GO-WITH-AMENDMENTS.** (See headline justification.)

Extraction (2–3 GPU-h local) *is* the probe's data collection; there is no cheaper way to get the grounded
keys. The oracle-ceiling kill-switch (§6.4) fires before any *further* (head) GPU. Against a graveyard of 20
dead directions and a terminus-seeking posture, a cheap, rigorous, discriminating negative on the closest
wave-2 cell to the proven encoder lever has real decision value, and a positive (esp. MHC-EN) advances the
binding performance clause regardless of the separate novelty ruling. **The GPU spend is justified
conditional on Amendments 1 & 2** (without them the probe risks a wasteful false-PASS into head-training).

### 6. Ceremony completeness — **RULING: complete, with two additions.**

Present and adequate: hash-freeze before submit (status block: implementation → **separate** independent code
review → hash-freeze → single Stage-E' submit), single-submit (§4), `JobHeldUser` wait-never-force (§4),
independent verdict review (status block), test-touch ledger (§9), `--limit 1` smoke to throwaway path (§4),
resumable per-video shards (§4). **Additions:** (i) the hash-freeze must include the pre-declared items
introduced by the amendments — the α-grid / Z-definition (Amendments 1–2), and the τ_live value if retained
(Amendment 4); (ii) enforce **raw-only transcription** of results per this project's numeric-provenance
discipline (report raw numbers from the primary JSON, no fabricated companion metrics).

---

## REQUIRED AMENDMENTS

| # | Blocking? | Amendment | Where |
|---|---|---|---|
| **1** | **BLOCKING** | Conditional-info probe (§5 C3-template arm) must use **`Z_best = concat(CLIP img+text, Qwen img+text)`** as the *binding* baseline (Qwen-only concat kept as secondary context), matching the C3 record's Z_best. CLIP caches exist for both datasets (verified: `data/CLIP_Embedding/{HateMM,MHC}/{train,dev_seen,test_seen}_openai_clip-vit-large-patch14-336_HF.pt`). Without this, W2-A re-opens the exact "beats Qwen-alone but is CLIP-encoder-redundant" false-pass that killed C3-nontarget. | §5, §6.1 gate 8, K9 |
| **2** | **BLOCKING** | Fix the handicapped kNN CONCAT geometry (fixed 50/50). Either **(a)** add a memory-fit **weight-optimized CONCAT-α** baseline to the must-lose set (α on a small grid, chosen on train∪val only, leak-free), so GROUNDED must beat the best-weighted concat; **OR (b)** demote raw bar K6 to advisory and make the weighting-invariant conditional-info probe (with Amdt 1's Z) the sole binding performance adjudicator. Pre-declare the choice. CONCAT-PCA does **not** substitute (unsupervised variance ≠ retrieval weighting). | §5, §6.4, §6.5, K5, K6 |
| 3 | non-blocking | Placebo (K3/§4 gate 3): use a **cross-video mismatched** transcript to match the cross-video permutation null (§6.6); keep within-video token-shuffle as a secondary diagnostic. | §4 gate 3, K3 |
| 4 | non-blocking | Demote **τ_live** to a logged diagnostic; the binding no-op VOID is the fixed pre-declared **present-set-median `cos ≥ 0.999`** (already in K2). τ_live self-calibrates and cannot fail on a weak-but-nonzero effect. | §4 gate 2, K2 |
| 5 | non-blocking | Pre-declare a **covered-rows-only** secondary Δ for HateMM (not primary, not dataset-shopping) so the ~8% empty-transcript dilution cannot mask a real covered-row effect. | §4, §6.6 |
| 6 | non-blocking | Gate-2 empty-branch (`cos ≥ 0.999` for empty transcripts): acknowledge the M-RoPE position-shift + `"(none)"`-attention confound; make it a **logged diagnostic** (report the empty-set cos distribution), not a HALT, or relax to a pre-declared looser bound. Binding content-check remains the placebo (K3). | §4 gate 2, K2 |
| 7 | minor | Add an assertion that the **vision-pad positions** (`input_ids==video_token_id`) are contiguous and count-match the grid gate in **both** forwards (grid gate checks count; add contiguity + identical mask logic in both), since G-recon-IMG validates the *prefix* span, not the vision-pad pool that produces `grd`/`ungrd_vis`. | §4 gate 0/1 |

---

## CONDITIONAL AUTHORIZATION TERMS (if the two blocking amendments land)

1. **Amend the prereg** (Amendments 1 & 2 blocking; 3–7 folded in) and re-hash the prereg + recon.
2. **Implementation.** A new message-builder (transcript-first, three content blocks) and a new vision-pad
   pooling function are the novel code surfaces — note `_build_messages`/`_encode` hardcode the video-first
   order and prefix/response spans and **must be re-authored**, not imported. Import verbatim only the
   forward-neutral helpers the prereg lists (`_sample_frame_indices`, decode helpers, `read_gt`,
   `IMG_INSTRUCTION`, `SPLIT_TO_OUTNAME`).
3. **Independent code review** (separate agent) — scrutinize especially: the transcript-first message
   construction + chat template rendering; the vision-pad mask vs prefix span; the G-recon-IMG byte-parity
   of the control forward; the leak-free memory-only fit of CONCAT-PCA / CONCAT-α / conditional-info PCA;
   the fail-closed no-test-touch guard.
4. **Smoke** (`--limit 1`, throwaway path; ≥20-video subset for any retained calibration) → **hash-freeze**
   (incl. all newly pre-declared constants) → **single Stage-E' submit**, local GPU, no `--time`,
   `JobHeldUser` → wait for auto-release (never force).
5. **Stage-P' probe** on the extracted float keys is **features-only and Modal-eligible** — CONFIRMED (no
   raw video leaves local; derived vectors only). Zero test-touch enforced.
6. **Independent verdict review** of the Stage-P' results before any DEAD/PASS call; raw-only transcription;
   the head-training formal stage (§11) remains **NOT authorized** here (separate prereg behind the oracle
   kill-switch).

---

## PROVENANCE (verified this review)
- Banked joint forward: `src/utils/generate_VideoMLLM_embedding_HF.py:241-251,254-323,349-359`.
- Causal backbone: `.../HateVideo/.../transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:723,989,991-998,
  1327-1381` (causal mask), `1546-1660` (`get_rope_index`, M-RoPE vision-position shift), `1809,1827`
  (masked_scatter). transformers **4.49.0**, HateVideo env.
- Retrieval vote: `src/utils/metrics.py:214,262-286,300` (`use_sim` signed-cosine rank-weighted vote; acc
  cut `vote≥0`; `tarc_vote_gamma=0` identity) — matches prereg §5/§6.2 exactly; "do NOT reimplement" is
  honored.
- REFLECTION §4 calibration/null mandate: `research-wiki/REFLECTION_mllm_integration_failures.md:37-45`.
- C3 template + Z_best baseline lesson: `refine-logs/C3_FUSION_PROBE_RECORD.md` (§PRE-DECLARED DESIGN,
  §RESULTS); epitaph `autoresearch/goal_mllm_plus3/state/directions_tried.json` (C3-nontarget, 19th).
- Bans: `directions_tried.json` banned_constraints (OCR veto, gold-in-method, single-dataset, external API,
  MLLM-scores-as-training-signal, P1–P5, kNN-pool expansion) — all W2-A claims (§4.4) confirmed.
- CLIP + Qwen caches present for both datasets: `data/CLIP_Embedding/{HateMM,MHC}/` (listed this review).
