---
type: experiment
node_id: exp:exp-w2a-grounded
title: "W2-A — cross-modal GROUNDED retrieval key (transcript-conditioned Qwen2.5-VL vision representation): G0-cond zero-training oracle screen (PRE-REGISTRATION, DRAFT-UNREVIEWED)"
idea_id: ""
status: PRE-REGISTRATION r1 APPROVED-WITH-AMENDMENTS (applied) — GO-WITH-AMENDMENTS on the 2-3 GPU-h spend; CLEARED FOR IMPLEMENTATION; NO code authored yet; NO submission authorized (code review + hash-freeze precede the single Stage-E' submit)
verdict: approved-with-amendments
confidence: n/a
date: "2026-07-15"
hardware: "Stage E' (grounded extraction, the ONLY GPU): ~2-3 GPU-h, 1x A100, single sbatch, TWO frozen Qwen2.5-VL-7B forwards per video (grounded transcript-first + img-control for G-recon; img_feats/text_feats already banked). Stage P' (probe): CPU only / cloud-eligible on derived float caches, minutes; paired LOO kNN over <=851 memory videos, zero test-touch."
duration: "Stage E': ~2-3 GPU-h. Stage P': minutes on CPU."
provenance: "PRE-REGISTRATION ONLY — NO runs executed, NO code authored. Reuses banked pooled cache data/CLIP_Embedding/{HateMM,MHC}/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF.pt (2026-07-02; verified 2026-07-15: HateMM 744/107/215, MHC-EN 549/80/161, Dv=Dt=3584, L2-normed, HateMM train has 1 zero-guard row; img_feats excludes transcript, text_feats IS a joint frames+transcript forward pooled at the response span). Native transcript = data/gt/<ds>/<split>.jsonl 'text' field (whisper-large-v3 ASR; coverage HateMM 88-95%, MHC-EN 100%). Extraction lineage + causal-masking finding + chosen key-pooling design: refine-logs/W2A_FORENSIC_RECON.md. House standard: research-wiki/experiments/exp-s2s-r3.md, refine-logs/S2S_PROBE_DESIGN.md. Gate mandate: research-wiki/REFLECTION_mllm_integration_failures.md §4. D1 template: refine-logs/C3_FUSION_PROBE_RECORD.md."
added: 2026-07-15T00:00:00Z
tags: ["hateful-video", "MLLM-encoder", "frozen-Qwen", "cross-modal-grounding", "transcript-conditioned-vision", "retrieval-key", "representation-geometry", "G0-cond", "oracle-kill-switch", "concat-must-lose", "D1", "HateMM", "MHC-EN", "pre-registered", "DRAFT-UNREVIEWED", "W2-A"]
---

# W2-A — cross-modal GROUNDED retrieval key (transcript-conditioned Qwen vision representation) (PRE-REGISTRATION)

> **STATUS: `r1 APPROVED-WITH-AMENDMENTS (applied)` · `GO-WITH-AMENDMENTS` on the 2–3 GPU-h spend ·
> `CLEARED FOR IMPLEMENTATION`.** The independent pre-registration review
> (`refine-logs/W2A_PREREG_REVIEW.md`, commit e6be76c) verified BOTH load-bearing claims TRUE (banked
> `text_feats` is a joint forward; Qwen LLM is causal ⇒ transcript-first required, incl. the M-RoPE
> flip-side) and returned APPROVED-WITH-AMENDMENTS. The two BLOCKING amendments (Amdt 1: `Z_best` 4-way
> concat as the binding conditional-info baseline; Amdt 2: CONCAT-geometry — **option (b) chosen**, kNN raw
> bar demoted to advisory, weighting-invariant conditional-info probe is the sole binding performance gate)
> and non-blocking 3–7 are folded **in place** below (tagged `(Amdt #)`; see §15). **NEXT:** an implementer
> authors the Stage-E' extractor (transcript-first message builder + vision-pad pool are the novel code
> surfaces — `_build_messages`/`_encode` must be RE-authored, not imported) + sbatch + Stage-P' probe, then
> a SEPARATE independent code review + hash-freeze (incl. the §16 pre-declared constants) precede the single
> Stage-E' submit. Stage P' consumes ZERO test touches. The downstream head-training formal stage (§11) is
> NOT authorized here. Companion forensic recon (READ FIRST): `refine-logs/W2A_FORENSIC_RECON.md`; review:
> `refine-logs/W2A_PREREG_REVIEW.md`.**

**verdict:** `approved-with-amendments` (r1 applied) · **confidence:** n/a · **line name:** W2-A (grounded key). Round-3 wave-2 LEAD candidate.

## 1. Purpose (one line) + the recon that reshaped this prereg

Test — with a **zero-training, paired, oracle-gated** screen before any head GPU — whether a
**transcript-conditioned Qwen visual representation** (the vision-token pool from a joint forward in which
the native transcript precedes the frames, so the vision tokens causally attend to it) is a **better
retrieval key** than the banked marginals, and specifically whether it carries **conditional information
beyond `concat(img_feats, text_feats)`** — on HateMM and MHC-EN.

**Two recon findings (`W2A_FORENSIC_RECON.md`) bind this design and MUST survive review:**
1. **`text_feats` is already a JOINT forward** (frames+title+transcript, response-span pool). So the img×text
   interaction is **already partly banked** and already in the retrieval key. W2-A's real claim is the
   weaker "a **vision-side re-pool** of the joint forward adds info the response-span pool does not" — a
   C3-nontarget-shaped redundancy question. **⇒ the binding D1 baseline is `Z_best` = concat(CLIP img+text,
   Qwen img+text) (8960-d; r1 Amdt 1), and it MUST LOSE (in the weighting-invariant conditional-info probe,
   r1 Amdt 2 option (b)) for W2-A to live** — a grounded key CLIP-redundant against `Z_best` is DEAD.
2. **Qwen2.5-VL is causally masked** (`modeling_qwen2_5_vl.py:723`). A video-first vision-span "grounded"
   key is a **provable no-op** (vision tokens cannot attend to a downstream transcript). **⇒ the primary key
   is extracted from a TRANSCRIPT-FIRST forward**; a "grounding-live" positive control (§4) fails-closed if
   the transcript did not actually move the vision representation.

## 2. The mechanism under test (cell definition)

**One-line mechanism.** Run **one** frozen Qwen2.5-VL-7B forward per video with the message content ordered
`[text: native transcript] → [video: 8 frames] → [text: IMG_INSTRUCTION]`. Because the LLM is causal and the
transcript precedes the frames, each vision token's last-layer hidden state is **modulated by the
transcript**. Pool the **vision-token span** (mean of the vision-pad token hidden states), L2-norm → the
**grounded key `grd`** (3584-d). Retrieval (kNN top-20, rank-weighted signed-cosine vote — `metrics.py`,
UNCHANGED) runs over `grd` instead of the pooled marginals. Two videos with the same implicit
visual↔speech (in)congruity — benign frames under a hateful voice-over, a reclaimed slur over friendly
footage — now retrieve each other because the incongruity is carried **in the key geometry**, not as a
decision-side conflict scalar.

**Injection point + bandwidth class.** Retrieval **representation** construction (representation-geometry,
D2 — the only class that ever cleared +3). Bandwidth: one 3584-d key carrying an interaction term the two
marginals cannot separately form. **Not decision-side** (D1 does not bite the mechanism *class*, but D1 is
the empirical threat because one marginal, `text_feats`, is itself a joint pool — §3, §6.4).

**Key-pooling design (SETTLED, `W2A_FORENSIC_RECON.md` §3.3).** Primary = transcript-first, **vision-span**
pool (`grd`). Single sensitivity = same forward, **prefix-span** pool (vision+trailing-instruction,
`grd_pfx`). The video-first vision-span pool is a documented no-op and is used ONLY as the ungrounded
reference in the grounding-live control (§4). No other pooling variant is in the primary probe.

## 3. Why this cell is open — escapes D1/D2/D3 and every epitaph

- **vs C3-nontarget (19th, DEAD_AT_FUSION):** C3 **generated** dense MLLM reasoning text and fused it as a
  **channel**; it died because the info was banked in the Qwen pathway (≤+0.016 over concat(CLIP,Qwen)).
  W2-A generates nothing, adds no channel — it re-pools the **native whisper transcript** (already in the
  pipeline) inside the encoder forward. Different object, no generation. **The honest carry-over:** W2-A's
  interaction is even more banked (same joint forward as `text_feats`), so the §5 `Z_best` conditional-info
  arm (r1 Amdt 1/2b) is the exact C3-style test, and C3's negative is the prior we are trying to beat.
- **vs B1/B2/B4 (encoder line, D7-dead / negative):** those swap **which encoder** produces the pooled
  vector. W2-A keeps the frozen Qwen-7B and changes the **key-construction operation** (grounding via
  reordering). Non-isomorphic on slot; shares the risk that the same frozen representation failed to convert
  MHC-EN when pooled (SAV #18), which the grounding-adds-interaction escape must overcome empirically.
- **vs encoder-swap (positive, D7):** W2-A is the closest wave-2 candidate to the encoder line — a
  grounded-ENCODING mechanism. Composability with the swap is noted; a HateMM-only win does not newly satisfy
  the ≥2-dataset goal (HateMM already passes via the swap), so MHC-EN is the binding co-primary (§6.6).
- **vs S2S / C2 (wave-1):** S2S keeps a **vision-only frame set** and changes the *matching*; W2-A keeps a
  **single grounded** key and changes what the key *contains* (adds transcript-conditioning). Orthogonal and
  composable (grounded frame-*sets* = the natural W2-A×S2S cross, deferred).
- **Bans (`directions_tried.json`):** single-dataset own-train memory ✓; no OCR ✓ (native ASR); no gold
  in-method ✓ (gold only as probe ceiling, REFLECTION §4); no cross-seed ensemble ✓; no external API ✓; no
  MLLM-generated text / no generation ✓; not a P1–P5 re-proposal ✓; no kNN-pool expansion ✓; local Qwen-7B ✓.

## 4. Stage E' — grounded extraction + the extraction-correctness gate (the ONLY GPU)

The grounded forward has **NO banked twin** (novel transcript-first ordering), so correctness is certified
from internal consistency + banked anchors on control forwards (`W2A_FORENSIC_RECON.md` §5). The extractor
**imports the banked helpers verbatim** from `src/utils/generate_VideoMLLM_embedding_HF.py` (`_sample_frame_
indices`, decode helpers, `read_gt`, `IMG_INSTRUCTION`, `SPLIT_TO_OUTNAME`) so every forward-affecting knob
is the banked one by construction (S2S §3 precedent).

**Per video, TWO forwards:**
- **GROUNDED forward** — content `[{text: transcript}, {video: frames}, {text: IMG_INSTRUCTION}]`; store
  `grd` = L2(mean of vision-pad hidden states) and `grd_pfx` = L2(mean of vision+trailing-instruction span).
- **IMG-CONTROL forward** — content `[{video: frames}, {text: IMG_INSTRUCTION}]` (byte-identical to the
  banked `img_feats` recipe); store `ungrd_vis` = L2(mean of vision-pad hidden states) and `img_recon` =
  L2(mean of the banked prefix span).

**Gate order (HALT on any failure):**
0. **Grid gate + vision-pad mask assert** (exact, free; Amdt 7) — `n_vis == grid_t·(grid_h//2)·(grid_w//2)`
   and per-group size, from the model's own `video_grid_thw` + `spatial_merge_size=2`, in **both** forwards;
   AND assert the vision-pad positions (`input_ids == video_token_id`) form a **single contiguous block**
   whose count equals the grid count, with **identical mask logic in both forwards**. (G-recon-IMG validates
   the *prefix* span, not the vision-pad pool that actually produces `grd`/`ungrd_vis`, so the pool span must
   be pinned independently.) HALT on violation.
1. **G-recon-IMG** (banked parity anchor) — `img_recon` vs banked `img_feats[v]`: **cos ≥ 0.9999 AND
   max-abs ≤ 1e-3** for every non-guard video. This is the team-lead-named "text-ablated forward reproduces
   banked img_feats" gate; it proves the harness IS the banked forward. Report the cos/max-abs distribution.
2. **Grounding-LIVE positive control** (the analog of S2S's temporal control). **BINDING no-op VOID (fixed,
   pre-declared numeric):** if the **present-transcript-set median** `cos(grd, ungrd_vis) ≥ 0.999`, grounding
   is a silent no-op → **probe VOID**. (Amdt 4) `τ_live` — the per-video `cos < τ_live` check calibrated on a
   ≥20-video smoke subset as `τ_live = median(present cos) − 0.5·(present−empty gap)` — is **demoted to a
   logged diagnostic** (it self-calibrates to whatever the model produces and cannot fail on a
   weak-but-nonzero effect, so it is NOT a HALT bar): report it, do not gate on it. (Amdt 6) The
   **empty-transcript branch** (expecting `cos(grd, ungrd_vis) ≈ 1`) is likewise a **logged diagnostic, NOT a
   HALT**: a non-empty `"(none)"` block before the frames shifts vision M-RoPE positions by a few tokens and
   lets vision attend to `"(none)"`, so exact ≥0.999 is not guaranteed by position-shift alone — report the
   empty-set cos distribution. The binding content-sensitivity check is the placebo (gate 3, K3), not this
   branch.
3. **Placebo-transcript control** (subset ≥50 videos, HALT; Amdt 3) — PRIMARY: `grd` recomputed with a
   **cross-video MISMATCHED** transcript (assign video *j*'s transcript to video *i*'s frames, `j≠i`, a
   length-comparable partner) must differ from true-`grd` by cos `< 0.999` — this matches the probe's
   **cross-video** permutation null (§6.6), so the check isolates transcript *content* (which video's words)
   from *position/length* (held comparable). SECONDARY (diagnostic, non-gating): the within-video
   token-shuffle variant (order-sensitivity only). If the mismatched-transcript `grd` does not move → the key
   reflects position/length, not content → **VOID**.
4. **Length/parity invariant** — `last_hidden.shape[0]==input_ids.numel()` (banked preflight); log the
   M-RoPE vision-position offset introduced by the transcript-before ordering (correct, `get_rope_index`).

**Zero-guard policy:** undecodable video → zero `grd`/`grd_pfx`/`ungrd_vis` (identical to banked); handled
identically in all probe arms. Empty-transcript videos are NOT zero-guarded (the forward runs with
`"Transcript: (none)"`); their grounded key ≈ ungrounded by construction (mechanism vacuous on those rows,
logged, paired-contrast unbiased). **sbatch:** `conda activate HateVideo`; NO `--time`; single submit;
`JobHeldUser` → wait for auto-release, never force; `--limit 1` smoke to a throwaway path permitted first.
**Local GPU only** (raw video, license-sensitive — off cloud per CLAUDE.md). Resumable per-video shards.

## 5. Stage P' — arms (paired, on identical frozen keys; CPU / cloud-eligible; ZERO test touch)

Memory = **train ∪ val only** (HateMM 851 incl. 1 zero-guard; MHC-EN 629). Leave-one-out: each memory video
is a query against the rest; retrieve top-20 by each arm's score; call the **real** vote
(`metrics.py:compute_metrics_retrieval`, `use_sim=True`, `majority_voting='arithmetic'`, `topk=20` — do NOT
reimplement). Only the pairwise score changes across arms. **Fail-closed no-test-touch guard** (S2S N4):
never open any `test_seen*` file; assert `len(memory)==851 / 629`.

| arm | retrieval key (cosine kNN) | role |
|---|---|---|
| **POOLED-IMG** | banked `img_feats` | ungrounded-vision reference (NOT the primary null) |
| **CONCAT** (advisory kNN reference; Amdt 2b) | `[img_feats ‖ text_feats]` (7168-d, each L2-normed) | the fixed-50/50 Qwen-marginals kNN baseline — **ADVISORY only** (handicapped geometry, see below); reported, non-gating |
| **CONCAT-PCA** (advisory, dim-matched) | train∪val-fit PCA of `[img_feats‖text_feats]` → 3584-d (leak-free, memory-only) | advisory dim-matched reference (does NOT fix the weighting handicap) |
| **CONCAT-α** (advisory, weight-tuned) | `[√α·img_feats ‖ √(1−α)·text_feats]`, α on a small grid `{0.3,0.4,0.5,0.6,0.7}` chosen on train∪val LOO only (leak-free) | advisory best-weighted-concat reference (shows GROUNDED is not just beating a bad blend) |
| **GROUNDED (primary)** | `grd` | transcript-conditioned vision key |
| **GROUNDED+TEXT (sensitivity)** | `[grd ‖ text_feats]` | does grounding add on top of the banked joint summary? (cannot rescue a failed primary) |
| **GROUNDED-PFX (sensitivity)** | `grd_pfx` | pooling-span robustness (cannot rescue a failed primary) |
| **label-oracle (Fano)** | pairwise `+1/−1` gold-label agreement | machine-validity calibration (§6.3) |
| **oracle-ceiling** | per-query gold-guided choice grd-vs-concat (§6.4) | conservative binding kill-switch; NEVER a result |

**Binding performance adjudicator (Amdt 2, option (b) — CHOSEN; rationale recorded per directive).** The kNN
CONCAT baseline uses a **fixed 50/50** similarity weighting (`cos(concat_i,concat_j) = ½·cos_img + ½·cos_text`),
so a kNN win for GROUNDED could reflect a *poor CONCAT retrieval geometry* rather than more information — a
false-PASS; and CONCAT-PCA does **not** fix it (unsupervised variance ≠ label-useful img/text weighting). We
therefore adopt reviewer **option (b)**: the raw kNN bar (§6.5) is **DEMOTED TO ADVISORY**, and the
**weighting-invariant C3-template conditional-info probe (with the Amdt-1 `Z_best` baseline) is the SOLE
binding performance adjudicator**. Rationale: the conditional-info probe's logistic head re-weights the
img/text halves freely (weighting-invariant), it is the house's own G0-cond D1 instrument (REFLECTION §4),
and it is the exact instrument that killed C3-nontarget — so it can *detect* the `text_feats`-redundancy
instead of being fooled by a handicapped kNN metric. Option (a) (a memory-fit weight-optimized CONCAT-α as a
*binding* must-lose baseline) was **rejected as the binding path**: it patches a metric-shopping surface (the
kNN blend) rather than removing it, and the linear probe already provides the weighting-invariant test option
(a) approximates; CONCAT-α is retained only as an **advisory** arm (above). The kNN paired Δ(GROUNDED −
{CONCAT, CONCAT-PCA, CONCAT-α}) is still reported (it mirrors the downstream vote and corroborates), but it
does **not** gate the verdict.

**BINDING D1 arm — C3-template conditional-info probe with `Z_best` (Amdt 1, BLOCKING).** Run the
`C3_FUSION_PROBE` machinery with `Z = Z_best = concat(CLIP img[1024], CLIP text[768], Qwen img[3584], Qwen
text[3584])` = **8960-d** — the pipeline's *actual best banked config*, matching the C3 record's binding
decision cell (CLIP caches verified on disk for **both** datasets:
`data/CLIP_Embedding/{HateMM,MHC}/{train,dev_seen,test_seen}_openai_clip-vit-large-patch14-336_HF.pt`,
id/order-aligned to the Qwen caches) — and `A = grd`: un-penalized aux block appended at `Z_best`'s
inner-CV-optimal `C`, 5×5 RepeatedStratifiedKFold, per-video clustered bootstrap, **label-oracle calibration
arm (accZA ≈ 1.0 or MACHINERY_INVALID)** reaching full Fano headroom, **≥150-permutation null of `A` across
videos as a DISTRIBUTION**. **Binding +0.040 triple rule (C3-verbatim):** (C1) best conditional Δacc point
≥ **+0.040**, (C2) per-video-clustered bootstrap CI-lower > 0, (C3) real > **all** ≥150 permutation maxima
(family-corrected). **The C3 lesson, verbatim and BINDING: a grounded key that beats Qwen-only concat but is
CLIP-encoder-redundant (dies against `Z_best`) is DEAD.** Qwen-only `concat(img_feats,text_feats)` (7168-d)
is kept ONLY as reported secondary context, never the binding baseline. Train∪val only, gold PROBE-ONLY.

## 6. Gates, bars, dataset rule (binding)

### 6.1 Gate order (mirrors S2S §7; Amdt 2b)
0–4. Stage-E' extraction gates (§4) — HALT / VOID.
5. **Fano machine-validity** (§6.3) — void if label-oracle key acc < 0.99 either dataset.
6. **Oracle-ceiling kill-switch** (§6.4) — DEAD, no head GPU, if oracle Δacc < +0.04 on **every** dataset
   (conservative: the handicapped fixed-50/50 CONCAT baseline makes this kill *harder* to trigger — the safe
   direction; it never wrongly kills).
7. **BINDING performance adjudicator — C3-template conditional-info probe vs `Z_best`** (§5, §6.5) — PASS
   requires the **+0.040 triple rule** against `Z_best` (8960-d). This is the **sole binding performance gate**.
8. **Advisory kNN corroboration** (§6.5) — the raw Δ(GROUNDED − {CONCAT, CONCAT-PCA, CONCAT-α}) bar
   (+0.05/+0.05 on HateMM) + permutation null + bootstrap are **reported as advisory** (they mirror the
   downstream vote) but do **NOT** gate the verdict (handicapped kNN geometry, Amdt 2b).
9. **Dataset rule** (§6.6) — assign outcome (a)/(b)/(c)/(d).

Sensitivity arms (GROUNDED+TEXT, GROUNDED-PFX) and all advisory CONCAT arms **cannot rescue a failed binding
adjudicator** (S2S N3).

### 6.2 Primary metric (declared before results; Amdt 2b)
**BINDING:** the C3-template conditional-info Δacc of `grd` over **`Z_best`** (8960-d) under the +0.040 triple
rule (§5, §6.1 gate 7). **ADVISORY (reported, non-gating):** the paired kNN **Δ(GROUNDED − CONCAT)** in
accuracy AND macro-F1, per dataset, LOO on train∪val, plus AUC — the pipeline's continuous rank-weighted
signed-cosine vote (acc cut `vote ≥ 0`; macro-F1 = goal metric, `metrics.py:309`). Test untouched.

### 6.3 Fano calibration arm (MANDATORY, REFLECTION §4)
Pairwise score `+1` if `label(q)==label(m)` else `−1` (tie-break by memory index). Vote acc **must reach
≥ 0.99** on both datasets, else the vote machine is VOID and **no negative verdict is acceptable**. Gold =
machine check only.

### 6.4 Oracle-ceiling kill-switch (pre-declared, binding — decides whether ANY head GPU is spent)
Per query `Q` with gold `y_Q`, compute the LOO vote margin under the CONCAT key `v_cat(Q)` and under the
GROUNDED key `v_grd(Q)`; the oracle selects, per query, the key whose margin most confidently votes the
correct label: `choose grd iff (2y_Q−1)·v_grd(Q) > (2y_Q−1)·v_cat(Q)` (tie → concat, so the oracle can never
score *below* concat). Report paired **Δ(oracle − CONCAT)**. This upper-bounds how much the grounded key
*could* add if we knew when to trust it. Gold enters only to pick which **key** to trust per query
(video-level label, no time-span gold). **Ordering expectation:** oracle Δ ≥ raw Δ; a raw Δ materially
exceeding the oracle ⇒ construction bug (investigate, do NOT auto-KILL).

> **KILL-SWITCH (binding).** If the oracle-ceiling paired **Δacc < +0.04** on **every** dataset, the
> grounded key carries no convertible information beyond `concat(img_feats, text_feats)` → **W2-A DEAD**,
> zero head GPU, cross-modal-grounding cell closed. The oracle ceiling is an upper bound and can **NEVER**
> be claimed as a result. **(Amdt 2b — conservative note.)** This kill-switch uses the fixed-50/50 CONCAT kNN
> geometry, which *inflates* the oracle gap (a handicapped baseline is easier to beat), so it fires only when
> even a lenient baseline shows no headroom — the safe direction. The **binding redundancy test remains the
> §5 `Z_best` conditional-info probe**, which is weighting-invariant and CLIP-augmented.

### 6.5 Raw survival bar (pre-declared, **ADVISORY — Amdt 2b**; prices in P3's "probe pass ≠ train gain")
> **RAW BAR (ADVISORY, non-gating).** On the **primary dataset (HateMM)**, mean paired **Δacc ≥ +0.05 AND
> Δmacro-F1 ≥ +0.05** for **GROUNDED − CONCAT** (and GROUNDED beats CONCAT-PCA and CONCAT-α in sign), with the
> bootstrap 5th-pct of the paired Δ **> 0** (D3) and the observed Δ **above the 95th pct of the permutation
> null** (§6.6). **This bar is reported as corroboration** (it mirrors the downstream vote) but is **NOT the
> binding gate** — the handicapped kNN CONCAT geometry can false-PASS (Amdt 2b), so the binding performance
> gate is the §5 `Z_best` conditional-info probe (§6.1 gate 7). **Bar derivation (identical logic to S2S
> §6.5):** the probe measures a zero-training LOO Δ on train∪val, optimistic on (i) near-duplicate
> self-retrieval + no generalization gap and (ii) an un-adapted baseline the head can later compensate (the
> P3 "probe passes, training flat" failure, ≥4× in the graveyard). Price a ~1.7× pessimism factor → +0.05.

### 6.6 Permutation null + dataset rule
- **Permutation null (≥100 fresh seeds, 0..99):** shuffle the per-video **grounded keys across videos**
  (destroying query↔key alignment), **same permutation both arms within a seed** (paired Δ preserved);
  observed Δ must exceed the 95th percentile. Report the full null distribution. (Advisory, per §6.5.)
- **Near-duplicate audit (A3, MANDATORY):** flag distinct memory pairs with `cos(grd_i, grd_j) ≥ 0.995` OR
  `cos(concat_i, concat_j) ≥ 0.995`; report counts at 0.98/0.99/0.995; re-run LOO dropping flagged
  neighbours — the GROUNDED advantage must survive (guards against duplicate-rediscovery, esp. MHC
  re-uploads).
- **Covered-rows-only HateMM secondary (Amdt 5; SECONDARY, non-primary, non-shopping).** Because HateMM has a
  ~8% empty-transcript tail on which the mechanism is vacuous (§4 zero-guard policy), pre-declare a
  **secondary** covered-rows-only view: recompute the binding `Z_best` conditional-info Δ (and the advisory
  kNN Δ) restricted to HateMM videos with a **non-empty transcript**, so the ~8% dilution cannot mask a real
  covered-row effect. This is reported alongside the full-set primary and is **not** a substitute for it and
  **not** a dataset-shopping surface (both are pre-declared now). MHC-EN's 100% coverage makes this a
  HateMM-only view.
- **Dataset rule (pre-declared; HateMM primary, MHC-EN binding-gap co-primary — the four fixed rows).**
  HateMM = primary mechanism-existence (G-recon anchor; highest overall prior) but with a declared **~8%
  empty-transcript vacuous-row dilution**. MHC-EN = binding-gap co-primary (advances the ≥2-dataset goal;
  **100% transcript coverage → fully exercises the grounding mechanism**), weaker historical prior
  (data/label-limited). Outcomes: **(a)** both clear → strongest, both license a formal stage; **(b)** HateMM
  clears, MHC-EN fails → mechanism real, composable with the swap, **binding gap NOT closed** (report as
  such, not "goal met"); **(c)** MHC-EN clears, HateMM fails → advances the goal on the binding dataset
  (report the HateMM null honestly); **(d)** neither → DEAD, cross-modal-grounding cell closed. **No post-hoc
  dataset shopping.**

## 7. D3 guards
Bootstrap ≥1000 LOO-query resamples per dataset, report 5/50/95 pct of paired Δacc/Δmacro-F1; a pass whose
5th-pct crosses 0 is **D3-FRAGILE**. No per-seed training variance (zero training) → no "3/3 seeds" claim at
Stage P'; the paired design cancels representation noise. Zero-guard rows + empty-transcript rows logged and
handled identically in all arms.

## 8. Novelty scope statement (honest; a user ruling, not decided here)
The raw operation — text/instruction-conditioned visual representations in decoder VLMs — is **established**
(iGVLM 2603.02748, TIE 2511.20770, Text-Guided Layer Fusion 2601.03100), and W2-A's realization (decoder
causal attention + reordering) is **more generic** than those trained conditioning pathways, not less. The
defensible novelty is **composite / domain-transfer only:** first use of a transcript-conditioned MLLM
visual representation as the **retrieval key** in hateful-video detection, so implicit visual–speech
incongruity enters the **retrieval geometry** rather than a decision-side conflict head (where all in-domain
incongruity work lives — CMFusion, MM-HSD, reasoning-fusion). **Whether this clears the novelty clause is a
D7-class USER RULING, identical in kind to S2S / B3-LoRA — NOT decided here.** This file decides only the
**performance** clause, and only its G0-cond screen. **Binding-gap honesty:** only an MHC-EN (or later ZH)
pass newly advances the ≥2-dataset goal.

## 9. Test-touch ledger (binding)
| stage | test data used? | touches |
|---|---|---|
| Stage E' extraction | test grounded keys **extracted and cached** (for the later formal stage) but **not scored** | 0 |
| Stage P' probe (all arms, incl. oracle ceiling + Fano + C3-template) | **train ∪ val only**; test never retrieved, never voted | **0** |
| Downstream formal stage (§11, not authorized here) | test scored under frozen 3-seed both-protocol ceremony | (spent later) |

Stage P' is a **ZERO test-touch** screen. Oracle/Fano/C3-template use **gold labels of train∪val only** as a
probe ceiling / machine check (REFLECTION §4 compliant). The probe script enforces fail-closed no-test-touch.

## 10. Cost estimate
| item | cost |
|---|---|
| Stage E' grounded extraction (both datasets, 1856 videos × 2 forwards) | **~2–3 GPU-h**, 1× A100, single sbatch (local) |
| placebo-transcript control (≥50-video subset) | negligible |
| Stage E' storage (grd, grd_pfx, ungrd_vis, img_recon; fp16) | **~55 MB**, sub-GB |
| Stage P' probe (all arms, both datasets, ≥100 kNN null seeds, ≥1000 bootstrap, C3-template `Z_best` ≥150 perms; loads banked CLIP+Qwen caches, features-only) | **minutes on CPU / cloud** |
| Downstream head-training formal stage (only if screen passes) | ~1–2 GPU-h (later prereg) |

## 11. Downstream head-training design sketch (NOT authorized; later prereg only)
Only if Stage P' survives: adapt the triplet+BCE head to the grounded key — a light projection `φ(grd)`
(single linear `3584→map_dim`, ≤1024) trained by the existing triplet-margin + BCE objective with the
grounded-key cosine substituted for the pooled cosine; inference vote unchanged (top-20). Composability with
the encoder swap and with S2S (grounded frame-sets) noted. Full formal pre-registration (3-seed, both
protocols, G-repro anchors, verbatim +0.03/+0.03 rule) written **after** the screen passes, separately
reviewed.

## 12. What-would-kill-this table
| # | killer | where |
|---|---|---|
| K0 | grid gate: `n_vis` ≠ grid count, or vision-pad positions not a contiguous grid-count block, in either forward (Amdt 7) | §4 gate 0 → HALT |
| K1 | G-recon-IMG cos < 0.9999 or max-abs > 1e-3 (fresh img-control ≠ banked) | §4 gate 1 → HALT |
| K2 | grounding-live: **present-set median** `cos(grd, ungrd_vis) ≥ 0.999` (transcript is a silent no-op). *(τ_live + empty-branch are logged diagnostics, NOT HALT — Amdt 4/6)* | §4 gate 2 → VOID |
| K3 | placebo: **cross-video mismatched** transcript does not move `grd` (`cos ≥ 0.999`) → key reflects position/length, not content (Amdt 3) | §4 gate 3 → VOID |
| K4 | Fano ±1 gold-key acc < 0.99 (vote machine void) | §6.3 → probe void |
| K5 | oracle-ceiling Δacc < +0.04 on **every** dataset (no headroom even vs the lenient CONCAT baseline — conservative) | §6.4 → DEAD, no head GPU |
| **K9 (BINDING)** | **C3-template conditional-info probe vs `Z_best`(8960-d): `grd` Δacc < +0.040 OR CI-lower ≤ 0 OR not > all ≥150 perm maxima — a grounded key CLIP-redundant against `Z_best` is DEAD (Amdt 1/2b, the SOLE binding performance gate)** | §5, §6.1 gate 7 |
| K6 (advisory) | raw HateMM kNN Δ(GROUNDED − CONCAT) acc < +0.05 OR mF1 < +0.05, or not beating CONCAT-PCA/CONCAT-α in sign — **corroborative, non-gating** (Amdt 2b) | §6.5 |
| K7 (advisory) | observed kNN Δ ≤ 95th pct of the key-shuffle permutation null | §6.6 |
| K7b | GROUNDED advantage does not survive near-dup-excluded retrieval | §6.6 |
| K8 (advisory) | bootstrap 5th-pct of paired kNN Δ crosses 0 (D3-fragile) | §7 |
| K10 | MHC-EN fails while HateMM passes → mechanism real but binding gap not closed (honest partial) | §6.6 rule (b), §8 |

## 13. Honest prior / expected outcome (declared before running)
**Prior: LOW–MODEST** (revised down from the scout's MODEST–FAIR by the recon: the interaction is already
partly banked in `text_feats`, making this a C3-nontarget-shaped redundancy question that the analogous C3
probe answered *no*; and the cheap reordering realization is more generic than the cited conditioning-pathway
prior art). Falsifiable: if the transcript-conditioned visual key does not carry **+0.040 conditional Δacc
over `Z_best` = concat(CLIP img+text, Qwen img+text)** (the binding C3-template triple rule, §5) on ≥1
dataset — while its oracle-ceiling clears +0.04 and the advisory kNN corroborates — then the interaction is
redundant with the banked marginals (including CLIP) and W2-A is dead. Most likely *informative* outcomes:
(b) HateMM-only (mechanism real, gap open), (d) both fail (cell closed), or a clean C3-style CLIP-redundancy
null. MHC-EN — 100% transcript coverage — is the honest fair-coin for whether grounding converts on the
binding dataset.

## 14. Connections
- Candidate spec: `research-wiki/ROUND3_CANDIDATES_WAVE2_2026-07-15.md` §W2-A.
- Forensic recon (READ FIRST — transcripts reality, causal-masking finding, key-pooling decision):
  `refine-logs/W2A_FORENSIC_RECON.md`.
- Independent pre-registration review (r1 amendments source): `refine-logs/W2A_PREREG_REVIEW.md`.
- House standard: `research-wiki/experiments/exp-s2s-r3.md`, `refine-logs/S2S_PROBE_DESIGN.md`.
- D1 conditional-info template: `refine-logs/C3_FUSION_PROBE_RECORD.md`.
- Gate mandate + calibration erratum: `research-wiki/REFLECTION_mllm_integration_failures.md` §4.
- Graveyard + bans + D-laws: `autoresearch/goal_mllm_plus3/state/directions_tried.json`.
- Extraction lineage: `src/utils/generate_VideoMLLM_embedding_HF.py`; causal backbone
  `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:723,989,1244`; retrieval core
  `src/utils/metrics.py:262-320`, `src/model/evaluate_rac.py`.

## 15. Revision history
- **2026-07-15 — v1 DRAFT-UNREVIEWED.** Initial pre-registration. Establishes Stage E' (grounded
  transcript-first extraction, only GPU), Stage P' (zero-training zero-test-touch oracle screen), the
  extraction-correctness gate suite (G-recon-IMG banked anchor + grid gate + grounding-live positive control
  + placebo-transcript control — the "text-ablated reproduces img_feats" analog for a forward with no banked
  twin), the SETTLED transcript-first vision-span key-pooling design (video-first is a documented no-op under
  causal masking), the **concat(img_feats,text_feats)-must-lose** binding D1 arm (because `text_feats` is
  itself a joint pool), the oracle-ceiling kill-switch (+0.04), the raw survival bar (+0.05 acc AND F1,
  P3-priced), the Fano machine-validity arm, the C3-template conditional-info confirmatory arm, the
  permutation-null distribution + near-dup audit, and the HateMM-primary / MHC-EN-binding-gap dataset rule.
  Prior revised to LOW–MODEST. Awaiting independent pre-registration review; NO code authored; NO submission
  authorized.
- **2026-07-15 — r1 APPROVED-WITH-AMENDMENTS (amendments applied).** Independent pre-registration review
  (`refine-logs/W2A_PREREG_REVIEW.md`) verified both load-bearing claims TRUE (joint `text_feats`; causal
  Qwen ⇒ transcript-first required; M-RoPE flip-side confirmed) and returned APPROVED-WITH-AMENDMENTS /
  GO-WITH-AMENDMENTS on the 2–3 GPU-h spend. Folded the two BLOCKING + five non-blocking amendments in place:
  - **Amdt 1 (BLOCKING)** — the C3-template conditional-info probe's binding baseline is now
    **`Z_best = concat(CLIP img+text, Qwen img+text)` (8960-d)**, matching the C3 record's decision cell
    (CLIP caches verified on disk both datasets, id/order-aligned); Qwen-only concat kept as secondary
    context. Closes the "beats Qwen-alone but CLIP-redundant" false-pass that killed C3-nontarget (§5, §6.1
    gate 7, K9, §16).
  - **Amdt 2 (BLOCKING) — CONCAT geometry, OPTION (b) CHOSEN.** The kNN CONCAT is a fixed-50/50 blend that can
    false-PASS (CONCAT-PCA does not fix it). **Decision + rationale (recorded per directive):** demote the raw
    kNN bar (§6.5, K6) to **ADVISORY** and make the **weighting-invariant `Z_best` conditional-info probe the
    SOLE binding performance adjudicator** (§6.1 gate 7). Chosen over option (a) because the conditional-info
    probe's logistic head re-weights the halves freely, it is the house's own G0-cond D1 instrument, and it is
    the exact instrument that killed C3-nontarget — option (a)'s weight-optimized CONCAT-α patches a
    metric-shopping surface rather than removing it, so CONCAT-α is retained only as an advisory arm. The
    oracle-ceiling kill-switch (§6.4) stays binding but is noted conservative (handicapped CONCAT ⇒ harder to
    kill). (§5, §6.1, §6.2, §6.4, §6.5, K5/K6/K9.)
  - **Amdt 3** — placebo is now a **cross-video mismatched** transcript (matches the cross-video permutation
    null); within-video token-shuffle kept as secondary diagnostic (§4 gate 3, K3).
  - **Amdt 4** — `τ_live` demoted to a logged diagnostic; the binding no-op VOID is the fixed pre-declared
    present-set-median `cos ≥ 0.999` (§4 gate 2, K2).
  - **Amdt 5** — pre-declared a **covered-rows-only HateMM secondary** Δ (non-primary, non-shopping) so the
    ~8% empty-transcript dilution cannot mask a covered-row effect (§6.6).
  - **Amdt 6** — the gate-2 empty-transcript-branch cos check demoted to a logged diagnostic (M-RoPE
    position-shift + `"(none)"`-attention confound); binding content-check is the placebo (§4 gate 2).
  - **Amdt 7** — added a vision-pad contiguity + grid-count assertion in **both** forwards (the vision-pad
    pool that produces `grd`/`ungrd_vis` is pinned independently of the G-recon-IMG prefix span) (§4 gate 0).
  - Added **§16 pre-declared constants (hash-frozen)** and enforced raw-only results transcription
    (numeric-provenance discipline). NO code authored; the transcript-first message builder + vision-pad pool
    are RE-authored (not imported); a separate independent code review + hash-freeze precede the single
    Stage-E' submit.

## 16. Pre-declared constants (hash-frozen at r1; re-verify at submit)
Frozen **now** so no post-hoc tuning is possible; the implementer hash-freezes this table with the scripts
before the single Stage-E' submit. Numbers are transcribed raw from the primary results JSON at verdict time
(numeric-provenance discipline — no fabricated companion metrics).

| constant | value | where |
|---|---|---|
| CONCAT-geometry option | **(b)** — kNN raw bar advisory; conditional-info probe is the sole binding perf. gate | §5, §6.1 |
| binding conditional-info baseline `Z_best` | `concat(CLIP img[1024], CLIP text[768], Qwen img[3584], Qwen text[3584])` = **8960-d** | §5 |
| secondary conditional-info context | Qwen-only `concat(img_feats,text_feats)` = 7168-d | §5 |
| conditional-info bar (triple rule) | C1 Δacc ≥ **+0.040**, C2 per-video-clustered bootstrap CI-lower > 0, C3 real > **all** perm maxima | §5, §6.1 gate 7 |
| conditional-info machinery | un-penalized aux at `Z_best` inner-CV-optimal C; 5×5 RepeatedStratifiedKFold; label-oracle calib accZA ≈ 1.0 (else MACHINERY_INVALID) | §5 |
| conditional-info permutation null | **≥150** permutations of `A=grd` across videos, as a DISTRIBUTION | §5 |
| oracle-ceiling kill-switch | Δacc < **+0.04** on **every** dataset → DEAD (conservative; per-query gold choice grd-vs-concat, tie→concat) | §6.4 |
| Fano machine-validity | ±1 gold-label key vote acc ≥ **0.99** both datasets else VOID | §6.3 |
| grounding-live binding VOID | present-transcript-set **median** `cos(grd, ungrd_vis) ≥ 0.999` → VOID | §4 gate 2 |
| τ_live (diagnostic only) | `median(present cos) − 0.5·(present−empty gap)`, ≥20-video smoke calib — logged, non-gating | §4 gate 2 |
| placebo (binding) | cross-video **mismatched** transcript must move `grd` (`cos < 0.999`); ≥50-video subset | §4 gate 3 |
| advisory kNN raw bar (HateMM) | Δ(GROUNDED−CONCAT) acc ≥ **+0.05** AND mF1 ≥ **+0.05**; beat CONCAT-PCA + CONCAT-α in sign — non-gating | §6.5 |
| advisory kNN permutation null | seeds **0..99** (≥100), key-shuffle across videos, same perm both arms; obs > 95th pct | §6.6 |
| CONCAT-α grid | α ∈ `{0.3,0.4,0.5,0.6,0.7}`, chosen on train∪val LOO only (leak-free), advisory | §5 |
| bootstrap | ≥**1000** LOO-query resamples; 5th-pct > 0 else D3-FRAGILE (advisory) | §7 |
| near-dup flag | `cos ≥ 0.995` (grd or concat); report at 0.98/0.99/0.995; SET advantage must survive exclusion | §6.6 |
| covered-rows-only HateMM | secondary Δ (binding + advisory) on non-empty-transcript rows; non-primary, non-shopping | §6.6 |
| memory sizes (fail-closed) | HateMM 851, MHC-EN 629 (train∪val); test never opened | §5, §9 |
| datasets | HateMM primary (~8% empty-transcript dilution), MHC-EN binding-gap co-primary (100% coverage) | §6.6 |
