---
type: experiment
node_id: exp:exp-w2a-grounded
title: "W2-A — cross-modal GROUNDED retrieval key (transcript-conditioned Qwen2.5-VL vision representation): G0-cond zero-training oracle screen (PRE-REGISTRATION, DRAFT-UNREVIEWED)"
idea_id: ""
status: PRE-REGISTRATION DRAFT-UNREVIEWED — awaiting fresh independent pre-registration review; NO code authored; NO submission authorized
verdict: draft-unreviewed
confidence: n/a
date: "2026-07-15"
hardware: "Stage E' (grounded extraction, the ONLY GPU): ~2-3 GPU-h, 1x A100, single sbatch, TWO frozen Qwen2.5-VL-7B forwards per video (grounded transcript-first + img-control for G-recon; img_feats/text_feats already banked). Stage P' (probe): CPU only / cloud-eligible on derived float caches, minutes; paired LOO kNN over <=851 memory videos, zero test-touch."
duration: "Stage E': ~2-3 GPU-h. Stage P': minutes on CPU."
provenance: "PRE-REGISTRATION ONLY — NO runs executed, NO code authored. Reuses banked pooled cache data/CLIP_Embedding/{HateMM,MHC}/{train,dev_seen,test_seen}_Qwen2.5-VL-7B-Instruct_HF.pt (2026-07-02; verified 2026-07-15: HateMM 744/107/215, MHC-EN 549/80/161, Dv=Dt=3584, L2-normed, HateMM train has 1 zero-guard row; img_feats excludes transcript, text_feats IS a joint frames+transcript forward pooled at the response span). Native transcript = data/gt/<ds>/<split>.jsonl 'text' field (whisper-large-v3 ASR; coverage HateMM 88-95%, MHC-EN 100%). Extraction lineage + causal-masking finding + chosen key-pooling design: refine-logs/W2A_FORENSIC_RECON.md. House standard: research-wiki/experiments/exp-s2s-r3.md, refine-logs/S2S_PROBE_DESIGN.md. Gate mandate: research-wiki/REFLECTION_mllm_integration_failures.md §4. D1 template: refine-logs/C3_FUSION_PROBE_RECORD.md."
added: 2026-07-15T00:00:00Z
tags: ["hateful-video", "MLLM-encoder", "frozen-Qwen", "cross-modal-grounding", "transcript-conditioned-vision", "retrieval-key", "representation-geometry", "G0-cond", "oracle-kill-switch", "concat-must-lose", "D1", "HateMM", "MHC-EN", "pre-registered", "DRAFT-UNREVIEWED", "W2-A"]
---

# W2-A — cross-modal GROUNDED retrieval key (transcript-conditioned Qwen vision representation) (PRE-REGISTRATION)

> **STATUS: `DRAFT-UNREVIEWED`. This pre-registration is NOT authorized to run. It awaits (1) a fresh
> independent pre-registration review, then (2) an implementer to author the Stage-E' extractor + sbatch +
> Stage-P' probe scripts, then (3) a SEPARATE independent code review + hash-freeze before the single
> Stage-E' submit. Stage P' consumes ZERO test touches. The downstream head-training formal stage (§11) is
> NOT authorized here — it is a later, separately pre-registered gate behind the Stage-P' oracle
> kill-switch. Companion forensic recon (READ FIRST): `refine-logs/W2A_FORENSIC_RECON.md`.**

**verdict:** `draft-unreviewed` · **confidence:** n/a · **line name:** W2-A (grounded key). Round-3 wave-2 LEAD candidate.

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
   C3-nontarget-shaped redundancy question. **⇒ the D1 baseline is `concat(img_feats, text_feats)`, and it
   MUST LOSE for W2-A to live.**
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
  interaction is even more banked (same joint forward as `text_feats`), so the §6.4 concat-must-lose arm is
  the exact C3-style test, and C3's negative is the prior we are trying to beat.
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
0. **Grid gate** (exact, free) — `n_vis == grid_t·(grid_h//2)·(grid_w//2)` and per-group size, from the
   model's own `video_grid_thw` + `spatial_merge_size=2`, in **both** forwards. Locates the vision span.
1. **G-recon-IMG** (banked parity anchor) — `img_recon` vs banked `img_feats[v]`: **cos ≥ 0.9999 AND
   max-abs ≤ 1e-3** for every non-guard video. This is the team-lead-named "text-ablated forward reproduces
   banked img_feats" gate; it proves the harness IS the banked forward. Report the cos/max-abs distribution.
2. **Grounding-LIVE positive control** (the analog of S2S's temporal control) — split by transcript
   presence: **present** videos must satisfy `cos(grd, ungrd_vis) < τ_live` (the transcript materially moved
   the vision rep); **empty-transcript** videos must satisfy `cos(grd, ungrd_vis) ≥ 0.999` (no transcript →
   no change, as causal masking predicts). `τ_live` is **pre-declared before the real run**, calibrated on a
   ≥20-video smoke subset as `τ_live = (median present-video cos) − 0.5·(present−empty gap)`, and frozen in
   the hash-freeze. If the present-set median cos ≥ 0.999 (grounding is a silent no-op) → **probe VOID**.
3. **Placebo-transcript control** (subset ≥50 videos, HALT) — `grd` with a **shuffled** transcript (tokens
   permuted within-video) must differ from true-`grd` by cos `< 0.999`; else the key reflects transcript
   *position/length*, not *content* → VOID.
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
| **CONCAT** (the D1 baseline) | `[img_feats ‖ text_feats]` (7168-d, each L2-normed then concatenated) | **the marginals baseline that MUST LOSE** — already contains the joint `text_feats` pool |
| **CONCAT-PCA** (dim-matched) | train-fold PCA of `[img_feats‖text_feats]` → 3584-d (leak-free, fit on memory only) | capacity/dim-matched D1 baseline |
| **GROUNDED (primary)** | `grd` | transcript-conditioned vision key |
| **GROUNDED+TEXT (sensitivity)** | `[grd ‖ text_feats]` | does grounding add on top of the banked joint summary? (cannot rescue a failed primary) |
| **GROUNDED-PFX (sensitivity)** | `grd_pfx` | pooling-span robustness (cannot rescue a failed primary) |
| **label-oracle (Fano)** | pairwise `+1/−1` gold-label agreement | machine-validity calibration (§6.3) |
| **oracle-ceiling** | per-query gold-guided choice grd-vs-concat (§6.4) | upper bound; NEVER a result |

**Binding primary decision (concat-must-lose, pre-declared arm ordering).** The reported **primary paired Δ
is `GROUNDED − CONCAT`** (NOT grounded − img). A grounded key that beats POOLED-IMG but not CONCAT is a
**KILL** — because CONCAT already contains `text_feats`, which is itself a joint-forward pool. GROUNDED must
beat **both CONCAT and CONCAT-PCA** (acc AND macro-F1) to survive.

**Mandatory D1 confirmatory arm (C3-template conditional-info probe).** In addition to the kNN arms, run the
`C3_FUSION_PROBE`-style linear conditional-info probe: `Z = concat(img_feats, text_feats)`, `A = grd`,
un-penalized aux block appended at `Z`'s inner-CV-optimal C, 5×5 RepeatedStratifiedKFold, per-video
clustered bootstrap, **label-oracle calibration arm (accZA ≈ 1.0 or MACHINERY_INVALID)**, **≥150-permutation
null of `A` across videos**. This answers "does `grd` carry conditional info beyond the marginals" at the
same rigor that killed C3-nontarget. Bar = **+0.040** conditional Δacc, CI-lower > 0, real > all permutation
maxima (the exact C3 triple rule). Train∪val only, gold PROBE-ONLY.

## 6. Gates, bars, dataset rule (binding)

### 6.1 Gate order (mirrors S2S §7)
0–4. Stage-E' extraction gates (§4) — HALT / VOID.
5. **Fano machine-validity** (§6.3) — void if label-oracle key acc < 0.99 either dataset.
6. **Oracle-ceiling kill-switch** (§6.4) — DEAD, no head GPU, if oracle Δacc < +0.04 on **every** dataset.
7. **Raw concat-must-lose bar + permutation null + bootstrap** (§6.5) — survival test on HateMM primary.
8. **C3-template conditional-info confirmatory arm** (§5) — must also favor `grd` over concat (+0.040 triple
   rule); a kNN pass that the conditional-info probe contradicts is treated as a retrieval-geometry artifact.
9. **Dataset rule** (§6.6) — assign outcome (a)/(b)/(c)/(d).

Sensitivity arms (GROUNDED+TEXT, GROUNDED-PFX, CONCAT-PCA-only) **cannot rescue a failed primary** (S2S N3).

### 6.2 Primary metric (declared before results)
Paired **Δ(GROUNDED − CONCAT)** in **accuracy AND macro-F1**, per dataset, LOO on train∪val, plus AUC. The
vote is the pipeline's continuous rank-weighted signed-cosine vote; acc cut `vote ≥ 0`; macro-F1 = goal
metric (`metrics.py:309`). Test untouched.

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
> be claimed as a result.

### 6.5 Raw survival bar (pre-declared; prices in P3's "probe pass ≠ train gain")
> **RAW BAR (binding).** On the **primary dataset (HateMM)**, mean paired **Δacc ≥ +0.05 AND Δmacro-F1
> ≥ +0.05** for **GROUNDED − CONCAT** (and GROUNDED must also beat CONCAT-PCA in sign), with the bootstrap
> 5th-pct of the paired Δ **> 0** (D3) and the observed Δ **above the 95th pct of the permutation null**
> (§6.6). **Bar derivation (identical logic to S2S §6.5):** the probe measures a zero-training LOO Δ on
> train∪val, optimistic on (i) near-duplicate self-retrieval + no generalization gap and (ii) an un-adapted
> baseline the head can later compensate (the P3 "probe passes, training flat" failure, ≥4× in the
> graveyard). Price a ~1.7× pessimism factor → require +0.05 raw to believe +0.03 converts downstream.

### 6.6 Permutation null + dataset rule
- **Permutation null (≥100 fresh seeds, 0..99):** shuffle the per-video **grounded keys across videos**
  (destroying query↔key alignment), **same permutation both arms within a seed** (paired Δ preserved);
  observed Δ must exceed the 95th percentile. Report the full null distribution.
- **Near-duplicate audit (A3, MANDATORY):** flag distinct memory pairs with `cos(grd_i, grd_j) ≥ 0.995` OR
  `cos(concat_i, concat_j) ≥ 0.995`; report counts at 0.98/0.99/0.995; re-run LOO dropping flagged
  neighbours — the GROUNDED advantage must survive (guards against duplicate-rediscovery, esp. MHC
  re-uploads).
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
| Stage P' probe (all arms, both datasets, ≥100 null seeds, ≥1000 bootstrap, C3-template ≥150 perms) | **minutes on CPU / cloud** |
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
| K0 | grid gate: `n_vis` ≠ grid count (wrong video_pad_id) or per-group size | §4 gate 0 → HALT |
| K1 | G-recon-IMG cos < 0.9999 or max-abs > 1e-3 (fresh img-control ≠ banked) | §4 gate 1 → HALT |
| K2 | grounding-live control: present-set median `cos(grd, ungrd_vis) ≥ 0.999` (transcript is a silent no-op) | §4 gate 2 → VOID |
| K3 | placebo control: `grd` insensitive to transcript CONTENT (only position/length) | §4 gate 3 → VOID |
| K4 | Fano ±1 gold-key acc < 0.99 (vote machine void) | §6.3 → probe void |
| K5 | oracle-ceiling Δacc < +0.04 on **every** dataset (no convertible info beyond concat) | §6.4 → DEAD, no head GPU |
| K6 | raw HateMM Δ(GROUNDED − CONCAT) acc < +0.05 OR mF1 < +0.05, OR GROUNDED does not beat CONCAT-PCA in sign | §6.5 |
| K7 | observed Δ ≤ 95th pct of the key-shuffle permutation null | §6.6 |
| K7b | GROUNDED advantage does not survive near-dup-excluded retrieval | §6.6 |
| K8 | bootstrap 5th-pct of paired Δ crosses 0 (D3-fragile) | §7 |
| K9 | C3-template conditional-info probe: `grd` Δacc over concat < +0.040 / CI-lower ≤ 0 / not > all perm maxima | §5, §6.1 gate 8 |
| K10 | MHC-EN fails while HateMM passes → mechanism real but binding gap not closed (honest partial) | §6.6 rule (b), §8 |

## 13. Honest prior / expected outcome (declared before running)
**Prior: LOW–MODEST** (revised down from the scout's MODEST–FAIR by the recon: the interaction is already
partly banked in `text_feats`, making this a C3-nontarget-shaped redundancy question that the analogous C3
probe answered *no*; and the cheap reordering realization is more generic than the cited conditioning-pathway
prior art). Falsifiable: if the transcript-conditioned visual key does not beat `concat(img_feats,
text_feats)` by a paired margin projecting to +3 on ≥1 dataset's oracle arm AND a raw +0.05/+0.05 on HateMM,
the interaction is redundant with the two banked marginals and W2-A is dead. Most likely *informative*
outcomes: (b) HateMM-only (mechanism real, gap open), (d) both fail (cell closed), or a clean C3-style null.
MHC-EN — 100% transcript coverage — is the honest fair-coin for whether grounding converts on the binding
dataset.

## 14. Connections
- Candidate spec: `research-wiki/ROUND3_CANDIDATES_WAVE2_2026-07-15.md` §W2-A.
- Forensic recon (READ FIRST — transcripts reality, causal-masking finding, key-pooling decision):
  `refine-logs/W2A_FORENSIC_RECON.md`.
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
