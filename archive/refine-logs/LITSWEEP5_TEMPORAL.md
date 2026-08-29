# LITSWEEP-5 — TEMPORAL-STRUCTURE axis (lit round 5 of 5, agent S1)

**Date:** 2026-07-25. **Mode:** ZERO GPU / SLURM / Modal. WebSearch/WebFetch + in-repo code/cache/doc
reading only. `autoresearch/…/state/` untouched. Repo HEAD at recon = `53957ad`.

**Charge.** FINAL sweep of the *temporal-structure* lens — the axis the task framed as "nobody ever
varied." **Finding after in-repo verification: the framing is false. Temporal structure is one of the
MOST heavily attacked axes in the campaign** — closed at four independent mechanistic levels with $0
conditional-info gates, a cloud order-kernel probe, a killed retrieval-object family, and an arithmetic
selection-lock. This note (a) maps that closure with citations, (b) isolates the genuinely virgin
sub-cells and prices them, (c) delivers the fresh-2026 delta (TANDEM 2601.11178 + MultiHateLoc + training-
free localisation), and (d) banks the paper-value. **Verdict: temporal axis CLOSED; no in-box candidate
clears the +0.030-on-≥2-datasets bar; one user-gated door-closer (motion/flow $0 gate) remains.**

---

## VERDICT LINE

> **CLOSE THE AXIS.** Every temporal *operator over the frozen representation* (order, arc, learned pool,
> segment granularity, frame count/density) is measured-dead. The only temporal moves that add genuinely
> new bits — raw motion/optical-flow (new modality) and mRoPE absolute-time/fps (new input signal) — are
> (i) not $0 (raw-video re-extraction, Modal-blocked), (ii) carry LOW priors with *prior evidence against*
> already banked (W2-C dynamics-magnitude class-invariance; CTF frame-group tensor = 0 conditional info),
> and (iii) reduce to the same conditional-redundancy / selection-lock walls that closed audio (F64),
> prosody (F41), and ISR (F66). The in-domain temporal literature (yang2025 2508.04900) *confirms hate is
> bursty* and that burstiness converts to accuracy **only via GOLD-timestamp trimming** — i.e. exactly the
> per-item selection our law-III bans. Temporal burstiness for classification is closed by arithmetic, not
> by lack of trying.

---

## 0. WHAT IS TEMPORAL IN OUR PIPELINE (verified in `src/utils/generate_VideoMLLM_embedding_HF.py`)

| dimension | current value | code |
|---|---|---|
| Frame **count** | `num_frames = 8` | `:131-136` |
| Frame **selection** | uniform `np.linspace(0, N-1, 8)` rounded — **no content signal, no RNG** | `:187-193` |
| Frames → model | `{"type": "video", "video": frames}` (list of 8 PIL) — **video mode** | `:282-292` |
| **fps / timestamps** | **NONE passed** to `processor(text=…, videos=[frames])` | `:304-316` |
| Temporal readout | mean-pool over vision+instruction span (img) / last-token (text) → L2-norm | `:331-364` |
| Banked object | one 3584-d pooled vector per stream; **per-frame tokens NOT stored** | `:477-484` |

**Two load-bearing extraction facts (answering the task's mRoPE question directly):**
1. Frames enter via Qwen2.5-VL **video mode**, so the model **does receive frame ORDER** — mRoPE assigns
   monotonic temporal position ids across the temporal-patch groups (`temporal_patch_size=2` → 8 frames =
   4 groups; confirmed by the S2S temporal control σ=[2,0,3,1], `W2C_FORENSIC_RECON.md §C`, which proved
   the group order axis is real and not scrambled by extraction).
2. **No `fps`/timestamp is ever passed** (`:311-316`). So the model is **PACING-BLIND**: it encodes the
   *sequence* (group 0 < group 1 < …) but not *absolute duration/tempo*. A 10-s clip and a 3-min clip look
   temporally identical up to the group ordering.
3. **Per-frame structure IS recoverable at $0 for the T=4 groups** — the frameset caches
   `data/CLIP_Embedding/{HateMM,MHC}/frameset_qwen7b_8f/{train,dev_seen}_frameset.pt` bank `g` `[N,T=4,3584]`
   per video (test_seen banked but audit-only, never opened). **CTF (F39) already ran the decisive $0
   conditional-info gate over exactly these** and killed the axis (§1).

---

## 1. THE TEMPORAL AXIS IS NOT VIRGIN — FOUR-LEVEL CLOSURE MAP

| level | mechanism tested | result | finding / record |
|---|---|---|---|
| **Order / sequence** | order-constrained kernel (soft-DTW + signed transition-set) over frame set, vs order-BLIND MeanMaxSim, with **within-video order-shuffle null** | CLIP-K4 **DEAD**: DTW obs Δacc **+0.0059 = null-95th exactly** (a random reorder buys the same edge; null-mean ≈ 0); TRANSITION strictly worse (−0.054 acc); bootstrap 5th<0 both datasets | W2-C (`W2C_ORDER_PRECHECK_RECORD.md`, cloud-triage); Qwen T=8 arm never ran — extinguished by S2S kill |
| **Retrieval-object over frame groups** | set-to-set / late-interaction (MeanMaxSim/Chamfer/ASYM) over Qwen frame-group tokens | **KILL both datasets** (HateMM +0.0035 fails +0.05 bar; MHC −0.0397); "pooling effectively lossless on these representations" | S2S **F38** (`S2S_PROBE_VERDICT_REVIEW.md`) |
| **Causal-prefix tensor / arc** | flat `[g_1..g_T]` (14336-d) + arc `Δ=g_T−g_1`, **supervised** conditional-info over `Z_best` (8960-d), label-oracle calibrated | **KILL $0**: flat **+0.0000** HateMM / **−0.0029** MHC; arc **−0.0049 / −0.0010**; calibration VALID (accZA=1.0000); higher-k monotone-negative = pure redundancy | CTF **F39** (`CTF_GATE_RECORD.md`) — operator-agnostic (kills fixed-mean, arc, AND learned attention-pool 1b) |
| **Segment granularity / burstiness** | independently re-encoded per-segment features → uniform (non-selecting) kNN-vote-mean; oracle decomposition | **NO-GO $0**: symmetric slice **+0.0012 / +0.0032** (flat); oracle headroom = symmetric(legal, ~0) **+** selection(banned) **91–98%** → convertible headroom is *formally disjoint* from the legal operator | ISR **F66** (`ISR_PREGATE_RECORD.md`) |
| Frame **count / density** | 8→16 frames, pooled dual-stream, paired 3-seed | **KILL both protocols**: val-sel −0.0077, final +0.0015; "8f already saturates" | frame16 **F67** |
| Frame **selection policy** | uniform vs content-aware / scene-change / motion / diversity keyframes @8f | priced **LOW/~0**: short video, 8f, query-agnostic, only 7–22% videos have near-dup slack; S2S near-dup-excluded arm ≈ null | LITSWEEP2 §3.2 (`LITSWEEP2_INPUT_FIDELITY.md`) |

**Unifying structural facts (why the level-1..4 kills are not coincidences):**
- **F35 cumulative-causal-prefix:** the T=4 groups are cumulative causal prefixes, not frame-local states —
  later groups have already attended over earlier ones, so "frame-local order semantics" is *unavailable*
  as a distinct coordinate. (Re-confirmed by F72: naked bidirectional-mask removal *craters* −10…−14 pts →
  the pipeline exploits the causal-prefix semantics, not a recoverable temporal-dynamics signal.)
- **F66 arithmetic selection-lock:** any *non-selecting symmetric* temporal operator converts ≈0; the only
  convertible temporal headroom is reachable *exclusively* by law-III-banned per-item selection. This makes
  "use temporal burstiness for classification" a closed arithmetic question, not an open engineering one.

**Ban-scope note (`directions_tried.json`):** the temporal-order-kernel revival is explicitly gated — "any
revival must first defeat F35 (groups are cumulative prefixes, not frame-local states) with a *different
representation object*." No operator over the banked frozen features can do this; only a **new information
source** (§2) is admissible, and it inherits the F49 alignment bar + F64/F41 conditional-redundancy screen.

---

## 2. GENUINELY VIRGIN SUB-CELLS — RANKED SHORTLIST (all LOW; none clears the bar)

### S1 — Raw motion / optical-flow as a NEW modality  ·  prior ≥+1 LOW (~4–6%), ≥+3 negligible
**Why virgin:** distinct from everything above — a temporal derivative of **pixels** (RAFT/GMFlow dense
flow, or classic TV-L1 flow-magnitude), not of appearance *embeddings*. The transition-set arm (W2-C) and
the arc Δ (CTF) are appearance-space differences of the *already-pooled Qwen vectors*; raw flow adds bits
the frozen forward never sees. **No ban binds it** (F67=count, S2S=operator, CTF=frozen-group tensor).

- **(a) Honest transplant:** dense flow per video → a pooled motion descriptor (flow-magnitude histogram,
  or a small frozen motion encoder) → auxiliary channel, screened by a CTF/APX-style $0 G0-cond
  conditional-info gate over `Z_best`; only if it clears +0.040 does it earn a fused-key GPU cell.
- **(b) $0 reuse:** **NONE** — needs raw-video re-extraction (**local SLURM only; Modal hard-blocked**,
  raw video never leaves the machine). But a cheap *proxy is already negative*: W2-C §C dynamics forensics
  on banked CLIP-K4 show dynamics-**magnitude** is near class-invariant (HateMM 0.301 vs 0.416; MHC 0.449
  vs 0.470) and **the hateful class is MORE static** (within-cos 0.899>0.874) — the reveal/escalation
  premise is *inverted* on the anchor.
- **(c) Strongest failure reason:** the **conditional-redundancy law** — audio-Whisper (F64) and eGeMAPS
  prosody (F41) both measured **exactly zero** conditional info over `Z_best`; motion faces the identical
  screen. Compounded by the F49 alignment bar (>0.663 for any new auxiliary source), W2-C class-invariance,
  the semantic-not-motion nature of hate (violence-detection's ~90% flow accuracy is a *different,
  motion-heavy task*, verified §3), and mean-pool attenuation (F65/F67).
- **(d) Prior vs house bar:** LOW; below +0.030-on-2-datasets by a wide margin.
- **Minimal-decisive cell:** HateMM only (the one image-converting dataset) — extract flow-magnitude
  descriptor (~0.3 GPU-h local) → $0 CTF-style G0-cond gate over `Z_best`. **Kill-switch:** point <+0.040
  OR CI-lower ≤0 → DEAD (matches the audio/prosody nulls, the expected outcome).

### S2 — mRoPE absolute-time / fps / duration injection  ·  prior ~1–2%
**Why virgin:** verified un-varied — extraction passes NO fps (§0), so the model is pacing-blind. Injecting
real per-video fps/duration engages Qwen2.5-VL's absolute-time encoding (a genuinely new input signal, not
an operator swap).
- **$0 reuse:** none (re-extraction). **Strongest failure reason:** CTF (F39) already measured the frame-
  **group** tensor the model *does* encode at +0.0000/−0.0029 conditional info — the finer temporal ids
  carry no convertible signal, so a *coarser* pacing scalar is even less likely; and F35 says the exploited
  structure is causal-prefix semantics, not tempo. Within-representation knob, no new external bits beyond a
  single duration scalar (which is a decision-side scalar → D1-dead by the diagnosis frame).
- **Minimal cell:** re-extract HateMM with real fps → $0 G0-cond gate. **Kill-switch:** <+0.040 (expected).

### S3 — Test-time multi-view frame-augmentation (TTA)  ·  OFF-AXIS, noted not claimed
Variance-reduction, **not a temporal-structure claim** (it averages independent stochastic reads; makes no
order/dynamics hypothesis, so it escapes Law-II precisely *because* it is not temporal). Already enumerated
as REDTEAM external-family **#5** (~10–15%, per-item legal, but **grazes the cross-seed-ensemble ban → needs
a user micro-ruling**). Deferred to the redteam/user; it is not my axis and I do not price it here.

---

## 3. FRESH-2026 DELTA (verified citations)

- **TANDEM: Temporal-Aware Neural Detection for Multimodal Hate Speech** (Koushik, Treharne, Kanojia;
  arXiv **2601.11178v2**, May 2026). Temporal operator = **scene-change keyframe selection** (= our
  LITSWEEP2 §3.2, priced LOW) + RL cross-modal grounding (SFT→GRPO/GSPO) that uses **optional GOLD
  temporal-segment + target-identity annotations**. **Trained classifier + RL**, not retrieval/kNN;
  backbones Qwen2.5-VL-7B + Qwen2-Audio-7B (LoRA). **HateMM acc 0.78 / macro-F1 0.79; MHC acc 0.67 /
  macro-F1 0.38.** → **NOT borrowable** (gold annotations + target-ID + RL-SFT = multiple constraint-box
  violations) and its **HateMM numbers sit BELOW our 0.8775/0.8791** — it *corroborates closure* (only adds
  temporal levers we priced dead or banned) and re-confirms our HateMM number stands.
- **MultiHateLoc** (Sun et al., arXiv **2512.10408**, WWW-2026): temporal **localization** with modality-
  aware temporal encoders + MIL; HateMM frame-mAP 0.645/AUC 0.799, MHC 0.445/0.750. Uses **frame-level GT**
  (gold) → out-of-box; **NO retrieval/kNN**. Localization, not the video-level classification we optimize.
- **Towards Training-free Multimodal Hate Localisation with LLMs** (Sun et al., arXiv **2602.09637**,
  Feb 2026): LLM zero-shot **localization**, not classification — noted, off our accuracy objective.
- **General temporal-pooling literature** corroborates the dichotomy that closes us: mean-pool "drowns
  critical moments in surrounding static content," attention-pool "collapses onto one or two pairs and
  discards global context." The attention-pool that *works* is exactly the per-item soft-selection our
  law-III / F66 bans. Consistent with F68's finding that SOTA video-embedding works use **no** temporal
  operator.

---

## 4. PAPER-VALUE (count = 3)

- **PV-T1 — Temporal-structure exhaustion as a NEGATIVE RESULT.** Four-level closure (order-kernel W2-C +
  causal-prefix tensor CTF/F39 + retrieval-object S2S/F38 + segment ISR/F66; plus frame-count F67), unified
  by F35 (cumulative-causal prefix) and F66's symmetric-legal-vs-selection-banned decomposition.
  Externally corroborated (mean-pool-drowns / attention-collapses; SOTA video-embedding uses no temporal
  operator). Analysis-chapter statement: *"why temporal structure does not convert in short hateful video
  under a pooled-MLLM representation."*
- **PV-T2 — Burstiness ↔ selection equivalence, in-domain grounded.** yang2025 (2508.04900) shows hateful
  videos contain long non-hate stretches and that trimming to **GOLD** hateful segments sharply improves
  classification — burstiness converts to accuracy **only through timestamp selection**, which is gold-
  supervised (banned in our deployed path). This is the *in-domain confirmation of F66's arithmetic lock*
  and the clean bridge from the P6/P10-b localization assets (wv-AUC 0.5755) to the classification law-I.
- **PV-T3 — Extraction methods/limitation note.** Qwen2.5-VL receives frame ORDER (monotonic mRoPE ids over
  T=4 groups) but is PACING-BLIND (no fps passed); with CTF's measured-zero on frame-group structure, this
  pins that the pipeline exploits cumulative-causal semantics, not temporal dynamics (consistent with the
  F72 bidir crater). Methods + limitations material.

---

## 5. USER-RULING FLAGS

1. **Motion/optical-flow door-closer** (S1): ~0.3 GPU-h local flow extraction + $0 G0-cond gate on HateMM —
   the only way to convert this axis from *priced-dead* to *measured-null*. User-gated (expected KILL).
2. **TTA multi-view** (S3): needs the cross-seed-ensemble micro-ruling already flagged in REDTEAM #5; also
   off the temporal-structure axis.
3. **Gold-timestamp trimming** (yang2025 mechanism): the *one demonstrated in-domain temporal lever*, and it
   is **banned** (no gold annotations in deployed path). If the user ever relaxes the gold-annotation ban
   for a localization-trained variant, this is the only temporal door with demonstrated conversion — but it
   exits the constraint box entirely.

---

## 6. PROVENANCE
- Code: `src/utils/generate_VideoMLLM_embedding_HF.py` (:131-193 sampler, :282-316 video-mode + no-fps,
  :331-364 pooled readout, :477-484 banked object).
- Kills read in full: `refine-logs/W2C_FORENSIC_RECON.md`, `refine-logs/W2C_ORDER_PRECHECK_RECORD.md`,
  `refine-logs/CTF_GATE_RECORD.md` (+ `CTF_G0COND_GATE_OUT.json`), `refine-logs/ISR_PREGATE_RECORD.md`,
  `refine-logs/LITSWEEP2_INPUT_FIDELITY.md`, `refine-logs/REDTEAM_EXTERNAL_FAMILIES.md`; findings
  F35/F38/F39/F41/F44/F49/F64/F66/F67/F68/F72 (`state/findings.jsonl`), `state/directions_tried.json`.
- Caches confirmed on disk: `data/CLIP_Embedding/{HateMM,MHC}/frameset_qwen7b_8f/` (`g` `[N,T=4,3584]`,
  train+dev; test_seen audit-only).
- Verified external: arXiv 2601.11178 (TANDEM), 2512.10408 (MultiHateLoc), 2508.04900 (yang2025),
  2602.09637 (training-free localisation); general temporal-pooling literature.
- Compute posture: ZERO GPU / SLURM / Modal; WebSearch/WebFetch + read-only in-repo inspection. Repo HEAD
  `53957ad`. One local commit, no push.
