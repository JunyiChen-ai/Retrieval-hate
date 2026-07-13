---
type: experiment
node_id: exp:exp-sav-f0
title: "SAV (C2): sparse attention-head feature mining from frozen Qwen2.5-VL for RGCL (tests the MHC-EN mean-pooling dilution hypothesis)"
idea_id: ""
verdict: DRAFT-REV2-AWAITING-DELTA-CHECK — Rev-2 applied re-review residuals R1–R3 + Rec-1..3; re-review pre-authorized APPROVED once R1–R3 land (delta-check only); no gate executed, nothing committed
confidence: n/a (design)
date: "2026-07-13"
status: >
  DRAFT-REV2-AWAITING-DELTA-CHECK. Recon + design + Rev-1 (M1–M5) + Rev-2 (re-review
  residuals R1–R3 + Rec-1..3, applied 2026-07-13; see §8 Revision history). The Rev-1
  re-review (refine-logs/SAV_F0_PREREG_REVIEW.md, appended section) returned REVISE(minor)
  with pre-authorization: APPROVED once R1–R3 land as written — a delta-check of the edited
  text suffices. No job submitted, no code changed, nothing committed. A-line
  (lb_scgp_global) is PAUSED (refine-logs/lb_scgp_global/A_LINE_PAUSE_DECISION.md);
  SAV is the LEAD experiment and the GPU is free.
hardware: >
  planned: 1x A100-SXM4-80GB (SLURM node foscsmlprd01). Per-head attention re-extraction
  is a single frozen forward (~same order as the existing frozen-Qwen extraction, tens of
  min/dataset); head-selection + linear probe are CPU/seconds. No --time. Cheapest C-line pilot.
provenance: >
  SAV method/numbers read from arXiv 2412.00142v3 (HTML) via WebFetch 2026-07-13.
  "Cached hidden states are insufficient" verified live against
  src/utils/generate_VideoMLLM_embedding_HF.py:18-52 (only last-layer mean-pooled 3584-d
  vectors are cached; no per-head, no per-layer). Encoder-swap MHC-EN failure cites
  exp-encoder-3seed.md file:line.
added: 2026-07-13T00:00:00Z
tags: ["hateful-video", "MLLM", "sparse-attention-vectors", "SAV", "feature-mining", "frozen-encoder", "RGCL", "pre-registered", "DRAFT-REV2-AWAITING-DELTA-CHECK", "MHC-EN", "dilution-hypothesis", "C-line", "SAV"]
---

# SAV — Sparse Attention-head feature mining from frozen Qwen2.5-VL (C2)

> **F-G1 KILL — CONFIRMED BY INDEPENDENT VERDICT REVIEW 2026-07-14 (refine-logs/SAV_F1_VERDICT_REVIEW.md). 18th negative route.** Machine verdict KILL (job 13099) upheld under corrected machinery (λ-grid widened; the deployed grid's edge-saturation had INFLATED SAV by crushing the higher-dim pooled baseline): MHC cell collapses to CI-including-0, HateMM no-harm violation confirmed and stronger, U-1 = pooled-equivalent null. Dilution hypothesis FALSIFIED — MHC-EN is data/label-limited. Single submissions 13058+13099 consumed; no F-G2.

> **PRE-REGISTRATION STATUS: DRAFT-REV2-AWAITING-DELTA-CHECK.** Pre-registers hypotheses,
> mechanism, gates, kill numbers, seeds, and protocol for the cheapest C-line candidate (C2).
> **Rev-1 (2026-07-13) applied the mandatory revisions M1–M5; Rev-2 (2026-07-13) applied the
> re-review residuals R1–R3 + Rec-1..3 from `refine-logs/SAV_F0_PREREG_REVIEW.md` (see §8).**
> The re-review pre-authorized **APPROVED once R1–R3 land as written (delta-check only)**.
> **No experiment is run, no code is changed, nothing is committed.** Delta-check + a
> user-visible report before any `sbatch`. **A-line (lb_scgp_global) is PAUSED
> (`refine-logs/lb_scgp_global/A_LINE_PAUSE_DECISION.md`) — SAV is the LEAD experiment; the GPU is free.**

**One-line mechanism.** The encoder-swap uses a **mean-pooled last-layer** Qwen hidden state
(3584-d) as the RGCL feature. SAV (Sparse Attention Vectors, ICCV 2025, arXiv 2412.00142) shows that
for discriminative/safety tasks a **tiny set (<5%) of attention heads** carries the signal, and
mean-pooling **dilutes** it. C2 mines the discriminative heads from a **frozen** Qwen2.5-VL forward
(head-selection on **train labels only**), and feeds the **selected-head feature vectors** into the
RGCL alignment-fusion head + kNN instead of the pooled hidden state. **This is a representation-level
change (the project's only proven +3 lever family, D2), and it supplies a falsifiable mechanism for
why the encoder swap FAILED on MHC-EN** while passing HateMM.

---

## 1. Hypothesis

**H (sparse-head discriminability).** The encoder-swap's frozen-Qwen feature is a mean over the
whole visual+instruction (or response) span of the **last layer** (`generate_VideoMLLM_embedding_HF.py:18-52`),
which averages a discriminative hate/benign subspace together with topic/community/format signal. On
**HateMM** the hate signal survives pooling (encoder swap +5.3 acc, `exp-encoder-3seed.md:184-198`);
on **MHC-EN** it is diluted below the pooling floor (encoder swap **fails both protocols**, mean Δacc
+0.019 val / +0.006 final, `exp-encoder-3seed.md:207,219,231`). Replacing the pooled feature with the
**SAV-selected sparse-head feature** should recover a discriminative subspace on MHC-EN and lift the
RGCL head **≥ +0.015 val acc over the pooled-feature baseline** — turning the encoder result from
"HateMM only" into the **≥ 2-dataset** claim the campaign goal requires.

**H0 (the null this line most likely dies on).** The pooled last-layer feature already contains the
hate subspace (a linear probe over pooled features on MHC-EN val ≈ SAV-head probe): MHC-EN's ceiling
is genuinely ~0.78-0.80 for *any* frozen Qwen read-out (both encoder arms sit in the 0.77-0.80 band,
`exp-encoder-3seed.md:240-241`), so no head-selection recovers a gain — the dilution hypothesis is
false and MHC-EN is data-limited, not pooling-limited. If the F-G1 probe does not clear the revised
G0-cond bar on val (§4 F-G1), C2 dies at ~1-3 GPU-hr with the dilution hypothesis cleanly falsified.

**Causal-claim scope (M3).** The SAV feature differs from the pooled baseline on **three** axes at once
— sparsity/granularity, **token position** (final-token vs span-mean), and **layer depth** (all-layer vs
last-layer only). A SAV win is therefore not automatically attributable to "sparse heads fix
mean-pooling dilution." The specific **"mean-pooling dilution"** attribution is licensed **only if** the
SAV feature beats the F-G1 isolating control **C-pos** (final-token full last-layer hidden state); if
C-pos already closes the MHC-EN gap the effect is a token-position artefact, and the causal claim is
re-scoped to "**final-token re-reading of the frozen encoder recovers discriminability**" (the goal —
does SAV-feature RGCL beat pooled-feature RGCL by +0.030 on test — is unaffected; this only governs the
paper-narrative mechanism claim). See F-G1 controls (C-pos / C-sparse) below.

---

## 2. Mechanism — SAV pinned to the exact procedure (arXiv 2412.00142v3, read 2026-07-13)

**SAV as published** (WebFetch of the HTML full text):
1. **Per-head feature.** From a **frozen** multimodal LLM, for each (layer l, head m) take the
   attention vector **h^{l,m}(x_i^T) at the final token position x_i^T** — one vector per (layer,head)
   per example. (NOT the pooled last-layer hidden state.)
2. **Head selection.** For each class c, compute the **centroid** μ_c^{l,m} of that head's vectors over
   the few-shot set; score each head by its **nearest-centroid (cosine) classification accuracy** on the
   labeled set; keep the **top-k heads** ℋ_SAV. SAV uses **≈20 heads (<5% of all heads)**.
3. **Inference.** **Majority vote** across the selected heads, each head a local nearest-centroid
   (cosine-to-class-means) classifier. Few-shot scale ≈ **20 examples/class**.
4. **Reported gains** (Table 1 — external motivation only, NOT a floor). VLGuard (LLaVA-OneVision)
   **≈+63%**. **M4(a) provenance flag:** the draft's earlier "+62.9% (31.0→94.3)" is internally
   **inconsistent** (94.3 − 31.0 = **+63.3**, not +62.9 — a ~0.4 discrepancy in an endpoint or the
   delta); these decimals are **PENDING PDF re-verification** and must NOT be transcribed into the paper
   until re-read from the arXiv 2412.00142 PDF (same discipline as the RA-HMD-number caveat). Other
   reported gains: MHaluBench +46.1%, vs **LoRA +7-20%**, BLINK +6.8%, NaturalBench +8.3-9.0%.
   **M4(b) model-mismatch caveat:** SAV's *results* are on **LLaVA-OneVision-7B and Qwen2-VL-7B** — the
   **same family as our encoder, but NOT the same model** (ours is **Qwen2.5-VL-7B**). The per-head
   extraction *mechanics* transfer; SAV's head-selection *efficacy on Qwen2.5-VL* is an **assumption
   F-G1 is testing, not a given result.**

**C2 adaptation (representation-level, not a decision-side bypass).** We keep RGCL's trained
alignment-fusion head + triplet-contrastive + top-20 kNN memory (the project's decision mechanism),
and only **swap the input feature**: pooled last-layer hidden state → **the concatenation (or stacked
per-head set) of the SAV-selected heads' final-token attention vectors**. Head selection uses **TRAIN
labels only** (nearest-centroid accuracy over the train memory — standard supervised training-data use,
NOT a gold-annotation channel; identical status to any classifier fitting on train labels). SAV's own
majority-vote classifier is used only inside F-G1 as a probe; the deployed decision stays the RGCL kNN.

**Non-isomorphism to the 14 closed routes + to C1.** All 14 closed MLLM routes injected a low-bandwidth
MLLM *output* into the decision side (REFLECTION §1). SAV changes **which coordinates of the frozen
encoder's own representation** feed the head — a pure representation-level edit, like the encoder swap.
Vs C1 (which LoRA-adapts the manifold), C2 leaves the backbone frozen and instead **re-reads** it
sparsely — orthogonal and far cheaper; the two can compose later (SAV heads of a LoRA-adapted backbone).

---

## 3. Why the cached features are INSUFFICIENT (F-G0 pre-verification, done)

**Verified live 2026-07-13:** the encoder-3seed extraction (`generate_VideoMLLM_embedding_HF.py:18-52`)
caches **only two 3584-d vectors per video** — `img_feats` = **mean of the LAST-layer hidden states**
over the visual+instruction span (L2-normed), and `text_feats` = mean of the LAST-layer hidden states
over the response-tail span. **No per-head vectors, no per-layer hidden states, no attention outputs are
stored.** SAV requires the **per-(layer,head) attention vector at the final token** — a strictly
different tensor at a different granularity. **Therefore SAV needs a fresh frozen forward with
per-head extraction** via a **forward hook on `self_attn.o_proj` input** (which captures the
already-computed value-weighted per-head output at the final token; NOT `output_attentions=True`, which
returns the softmax weight matrices — the wrong object and a large tensor over thousands of visual
tokens), producing L×H head vectors/video. This is the concrete work item that F-G0 confirms and F-G1
executes. **(L, H, head_dim) verified live 2026-07-13** against the local `config.json`:
`num_hidden_layers=28`, `num_attention_heads=28`, `hidden_size=3584` ⇒ head_dim = 3584/28 = **128**;
`num_key_value_heads=4` (GQA — does not affect the count, since the per-query-head **output** is
head_dim=128 and 28 concatenate to 3584 before o_proj). So **28×28 = 784 query-head positions**, and
SAV's <5% ≈ **20-39 heads**, matching SAV's "~20 heads". **We commit to extracting ALL 784 heads (M5a):**
the full per-head final-token cache is ~784×128×4 B ≈ **401 KB/video** ⇒ ~1,000 videos × 3 datasets ≈
**~1.2 GB total (fp32; ~0.6 GB fp16)** — trivial. No layer-subset hedge (a subset knob would be a hidden
degree of freedom in head selection).

---

## 4. Gate sequence (numeric kill numbers + cost + test-touch discipline)

**Control floor (the pooled-feature RGCL / encoder swap), with provenance:**
- **MHC-EN** frozen-Qwen (the dilution target): val-sel 3-seed mean **0.7805 acc / 0.7219 F1**,
  final-ep **0.7847 acc / 0.7425 F1** (`exp-encoder-3seed.md:164-170,257-259`). Encoder swap here
  FAILS both protocols (`:207,219,231`) — the gap C2 must open.
- **HateMM** frozen-Qwen (no-harm control): val-sel mean **0.8729 acc / 0.8648 F1**, final-ep
  **0.8682 / 0.8591** (`exp-encoder-3seed.md:154-159,251-253`).

### F-G0 — SAV procedure pinned + cached-state insufficiency + extraction feasibility + anti-drift guards (cost: 0 GPU + one bounded reproduction check)
- **Done in this draft:** SAV selection procedure pinned (§2, arXiv 2412.00142v3); cached hidden states
  confirmed insufficient — per-head re-extraction required (§3); (L, H, head_dim) = (28, 28, 128), GQA
  kv=4, 784 head-positions verified live (§3); storage bounded at ~1.2 GB — **all 784 heads extracted**
  (M5a; no layer-subset hedge).
- **Remaining F-G0 checks:** (a) per-head extraction mechanism pinned = **forward hook on
  `self_attn.o_proj` input** capturing the value-weighted per-head output at the final token (NOT
  `output_attentions`, §3) — kernel-agnostic (o_proj is a distinct Linear regardless of SDPA/flash-attn).
- **(b) Reproduction guard (M2a + R2 — anti-drift, MANDATORY before any SAV comparison is valid).**
  Two-tier, pre-declared (R2):
  - **PRIMARY (feature-level): per-video cosine** between the **fresh-forward pooled feature**
    (span-mean last-layer hidden state, L2-normed, recomputed by the fresh per-head forward pipeline)
    and the **cached** pooled feature (`{split}_Qwen2.5-VL-7B-Instruct_HF.pt`), for BOTH `img_feats`
    and `text_feats`, over every train+val video of the gated dataset. **Pass threshold (pre-declared):
    min per-video cosine ≥ 0.999.** This is the deciding statistic: it is continuous, per-video, and
    not quantized by the tiny val set.
  - **SECONDARY (confirmatory, probe-level):** a probe/RGCL read-out over the fresh-forward pooled
    feature must land **within ±0.010 val acc** of the cached pooled-feature floor. **Quantization fact,
    stated up front (R2):** on the 80-sample MHC-EN val, 1 prediction flip = 0.0125 > 0.010, so ±0.010
    permits **zero flips** — i.e. the secondary check demands exact prediction reproduction and CAN trip
    on benign bf16/kernel nondeterminism. It is therefore read as **confirmatory only**: if the PRIMARY
    feature-level check passes and the secondary trips, the guard PASSES (record the discrepancy); if the
    PRIMARY fails, the guard FAILS regardless of the secondary. No post-hoc tolerance amendment is needed
    or permitted — this two-tier rule is the pre-declared tolerance.
  If the guard FAILS, the fresh forward has drifted (frame sampler / prompt / span / dtype) relative to
  the banked encoder-swap extraction, and **no SAV-vs-pooled comparison is admissible** — fix the drift
  first. This guard is what makes the three-axis F-G1 comparison (§4 F-G1, M3) attributable to the
  read-out change rather than a pipeline diff.
- **(c) Frame-source pin (M2b).** The fresh forward MUST read frames from the **same source as the cached
  pooled extraction**: decode the **symlinked mp4s** at `data/video/<dataset>/All/<id>.mp4` (verified: the
  per-file entries are symlinks into the raw stores, e.g. `/data/jehc223/HateMM/video/…`) via the **same
  decord→PyAV 8-frame sampler** used by `generate_VideoMLLM_embedding_HF.py` (§frame sampler, lines
  155-233). **Do NOT read from `data/lora_frames/`** (the pre-extracted frame directory used only by the
  LoRA-SFT pipeline — a *different* frame source). This is a direct hedge against the project's
  **symlink-topology / symlinked-mp4 decode-mismatch burn**; a frame-source swap alone would confound the
  result and would also trip the (b) reproduction guard.
- **(d) Deferred-import audit + independent code review (M2c).** The per-head hook / extraction is **new
  model-internals code** — exactly the class the project's deferred-import audits target. Before any GPU
  submit: run a **deferred-import audit** (no lazy/conditional import silently changing the forward path)
  AND route the hook code through **`codex-code-review`** (independent code review). Both are prerequisites
  to F-G1, not optional.
- **Kill:** if per-head attention output cannot be extracted from Qwen2.5-VL under our stack with a
  localized reviewed edit, OR the reproduction guard (b) FAILS its PRIMARY feature-level check
  (min per-video cosine < 0.999 against the cached pooled features — irreducible pipeline drift), OR
  the deferred-import/code-review audit (d) surfaces an unfixable forward-path change → line does not
  reach a training GPU (re-scope or drop). Deliverable: a feasibility + reproduction-guard +
  frame-source-pin note.

### F-G1 — matched-capacity MDL/codelength probe, multi-seed (≥5), CI-excludes-0, G0-cond-compliant (cost: ~2-3 GPU-hr; MHC-EN + HateMM + MHC-ZH val)
- **This is the deciding cheap gate and the direct test of the MHC-EN dilution hypothesis. Rev-1 (M1)
  rebuilds it to the project's own G0-cond bar** (`REFLECTION_mllm_integration_failures.md:37-43`). The
  previous **single-seed +0.015-val-acc** threshold is **RETIRED**: MHC-EN val = **80 samples** (verified,
  `data/gt/MHC/val.jsonl`), so 1 acc point = 1.25% and +0.015 ≈ **1.2 flipped examples** — inside the
  ±1-2pt noise floor the project itself established (the exact trap behind TARC's false-positive cell and
  the archive-as-key withdrawal). Prerequisite: F-G0(b) reproduction guard + F-G0(c) frame pin + F-G0(d)
  import/code audit must all pass first.

- **Extraction & selection (multi-seed, ≥5) — WHAT THE SEEDS VARY (R1a, pre-declared).** Frozen
  Qwen2.5-VL-7B per-head extraction (§3, all 784 heads, o_proj-input hook) on train + val of **all three
  evaluation sets**. The extraction is a **deterministic frozen forward and is run ONCE** (seeds do not
  and cannot vary it); likewise, nearest-centroid selection on the full train set would be deterministic.
  If nothing varied, 5 seeds would be 5 identical replicates, the cross-seed CI would degenerate to a
  point, and "CI excludes 0" would be trivially true — illusory rigor. Therefore the **≥5 seeds
  (seeds 0-4) inject genuine variation in BOTH of the following, pre-declared:**
  - **(i) head-selection subsample draws:** each seed draws an SAV-style few-shot selection subset
    (**20 examples/class, sampled without replacement from train**) and runs the nearest-centroid
    cosine-accuracy ranking (top-k swept over {10, 20, 40}) on that draw — so head-set stability across
    draws is itself measured and reported;
  - **(ii) probe train-split resampling:** each seed re-draws the probe's train subset (80% of train,
    stratified) for fitting the matched-capacity probe g; centroids/probe fit use the seed's subset only.
  The deciding statistic is thus a **cross-seed distribution over genuinely varying replicates**, not a
  single point estimate and not identical copies.

- **Primary metric = MDL / codelength (NOT accuracy), capacity-matched (M1c; REFLECTION §4(ii)); exact
  estimator PINNED (R1c).** For a probe g of the **same capacity as the deployed RGCL head**:
  - **MDL estimator (pre-declared): HOLDOUT LOG-LOSS** — the description length of the val labels given
    a probe fit on the (seed's) train subset only: **L(feature) = Σ_{i∈val} −log₂ p̂_g(y_i | x_i) bits**,
    with p̂ clipped to [10⁻⁶, 1−10⁻⁶]. NOT prequential/online coding (no ordering ambiguity, no online
    schedule to tune). The statistic is **ΔL = L(pooled) − L(SAV)** (bits the sparse-head feature saves),
    one ΔL per seed.
  - **bits→acc conversion rule (pre-declared): the Fano / inverse-binary-entropy projection** (REFLECTION
    §4(iii) "Fano/经验斜率" — we pick Fano and commit; the empirical-slope alternative is NOT used, since
    fitting a slope would itself be a post-hoc degree of freedom). Concretely: per-example mean codelength
    ℓ = L/n_val bits; projected ceiling accuracy **acc(ℓ) = 1 − h₂⁻¹(min(ℓ, 1))** where h₂⁻¹ is the lower
    inverse of the binary entropy function; **projected downstream gain = acc(ℓ_SAV) − acc(ℓ_pooled)**,
    computed per seed and bootstrapped per the R1b clustering rule below.
  Codelength (not accuracy) is the mandated primary read because accuracy on an 80-sample val is quantised
  at 1.25%/example, whereas codelength integrates the full predictive distribution and is the §4-recipe
  conditional-information metric.

- **Co-primary accuracy probe (matched capacity, multi-seed, bootstrap CI excludes 0) — justified
  substitute (M1c).** SAV is a **representation-level feature swap** (D2 family), NOT a low-bandwidth
  decision-side auxiliary signal, so a capacity-matched **multi-seed accuracy** probe is a legitimate
  co-primary here: the quantity of interest is directly the feature's linear/kNN separability, with no
  bandwidth bottleneck to hide behind. Report the cross-seed **Δacc** with a **bootstrap CI that must
  exclude 0**, computed by the **example-level clustered bootstrap pinned below (R1b)** — seed×example
  draws are correlated and are NEVER pooled as if independent; single-seed point estimates are never used
  to decide. This is the explicitly-justified accepted substitute the review permits for a feature-swap
  route.

- **Evaluation sets + combined decision rule (M1 — more than the 80-sample MHC-EN val alone).**
  - **MHC-EN val (80) — dilution target. Bootstrap clustered at the EXAMPLE level (R1b, pre-declared).**
    The earlier "≈400 val-example draws" phrasing was a **statistical error and is RETIRED**: the same 80
    val examples recur across seeds, so seed×example are correlated draws — pooling them as if independent
    would spuriously narrow the CI by ~√5. The pre-declared procedure: **resample the 80 MHC-EN val
    examples with replacement; within each bootstrap draw, average each drawn example's paired
    per-example delta (SAV − pooled) across the ≥5 seeds first, then average over the draw** (10,000
    draws). **Effective n stays 80** — seeds reduce per-example variance, they do not multiply n. The same
    clustered rule applies to the ΔL and projected-gain bootstraps and to HateMM (n=107) / MHC-ZH (n=78).
    Per-example resolution (1.25%) and effective n are stated so the bar sits provably **above the
    ±1-2pt floor**.
  - **HateMM val (107) — no-harm control.** SAV must NOT regress HateMM (ΔL ≥ 0 within CI; Δacc CI not
    below −0.010) — it must not trade the one banked encoder-swap win for the MHC-EN target.
  - **MHC-ZH val (78) — secondary / completeness.**
  - **Combined rule (pre-declared):** the line proceeds to F-G2 **only if**, on **MHC-EN**, **ΔL > 0 with
    cross-seed bootstrap CI excluding 0** AND the bits→acc projection below clears its bar, **AND HateMM
    does not regress**. A pass on HateMM/MHC-ZH alone does not carry the line (MHC-EN is the target).

- **Projected-gain-vs-noise-floor argument + EXPLICIT BAR (M1b + G0-cond).** Convert the codelength
  advantage ΔL to a **projected downstream accuracy gain** (bits→acc), stating the effective val n and
  per-example resolution. **The pre-declared bar: the projected gain must exceed `+0.030 acc + the ±1-2pt
  noise band`, with the cross-seed bootstrap CI on the projection excluding 0.** This is the project's
  G0-cond threshold verbatim (`REFLECTION:41`: "bits→acc 换算后投影增益必须 > +3 acc + 噪声带,多 seed
  bootstrap CI 排除 0"). **Noise-band number PINNED (Rev-2b main-loop ruling, 2026-07-13): 0.010, so
  the numeric bar = +0.040 projected acc.** Rationale: protocol consistency with the A-line G0-cond
  probe precedent (`refine-logs/lb_scgp_global/M1_G0COND_PROBE_RECORD.md` used +0.030 + 0.01 = +0.040);
  a per-gate drifting bar invites protocol-inconsistency criticism. Implemented as
  `NOISE_BAND_ACC = 0.010` / `PROJECTED_GAIN_BAR = 0.040` in `scripts/analysis/sav_f0_common.py`,
  echoed in the F-G1 verdict JSON `config`. A probe that clears the old +0.015 but **projects below
  +0.030+noise does NOT license the GPU spend.** (Oracle sanity, REFLECTION §4(iii): if even a
  gold-label-selected head set projects < +0.030, the sparse-head family is dead — run the
  oracle-selection projection as the cheapest possible pre-check; gold used for probing only, compliant.)

- **Probe-stream scope PINNED: img-stream per-head only (Rev-2b main-loop ruling, 2026-07-13).**
  F-G1's per-head extraction and all probe arms operate on the **img forward** (visual+instruction
  "prefix" span) — the literal mean-pooling-dilution target of H, and the only stream where the C-pos
  position control is non-degenerate (the text stream's pooled read-out is already a near-final-token
  response tail). The **text stream remains pooled-only** (forward-passed for the F-G0(b) reproduction
  guard, which checks BOTH cached streams; no text per-head cache). **Text-stream / concat per-head
  extraction is DEFERRED as an F-G2-stage option only if SAV wins F-G1** — pre-declared here so it
  cannot become a post-hoc degree of freedom.

- **Isolating controls for the 3-axis confound (M3).** SAV feature vs pooled baseline differ on
  sparsity/granularity **and** token position (final-token vs span-mean) **and** layer depth (all-layer vs
  last-layer). To attribute any gain to sparse-head re-reading (not position/layer), F-G1 adds two
  matched-capacity control probes:
  - **(C-pos) final-token, full last-layer hidden state** — isolates the **token-position** axis at pooled
    granularity. **If C-pos already closes the MHC-EN gap, the effect is a position artefact, not
    sparse-head dilution recovery** → re-scope the causal claim per §1 (M3 scope note).
  - **(C-sparse) the selected heads' output vectors, span-mean pooled over TOKEN POSITIONS (Rec-2:
    pooling is over the token axis — the same span as the cached extraction — NOT an average across
    heads; the per-head identity is kept and the pooled per-head vectors are concatenated)** — isolates
    **sparsity** from token position.
  The "mean-pooling dilution" attribution (§1/§6) is licensed **only if the SAV feature beats C-pos.**

- **Cheap upper-bound references (Recommended, review §7) + U-1 as the pre-declared attribution
  tie-breaker (Rec-1).** Also probe **(U-1) the full 784-head concatenation** and **(U-2) the single best
  head** as head-information ceilings, so a kill reads "no head subset carries information the pooled
  feature lacked" (U-1 ≈ pooled ⇒ nothing to mine) rather than only "top-k-by-nearest-centroid didn't."
  SAV's own majority-vote read-out is retained as a secondary reference. **Tie-breaker (pre-declared):
  if SAV beats C-pos, attribution between "sparse heads" and "multi-layer pre-o_proj head-space read-out"
  is still open (C-pos isolates position only; layer depth remains entangled with sparsity). The U-1
  reading breaks the tie: U-1 ≈ SAV ⇒ the gain is the head-space/multi-layer read-out, NOT sparsity per
  se (report the mechanism accordingly); SAV > U-1 ⇒ sparsity itself contributes (selection removes noise
  the full concatenation carries).** **U-1 capacity-matching / regularization (pre-declared):** U-1's probe
  input is **784×128 = 100,352-d** (vs 2,560-d for top-20), so an unregularized probe would overfit its
  way into an illusory ceiling; the U-1 probe uses the SAME probe family g with **L2 regularization chosen
  on the probe's train subset only (5-fold CV within the seed's 80% train split, λ swept over
  {10⁻⁴..10²} log-spaced)** — never tuned on val.

- **Kill:** on MHC-EN, ΔL CI includes 0, **OR** the bits→acc projection does not clear +0.030+noise (CI
  excludes 0) at any swept k, **OR** the no-harm HateMM regresses → **dilution hypothesis FALSIFIED, kill
  the line.** MHC-EN is then data-limited, not pooling-limited (H0), and no RGCL integration is attempted.
  (SAV analogue of the conditional-information gate: at the capacity of the deployed head, the sparse heads
  carry no label information beyond the pooled feature.)

### F-G2 — RGCL integration, 3-seed paired, both protocols, VAL (cost: ~2-3 GPU-hr; only if F-G1 passes)
- **Pre-flight — carry-forward head set PINNED pre-F-G1 (Rev-2a execution note 1, delta-check note 1).**
  The head set that DEPLOYS to F-G2 (and thence F-G3) is the **deterministic full-train
  nearest-centroid selection** — i.e. run the canonical SAV head-selection procedure once over the
  ENTIRE train set (all train labels, no subsampling → deterministic), take the top-k for the k that
  F-G1 promotes. The five F-G1 seed draws (20/class subsamples) are used ONLY to power the cross-seed CI
  and to **report head-set consensus (intersection size / mean pairwise Jaccard) as a stability
  diagnostic** — they do NOT choose the deployed head set. This is pinned BEFORE any F-G1 result is seen
  (it does not affect F-G0/F-G1 and cannot be back-fit to results); it is the `oracle_order`/full-train
  selection already emitted by the F-G1 engine, carried forward unchanged.
- **Run:** feed the selected-head feature into the RGCL alignment-fusion head + triplet-contrastive + kNN
  (the deployed decision), MHC-EN (+ HateMM no-harm), **seeds 0/1/2 paired vs the pooled-feature RGCL
  floor**, both protocols, judged on **val**.
- **Pass (pre-declared; R3a adds the CI co-requirement).** On MHC-EN, under a stated protocol
  (val-judged to preserve the test touch), ALL of:
  (i) 3-seed mean paired **Δacc ≥ +0.015 AND ΔF1 ≥ +0.015** vs the pooled-feature floor;
  (ii) **sign ≥ 2/3 seeds**;
  (iii) **example-level paired bootstrap CI excludes 0** (same clustered rule as F-G1/R1b: resample the
  80 val examples, average each example's paired delta across the 3 seeds within a draw, 10,000 draws) —
  mean + sign alone can still be noise-carried on 80 samples.
  **The bar stays +0.015 (per the re-review's explicit ruling AGAINST raising it to +0.030):** F-G2's job
  is not to establish the effect (the rebuilt F-G1 does that at G0-cond strength) but to check the probe
  gain **survives the trained RGCL head** before spending the one test touch; +0.030 on an 80-sample val
  (2.4 examples) at 3 seeds would risk killing a true +0.030-test effect on val noise.
- **Kill (R3b — HateMM no-harm number stated IN the rule, not only in the run spec):** MHC-EN does not
  clear the pass rule (i)–(iii) under either protocol, **OR HateMM regresses: 3-seed mean paired Δacc
  below −0.010 vs the pooled-feature HateMM floor (same no-harm number as F-G1)** → kill (the probe gain
  did not survive the trained RGCL head — the recurring "clean probe, flat trained metric" pattern the
  campaign has hit ≥ 4× must be checked HERE, on val, before any test touch).

### F-G3 — single test touch, +0.030 bar (the ONE sanctioned read; only if F-G2 passes)
- **Run:** the F-G2-surviving config, **MHC-EN + HateMM**, seeds 0/1/2, SAV-feature RGCL vs pooled-feature
  RGCL, both protocols, judged by the **exact enc3seed decision rule** (`exp-encoder-3seed.md:73-85`).
- **Pass:** mean paired **Δacc ≥ +0.030 AND Δmacro-F1 ≥ +0.030 AND sign 3/3**, on **≥ 1 dataset**, stated
  protocol. If MHC-EN passes, the encoder story becomes **≥ 2 datasets** (HateMM already passes frozen).
- **Test-touch budget:** the **only** C2 test read (val used for all F-G0/1/2 selection). One serial
  sbatch; no further C2 test read after F-G3.

**Ceremony.** Each gate = one pre-registered serial sbatch, single-submit-per-lineage. Per-head
extraction / hook code (touches model internals) routes through `codex-code-review` before GPU submit.
**GPU scheduling (Rec-3, updated 2026-07-13): A-line (lb_scgp_global) is PAUSED at a zero-GPU G0-cond
kill (`refine-logs/lb_scgp_global/A_LINE_PAUSE_DECISION.md`) — SAV is the LEAD experiment and the GPU
is free; no queueing constraint remains.**

---

## 5. Hard rules upheld

- **No gold annotations:** head selection uses TRAIN binary labels only (standard supervised training
  data; NOT a gold-annotation channel). DEV labels for selection, TEST labels only at F-G3 metric.
- **No OCR channel** (user veto 2026-07-13). **No cross-seed ensembles.** **Local open weights only** —
  Qwen2.5-VL-7B is the only fully-downloaded VL checkpoint (16 GB, verified; 32B/72B/Qwen3-VL are NOT
  present locally), so C2 is a 7B pilot, which matches SAV's own 7B setting.
- **OCR-veto note (M5b, non-blocking; line cite corrected per Rec-3).** The frozen encoder's *img*
  instruction already reads "Describe the people, symbols, gestures, and **on-screen text** in this video"
  (`generate_VideoMLLM_embedding_HF.py:45-47` — 44 is the comment line). This is **pre-existing banked encoder-swap baseline
  behaviour** — part of the extraction contract SAV re-reads sparsely — and is **NOT a new OCR channel**,
  so it does **not** violate the OCR veto. C2 adds no OCR. Recorded here only so a later reviewer does not
  mistake the existing prompt phrasing for a fresh OCR channel.

---

## 6. Where this line most likely dies (honest prior)

**Most likely: F-G1, via H0.** Both encoder arms already sit in MHC-EN's 0.77-0.80 band
(`exp-encoder-3seed.md:240-241`), consistent with a data/label ceiling rather than a pooling artifact;
a matched-capacity probe over pooled features may already reach that ceiling, so SAV heads add nothing on
val and the dilution hypothesis is falsified at ~1-3 GPU-hr. **Second: F-G2**, the campaign's recurring
"probe passes, trained metric flat" failure (P3/P11/TARC) — the val probe gain does not survive the RGCL
head. **The upside that justifies the cheap pilot:** SAV is the single lowest-cost route that (a) is
representation-level (the only proven +3 family) and (b) supplies a *falsifiable causal account* of the
MHC-EN encoder failure. Even a clean F-G1 kill is paper-usable ("MHC-EN is data-limited, not
pooling-limited — sparse-head re-reading of the frozen encoder does not recover it") — **subject to the
M3 scope**: if the F-G1 control **C-pos** (final-token full last-layer hidden) already closes the gap, the
kill/pass narrative is re-scoped to *final-token re-reading* rather than *sparse-head dilution recovery*
(§1 causal-claim scope). An F-G1 pass is the cheapest path to the ≥ 2-dataset headline. **(Rev-2a execution note 3,
delta-check note 3: the earlier "run it before the more expensive C1" clause is stale — C1 is
`KILL_CONFIRMED` (`refine-logs/C1_KILL_REVIEW.md`); SAV/C2 is the standalone lead C-line pilot, no
longer sequenced against C1.)**

---

## 7. Status / next step

**DRAFT-REV2-AWAITING-DELTA-CHECK (Rev-2, 2026-07-13; see §8).** The Rev-1 re-review
(`refine-logs/SAV_F0_PREREG_REVIEW.md`, appended "Rev-1 RE-REVIEW" section) found all of M1–M5 applied
faithfully — no design-level objection remains — and returned **REVISE (minor, text-only statistics
pinning)** with **pre-authorization: APPROVED once R1–R3 land as written (a delta-check of the edited
text suffices; no further full re-review cycle).** Rev-2 has now applied R1 (F-G1 seed-variation
declaration, example-level clustered bootstrap, pinned holdout-log-loss MDL estimator + Fano bits→acc
rule), R2 (two-tier reproduction guard: feature-level per-video cosine ≥ 0.999 PRIMARY, ±0.010 val probe
SECONDARY/confirmatory, flip-quantization stated), R3 (F-G2 example-level paired bootstrap CI-excludes-0
co-requirement + explicit HateMM −0.010 no-harm kill number; bar stays +0.015 per the reviewer's ruling),
and Rec-1..3 (U-1 tie-breaker + 100,352-d probe regularization; C-sparse token-axis pooling clarified;
stale A-line queueing clauses updated — A-line PAUSED, SAV is the lead experiment, GPU free; OCR-note
cite fixed to :45-47). **Next step: the pre-authorized delta-check of R1–R3 + a user-visible report;
then F-G0 remaining checks (hook feasibility, reproduction guard, deferred-import audit +
codex-code-review) before the F-G1 sbatch.** No submission until the delta-check and the user-visible
report are done.

---

## 8. Revision history

### Rev-1 2026-07-13 applied SAV_F0_PREREG_REVIEW M1–M5

Fresh 0-context reviewer verdict on the DRAFT-UNREVIEWED pre-registration was **REVISE** — 5 mandatory
revisions before any `sbatch` (`refine-logs/SAV_F0_PREREG_REVIEW.md`). All five applied:

- **M1 — F-G1 under-powered / not G0-cond-compliant.** F-G1 rebuilt (§4) as a **multi-seed (≥5)
  matched-capacity probe**: primary metric = **MDL/codelength** (REFLECTION §4(ii)); co-primary =
  capacity-matched multi-seed **accuracy** probe with a **bootstrap CI excluding 0** (explicitly justified
  as a legitimate co-primary because SAV is a representation-level feature swap, not a low-bandwidth
  decision-side signal — the accepted substitute the review permits); added the **projected-gain-vs-
  noise-floor** argument with the **explicit bar = projected > +0.030 acc + noise band, CI-excludes-0**
  (project G0-cond threshold), plus the oracle-selection pre-check; deciding read now spans **MHC-EN val +
  HateMM val + MHC-ZH val** with a stated combined rule and cross-seed pooling to widen the effective n
  (no longer the 80-sample MHC-EN val alone). The retired single-seed +0.015 threshold sat inside the
  ±1-2pt floor.
- **M2 — anti-drift / anti-deferred-import.** F-G0 gains (a) a **fresh-forward reproduction guard**
  (full-last-layer-hidden read-out over the fresh forward must reproduce the cached pooled floor within
  ±0.010 val before ANY SAV comparison is valid); (b) an explicit **frame-source pin** — decode the same
  **symlinked mp4s** at `data/video/<dataset>/All/<id>.mp4` via the same decord→PyAV 8-frame sampler as
  the cached extractor, **NOT** `data/lora_frames/` (references the symlink-topology / symlinked-mp4
  decode-mismatch burn); (c) a required **deferred-import audit + `codex-code-review`** of the hook code
  before any GPU submit.
- **M3 — 3-axis confound resolved/scoped.** F-G1 adds two isolating control probes — **(C-pos)**
  final-token full last-layer hidden state (isolates token position) and **(C-sparse)** mean-pooled
  selected-head vectors (isolates sparsity from position); the "**mean-pooling dilution**" causal claim
  in §1/§6 is **re-scoped** and licensed only if the SAV feature beats C-pos (else "final-token
  re-reading recovers discriminability").
- **M4 — external-number provenance + model mismatch.** §2 flags the **VLGuard "+62.9% (31.0→94.3)"**
  internal inconsistency (Δ = +63.3) as **PENDING PDF re-verification**, and adds the explicit
  **Qwen2-VL-7B (SAV results) vs Qwen2.5-VL-7B (our encoder)** model-mismatch caveat (head-selection
  efficacy on 2.5 is an assumption F-G1 tests, not a given).
- **M5 — two tightenings.** §3/F-G0 **commit to extracting all 784 heads** (dropped the layer-subset
  hedge — full extraction ≈1.2 GB total, trivial, and the hedge was a hidden degree of freedom in head
  selection); §5 records the **on-screen-text** encoder-prompt clause as **pre-existing baseline
  behaviour, not a new OCR channel** (user OCR veto stands).
- **Recommended (review §7).** F-G1 adds **full-784-head-concat (U-1)** and **best-single-head (U-2)**
  cheap upper-bound reference probes.

**Status after Rev-1: DRAFT-REVISED-AWAITING-REREVIEW.** No gate executed, no code changed, nothing
committed. Awaiting a fresh 0-context RE-REVIEW + a user-visible report before any `sbatch`.

### Rev-2 2026-07-13 applied re-review residuals R1–R3 + Rec-1..3

The Rev-1 re-review (same reviewer, second pass; appended to `refine-logs/SAV_F0_PREREG_REVIEW.md`)
verified M1–M5 as faithfully applied (in two places stronger than asked), found **no design-level
objection**, and returned **REVISE (minor)** with **pre-authorization: APPROVED once R1–R3 land as
written (delta-check only)**. All residuals applied, text-only, no design change, no new cost:

- **R1a — seeds now inject declared, genuine variation (F-G1 §Extraction & selection).** Stated that the
  frozen-forward extraction is deterministic and run once, and that full-train nearest-centroid selection
  would be deterministic — so the ≥5 seeds are pre-declared to vary (i) SAV-style **head-selection
  subsample draws** (20/class from train, without replacement) and (ii) **probe train-split resampling**
  (stratified 80% of train). Identical-replicate degenerate-CI risk named and closed.
- **R1b — "≈400 val-example draws" pooling error RETIRED (F-G1 §Evaluation sets + co-primary bullet).**
  Seed×example draws are correlated, not ~400 independent samples; pre-declared the **example-level
  clustered bootstrap** (resample the 80 MHC-EN val examples; within each draw average each example's
  paired delta across seeds first; 10,000 draws). **Effective n stays 80** — seeds reduce per-example
  variance, they do not multiply n. Same clustered rule applied to ΔL, the projected-gain bootstrap, and
  HateMM (107) / MHC-ZH (78).
- **R1c — MDL estimator + bits→acc rule PINNED (F-G1 §Primary metric).** Estimator = **holdout log-loss**
  (description length of val labels under a train-fit probe; p̂ clipped to [10⁻⁶, 1−10⁻⁶]); NOT
  prequential/online (ordering ambiguity). Conversion = **Fano / inverse-binary-entropy projection**
  acc(ℓ) = 1 − h₂⁻¹(min(ℓ,1)); the empirical-slope alternative explicitly NOT used (a fitted slope would
  be a post-hoc degree of freedom).
- **R2 — two-tier reproduction guard (F-G0(b) + kill clause).** PRIMARY = **feature-level per-video
  cosine ≥ 0.999** between fresh-forward pooled features and the cached ones (img + text, all train+val
  videos); SECONDARY/confirmatory = the ±0.010 val probe, with the **flip-quantization fact stated**
  (±0.010 on 80-sample val = zero flips; can trip on benign bf16 nondeterminism); primary-pass +
  secondary-trip = guard PASSES with recorded discrepancy. No post-hoc tolerance amendment possible.
- **R3 — F-G2 statistics tightened; +0.015 bar KEPT (reviewer explicitly ruled against raising to
  +0.030).** (a) Pass rule gains the **example-level paired bootstrap CI-excludes-0** co-requirement
  (same clustered rule as F-G1); (b) the **HateMM no-harm kill number (3-seed mean paired Δacc <
  −0.010)** now stated inside F-G2's kill rule, not only in the run spec.
- **Rec-1 — U-1 pre-declared as the sparsity-vs-layer-depth attribution tie-breaker** if SAV beats C-pos
  (U-1 ≈ SAV ⇒ head-space/multi-layer read-out, not sparsity per se; SAV > U-1 ⇒ sparsity contributes),
  with the **100,352-d U-1 probe regularization pinned** (same probe family, L2, λ by 5-fold CV within
  the seed's train split, never tuned on val).
- **Rec-2 — C-sparse construction clarified:** selected heads' outputs **span-mean pooled over token
  positions** (token-axis pooling, per-head identity kept, pooled per-head vectors concatenated — not an
  average across heads).
- **Rec-3 — editorial:** stale "C-line queues behind A-line M2/M3" clauses updated everywhere (A-line is
  **PAUSED**, `refine-logs/lb_scgp_global/A_LINE_PAUSE_DECISION.md`; SAV is the **lead experiment**, GPU
  free); §5 OCR-note cite corrected to `generate_VideoMLLM_embedding_HF.py:45-47`.

**Status after Rev-2: DRAFT-REV2-AWAITING-DELTA-CHECK.** No gate executed, no code changed, nothing
committed. Awaiting the pre-authorized delta-check of R1–R3 + a user-visible report before any `sbatch`.

### Rev-2a 2026-07-13 execution-notes pinning (delta-check non-blocking notes 1 & 3)

Applied by the implementation agent alongside the F-G0/F-G1 code build (no design change, no gate run,
nothing committed by this edit). These are the ONLY two prereg edits made during implementation:

- **Exec note 1 (delta-check note 1) — carry-forward head set pinned PRE-F-G1 (§4 F-G2 Pre-flight).**
  The head set deployed to F-G2/F-G3 is the **deterministic full-train nearest-centroid selection**
  (canonical SAV over the entire train set, top-k for the F-G1-promoted k); the five 20/class seed draws
  power only the F-G1 cross-seed CI and a reported head-set consensus/stability diagnostic (intersection
  size + mean pairwise Jaccard) — they do not choose the deployed set. Pinned before any F-G1 result is
  seen; realised in code as the F-G1 engine's full-train `oracle_order` arm, carried forward unchanged.
- **Exec note 3 (delta-check note 3) — §6 stale "before the more expensive C1" editorial fixed.** C1 is
  `KILL_CONFIRMED` (`refine-logs/C1_KILL_REVIEW.md`); the clause is annotated as stale and SAV/C2 is the
  standalone lead C-line pilot.

Delta-check note 2 (C-sparse span-mean per-head vectors emitted in the same extraction pass) is realised
directly in the F-G0 extractor (`sav_f0_extract.py` emits `img_head_spanmean` alongside `img_head_final`),
so no prereg text change was required for it. Code: `scripts/analysis/sav_f0_{common,extract,guard,probe}.py`,
`scripts/wrappers/sav_f0.sh`, `scripts/slurm/sav_f0.sbatch`; impl self-audit: `refine-logs/SAV_F0_IMPL_NOTES.md`.

### Rev-2b 2026-07-13 main-loop rulings: noise band 0.010 (A-line precedent), img-stream-only scope pinned

Two main-loop rulings on the implementation agent's open questions, applied pre-F-G1 (no gate run,
nothing committed, no design change beyond pinning two already-flagged constants/scopes):

- **Noise band = 0.010 ⇒ F-G1 projected-gain bar = +0.040 acc.** The implementation had provisionally
  pinned the "±1-2pt noise band" at its upper end (0.020 ⇒ bar 0.050); the main loop ruled **0.010**
  for protocol consistency with the A-line G0-cond probe precedent
  (`refine-logs/lb_scgp_global/M1_G0COND_PROBE_RECORD.md`: +0.030 + 0.01 = +0.040) — a per-gate
  drifting bar invites protocol-inconsistency criticism. Applied in §4 F-G1 (bar paragraph),
  `scripts/analysis/sav_f0_common.py` (`NOISE_BAND_ACC`/`PROJECTED_GAIN_BAR`), and the F-G1 verdict
  JSON `config` echo.
- **Probe stream = IMG only, ACCEPTED and pre-declared.** F-G1 per-head extraction + all probe arms on
  the img (visual+instruction "prefix") forward — the literal dilution-hypothesis target and the only
  stream where C-pos is non-degenerate; text stream stays pooled-only for the F-G0(b) guard.
  **Text-stream / concat per-head extraction DEFERRED as an F-G2-stage option only if SAV wins F-G1**
  (pre-declared in §4 F-G1 so it cannot become a post-hoc degree of freedom).
- Open questions 3 & 4 (uniform StandardScaler + L2 LogisticRegressionCV probe family across arms;
  U-1 CPU pole) accepted as-is, no change.
