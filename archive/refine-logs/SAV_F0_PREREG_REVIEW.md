# TASK B — Pre-registration review of exp-sav-f0.md (C2: SAV sparse attention-head mining)

**Reviewer:** fresh zero-prior-context reviewer, 0-context ceremony style. Read-only. No GPU, no
submissions, no code edits.
**Date:** 2026-07-13.
**Object:** `research-wiki/experiments/exp-sav-f0.md` (DRAFT-UNREVIEWED). Anchor: Sparse Attention
Vectors (SAV), ICCV 2025, arXiv 2412.00142.

## VERDICT: **REVISE** — 5 mandatory revisions before any `sbatch`.

The design is structurally sound, cheap, representation-level (the project's only proven +3 family),
non-isomorphic to the 14 closed routes, and carries a genuine falsifiable null. It is worth doing and
should **not** be prematurely killed. But the **deciding cheap gate (F-G1) is under-powered and
confounded as drafted**, and it does not meet the project's own G0-cond bar (REFLECTION §4). Fix the
five items below; then it is APPROVED-able.

---

## 1. SAV procedure vs the actual paper (arXiv 2412.00142v3, WebFetch 2026-07-13)

The draft's §2 description matches the paper on every load-bearing point:

| draft §2 claim | paper (fetched) | match |
|---|---|---|
| Per-head feature = attention vector **h^{l,m}(x_i^T) at the final token** (not pooled hidden state) | "we compute the attention vector h^{l,m}(x_i^T) for head m from layer l for the final token x_i^T" | ✅ |
| Head selection = per-class **centroid** (mean), score each head by **nearest-centroid cosine accuracy**, keep top-k | "compute its centroid (mean) attention vector … measure the classification accuracy of each head" via cosine | ✅ |
| **≈20 heads (<5%)** | "just 20 vectors are enough … less than 5% of the heads" | ✅ |
| Inference = **majority vote** across selected heads, each a local nearest-centroid classifier | "a global classifier that counts the majority vote across all local classifiers" | ✅ |
| Few-shot ≈ **20 examples/class** | "just 20 examples per label are necessary" | ✅ |
| Base MLLMs LLaVA-OneVision-7B and **Qwen2-VL-7B** | both confirmed | ✅ (see M4) |

**Procedure description: ACCURATE.** No misreading of the method.

**Two accuracy nits (→ M4):**
- The draft cites **VLGuard +62.9% (31.0→94.3)**; 94.3 − 31.0 = **+63.3**, not +62.9. Either endpoint
  or delta is off by ~0.4. Not load-bearing (external motivation, not a floor), but it is a transcribed
  external number and must be re-verified against the PDF like the RA-HMD numbers.
- SAV validated on **Qwen2-VL-7B**; our encoder is **Qwen2.5-VL-7B**. Same family, *not* the same
  model. The extraction mechanics transfer; SAV's head-selection *efficacy* on 2.5 is an assumption,
  not a result. The draft says "same family" — make that caveat explicit.

---

## 2. Numeric-claim provenance audit (every number → source, re-checked)

I re-derived every floor the draft cites from `exp-encoder-3seed.md` and confirmed the arithmetic.

| draft number | source cited | independent re-derivation | status |
|---|---|---|---|
| MHC-EN frozen-Qwen val-sel **0.7805 / 0.7219** | `exp-encoder-3seed.md:164–170,257–259` | acc (0.7888+0.7826+0.7702)/3 = 0.7805; F1 (0.7378+0.7283+0.6997)/3 = 0.7219 | ✅ |
| MHC-EN frozen-Qwen final-ep **0.7847 / 0.7425** | same | acc (0.8012+0.7702+0.7826)/3 = 0.7847; F1 (0.7596+0.7203+0.7475)/3 = 0.7425 | ✅ |
| HateMM frozen-Qwen val-sel **0.8729 / 0.8648** | `exp-encoder-3seed.md:154–159,251–253` | acc (0.8698+0.8651+0.8837)/3 = 0.8729; F1 (0.8606+0.8586+0.8753)/3 = 0.8648 | ✅ |
| HateMM frozen-Qwen final-ep **0.8682 / 0.8591** | same | acc (0.8605+0.8605+0.8837)/3 = 0.8682; F1 (0.8507+0.8514+0.8753)/3 = 0.8591 | ✅ |
| encoder-swap MHC-EN mean Δacc **+0.019 val / +0.006 final**, FAILS both | `exp-encoder-3seed.md:207,219,231` | val-sel mean +0.0186; final-ep mean +0.0062; verdict FAIL/FAIL | ✅ |
| MHC-EN "0.77–0.80 band regardless of encoder" | `exp-encoder-3seed.md:240–241` | consistency note verbatim | ✅ |

**Cache-insufficiency claim (F-G0) — VERIFIED LIVE.** `src/utils/generate_VideoMLLM_embedding_HF.py`
(read, lines 9–52): caches exactly **two 3584-d vectors/video** — `img_feats` = **mean of the
LAST-layer hidden states** over the visual+instruction span (L2-normed); `text_feats` = mean of the
LAST-layer hidden states over the response-tail span (L2-normed). **No per-head, no per-layer, no
attention outputs.** The draft's §3 claim ("cached hidden states are insufficient; SAV needs a fresh
per-head forward") is **correct**.

**(L, H, head_dim) — VERIFIED LIVE** against the local config
(`models--Qwen--Qwen2.5-VL-7B-Instruct/.../config.json`): `num_hidden_layers=28`,
`num_attention_heads=28`, `hidden_size=3584` ⇒ head_dim = 3584/28 = **128**; `num_key_value_heads=4`
(GQA). So **28×28 = 784 query-head positions**, matching the draft's "≈784 head-positions, <5% ≈
20–39 heads" exactly. GQA does not affect this: the per-query-head attention **output** is head_dim=128
and 28 of them concatenate to 3584 before o_proj.

**Provenance verdict:** every internal (floor) number has correct file:line and matches its source.
Only the two *external* SAV numbers need PDF re-verification (M4).

---

## 3. Gate structure — fail-closed audit

| requirement | draft status | assessment |
|---|---|---|
| F-G0 / F-G1 kill numbers pre-declared | F-G0 kill = extraction infeasible / storage intractable; F-G1 kill = SAV probe < pooled +0.015 val at any k | ✅ declared, but F-G1 threshold under-powered → **M1** |
| Matched-capacity probe | F-G1 baseline probe = **same** linear/kNN probe over pooled feature vs same probe over SAV-head feature | ✅ capacity matched — but confounded on other axes → **M3** |
| Multi-seed CI | F-G1 = **single seed** ("same val split, same seed"); F-G2 = 3-seed | ✗ the *deciding cheap gate* is single-seed → **M1** |
| Projected-gain-vs-noise-floor argument | absent at F-G1 (raw +0.015 val-acc threshold, no noise-floor argument) | ✗ → **M1** |
| MDL/codelength (REFLECTION §4 recipe) | absent (F-G1 uses accuracy, not codelength) | ✗ → **M1** (accept capacity-matched multi-seed probe as substitute *if* justified) |
| Test-touch budgeted | F-G3 only; val for F-G0/1/2 | ✅ |
| Single-submit ceremony | "each gate = one pre-registered serial sbatch, single-submit-per-lineage" | ✅ |
| No gold annotations | head selection uses **train binary labels only**; dev for selection; test only at F-G3 metric | ✅ |
| Label use limited to train labels | ✅ (nearest-centroid over train memory = standard supervised training-data use) | ✅ |
| No OCR / no cross-seed ensembles / local 7B only | upheld §5 | ✅ (one note → M5) |
| Code-review of model-internals code | "per-head extraction / hook code routes through codex-code-review" | ✅ (strengthen → M2) |

**The structure is fail-closed in the right places** (test touch preserved to F-G3, val-judged gates,
pre-declared kills, both protocols at F-G2/G3). The failure is concentrated in **F-G1's power and
confounding**, which is the gate that decides whether the line proceeds at all.

---

## 4. Known-failure-mode hunt (project burn history)

- **Deferred imports:** the per-head hook / extraction is *new* model-internals code — the exact class
  the project's deferred-import audits target. F-G0 must include a deferred-import audit → **M2**.
- **Plan/config drift & smoke-tests-bypassing-frozen-entries:** **the largest risk here.** The SAV
  feature (final-token, per-head, all-layer attention output) differs from the pooled baseline
  (span-mean, full-hidden, last-layer-only) on **three** axes: granularity, token position, and layer
  depth. A fresh forward that silently uses a different frame sampler / prompt / span than the cached
  pipeline would confound any result. There is **no reproduction guard** in the draft (the C1 draft has
  one: E-G0(e) "λ/no-op reproduces the baseline to 4 dp"). → **M2** (add a reproduction guard: a
  full-hidden probe over the fresh forward must reproduce the pooled-feature floor) and **M3** (add an
  isolating control for the position/layer confound).
- **Storage-topology / symlinked mp4s:** the burn history flags symlinked-mp4 decode mismatches. The
  fresh forward must read frames from the **same source** as the cached extraction (pre-extracted
  `data/lora_frames/` vs decoded from symlinked mp4). F-G0 must pin the frame source → folded into M2.
- **Val-selection artifacts:** MHC-EN val = **80 samples** (verified: `data/gt/MHC/val.jsonl` = 80).
  1 acc point = 1.25%. A single-seed +0.015 threshold ≈ **1.2 flipped examples** — inside the ±1–2pt
  noise floor the project itself established. This is the same trap that produced TARC's false-positive
  cell and the archive-as-key withdrawal. → **M1**.

---

## 5. VRAM / runtime / storage sanity (per-head extraction, ~1k videos × 3 datasets, 1–2 A100-80G)

**Credible, with one over-cautious statement to relax.**

- **VRAM:** a single frozen Qwen2.5-VL-7B forward (8 frames, `image_max_pixels 65536`) fits **one**
  A100-80G in bf16 — P9 proved this for the LoRA forward (`exp-e2eq-e0.md:222–227`); a frozen forward
  is strictly lighter. A forward **hook on `self_attn.o_proj` input** adds ~zero VRAM (it captures an
  already-computed [.., 28×128] tensor at the final token). ✅
- **Do NOT use `output_attentions=True`** to get the head vectors: that returns the softmax
  **attention-weight matrices** (seq_len² × heads × layers), which over an 8-frame video (thousands of
  visual tokens) is a large tensor and is also the *wrong object* (SAV wants the value-weighted head
  **output**, not the weights). The draft **correctly** identifies this (§F-G0(b): hook o_proj input).
  Keep that; the o_proj-input hook is the right and cheap mechanism and works regardless of the
  SDPA/flash-attn kernel (o_proj is a distinct Linear in Qwen2.5). ✅
- **Runtime:** single frozen forward per video = same order as the existing frozen extraction (tens of
  min/dataset). Head-selection + linear probe are CPU/seconds. ✅
- **Storage (draft over-worries):** 784 heads × 128 dim × 4 B (fp32) = **~401 KB/video**; keeping only
  the final-token vector. ~1,000 videos × 3 datasets ≈ **~1.2 GB total** (fp32; ~0.6 GB fp16). This is
  **trivial** — the draft's F-G0(c) "likely keep only a candidate layer subset if full is too large" is
  unnecessary caution; full extraction is fine. Relaxing it also avoids a layer-subset selection knob
  that could itself become a hidden degree of freedom. → **M5** (minor).

**Runtime/VRAM verdict: the draft's estimate is credible on 1 GPU;** storage is a non-issue.

---

## 6. MANDATORY REVISIONS (numbered)

**M1 — F-G1 must meet the project's own G0-cond bar; the current +0.015 single-seed val threshold is
inside the noise floor.** MHC-EN val = 80 samples ⇒ +0.015 ≈ 1.2 examples. As drafted, F-G1 can pass
on noise or kill a real small effect — the exact failure the reflection institutionalized against.
Required:
  (a) Make F-G1 **multi-seed** (≥3 seeds; extraction is cheap) and judge the probe delta by a
      **bootstrap/across-seed CI that excludes 0**, not a single-seed point estimate ≥ +0.015.
  (b) Add the **projected-gain-vs-noise-floor** argument: state the effective val n, the per-example
      resolution, and why the chosen bar sits above the ±1–2pt floor (or widen the effective sample by
      pooling val across seeds / bootstrapping).
  (c) Either adopt an **MDL/codelength** read (REFLECTION §4(ii)) **or** explicitly justify that a
      capacity-matched **multi-seed** accuracy probe with a CI-excludes-0 rule is the accepted
      substitute for this *representation-level* (not auxiliary-signal) route — and say so, since SAV
      is a feature swap, not a low-bandwidth decision-side signal.

**M2 — Add a reproduction guard + extraction-hygiene audit to F-G0 (anti-drift, anti-deferred-import).**
The fresh per-head forward must be proven to be the *same pipeline* as the cached pooled extraction,
differing only in the read-out. Required:
  (a) A **reproduction guard**: a probe/RGCL read-out over the **full last-layer hidden state** taken
      from the *fresh* forward must reproduce the cached pooled-feature floor to a stated tolerance
      (analogue of the C1 draft's E-G0(e) 4-dp guard). If it does not, the fresh forward has drifted
      (frame sampler / prompt / span / dtype) and no SAV comparison is valid.
  (b) Pin the **frame source** (pre-extracted `data/lora_frames/` vs symlinked-mp4 decode) to match the
      cached extraction exactly.
  (c) A **deferred-import audit** + `codex-code-review` of the hook code *before* any GPU submit
      (the draft names codex-code-review; make the deferred-import audit explicit, per burn history).

**M3 — Resolve (or scope) the 3-axis confound in the F-G1 comparison.** SAV feature vs pooled baseline
differ in granularity **and** token position (final-token vs span-mean) **and** layer depth (all-layer
vs last-layer). A SAV win is therefore not attributable to "sparse heads fix mean-pooling dilution"
alone — the draft's headline causal claim. Required, at least one of:
  (a) Add an isolating control probe — e.g. **final-token full last-layer hidden state** (isolates the
      position axis) and/or **mean-pooled selected-head vectors** (isolates the sparsity axis from
      position) — so the dilution claim is tested against the right counterfactual; **or**
  (b) Explicitly **re-scope** the causal claim in H/§6 to "sparse-head *re-reading of all layers at the
      final token* recovers discriminability" and drop the specific "mean-pooling dilution" attribution
      unless (a) isolates it. (The **goal** — does SAV-feature RGCL beat pooled-feature RGCL by +0.030
      on test — is unaffected; this is about not over-claiming the mechanism in the paper narrative.)

**M4 — Re-verify the external SAV numbers against the PDF and fix the Qwen2-VL vs 2.5 caveat.**
  (a) VLGuard "+62.9% (31.0→94.3)" is internally inconsistent (Δ = 63.3); re-read Table 1 decimals from
      the PDF and correct, same discipline as the RA-HMD-number caveat.
  (b) State plainly that SAV's *results* are on Qwen2-VL-7B (+ LLaVA-OneVision-7B), and that transfer of
      its head-selection efficacy to **Qwen2.5-VL-7B** is an assumption F-G1 is testing, not a given.

**M5 — Two minor tightenings.**
  (a) Drop the F-G0(c) "keep only a layer subset if too large" hedge — full per-head extraction is
      ~1.2 GB total (trivial); a layer-subset knob is an unnecessary hidden degree of freedom. Extract
      all 784 heads.
  (b) Note (non-blocking) that the frozen encoder's *img instruction* already mentions "on-screen
      text" (`generate_VideoMLLM_embedding_HF.py:45–47`); this is part of the **pre-existing** banked
      encoder-swap baseline, not a new OCR channel, so it does **not** violate the OCR veto — but record
      it so a later reviewer doesn't mistake it for one.

---

## 7. What is already right (keep, do not weaken)

- **Non-isomorphism** to the 14 closed routes is real: SAV changes *which coordinates of the frozen
  encoder's own representation* feed the head — representation-level, like the encoder swap — not a
  low-bandwidth MLLM output injected on the decision side. This is the correct family (D2).
- **The falsifiable null is genuine and cheap:** if a matched-capacity probe over pooled features
  already reaches MHC-EN's ceiling, the dilution hypothesis is falsified at ~1 GPU-hour — a
  paper-usable negative ("MHC-EN is data-limited, not pooling-limited"). Both a pass and a clean kill
  have value; this justifies running it before the (killed) C1.
- **Test-touch discipline, both-protocol judging, single-submit ceremony, no-gold-annotation
  isolation, codex-code-review of internals** are all correctly specified.
- **The o_proj-input hook mechanism is the correct, cheap, kernel-agnostic way to get per-head
  outputs** — and the draft correctly rejects the naive `output_attentions` path.

**Recommended (not mandatory):** add a cheap **upper-bound reference** to F-G1 — the probe accuracy of
the **full 784-head concatenation** and of the **single best head** — so a kill reads "no head subset
carries information the pooled feature lacked" rather than only "top-k-by-nearest-centroid didn't."

---

## 8. Bottom line

**REVISE.** C2/SAV is the right kind of bet — cheap, frozen, representation-level, non-isomorphic, with
a real falsifiable account of the MHC-EN encoder failure — and should proceed after fixing the deciding
gate. The blocking issues are all at **F-G1**: it is single-seed on an 80-sample val with a +0.015
threshold inside the noise floor (M1), lacks a fresh-forward reproduction guard (M2), and confounds
head-sparsity with token-position and layer-depth (M3), plus two provenance/scoping fixes (M4, M5).
Land M1–M5 and F-G1 becomes a properly powered, confound-controlled, G0-cond-compliant falsification
gate — at which point the pre-registration is APPROVED-able.

---

### Provenance index (this review)

- SAV method/numbers: arXiv 2412.00142v3 (HTML, WebFetch 2026-07-13).
- Frozen-Qwen floors: `research-wiki/experiments/exp-encoder-3seed.md:154–170,240–241,251–259`.
- Cache contract (insufficiency): `src/utils/generate_VideoMLLM_embedding_HF.py:9–52` (read live).
- Qwen2.5-VL-7B dims (28L×28H, head_dim 128, GQA 4): local `config.json` (read live 2026-07-13).
- Set sizes: `data/gt/MHC/val.jsonl`=80, `test.jsonl`=161, `train.jsonl`=549; `MHC_zh` val 78 / test
  149 / train 579; `HateMM` val 107 / test 215 / train 744 (counted live).
- G0-cond recipe / noise-floor doctrine: `research-wiki/REFLECTION_mllm_integration_failures.md:37–43`;
  `research-wiki/LITERATURE_mllm_integration_2026-07-13.md:60–65`.

---
---

# Rev-1 RE-REVIEW (same reviewer, 2026-07-13, second pass)

**Object:** `research-wiki/experiments/exp-sav-f0.md` at status DRAFT-REVISED-AWAITING-REREVIEW
(Rev-1, §8 revision history). Checked against my own M1–M5 + recommended items above.
**Context change acknowledged:** A-line lb_scgp_global is PAUSED at a zero-GPU G0-cond kill
(`refine-logs/lb_scgp_global/A_LINE_PAUSE_DECISION.md`, verified to exist and to say what the
coordinator says it says) — SAV is now the lead experiment; this re-review is the last gate before
an execution-authorization cycle. That raises, not lowers, the bar for statistical soundness of the
deciding gate.

## VERDICT: **REVISE (minor — Rev-2 is text-only statistics pinning, no design change).**

All five mandatory items M1–M5 are applied faithfully and, in two places, better than I asked
(≥5 seeds; oracle-selection pre-check added). **No design-level objection remains.** Three residual
items (R1–R3) must land in the pre-registration text before `sbatch` because they pin the deciding
gate's statistics; all are small edits. **Upon faithful application of R1–R3 as written, this
pre-registration is APPROVED — a delta-check of the edited text suffices; no further full re-review
cycle is needed.**

## 1. Item-by-item compliance check

| item | required | Rev-1 state | verdict |
|---|---|---|---|
| M1a multi-seed | ≥3 seeds, CI excludes 0 | **≥5 seeds (0–4)**, bootstrap CI excludes 0, cross-seed distribution decides | ✅ exceeds — **acceptable** (coordinator's question answered: 5 > 3 is strictly better), but conditional on R1a: the seeds must inject *genuine* variation |
| M1b noise-floor argument | projected-gain-vs-noise argument, effective n stated | explicit bar = projected > +0.030 acc + noise band, CI-excludes-0; per-example resolution (1.25%) and effective n stated; G0-cond quote verbatim (`REFLECTION:41`) | ✅ (one statistical error in the "≈400 draws" claim → R1b) |
| M1c MDL or justified substitute | codelength primary OR justified capacity-matched accuracy | **MDL/codelength primary** + capacity-matched accuracy **co-primary** with the exact justification I required (representation-level feature swap, not low-bandwidth aux signal) | ✅ (estimator not pinned → R1c) |
| M1 (scope widening) | — (my review implied EN-only decision was fragile) | decision spans MHC-EN + HateMM no-harm + MHC-ZH secondary with a pre-declared combined rule; EN remains the carrying target | ✅ good addition |
| M2a reproduction guard | fresh-forward full-hidden read-out reproduces pooled floor, stated tolerance | present, ±0.010 val acc, admissibility-blocking | ✅ present (tolerance quantization issue → R2) |
| M2b frame-source pin | same source as cached extraction | pinned to **symlinked mp4s** `data/video/<ds>/All/<id>.mp4` via the same decord→PyAV 8-frame sampler; lora_frames explicitly forbidden | ✅ **independently verified this pass**: extractor reads `--video_dir ./data/video`, path `<dataset>/All/<id>.mp4` (`generate_VideoMLLM_embedding_HF.py:73-76,334,341`); `data/video/HateMM/All/*.mp4` and `data/video/MHC/All/*.mp4` confirmed symlinks into `/data/jehc223/HateMM/video/` and `/data/jehc223/Multihateclip/English/video_mp4/` (checked live). The pin is factually correct and matches the cached pipeline — my original review's uncertainty about which source the cached extraction used is resolved: it is the symlinked mp4s, NOT lora_frames, so Rev-1 chose correctly |
| M2c deferred-import audit + review | explicit, pre-submit | both required as F-G1 prerequisites, kill-wired | ✅ |
| M3 confound controls | isolating control(s) or re-scope | **both**: C-pos + C-sparse controls AND the causal claim re-scoped in §1/§6, licensed only if SAV beats C-pos | ✅ (attribution nuance → Rec-1; C-sparse wording → Rec-2) |
| M4a VLGuard decimals | flag pending PDF | flagged, transcription-blocked until PDF re-read | ✅ |
| M4b Qwen2-VL vs 2.5 | explicit assumption caveat | present, framed as "assumption F-G1 is testing" | ✅ |
| M5a all 784 heads | drop layer-subset hedge | committed, storage bounded (~1.2 GB), hedge explicitly retired as a hidden DoF | ✅ |
| M5b OCR-veto note | record on-screen-text prompt clause | present in §5 | ✅ (line cite is 44-46; the constant actually spans 45-47 — trivial, Rec-3) |
| Recommended U-1/U-2 | upper-bound probes | both added | ✅ |

## 2. Residual mandatory items (Rev-2, text-only)

**R1 — Pin the F-G1 statistics precisely (the CI must be real, not decorative).**
  (a) **State what the ≥5 seeds actually vary.** The extraction is a deterministic frozen forward;
      nearest-centroid head selection on the full train set is also deterministic. If nothing varies,
      5 seeds produce 5 identical replicates, the cross-seed CI degenerates to a point, and
      "CI excludes 0" becomes trivially true for any nonzero delta — illusory rigor of exactly the
      kind the burn history punishes. The seeds must inject genuine variation: e.g. resampled
      head-selection subsets (SAV-style ~20/class few-shot draws from train) and/or resampled probe
      train splits. Declare which.
  (b) **Fix the pooling claim.** "val pooled across the ≥5 seeds (≈400 val-example draws)" is a
      statistical error as written: the same 80 val examples across seeds are correlated (identical,
      if (a) is not fixed), and seed×example are NOT ~400 independent draws — pooling replicates
      narrows the CI by ~√5 spuriously. Required: bootstrap **clustered at the example level**
      (resample the 80 MHC-EN val examples; within each draw, average the per-example paired delta
      across seeds). Effective n stays 80; seeds reduce per-example variance, they do not multiply n.
  (c) **Pre-declare the exact MDL estimator and bits→acc rule.** "Prequential/online codelength of
      the val labels" is ambiguous (online coding over what ordering? or holdout log-loss = description
      length of val given a train-fit probe?). Pin one (holdout log-loss is the simplest defensible
      choice here), and pin the bits→acc conversion (Fano bound vs empirical slope, per REFLECTION
      §4(iii)'s "Fano/经验斜率") so the +0.030+noise projection is computed one pre-declared way.

**R2 — Reproduction-guard tolerance is quantized to zero flips; pre-declare a feature-level primary
check to avoid a post-hoc amendment.** ±0.010 val acc on the 80-sample MHC-EN val allows **zero**
prediction flips (1 flip = 0.0125 > 0.010), i.e. the guard as written demands exact prediction
reproduction and can trip on benign bf16/kernel nondeterminism — forcing exactly the kind of
after-the-fact tolerance renegotiation the ceremony rules exist to prevent. Required: make the
**primary** guard statistic feature-level — e.g. per-video cosine between the fresh-forward pooled
feature and the cached feature, threshold pre-declared (≥0.999 or a max-abs-diff bound) — with the
±0.010 probe read as secondary/confirmatory, and state the flip-quantization fact in the text. (The
guard's failure direction is safe/fail-closed either way; this is about not baking in a gate that
predictably needs amending.)

**R3 — F-G2 (answering the coordinator's explicit question).** Keeping F-G2 at **+0.015 val is
ACCEPTABLE and I do NOT ask for it to be raised to +0.030.** Rationale, both directions: F-G2's job is
no longer to establish the effect (the rebuilt F-G1 upstream now does that at G0-cond strength); it is
a *survival* check that the probe gain survives the trained RGCL head before spending the one test
touch. Raising it to +0.030 on an 80-sample val (2.4 examples) at 3 seeds would risk killing a true
+0.030-test effect on val noise — the premature-kill direction. The revision agent's judgment call is
endorsed. **Two small tightenings required, same statistics discipline as F-G1:**
  (a) add the example-level paired bootstrap CI-excludes-0 co-requirement to the F-G2 pass rule (mean
      +0.015 + sign ≥2/3 alone can still be noise-carried on 80 samples);
  (b) the HateMM no-harm check appears in F-G2's run spec but has no kill number in its pass/kill
      rule — state it explicitly (reuse F-G1's: mean paired Δacc not below −0.010).

## 3. Recommended (non-blocking)

- **Rec-1:** If SAV beats C-pos, attribution between "sparse heads" and "multi-layer pre-o_proj
  head-space read-out" is still open (C-pos isolates position only; layer depth remains entangled).
  Pre-declare the U-1 reading as the tie-breaker: U-1 ≈ SAV ⇒ the gain is the head-space/multi-layer
  read-out, not sparsity per se. Also note U-1's capacity-matching needs care (784×128 = 100,352-d
  probe input vs 2,560-d for top-20) — state the regularization.
- **Rec-2:** Clarify C-sparse construction in one phrase: the selected heads' outputs **span-mean
  pooled over token positions** (the wording "mean-pooled selected-head vectors" — inherited from my
  own review — is ambiguous between token-pooling and head-averaging).
- **Rec-3:** Editorial: (i) the multiple "C-line queues behind A-line M2/M3" clauses are now stale —
  A-line is PAUSED (`A_LINE_PAUSE_DECISION.md`) and SAV is the lead; harmless (queueing behind a
  paused line = no contention) but update at next touch; (ii) the §5 OCR-note line cite should be
  `generate_VideoMLLM_embedding_HF.py:45-47` (44 is the comment line).

## 4. Bottom line

Rev-1 is a faithful, in places stronger-than-asked, application of M1–M5. The residuals are all
pin-the-statistics edits on the deciding gate — no experiment design changes, no new controls, no new
cost. **REVISE (minor); pre-authorized APPROVED once R1–R3 land as written** (delta-check only).
The route remains worth running: representation-level, cheapest C-line pilot, genuine falsifiable
null, and now the project's lead experiment with GPU free.

### Additional live verifications this pass

- Extractor video path: `src/utils/generate_VideoMLLM_embedding_HF.py:73-76` (`--video_dir`,
  default `./data/video`, "<dataset>/All/<id>.mp4"), `:334` (`video_root`), `:341` (`video_path`).
- Symlink topology: `data/video/HateMM/All/hate_video_100.mp4 -> /data/jehc223/HateMM/video/...`,
  `data/video/MHC/All/01ygFLVdj8s.mp4 -> /data/jehc223/Multihateclip/English/video_mp4/...`
  (ls -la, 2026-07-13).
- decord→PyAV sampler present in the cited region (`_decode_with_decord` at ~:155).
- `refine-logs/lb_scgp_global/A_LINE_PAUSE_DECISION.md` exists (2026-07-13 21:46); confirms A-line
  PAUSE at zero-GPU G0-cond kill and GPU redirect to C-line with SAV as lead.

---
---

# Rev-2 DELTA-CHECK (same reviewer, 2026-07-13, third pass — final)

**Object:** `research-wiki/experiments/exp-sav-f0.md` at status DRAFT-REV2-AWAITING-DELTA-CHECK.
Scope per the pre-authorization in the Rev-1 re-review: verify R1–R3 (+ Rec-1..3) landed **as
written**; no new full review cycle.

## VERDICT: **APPROVED.**

All residuals landed faithfully; in one place (R2) stronger than asked. The pre-registration is now,
in my judgment, a properly powered, confound-controlled, fail-closed, G0-cond-compliant design. No
blocking items remain.

## Delta-by-delta verification (against the residuals as I wrote them)

| residual | required | Rev-2 text (exp-sav-f0.md) | verdict |
|---|---|---|---|
| **R1a** | declare what the ≥5 seeds vary; name the degenerate-CI failure | F-G1 "Extraction & selection": extraction declared deterministic and **run once**; full-train nearest-centroid selection acknowledged deterministic; seeds pre-declared to vary **(i)** SAV-style head-selection subsample draws (20/class, without replacement, from train — matching SAV's own few-shot scale) and **(ii)** stratified 80% probe train-split resampling; the identical-replicates → degenerate-CI → "trivially excludes 0" failure is named verbatim; head-set stability across draws is itself reported | ✅ as written |
| **R1b** | retire "≈400 draws"; example-level clustered bootstrap; n stays 80 | the "≈400 val-example draws" phrasing explicitly called "a statistical error and RETIRED"; procedure = resample the 80 val examples with replacement, average each drawn example's paired delta across seeds **first**, 10,000 draws; "effective n stays 80 — seeds reduce per-example variance, they do not multiply n"; clustered rule extended to ΔL, the projected-gain bootstrap, HateMM (107), MHC-ZH (78); co-primary bullet states seed×example draws are never pooled as independent | ✅ as written |
| **R1c** | pin MDL estimator + bits→acc rule | estimator pinned = **holdout log-loss** (Σ −log₂ p̂ over val, probe fit on the seed's train subset only, p̂ clipped to [10⁻⁶, 1−10⁻⁶]); prequential/online explicitly rejected (ordering ambiguity); conversion pinned = **Fano / inverse-binary-entropy** acc(ℓ)=1−h₂⁻¹(min(ℓ,1)); empirical slope explicitly NOT used (post-hoc DoF) — the exact recommendation | ✅ as written |
| **R2** | feature-level primary guard, ±0.010 demoted to secondary, flip-quantization stated | two-tier guard: PRIMARY = **min per-video cosine ≥ 0.999**, fresh-vs-cached pooled features, **both img_feats and text_feats, over every train+val video** (broader than I asked — I suggested a per-video cosine; Rev-2 pins min over both streams and both splits); SECONDARY = ±0.010 probe, confirmatory only, with the zero-flips quantization fact stated; primary-pass + secondary-trip = PASS with recorded discrepancy; primary-fail = FAIL regardless; "No post-hoc tolerance amendment is needed or permitted"; F-G0 kill clause updated to the primary check | ✅ stronger than asked |
| **R3a** | example-level paired bootstrap CI-excludes-0 in the F-G2 pass rule | pass rule now (i) mean +0.015 both metrics, (ii) sign ≥2/3, (iii) example-level paired bootstrap CI excludes 0 (same clustered rule, 10k draws); my ruling against raising the bar to +0.030 is quoted with its rationale | ✅ as written |
| **R3b** | HateMM no-harm kill number inside the F-G2 kill rule | kill rule now contains "HateMM regresses: 3-seed mean paired Δacc below −0.010 vs the pooled-feature HateMM floor (same no-harm number as F-G1)" | ✅ as written |
| **Rec-1** | U-1 tie-breaker + U-1 probe regularization | pre-declared both readings (U-1 ≈ SAV ⇒ multi-layer head-space read-out, not sparsity; SAV > U-1 ⇒ sparsity contributes); 100,352-d capacity risk named; regularization pinned (same probe family, L2, λ ∈ {10⁻⁴..10²} by 5-fold CV within the seed's 80% train split, never tuned on val) | ✅ |
| **Rec-2** | C-sparse construction clarified | "span-mean pooled over TOKEN POSITIONS … NOT an average across heads; per-head identity kept, pooled per-head vectors concatenated" | ✅ |
| **Rec-3** | stale A-line clauses; OCR cite | scheduling clause rewritten (A-line PAUSED, SAV lead, GPU free, no queueing constraint) in header/status/§4-Ceremony/§7; OCR cite corrected to `:45-47` with "44 is the comment line" | ✅ |

Front-matter verdict line, §7 status, and the §8 Rev-2 history entry are consistent with the body.

## Non-blocking notes for the execution phase (do NOT hold the sbatch for these)

1. **Pin the carry-forward head set before F-G2.** F-G1's five seeds each select heads from a 20/class
   draw (correct for the stability-measured CI), but the draft does not yet say **which head set
   deploys to F-G2** (e.g. re-selection on the full train set — deterministic — or the consensus/
   intersection of the 5 draws). Any choice is fine; it just must be written down in the F-G2
   pre-flight BEFORE F-G1 results are seen (it does not affect F-G0/F-G1 and so does not block this
   approval).
2. **C-sparse needs span-mean per-head vectors — emit them in the same extraction pass.** The
   committed cache is final-token per-head (784×128/video). C-sparse (Rec-2 wording) requires each
   selected head's output span-mean pooled over token positions — computable as a running mean during
   the same forward, roughly doubling the cache to ~2.4 GB total (still trivial), but the hook code
   must know to emit it. Flag for the F-G0 implementation + codex-code-review checklist so it does not
   surface as a surprise plan amendment.
3. **Trivial editorial:** §6 still ends "Run it before the more expensive C1" — C1 is now
   KILL_CONFIRMED (refine-logs/C1_KILL_REVIEW.md); harmless, update at next touch.

## Final status

**APPROVED** per the Rev-1 re-review pre-authorization. The remaining path to GPU is the draft's own:
user-visible report → F-G0 execution (hook feasibility + two-tier reproduction guard + frame-source
pin + deferred-import audit + codex-code-review) → F-G1 sbatch (single-submit ceremony). This
reviewer's role is complete; notes 1–2 above should be folded into the F-G0/F-G2 pre-flight
checklists by the executing agent.
