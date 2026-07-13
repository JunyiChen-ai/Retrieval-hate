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
