# C3 NON-TARGET content pilot — G0-cond gate DESIGN (pre-registration)

Date: 2026-07-14
Author = **Claude Opus 4.8** (`claude-opus-4-8`), C3-nontarget G0 pilot executor.
conda `HateVideo`; SLURM `sbatch` only (NEVER `--time`); single generation job, 1 GPU.
No commits (archiver handles commits).

**This document is fixed BEFORE any probe number is seen.** It declares ONE generation
prompt family, the sample, the evidence pack, the embedding pathway, the corrected-machinery
conditional probe, and a single pre-declared decision rule. It is the anti-gate-hacking
contract for the C3 non-target channel.

---

## 0. Why this pilot exists (provenance chain)

- `research-wiki/REFLECTION_mllm_integration_failures.md` §4 institutes the **G0-cond gate**:
  any auxiliary-signal route must pass a zero-GPU conditional-information probe (does A add
  label info *beyond* the frozen features Z?) BEFORE any experiment, with a **mandatory
  label-oracle calibration arm** (feed gold label as A; it MUST reach ~full Fano headroom, else
  the machinery is invalid — the 2026-07-14 calibration mandate born from the crush bug).
- `refine-logs/C3_PROBE_VERDICT_REVIEW.md`: the corrected machinery (Z standardized alone at its
  optimal C; the auxiliary block appended as a raw, effectively-unpenalized encoding × s so a
  shared heavy L2 cannot crush it). Label-oracle then hits accZA = 1.0000 exactly.
- `refine-logs/C3_REAL_PREDICTOR_PROBE.md`: the **target** channel of C3 is **DEAD on a real
  predictor** (Δacc ≈ 0, no CI clears 0, ≥4× below the +0.040 bar). Verdict there:
  *"C3 as a family now requires a NON-target content pilot to stay alive."* This is that pilot.
- `research-wiki/LITERATURE_mllm_integration_2026-07-13.md` §1-C3: the surviving C3 content =
  **world-knowledge / implicit-reasoning** text (coded language, dog-whistles, symbols, gestures,
  scene context). Explicitly **NOT** target-category restating, **NOT** transcript restating, and
  **NO OCR channel** (user veto 2026-07-13). The banked encoder prompt's pre-existing on-screen-text
  clause is baseline and is *not* replicated here — this generation prompt adds no OCR emphasis.

## 1. ANTI-GATE-HACKING RULES (binding)

1. **ONE pre-declared prompt family** (§3), fixed before seeing any probe number.
2. **No prompt iteration against the probe.** The generation prompt, sample, embedding pathway,
   probe machinery, and decision rule below are frozen by this document. If any probe number
   later suggests a "better" prompt, that is a NEW pre-registered pilot, not an edit of this one.
3. **If the gate fails, C3-nontarget is DEAD at G0** (`C3_NONTARGET_DEAD_AT_G0`). Any revival
   requires a *new* pre-registered justification (new content family, new dataset, or a documented
   machinery defect), never a re-run of this prompt with tweaks.
4. **No gold annotations touch the generation.** Labels are used ONLY (a) to draw the stratified
   sample (SAMPLING only — see §2) and (b) inside the probe (gate role). The generation prompt
   receives frames + title + ASR only; never the label, never the gold target.

## 2. Sample (stratified by label — labels used for SAMPLING ONLY)

- **300 HateMM train** + **300 MHC-EN train** videos, drawn from `data/gt/{HateMM,MHC}/train.jsonl`.
- Stratified by gold label to preserve each dataset's train hate-rate, so the probe has balanced
  power in both classes:
  - HateMM train N=744, hate-rate 0.4005 → **120 hate + 180 non-hate**.
  - MHC-EN train N=549, hate-rate 0.3060 → **92 hate + 208 non-hate**.
- **Labels are consumed here ONLY to choose which ids to generate for** (probe-power balancing).
  The label never enters the generation prompt or the MLLM context. This use is recorded and is
  the only sampling-time gold touch.
- Deterministic: within each label group the ids are sorted, then sampled without replacement with
  `numpy.random.default_rng(20260714)`. The chosen ids + labels are written once to
  `artifacts/c3_nontarget/<dataset>_sample300.json` (the stratification record) and reused by the
  probe. If the manifest already exists it is loaded, never re-drawn.
- Videos that fail to decode (0 frames) are kept in the sample with a zero A_text vector and
  `ok=false` (mirrors the banked extractor's zero-guard); the probe records/handles them.

## 3. THE single generation prompt family (frozen)

**Evidence pack per video (mirrors `src/utils/generate_VideoMLLM_embedding_HF.py`'s loader):**
8 uniformly-sampled RGB frames passed as a video, plus the title and the ASR transcript formatted
exactly as the banked extractor does:
`"\nTitle: " + (title or "(none)") + "\nTranscript: " + (transcript or "(none)")`.

**Generation instruction (verbatim, the ONLY prompt; greedy, `max_new_tokens=256`):**

> You are an expert analyst of implicit and coded hateful content in short videos. You are shown
> several frames sampled from a video, its title, and an automatic speech transcript. Write ONE
> dense analytical paragraph (about 150-220 words) that decodes the IMPLICIT and CODED signals a
> casual viewer would miss. Reason about: coded language, dog-whistles, slurs-by-allusion,
> euphemisms and in-group jargon; the real-world meaning of any symbols, flags, insignia, hand
> gestures, memes, or numeric codes that appear; and how the scene, setting, or behaviour reframes
> otherwise-neutral words. Use world knowledge to explain what these things REFERENCE and why they
> could signal hostility. Constraints: do NOT make "naming the targeted group/category" your main
> content — decode mechanisms, not labels; do NOT restate, quote, or paraphrase the transcript; do
> NOT transcribe or read out any on-screen text. If nothing implicit or coded is present, say so
> and briefly explain why the content reads as benign. Output only the paragraph.

This prompt: (i) elicits world-knowledge / implicit-reasoning content per LITERATURE §1-C3;
(ii) forbids target-category restating; (iii) forbids ASR restating; (iv) forbids OCR/on-screen-text
reading (honours the user veto and the C3 spec). It is a single family; no variants.

**Decode:** greedy (`do_sample=False, num_beams=1`), `max_new_tokens=256`, Qwen2.5-VL-7B-Instruct
local (`Qwen/Qwen2.5-VL-7B-Instruct`, bf16, sdpa, `max_pixels=360*420`, `num_frames=8`) — identical
model/processor construction to the banked extractor.

## 4. Embedding pathway for A_text (the SAME frozen text pathway the pipeline uses)

The pipeline has no separate text encoder — the banked `text_feats` are produced by
`generate_VideoMLLM_embedding_HF.py::_encode(frames, TEXT_INSTRUCTION+title+transcript,
span="response")`: a frozen Qwen2.5-VL forward, mean of the last-layer hidden states over the
trailing assistant-header span, L2-normed → 3584-d. Because this pathway runs the multimodal model,
**embedding needs the GPU, so the embedding step is folded INTO the generation job** (one job does
generate → embed per video; decided before submission).

**A_text construction (identical pathway, content swapped to the generated analysis):** reuse the
extractor's `_encode(..., span="response")` verbatim on the SAME 8 frames with prompt
`TEXT_INSTRUCTION + "\nAnalysis: " + generated_text` (the generated dense analysis takes the content
slot that title+transcript occupied). Title/transcript are **not** re-fed here, so any marginal gain
of A_text over the banked Z is genuinely from the newly generated world-knowledge content, not a
re-encoding of inputs already in Z. Output: 3584-d L2-normed vector, same space/pooling as banked
`text_feats`. Frames are decoded once and reused for both the generation and the embedding forward.

## 5. Corrected-machinery conditional probe (`c3_nontarget_probe.py`, CPU)

Mirrors `scripts/analysis/c3_real_predictor_probe.py` (the corrected machinery adjudicated in
`C3_PROBE_VERDICT_REVIEW.md`), restricted to the sampled ids:

- **Z (frozen features)** = `concat([img_feats, text_feats])` from BOTH banked caches, per cell:
  `data/CLIP_Embedding/{HateMM,MHC}/train_{openai_clip-vit-large-patch14-336_HF,
  Qwen2.5-VL-7B-Instruct_HF}.pt`, subset to the 300 sampled videos. 4 cells = {HateMM,MHC} × {CLIP,Qwen}.
- **Machinery:** standardize **Z alone** (fit on train fold), keep Z at its **Z-only inner-CV-optimal
  C** (`C_GRID={1e-3,1e-2,1e-1,1}`); the auxiliary **block** is appended as a raw, standardized block
  × s (s=50 ⇒ effectively unpenalized — the shared-L2 crush cannot recur). `RepeatedStratifiedKFold`
  5×5; MDL held-out bits (`-log2 p_true`); **example-clustered (per-video) bootstrap B=5000**; Fano
  bits→acc projection; bar **+0.040**.
- **Dimensionality handling for the dense A_text (design decision, calibration-consistent):** the
  s-trick un-penalizes the appended block; un-penalizing a full 3584-dim block on ~240 train rows
  would *overfit*, not *crush*. So A_text enters the SAME un-penalized appending path but as a
  **train-fold PCA block** (PCA fit on the train fold only — unsupervised, leak-free — then the k
  scores standardized and appended × s). This makes the label-oracle (a 2-col one-hot block) and
  A_text use the **identical appending machinery**, differing only in the block, so the label-oracle
  calibration directly validates the A_text path. k is **swept over {8,16,32,64}**; the gate reads the
  **best-k per cell** (a gate is permissive — give the signal its fairest un-overfit chance).
  **Secondary robustness read:** full-3584-dim A_text under a combined inner-CV-tuned C (standard
  capacity-matched probe); reported but not the primary gate metric.
- **Arms (per cell):**
  1. `baseline` — g(Z) [Z-only reference].
  2. `text_pca_k{8,16,32,64}` — g'([Z, PCA_k(A_text)×s]) [the C3 non-target signal; best-k = gate].
  3. `label_oracle` — g'([Z, onehot(gold label)×s]) [**CALIBRATION — MUST hit accZA ≈ 1.0**].
  4. `shuffled_text` — g'([Z, PCA_k(A_text) row-permuted ×s]) [null control; expect ~0].
  5. `text_full_cvC` — g'([Z_std, A_text_std], combined CV-tuned C) [secondary full-dim robustness].
- **Seeds:** `rng=20260714`, shuffle-seed `12345` (same as prior probes).

## 6. PRE-DECLARED DECISION RULE (frozen)

Let calibration = the label-oracle arm reaching **accZA ≥ 0.99** (~full Fano headroom) on all 4 cells.

- **If calibration FAILS on any cell → `MACHINERY_INVALID`** (no C3 verdict may stand; the crush/overfit
  pathology must be fixed and re-pre-registered).
- **Else**, evaluate the C3 non-target text channel (`text_pca` best-k per cell), under BOTH admissible
  readings of "projected Δacc" — direct held-out Δacc and the Fano bits→acc projection:
  - **PROCEED (`C3_NONTARGET_PROCEED`)** — text channel advances toward a prereg — **iff** on **≥1
    dataset** (best cell of the two encoders) the projected Δacc **point estimate ≥ +0.040** **AND**
    its 95% CI **lower bound > 0**.
  - **Otherwise → `C3_NONTARGET_DEAD_AT_G0`.** The non-target content channel is closed at G0; per the
    anti-gate-hacking rules, revival requires a new pre-registration.

The null (`shuffled_text`) must sit at ~0; if the real `text_pca` cannot beat the shuffled floor it
cannot count. The secondary `text_full_cvC` read is corroborating only and does not by itself promote.

## 7. Artifacts & provenance

- Design (this file): `refine-logs/C3_NONTARGET_PILOT_DESIGN.md`.
- Generation+embed: `scripts/analysis/c3_nontarget_gen.py` (resumable per-video, atomic os.replace,
  symlink-tolerant loader imported verbatim from the banked extractor) +
  `scripts/slurm/c3_nontarget_gen.sbatch` (1 GPU / 8 CPU / 48G, no `--time`, single submit).
- Outputs (in-repo): `artifacts/c3_nontarget/<dataset>/text/<id>.json` (generated paragraph, prompt
  sha, ok flag — NO label), `artifacts/c3_nontarget/<dataset>/emb/<id>.npy` (3584-d A_text),
  `artifacts/c3_nontarget/<dataset>_sample300.json` (stratification manifest).
- Probe: `scripts/analysis/c3_nontarget_probe.py` (CPU) → `refine-logs/C3_NONTARGET_PILOT_OUT.json`.
- Record + verdict: `refine-logs/C3_NONTARGET_PILOT_RECORD.md`.
- Gold usage is **PROBE-ONLY** + sampling-only; never in-method, never on val/test. No network on the
  compute node (`HF_HUB_OFFLINE=1`). Not committed (archiver handles commits).

## Required statements

- No performance/accuracy claim on any held-out benchmark; all accuracy/codelength numbers are
  train-only cross-validation used solely to measure conditional information / audit the probe.
- Single generation prompt family, fixed by this document before any probe number was observed.
