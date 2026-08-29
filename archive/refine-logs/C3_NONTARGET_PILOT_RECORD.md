# C3 NON-TARGET content pilot — G0-cond gate RECORD

Date: 2026-07-14
Executor = **Claude Opus 4.8** (`claude-opus-4-8`), C3-nontarget G0 pilot executor.
Pre-registration: `refine-logs/C3_NONTARGET_PILOT_DESIGN.md` (frozen BEFORE any probe number;
single prompt family sha `f9a941611409`; no prompt iteration occurred).

## VERDICT (one line)

**C3_NONTARGET_PROCEED** — the mandatory label-oracle calibration arm hits **exactly full Fano
headroom on all four cells** (accZA = 1.0000, headroom fraction 1.000 → machinery VALID), and the
pre-declared proceed condition (projected Δacc point ≥ +0.040 **and** CI-lower > 0 on ≥1 dataset)
is **met on MHC-EN/CLIP**: `text_pca_k8` direct Δacc **+0.0533 [+0.0173, +0.0900]**, independently
corroborated at k16 (**+0.0440 [+0.0040, +0.0847]**) and by the secondary full-dim arm
(**+0.0440 [+0.0020, +0.0847]**). HateMM is flat/negative everywhere and MHC/Qwen is flat — the
effect is a **single-cell (weak-encoder) effect**, and three fragility caveats below are BINDING
context for any follow-up prereg. This is a gate pass, not a performance claim.

## 1. Generation job (provenance)

- Job **13101** (`c3nt_gen`), single submission, 1×A100-80GB / 8 CPU / 48G, **COMPLETED**
  ExitCode 0:0, Elapsed **01:25:12** (start 2026-07-14T04:56:02+12:00, done 06:21:13+12:00).
  Log: `slurm/logs/c3nt_gen_13101.out`. No resubmission needed.
- Artifacts: **600/600** (300 HateMM + 300 MHC-EN): `artifacts/c3_nontarget/<ds>/text/<id>.json`
  + `<ds>/emb/<id>.npy` + `<ds>_sample300.json` manifests (stratified by label — labels used for
  SAMPLING ONLY, recorded in-manifest; HateMM 120 hate/180 non-hate, MHC 92/208, seed 20260714).
- Model Qwen2.5-VL-7B-Instruct local, bf16, greedy, max_new_tokens=256, 8 frames + title + ASR
  (evidence pack mirrors `src/utils/generate_VideoMLLM_embedding_HF.py`; loader imported verbatim,
  symlink-tolerant). One HateMM video failed decode → zero-vector guard (`ok=false`), MHC zero.
- Text quality audit (599 ok texts): **all unique**; ASR 4-gram overlap mean 0.013 (HateMM) /
  0.015 (MHC), max 0.10/0.13 → the no-transcript-restating constraint held; explicit-benign
  statements 5/11; mean length ~180 words. Spot-read confirms on-spec content (decodes coded
  language/slur-by-allusion/scene context; no target-category naming as main content; no OCR).
- A_text = the SAME frozen text pathway as banked `text_feats` (`_encode(..., span="response")`
  reused verbatim; 3584-d L2-normed), embedding folded into the generation job (decided pre-submit).

## 2. Probe (corrected machinery) & calibration gate

Script `scripts/analysis/c3_nontarget_probe.py` (CPU, ~75 min, exit 0); results
`refine-logs/C3_NONTARGET_PILOT_OUT.json`; log `refine-logs/C3_NONTARGET_PILOT_run.log`.
Machinery per design §5: Z = concat(img,text) banked feats subset to the 300 ids, standardized
alone at its Z-only CV-optimal C; auxiliary block appended un-crushed (raw/std × s=50); A_text via
train-fold PCA k∈{8,16,32,64}; label-oracle = 2-col one-hot on the identical appending path;
shuffled-A null; 5×5 RepeatedStratifiedKFold; per-video clustered bootstrap B=5000; bar +0.040.

**MANDATORY CALIBRATION — PASS on all four cells (machinery VALID):**

| cell | C_Z | accZ | label accZA | label Δacc | headroom frac |
|---|---|---|---|---|---|
| HateMM/CLIP | 0.001 | 0.8353 | **1.0000** | +0.1647 | **1.000** |
| HateMM/Qwen | 0.001 | 0.8413 | **1.0000** | +0.1587 | **1.000** |
| MHC/CLIP | 1.0 | 0.7307 | **1.0000** | +0.2693 | **1.000** |
| MHC/Qwen | 1.0 | 0.7980 | **1.0000** | +0.2020 | **1.000** |

**Fano-metric validity note:** on both MHC cells the baseline codelength bits_Z > 1.0
(1.1374 / 1.1259), so the binary-entropy Fano ceiling is pinned at 0.5 and the Fano projection is
**saturated/uninformative on MHC** (label-oracle fano = +0.5000 exactly). The **direct held-out
Δacc** is therefore the operative metric on MHC; on HateMM (bits_Z 0.61/0.55) both metrics operate.

## 3. All arms — direct Δacc [95% CI] (the operative metric), Δbits, Fano

### HateMM (300; zero_Atext=1)

| cell | arm | accZA | Δacc [CI] | Δbits [CI] | fano [CI] |
|---|---|---|---|---|---|
| CLIP | text_pca_k8 | 0.8307 | −0.0047 [−0.0427,+0.0327] | +0.0270 [−0.0382,+0.0894] | +0.0106 [−0.0163,+0.0332] |
| CLIP | text_pca_k16 | 0.8333 | −0.0020 [−0.0367,+0.0327] | +0.0184 [−0.0507,+0.0834] | +0.0073 [−0.0218,+0.0312] |
| CLIP | text_pca_k32 | 0.8227 | −0.0127 [−0.0500,+0.0247] | −0.0255 [−0.1145,+0.0596] | −0.0105 [−0.0536,+0.0218] |
| CLIP | text_pca_k64 | 0.7887 | −0.0467 [−0.0847,−0.0080] | −0.2242 [−0.3801,−0.0852] | −0.1154 [−0.3283,−0.0353] |
| CLIP | shuffled_k8 (null) | 0.8193 | −0.0160 [−0.0347,+0.0020] | −0.0093 [−0.0289,+0.0098] | −0.0038 [−0.0123,+0.0039] |
| CLIP | text_full_cvC (2nd) | 0.8260 | −0.0093 [−0.0413,+0.0213] | +0.0491 [−0.0027,+0.0982] | +0.0188 [−0.0011,+0.0364] |
| Qwen | text_pca_k8 | 0.8413 | +0.0000 [−0.0187,+0.0193] | −0.0365 [−0.0769,+0.0013] | −0.0136 [−0.0323,+0.0005] |
| Qwen | text_pca_k16 | 0.8380 | −0.0033 [−0.0247,+0.0180] | −0.0595 [−0.1069,−0.0155] | −0.0225 [−0.0473,−0.0055] |
| Qwen | text_pca_k32 | 0.8320 | −0.0093 [−0.0353,+0.0160] | −0.0779 [−0.1434,−0.0194] | −0.0300 [−0.0652,−0.0068] |
| Qwen | text_pca_k64 | 0.7960 | −0.0453 [−0.0747,−0.0173] | −0.2688 [−0.3964,−0.1556] | −0.1273 [−0.3245,−0.0599] |
| Qwen | shuffled_k8 (null) | 0.8393 | −0.0020 [−0.0207,+0.0160] | −0.0062 [−0.0350,+0.0212] | −0.0022 [−0.0130,+0.0078] |
| Qwen | text_full_cvC (2nd) | 0.8440 | +0.0027 [−0.0100,+0.0167] | −0.0083 [−0.0354,+0.0182] | −0.0030 [−0.0139,+0.0065] |

→ **HateMM: no cell passes anything.** The dense text channel is flat-to-harmful on the dataset
where the Qwen encoder swap is strongest — consistent with D1 redundancy (the frozen features
already carry this content there).

### MHC-EN (300; zero_Atext=0) — Fano saturated, direct Δacc operative

| cell | arm | accZA | Δacc [CI] | Δbits [CI] |
|---|---|---|---|---|
| CLIP | **text_pca_k8** | 0.7840 | **+0.0533 [+0.0173,+0.0900]** | +0.0482 [−0.1414,+0.2467] |
| CLIP | **text_pca_k16** | 0.7747 | **+0.0440 [+0.0040,+0.0847]** | −0.0481 [−0.2608,+0.1737] |
| CLIP | text_pca_k32 | 0.7687 | +0.0380 [−0.0047,+0.0807] | −0.2195 [−0.5031,+0.0400] |
| CLIP | text_pca_k64 | 0.7413 | +0.0107 [−0.0347,+0.0560] | −1.1486 [−1.6439,−0.7092] |
| CLIP | shuffled_k8 (null) | 0.7533 | **+0.0227 [+0.0020,+0.0447]** | −0.0603 [−0.1847,+0.0626] |
| CLIP | shuffled_k16 (null) | 0.7380 | +0.0073 [−0.0187,+0.0333] | −0.1002 [−0.2546,+0.0392] |
| CLIP | **text_full_cvC (2nd)** | 0.7747 | **+0.0440 [+0.0020,+0.0847]** | +0.4963 [+0.3089,+0.6890]* |
| Qwen | text_pca_k8 | 0.7967 | −0.0013 [−0.0273,+0.0233] | −0.0170 [−0.1584,+0.1227] |
| Qwen | text_pca_k16 | 0.8020 | +0.0040 [−0.0253,+0.0333] | −0.0257 [−0.1945,+0.1518] |
| Qwen | text_pca_k32 | 0.7960 | −0.0020 [−0.0347,+0.0307] | −0.1164 [−0.3510,+0.1072] |
| Qwen | text_pca_k64 | 0.7527 | −0.0453 [−0.0833,−0.0073] | −0.9325 [−1.3557,−0.5160] |
| Qwen | shuffled_k8 (null) | 0.7933 | −0.0047 [−0.0227,+0.0140] | −0.0493 [−0.1675,+0.0647] |
| Qwen | text_full_cvC (2nd) | 0.7993 | +0.0013 [−0.0247,+0.0273] | +0.4438 [+0.2756,+0.6244]* |

\* the full-dim arm's Δbits on MHC is **confounded** (baseline at C_Z=1.0 has inflated held-out
codelength vs the full arm's C_full=0.001/0.01 — different regularization, not conditional
information; MHC/Qwen shows the same +0.44 Δbits with Δacc ≈ 0, proving the inflation). The full
arm's **Δacc** is the honest secondary read.

## 4. Pre-declared decision rule — evaluation

> PROCEED iff projected Δacc point ≥ +0.040 AND 95% CI-lower > 0 on ≥1 dataset; calibration must
> pass everywhere; the real arm must beat the shuffled floor.

- **Calibration:** PASS 4/4 (exactly full headroom) → no `MACHINERY_INVALID`.
- **Proceed condition:** MET on **MHC-EN via CLIP**: k8 **+0.0533 [+0.0173,+0.0900]** (point ≥
  +0.040 ✔, CI-lower > 0 ✔); k16 +0.0440 [+0.0040,+0.0847] also passes independently; secondary
  full-dim +0.0440 [+0.0020,+0.0847] corroborates (not a single-k artifact).
- **Null clause:** the MHC/CLIP shuffled-k8 null is **not ~0** (+0.0227 [+0.0020,+0.0447]) — 1 of
  16 null cells excludes 0 (≈ the multiple-testing expectation, but it sits at the trigger cell).
  The real arm **beats the shuffled floor point-wise** (+0.0533 > +0.0227), satisfying the design's
  literal clause; the **paired** real−shuffled contrast (same folds, per-video, B=5000;
  `refine-logs/C3_NONTARGET_PILOT_nullcheck.json`, deterministic rerun reproduced OUT.json to 1e-9):
  k8 **+0.0307 [−0.0087,+0.0713]**, k16 **+0.0367 [−0.0067,+0.0807]** — positive but **not
  individually significant**. Recorded as fragility caveat F2, not a veto (the frozen rule's veto
  reading is point-wise).

**VERDICT = C3_NONTARGET_PROCEED** (gate role only; no benchmark claim).

## 5. BINDING fragility caveats for the follow-up prereg

- **F1 — single-cell, weak-encoder effect.** The signal exists ONLY at MHC-EN×CLIP (accZ 0.7307);
  MHC×Qwen is flat (best +0.0040) and HateMM is flat/negative on both encoders. Crucially,
  MHC/CLIP+text accZA = **0.7840 < 0.7980 = MHC/Qwen baseline**: on this probe the text channel
  lifts the weak encoder toward, but not past, what the stronger banked encoder already achieves.
  A prereg must therefore test A_text **on top of the pipeline's best configuration** (or as a
  CLIP+Qwen+text fusion) — a gain that only reproduces the Qwen floor by another route has no
  main-table value.
- **F2 — null floor at the trigger cell.** Shuffled-noise columns alone gain +0.0227 there (probable
  variance-reduction/regularization side-effect of appending unpenalized columns at C_Z=1.0);
  real−shuffled = +0.031/+0.037, CI includes 0. The conditional-information component net of the
  mechanical floor is likely below the +0.040 bar on its own.
- **F3 — codelength does not corroborate.** At the trigger cell Δbits ≈ +0.05 [−0.14,+0.25]
  (straddles 0); the Δacc gain is threshold-level, not likelihood-level. Combined with the MHC Fano
  saturation, the MDL leg of the gate is inconclusive on MHC (neither confirms nor refutes).
- Sample = 300/dataset (gate power, not experiment power); k selected as best-of-4 per the frozen
  permissive-gate design (k8/k16/full-dim agreement mitigates but does not eliminate selection).

## 6. Anti-gate-hacking compliance

Single prompt family (sha `f9a941611409` on all 600 artifacts), fixed in the design doc before any
probe number; zero prompt iteration; sample manifests drawn once (seed 20260714) and never redrawn;
decision rule evaluated exactly as frozen. The §4 null-clause paired contrast is a post-hoc
*machinery-health* check demanded by the design's own text, not a rule change.

## 7. Provenance / reproduction

- Design: `refine-logs/C3_NONTARGET_PILOT_DESIGN.md`. Generation:
  `scripts/analysis/c3_nontarget_gen.py` + `scripts/slurm/c3_nontarget_gen.sbatch` (job 13101).
- Probe: `scripts/analysis/c3_nontarget_probe.py` → `refine-logs/C3_NONTARGET_PILOT_OUT.json` +
  `..._run.log`; null check: `refine-logs/C3_NONTARGET_PILOT_nullcheck.json`.
- Data (read-only): `data/CLIP_Embedding/{HateMM,MHC}/train_{openai_clip-vit-large-patch14-336_HF,
  Qwen2.5-VL-7B-Instruct_HF}.pt`; `data/gt/{HateMM,MHC}/train.jsonl`; videos via
  `data/video/<ds>/All/*.mp4` symlinks. Seeds: rng 20260714, shuffle 12345, CV folds 1000+rep.
- Gold usage: labels for stratified SAMPLING (recorded) + PROBE-ONLY (calibration arm/targets);
  never in the generation prompt, never in-method, never on val/test. `HF_HUB_OFFLINE=1` on the
  compute node. Not committed (archiver handles commits).

## Required statements

- No performance/accuracy claim on any held-out benchmark; all accuracy/codelength numbers are
  train-subset cross-validation used solely to measure conditional information / audit the probe.
- Write scope = this file + design doc + `scripts/analysis/c3_nontarget_{gen,probe}.py` +
  `scripts/slurm/c3_nontarget_gen.sbatch` + `refine-logs/C3_NONTARGET_PILOT_{OUT,nullcheck}.json` +
  `..._run.log` + `artifacts/c3_nontarget/**` + `scripts/analysis/.c3nt_gen_jid`. Not committed.
