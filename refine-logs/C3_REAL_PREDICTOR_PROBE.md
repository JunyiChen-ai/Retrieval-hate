# C3 target channel — REAL-PREDICTOR conditional-information probe

Date: 2026-07-14
Executor = **Claude Opus 4.8** (`claude-opus-4-8`), C3 real-predictor probe executor,
CPU-only, conda `HateVideo`, no GPU / no SLURM / no commits (archiver commits).

**Purpose.** After `refine-logs/C3_PROBE_VERDICT_REVIEW.md` OVERTURNED
`TARGET_CONTENT_CAPPED` — the corrected oracle (perfect, full-coverage) target ceiling
is **+0.0487** on both encoders, above the +0.040 bar — the prescribed next step is to
**decide the C3 target channel on a REAL predictor, not the oracle**. That review's own
nuance (§6): "the oracle ceiling clears the bar → you cannot foreclose the target channel
on the ceiling; decide it on a real predictor," and it does **not** imply a real target C3
will work. This probe supplies that real-predictor read.

**Real predictor.** `data/gt/{HateMM,MHC}/target_pred_qwen7b.json` — Qwen2.5-VL-7B
target/community predictions from the TARC campaign, injected as a 9-way one-hot
(8 named communities + none/unparsed). This is the same low-bandwidth decision-side
target one-hot the oracle used, but produced by a real MLLM instead of gold.

## VERDICT (one line)

**C3_TARGET_DEAD_ON_REAL_PREDICTOR.** The mandatory label-oracle calibration arm reaches
full Fano headroom on all four cells (accZA = **1.0000**, headroom fraction **1.000**), so
the machinery is VALID and the negative read is admissible. The real Qwen-7B predicted-target
arm carries essentially **no conditional information beyond the frozen encoder**: direct
held-out Δacc ∈ {**−0.0019, +0.0040, +0.0015, −0.0073**} and Fano-projected Δacc ∈
{**+0.0094, +0.0077, −0.0111, −0.0136**} across HateMM/MHC × CLIP/Qwen — every point estimate
is **≥ 4× below the +0.040 bar**, and **no cell's Δacc CI lower bound exceeds 0** under either
metric. The pre-declared proceed condition (projected Δacc CI-lower > 0 **and** point ≥ +0.040
on ≥1 dataset) is met **nowhere**. The gold-oracle consistency arm reproduces the review's
**+0.0487** on both HateMM encoders exactly, so the collapse from +0.0487 (oracle) to ≈0 (real)
is the imperfect-predictor gap, not a machinery artifact. **C3 as a family therefore requires a
NON-target content pilot to stay alive** (not designed here — stated only), consistent with the
TARC test-flat result and the review's oracle-marginality nuance.

---

## 1. Prediction-file format & coverage audit

Both files are `{video_stem: {"primary": int, "raw": str}}`, `primary ∈ {−1..7}` with
`−1 == "None"`; the community→code mapping matches the gold `code_dict` exactly
(`Blacks 0, Jews 1, Whites 2, Others 3, LGBTQ 4, Muslims 5, Sexits 6, Asian 7`). All
`primary` values are ints (no unparsed sentinels inside the file). Injection = 9-col one-hot:
cols 0–7 = named communities, col 8 = **none/unparsed** (predicted `−1` **and** any train id
absent from the file both map to col 8).

| dataset | train N | covered by pred | coverage | pred = none(−1) | unparsed/missing | missing-id labels |
|---|---|---|---|---|---|---|
| HateMM | 744 | 744 | **1.0000** | 172 | 0 | — |
| MHC-EN | 549 | 545 | **0.9927** | 200 | 4 | 4 non-hate → col 8 |

**Predicted-target-alone Bayes (majority vote, machinery-independent):**
- HateMM **0.6707** vs base-rate 0.5995 (hate rate 0.4005) and Z-alone 0.8239/0.8384 → real
  target has *some* marginal signal but far below the encoder.
- MHC **0.7031** vs base-rate 0.6940 (hate rate 0.3060) → **barely above base rate**;
  predicted target is near-useless on MHC before any conditioning.

Predicted-primary → [non-hate, hate] cross-tabs (source of the weak marginal signal):
- HateMM: `−1:[161,11] · Blacks:[134,186] · Jews:[64,62] · Whites:[48,19] · Others:[14,5] ·
  LGBTQ:[12,9] · Muslims:[4,5] · Sexits:[4,1] · Asian:[5,0]` (only Blacks/none are informative).
- MHC: `unparsed:[4,0] · −1:[151,49] · Blacks:[16,13] · Jews:[0,2] · Whites:[19,19] ·
  Others:[42,12] · LGBTQ:[86,21] · Muslims:[3,6] · Sexits:[46,44] · Asian:[14,2]`
  (categories split near-randomly on the label → little separating power).

MHC has **no gold `target_map.json`** (only the prediction file), so the gold-oracle
consistency arm is HateMM-only — this is a real-predictor-only read on MHC, as expected.

## 2. Machinery (CORRECTED, per review §3) & calibration gate

Script: `scripts/analysis/c3_real_predictor_probe.py`. Corrected machinery = standardize **Z
alone** (fit on train fold), keep Z at its Z-only CV-optimal C (`C_GRID={1e-3,1e-2,1e-1,1}`),
append **A as raw one-hot × s** with **s=50** (large s ⇒ A effectively unpenalized so the
shared-L2 crush cannot recur; robustness at s=100). Otherwise mirrors
`c3_g0cond_oracle_probe.py`: RepeatedStratifiedKFold **5×5**, MDL held-out bits
(`−log2 p_true`), example-clustered (per-video) bootstrap **B=5000**, Fano bits→acc
projection, bar **+0.040**. Chosen C: HateMM 0.001 (both enc), MHC 0.01 (both enc).

**MANDATORY CALIBRATION ARM (label-oracle → must hit ~full Fano headroom).** All four cells
pass exactly:

| dataset/enc | accZ | label accZA | headroom (1−accZ) | label Δacc | headroom fraction | **PASS** |
|---|---|---|---|---|---|---|
| HateMM/CLIP | 0.8239 | **1.0000** | 0.1761 | +0.1761 | **1.000** | ✔ |
| HateMM/Qwen | 0.8384 | **1.0000** | 0.1616 | +0.1616 | **1.000** | ✔ |
| MHC/CLIP    | 0.7621 | **1.0000** | 0.2379 | +0.2379 | **1.000** | ✔ |
| MHC/Qwen    | 0.8047 | **1.0000** | 0.1953 | +0.1953 | **1.000** | ✔ |

The known-perfect feature is credited its full analytic headroom to the last digit → the
machinery is a faithful ceiling-measurement and **no "signal is capped" negative is a crush
artifact** (the 2026-07-14 REFLECTION §4 mandate is satisfied). Machinery is **VALID**.

## 3. All arms — Δbits/video and projected Δacc with 95% CIs

Δacc(direct) = held-out accuracy gain from appending A; Δbits = MDL bits saved/video;
fano = Fano bits→acc projected gain. Per-video clustered bootstrap B=5000.

### HateMM (train 744)

| enc | arm | accZA | **Δacc(direct) [CI]** | Δbits/vid [CI] | fano Δacc [CI] |
|---|---|---|---|---|---|
| CLIP | label_oracle (calib) | 1.0000 | +0.1761 [+0.1492,+0.2024] | +0.5874 [+0.5408,+0.6359] | +0.1441 [+0.1266,+0.1639] |
| CLIP | **gold_target** (consistency) | 0.8726 | **+0.0487 [+0.0220,+0.0750]** | +0.1648 [+0.1228,+0.2055] | +0.0562 [+0.0416,+0.0712] |
| CLIP | **pred_target (REAL)** | 0.8220 | **−0.0019 [−0.0180,+0.0145]** | +0.0245 [−0.0002,+0.0475] | +0.0094 [−0.0001,+0.0181] |
| CLIP | pred_target s=100 (robust) | — | −0.0035 [−0.0202,+0.0126] | +0.0214 [−0.0037,+0.0462] | +0.0082 [−0.0015,+0.0178] |
| CLIP | shuffled_pred (null) | 0.8113 | −0.0126 [−0.0210,−0.0048] | −0.0066 [−0.0130,−0.0003] | −0.0026 [−0.0052,−0.0001] |
| Qwen | label_oracle (calib) | 1.0000 | +0.1616 [+0.1368,+0.1874] | +0.5370 [+0.4739,+0.6046] | +0.1253 [+0.1037,+0.1510] |
| Qwen | **gold_target** (consistency) | 0.8871 | **+0.0487 [+0.0301,+0.0672]** | +0.1391 [+0.0976,+0.1805] | +0.0446 [+0.0304,+0.0603] |
| Qwen | **pred_target (REAL)** | 0.8425 | **+0.0040 [−0.0081,+0.0161]** | +0.0220 [−0.0015,+0.0456] | +0.0077 [−0.0005,+0.0164] |
| Qwen | pred_target s=100 (robust) | — | +0.0043 [−0.0086,+0.0169] | +0.0185 [−0.0074,+0.0433] | +0.0065 [−0.0027,+0.0157] |
| Qwen | shuffled_pred (null) | 0.8325 | −0.0059 [−0.0126,+0.0005] | −0.0089 [−0.0170,−0.0010] | −0.0032 [−0.0063,−0.0003] |

### MHC-EN (train 549) — no gold target, real-predictor-only

| enc | arm | accZA | **Δacc(direct) [CI]** | Δbits/vid [CI] | fano Δacc [CI] |
|---|---|---|---|---|---|
| CLIP | label_oracle (calib) | 1.0000 | +0.2379 [+0.2051,+0.2714] | +0.6865 [+0.6114,+0.7650] | +0.1837 [+0.1512,+0.2236] |
| CLIP | **pred_target (REAL)** | 0.7636 | **+0.0015 [−0.0102,+0.0135]** | −0.0232 [−0.0452,−0.0015] | −0.0111 [−0.0232,−0.0007] |
| CLIP | pred_target s=100 (robust) | — | +0.0000 [−0.0120,+0.0120] | −0.0247 [−0.0477,−0.0029] | −0.0118 [−0.0248,−0.0013] |
| CLIP | shuffled_pred (null) | 0.7592 | −0.0029 [−0.0138,+0.0077] | −0.0184 [−0.0401,+0.0018] | −0.0087 [−0.0203,+0.0008] |
| Qwen | label_oracle (calib) | 1.0000 | +0.1953 [+0.1661,+0.2251] | +0.6933 [+0.5836,+0.8112] | +0.1869 [+0.1403,+0.2510] |
| Qwen | **pred_target (REAL)** | 0.7974 | **−0.0073 [−0.0164,+0.0015]** | −0.0280 [−0.0456,−0.0094] | −0.0136 [−0.0254,−0.0045] |
| Qwen | pred_target s=100 (robust) | — | −0.0095 [−0.0193,+0.0004] | −0.0323 [−0.0523,−0.0125] | −0.0158 [−0.0295,−0.0058] |
| Qwen | shuffled_pred (null) | 0.8033 | −0.0015 [−0.0146,+0.0113] | −0.0192 [−0.0497,+0.0105] | −0.0092 [−0.0267,+0.0049] |

**Consistency / sanity reads:**
- **Gold-oracle reproduces the review exactly**: +0.0487 on both HateMM encoders (review §4:
  CLIP +0.0487 [+0.0220,+0.0750], Qwen +0.0487 [+0.0298,+0.0685]) → this probe's corrected
  machinery is the same faithful ceiling instrument; the real-vs-oracle gap is genuine.
- **Null control** (shuffled predicted target) is ~0/slightly negative everywhere → appending
  an aligned-but-uninformative one-hot costs a hair of capacity; the real signal must beat this
  floor to count, and on MHC it does **not** (real Δbits ≈ shuffled Δbits, both negative).
- On **MHC the real predictor is anti-informative in codelength** (Δbits CI entirely negative:
  CLIP [−0.0452,−0.0015], Qwen [−0.0456,−0.0094]) — the predicted community adds noise beyond
  the encoder, matching its ≈base-rate stand-alone Bayes (0.703 vs 0.694).

## 4. Decision-rule evaluation (pre-declared)

> Proceed toward prereg ONLY if the real-predictor arm's projected Δacc **CI lower bound > 0**
> **AND** point estimate **≥ +0.040** on **at least one dataset**; else
> **C3_TARGET_DEAD_ON_REAL_PREDICTOR**.

Evaluated under **both** admissible readings of "projected Δacc" (direct held-out Δacc — the
metric the review compared to the +0.040 bar for its +0.0487 headline; and the Fano bits→acc
projection). The rule requires the *best* dataset to clear both sub-conditions.

| cell | Δacc(direct) pt | CI-lower>0? | ≥+0.040? | fano Δacc pt | CI-lower>0? | ≥+0.040? |
|---|---|---|---|---|---|---|
| HateMM/CLIP | −0.0019 | no | no | +0.0094 | no (−0.0001) | no |
| HateMM/Qwen | +0.0040 | no | no | +0.0077 | no (−0.0005) | no |
| MHC/CLIP    | +0.0015 | no | no | −0.0111 | no | no |
| MHC/Qwen    | −0.0073 | no | no | −0.0136 | no | no |

- **Point-estimate condition:** best real-predictor Δacc is **+0.0094** (Fano, HateMM/CLIP) /
  **+0.0040** (direct, HateMM/Qwen) — both **≪ +0.040** (≥ 4× short). FAIL on every cell,
  every metric.
- **CI-lower-bound condition:** **no** cell has a Δacc CI strictly above 0 under either metric
  (the two closest, HateMM Fano, have lower bounds −0.0001 / −0.0005). FAIL everywhere.
- Robustness at s=100 confirms (pred_target Δacc stays within ±0.01 of 0).

Both sub-conditions fail on all four cells → **the proceed condition is met nowhere**.

## 5. VERDICT & consequence

**VERDICT = C3_TARGET_DEAD_ON_REAL_PREDICTOR.**

The calibration gate passes (machinery VALID — this is a real read, not a crush artifact), and
the real Qwen-7B predicted-target channel is dead: at best +0.0094 projected Δacc, at worst
anti-informative (MHC), with no CI clearing 0 and nothing within 4× of the +0.040 bar. This is
fully consistent with (i) the review's explicit nuance that the +0.0487 oracle ceiling — sitting
just above the bar with CI down to +0.022/+0.030 — does **not** imply a real, imperfect,
kNN-delivered target C3 will work; (ii) the TARC campaign's test-flat result on the same target
signal (the 14th pre-registered negative); and (iii) the standing D1 diagnosis
(REFLECTION §2): the target one-hot is killed by **redundancy** — its stand-alone Bayes
(0.671 HateMM / 0.703 MHC) is below the encoder, so once conditioned on Z it adds ~0 bits.

**Consequence for C3 as a family (stated, not designed here):** the target-content variant of
C3 is closed on a real predictor. **C3 as a family now requires a NON-target content pilot**
(dense MLLM world-knowledge / implicit-reasoning text that is *not* target-category and not a
transcript/OCR restatement — per REFLECTION §3 route 2) to stay alive; that pilot must itself
pass this same G0-cond gate (with the mandatory label-oracle calibration arm) before any GPU.
Designing that non-target pilot is out of scope for this probe.

## 6. Provenance / reproduction

- Script (new): `scripts/analysis/c3_real_predictor_probe.py`; results:
  `refine-logs/C3_REAL_PREDICTOR_PROBE_OUT.json`; run log:
  `refine-logs/C3_REAL_PREDICTOR_PROBE_run.log`. conda `HateVideo`, CPU-only, thread-capped
  (OMP/BLAS=4), max_iter=2000, ~3 min. Per-cell checkpoint + auto-resume (the non-SLURM
  process was SIGKILLed once mid-run by the login-node reaper with no traceback; the resume
  wrapper completed all four cells on the next attempt — numbers are deterministic, seed
  20260714 / shuffle 12345).
- Corrected machinery adjudicated in `refine-logs/C3_PROBE_VERDICT_REVIEW.md` (§3 fix,
  `..._diag.py`); gold-oracle consistency target = its §4 numbers (+0.0487 both encoders).
- Data (read-only): `data/CLIP_Embedding/{HateMM,MHC}/train_{openai_clip-vit-large-patch14-336_HF,
  Qwen2.5-VL-7B-Instruct_HF}.pt`; real predictor
  `data/gt/{HateMM,MHC}/target_pred_qwen7b.json`; gold `data/gt/HateMM/target_map.json`.
- Gold usage is **PROBE-ONLY** (gold target + gold label appended as features / probe targets
  and used only to measure conditional information + calibrate the probe; never in-method,
  never on val/test). No GPU/SLURM/network. Not committed (archiver handles commits).

## Required statements
- No performance/accuracy claim on any held-out benchmark; all accuracy/codelength numbers are
  train-only cross-validation used solely to measure conditional information / audit the probe.
- Write scope = this file + `scripts/analysis/c3_real_predictor_probe.py` +
  `refine-logs/C3_REAL_PREDICTOR_PROBE_OUT.json` + `..._run.log`. Not committed.
