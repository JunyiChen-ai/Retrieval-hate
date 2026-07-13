# C3-NONTARGET FUSION probe — G0-cond gate on the BEST banked configuration

Date: 2026-07-14
Executor = **Claude Opus 4.8** (`claude-opus-4-8`, 1M ctx), C3-nontarget FUSION probe executor,
CPU-only, conda `HateVideo`, no GPU/SLURM, no commits (archiver handles commits).

**What this is.** The review-prescribed cheap falsification of the C3-nontarget generated-text channel
on top of the pipeline's **best banked configuration** (`refine-logs/C3_NONTARGET_VERDICT_REVIEW.md` §4,
the MANDATORY prereg endpoint). The pilot (`refine-logs/C3_NONTARGET_PILOT_RECORD.md`) found the text
channel's only signal at MHC-EN × **CLIP** (weak encoder, accZ 0.7307), where it lifted accZA to 0.7840
— still **below** the MHC/Qwen Z-only baseline 0.7980. The verdict review confirmed the effect is real
conditional information but **encoder-redundant**: on the strong encoder (Qwen) the same text was flat
(+0.0040), so the pipeline already banks that content by using Qwen. The binding question left open:
does A_text add anything **on top of the best banked features** (concat(CLIP,Qwen))? **Honest prior ≈ 0.**

**This is a gate probe, not a performance claim.** All accuracy numbers are train-subset cross-validation
used solely to measure conditional information and audit the probe. No held-out benchmark claim is made.

---

## PRE-DECLARED DESIGN (frozen BEFORE any probe number is computed)

### Features
- **Z_best (PRIMARY) = concat(CLIP img+text, Qwen img+text)** = 8960-d (1024+768+3584+3584), both
  banked caches subset to the same 300 pilot ids in the same order. This is the pipeline's best banked
  configuration (concat of both encoders' pooled features).
- **Z = Qwen-alone (SECONDARY context) = concat(Qwen img+text)** = 7168-d. Reported for context (it is
  the strong-encoder baseline the review says the CLIP effect fails to cross).
- **A_text** = the 3584-d generated-text embeddings from generation job 13101
  (`artifacts/c3_nontarget/<ds>/emb/<id>.npy`, L2-normed, same frozen text pathway as banked
  `text_feats`). Identical artifacts the pilot and the verdict review consumed.

### Machinery (corrected, per `C3_PROBE_VERDICT_REVIEW.md` + the verdict review's permutation methodology)
- Z standardized ALONE at its Z-only inner-CV-optimal `C_Z` (grid {0.001,0.01,0.1,1.0},
  StratifiedKFold rs=0). Auxiliary block appended standardized × s=50 (effectively **un-penalized**,
  refit at the same `C_Z`) so a shared heavy L2 cannot crush the aux columns.
- **Text arms (decision family):** `text_pca_k8`, `text_pca_k16` — train-fold PCA of A_text (fit on the
  train fold only → leak-free), k sliced from a single kmax PCA, standardized × s=50, appended, refit at
  `C_Z`. (k32/k64 point estimates reported as context; they degraded in the pilot and are NOT in the
  decision family.)
- **Full-dim capacity-matched secondary arm:** `text_full_cvC` = [Z_std, A_text_std] under a combined
  inner-CV-tuned C (mirror of the pilot's secondary arm).
- 5×5 RepeatedStratifiedKFold (random_state 1000+rep), per-video correctness averaged over reps.
  Example-clustered (per-video) bootstrap B=5000 on Δacc = cor_ZA − cor_Z (each row = one video, so the
  per-video clustered bootstrap is a paired row resample over the 300 videos). Bar = **+0.040**.

### MANDATORY calibration (per REFLECTION §4, 2026-07-14 addendum)
- **Label-oracle arm** on every cell: 2-col one-hot of the gold label appended raw × s=50 on the
  identical appending path. Must reach **accZA ≈ 1.0** (headroom fraction ≈ 1.0). Any cell whose oracle
  fails ⇒ **MACHINERY_INVALID** for that cell (the "aux columns crushed by shared L2" pathology), and no
  "signal is capped" negative may be accepted from it. Gold used PROBE-ONLY.

### MANDATORY permutation null (as a DISTRIBUTION, not a single seed)
- On the decision cell (and the HateMM no-harm cell), the null is measured as a distribution over
  **≥100 fresh permutations** of A_text across videos (seeds `default_rng(70000+si)`). Report the
  per-k null Δacc **mean / SD / max / [2.5,50,97.5] quantiles**, and the **max-over-k** null distribution
  (max over the decision family {k8,k16} per permutation) for the family (selection) correction. A single
  shuffle seed (12345) is reported for continuity with the pilot but is NOT the null.

### PRE-DECLARED DECISION RULE (from the verdict review §4; frozen)
On **Z_best (MHC-EN, the only cell with prior signal)**, C3-nontarget proceeds to prereg **ONLY IF ALL**
of the following hold:
1. **(C1)** the text arm's direct held-out **Δacc point ≥ +0.040** (best of the decision family {k8,k16}), AND
2. **(C2)** its example-clustered bootstrap **CI-lower > 0**, AND
3. **(C3)** the **real max-over-k exceeds ALL permutation maxima** (real max-over-{8,16} > every one of the
   ≥100 permutation max-over-{8,16} draws, i.e. permutation p = 0).

Calibration (label-oracle ≈ full headroom) must pass on the decision cell, else **MACHINERY_INVALID**.

- **PROCEED verdict** (`C3_FUSION_PROCEED`): C1 ∧ C2 ∧ C3 all met on Z_best × MHC-EN.
- **Anything less** ⇒ **`C3_NONTARGET_DEAD_AT_FUSION`** — the 19th pre-registered negative.
- **No-harm context (advisory only):** report HateMM Δacc on Z_best; it does not gate the verdict.

**Honest prior on PROCEED ≈ 0** (the review's stated prior). Run exactly and let the numbers decide.

Script: `scripts/analysis/c3_fusion_probe.py` (CPU, checkpointed + auto-resume against the login-node
reaper). Results: `refine-logs/C3_FUSION_PROBE_OUT.json` + `..._run.log`.

---

## RESULTS

Script `scripts/analysis/c3_fusion_probe.py` (CPU, completed exit 0, elapsed 1856s final leg +
checkpointed earlier legs); results `refine-logs/C3_FUSION_PROBE_OUT.json`; log `..._run.log`.
Point estimates and calibration are deterministic (checkpointed per cell; survived one login-node
reap via auto-resume). Permutation null = **150/150** fresh permutations completed on BOTH Z_best
cells. The script's independently computed verdict block (`OUT.json['verdict']`) matches this record:
`calib_pass=true, C1=false, C2=false, C3=false, VERDICT=C3_NONTARGET_DEAD_AT_FUSION`.

### Calibration (MANDATORY — machinery validity)

| cell | Z_dim | C_Z | accZ | label_accZA | headroom frac | PASS |
|---|---|---|---|---|---|---|
| **MHC / Z_best** | 8960 | 0.001 | 0.7827 | **1.0000** | **1.000** | ✔ |
| MHC / Qwen-alone | 7168 | 1.0 | 0.7980 | **1.0000** | **1.000** | ✔ |
| **HateMM / Z_best** | 8960 | 0.01 | 0.8420 | **1.0000** | **1.000** | ✔ |
| HateMM / Qwen-alone | 7168 | 0.001 | 0.8413 | **1.0000** | **1.000** | ✔ |

→ Label-oracle hits exactly full Fano headroom on every cell (accZA = 1.0000). **Machinery VALID**
(no `MACHINERY_INVALID`); the aux-column-crush pathology that overturned the sibling C3-oracle/SAV
probes is absent here, so a negative read is admissible.

### Text arms — direct held-out Δacc [per-video-clustered 95% CI], on Z_best

**MHC-EN / Z_best (PRIMARY DECISION cell), accZ = 0.7827:**

| arm | accZA | Δacc [CI] |
|---|---|---|
| text_pca_k8  | 0.7947 | **+0.0120 [−0.0187, +0.0427]** |
| text_pca_k16 | 0.7987 | **+0.0160 [−0.0187, +0.0507]** |
| text_pca_k32 (context) | 0.7973 | +0.0147 [−0.0233, +0.0527] |
| text_pca_k64 (context) | 0.7533 | −0.0293 [−0.0680, +0.0113] |
| text_full_cvC (secondary) | 0.7980 | +0.0153 [−0.0093, +0.0413] |
| shuffled (seed 12345), k8 / k16 | — | +0.0153 / +0.0013 |

Best of the decision family {k8,k16} = **k16 +0.0160 [−0.0187,+0.0507]**. Real max-over-{8,16} = **+0.0160**.

The pilot's weak-encoder signal (MHC/CLIP-alone k8 **+0.0533**) **collapses to +0.0120/+0.0160 on the
best banked features** — the text adds essentially nothing above the noise band once the strong Qwen
channel is present. Note the single-seed shuffle floor (+0.0153) is on par with the real arm (+0.0120),
and the secondary full-dim arm (+0.0153) matches: all reads cluster around a ~+0.015 mechanical floor,
none reaching the +0.040 bar, every CI straddling 0.

**HateMM / Z_best (no-harm advisory), accZ = 0.8420:** text_pca_k8 −0.0047 [−0.0213,+0.0113], k16
−0.0087 [−0.0280,+0.0100], full-dim +0.0040 [−0.0113,+0.0200]. Flat-to-slightly-negative — **no harm,
no help** (consistent with D1 redundancy on the dataset where the Qwen encoder swap is strongest).

### Secondary context — Qwen-alone (reproduces the pilot / verdict review)

**MHC-EN / Qwen-alone, accZ = 0.7980:** text_pca_k8 −0.0013 [−0.0260,+0.0240], k16 +0.0040
[−0.0253,+0.0327], full-dim +0.0013 [−0.0260,+0.0293] — flat, reproducing the verdict review's
"+0.0040 on the strong encoder" to the digit. Confirms the effect is encoder-redundant, and confirms
Z_best (0.7827) sits at/below the Qwen-alone baseline (0.7980) — the concat does not itself help here.

### Permutation-null distribution (150 fresh permutations of A_text across videos), on Z_best

**MHC-EN / Z_best (decision cell), n = 150 fresh permutations (seeds `default_rng(70000+si)`):**

| statistic | k8 null | k16 null | max-over-{8,16} (family) |
|---|---|---|---|
| mean | **−0.0031** | **−0.0090** | **−0.0019** |
| SD | 0.0088 | 0.0102 | 0.0091 |
| max | **+0.0207** | **+0.0287** | **+0.0287** |
| quantiles [2.5, 50, 97.5] | [−0.0175, −0.0047, +0.0167] | [−0.0262, −0.0097, +0.0109] | [−0.0162, −0.0027, +0.0169] |

- Null centered at ≈ 0 (slightly negative) — machinery unbiased, no free-column floor; consistent with
  the verdict review's 150-perm finding on CLIP-alone.
- **p(perm max-over-k ≥ real max +0.0160) = 0.047 (7/150)** → the real arm does **NOT** exceed all
  permutation maxima (perm-family max = +0.0287 > real +0.0160). Per-k: p(perm_k8 ≥ +0.0120) = 0.080,
  p(perm_k16 ≥ +0.0160) = 0.013.
- **p(perm max-over-k ≥ +0.040 bar) = 0.000 (0/150)** — the machinery never manufactures a
  bar-clearing gain under the null, so a true +0.040 signal would have been detectable. The real
  +0.0160 sits at only the ~95th percentile of the family null — marginal at best, an order of
  magnitude weaker than the pilot's ≈6σ CLIP-alone read (which had permutation p < 0.007).
- The pilot-continuity single shuffle (seed 12345, k8 = +0.0153) is again an upper-tail draw of this
  distribution (~96th pct of the k8 null) — reconfirming the review's diagnosis of that unlucky seed.

**HateMM / Z_best (no-harm advisory), n = 150:** k8 null mean −0.0014 (SD 0.0063, max +0.0200);
max-over-{8,16} mean −0.0009 (max +0.0200). Real arms are negative (−0.0047/−0.0087; real
max-over-k = −0.0047, sitting at the ~70th-percentile-from-top of the null, p(perm ≥ real) = 0.700 —
i.e. indistinguishable from permuted text), so no permutation question arises — the advisory cell is
no-harm/no-help.

**Qwen-alone context cells (final file):** HateMM/Qwen accZ 0.8413, k8 +0.0000 [−0.0193,+0.0180],
k16 −0.0033 [−0.0253,+0.0187], full-dim +0.0027 [−0.0107,+0.0167] — reproduces the pilot's
HateMM/Qwen cell (flat), as expected.

### Decision-rule evaluation (Z_best × MHC-EN)

| condition | requirement | observed | met? |
|---|---|---|---|
| **C1** | best-decision-k Δacc point ≥ +0.040 | k16 = **+0.0160** | **✗ NO** |
| **C2** | its bootstrap CI-lower > 0 | CI-low = **−0.0187** | **✗ NO** |
| **C3** | real max-over-k > all perm maxima | real +0.0160 < perm max +0.0287 (p = 0.047) | **✗ NO** |
| calibration | label-oracle ≈ full headroom | accZA 1.0000 | ✔ YES |

**All three conditions fail independently.** C1: point +0.0160 is <½ the bar. C2: CI-lower −0.0187
well below 0. C3: 7 of 150 permutation family-maxima exceed the real max (the real arm is not even
individually significant against the permutation null at the family level). The rule requires all
three; the failure is triple, not marginal.

## VERDICT

**C3_NONTARGET_DEAD_AT_FUSION** — the 19th pre-registered negative.

On the best banked configuration Z_best = concat(CLIP,Qwen), the C3-nontarget generated-text channel
does **not** clear the +0.040 conditional-information bar: best decision-family read +0.0160
[−0.0187,+0.0507] (C1 ✗, C2 ✗), and it does not separate from the 150-permutation family null
(p = 0.047, 7 permutation maxima above it; C3 ✗). The pilot's +0.0533 was a **weak-CLIP-encoder**
effect the verdict review had already diagnosed as encoder-redundant; on Qwen-alone the text is flat
(+0.0040, reproduced to the digit) and on the CLIP+Qwen fusion it collapses to noise-band level.
Calibration passed on all four cells (label-oracle accZA = 1.0000 → machinery VALID), and the
permutation null shows the machinery cannot manufacture a bar-clearing gain (0/150 ≥ +0.040) — so
this is a genuine null, not a capped-signal artifact. No-harm context: HateMM Z_best Δacc is
flat/slightly negative (no degradation, no help). **The prereg does not proceed. Honest prior (≈ 0)
confirmed. Do not re-propose the C3-nontarget text channel as a decision-side or feature-append
signal at 7B; the only surviving MLLM text role remains the encoder pathway itself.**

## Provenance / reproduction

- Prescription: `refine-logs/C3_NONTARGET_VERDICT_REVIEW.md` §4. Pilot record:
  `refine-logs/C3_NONTARGET_PILOT_RECORD.md`. Permutation methodology reused verbatim from
  `refine-logs/c3nt_verdict_review_diag.py`.
- Probe: `scripts/analysis/c3_fusion_probe.py` (CPU-only, conda `HateVideo`, checkpoint-per-cell +
  auto-resume; NSEED=150) → `refine-logs/C3_FUSION_PROBE_OUT.json` + `refine-logs/C3_FUSION_PROBE_run.log`.
- Data (read-only): `data/CLIP_Embedding/{MHC,HateMM}/train_{openai_clip-vit-large-patch14-336_HF,
  Qwen2.5-VL-7B-Instruct_HF}.pt`; `artifacts/c3_nontarget/{MHC,HateMM}_sample300.json` +
  `{MHC,HateMM}/emb/*.npy` (job 13101 artifacts, untouched). Label-order equality between both encoder
  caches and the sample manifests asserted at load (300/300, no missing ids, MHC 92 pos / HateMM 120 pos).
- Seeds: CV folds rs=1000+rep (identical to pilot); permutations `default_rng(70000+si)` (identical
  family to the verdict-review diag); bootstrap B=5000, seeds 20260714+k; continuity shuffle 12345.
- Gold labels used PROBE-ONLY (stratification recorded in-manifest by the pilot; oracle arm + targets
  here); no validation/test content touched.

## Required statements

- No performance/accuracy claim on any held-out benchmark; all accuracy numbers are train-subset
  cross-validation used solely to measure conditional information and audit the probe.
- Write scope = this file + `scripts/analysis/c3_fusion_probe.py` +
  `refine-logs/C3_FUSION_PROBE_OUT.json` + `refine-logs/C3_FUSION_PROBE_run.log`. Not committed
  (archiver handles commits). No prereg / config / CLAUDE.md / settings mutated. No SLURM jobs
  submitted; zero GPU consumed.
