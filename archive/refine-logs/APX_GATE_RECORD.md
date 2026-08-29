# APX G0-cond audio gate — Wave-3 candidate #3 (NON-BINDING $0 pre-check)

**Date:** 2026-07-16. **Executor:** CTF/APX gate executor (CPU-only, conda `HateVideo`, no GPU).
**Status of this record:** **NON-BINDING prior-mover / cheap-kill screen** (pre-ceremony; precedent = the
W2-C CLIP-K4 pre-check, `ad48dcc`). No prereg freeze; full **raw-only** transcription discipline (every
number below copied verbatim from `refine-logs/APX_G0COND_GATE_OUT.json`).
**Design followed VERBATIM:** `refine-logs/WAVE3_CANDIDATES.md` CANDIDATE 3 sections (d)/(e) (commit `0ee06df`).

---

## 0. Headline

**KILL-side (clean).** The whole-video openSMILE **eGeMAPSv02** prosodic embedding (88-d: pitch, loudness,
jitter/shimmer, spectral) carries **no** conditional label information over `Z_best = concat(CLIP img+text,
Qwen img+text)` (8960-d) on HateMM. Best decision-family point estimate **−0.0038** (k8, CI `[-0.0113,+0.0033]`);
even the full 88-d capacity-matched arm is **+0.0005** (CI `[-0.0031,+0.0042]`) — essentially exactly zero,
an order of magnitude under the +0.040 bar with CI straddling 0. Calibration (K-APX-1) is **VALID**
(label-oracle accZA = 1.0000, headroom-fraction 1.000) → genuine null, not a machinery artifact.
**K-APX-0 fires → the acoustic axis prior is slashed at ~$0, NO audio-encoder download escalation.**
(This also de-risks/gates candidate #4 AVC, which sits behind APX's audio gate.)

---

## 1. Provenance

| item | value |
|---|---|
| probe script | `scripts/analysis/apx_g0cond_gate.py` sha256 **`c338de8cea7198168ec7c9cc96f9c9558667939d3aaf401c559ee349bcc7b5bd`** |
| extractor script | `scripts/analysis/apx_extract_egemaps.py` sha256 **`6f25d8b98177798541251cb805cbb8b5a677bb241fa9ba15c62b30c0b099aacb`** |
| extraction sbatch | `scripts/slurm/apx_egemaps_extract.sbatch` sha256 **`57dae37eb6013411e62fe3e5ab13d575568665cd446372186df84f9f0dfe5032`** |
| machinery source (reused VERBATIM) | `scripts/analysis/c3_fusion_probe.py` sha256 `9091e2c3443d4826144f820217e37d43d26d282d334b0b35bea7cb4ae9748b3c` |
| feature extractor | openSMILE **2.6.0** (python `opensmile`, bundled SMILExtract binary; **no model download**), feature set **eGeMAPSv02**, level **Functionals**, **88-d** |
| audio decode | ffmpeg `9c33b2f` (`-vn -ac 1 -ar 16000`, 16 kHz mono wav); read via `soundfile`; `opensmile.Smile.process_signal` |
| eGeMAPS cache | `data/audio/HateMM/egemaps_v02_trainval.pt` sha256 **`0bcb11cd5a552c5ea4c178e0bb4d969edc8a52d20d3b911a523703b670ff8944`** (gitignored derived data; per-id shards under `.../egemaps_v02/<id>.npy`) |
| extraction job | SLURM **13203** COMPLETED 0:0, elapsed 00:08:53, CPU-only (no `--gres`), 8-way parallel; log `slurm/logs/apx_egemaps_13203.log`; manifest `data/audio/HateMM/egemaps_v02_manifest.json` (`status={ok:851}`, `n_zero_rows=0`, no NaN) |
| probe run | `conda run -n HateVideo python3 scripts/analysis/apx_g0cond_gate.py`, **LOCAL login-node CPU** (not Modal), no GPU, elapsed **154 s**; outputs `refine-logs/APX_G0COND_GATE_OUT.json` + `..._run.log` |
| env | conda `HateVideo`: sklearn 1.5.2, numpy 1.26.4, torch 2.6.0+cu124 |

### 1.1 Infrastructure failure + fix (documented per instruction; NO scientific spend)

The first extraction submission **job 13200 FAILED** (exit 1:0, 3 s). Root cause (from
`slurm/logs/apx_egemaps_13200.log`): the sbatch **prepended** the ExMRD env's `bin` to `PATH` to obtain
`ffmpeg`, which **shadowed HateVideo's `python`** → bare `python` resolved to `ExMRD/bin/python` (no
`opensmile`) → `ModuleNotFoundError: No module named 'opensmile'`. The conda activation itself was correct
(`source .../conda.sh && conda activate HateVideo`), i.e. this was **not** the `source activate` no-op trap.
**Fix** (sbatch → sha256 `57dae37e…`): (i) **append** ExMRD to `PATH` instead of prepend (ffmpeg findable,
conda python un-shadowed); (ii) call the HateVideo interpreter by **absolute path**
`/data/jehc223/miniconda3/envs/HateVideo/bin/python`; (iii) fail-fast preflight
`python -c "import opensmile, soundfile"`. Resubmitted as **13203** (above), COMPLETED clean. Job 13200
produced **no numbers** (crashed before any feature was computed) → CPU-only resubmission, no scientific spend.

### 1.2 Baseline `Z_best` and scope

Baseline caches (features-only, read-only, train + dev_seen): `data/CLIP_Embedding/HateMM/{train,dev_seen}_
{openai_clip-vit-large-patch14-336_HF, Qwen2.5-VL-7B-Instruct_HF}.pt` → `Z_best` = CLIP img 1024 + CLIP text
768 + Qwen img 3584 + Qwen text 3584 = **8960-d** (dims + per-video id/label agreement asserted at load).
**Scope: HateMM only** (design is HateMM-primary — "run the C3-template conditional-info of eGeMAPS over
`Z_best` **on HateMM**"; MHC-EN is data/label-limited per WAVE3 §0, not the acoustic anchor). N = 744+107 =
**851** train∪val (341 pos). **Zero test-touch**: `test_seen` videos and caches never enumerated/opened.

---

## 2. Machinery (reused VERBATIM from `c3_fusion_probe.py`)

Identical to the CTF gate and the C3-template: Z standardized ALONE at its Z-only inner-CV-optimal `C_Z`
(grid {0.001,0.01,0.1,1.0}, `rs=0`); aux block appended standardized × **s=50** (un-penalized, refit at `C_Z`);
aux via train-fold PCA (leak-free); **5×5 RepeatedStratifiedKFold** (rs=1000+rep); per-video correctness
averaged; per-video-clustered **bootstrap B=5000** on Δacc; **bar +0.040**; mandatory **label-oracle
calibration** (accZA ≥ 0.99 or MACHINERY_INVALID); ≥150-permutation null available only to confirm a
would-be pass. Aux arms: `audio_pca_k{8,16}` (decision family + max-over-k), `audio_pca_k{32,64}` (context),
`audio_full_cvC` (full 88-d capacity-matched at combined CV-tuned C), `shuffled` (seed 12345, continuity).
The **only** data-layer change vs CTF/C3 is the aux block = the eGeMAPSv02 88-d vector, aligned to `Z_best`
by id (canonical order = the eGeMAPS cache order = frameset train ⊕ dev_seen).

---

## 3. Raw results (verbatim from `APX_G0COND_GATE_OUT.json`) — HateMM (N=851, n_pos=341, accZ=0.8383, C_Z=0.01)

**Calibration (K-APX-1):** label-oracle **accZA = 1.0000**, headroom-fraction **1.000**, `PASS = True` →
machinery **VALID** (the aux-column-crush pathology is absent; a negative read is admissible).

| arm | accZA | Δacc | per-video 95% CI |
|---|---|---|---|
| **audio_pca_k8** (decision, best) | 0.8345 | **−0.0038** | [−0.0113, +0.0033] |
| audio_pca_k16 (decision) | 0.8294 | −0.0089 | [−0.0174, −0.0007] |
| audio_pca_k32 (context) | 0.8249 | −0.0134 | [−0.0254, −0.0012] |
| audio_pca_k64 (context) | 0.8066 | −0.0317 | [−0.0479, −0.0153] |
| **audio_full_cvC** (full 88-d, C_full=0.01) | 0.8388 | **+0.0005** | [−0.0031, +0.0042] |
| shuffled (seed 12345) k8 / k16 | — | −0.0080 / −0.0101 | — |

`real_max_over_kdec` (best of {k8,k16}) = **−0.0038**.

**Structural read.** Every arm sits at or below zero. The full-capacity 88-d arm (`audio_full_cvC = +0.0005`,
CI straddling 0) is the strictest read — it gives eGeMAPS its entire dimensionality under a tuned penalty and
still adds nothing, so the KILL is **not** a PCA-underpowering artifact. The PCA arms degrade monotonically
with k (k64 −0.0317), the classic **pure-redundancy dilution** signature (added prosodic dimensions are noise
the head cannot zero out), and calibration at full headroom (1.0000) proves the machinery *can* convert real
information. This is exactly the F31 hazard realized: the whisper-large-v3 transcript already banked in
`text_feats` carries the spoken-hate content, so classical prosody adds no **conditional** label information.

---

## 4. Mechanical kill/pass evaluation (design arithmetic quoted verbatim; NON-binding)

Quoted verbatim from `WAVE3_CANDIDATES.md` §(e):

> **K-APX-0 ($0, eGeMAPS):** conditional info over `Z_best` **< +0.040** or CI-lower **≤ 0** → acoustic axis
> prior slashed, **no download escalation.**
>
> **K-APX-1 (calibration):** accZA < 0.99 → MACHINERY_INVALID.

Evaluated mechanically on the binding point estimate (best of decision family {k8,k16}, per the C3-template):

| quantity | rule | observed | fires? |
|---|---|---|---|
| point Δacc (best-k=8) | < +0.040 → kill | **−0.0038** | **✗ kill** (well under bar) |
| bootstrap CI-lower | ≤ 0 → kill | **−0.0113** | **✗ kill** (≤ 0) |
| calibration accZA | < 0.99 → MACHINERY_INVALID | **1.0000** | ✔ VALID |

**K-APX-0 is a 2-condition OR-kill; both conditions fire independently → KILL.** (Interpretation note: unlike
K-CTF-1, K-APX-0 as written lists only two kill conditions — no "real ≤ perm-null" term — so the ≥150-perm
null was neither triggered nor required; it would only discriminate a would-be pass with point ≥ +0.040 AND
CI-lower > 0. Both fail here, so it is moot.) Calibration VALID → the KILL is **credited** (not MACHINERY_INVALID).

**APX verdict (from `OUT.json`):** `K_APX_0_verdict = "KILL"`,
`acoustic_axis = "prior_slashed_no_download_escalation"`.

The survival bar in §(e) — "eGeMAPS gate clears **AND** user grants an audio-encoder download ruling **AND**
the audio-fused key clears the S2S raw bar on HateMM" — is **not reached**: the gate did not clear, so **no
download ruling is requested and no GPU is spent**. Candidate #4 (AVC), explicitly gated behind APX's $0 audio
gate, does not start.

---

## 5. Verdict (NON-binding)

**KILL-side.** Classical whole-video prosody (openSMILE eGeMAPSv02, 88-d) adds **no conditional label
information** over `Z_best` on HateMM — best-k −0.0038 (CI [−0.0113,+0.0033]) and full-88-d +0.0005 (CI
[−0.0031,+0.0042]), both an order of magnitude under the +0.040 bar with CIs straddling/below 0, calibration
VALID (label-oracle accZA = 1.0000). This is the F31 hazard confirmed at ~$0: the ASR transcript already banks
the spoken-hate content, so prosody-as-retrieval-channel is **redundant**, not additive. Because eGeMAPS
*upper-bounds* the cheap realization (and only weakly lower-bounds a learned audio encoder), the classical
probe **de-risks the whole acoustic axis downward** — the **acoustic-axis prior is slashed and NO
audio-encoder download escalation is warranted**; candidate #4 (AVC), gated behind this probe, does not start.

**This is a NON-BINDING pre-ceremony screen.** It authorizes no prereg and requests no download ruling; it
recommends parking the audio axis (APX + AVC) at zero cost. Whether to log APX as a formal pre-registered
negative is the team lead's call.

---

## 6. Required statements

- **No performance / accuracy claim** on any held-out benchmark. All accuracy numbers are train∪val
  cross-validation used **solely** to measure conditional information and audit the probe (C3-template usage).
- **Zero test-touch:** only train + dev_seen HateMM ids were enumerated (canonical order = frameset train ⊕
  dev_seen); `test_seen` videos/caches never opened. Gold labels used **PROBE-ONLY** (calibration + CV strat).
- **Raw videos never left local:** ffmpeg decoded the local mp4s on-node; only derived 88-d float vectors were
  produced. No Modal, no network, no GPU, no model download. This executor created/stopped **zero** Modal apps.
- **Raw-only:** every number transcribed verbatim from `APX_G0COND_GATE_OUT.json`; no companion metric fabricated.
- **Write scope:** `scripts/analysis/apx_extract_egemaps.py`, `scripts/slurm/apx_egemaps_extract.sbatch`,
  `scripts/analysis/apx_g0cond_gate.py`, `refine-logs/APX_G0COND_GATE_OUT.json`,
  `refine-logs/APX_G0COND_GATE_run.log`, this record, and derived caches under `data/audio/HateMM/` (gitignored).
  No prereg / config / CLAUDE.md / state files mutated. Committed (scripts + record + probe outputs);
  **not pushed**.
