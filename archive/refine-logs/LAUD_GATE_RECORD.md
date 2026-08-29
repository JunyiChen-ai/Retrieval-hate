# LEARNED-AUDIO axis — G0-cond gate record (K-LAUD, Whisper-encoder hidden-state stream)

**Date:** 2026-07-20 NZST. **Executor:** learned-audio executor (CPU/SLURM, conda `HateVideo`).
**Binding design:** `refine-logs/AUDIO_AXIS_FORENSIC_RECON.md` (commit `166f9e2b`) §2-5, followed VERBATIM.
**Status of THIS section (header):** **PRE-DECLARATION — bars frozen BEFORE any dev-label read / any gate
number.** Extraction results (Stage B) and gate numbers (Stage C) are appended below AFTER this header is
committed. Full raw-only transcription discipline: every Stage-C number will be copied verbatim from
`refine-logs/LAUD_GATE_OUT.json`.

The axis: **Whisper-large-v3 ENCODER hidden-state embeddings as a NEW input stream** — the #1-ranked
in-box gap in both red-team audits (`REDTEAM_BAN_SCOPE_AUDIT.md` §GAP-1, `REDTEAM_UNTESTED_CELLS.md` C3).
The only prior audio work (F41/APX) killed a **classical eGeMAPS-88-d whole-video prosody** vector on
**HateMM only**, and that record itself concedes eGeMAPS "only weakly lower-bounds a learned audio encoder."
A learned Whisper-encoder representation and the **entire MHC-EN audio axis** are genuinely untested.

---

## 0. Pre-declared kill-switches (FROZEN — no adjustment after seeing numbers)

**Machinery** (VERBATIM `scripts/analysis/c3_fusion_probe.py` via the APX fork): Z standardized ALONE at
its Z-only inner-CV-optimal `C_Z`; aux block (Whisper video vector) standardized × s=50 (effectively
un-penalized), refit at `C_Z`; aux via train-fold PCA (leak-free), decision family k∈{8,16} (+ {32,64}
context); 5×5 RepeatedStratifiedKFold rs=1000+rep; per-video-clustered bootstrap B=5000 on Δacc;
permutation null as a distribution over **≥150** fresh permutations (only computed to confirm a would-be
pass). Binding point = best of the decision family {k8,k16}. **Dev-side only** (train∪val), **single gate
read**, **zero test-touch**.

- **K-LAUD-1 (calibration):** label-oracle `accZA < 0.99` → **MACHINERY_INVALID** (no verdict credited).
- **K-LAUD-0 (kill / promote — K9 house standard, bit-consistent with APX/W2-A):**
  best-decision-k point **Δacc < +0.040** OR bootstrap **CI-lower ≤ 0** OR (on a would-be pass) **not >
  all ≥150 permutation maxima** → **KILL** → audio-axis prior slashed, **no head GPU**; and — subject to
  recon caveat §0.2 (Whisper is a speech-ASR encoder, weakest on the non-speech residual) — **no
  general-audio (AST/BEATs/wav2vec2) download escalation from Whisper evidence alone**.
  - **+0.030 ≤ Δacc < +0.040 with CI-lower > 0 = HONEST-PARTIAL flag** (documented near-miss, NOT an
    auto-promote; the frozen machinery bar stays **+0.040** so the number is comparable to the graveyard).
- **K-LAUD-2 (EN blank-cell leg):** the identical screen on **MHC-EN** — the binding-gap dataset whose
  audio conditional-info was **never measured, even classically** (APX was HateMM-only). Genuine blank-cell
  fill, not a re-run. **ZH is the third leg.**
- **Two Z conditioning arms per dataset — a PASS must clear BOTH** (recon §4, matches the W2-A "Z_best
  8960-d is the sole binding" rule):
  - **deployed_7168** — the per-dataset DEPLOYED winning encoder (img⊕text): the honest "does audio add
    over what we actually deploy?".
  - **strict_8960** — the exact W2-A/APX `Z_best` = CLIP img⊕text ⊕ **frozen**-Qwen img⊕text; guards the
    weak-deployed-encoder loophole.

| dataset | role | deployed_7168 encoder | strict_8960 = CLIP(1792) ⊕ frozen-Qwen(7168) |
|---|---|---|---|
| HateMM | acoustic anchor (decision) | `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF` | CLIP + `Qwen2.5-VL-7B-Instruct_HF` |
| MHC (EN) | **K-LAUD-2 blank-cell** | `Qwen2.5-VL-7B-Instruct_HF` (frozen) | CLIP + `Qwen2.5-VL-7B-Instruct_HF` |
| MHC_zh (ZH) | third leg | `Qwen2.5-VL-7B-Instruct-LoRA_HF` | CLIP + `Qwen2.5-VL-7B-Instruct_HF` |

**Promote → head GPU** iff the **HateMM anchor** clears the triple over **both** Z arms → formal prereg for
a 3-seed head-fusion run (audio as 3rd stream into the `align` Hadamard head), with its own oracle/KS bars.

**Honest prior (recon §0/§6): LOW ~10–15%**, higher on HateMM than EN. *Against:* F31 redundancy (the
large-v3 transcript already banks spoken hate into deployed `text_feats`); the APX eGeMAPS null; Whisper's
speech-ASR encoder is weakest on the non-speech residual that is the only blessed increment. *For:* eGeMAPS
compressed audio into 88 hand-crafted scalars, a learned encoder keeps acoustic-event structure; MHC-EN
audio is a blank cell. **D7 novelty is thin** — learned-audio = catch-up to SOTA inputs (all 3 baselines
already fuse audio); a gain is a **performance/ablation row**, never a novelty win.

---

## 1. Representation + provenance (FROZEN)

**Video-level aux vector (the $0-gate input):** per video, PyAV-decode 16 kHz mono → 30 s chunks → Whisper
`get_encoder()` last-hidden `[1500, d]` per chunk → **mean-pool ⊕ max-pool** over the 1500 time-steps
(`2d`/chunk) → **mean over chunks** → one **`2d`-dim** vector/video. Primary model
**openai/whisper-large-v3** (`d_model=1280` → **2560-d**). `d_model` read dynamically from the loaded
encoder (not hardcoded), so the pooled dim is correct for any Whisper size.
**Segment variant (banked, NOT used in the gate):** concatenated encoder frame sequence `[n_chunks*1500, d]`
split into **K=4 uniform windows** → mean⊕max per window → `[4, 2d]`. Frozen now for a later localization
tie-in.

| item | value |
|---|---|
| extractor | `scripts/analysis/laud_extract_whisper.py` sha256 **`d51666a3dea4f4f7210dde9cff22e91e17ca90d69edd8edf6671721ad04d7823`** |
| extraction sbatch | `scripts/slurm/laud_extract.sbatch` sha256 **`88127bfa5639808ffcccf7a4b3d443cca4d735df28c6e168cc1021d9bdda3433`** |
| gate | `scripts/analysis/laud_g0cond_gate.py` sha256 **`b601013a13727973536c16e199276ae30c2d475c423fe52e090d006668e60594`** |
| machinery source (fork target, verbatim) | `scripts/analysis/apx_g0cond_gate.py` sha256 `c338de8cea7198168ec7c9cc96f9c9558667939d3aaf401c559ee349bcc7b5bd` |
| machinery source (verbatim template) | `scripts/analysis/c3_fusion_probe.py` sha256 `9091e2c3443d4826144f820217e37d43d26d282d334b0b35bea7cb4ae9748b3c` |
| audio decode | `decode_audio_pyav` (PyAV 17.0.0, no ffmpeg binary / no torchaudio) imported VERBATIM from `src/utils/generate_segment_asr_HF.py` sha256 `c52cba69648e10b1c87ee4b06182ccd9496a3d274350606e4d61f21bf7a0b394` |
| Whisper weights | `openai/whisper-large-v3` snapshot `06f233fe…` (`model.safetensors` 3.087 GB, on disk; `HF_HUB_OFFLINE=1`, **no download**) |
| scope | **train∪val only**, per dataset: HateMM **851** (n_pos 341), MHC-EN **629**, MHC-ZH **657** — id-set == deployed==frozen==CLIP cache == raw mp4 (verified). **Zero test-touch.** |
| aux cache (per DS) | `data/audio/<DS>/whisper_whisper-large-v3_trainval.pt` (`ids`,`emb`[N,2560],`seg_emb`[N,4,2560],`labels`,`model_tag`,`d_model`,`pool`,`num_subclips`) + `..._manifest.json` |

### 1.1 Discrepancy log (recon vs on-disk ground truth)

The recon §2 states whisper-**base** is `d_model=384` / 4 encoder layers (and "corrects the tasking's 512").
On-disk ground truth (`WhisperConfig.from_pretrained` this session): whisper-base is **`d_model=512` / 6
layers / 80 mels** → base mean⊕max would be **1024-d**, not 768-d. This affects the **secondary** model only;
the **primary** large-v3 is confirmed **`d_model=1280` / 32 layers / 128 mels → 2560-d**, exactly as the
recon and this design specify. Resolution: the extractor reads `d_model` dynamically, so it is correct for
either size; **large-v3 is the model run for this gate** (the recon's recommended zero-download first shot).
No other tasking/recon conflict encountered.

---

## 2. Stage B — extraction results (job 13295)

SLURM **13295** (`laud_extract`) COMPLETED `0:0`, **elapsed 00:24:23** wall (extraction compute 663 s),
1×A100 GPU-light, 8 CPU / 48 G, `HF_HUB_OFFLINE=1`, MaxRSS 4.94 G. Log `slurm/logs/laud_extract_13295.log`.
Model `openai/whisper-large-v3` (`d_model=1280` → **pooled 2560-d**). All three caches
`data/audio/<DS>/whisper_whisper-large-v3_trainval.pt` (+ `..._manifest.json`, per-video `.npz` shards):

| dataset | N (exp) | n_pos | emb | seg | n_zero rows | n_nan | status | 3 example norms | cache sha256 |
|---|---|---|---|---|---|---|---|---|---|
| HateMM | **851** (851 ✓) | 341 | (851, 2560) f32 | (851, 4, 2560) | 0 | 0 | ok:851 | 72.51 / 65.68 / 71.50 | `4a6b0bb2fe4b26f6d16a7f1a4d6ff2680fad306593758eb3072eeb96e3214070` |
| MHC (EN) | **629** (629 ✓) | 193 | (629, 2560) f32 | (629, 4, 2560) | 0 | 0 | ok:629 | 69.01 / 71.57 / 71.58 | `203b67343c82082c295742d4ca4517f155cf0ca29fa9a7064eef2b396ede9d54` |
| MHC_zh (ZH) | **657** (657 ✓) | 208 | (657, 2560) f32 | (657, 4, 2560) | 0 | 0 | ok:657 | 61.14 / 72.35 / 69.50 | `e2b046276797c8b402d5cab546311411e1c0bb752c29ca6a4c785984b9d22c9e` |

Audio coverage 100% as the recon promised — **no `no_audio` rows, no decode failures, no NaN** anywhere.

## 3. Stage C — gate results (single read; verbatim from `LAUD_GATE_OUT.json`, run `LAUD_GATE_run.log`)

Gate `laud_g0cond_gate.py` on local login-node CPU, **elapsed 530 s**, `LAUD_GATE_OUT.json` sha256
`353bba844faf9cfb373709b138b47153c4a82c1a2d9ac714f0f6fd97ccacae98`. **K-LAUD-1 VALID for all 6 cells**
(label-oracle `accZA = 1.0000`, headroom-fraction 1.000 everywhere → genuine nulls, not machinery
artifacts). **No perm-null was computed on any cell** — correct per the pre-declared spec: perm-null runs
only to confirm a would-be pass, and **no arm cleared C1&C2**, so C3 is `None` throughout.

Binding decision = best of the decision family {k8, k16}. Numbers below are copied verbatim (run-log lines
cited).

| dataset | Z arm | accZ | best-k Δacc | bootstrap CI | full-2560 Δacc | K-LAUD-0 |
|---|---|---|---|---|---|---|
| HateMM | deployed_7168 | 0.8712 | **+0.0014** (k16) | [−0.0075, +0.0103] | +0.0063 | **KILL** (L1-7) |
| HateMM | strict_8960 | 0.8383 | **+0.0014** (k8) | [−0.0073, +0.0106] | +0.0052 | **KILL** (L8-14) |
| MHC (EN) | deployed_7168 | 0.7847 | **+0.0041** (k8) | [−0.0079, +0.0159] | +0.0038 | **KILL** (L15-21) |
| MHC (EN) | strict_8960 | 0.7971 | **−0.0013** (k8) | [−0.0130, +0.0102] | −0.0032 | **KILL** (L22-28) |
| MHC_zh (ZH) | deployed_7168 | 0.8770 | **−0.0052** (k16) | [−0.0155, +0.0052] | −0.0018 | **KILL** (L29-35) |
| MHC_zh (ZH) | strict_8960 | 0.8228 | **−0.0082** (k8) | [−0.0180, +0.0009] | −0.0094 | **KILL** (L36-42) |

**Every** decision-family point estimate is an **order of magnitude under the +0.040 bar** (global max
across all 6 cells × {k8,k16} = **+0.0041**, EN/deployed), **every** bootstrap CI straddles 0 (or is
entirely negative), and no cell reaches even the +0.030 honest-partial band. The higher-capacity context
arms trend **negative** (k32/k64 down to −0.0298/−0.0320 on ZH; audio adds noise, not signal, as capacity
grows) and the shuffled-audio controls sit at ≈0 (−0.0149…+0.0060) — a clean null, not an under-powered
one. Per-dataset combined ruling (a PASS must clear BOTH Z arms): **HateMM KILL, EN KILL, ZH KILL**.

## 4. Verdict — KILL (all three datasets, both Z arms)

**The learned Whisper-large-v3-ENCODER audio stream carries no conditional label information over the
deployed encoder (7168-d) OR the strict W2-A/APX `Z_best` (8960-d), on any of the three datasets.**
K-LAUD-0 fires on every arm; **`promote_head_gpu = False`** → the acoustic-axis prior is slashed at ~$0,
**no head GPU is spent**. K-LAUD-2 (the EN blank-cell fill) is now measured: EN audio conditional-info is
**also null** (best +0.0041 deployed / −0.0013 strict) — the F44 label-limited-EN wall was not the obstacle
here; the audio simply adds nothing over the deployed multimodal representation. This is the **6th
no-conversion audio datum** and closes the eGeMAPS→learned-encoder gap the APX record flagged: a *learned*
Whisper encoder confirms the *classical* eGeMAPS null rather than overturning it. The mechanism matches the
recon's F31 hazard — the large-v3 **transcript** already banks spoken hate into the deployed `text_feats`,
leaving no non-lexical residual the Whisper encoder can add.

**SCOPE — this closes only the WHISPER realization, not the learned-audio axis (recon §0.2, binding).**
Whisper is a **speech-ASR** encoder, optimized to transcribe speech and therefore **weakest exactly on the
non-speech events** (music, chants, screams, gunshots, tone) that motivated the axis. A Whisper-encoder null
must **not** close the whole learned-audio axis: a **general-audio encoder (AST / BEATs / wav2vec2)** is the
proper closer. Per K-LAUD-0, that download escalation is **not** licensed *from this Whisper evidence alone*
— it proceeds only if the loop judges the non-speech caveat decisive. **D7 novelty was thin regardless**
(learned-audio = catch-up to SOTA inputs; all three baselines already fuse audio), so even a gain would have
been a performance/ablation row, never a novelty win.

The K=4 segment-level variant (`seg_emb [N,4,2560]`) is banked in the caches, unused by this gate, and
remains available for a future localization tie-in without re-extraction.

---
**Discipline honored:** ZERO test-touch (train∪val only, test split never enumerated); single gate read
(no bar adjustment after seeing numbers — bars frozen at commit 7ff217f before any dev-label read); no
`state/` mutation; no model download (`HF_HUB_OFFLINE=1`, weights on disk); no Modal (raw media stayed
local). Local commits only, not pushed.
