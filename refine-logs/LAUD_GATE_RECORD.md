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

## 2. Stage B — extraction results  *(APPENDED AFTER JOB COMPLETES — placeholder)*

_job id, runtime, per-dataset counts (851/629/657), n_zero_rows, NaN check, 3 example norms, cache sha256._

## 3. Stage C — gate results  *(APPENDED AFTER THE SINGLE GATE READ — placeholder)*

_per dataset × Z arm: baseline accZ, label-oracle accZA (K-LAUD-1), best-decision-k Δacc + bootstrap CI,
perm-null max band (if a would-be pass), K-LAUD-0 ruling, per-dataset combined ruling
(KILL / HONEST-PARTIAL / PASS), promote_head_gpu._

## 4. Verdict  *(APPENDED — placeholder)*

_HateMM anchor ruling + EN (K-LAUD-2) + ZH; whether the axis survives the $0 screen or the prior is slashed._
