# LEARNED-AUDIO AXIS — FORENSIC RECON (Whisper-ENCODER hidden-state stream, GAP-1)

**Agent:** forensic-recon (zero-GPU). **Date:** 2026-07-20 NZST.
**Mission.** Reconstruct a GO/NO-GO + full execution skeleton for the LEARNED-AUDIO cell — Whisper
**encoder** hidden-state embeddings as a **new input stream** — WITHOUT submitting any job or authoring
any prereg. This is the axis the red-team ban-scope audit ranked **GAP-1 / rank-1** (cheapest, in-box,
blessed gain source): `refine-logs/REDTEAM_BAN_SCOPE_AUDIT.md` §GAP-1 + `refine-logs/REDTEAM_UNTESTED_CELLS.md`
cell **C3**.

**Discipline honored.** CPU-only reading + forensic arithmetic + light CPU probes (config imports,
`ffprobe`, JSONL census). **ZERO** GPU / SLURM / Modal / model download / prereg / `state/` mutation /
test-touch. Every disk fact below is cited to a path and was verified this session.

---

## 0. VERDICT (headline)

**GO — run the two-stage screen** (one small local extraction job → a $0 CPU conditional-info gate).
This is the single strongest refutation of "audio axis closed": the only audio work ever done (APX/F41)
killed a **classical eGeMAPS-88-d whole-video prosody** vector on **HateMM only**, and that record *itself
concedes* eGeMAPS "only **weakly** lower-bounds a learned audio encoder." A learned Whisper-encoder
representation and **the entire MHC-EN audio axis** are genuinely untested. Whisper weights are **already
on disk (zero download)**, audio coverage is **100%**, and the exact K9/APX gate machinery is forkable
verbatim.

**But GO with three honest caveats, load-bearing for how the result is read:**
1. **Prior is LOW (~10–15%).** The F31 hazard is real: the whisper-large-v3 *transcript* already banks
   spoken hate into the deployed `text_feats`. The blessed increment is only the **non-lexical, non-speech**
   residual (music, chants, screams, gunshots, tone, laugh-over-slur).
2. **Whisper is a speech-ASR encoder** — optimized to *transcribe speech*, so it is weakest exactly on the
   non-speech events that motivate the axis. A Whisper-encoder null therefore **must not** close the whole
   learned-audio axis: a general-audio encoder (AST / BEATs / wav2vec2, download-gated) is the proper
   *closer*, or we repeat the APX overreach one level up. Whisper is the correct **zero-download first shot**,
   not the last word.
3. **D7 novelty is thin (catch-up).** "Add audio" is HateMM's 2023 founding contribution; all three
   baselines already fuse audio. Even a gain is a **performance / ablation row**, not a novelty win. This
   axis buys a possible 2nd/3rd passing dataset, not a method-novelty claim.

**Total GPU cost:** extraction ≈ **1–2 GPU-h** (all 3 datasets, one job; budget 3 GPU-h); the gate is
**$0 CPU**. No download. No test-touch.

---

## 1. RAW MEDIA — coverage & audio tracks (checklist item 1)

**Local raw video (`data/video/<DS>/All/*.mp4`):**

| dataset | raw mp4 on disk | train / val / test (`data/gt/<DS>/*.jsonl`) | train∪val (screen scope) |
|---|---|---|---|
| **HateMM** | **1066** | 744 / 107 / 215 | **851** |
| **MHC (EN)** | **790** | 549 / 80 / 161 | **629** |
| **MHC_zh (ZH)** | **806** | 579 / 78 / 149 | **657** |

(The EN 790 < gt-total 790 present; the audit's "792/1000" is the *published* set — locally 790 mp4 cover
the 549+80+161 = 790 split ids. ZH 806 = 579+78+149.)

**Audio-track census — EXACT, from the banked ASR JSONLs** (`data/ASR/<DS>/*_asrK4_whisper-large-v3.jsonl`,
which record a per-video `audio_ok` flag + `duration` computed at ASR time):

| dataset | videos with ASR record | `no_audio` (audio_ok=false) | empty transcript | total audio (min) |
|---|---|---|---|---|
| HateMM | 1066 (all splits) | **0** | **0** | 2592.5 |
| MHC (EN) | 549 (train split only¹) | **0** | **0** | 321.2 |
| MHC_zh | 806 (all splits) | **0** | **0** | 422.9 |

¹ EN dev/test ASR-K4 was never generated, but that is irrelevant to audio extraction (which decodes video
directly); the EN dev/test **embeddings** exist and the raw mp4s are present, so the EN screen scope
(train∪val=629) is fully extractable.

A CPU `ffprobe` spot-check of 12 videos/dataset independently found **0/36 with no audio stream** (all
`aac` audio + `h264` video). **Load-bearing fact: audio coverage is effectively 100% — there is no
silent-video / zero-audio-row pathology to correct for** (contrast the W2-A empty-transcript dilution).

**`ffmpeg` / decode:** `ffmpeg` + `ffprobe` present (`/data/jehc223/miniconda3/bin/ffmpeg`, v9c33b2f). But
the deployed audio-decode path does **not** need the ffmpeg binary *or* torchaudio: the existing ASR
extractor `src/utils/generate_segment_asr_HF.py:101` (`decode_audio_pyav`) uses **PyAV** (`import av`,
**PyAV 17.0.0** confirmed) → `AudioResampler(format=s16, layout=mono, rate=16000)` → float32 mono. This is
the exact contract Whisper expects (16 kHz mono) and is reusable verbatim. **`torchaudio` is MISSING** in
`HateVideo` — a red herring; the PyAV path makes it unnecessary.

---

## 2. WHISPER — weights, API, cost (checklist item 2)

**On-disk weights (zero download):** `~/.cache/huggingface/hub/` (= `/data/jehc223/home/.cache/...`):

| model | snapshot | weight file | size | dims |
|---|---|---|---|---|
| **openai/whisper-large-v3** | `06f233fe…` | `model.safetensors` | **3.087 GB** (complete) | d_model **1280**, 32 enc layers |
| **openai/whisper-base** | (single) | `model.safetensors` | **290 MB** (complete) | d_model **384**, 4 enc layers |

Both were pulled for the ASR transcript step (`--model openai/whisper-large-v3`,
`generate_segment_asr_HF.py:60`; the banked transcripts in `text_feats` are large-v3 output — **this is
the F31 hazard**).

**Encoder-only forward is supported** in `transformers 4.49.0` (verified by CPU import, no forward):
`WhisperModel.get_encoder()` exists; `WhisperConfig` loads large-v3 (`d_model=1280`,
`max_source_positions=1500`, `encoder_layers=32`) and base (`d_model=384`, `encoder_layers=4`);
`WhisperFeatureExtractor` reports `sampling_rate=16000`, `chunk_length=30`, `n_samples=480000`. So one
encoder forward = a 30 s mel → **1500 time-steps × d_model**.

> **Correction to the tasking:** base is **d_model = 384**, not 512. So per-chunk mean-pool = 1280
> (large-v3) / **384** (base); the mean⊕max design below = 2560 / 768.

**Cost arithmetic (encoder-forward + decode).** Screen scope = train∪val only (zero test-touch). From the
per-video `duration` sums: 30 s-chunk counts ≈ **HateMM 4,209** (2104.7 min), **MHC-EN ~736** (train
321.2 min + est. dev ~47 min), **MHC-ZH ~693** (346.7 min) → **≈ 5,640 chunks total**.

- **On 1×A100 (recommended):** large-v3 encoder forward on a 30 s mel batched ≈ 20–50 ms/chunk ⇒
  5,640 chunks ≈ **5 min pure GPU**. The real cost is PyAV audio decode (CPU, ~48 h of audio at ~30–100×
  realtime ⇒ ~1 h single-thread / ~8 min 8-way). **Whole job (all 3 datasets) ≈ 1–2 GPU-h with margin** —
  the same scale as the eGeMAPS precedent (job 13203: 851 HateMM videos, CPU-only, **8m53s**, 8-way).
- **CPU-only overnight?** **base: YES, realistically** — 4-layer/384-d forward ≈ 0.3 s/chunk ⇒ 5,640 ×
  0.3 s ≈ **28 min single-thread**. **large-v3: MARGINAL single-thread** — 32-layer/1280-d/seq-1500 forward
  ≈ 3 s/chunk ⇒ 5,640 × 3 s ≈ **4.7 h single-thread** (≈ 40 min 8-way). Crucially, a 4.7 h *bare login-node*
  process would be **reaped** (CLAUDE.md: login=compute). **⇒ Even the "CPU-only" variant must go through a
  CPU SLURM job** (precedent: eGeMAPS 13203 was CPU-only SLURM, no `--gres`). Cleanest: one GPU-light SLURM
  job (fastest) for large-v3, or a CPU SLURM job for base.

---

## 3. REPRESENTATION DESIGN (checklist item 3, pre-specified)

**Video-level vector (the $0-gate input):** for each video, decode 16 kHz mono (PyAV) → split into 30 s
chunks → Whisper `get_encoder()` last-hidden-state `[1500, d]` per chunk → **mean-pool ⊕ max-pool over the
1500 time-steps** = `2d` per chunk → **mean over chunks** → one **`2d`-dim** vector/video (large-v3
**2560-d**, base **768-d**).

**Segment-level variant (banked for a later localization tie-in, NOT used in the $0 gate):** split each
video's encoder frames into **K=4 uniform windows** — matching the existing K=4 sub-clip / ASR-window
contract (`generate_segment_asr_HF.py:138` `window_time_bounds`, M=16→K=4) — mean⊕max per window →
`4 × 2d` per video. Reserved for a future weak-supervision / HateClipSeg localization arm; extracted in the
same pass at ~zero marginal cost.

**Justification (≤3 sentences).** Mean-pool captures the dominant acoustic content and max-pool captures
transient events (a scream / gunshot / slur burst) that a mean washes out — the exact non-speech residual
that motivates the axis over eGeMAPS's whole-video functionals. Averaging over 30 s chunks gives one
fixed-width video vector directly pluggable into the banked conditional-info gate (no sequence model, no
new training). The K=4 variant is pre-registered now so the segment representation is frozen before any
localization use, avoiding a post-hoc design d.o.f.

---

## 4. GATE DESIGN — the decisive $0 CPU screen (checklist item 4)

**Fork target:** `scripts/analysis/apx_g0cond_gate.py` (which itself reuses `scripts/analysis/c3_fusion_probe.py`
**VERBATIM** — the same C3-template conditional-info machinery that rendered the binding W2-A/K9 and APX
verdicts). The **only** data-layer change: swap the eGeMAPS-88-d aux block for the Whisper-encoder
`2d`-vector, and generalize the `Z` loader to the per-dataset **current best representation**.

**Baseline `Z` per dataset (all caches confirmed on disk, exposing `img_feats`/`text_feats`/`labels`/`ids`):**

| dataset | deployed winning encoder (primary Z, 7168-d) | cache |
|---|---|---|
| HateMM | LoRA-**curric** img+text | `Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt` |
| MHC-EN | **frozen**-Qwen img+text | `Qwen2.5-VL-7B-Instruct_HF.pt` |
| MHC-ZH | LoRA img+text | `Qwen2.5-VL-7B-Instruct-LoRA_HF.pt` |

- **Primary arm** conditions audio over the **deployed** 7168-d Z (Qwen img 3584 ⊕ text 3584) — the honest
  "does audio add over what we actually deploy?".
- **Strict-confirm arm** conditions over the full **8960-d Z_best** = CLIP img 1024 ⊕ CLIP text 768 ⊕ Qwen
  img 3584 ⊕ Qwen text 3584 (all caches present) — the exact W2-A/APX baseline; guards the weak-deployed-
  encoder loophole. **A PASS must clear BOTH** (matches the W2-A "Z_best 8960-d is the sole binding" rule).

**Machinery (inherited verbatim):** Z standardized alone at its Z-only inner-CV-optimal `C_Z`; aux block
standardized × s=50 (effectively un-penalized), refit at `C_Z`; aux via train-fold PCA (leak-free), k ∈
{8,16} decision family (+ {32,64} context); 5×5 RepeatedStratifiedKFold; per-video-clustered bootstrap
B=5000; **mandatory label-oracle calibration** (accZA ≥ 0.99 or MACHINERY_INVALID); permutation null as a
distribution over ≥150 fresh permutations (only to confirm a would-be pass).

**Kill / promote bars (K9 house standard):**
- **K-LAUD-1 (calibration):** label-oracle accZA < 0.99 → **MACHINERY_INVALID** (no verdict credited).
- **K-LAUD-0 (kill):** best-decision-k point Δacc **< +0.040** OR bootstrap CI-lower **≤ 0** OR (on a
  would-be pass) not **> all** ≥150 perm maxima → **audio-axis prior slashed, no head GPU**, and — subject
  to caveat §0.2 — no general-audio download escalation *from Whisper evidence alone*.
- **Promote:** clears the triple → the axis survives the $0 screen → go to prereg (§5) for the formal
  3-seed head-fusion validation.

> **Reconciliation with the tasking's "+0.03 dev" bar.** I hold the **house +0.040** (bit-consistent with
> APX/W2-A/K9; every prior binding conditional-info gate used +0.040). The **+0.03–0.04 band is an
> honest-partial / MODEST flag**, not an auto-promote — the loop may relax to +0.03 as a *documented*
> decision, but the frozen machinery bar stays +0.040 so the number is comparable to the graveyard.
> Kill on Δacc ≤ 0 or CI-lower ≤ 0 exactly reproduces the K9 kill.

**K-LAUD-2 (EN blank-cell leg):** run the identical screen on **MHC-EN** — the binding-gap dataset whose
audio conditional-info was **never measured, even classically** (APX was HateMM-only). This is a genuine
blank-cell fill, not a re-run. (ZH included as a third leg; HateMM is the acoustic anchor per the
richer/noisier hate-video content.)

---

## 5. SLURM / MODAL BOUNDARY + EXECUTION SKELETON (checklist item 5 — plan only, NOT submitted)

**Extraction = ONE local SLURM job.** Raw audio never leaves the machine (CLAUDE.md hard rule + the
`modal_probe_runner.py` hard block); only derived `2d`-float `.pt` caches + a manifest are produced. This
is a **formal-side** small job, precedent = APX eGeMAPS 13200/13203. NOT a Modal probe (raw media can't go
to Modal; the derived vectors *could*, but the extraction that produces them is local).

**Stage A — extract (local SLURM, ~1–2 GPU-h, GPU-light; or CPU-SLURM for base):**
1. New script `scripts/analysis/laud_extract_whisper.py`: reuse `decode_audio_pyav` (PyAV, no torchaudio),
   `WhisperFeatureExtractor` + `WhisperModel.get_encoder()`, mean⊕max pool per §3, over **train∪val only**
   (zero test-touch), per dataset. Write `data/audio/<DS>/whisper_<tag>_trainval.pt` (`ids`, `emb`,
   `labels`, `model_tag`, `pool`) + a `..._manifest.json` (`status`, `n_zero_rows`, NaN check) — mirroring
   the eGeMAPS cache/manifest layout.
2. New sbatch `scripts/slurm/laud_extract.sbatch` (no `--time`; absolute HateVideo python by path;
   fail-fast preflight `import av, transformers`; **append** any aux env to `PATH` to avoid the 13200
   python-shadow trap documented in `APX_GATE_RECORD.md` §1.1).

**Stage B — gate (LOCAL login-node CPU, $0, no SLURM needed):**
3. New script `scripts/analysis/laud_g0cond_gate.py` = fork of `apx_g0cond_gate.py` with the two Z arms
   (§4) + Whisper aux loader. Run per dataset; write `refine-logs/LAUD_G0COND_GATE_OUT.json` +
   `..._run.log`. Full raw-only transcription discipline (numbers copied verbatim from the JSON).

**This recon submits NOTHING.** All three scripts are to be authored in the follow-up prereg stage, not now.

---

## 6. RISKS / PRIORS (checklist item 6)

- **Audio quality:** 100% audio coverage, 0 silent videos (§1) — no missing-modality masking needed. But
  hate-video audio is **music-heavy / mixed** (background tracks, memes, clips), which *helps* the
  non-speech hypothesis (music/genre is real signal) yet *hurts* Whisper specifically (its ASR encoder
  down-weights non-speech).
- **The F44 label-limited-EN wall does NOT bar this channel.** Ban-scope audit verbatim
  (`REDTEAM_BAN_SCOPE_AUDIT.md` §GAP-1): *"Audio is a genuine **new-input channel** … so the F44 'no
  representation lever converts label-limited EN' wall does **not** apply to it: a new modality can add
  signal a representation lever cannot."* This is exactly why EN-audio (K-LAUD-2) is worth the blank-cell
  fill despite EN being closed to every *representation* lever.
- **Honest prior + mechanism.** **LOW ~10–15%**, higher on HateMM than EN.
  *Against:* F31 redundancy — the large-v3 transcript already banks spoken hate into the deployed
  `text_feats`; the APX eGeMAPS null (+0.0005 over Z_best, calib 1.0) shows whole-video prosody adds nothing
  conditional; D1 conditional-redundancy; **and Whisper's speech-ASR encoder is weakest on the non-speech
  events that are the only blessed residual** (caveat §0.2).
  *For:* eGeMAPS compresses audio into 88 hand-crafted scalars — a learned encoder keeps acoustic-event
  structure the transcript misses entirely; **and MHC-EN audio is a blank cell** (never screened at any
  fidelity), so its prior is not pinned by any measurement.
- **D7 novelty status — thin, stated plainly.** Learned-audio = **catch-up to SOTA inputs**. HateMM (2023)
  and the MultiHateClip baselines already fuse audio; a gain here is a **performance/ablation row**, not a
  novel-MLLM-role win. It can only buy a passing dataset, never the novelty claim (D7).

---

## 7. VERDICT + COST LEDGER + KILL-SWITCH SKELETON (checklist item 7)

**GO — extract (Stage A) then $0 gate (Stage B).** It is the #1-ranked in-box gap in **both** red-team
audits; the APX kill is provably scoped to *classical prosody, HateMM-only*; the tools are all on disk. GO
is bounded by the three §0 caveats: LOW prior, Whisper-speech-encoder weakness (⇒ a null closes only the
*Whisper* realization, not the learned-audio *axis*), and thin D7 novelty.

**Cost ledger:**

| stage | resource | cost | download | test-touch |
|---|---|---|---|---|
| A · extract (3 datasets, one job) | local SLURM, GPU-light (large-v3) or CPU (base) | **~1–2 GPU-h** (budget 3) | **none** | none |
| B · conditional-info gate (3 datasets) | local login-node CPU | **$0** (~few min/dataset) | none | none |
| **total to a verdict** | | **~1–2 GPU-h + $0** | **none** | **zero** |

**Kill-switch skeleton (what the follow-up prereg would FREEZE):**
- **K-LAUD-1** calibration accZA ≥ 0.99 (else MACHINERY_INVALID).
- **K-LAUD-0** HateMM: kill iff Δacc < +0.040 OR CI-lower ≤ 0 OR not > all ≥150 perm maxima, over **both**
  Z arms (deployed-7168 AND strict-8960). Kill ⇒ prior slashed, no head GPU; general-audio (AST/BEATs)
  download escalation only if the loop judges the non-speech caveat (§0.2) decisive.
- **K-LAUD-2** MHC-EN identical screen (blank-cell fill) + ZH third leg.
- **Promote** iff the triple clears (HateMM primary) ⇒ formal prereg for a **3-seed head-fusion** run
  (audio as 3rd stream into the `align` Hadamard head, local SLURM), with its own oracle/KS bars.

**What the prereg would freeze (hashes):** extraction script + Whisper `model_tag` (large-v3 primary, base
secondary) + pooling spec (§3, mean⊕max + K=4 variant) + PyAV decode contract (16 kHz mono) + gate script
(fork of `apx_g0cond_gate.py`) + Z definitions per dataset + bars (+0.040 triple) + calibration + perm-seed
base + bootstrap seed + per-dataset feature-cache sha256 + manifest + the zero-test-touch declaration
(train∪val only).

---

## PROVENANCE (verified this session)
- Kill records read: `refine-logs/REDTEAM_BAN_SCOPE_AUDIT.md` (§GAP-1, rank table),
  `refine-logs/REDTEAM_UNTESTED_CELLS.md` (C3, §0 probe), `refine-logs/APX_GATE_RECORD.md` (F41 eGeMAPS
  null + §1.1 python-shadow trap + §5 "weakly lower-bounds a learned encoder"), `refine-logs/W2A_PROBE_VERDICT_REVIEW.md`
  (K9 discipline, Z_best 8960 binding rule).
- Machinery: `scripts/analysis/apx_g0cond_gate.py` (fork target), `scripts/analysis/c3_fusion_probe.py`
  (verbatim conditional-info template), `src/utils/generate_segment_asr_HF.py:101,138` (PyAV decode +
  window contract).
- Disk facts: Whisper weights `~/.cache/huggingface/hub/models--openai--whisper-{large-v3,base}` (3.087 GB /
  290 MB); `transformers 4.49.0` `WhisperModel.get_encoder` (d_model 1280/384); `PyAV 17.0.0`; `torchaudio`
  absent; raw video `data/video/{HateMM,MHC,MHC_zh}/All` (1066/790/806); audio census + durations from
  `data/ASR/<DS>/*_asrK4_whisper-large-v3.jsonl` (0 no-audio, 0 empty); best-rep caches
  `data/CLIP_Embedding/<DS>/{train,dev_seen}_*.pt` (train∪val 851/629/657, img+text 3584 each); gt splits
  `data/gt/<DS>/{train,val,test}.jsonl`.
- **Required statements:** ZERO GPU / SLURM / Modal / download / training / test-touch spent by this recon;
  no held-out metric read; no `state/`, prereg, config, or frozen artifact mutated; no Modal app created.
  One deliverable (this file). Committed on `main`, **not pushed**.
</content>
</invoke>
