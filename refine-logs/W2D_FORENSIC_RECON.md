# W2-D Forensic Recon — Acoustic-channel retrieval (new-input axis)

**Agent:** zero-GPU / zero-SLURM / zero-Modal forensic **PRE-recon** (pure code + cache + disk + lit-note
reading; small CPU-seconds `ls`/`grep`/config inspection only — no vote, no probe, no extraction).
**Target:** wave-2 candidate **W2-D** ("acoustic-channel retrieval", new-input axis). Ranked **#4/5** in
`research-wiki/ROUND3_CANDIDATES_WAVE2_2026-07-15.md`, prior LOW–MODEST, novelty "HONEST, THIN"; currently
gated **"queue only if A/B/C stall."** This recon exists to keep the ammo pool loaded, not to promote.
**Conditioning kills read in full:** `W2B_VERDICT_REVIEW.md` (W2-B KILLED, outcome (d)), `W2E_FORENSIC_RECON.md`
(W2-E NO-GO at recon, D7 meta-family), `W2C_FORENSIC_RECON.md` + `W2C_ORDER_PRECHECK_RECORD.md`.
**Ground truth read (not memory):** `src/utils/metrics.py:262-320` (top-20 rank-weighted signed-cosine vote;
sim is a multiplicative WEIGHT `:268-270`), `src/model/evaluate_rac.py:80-155` (memory = `IndexFlatIP`, one
L2-normed head-projected vector per train video). Disk / HF-cache / dataset facts verified this session (§A,§D).
**Repo HEAD at recon:** `ad48dcc`. Zero GPU / SLURM / Modal used.

---

## VERDICT LINE

> **CONDITIONAL — REMAIN "only-if-stall"; NOT promotable to probe-design as posed.** W2-D is the one wave-2
> candidate that adds a genuinely new *signal source* (acoustic), but it faces **two structural gates it does
> not clear on current evidence, and a third it can only clear via a user-ruled new-model download:**
> **(1) D7 novelty — FAILS with any local encoder.** The only audio-capable models on disk are Whisper
> (base + large-v3); **Whisper is an ASR model, NOT an MLLM**, so a Whisper-encoder acoustic channel carries
> **zero MLLM content** → cannot satisfy D7-tightened ("the MLLM must be load-bearing"), and "add an audio
> channel" is **literally the 2023 HateMM contribution** (BERT⊙ViT⊙**MFCC**) that *all three* benchmark
> baselines already implement (HateMM MFCC/AudioVGG19; MM-HSD wav2vec2; MultiHateClip MFCC/Wav2Vec2-BERT).
> So the local version is a **pure catch-up, performance-only side-dish** with no novel-contribution path.
> **(2) The only D7-defensible variant needs a NEW MODEL DOWNLOAD (user ruling required).** An
> audio-*capable MLLM* (Qwen2.5-Omni / Qwen2-Audio) is **absent from disk** — Qwen2.5-VL has **no audio
> tower**, nor does any VLM on disk (InternVL*, Qwen3-VL, LLaVA-NeXT-Video) — and prior 32B/72B download
> attempts failed/are-absent (MEMORY). The "audio-visual temporal-correspondence as a retrieval structure"
> composite is **thin and adjacent to the already-KILLED W2-K1 incongruity line**.
> **(3) Signal prerequisite — CONDITIONAL headroom is thin.** Audio *does* carry hate signal (HateMM
> audio-only macro-F1 **0.669**), but it contributed only **~2–3%** to a *weak* 2023 fusion (Vosk 22%-OOV ASR
> + ViT+LSTM). Our pipeline's text channel is **Whisper-large-v3 ASR → mpnet** (far stronger) and vision is
> CLIP-L/336 / Qwen-7B — so the **conditional** acoustic headroom *given the strong Z_best* is plausibly **at
> or below the +0.040 G0-cond bar** (audio's linguistic content is already ingested by a good ASR; only
> paralinguistics remain).
> **Conditions to ever promote:** (i) **all** higher-ranked live lines exhausted/stalled (W2-A lead needs its
> grounded extraction; W2-C rides S2S frameset; S2S itself untested); AND (ii) a **user ruling** on either
> authorising a new audio-MLLM download OR explicitly accepting a non-novel, performance-only Whisper-encoder
> audio side-dish; AND (iii) it must pass a **cheap, local, zero-GPU-ish Whisper-encoder G0-cond
> conditional-info gate FIRST** (the correct pre-GPU kill — if audio-features-vs-`concat(Z)` shows < +0.040
> oracle headroom the line dies for ~$0 regardless of the novelty/download question).
> **Prior: LOW.** On current evidence it will not clear the novelty gate locally and likely not the G0-cond
> signal gate either; keep it parked as ammo, do not spend ceremony/GPU.

**One-line:** *W2-D is real-new-signal but D7-failing-locally, download-gated for its only novel variant, and
conditionally-thin on signal — stay only-if-stall; the honest promotion path is a local Whisper-encoder
G0-cond probe as a cheap kill, not a novelty bet.*

---

## A. MECHANISM — sharpest variant, what acoustic info is new, where it slots, which encoder

**What the pipeline captures today (verified).** Frames + transcript-text only. Text channel =
**Whisper-large-v3 ASR** (`data/ASR/{HateMM,MHC,MHC_zh}/{split}_asrK{4,30}_whisper-large-v3.jsonl`) →
sentence-transformer **mpnet512** embeddings (`data/CLIP_Embedding/*/{split}_transcript_mpnet512_HF.pt`) and,
for sub-clips, CLIP-text of the window transcript (`generate_subclip_mm_embedding_HF.py`). Vision =
CLIP-ViT-L/336 pooled and Qwen2.5-VL-7B pooled (3584-d). Memory/vote = `evaluate_rac.py:80-155` +
`metrics.py:262-320` (one vector/video, top-20 rank×signed-cosine).

**What acoustic info the ASR transcript does NOT capture (the true residual).** A strong ASR transcript
already ingests the *lexical/semantic* speech content (slurs, threats, named targets). What remains
genuinely acoustic-only = **paralinguistics**: aggression prosody (shouting/anger intonation), speaker
affect, **music genre / hateful chants**, sound effects, **laughter over a slur**, crowd/mob noise, clapping,
gunshots. That is the *only* marginal bandwidth W2-D can add over Z_best — and it is a **narrow, paralinguistic
residual**, not the bulk of the acoustic hate signal (most of which the Whisper transcript has already
transcribed into the text channel). This is the crux of the conditional-info problem in §C.

**Sharpest variant + where it slots.** To respect the diagnosis frame, the acoustic key must enter as a
**representation-level feature channel concatenated into the retrieval key (D2-favoured)** — e.g.
`key = L2norm([Z_best ⊕ a])` where `a` = a frozen audio-encoder vector per video, fed through the *unchanged*
top-20 kNN vote — **NOT** a decision-side audio-vote re-weighting or an audio-agreement scalar (that is the
**D1 death** pattern the frame names: a low-bandwidth decision signal). The candidate text's option (b)
(audio-visual temporal-correspondence keys) is the only variant with a novelty sliver but is thin (§B).

**Which encoder — what is already on disk (verified, full HF-cache scan
`/data/jehc223/home/.cache/huggingface/hub/`):**
- **Whisper-base** (`d_model=512`, 6 enc layers, 80 mel) and **Whisper-large-v3** (`d_model=1280`, 32 enc
  layers, 128 mel) — the **only** audio-capable models on disk. Whisper's **encoder hidden states** are a
  well-known cheap general-audio feature (mean-pool the encoder output) → **already-local, no download**.
- **NO** CLAP, wav2vec2/wavLM/HuBERT/data2vec, AST/BEATs/PANNs/VGGish, EnCodec, SpeechT5, or any **audio
  MLLM** (Qwen2.5-Omni / Qwen2-Audio) anywhere in the cache. Explicitly checked.
- **CRITICAL INFRA FLAG — Qwen2.5-VL has NO audio tower.** Neither does any other MLLM on disk (InternVL2-8B,
  InternVL3-38B/78B/3.5-30B/38B, Qwen3-VL-8B/235B, LLaVA-NeXT-Video-34B) — all are vision+text. An
  **audio-capable MLLM is a NEW MODEL DOWNLOAD** (Qwen2.5-Omni-7B ≈ tens of GB). Per MEMORY, prior 32B/72B
  download attempts **failed / are absent despite logs**. This is a **hard user ruling**, not an assumption.

**Does the local Whisper option satisfy D7 "MLLM integration"? — NO (dispositive).** Whisper is a speech
**recognition** model (encoder-decoder ASR), categorically **not an MLLM**. A Whisper-encoder acoustic channel
is an **encoder-class feature add** — the exact D7-tightened anti-pattern ("encoder swaps / generic tricks
with an MLLM coat of paint"), except here there is not even a coat of paint: the audio encoder is not an LLM
at all. So **W2-D with a local encoder structurally FAILS the novelty clause** and can only ever be a
**performance side-dish**, not a contribution — and even the performance is catch-up (§B).

---

## B. NOVELTY vs D7 — audio-for-hate-video is a SOLVED 2023 baseline, not a mechanism

**Standalone: FAILS D7-tightened, hard.** "Add an audio channel to hate-video detection" is not novel — it is
**the founding HateMM 2023 contribution**: the SOTA-anchor model is literally BERT⊙ViT⊙**MFCC** (audio),
`das2023` note L19/L25/L28. Every core benchmark baseline already fuses audio:
- **HateMM (das2023):** AUDIO = MFCC / AudioVGG19; fusion = BERT⊙ViT⊙MFCC → acc 0.798 / **M-F1 0.790**.
- **MM-HSD (2508.20546, current HateMM video SOTA M-F1 0.874):** audio = **wav2vec2-xlsr-53**, tetra-modal
  (T+A+V+OCR); a listed baseline is **HXP+CLAP+CLIP 0.848** (CLAP = audio-text contrastive).
- **MultiHateClip (wang2024):** audio baselines **MFCC / Wav2Vec2-BERT**; late-fusion M1 = mBERT⊙MFCC⊙ViViT;
  the dataset ships **contributing-modality attribution labels** — audio is a first-class, already-studied
  modality on the exact binding-gap dataset.

So W2-D standalone is a **catch-up to a 2-year-old baseline**, not a novel mechanism. Our paper's
baseline-comparison story makes this worse: our numeric targets **are** these audio-fusing models
(HateMM 0.790, MM-HSD 0.874). "We added audio too" concedes the differentiator; the paper's stated novelty
axis is **retrieval-guided contrastive**, and audio does not live there.

**Composite: THIN, and adjacent to a KILLED line.** The only defensible framing is "audio-visual
**temporal-correspondence** carried in a *retrieval representation* (not plain late-fusion)" — first use of
acoustic-visual co-occurrence geometry in a hate-video contrastive kNN head. But (i) it is a small sliver,
(ii) it is **adjacent to W2-K1** (cross-modal incongruity/co-occurrence), which was **KILLED at ideation**
(D1 + mature sarcasm/incongruity prior art), and (iii) to be genuinely **MLLM-load-bearing** it needs an
**audio MLLM** (new download) — a plain frozen audio encoder makes the "MLLM" adjective inert exactly as it
was for W2-E's k-means. Net: no clean novel-contribution path without the download, and even with it the
mechanism is thin and prior-art-adjacent.

**Non-isomorphism vs graveyard (mechanism):** no dead route uses audio (P1–P11/TARC/B*/C* are vision+text);
not OCR (distinct channel, but OCR is user-vetoed and MM-HSD's gain came from OCR, not audio — so we cannot
even borrow their fusion trick); not an encoder-*swap* (adds a modality). **No ban collision** — but D7 and
the download question are the walls, not the graveyard.

---

## C. SIGNAL PREREQUISITE + G0-cond ORACLE SKETCH

**Zero-GPU signal evidence (lit notes).** Audio carries *some* standalone hate signal but is the **weakest
unimodal** and its *marginal* is small:
- **HateMM unimodals (das2023 L28):** text HateXplain **0.733 macro-F1** / vision ViT **0.733** / **audio
  AudioVGG19 0.669** (acc 0.690). Audio-only is real (≫ chance) but below text and vision.
- **Marginal contribution:** "**all modalities contribute ~2–3% each**" to the 0.790 fusion — measured
  against **weak** companions (Vosk ASR, ~22% OOV; ViT+LSTM; fastText/BERT).
- **MM-HSD ablation:** best *single* modality = **transcript 0.816**; audio is a contributor but the headline
  CMA lift (0.846→0.874) is driven by **OCR-as-query** (vetoed for us), not audio.

**The conditional-info problem (why the prior is LOW).** The 2–3% audio marginal was earned against a
*weak* text/vision stack. **Our** text channel is Whisper-large-v3→mpnet (far better ASR than Vosk) and vision
is CLIP-L/336 or Qwen-7B (the encoder-swap positive, +5.3–5.6 on HateMM). The stronger and more complete the
transcript + visual scene, the **smaller** the residual acoustic headroom — because a good ASR has already
transcribed the spoken hate into the text channel, leaving only the paralinguistic residual (§A). So
`I(label ; audio | Z_best)` — the exact quantity G0-cond measures — is plausibly **at or under +0.040**.
This is not a prediction of zero (paralinguistics are real), but it is the honest reason the prior is LOW,
not MODEST.

**G0-cond gate sketch (mandatory before any auxiliary-signal GPU).** Reuse the W2-B/S2S instrument verbatim:
- **Sole primary contrast:** paired LOO kNN, **`concat(Z_best ⊕ a)` vs `Z_best`** in acc AND macro-F1, where
  `Z_best` = the honest strong baseline `concat(Qwen_img, mpnet_text)` (the 7168-d flat key W2-E used) and
  `a` = frozen audio-encoder vector. Encoder / pooling / fusion-weight are **sensitivity-only, never
  survival-determining** (bars hyperparameter shopping, mirrors W2-B B2).
- **Oracle kill-switch FIRST (admissible, non-leaking):** the W2-B pattern — gold picks, *per query*, whether
  to trust the audio-augmented or the audio-free key (i.e. does an audio-improved neighbourhood *exist*);
  memory side never oracle-selected. **DEAD iff oracle Δ(oracle − Z_best) < +0.040 on every dataset.** Do NOT
  use gold-label audio-routing (label-leaking, inadmissible).
- **Fano ≥ 0.99** (±1 gold-label key vote) — verdict admissible only if the vote machine is valid.
- **Conversion-taxed raw bars:** HateMM anchor Δacc AND ΔmF1 ≥ **+0.05** vs Z_best (rank-only corroborated);
  MHC-EN survival Δ ≥ **+0.03/+0.03** (the candidate's own falsifiable bar).
- **Permutation null ≥ 100** (audio-key ↔ video shuffled across videos, same-perm both arms); **bootstrap
  1000**, paired Δ 5th-pct > 0 (D3-fragility guard). Fail-closed: no `test_seen`; assert memory
  V == 851/629(/657).

**Extraction cost (raw audio → features).** Fully **local** (raw video **never** to cloud):
`ffmpeg -i video.mp4 -ac 1 -ar 16000 wav` (CPU, seconds/video; `ffmpeg`/`ffprobe` present at
`/data/jehc223/miniconda3/envs/ExMRD/bin/`) → **Whisper-encoder forward, mean-pooled** (whisper-base CPU-OK;
large-v3 GPU-light) → one `a` vector/video. Only the derived `.pt` audio features (+ labels) are Modal-eligible
for the CPU probe. A CLAP/wav2vec2 route would be a stronger audio feature but is **another download**.

---

## D. DATA / INFRA PLAN

**Raw-video availability (verified on disk — corrects the candidate's "MHC-only" claim).** Raw `.mp4` are
present locally for **all three** datasets, so audio is extractable everywhere (not just MHC):
`data/video/HateMM/All` = **1066** mp4, `data/video/MHC/All` = **790**, `data/video/MHC_zh/All` = **806**.
(Whether every clip has a non-silent audio track is unverified at recon — BitChute/YouTube/Bilibili sources
generally do; a `ffprobe` silent-fraction pass is a cheap first extraction-time check. **No silent-video
fraction is documented** in the wiki notes.)

**Audio cache status.** `data/audio/MHC` **exists but is EMPTY** (0 bytes, confirmed) — matches the candidate
text; there is **no** `data/audio/HateMM` or `data/audio/MHC_zh` dir at all. **No audio features are banked;
extraction is required before any probe.**

**Extraction pipeline sketch (local queue).** For each split of each targeted dataset: mp4 → 16 kHz mono wav
(ffmpeg, CPU) → Whisper-encoder mean-pool (or wav2vec2/CLAP if a download is authorised) → write
`data/audio/<DS>/{split}_audio_<enc>.pt = {ids, audio_feats, labels}` mirroring the pooled-feature cache
schema so the W2-B loader/vote/oracle/Fano/null/bootstrap machinery is reusable verbatim.

**Caches land** in `data/audio/<DS>/` (local); derived `.pt` + labels sync to Modal volume `rgcl-features`
for the CPU probe. **Modal eligibility:** derived audio features only ✓; **raw audio ⊂ the raw-video license
question → keep local** (the `modal_probe_runner.py` hard block on raw video stands). MHC-EN is the binding
gap and is the priority target; HateMM is the anchor and is also extractable.

---

## E. KILL-BAR SKETCH (per house discipline)

**G0-cond FIRST** (§C) — conditional info of the audio-augmented key beyond `Z_best`, oracle-gated at +0.040
on every dataset. This is the pre-GPU kill and the cheapest exit: it can be run on **Whisper-encoder features
alone** (local extraction, CPU probe) *before* any audio-MLLM download or novelty ruling — if there is no
convertible acoustic signal, the whole line dies for ~$0 and the D7/download question never has to be asked.

Then, only if G0-cond survives and the user rules the novelty/download question, the standard probe bars:
sole-primary-arm (concat-vs-Z_best, one encoder pre-declared), oracle kill-switch, Fano ≥ 0.99, conversion-
taxed +0.05 (HateMM) / +0.03 (MHC-EN), perm-null ≥ 100, bootstrap 1000 with 5th-pct > 0, fail-closed on
`test_seen`. Pre-declared expectation (falsifiable): the concat key Δ lands within the perm-null on ≥1 dataset
because the paralinguistic residual is small given a strong ASR transcript.

---

## F. PRIOR — LOW; verdict CONDITIONAL (stay only-if-stall)

**Prior: LOW** (candidate said LOW–MODEST; recon lowers to LOW, conditioned on D1/D2/D3 + the fact that ASR is
already in the pipeline + the baseline-catch-up novelty wall).

- **D1 (decision-side low-bandwidth redundant, 5 prior kills):** does not bite *if* the audio key is
  representation-level concat (D2); bites hard if it degrades to an audio-vote/agreement scalar — so the
  design must stay representation-level. Manageable, not fatal.
- **D2 (only representation-level levers cleared +3):** W2-D *can* enter the winning class (new high-bandwidth
  input channel is REFLECTION §3.2's second D2 class) — this is its one genuine advantage over W2-E. But
  entering the class ≠ clearing it; the conditional-info evidence (§C) says the realized headroom is thin.
- **D3 (±1–2pt noise floor):** MHC-EN is data/label-limited (SAV); a thin paralinguistic marginal is squarely
  in the noise band, so even a nominal positive would be fragile.
- **Novelty is the dispositive wall, not performance.** Even if the number came back positive, the **local**
  version yields **no novel contribution** (Whisper ≠ MLLM; audio-fusion = 2023 baseline). Unlike W2-C (which
  escapes into a novel-if-accepted D2 mechanism at ~0 marginal cost), W2-D's only novel path is
  **download-gated** (audio MLLM) and **thin/adjacent-to-killed** (temporal-correspondence ≈ W2-K1). This is
  closer to the W2-E logic ("cannot yield a contribution regardless of the number, at real ceremony/GPU cost")
  — except W2-D at least has a real signal source, which is why it stays parked as ammo rather than a
  recon-KILL.
- **Baseline-catch-up risk is real and central:** the paper's targets already fuse audio; adding it concedes
  the retrieval-contrastive differentiator.

**Recommended gating: STAY "only-if-stall" (do NOT promote, do NOT hard-kill).** Keep W2-D loaded as the
new-signal-source fallback. It becomes probe-worthy **only** when: (i) W2-A / W2-C / S2S / W2-B are all
exhausted or stalled; AND (ii) the user issues a ruling — either **authorise a new audio-MLLM download**
(Qwen2.5-Omni/Qwen2-Audio; flag prior-download failures) so the composite can be genuinely MLLM-load-bearing,
OR **explicitly accept a Whisper-encoder audio channel as a performance-only, D7-non-novel side-dish** for the
paper; AND (iii) it clears the **local Whisper-encoder G0-cond conditional-info gate** run FIRST as the cheap
pre-GPU kill. Absent (i)–(iii), no ceremony, no GPU.

*Falsifiable (candidate's, endorsed):* if an audio-fused retrieval key does not beat `Z_best =
concat(img,text)` by a paired +0.03 on MHC-EN (and +0.05 on HateMM) with oracle headroom > +0.040, hate's
acoustic signal is already banked in the transcript/visual pathway and W2-D is dead.

---

## Provenance
- Code: `src/utils/metrics.py:262-320` (vote; sim-as-weight `:268-270`), `src/model/evaluate_rac.py:80-155`
  (memory = IndexFlatIP, one vector/video).
- HF cache (full scan, `/data/jehc223/home/.cache/huggingface/hub/`): audio models = **only** whisper-base
  (`d_model=512`) + whisper-large-v3 (`d_model=1280`, 32 enc layers, 128 mel); NO CLAP/wav2vec2/HuBERT/AST/
  BEATs/Omni/Qwen2-Audio; MLLMs on disk (Qwen2.5-VL-7B, Qwen3-VL, InternVL*, LLaVA-NeXT-Video) all lack an
  audio tower.
- Datasets: raw mp4 present `data/video/{HateMM:1066, MHC:790, MHC_zh:806}/All`; `data/audio/MHC` EMPTY, no
  HateMM/MHC_zh audio dir; text channel = `data/ASR/*/*_whisper-large-v3.jsonl` → `*_transcript_mpnet512_HF.pt`;
  `ffmpeg`/`ffprobe` at `/data/jehc223/miniconda3/envs/ExMRD/bin/`.
- Lit: `research-wiki/papers/das2023_hatemm_multimodal_dataset.md` (audio MFCC/AudioVGG19 0.669; ~2–3%/modality;
  Vosk ASR), `research-wiki/papers/cspedessarrias2025_mmhsd_multimodal_hate.md` (wav2vec2, CLAP baseline,
  transcript 0.816, SOTA 0.874, OCR-driven), `research-wiki/papers/wang2024_multihateclip_multilingual_benchmark.md`
  (MFCC/Wav2Vec2-BERT, modality-attribution labels), `research-wiki/paper.bib`.
- Kills/bans: `refine-logs/W2B_VERDICT_REVIEW.md`, `refine-logs/W2E_FORENSIC_RECON.md`,
  `refine-logs/W2C_FORENSIC_RECON.md`, `refine-logs/W2C_ORDER_PRECHECK_RECORD.md`,
  `autoresearch/goal_mllm_plus3/state/directions_tried.json` (dead ids + banned_constraints incl. OCR veto,
  W2-K1 incongruity kill).
- Candidate text: `research-wiki/ROUND3_CANDIDATES_WAVE2_2026-07-15.md` §W2-D (ranked #4/5).
- Repo HEAD at recon: `ad48dcc`. Zero GPU / SLURM / Modal used.
