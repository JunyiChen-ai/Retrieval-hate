# CLAP general-audio channel — G0-cond GATE SPEC (pre-declaration)

**Date:** 2026-07-27 NZST. **Executor:** CLAP lane executor (CPU/SLURM, conda `HateVideo`).
**Repo sha at freeze:** `516c3cb59a1ad47273fd55468caed15b83192ddd`.
**Status of this document:** **PRE-DECLARATION.** Every kill-switch, bar, representation choice,
stratum definition and arm role below is frozen **BEFORE any CLAP feature is extracted** and
therefore before any gate number exists. Numbers are appended to
`refine-logs/CLAP_GATE_RECORD.md`, never to this file.

**User gate:** opened 2026-07-27 (CLAP download authorized). This document authorizes **extraction +
a $0 CPU conditional-information gate only**. It authorizes **no training GPU** and **no prereg**.

---

## 0. The candidate and the honest prior

**Axis:** CLAP (Contrastive Language-Audio Pretraining) **general-audio semantic embeddings** as a new
input stream for HateMM hateful-video detection.

**Target cluster (from `refine-logs/ERRPAT_HateMM_2026-07-26.md` §5, §7, §8):** **FN1 — "speech-poor
visual hate"**, n=7 of 27 stable test errors (25.9%), all 7 wrong in 3/3 seeds. Rule: `y=1` and
≤25 transcript words. Median 6 words, median duration 24.4 s, median gold-span fraction 0.632,
top-20 retrieval purity toward the true label 0.112. Exemplars: `hate_video_329` (**0** transcript
words, 7.1 s, span covers 70% of runtime, 3-seed mean vote −0.7491); `hate_video_10` (4 words,
transcript is `"🎼 In and and.🎼And."` — music, no speech, span 98.8%). The deployed model predicts
**non-hate on 30/30 test items with ≤1 transcript word, in 6/6 seed×protocol cells**
(ERRPAT §4.2). **Ceiling if all 7 FN1 items flip and nothing else breaks: +0.0326 acc**
(1 test item = 0.00465 on n=215).

**Mechanism claim under test.** ERRPAT §4.3 established that the deployed kNN vote inherits the
memory bank's *length-conditional class prior* (train `P(hate | 0-1 words) = 0.1096` vs
`P(hate | 401+ words) = 0.5538`; Spearman ρ=0.5817 between a query's word count and its retrieved
neighbours' word count). A speech-poor hate video therefore lands in a low-base-rate region and the
vote goes negative **before any content evidence is consulted**. CLAP is the only remaining
un-vetoed channel that can supply content evidence for exactly those items, because its signal
(music, chanting, screaming, crowd noise, gunshots, tonal affect, sound effects) is defined on the
*non-speech* audio that both existing channels miss.

**THE PRIOR IS LOW. The audio axis is 0-for-2 and this is the third bite:**

| id | axis | representation | result | scope |
|---|---|---|---|---|
| **F41 / APX** (`APX_GATE_RECORD.md`) | classical prosody | openSMILE eGeMAPSv02, 88-d whole-video functionals | **KILL** — best-k Δacc **−0.0038** CI[−0.0113,+0.0033]; full-88-d **+0.0005** CI[−0.0031,+0.0042]; calib 1.0000 | HateMM only |
| **F64 / LAUD** (`LAUD_GATE_RECORD.md`) | learned speech-ASR encoder | Whisper-large-v3 encoder hidden, mean⊕max, 2560-d | **KILL** — global max over 6 cells × {k8,k16} = **+0.0041**; every CI straddles 0; calib 1.0000 everywhere | HateMM + EN + ZH, both Z arms |

Both nulls have the same published mechanism (**the F31 hazard**): the whisper-large-v3 **transcript**
already banks spoken hate into the deployed `text_feats`, leaving no lexical residual for an audio
channel to add. **The burden of proof is on CLAP.**

**Why CLAP is nonetheless a distinct measurement, not a re-run.** Both prior records scope their own
kills narrowly and both name this successor explicitly:
* APX §5 concedes eGeMAPS "only **weakly** lower-bounds a learned audio encoder";
* `AUDIO_AXIS_FORENSIC_RECON.md` §0.2 (binding on LAUD) states a Whisper null "**must not** close the
  whole learned-audio axis: a **general-audio encoder (AST / BEATs / wav2vec2)** is the proper
  *closer*, or we repeat the APX overreach one level up";
* LAUD §4 repeats this verbatim as the scope of its own kill: Whisper is a **speech-ASR** encoder,
  trained to transcribe speech, and is therefore **weakest exactly on the non-speech events** that
  motivate the axis.

CLAP differs from both predecessors on the axis that matters: it is trained by contrastive alignment
of audio to **free-text descriptions of general sound events**, so its embedding space is organised
by *what the audio is* (a chant, a scream, a music genre, a gunshot) rather than by prosodic scalars
(eGeMAPS) or by phonetic content for transcription (Whisper). This gate is the designated **closer**
for the learned-audio axis.

**Stated prior: LOW, ~10-15%** for the primary global gate; somewhat higher (~20%) for the FN1
stratum read (§5), because the stratum restriction removes the redundancy that killed both
predecessors. **A KILL here closes the general-audio axis** — that is the point of running it.

**D7 novelty status — thin, stated plainly and unchanged from LAUD §4.** "Add audio" is HateMM's own
2023 founding contribution and all three baselines already fuse audio. Even a PASS buys a
**performance / ablation row**, never a method-novelty claim.

---

## 1. Model choice (FROZEN)

**Primary checkpoint: `laion/larger_clap_general`.**

Confirmed from the fetched `config.json` / `preprocessor_config.json` (snapshot
`ada0c23a36c4e8582805bb38fec3905903f18b41`, hub metadata read 2026-07-27, **before** any weight
download):

| property | value |
|---|---|
| `projection_dim` (joint audio-text space) | **512** |
| audio tower | `clap_audio_model` (HTSAT), `hidden_size` **1024** |
| feature extractor sampling rate | **48000 Hz** |
| `nb_max_samples` | **480000** = **10.0 s** at 48 kHz |
| `nb_max_frames` / `feature_size` (mel bins) | 1000 / **64** |
| `padding` / `truncation` defaults | `repeatpad` / `rand_trunc` |
| total repo size | **0.78 GB** |

**Justification vs the FN1 content (required by tasking).** FN1 is *defined by the absence of
speech*: median 6 transcript words, two members with zero. The discriminating content is music,
chanting, ambient and effect audio. Among the LAION CLAP releases:
* `larger_clap_general` is trained on the **broadest** mixture (music + speech + general/AudioSet-class
  events) and therefore has the widest **non-speech event vocabulary** — chanting, crowd, screaming,
  gunshot, sirens, musical genre. This is the on-mechanism match for FN1 and is the primary.
* `larger_clap_music_and_speech` is specialised to music+speech and is **narrower on event sounds**;
  it would re-introduce the speech bias that made Whisper the wrong instrument (LAUD §4).
* `clap-htsat-unfused` is the smaller/older base model (0.62 GB, weaker general-audio coverage).

**No secondary checkpoint will be downloaded or run.** Trying a second CLAP variant after seeing the
first one's numbers is bar-shopping; if the primary fails, the axis fails.

**Download boundary.** Weights are pulled to the shared HF hub cache
`/data/jehc223/home/.cache/huggingface/hub/models--laion--larger_clap_general/` — the same layout as
every existing checkpoint (Qwen2.5-VL-7B, whisper-large-v3, CLIP-L/14-336). Disk headroom at freeze:
**1.5 TB free** on `/data` (0.78 GB required). Only the **audio tower** is used at extraction; the
RoBERTa text tower is loaded but never forwarded.

---

## 2. Extraction recipe (FROZEN)

**Raw media boundary (CLAUDE.md hard rule).** Audio is decoded from the **local** mp4s on-node. Raw
video and raw audio **never leave the machine**; no Modal, no network at extraction time
(`HF_HUB_OFFLINE=1` once weights are cached). Only derived float `.pt` / `.npz` caches are produced.

**Source media.** `data/video/HateMM/All/<id>.mp4` — **1066** files on disk, exactly covering the
744/107/215 split ids in `data/gt/HateMM/{train,val,test}.jsonl`. (The "1083" figure in the tasking
is the `hate_spans.json` entry count, which includes 17 ids outside the splits; the extractable
universe is **1066**.)

**Decode.** `decode_audio_pyav` imported **VERBATIM** from `src/utils/generate_segment_asr_HF.py`
(sha256 `c52cba69648e10b1c87ee4b06182ccd9496a3d274350606e4d61f21bf7a0b394`) — PyAV 17.0.0, no ffmpeg
binary, no torchaudio (absent from `HateVideo`). **`target_rate=48000`, mono** (CLAP's native rate;
the LAUD/Whisper path used 16 kHz — this is the one decode-contract change and it is required by the
model, not chosen).

**Windowing policy.** Per video: split the decoded waveform into **consecutive non-overlapping 10.0 s
windows (480000 samples)**, matching `nb_max_samples` exactly. A trailing short window and any video
shorter than 10 s are handled by CLAP's native `repeatpad`.

> **Determinism note (load-bearing).** The `ClapFeatureExtractor` default `truncation='rand_trunc'`
> is **nondeterministic** for inputs *longer* than `nb_max_samples`. Exact-10 s windowing means no
> input ever exceeds `nb_max_samples`, so the random-truncation path is never entered. This is
> **verified empirically in the smoke test** (§2.1 D-1), not assumed.

**Pooling policy.** Each window yields one embedding. Per video the window embeddings are reduced by
**mean-over-windows ⊕ max-over-windows** (concatenation). Mean captures dominant acoustic content;
**max is the FN1-critical term** — a single 7 s chant or scream inside a 150 s video survives a max
and is washed out by a mean. This is the same mean⊕max motivation the LAUD design used, moved from
*within-chunk frames* to *across-windows*, which is where the transient-event structure lives for a
clip-level encoder.

**Two representations are banked from the SAME forward pass (zero marginal cost):**

| tag | object | per-window dim | pooled video dim | role |
|---|---|---|---|---|
| **`proj`** | `ClapModel.get_audio_features()` — the L2-space **projected joint audio-text embedding** | 512 | **1024** | **BINDING PRIMARY** |
| `hidden` | `audio_model(...).pooler_output` — HTSAT pooled hidden **before** the projection | 1024 | **2048** | pre-declared SECONDARY (§4.4) |

`proj` is binding because it *is* the CLAP object: the language-aligned general-audio semantics that
distinguish this candidate from eGeMAPS and Whisper. `hidden` is the pre-projection analogue of the
Whisper hidden-state block that F64 already killed, and its role is fixed in §4.4 **before** any
number is seen.

**Splits extracted.** All three (`train` 744, `val` 107, `test` 215). Two **separate** cache files
are written so the gate physically cannot reach the test rows:
* `data/audio/HateMM/clap_larger_clap_general_trainval.pt` — **the only file the gate opens**
* `data/audio/HateMM/clap_larger_clap_general_test.pt` — written, then untouched by this lane

> **Certification language this permits (and its limit).** Extracting *inputs* for the test split is
> not a test read: no test label is loaded by the extractor, no test-set metric is computed anywhere
> in this lane, and the gate script hard-codes the `_trainval` path. This is exactly how every other
> feature cache in the repo is built (`test_seen_*.pt` exists for CLIP, frozen-Qwen and LoRA-curric).
> It is a **deliberate, declared deviation** from the APX/LAUD records' stronger "the test split was
> never enumerated" wording, taken so that a PASS does not need a second queue slot. The record will
> state the weaker, true claim and will **not** copy the LAUD sentence.

**Per-video checkpointing.** Each id is written to `data/audio/HateMM/clap_larger_clap_general/<id>.npz`
(`proj`, `hidden`) as computed, so a SLURM reap/requeue resumes rather than restarts (LAUD precedent).

**Canonical id order** = `data/gt/HateMM/train.jsonl` ⊕ `val.jsonl` (train then val, file order) —
bit-identical to the LAUD/APX convention, so the aux block aligns to `Z` by construction.

### 2.1 Compute, and the pre-declared smoke gate

**Resource: CPU-only SLURM, 8 CPU** (per tasking; **not** 16 — the standing infra rule forbids two
concurrent 16-CPU jobs, and other lanes are live). No `--gres`. No `--time` (house rule).
`JobHeldUser` is expected and is **waited out, never force-released**. **Max one job in flight from
this lane**; `squeue -u jehc223` checked immediately before submission (clear at freeze time).

Arithmetic: HateMM total audio 2592.5 min = 155,550 s ⇒ **≈15,600 windows** of 10 s. HTSAT-large
forward on a 64×1000 mel ≈ 0.15-0.4 s/window single-thread ⇒ ≈40-100 min single-thread, well under an
hour 8-way batched; PyAV 48 kHz decode of ~43 h of audio adds ~5-10 min 8-way. **Budget: < 2 h wall,
zero GPU.** If the smoke shows this is wrong by more than 3×, the job is switched to 1 GPU (one job,
coordinated via `squeue`) and the switch is recorded — this is a *compute-path* decision and touches
no bar.

**Pre-declared smoke gate (run first, ~20 videos, before the full job):**
* **D-1 determinism:** the same video encoded twice yields **bit-identical** `proj` and `hidden`
  (confirms the `rand_trunc` path is never entered). **Fail ⇒ stop**, re-specify windowing.
* **D-2 sanity:** no NaN, no all-zero row, embedding norms in a plausible band, window counts match
  `ceil(duration/10 s)`.
* **D-3 discriminative liveness:** `proj` embeddings of two obviously different audio clips are not
  near-identical (guards a silently broken feature-extractor contract). Cosine spread reported.

The smoke reads **no labels** and produces **no accuracy number**; it cannot influence any bar.

---

## 3. Baseline `Z` (FROZEN)

Identical to the LAUD/APX definitions. HateMM only (the acoustic anchor and the only dataset whose
FN1 cluster motivates the channel; F81 scoped CLAP HateMM-only).

| arm | definition | dim | cache |
|---|---|---|---|
| **`deployed_7168`** | the **deployed** winning encoder, img ⊕ text — "does CLAP add over what we actually deploy?" | 7168 | `train,dev_seen_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt` |
| **`strict_8960`** | the exact W2-A/APX/LAUD `Z_best` = CLIP img(1024) ⊕ CLIP text(768) ⊕ frozen-Qwen img(3584) ⊕ frozen-Qwen text(3584) — guards the weak-deployed-encoder loophole | 8960 | `..._openai_clip-vit-large-patch14-336_HF.pt` + `..._Qwen2.5-VL-7B-Instruct_HF.pt` |

**A PASS must clear BOTH arms** (the W2-A "`Z_best` 8960-d is the sole binding" rule, as applied by
LAUD §0). Scope N = 744 + 107 = **851** train∪val, **341 positive**. Per-id label agreement across
every cache is asserted at load. Gold labels are used **PROBE-ONLY** (calibration arm + CV
stratification).

---

## 4. G0-cond gate — machinery and the FROZEN kill-switches

### 4.1 Machinery (inherited VERBATIM, no data-layer novelty except the aux block)

Fork of `scripts/analysis/laud_g0cond_gate.py` (sha256 `b601013a…`), which forks
`apx_g0cond_gate.py` (sha256 `c338de8c…`), which reuses `c3_fusion_probe.py` (sha256 `9091e2c3…`)
**VERBATIM** — the same C3-template conditional-information machinery that rendered the binding
W2-A / K9 / APX / LAUD verdicts. Constants are **unchanged**: `Z` standardized alone at its Z-only
inner-CV-optimal `C_Z` (grid {0.001,0.01,0.1,1.0}, `rs=0`); aux block standardized × `s=50`
(effectively un-penalized), refit at `C_Z`; aux via **train-fold** PCA (leak-free); decision family
k ∈ {8,16} (+ {32,64} reported as context); 5×5 `RepeatedStratifiedKFold` (`rs=1000+rep`); per-video
correctness averaged; per-video-clustered bootstrap **B=5000** on Δacc (`BOOT_SEED=20260714`);
shuffled-aux control (`seed=12345`); permutation null over **≥150** fresh permutations, computed
**only** to confirm a would-be pass.

**The ONLY data-layer change vs LAUD:** the aux block becomes the CLAP video vector (§2). Nothing
else — no constant, no seed, no CV scheme, no bar — is touched, so every number below is directly
comparable to the F41 and F64 graveyard entries.

### 4.2 Primary kill-switches (binding)

* **K-CLAP-1 (calibration).** Label-oracle `accZA < 0.99` → **MACHINERY_INVALID**, no verdict
  credited. (F41 and F64 both scored 1.0000; a failure here means the machinery, not the signal, is
  at fault.)
* **K-CLAP-0 (kill / promote — K9 house standard, bit-consistent with APX/LAUD).** On the **`proj`**
  aux block, binding point = **best of the decision family {k8, k16}**:
  * best-decision-k point **Δacc < +0.040** → **KILL**, **OR**
  * bootstrap **CI-lower ≤ 0** → **KILL**, **OR**
  * (on a would-be pass only) real max-over-k **not > all ≥150 permutation maxima** → **KILL**.
  * **A PASS must clear the triple on BOTH `Z` arms** (§3).
* **K-CLAP-2 (honest-partial band).** `+0.030 ≤ Δacc < +0.040` with CI-lower > 0 on both arms →
  **HONEST-PARTIAL**: a documented near-miss, **NOT an auto-promote**. The machinery bar stays
  +0.040 so the number stays comparable to the graveyard.

**Consequence of KILL (pre-declared).** The general-audio axis is closed: **no training GPU**, **no
prereg**, **no second CLAP variant**, **no AST/BEATs/wav2vec2 escalation**. Combined with F41
(classical prosody) and F64 (speech-ASR encoder), a CLAP null closes the audio axis at all three
representational levels and the record will say so.

### 4.3 Why a global +0.040 bar is retained despite a +0.0326 cluster ceiling (declared tension)

FN1's whole arithmetic ceiling is **+0.0326 test acc**, which is *below* the +0.040 gate bar. Stated
plainly: **the primary gate is deliberately harder than the target cluster's entire ceiling.** Two
reasons this is the right choice and not a rigged one:

1. The two quantities are **not the same measurement**. The gate bar is Δacc on 851 train∪val
   cross-validated items and asks "does this channel carry enough conditional label information
   anywhere in the data to justify GPU"; the +0.0326 is a test-set cluster ceiling. A channel that
   genuinely fixes FN1 would also be expected to move non-FN1 items.
2. Comparability is the asset. F41 and F64 were judged at exactly +0.040. Moving the bar for the
   third audio candidate — after two nulls — would make the third number incomparable to the first
   two and would be indistinguishable from bar-shopping.

The tension is nonetheless **real**: a signal concentrated in a 232/851 (27%) stratum is diluted
~3.7× at the global level, so a within-stratum effect of +0.15 is needed to register +0.040
globally. That is precisely why the stratum read in §5 exists, is **declared now**, and has its own
strict, independent, pre-registered bars — rather than being reached for after a disappointing
primary number.

### 4.4 The `hidden` secondary arm — role fixed BEFORE any number

The pre-projection `hidden` block (2048-d) is run through the identical machinery and **fully
reported**. Its role is fixed here:
* `hidden` **cannot produce a PASS.** The binding object is `proj` (§2).
* If `hidden` clears the triple on both arms while `proj` does not, the outcome is recorded as
  **DISCORDANT**. That is **not** a promote and **not** a prereg trigger; it is a documented
  escalation *request* returned to the main loop for a ruling. This rule exists so that a real
  signal in the pre-projection space is neither buried nor allowed to become a back-door GO.
* Reporting both costs one extra CPU gate run and zero extra extraction.

---

## 5. K-CLAP-3 — the FN1-targeted stratum read (SECONDARY, narrow path, bars frozen here)

**Motivation.** §4.3: the global bar systematically under-detects a stratum-localised signal, and the
FN1 mechanism predicts *exactly* a stratum-localised signal. This read is declared **before**
extraction so it cannot become a consolation prize.

**Stratum definition (frozen).** Train∪val items with **`len(gt["text"].split()) ≤ 25`** — the
identical rule and identical word-count function used to define FN1 in `ERRPAT_HateMM_2026-07-26.md`
§5 (`errpat_hatemm_clusters.py:131`). Counts computed from the gt files at freeze time, **no labels
beyond these counts consulted**:

| stratum | train | val | **train∪val** | hate | P(hate) |
|---|---|---|---|---|---|
| **≤25 words (FN1 rule) — PRIMARY stratum** | 194 (34 hate) | 38 (8 hate) | **232** | **42** | **0.1810** |
| ≤1 word (empty-transcript analogue) — context only | 73 (8 hate) | 14 (0 hate) | 87 | 8 | 0.0920 |

The ≤1-word bin has **8 positives** and is reported as **underpowered context only**; it can render
no verdict. The ≤25-word stratum (42 positives) is the unit of K-CLAP-3.

**Metric: out-of-fold ROC AUC, not Δacc.** Within the stratum the base rate is 0.1810, so a
constant-predict-non-hate rule already scores 0.819 accuracy and Δacc has almost no dynamic range.
AUC asks the question that actually matters — *does CLAP rank the 42 speech-poor hate videos above
the 190 speech-poor non-hate videos at all?* Same 5×5 `RepeatedStratifiedKFold` CV, logistic on the
standardized `proj` block, out-of-fold scores pooled; per-video-clustered bootstrap B=5000 for the CI.

**Three reads, all reported:**
* **(a) marginal:** CLAP-alone stratum AUC.
* **(b) head-to-head vs the killed predecessor:** the identical stratum AUC computed on the **already
  banked** Whisper-large-v3 block (`data/audio/HateMM/whisper_whisper-large-v3_trainval.pt`, sha256
  `4a6b0bb2…`, same 851 ids, same order). **Free**, and it is the direct test of the LAUD §4 scope
  claim — *is general-audio semantics better than an ASR encoder where speech is absent?*
* **(c) conditional:** ΔAUC of `[Z_deployed ⊕ CLAP]` over `Z_deployed` alone, restricted to the
  stratum.

**K-CLAP-3 bars (FROZEN):**
* **KILL-side** iff stratum AUC **≤ 0.60** OR bootstrap CI-lower **≤ 0.50**.
* **NARROW-GO** iff **all three** hold:
  1. stratum AUC **≥ 0.65** with bootstrap **CI-lower > 0.55**; **and**
  2. CLAP stratum AUC **exceeds the banked Whisper stratum AUC by ≥ +0.05** (establishes that the
     signal is *general-audio semantics*, not merely "some audio encoder", since Whisper audio is
     already a measured null); **and**
  3. conditional ΔAUC over `Z_deployed` within the stratum **> 0 with bootstrap CI-lower > 0**
     (establishes the signal is *additive*, not a re-encoding of what `Z` already has).
* Anything between KILL-side and NARROW-GO is **INCONCLUSIVE-NARROW** and is treated as a KILL for
  action purposes (no GPU, no prereg) while being reported honestly.

**What a NARROW-GO buys, and what it does not.** It authorizes **only** the drafting of a prereg for
a *speech-poor-targeted* cell (CLAP fused as a third stream, evaluated with FN1 as the pre-declared
target), and it authorizes **no submission** — the formal prereg-review ceremony and the main loop's
ruling stand between it and any GPU. It **never** supports a general "audio helps hateful-video
detection" claim; the two prior nulls own that statement.

**Power, stated honestly and in advance.** With 42 positives and 190 negatives the standard error on
an AUC near 0.65 is ≈0.045, so the CI-lower>0.55 requirement sits close to the edge of what this
stratum can resolve. This is a deliberate design limit: it is the *only* stratum the FN1 mechanism
licenses, and inflating it by relaxing the word threshold after seeing numbers is forbidden by this
document. **The ≤25-word threshold is frozen and will not be moved.**

### 5.1 Pre-declared confound diagnostics (reported, non-binding)

ERRPAT §4.3 showed transcript length is a *bias direction in the deployed score*. A CLAP embedding
could re-encode "this clip is short / has no speech / is music" and thereby correlate with the same
length prior rather than with hatefulness. Therefore the record will also report:
* Spearman ρ between the CLAP-alone out-of-fold score and `n_words`, globally and within the stratum;
* the shuffled-aux control (already in the machinery) for every arm;
* the stratum-restricted class base rates above.
These are **diagnostics**: they contextualise a result but move no bar.

---

## 6. Post-gate plan (declared in advance)

| outcome | action |
|---|---|
| **K-CLAP-1 fires** (calib < 0.99) | MACHINERY_INVALID — no verdict credited; debug machinery, re-run once, no bar change. |
| **KILL** (K-CLAP-0 fires; K-CLAP-3 KILL-side or INCONCLUSIVE-NARROW) | Write `CLAP_GATE_RECORD.md` with the full number set, log as the **3rd pre-registered audio negative**, declare the **general-audio axis closed** (all three representational levels: classical prosody F41, speech-ASR encoder F64, general-audio semantics F90). **Stop.** No GPU, no prereg, no second variant, no AST/BEATs escalation. Return the verdict to the main loop. |
| **HONEST-PARTIAL** (K-CLAP-2) | Documented near-miss. No auto-promote. Returned to the main loop as a ruling request with the full numbers. |
| **DISCORDANT** (`hidden` passes, `proj` does not, §4.4) | Documented escalation request to the main loop. No prereg by this lane. |
| **NARROW-GO** (K-CLAP-3 clears all three conditions, K-CLAP-0 did not fire MACHINERY_INVALID) | Draft — **not submit** — a prereg for the FN1-targeted 3-seed audio-third-stream cell; hand to the main loop for the prereg-review ceremony. |
| **PASS** (K-CLAP-0 clears the triple on both `Z` arms) | Draft — **not submit** — the prereg for the formal 3-seed audio-third-stream cell (CLAP into the `align` Hadamard head), with its own oracle/KS bars. Hand to the main loop for review + freeze. |

**In no branch does this lane submit a training job, touch a test metric, or write to
`autoresearch/.../state/`.**

---

## 7. Artifacts and discipline

**To be written by this lane:**
`scripts/analysis/clap_extract.py`, `scripts/slurm/clap_extract.sbatch`,
`scripts/analysis/clap_g0cond_gate.py`, `refine-logs/CLAP_G0COND_GATE_OUT.json`,
`refine-logs/CLAP_G0COND_GATE_run.log`, `refine-logs/CLAP_GATE_RECORD.md`, this spec, and derived
caches under `data/audio/HateMM/` (gitignored). Every script sha256 is recorded in the record.

**Discipline honored (to be certified in the record):**
* **G0-cond is mandatory before any training GPU** — this gate *is* that gate; no GPU head is spent
  either way.
* **Raw media never leaves the machine** — local ffmpeg-free PyAV decode; no Modal; the
  `modal_probe_runner.py` hard block is untouched.
* **SLURM for anything heavy**, `sbatch`, no `--time`, `JobHeldUser` waited out, **one job in
  flight**, 8 CPU (never two concurrent 16-CPU jobs).
* **Single gate read.** Bars frozen in this document before any feature exists. No bar, threshold,
  stratum boundary or arm role is adjusted after seeing a number. If any number below is re-run, the
  re-run and its reason are logged.
* **Numeric provenance:** 4 dp, every number transcribed **verbatim** from
  `CLAP_G0COND_GATE_OUT.json`; no companion metric fabricated.
* **Test contact:** no test label read, no test-set metric computed, gate hard-codes `_trainval`.
  The weaker-but-true certification of §2 is used; the LAUD "never enumerated" sentence is not
  copied.
* Local commits only, **never pushed**.

---

**FROZEN 2026-07-27, before any CLAP weight was downloaded and before any feature was extracted.**
