# R11-SEG — pre-registration (FROZEN)

**Date frozen**: 2026-08-18
**Round**: 11, follow-on to `idea-stage/IDEA_REPORT.md` §14 and
`research-wiki/TEMPORAL_SPAN_LANDSCAPE_2026-08-18.md` §10.3 / §14.5.
**Status**: frozen. This file is committed to git *before* any arm metric exists.
No number in the arms/B3 sections below has been computed at freeze time.

---

## 0. What this pilot asks

§14.5 left exactly one live branch on the temporal axis: **HateClipSeg is the only
non-degenerate temporal substrate** available to this project, and its *online
per-timestamp classification* task is formally **temporal action segmentation (TAS)**,
not detection — 100% coverage by construction, no background class, project-scale data.
§10.3 of the landscape names TAS as the **one mechanism family that is not on the
structurally-invalid list** at coverage 0.8-1.0, and notes it has never been pointed at hate.

**The question.** On HateClipSeg's online per-timestamp task, does a TAS-family causal
temporal model beat a per-window independent head (no temporal context) and a
broadcast control, on a frozen split, over ≥10 seeds, with a video-clustered paired
bootstrap CI that excludes zero and clears a pre-declared smallest worthwhile gain?

This is a **method-accuracy** question (Accuracy / Macro-F1), not a localization-metric
question. It is therefore inside the project's method-paper-only rule. §14.5's own two
conditions are carried: no SOTA claim is available on the 90.8% subset (§4 below), and
B3's published modality inversion is reproduced under matched heads *before* anything is
built on it (§6 below).

---

## 1. Data state on disk (verified 2026-08-18, before freeze)

| asset | path | state |
|---|---|---|
| gold segment annotation | `data/gt/HateClipSeg/gold_segments.json` | 395 videos, 10 572 segments, `{duration, platform, n_segments, segments:[[start,end,multihot6]]}`; class order `0 normal, 1 hateful, 2 insulting, 3 sexual, 4 violence, 5 harm`; segments tile each video contiguously from 0 |
| frozen split | `data/gt/HateClipSeg/p11_split.json` | **237 / 39 / 119**, seed 0, stratified on `has_toxic_second`, `n_total=395`; unconsumed before this pilot |
| durations | `data/gt/HateClipSeg/video_durations.jsonl` | 395 |
| visual K=30 | `data/CLIP_Embedding/HateClipSeg/test_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt` | `subclip_img_feats [11850,1024]`, `subclip_parent`, 30 windows × 395 videos, CLIP-L/14-336 pooled over 4 frames per window (120 frames per video) |
| ASR K=30 | `data/ASR/HateClipSeg/test_seen_asrK30_whisper-large-v3.jsonl` | 395 videos, whisper-large-v3, word/sentence `chunks` **plus** pre-computed `window_bounds` and `window_text` on the same K=30 grid; 5 130 / 11 850 windows carry non-empty speech text; all 395 English; `audio_ok` on 394 |
| OCR K=30 | `data/OCR/HateClipSeg/ocr_windows_K30.jsonl` | 11 850 lines, easyocr, one midpoint frame per window; 8 665 windows carry ≥1 detection, 8 331 survive a conf ≥ 0.5 filter |
| 72B MLLM window scores | `data/MLLM_scores/HateClipSeg/test_seen_segscoreK30_*.jsonl` | present, **not used by this pilot** |
| raw video | `/home/jehc223/data/HateClipSeg/videos/` (symlinked at `data/video/HateClipSeg/All`) | 395 files, 4.2 GB, local only |
| **audio features** | — | **absent at freeze time; extracted for this pilot** (§3) |

**Canonical grid.** Window `k` of every channel is the interval stored in the ASR file's
`window_bounds[k]`. Those bounds are derived from exactly the frame partition that produced
the CLIP `subclipK30` tensor (120 frames sampled by `np.linspace(0, N-1, 120)`, split into
30 contiguous groups of 4), so window `k` of the ASR file, the OCR file and the CLIP tensor
are the same interval. Verified on all 395 videos: contiguous, starts at 0.0, ends within
2 s of the ffprobe duration, no gaps. Median window ≈ 8.0 s against a median gold segment
of 8.12 s. Built by `scripts/r11_seg/build_grid.py` into
`idea-stage/r11_seg/out/grid_labels.npz`.

**Design-time descriptive statistics, TRAIN split only** (these are properties of the data,
not of any arm; computed before the freeze and recorded here for auditability):
window offensive base rate **0.4495**; per-timestamp (0.25 s) base rate **0.4521** over
223 690 timestamps; label-change rate between adjacent windows **0.1547**; 73.4% of windows
contain ≥1 raw gold segment boundary (mean 0.853 per window); video-level any-offensive
rate 0.8734; per-window class rates hateful 0.2530, insulting 0.3097, sexual 0.0368,
violence 0.1440, harm 0.0013.

**Consequence recorded now, before results:** the raw-boundary target is near-saturated on
this grid (73.4% of windows contain a boundary) and is therefore **not** usable as the
boundary task in §6. The label-change-point target (15.5%) is used instead, and that
substitution is a declared deviation from the published proposal-level boundary metric.

---

## 2. Task definition (the paper's, verified from arXiv 2508.01712v2 HTML)

HateClipSeg task (3), *Online Hateful Video Classification*: given a temporally aligned
feature sequence, the model at time `t_i` may access only `[t_{i-N}, t_i]` and emits a
**binary** label (offensive = any of the five offensive classes, vs normal) for that
timestamp. The paper's protocol: 32 s context window, 0.25 s stride, metrics
**Accuracy and Macro-F1**. Published baseline (LSTR, 80/20 split, full 435-video corpus):
V 57.99 / 57.52, T 58.86 / 56.51, A 61.05 / 60.84, V+T+A **63.21 / 62.75**; StreamSense
(WWW 2026) reports 72.10 / 72.06 on the same task.

**Our instantiation and where it deviates.** Same label definition (binary offensive vs
normal), same metrics (Accuracy, Macro-F1), same online causality constraint
(the model at window `k` sees windows `0..k` only). Deviations, all declared here:

1. **Temporal resolution.** Predictions are emitted per K=30 window (median ≈ 8.0 s), not
   per 0.25 s timestamp. Evaluation is still *per timestamp at 0.25 s stride*: each
   timestamp takes the prediction of the window that contains it, and its gold label is the
   label of the gold segment that contains it. The metric therefore has the paper's shape;
   the model's temporal resolution is coarser.
2. **Context.** Unbounded causal history (the whole prefix `0..k`) rather than a 32 s window.
   The same unbounded prefix is given to every parameterized arm, including the controls, so
   the arm comparison is not advantaged by it.
3. **Corpus.** The 90.8% surviving subset (395/435) with non-random attrition, and our own
   237/39/119 split, not the paper's 80/20.
4. **Encoders.** CLIP-L/14-336 visual (not ViT-L on the paper's schedule), CLIP text over
   whisper-large-v3 window transcript (not BERT-Base on 2 s windows), OCR text (the paper has
   no OCR channel at all), wav2vec2-emotion (same family as the paper's Wav2Vec-Emotion).

**Comparability ruling, frozen.** Deviations 1, 3 and 4 make our absolute numbers
**not comparable** to 62.75 or 72.06. Those published numbers are quoted for context and
are **not** a gate. The gate in §5 is **internal only** — arm vs arm on the identical frozen
subset and split. Any result document must repeat this and must carry the
`DATASET_hateclipseg.md §4` selection-bias statement.

---

## 3. Features (extraction performed before the freeze; no arm metric computed)

All channels are per (video, window) on the canonical grid, all encoders frozen, no
fine-tuning anywhere in this pilot.

| tag | channel | encoder | dim | script |
|---|---|---|---|---|
| `V` | visual | `openai/clip-vit-large-patch14-336` image tower, mean of 4 frames | 1024 | already on disk |
| `T` | speech text | same CLIP **text** tower over the window's whisper transcript | 768 | `scripts/r11_seg/extract_text_feats.py` |
| `O` | on-screen text | same CLIP text tower over the window's OCR text (conf ≥ 0.5) | 768 | same |
| `A` | audio / prosody | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`, masked mean of last hidden state | 1024 | `scripts/r11_seg/extract_audio_feats.py` |
| `E` | prosody functionals | openSMILE eGeMAPSv02 functionals per window | 88 | same |

Empty ASR/OCR windows get a zero vector plus an explicit presence flag; that is an honest
"no text", not an imputation. Every channel is L2-normalised, then per-dimension z-scored
with statistics fitted on **TRAIN windows only**.

**LoRA note.** The task brief allowed falling back on the deployment pipeline
(frozen Qwen2.5-VL + LoRA) if features were missing. It is not used: the K=30 visual grid
already exists, and the LoRA adapter in this project was trained on a *different* dataset,
so using it here would import an unquantified transfer confound into a first pilot. All
encoders in this pilot are off-the-shelf and frozen. Recorded so the choice is auditable.

Primary input `ALL` = concat[V, T, O, A] + 2 presence flags = **3586 d**.
Secondary input `V` = visual only = 1024 d, run for every arm.
`E` is reserved for the §6 robustness check and is not in `ALL`.

---

## 4. Arms

Every parameterized arm begins with the **same** shared linear projection `D → 256`,
GELU, dropout 0.1, so all arms receive identical input treatment and differ only in the
temporal operator. All arms are causal.

| id | arm | temporal operator | supervision | role |
|---|---|---|---|---|
| **A0** | `CONST` | none | none | predict the TRAIN-majority class at every timestamp. Floor. |
| **A1** | `BCAST-CAUSAL` | causal prefix mean of window features (`mean(0..k)`) → linear | per-window | **broadcast control.** No within-video temporal resolution beyond a running average. This is the §1.3-landscape "video-level prediction wearing a timeline", made online-legal. |
| **A1b** | `BCAST-VIDEO` | whole-video mean → linear, broadcast to all windows | per-window | non-causal reference. **Violates the online protocol**; reported as a diagnostic only, never as a gate. |
| **A2** | `PERWIN` | none (window `k` only) | per-window | **primary comparator.** The "no temporal context" independent-head baseline, i.e. HateClipSeg's trimmed-classification head applied online. |
| **A3** | `MIL-TOPK` | per-window logits, top-33% MIL pooling to a video logit | **video-level label only** | the structurally-invalid family (landscape §10.2; MultiHateLoc's mechanism, its own tuned K). Priced, not expected to win. |
| **A4** | `CTCN` | **the segmentation arm.** MS-TCN-style multi-stage causal dilated TCN: S=2 stages × L=5 layers, channels 64, kernel 3, dilation `2^l`, residual, causal left-padding. Per-window sigmoid. Loss = BCE + λ·truncated-MSE smoothing (MS-TCN defaults λ=0.15, τ=4). **No background class, no top-k pooling, no softmax-over-time, no intra-video contrastive negatives.** | per-window | the candidate |
| **A5** | `CTCN-NOSMOOTH` | A4 with λ = 0 | per-window | ablation: is the TAS smoothing term load-bearing? |
| **A6** | `CTRANS` | causal Transformer encoder, 2 layers, 4 heads, d=256, learned position embedding, causal mask | per-window | the LSTR-family shape (attention over history) — the published-baseline architecture family, distinct from A4's TAS convolution |

Training, **identical for A1, A1b, A2, A3, A4, A5, A6**: AdamW, lr 1e-3, weight decay 1e-2,
batch 32 videos, 40 epochs, cosine schedule, BCE with `pos_weight = 1.0` (base rate 0.4495,
near-balanced), decision threshold fixed at **0.5** (never tuned). Model selection = the
epoch with the best **val** per-timestamp Macro-F1 on the 39 val videos; test is evaluated
once, with that epoch. A0 has no training.

**Seeds.** 12 seeds, **2200–2211**. Seeds control weight init and batch order only; the
split is fixed. Bootstrap seed **2299**. Circular-shift seed **2298**. All chosen outside
the reserved ranges 0-119, 400-429, 500-529, 600-629, 700-729, 1300-1524, 41000-41029, and
outside R13-SPAN's 2000-2021.

---

## 5. Decision rule (frozen; no arm number exists at the time of writing)

**Primary endpoint**: per-timestamp (0.25 s stride) **Macro-F1** on the **test** split
(119 videos), seed-averaged over the 12 seeds. Accuracy is reported alongside but is not
the gate.

**Primary contrast**: `Δ_main = A4 (CTCN) − A2 (PERWIN)` on input `ALL`.
**Secondary gate**: `Δ_bcast = A4 (CTCN) − A1 (BCAST-CAUSAL)` on input `ALL`.

**Uncertainty**: paired bootstrap over **videos** (not windows, not timestamps —
30 windows inside one video are not independent), 10 000 resamples, seed 2299, resampling
the 119 test video ids with replacement and recomputing the metric on the seed-averaged
per-window probability of each arm. Two-sided 95% percentile CI.

**Smallest worthwhile gain**: `δ = +1.0 Macro-F1 point` (0-100 scale). Rationale fixed
here: the published spread between LSTR's weakest and strongest modality configuration on
this exact task is 6.2 points, and the project's standing position is that real, stackable
incremental gains are acceptable; a gain below one point on a 119-video test set is not
distinguishable from encoder or seed noise.

| verdict | condition |
|---|---|
| **GO** | `Δ_main > 0` with 95% CI excluding zero, **AND** `Δ_bcast > 0` with 95% CI excluding zero, **AND** the point estimate of `Δ_main ≥ δ` |
| **AMBIGUOUS** | `Δ_main > 0` with CI excluding zero but point estimate `< δ`, **or** `Δ_main > 0` with a CI containing zero |
| **KILL** | `Δ_main ≤ 0`, **or** `Δ_bcast ≤ 0`, **or** the 95% CI **upper** bound of `Δ_main` is below `δ` (kill by equivalence — the arm is measurably not worth a point) |

Ordering note: the equivalence kill takes precedence over AMBIGUOUS. A CI that excludes
zero *and* lies entirely below +1.0 is a **KILL**, not a win.

**Secondary, reported but not gates**: the same contrasts on input `V`; A5 vs A4 (is the
smoothing term load-bearing); A6 vs A4 (does the TAS convolution beat the LSTR-family
attention shape); A3 vs A2 (the price of the invalid-family mechanism); A1b vs A1 (how much
of the score is non-causal video-level information); per-window Macro-F1 as a robustness
read-out of the per-timestamp primary.

**Test discipline.** Test *labels* are read exactly once, after all training and model
selection are complete, in a single scripted evaluation pass. No threshold, epoch,
hyperparameter, feature set or arm is selected on test. Val (39 videos) carries all
selection. Single submission: the runner is executed once; a crash-and-fix is recorded as a
deviation with the reason.

---

## 6. B3 pre-check — the modality inversion, under matched heads

§14.5 condition 2 requires this **before** anything is built on B3.

**The published claim.** HateClipSeg arXiv 2508.01712v2, verified in round 11: Table 4
localization F1@tIoU 0.5 — visual **52.65** > text 34.60 > audio **25.40**; Table 5 online
per-timestamp Macro-F1 — audio **60.84** > visual **57.52** > text 56.51. Read together:
*prosody labels the moment, pixels draw the boundary.* Those two rows come from **two
different architectures** (ActionFormer vs LSTR) with **two different context windows**, so
the inversion may be an architecture confound rather than a modality fact.

**The minimal matched check, frozen.** A 2 × 2:

- **modality**: `VIS` = CLIP visual (1024 d) · `AUD` = wav2vec2-emotion (1024 d).
  Equal width, both frozen, both z-scored on train, both fed through the *identical*
  `1024 → 256` projection.
- **task**: `LABEL` = per-window binary offensive `y_win` · `CHANGE` = per-window binary
  label-change `y_change[k] = 1[y_win[k] ≠ y_win[k-1]]` (`k=0` → 0), train base rate 0.1547.
- **head**: the **same** A4 `CTCN`, same hyperparameters, same parameter budget, same
  optimizer, same 40 epochs, same val-selected epoch, same 12 seeds 2200-2211. The only
  thing that changes across the four cells is the input tensor and the target vector.
- **grid / context**: identical (K=30, unbounded causal prefix) in all four cells.
- **metric**: `LABEL` → per-window Macro-F1. `CHANGE` → **average precision** for the
  change class (base rate 0.155, so AP not accuracy), with Macro-F1 at threshold 0.5
  reported alongside.
- **CIs**: video-clustered paired bootstrap, 10 000 resamples, seed 2299, on the 119 test
  videos.
- **control**: a **within-video circular shift** of the feature sequence (roll each video's
  30 window vectors by a per-video random offset in 1..29, seed 2298), retrained from
  scratch, run for both modalities on `LABEL`. If shifted-`AUD` matches real-`AUD` on
  `LABEL`, then the audio channel is carrying a *video-level* property (speaker, channel,
  recording condition) rather than a *moment* property, and the claim "prosody carries the
  moment label" is unsupported regardless of the 2×2 outcome.
- **robustness**: the whole 2×2 is repeated with `E` (eGeMAPS 88-d) substituted for `AUD`,
  reported but not a gate — it tests whether the audio result depends on the wav2vec2
  encoder or on prosody as such.

**B3 verdict rule, frozen:**

- **REPRODUCED** iff `AUD > VIS` on `LABEL` with 95% CI excluding zero, **and**
  `VIS > AUD` on `CHANGE` with 95% CI excluding zero, **and** the circular-shift control
  shows real-`AUD` > shifted-`AUD` on `LABEL` with CI excluding zero.
- **NOT REPRODUCED** if either direction fails, or if the shift control fails.
- **CONFOUNDED** if the shift control fails while the 2×2 succeeds — i.e. the inversion is
  visible but the audio channel is not carrying moment-level information.

**Declared deviation.** The published boundary evidence is proposal-level F1@tIoU from
ActionFormer. Our `CHANGE` task is a label-change-point proxy on an ~8 s grid, chosen
because the raw gold-boundary target is near-saturated on this grid (73.4% of train windows
contain a boundary; §1). A negative `CHANGE` result therefore **does not refute** the
published proposal-level number. What this check can conclude is narrower and is exactly
what §14.5 asks: whether the inversion survives when architecture, context window, feature
width, parameter budget and grid are held constant.

**Dependency.** B3 is a *pre-check*, reported in full whatever it says. Its outcome does
not change the §5 arms or the §5 decision rule — those are frozen independently. It gates
only whether any *future* work may build on the modality-asymmetry claim.

---

## 7. Red lines and their concrete implementation

1. **Zero test-label tuning.** Test labels are opened once, by the final scripted eval pass,
   after training and val-based epoch selection. The runner asserts that no test id appears
   in the training or validation id lists and writes the assertion result to the log.
2. **Decision rule frozen before running.** This file is committed to git before
   `scripts/r11_seg/run_pilot.py` is executed. The commit hash is recorded in the result doc.
3. **Blindness.** At freeze time no arm metric, no B3 cell metric, and no val or test score
   of any kind has been computed. The only numbers computed so far are the TRAIN-split
   descriptive statistics in §1 and the extraction coverage counts in §1/§3.
4. **Single submission.** One run of the runner. Any re-run is a recorded deviation with its
   cause; the decision rule is never re-opened.

---

## 8. Cost and scope

Local only, RTX 5090, no cloud, no API, ¥0. Feature extraction: CLIP text ≈ 1 min,
audio (ffmpeg + openSMILE + wav2vec2 over 11 850 windows ≈ 26 h of audio) ≈ 15 min.
Training: ~250 runs of a ≤1 M-parameter model over a `[237,30,3586]` tensor, minutes total.
No raw video leaves the machine.

---

## 9. Files

- `scripts/r11_seg/build_grid.py` — canonical grid + gold labels + train-only statistics
- `scripts/r11_seg/extract_text_feats.py` — CLIP text over ASR / OCR window text
- `scripts/r11_seg/extract_audio_feats.py` — wav2vec2-emotion + eGeMAPS per window
- `scripts/r11_seg/run_pilot.py` — arms, B3, bootstrap, single-submission runner
- `idea-stage/r11_seg/out/` — `grid_labels.npz`, `text_feats.npz`, `audio_feats.npz`, results JSON
- `idea-stage/R11_SEG_PILOT_RESULT.md` — written after the run, KILL causes included
