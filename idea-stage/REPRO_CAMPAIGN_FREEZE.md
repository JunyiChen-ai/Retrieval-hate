# REPRO_CAMPAIGN_FREEZE — label-free frame-level baseline reproduction, unified protocol

Frozen 2026-08-19. **Nothing in this file may be changed after the commit that introduces it.**
Any later change is a numbered deviation appended to §12, with the reason and the date, and never a
silent edit.

Campaign source plan: label-free frame-level baseline reproduction (user rulings 2026-08-19).
Source shortlist: `research-wiki/LABELFREE_FRAMELEVEL_BASELINES_2026-08-18.md` (ea24981).
Source benchmark measurement: `research-wiki/TEMPORAL_SPAN_LANDSCAPE_2026-08-18.md`.

## 0. What this campaign is and is not

This is a **baseline table**, not a candidate trial. Every method listed in §9 is run once under the
protocol below and its numbers are **reported**. No method receives a GO / KILL verdict, no method
is "promoted", and no decision rule in this document selects a winner. The only pass/fail judgement
anywhere in the campaign is the **transplant-fidelity check** in §7, which asks "did we port the
published pipeline correctly", not "is this method good".

Purpose:
1. give any future localisation claim of ours a first-hand baseline floor on our data;
2. fill the literature gap that none of these methods has been compared frame-level in the hate domain;
3. measure 2025-26 mechanisms against the older LAVAD skeleton in our domain.

## 1. Frame grid

- **Canonical evaluation grid: 4 fps.** Frame index `i` of a video denotes the instant
  `t_i = i / 4` seconds, matching the decoding contract of
  `scripts/r16_detbase/extract_dense_clip.py` (ffmpeg `fps=4` filter; output frame `i` is the video
  content at `t = i/4`).
- Gold frame count for a video of ffprobe duration `D` is `T_gt = floor(D * 4)` (frames at
  `t = 0, 0.25, …` strictly below `D`). Feature arrays may differ from `T_gt` by a few frames
  because of container/stream duration disagreement; **evaluation truncates both gold and score to
  `T = min(T_gt, T_feat)`** and the per-video `(T_gt, T_feat)` pair is recorded in the result file.
  A video whose `|T_gt - T_feat| > 8` frames (2 s) is flagged in the run report and excluded from
  no table — it is reported, not dropped.
- A **secondary 1 fps grid** (`t_i = i`, `T = floor(D)`) is built and stored alongside the 4 fps
  grid, for one purpose only: reproducing the published `TEMPORAL_SPAN_LANDSCAPE §1.3` oracle
  numbers, which were measured at 1 fps. All campaign result tables use 4 fps.
- Methods whose native output is coarser than 4 fps (segment scores, 16-frame windows, 1 fps
  captions) are **upsampled by piecewise-constant broadcast** onto the 4 fps grid. This is the
  standard LAVAD/VAD convention and is recorded per method in the result table's `native_rate`
  column. No smoothing, no interpolation is added by us beyond what the method itself specifies.

## 2. Metrics

Computed by one shared evaluator, never re-implemented per method.

- **Frame ROC-AUC** and **frame PR-AUC (= average precision)**, computed on the **pooled frame set
  of the whole evaluated split** (all videos concatenated, one global ranking). This is the
  MultiHateLoc / LELA convention and the convention under which the §3 controls were measured.
  `sklearn.metrics.roc_auc_score` / `average_precision_score`, default tie handling.
- **F1@tIoU ∈ {0.30, 0.50, 0.70}**, proposal-level, HateClipSeg-paper convention (F1 over matched
  predicted intervals vs gold intervals; each gold interval may be matched at most once, greedy by
  descending proposal score). **Reported only by methods that natively emit intervals.** A
  score-curve method does not get a thresholded interval invented for it; the cell reads `n/a`.
- Reported to 4 decimal places. No confidence intervals for deterministic single-run methods; for
  stochastic methods see §6.

Secondary, always reported next to the primary numbers because §3 shows the primaries are
compressible: **positive base rate** of the evaluated frame pool, and **oracle-normalised AP**
`(AP - AP_random) / (AP_broadcast - AP_random)`, which reads 0 at the random floor and 1 at the
zero-temporal-resolution ceiling. This is a descriptive rescaling, not a new metric to optimise.

## 3. Mandatory controls (every dataset, every table)

Two controls are computed once per dataset and printed in every table as fixed rows:

1. **Gold video-level broadcast (zero-temporal-resolution ceiling).** Score = 1 for every frame of
   every video the gold marks as hateful/toxic at video level, 0 for every frame of every other
   video. It has a perfect video-level classifier and no localisation ability whatsoever. Frozen
   expected values, from `TEMPORAL_SPAN_LANDSCAPE §1.3`, **at 1 fps on the full corpora**:

   | Dataset | 1 fps positive base rate | broadcast frame AP |
   |---|---|---|
   | HateMM (all 1,083) | 0.2869 | **0.675** |
   | MHC-EN (793 rows with local media) | 0.2466 | **0.786** |
   | MHC-ZH (814) | 0.2539 | **0.853** |
   | HateClipSeg (395, any-toxic) | 0.4638 | **0.530** |

   §11 requires the GT parser to reproduce these four AP values at 1 fps before any method is run.
   Tolerance **±0.005 absolute**; a miss is investigated and explained before the campaign proceeds.

2. **Uniform random score floor.** Score ~ U(0,1) i.i.d. per frame, 20 draws, mean ± sd reported.
   Seeds 20250819 + k, k = 0..19.

A reported method that does not clear the random floor is reported as such. A reported method that
does not clear the broadcast ceiling is reported as such. Neither outcome kills anything.

## 4. Ground-truth conventions (frozen)

Positive = "this instant is inside an annotated hateful/toxic span". Spans are half-open `[a, b)`.
A frame at `t` is positive iff some gold span has `a <= t < b`.

- **HateMM** — `data/gt/HateMM/hate_spans.json`. Fields `duration`, `spans` (list of `[start, end]`
  in seconds), `label`. 1,083 entries. Video-level positive = `label == 1`. Entries with
  `label == 1` and an empty or zero-length span list keep their frames in the pool with **zero
  positive frames** (they are not dropped); the count of such videos is reported.
- **MHC-EN / MHC-ZH** — `data/gt/mhc_votes/mhc_{English,Chinese}_{train,valid,test}.tsv`, column
  `Duration`. Format is a Python literal string, e.g. `[(0, 10)]` or `[]`, parsed with
  `ast.literal_eval`; endpoints are **integer seconds**. A row with `[]` has no annotated span.
  Video-level positive = `Majority_Voting != 'Normal'` (i.e. Hateful or Offensive), which is the
  label the span annotation accompanies. **Spans are clipped to `[0, D)` against the local video's
  ffprobe duration**; a span extending past `D` is truncated and the truncation is counted and
  reported. The English TSVs contain 1,001 rows for 1,000 unique IDs — the duplicate row is
  deduplicated keeping the first occurrence, and the duplicate ID is named in the run report.
  Rows whose video is not on local disk are excluded from every pool (EN 208, ZH 186 such rows).
- **HateClipSeg** — `data/gt/HateClipSeg/gold_segments.json`, original released segments, no
  re-gridding. Each segment is `[start, end, multi-hot(6)]` with class order
  `0 normal, 1 hateful, 2 insulting, 3 sexual, 4 violence, 5 harm`. **Primary binary label is
  "any toxic" = any of classes 1..5**, matching `TEMPORAL_SPAN_LANDSCAPE §1.2/§1.3` and
  `scripts/r11_seg/build_grid.py`. The `hateful-only` (class 1) variant is built and stored as a
  secondary array but is not the headline. Durations from
  `data/gt/HateClipSeg/video_durations.jsonl`.

The parser writes `data/gt/frame_gt_4fps/{HateMM,MHC,MHC_zh,HateClipSeg}.npz`, one array per video
plus a 1 fps mirror and the metadata needed by §1 and §3. **Reading annotation files to build an
evaluator is not test-set tuning** (red line 1 forbids using test labels to choose a
hyper-parameter, not to score).

## 5. Splits

- Each dataset keeps its **existing frozen project split**:
  `data/gt/{HateMM,MHC,MHC_zh}/{train,val,test}.jsonl` and
  `data/gt/HateClipSeg/p11_split.json`.
- **Headline table = test split.** Every method gets exactly one test-split evaluation (§10, red
  line 4).
- Zero-shot / training-free methods have nothing to fit. Where a method needs a threshold, a
  smoothing width, a prompt choice among a fixed set, or any other free knob, it is chosen on
  **val** and then frozen; the chosen value is written into the method's own run record before the
  test call.
- The §3 broadcast/random controls and the §11 reproduction check are computed on the **full corpus**
  (all splits pooled) because the published numbers they reproduce were measured that way; the
  per-split control values are also stored and are what the headline table uses.

## 6. Randomness and seeds

- Deterministic pipeline (fixed weights, greedy decoding, no sampling) → **one run**, no error bar.
- Any stochastic element (sampling decode, random init, random subsampling, test-time adaptation) →
  **3 seeds: 20250819, 20250820, 20250821**; report mean ± sd across seeds.
- The uniform random floor uses its own 20 seeds (§3).
- Every run records the resolved seed, the git commit of this repo, the model revision hash, GPU
  model, torch version, and the wall-clock time, in the method's `run_meta.json`.

## 7. Transplant-fidelity check (the only pass/fail in the campaign)

Two methods have already been ported to our datasets by a published third party (LELA,
arXiv 2602.09637). Their numbers are the alignment target:

| Method | Dataset | published PR-AUC / ROC-AUC | source |
|---|---|---|---|
| LAVAD | HateMM | 0.5781 / 0.6163 | LELA table |
| LAVAD | MultiHateClip | 0.5865 / 0.6302 | LELA table |
| URF-HVAA | HateMM | 0.6239 / 0.5674 | landscape doc row 2 |
| URF-HVAA | MultiHateClip | 0.6147 / 0.5626 | landscape doc row 2 |

**Our reproduction is called a successful transplant if it lands within ±0.03 absolute of the
target on both metrics.** Outside ±0.03 the number is still reported, flagged
`transplant=OUT_OF_TOLERANCE`, and the discrepancy is investigated and written up. This check
never removes a method from the table.

Caveat recorded up front: LELA does not state which MultiHateClip language(s) its "MultiHateClip"
column pools. We report EN and ZH separately and compare the target against both.

## 8. Text-injection variants (our declared adaptation)

Every entry in the shortlist scores visual frames only. Where a method's interface admits text — a
caption stream, a free-text query, a subtitle channel, an LLM prompt slot — we additionally run an
**`+text` variant** that injects the on-disk ASR and OCR caches for that video/window, and report it
**as a separate row** clearly marked `variant=+text (ours)`. Rules:

- The base row is always the faithful, unmodified port. `+text` never replaces it.
- Text comes only from the frozen caches: `data/ASR/<DS>/*_asrK4_whisper-large-v3.jsonl` and
  `data/OCR/<DS>/ocr_windows_K30.jsonl` (see §13 for the convention decision). No new
  transcription or OCR is run per method.
- The injected string for a method's native window is the concatenation, in time order, of the ASR
  chunks and OCR window texts that **overlap** that window, ASR first then OCR, separated by
  `" | "`. Empty when neither channel has text.
- A method that has no text slot gets no `+text` row. We do not invent one.
- MHC-ZH text is injected in Chinese as cached; no translation step.

## 9. Reproduction roster and waves

Supervision class is a table column, not a filter. Columns: `label-free` (no target-dataset labels
at all), `one-class` (needs a "these are normal" pool), `aux-temporal-pretrain` (checkpoint trained
with temporal annotation on a disjoint corpus).

**Wave 0 — floors** (`label-free`): ZS-CLIP; ZS-ImageBind (image and video variants, audio on);
LLaVA-1.5 direct scoring (from the LAVAD repo); Qwen2.5-VL-7B native temporal grounding
(TempSamp-R1 / lmms-eval prompt convention, emits intervals → F1@tIoU applies).

**Wave 1 — main** (`label-free` unless noted): URF-HVAA (with LAVAD on the same captions as its
ablation floor); LaGoVAD (`aux-temporal-pretrain`); UniTime (`aux-temporal-pretrain`; six hate
categories as six queries, category wording from the HateMM/MHC target maps); AV²A (audio-visual,
per-second).

**Wave 2 — completion**: MULDE and CLAP (`one-class`, existing feature vectors); VADTree (needs a
GEBD checkpoint); EventVAD (needs RAFT; the optical-flow surveillance prior is a known domain risk,
run and recorded anyway); T3AL (class list = six hate categories); SeViLA Localizer
(`aux-temporal-pretrain`). Install-gated, run only if they build:
OmniVTG, BAGLM, ZS-STVG, LAVIDA, OV-AVEL, DASM, FLAM, FineLAP — the reject list is written down
with the failure reason, not silently dropped.

**Excluded up front, with reason**: MoniTor (repo is a documentation stub); PANDA, DART, Memory
Matters (no code, not re-implemented this round); O-VAD (industrial object-state mechanism, no
domain fit); T\* (needs detectable objects); TFVTG (requires a paid GPT-4 endpoint); any paid API
method (cited from the literature, never run). Ruling: **zero paid API spend for the whole
campaign.**

Datasets: **all four** — HateMM, MHC-EN, MHC-ZH, HateClipSeg — for every method that runs.

## 10. Red lines (never relaxed)

1. **Zero test-label tuning.** No test-split label is used to select any knob of any method.
   Building the GT parser and computing the §3 controls is scoring, not tuning.
2. **This protocol is frozen before any number exists.** Committed before the first run.
3. **Blindness.** No method's metric is computed while its adapter is being written or debugged;
   smoke tests check output shape and score range only.
4. **One test call per method per dataset.** A repeat requires a written deviation in §12 stating
   what broke and why the first call is void.

## 11. Phase A acceptance gates

Phase A (assets) is done only when all of the following hold, and each is recorded in
`idea-stage/repro_campaign/PHASE_A_STATUS.md`:

- **G1** The GT parser reproduces the four §3 broadcast AP values at 1 fps within ±0.005.
- **G2** Dense 4 fps CLIP-L/336 features exist for HateMM 1,083, MHC-EN 792, MHC-ZH 814, in the
  HateClipSeg cache format: one `float32` `.npy` per video, shape `(T, 1024)`, `T ≈ 4·D`.
- **G3** Re-extracting 3 HateClipSeg videos with the campaign's extraction command reproduces the
  existing `dense4fps_clipL336` cache; agreement recorded bit-exact or with the measured numeric
  tolerance.
- **G4** wav2vec2-emotion 4 fps features exist for the same video sets, `(T, 1024)`, index-aligned
  to the visual array.
- **G5** The cache backfill of §13 is complete or its shortfall is named.

## 12. Deviations

**D1 — 2026-08-19 — the broadcast oracle's positive video set is span-derived, not label-derived.**
*What §4 said at freeze:* MHC video-level positive = `Majority_Voting != 'Normal'`, HateMM = `label == 1`.
*What broke:* under that rule the §3 G1 gate failed on both MHC datasets — measured 1 fps broadcast
AP 0.7150 (EN, target 0.786) and 0.7255 (ZH, target 0.853), while the positive base rate matched the
published value to four decimals (EN 0.2463 vs 0.2466, ZH 0.2539 vs 0.2539). A matching base rate
with a mismatched AP isolates the fault to the oracle's *denominator* — which videos get score 1 —
and not to the frame labels, the durations or the grid.
*Cause:* the MHC vote TSVs disagree with themselves. 25 EN and 35 ZH videos carry an annotated
`Duration` span while their majority vote is `Normal`; 24 EN and 28 ZH carry a non-Normal majority
vote with no span at all. The published landscape oracle is "score 1 for every second of every video
the gold says is hateful/toxic", and the gold statement that produces frames is the **span
annotation**, not the majority vote.
*New rule, used from now on:* `y_video = 1` iff the annotation lists at least one span, evaluated
**before** clipping to `[0, D)` (so a video whose only span lies past its own duration still counts
as a positive-scoring video with zero positive frames, exactly as HateMM's one span-less `label == 1`
video does). The dataset's own video-level class label is still stored, as `y_video_ann`, and is
descriptive only — it is not used by any control or metric.
*Effect:* all four G1 gates pass. 1 fps broadcast AP HateMM 0.6750 (target 0.675), MHC-EN 0.7835
(0.786), MHC-ZH 0.8543 (0.853), HateClipSeg 0.5298 (0.530); max |diff| 0.0025, tolerance 0.005.
*Scope:* affects the §3 control rows and the `y_video` array only. No method score, no split, no
seed, no red line is touched. Recorded before any method was run.

**D2 — 2026-08-19 — two HateMM source files have no video stream; G2/G4 read 1081/1083.**
*What §11 said at freeze:* G2/G4 require dense features for HateMM 1,083.
*What happened:* `hate_video_147` (150.14 s) and `hate_video_292` (134.95 s) contain a single AAC
audio stream and no video stream at all (`ffmpeg -map 0:v:0` → "Stream map matches no streams").
There is no visual frame to encode; this is a property of the released media, not a pipeline fault.
*Rule:* both ids are named in `NO_VIDEO_STREAM` in `scripts/repro_campaign/verify_phase_a.py`, the
gate reports `complete_or_explained`, and the exact count `complete` is still recorded next to it.
No zero array is fabricated for either video — a method that needs visual features simply has no
row for them.
*Scope:* neither video is in any frozen HateMM split (`train`/`val`/`test`), so no headline table is
affected; only the full-corpus §3 control row loses two videos' frames. Six further HateMM videos
have a video stream but no decodable audio (`hate_video_108`, `hate_video_17`, `non_hate_video_132`,
`non_hate_video_2`, `non_hate_video_218`, `non_hate_video_252`); their `dense4fps_w2vemo` array is
zero-filled, as the HateClipSeg pipeline already does, and none of the six is in a split either.

## 13. Cache backfill conventions

- **ASR**: `whisper-large-v3`, K=4 sub-clip window alignment, produced by
  `src/utils/generate_segment_asr_HF.py`, output
  `data/ASR/<DS>/{train,dev_seen,test_seen}_asrK4_whisper-large-v3.jsonl`. This matches the
  existing MHC and MHC_zh files. Backfill target: MHC-EN `dev_seen` and `test_seen`.
- **OCR**: PaddleOCR, **K=30** midpoint window grid (`t_k = (k+0.5)·D/K`), produced by
  `scripts/ocr_cache/extract_ocr_windows.py`, output `data/OCR/<DS>/ocr_windows_K30.jsonl`.
  Backfill target: MHC and MHC_zh `train` + `dev`.
  **Decision, recorded here rather than silently**: the campaign brief said "OCR, K4". Every OCR
  cache on disk, including the already-built `data/OCR/MHC_test` and `data/OCR/MHC_zh_test`, is
  K=30. Building train/dev at K=4 would make them incomparable with the test files they exist to
  complete. **K=30 is used**, matching the existing test-split caches. ASR stays K=4, matching the
  existing ASR files. Windowed text is mapped onto a method's native window by time overlap (§8),
  so neither K is load-bearing for the campaign metrics.
- **HateMM `test_seen_subclipK30`** CLIP tensor: built with the existing sub-clip pipeline, same
  contract as the existing `train_` and `dev_seen_` K30 tensors.

## 14. Result table schema

`idea-stage/REPRO_CAMPAIGN_RESULTS.md`, one row per (method × dataset × variant), columns:

| column | meaning |
|---|---|
| `method` | method name as published |
| `wave` | 0 / 1 / 2 |
| `dataset` | HateMM / MHC-EN / MHC-ZH / HateClipSeg |
| `split` | test (headline) |
| `supervision` | label-free / one-class / aux-temporal-pretrain |
| `variant` | base / `+text (ours)` |
| `native_rate` | the method's own output rate before broadcast to 4 fps |
| `frame_ROC_AUC` | 4 dp; mean ± sd if stochastic |
| `frame_PR_AUC` | 4 dp; mean ± sd if stochastic |
| `F1@0.3` / `F1@0.5` / `F1@0.7` | interval-emitting methods only, else `n/a` |
| `AP_norm` | oracle-normalised AP (§2) |
| `n_frames` / `base_rate` | evaluated pool size and positive rate |
| `seeds` | 1 or the three seeds |
| `transplant` | `OK` / `OUT_OF_TOLERANCE` / `n/a` (§7) |
| `gt_convention` | pointer to the §4 clause used |
| `run_dir` | `idea-stage/repro_<name>/` |
| `notes` | install/port caveats |

Fixed control rows per dataset: `GOLD_BROADCAST` and `RANDOM_UNIFORM`.
Stratified sub-tables for HateMM / MHC: single-span videos vs multi-span videos, because the
coverage degeneracy differs sharply between them (HateMM single-span 72.8%, MHC-EN 95.8%,
MHC-ZH 98.2%).

## 15. Reporting stance

Report what was run, what number came out, and what it means for a future localisation claim.
No method in this campaign is judged good or bad. The campaign's own headline finding is expected
to be a comparison of every reproduced number against the §3 broadcast ceiling; that comparison is
a statement about the benchmarks, not about the methods.
