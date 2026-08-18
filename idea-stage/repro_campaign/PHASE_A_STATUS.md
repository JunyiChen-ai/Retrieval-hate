# REPRO campaign — Phase A status (assets), 2026-08-19

Protocol: `idea-stage/REPRO_CAMPAIGN_FREEZE.md`, frozen and committed as **74b9d87** before any
number was computed. Deviations D1 and D2 are recorded in §12 of that file.
Machine: single RTX 5090 (32 GB), conda `HateVideo`, torch 2.7.1+cu128. Zero paid API spend.

## 1. Acceptance gates (§11) — all pass

| gate | what it checks | result |
|---|---|---|
| **G1** | frame GT reproduces the published 1 fps broadcast oracle within ±0.005 | **PASS** on all four datasets |
| **G2** | dense 4 fps CLIP-L/336 for HateMM / MHC-EN / MHC-ZH in the HateClipSeg format | **PASS** (1081/1083 + 792 + 814; two exclusions named, D2) |
| **G3** | new extractor reproduces the existing HateClipSeg cache | **PASS, bit-identical**, both channels |
| **G4** | wav2vec2-emotion 4 fps, index-aligned to the visual array | **PASS**, same counts as G2 |
| **G5** | cache backfill complete | **PASS**, all five targets written |

Machine-readable: `idea-stage/repro_campaign/phase_a_gates.json`,
`idea-stage/repro_campaign/gt_controls.json`,
`idea-stage/repro_campaign/g3_pipeline_consistency.json`.

## 2. G1 — broadcast oracle cross-check against the landscape document

`research-wiki/TEMPORAL_SPAN_LANDSCAPE_2026-08-18.md §1.3` measured a zero-temporal-resolution
oracle at 1 fps on the full corpora. Rebuilt from scratch by
`scripts/repro_campaign/build_frame_gt.py`:

| dataset | base rate (target) | broadcast AP (target) | \|diff\| |
|---|---|---|---|
| HateMM, 1083 videos | 0.2870 (0.2869) | **0.6750** (0.675) | 0.0000 |
| MHC-EN, 792 videos | 0.2463 (0.2466) | **0.7835** (0.786) | 0.0025 |
| MHC-ZH, 814 videos | 0.2539 (0.2539) | **0.8543** (0.853) | 0.0013 |
| HateClipSeg, 395 videos, any-toxic | 0.4637 (0.4638) | **0.5298** (0.530) | 0.0002 |

Tolerance 0.005. The first attempt failed on both MHC datasets (0.7150 / 0.7255) with the base rate
already matching to four decimals, which isolated the fault to the oracle's positive **video set**
rather than to the frame labels. Cause and fix are deviation **D1**: the MHC vote TSVs disagree with
themselves — 25 EN and 35 ZH videos carry an annotated span under a `Normal` majority vote, and
24 EN / 28 ZH carry a non-Normal vote with no span — and the published oracle keys on the span
annotation, not the vote. `y_video` is now span-derived; the dataset's own class label is kept
alongside as `y_video_ann` and is descriptive only.

## 3. Canonical 4 fps controls (what every method will be reported against)

Full corpus / test split, 4 fps, pooled frames.

| dataset | split | frames | base rate | broadcast AP | broadcast ROC | random AP (20 seeds) |
|---|---|---|---|---|---|---|
| HateMM | full | 625,249 | 0.2863 | 0.6735 | 0.9028 | 0.2864 ± 0.0006 |
| HateMM | test | 116,975 | 0.2421 | 0.5829 | 0.8857 | 0.2423 ± 0.0013 |
| MHC-EN | full | 110,735 | 0.2441 | 0.7767 | 0.9536 | 0.2446 ± 0.0014 |
| MHC-EN | test | 22,337 | 0.2734 | 0.7664 | 0.9427 | 0.2737 ± 0.0026 |
| MHC-ZH | full | 102,153 | 0.2538 | 0.8537 | 0.9709 | 0.2543 ± 0.0014 |
| MHC-ZH | test | 18,199 | 0.2648 | **0.9191** | 0.9842 | 0.2646 ± 0.0038 |
| HateClipSeg | full | 375,330 | 0.4636 | 0.5297 | 0.6164 | 0.4635 ± 0.0010 |
| HateClipSeg | test | 114,097 | 0.4712 | 0.5437 | 0.6260 | 0.4721 ± 0.0016 |

Random ROC-AUC is 0.499–0.501 everywhere, as it must be. Single-span fraction among span-carrying
videos: HateMM 0.728, MHC-EN 0.951, MHC-ZH 0.985, HateClipSeg 0.220 — the same ordering the
landscape document reports, and the reason the MHC-ZH test broadcast ceiling sits at 0.9191.

## 4. Products

### 4.1 Frame ground truth — `data/gt/frame_gt_4fps/`

| file | contents |
|---|---|
| `{HateMM,MHC,MHC_zh,HateClipSeg}.npz` | per video: `y4` (4 fps int8 labels), `y1` (1 fps mirror), `spans` clipped to `[0,D)`, `duration`, `split`, `y_video` (span-derived), `y_video_ann` (dataset label), `n_spans`; HateClipSeg also `y4_hateonly` |
| `durations_<DS>.json` | ffprobe duration cache, 3,084 videos |

Total 380 KB. Span truncation against the local video duration: HateMM 4, MHC-EN 16, MHC-ZH 191,
HateClipSeg 84 spans clipped.

### 4.2 Dense 4 fps features — `data/CLIP_Embedding/<DS>/dense4fps_{clipL336,w2vemo}/`

float32 `.npy` per video, `(T, 1024)`, `T = floor(4·D)`.

| dataset | clipL336 | w2vemo | size each | wall time |
|---|---|---|---|---|
| HateMM | 1081 / 1083 | 1081 / 1083 | 2.4 GiB | 3,095 s |
| MHC-EN (`video_mp4` transcodes) | 792 / 792 | 792 / 792 | 437 MiB | 892 s |
| MHC-ZH | 814 / 814 | 814 / 814 | 403 MiB | 906 s |
| HateClipSeg (pre-existing) | 395 / 395 | 395 / 395 | 1.43 GiB | — |

New this phase: 6.5 GiB, 4,893 s of GPU wall time in one serial job. Disk went 403 GB → 526 GB used
of 1.8 TB (the extra ~117 GB is this phase's features plus the other agent's model downloads);
1.2 TB free.

Frame-count agreement with the GT grid, on a 25-video random sample per dataset: `T − T_gt` in
`[-1, 1]` everywhere, zero videos off by more than the 2 s tolerance.

### 4.3 Pipeline-drift check (G3)

Three HateClipSeg videos (`bit_0c3iRc8b0CPF`, `bit_0EHvMSiEHVoc`, `bit_0SYLs1h6WtM2`) were
re-extracted through the new script into a side directory and compared to the cache built by
`scripts/r16_detbase/extract_dense_{clip,at}.py`:

| channel | shape agreement | max abs diff | verdict |
|---|---|---|---|
| `dense4fps_clipL336` | 3/3 | **0.0** | bit-identical |
| `dense4fps_w2vemo` | 3/3 | **0.0** | bit-identical |

The new extractor is the same pipeline, not a re-implementation of it. The side directory was
deleted after the comparison.

### 4.4 Cache backfill

| target | status |
|---|---|
| MHC-EN `dev_seen` ASR | 80 records, whisper-large-v3, K=4, audio_ok 80/80, no empty-transcript video |
| MHC-EN `test_seen` ASR | 161 records, same settings, audio_ok 161/161 |
| MHC-EN train+dev OCR | 629 videos, 18,870 windows, 0 failures, 12 MB, 99.8% of videos carry some text |
| MHC-ZH train+dev OCR | 657 videos, 19,710 windows, 0 failures, 17 MB, 99.7% carry some text |
| HateMM `test_seen_subclipK30` | 215 videos, 6,450 sub-clips, 1024-d, 0 zero-vector guards, 26.5 MB |

whisper-large-v3 weights downloaded (`model.safetensors`, 3.09 GB; the redundant fp32 shard set was
cancelled once the fp16 weights were verified loadable).

**Convention decision, recorded in freeze §13 rather than made silently.** The campaign brief asked
for OCR at K=4. Every OCR cache on disk — including the already-built `data/OCR/MHC_test` and
`data/OCR/MHC_zh_test` — is K=30 on the midpoint grid `t_k = (k+0.5)·D/K`. Building train/dev at
K=4 would have made them incomparable with the test halves they exist to complete, so **K=30 was
used for OCR**. ASR stays at **K=4**, which is what the existing MHC and MHC_zh ASR files use.
The `+text` injection rule (freeze §8) maps window text onto a method's native window by time
overlap, so neither K is load-bearing for any campaign metric.

## 5. Known gaps and data facts, stated rather than hidden

- **2 HateMM videos have no video stream.** `hate_video_147` and `hate_video_292` are audio-only
  containers; there is no frame to encode. Neither is in any frozen split, so no headline table is
  affected. Deviation D2. No zero array was fabricated for them.
- **6 HateMM videos have a video stream but no decodable audio** (`hate_video_108`, `hate_video_17`,
  `non_hate_video_132`, `non_hate_video_2`, `non_hate_video_218`, `non_hate_video_252`); their
  `w2vemo` array is zero-filled, matching what the HateClipSeg pipeline already does. None is in a
  split.
- **MHC-EN has 792 unique local videos, not 793.** The English vote TSVs hold 1,001 rows for 1,000
  unique ids; the duplicate row is dropped keeping the first occurrence. The landscape document's
  "793 videos w/ local media" counted rows. The G1 gate passes either way.
- **208 EN and 186 ZH annotated ids have no local media** and are excluded from every pool.
- **MHC-ZH has 191 spans that run past the local video's duration** and are clipped to it. This is
  by far the largest truncation count of the four datasets and is worth remembering when reading
  MHC-ZH numbers.
- **1 HateMM video is labelled hateful with no span** (`label == 1`, empty span list). It stays in
  the pool contributing zero positive frames.
- OCR decoding fell back from decord to ffmpeg on a substantial fraction of MHC videos (decord
  cannot find the video stream index in those containers). The fallback is the committed code path
  and produced 0 failures on either dataset.

## 6. Reproduce

```
python scripts/repro_campaign/build_frame_gt.py            # frame GT + controls + G1
bash   scripts/repro_campaign/run_extract.sh              # dense 4 fps, 3 datasets, serial
bash   scripts/repro_campaign/run_ocr_backfill.sh         # MHC + MHC_zh train/dev OCR
bash   scripts/repro_campaign/run_asr_backfill.sh         # MHC-EN dev/test ASR
python src/utils/generate_subclip_embedding_HF.py --dataset HateMM \
       --num_frames 120 --num_subclips 30 --splits test   # HateMM test K30
python scripts/repro_campaign/verify_phase_a.py           # gates G1-G5
```

All long jobs ran detached under `nohup`/`setsid` with `logging/runs/<task>/run.{log,pid}`
(`repro_extract`, `repro_ocr`, `repro_asr`, `repro_subclipK30`, `repro_whisper_dl`) and print
parseable `PROGRESS` / `[progress]` lines. The extractors are idempotent — an existing output file
is skipped and every write goes through a `.tmp` + `os.replace`, so a kill never leaves a truncated
array.

## 7. What Phase A does not include

Model downloads and per-repo smoke tests (freeze §9 roster: BLIP-2, VideoLLaMA3-7B, Llama-3.1-8B,
ImageBind, LaGoVAD / UniTime / AV²A / SeViLA checkpoints, MULDE / CLAP dependencies) were run by a
separate worker in parallel and are reported separately. No method has been run and no method
metric has been computed — freeze §10 red line 3.
