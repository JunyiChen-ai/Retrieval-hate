# DATASET: HateClipSeg (segment-level hateful video, downloaded subset)

**Date**: 2026-07-04 | **Status**: DOWNLOADED & VERIFIED — evaluable
**Source**: https://github.com/Social-AI-Studio/HateClipSeg (annotations only; videos re-fetched from platforms)
**Local paths**:
- Annotations: `/data/jehc223/HateClipSeg/Dataset/{video,segment}_level_annotation.csv`
- Videos: `/data/jehc223/HateClipSeg/videos/` (`<vid>.{mp4,webm,mkv}`, 395 files, **4.45 GB**)
- Pilot/download artifacts: `/data/jehc223/HateClipSeg/pilot/` (scripts, `sample50.json`, `probe_results.tsv`, `download_status.tsv`)
- SLURM job: `scripts/slurm/hateclipseg_download.sbatch` (job 12232, COMPLETED, ~70 min)

## 1. Annotation structure (matches paper: 435 videos / 11,714 segments — verified exactly)

- **Video ID**: `<platform>_<raw_id>`; platforms: `bit` = BitChute (363), `yt` = YouTube (72).
- **video_level_annotation.csv**: `Video Id, Video-Level Label (multi-label list), Target Victim (list, 21 categories + Other)`.
- **segment_level_annotation.csv**: per video a list of multi-hot vectors (`0:normal 1:hateful 2:insulting 3:sexual 4:violence 5:harm`) + list of `[start,end]` second timestamps. Segments tile the video contiguously from 0 to video end.
- Duration: total 28.8 h; mean 238 s, median 240 s, max 350 s per video.
- **No official train/val/test split is released** — any split must be defined (and reported) by us.

## 2. 50-video survival pilot (stratified by rarest-label × platform, seed 42)

Overall: **47/50 alive = 94%**.

| stratum | alive | | stratum | alive |
|---|---|---|---|---|
| platform bit | 39/41 (95%) | | hateful | 18/18 |
| platform yt | 8/9 (89%) | | insulting | 8/8 |
| harm | 2/3 | | violence | 8/8 |
| sexual | 6/7 (86%) | | normal | 5/6 (83%) |

Gate (≥50%) passed → proceeded to full-corpus download.

## 3. Full download results (yt-dlp 2026.03.17, `-S res:480`, rate-limited 4s BitChute / 2s YT)

**395/435 videos = 90.8% recovered** (pilot estimate 94% was mildly optimistic, within sampling error).

| | videos | recovered | % | segments | recovered | % |
|---|---|---|---|---|---|---|
| **total** | 435 | **395** | **90.8%** | 11,714 | **10,604** | **90.5%** |
| bitchute | 363 | 338 | 93.1% | | | |
| youtube | 72 | 57 | 79.2% | | | |
| normal | 55 | 50 | 91% | 6,491 | 5,864 | 90% |
| hateful | 194 | 180 | 93% | 2,363 | 2,209 | 93% |
| insulting | 280 | 252 | 90% | 2,920 | 2,620 | 90% |
| sexual | 69 | 59 | 86% | 372 | 307 | 83% |
| violence | 192 | 170 | 89% | 1,281 | 1,146 | 89% |
| harm | 18 | 13 | 72% | 39 | 31 | 79% |

(Label rows are multi-label, non-exclusive.)

**Failure taxonomy (40 dead)**: BitChute 404 removed (20), BitChute 403 blocked (5+1), YT "not available" (7), YT account terminated (3), YT ToS removal (2), YT age-gate requiring login cookies (2 — recoverable only with authenticated cookies).

**Integrity**: all 395 files pass ffprobe; **duration vs annotated last-segment end: median Δ = +0.00 s, zero videos off by >2 s** → segment timestamps apply to the downloaded files verbatim, no re-alignment needed.

**Formats**: 364 mp4 / 27 webm / 4 mkv, capped at ≤480p (BitChute sources are natively ≤480p mostly). Disk after download: 259G/290G quota.

## 4. Selection-bias statement (must accompany any result on this subset)

Any evaluation on this corpus is on the **90.8% surviving subset as of 2026-07-03**, not the paper's full 435-video set:
1. **Attrition is non-random**: platform moderation removes the most extreme content first — YouTube attrition (20.8%) ≫ BitChute (6.9%), and the rarest/most severe strata are hit hardest (harm 72%, sexual 86% video coverage vs hateful 93%). Absolute metrics on the subset are therefore **not comparable** to numbers reported on the full corpus in the HateClipSeg paper; only method-vs-method comparisons *on the identical subset* are valid.
2. Coverage is still ≥83% of segments for every label except harm (31/39 segments — too few for per-class harm conclusions; report harm only in aggregate or with wide CIs).
3. Reproducibility: `pilot/download_status.tsv` freezes the exact surviving-ID list; publish it (IDs only) with any results.

## 5. Evaluability verdict

**HateClipSeg is USABLE as the primary temporal-localization evaluation set** (10.6k gold segments across 395 videos dwarfs HateMM's hate_snippet annotations), with HateMM hate_snippet retained as a secondary/confirmatory set. Caveats: declare the 90.8% subset + selection bias; exclude or aggregate the `harm` class; define our own split (none released upstream).
