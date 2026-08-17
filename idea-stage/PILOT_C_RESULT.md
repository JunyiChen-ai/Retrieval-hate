# P-C — on-screen-text provenance separability: results

Run 2026-08-09. The decision rule was frozen **before** any candidate number was computed, in
`idea-stage/PILOT_FREEZE_2026-08-09.md` (section P-C, written 2026-08-09 08:29). Single
submission: synthetic smoke → label-permuted null → one real run, in one background driver, with
no edit to the script between stages and no re-run afterwards.

- Script: `idea-stage/pilot_c_ocr_provenance.py`
- Driver: `logging/runs/pilot_c/run.sh`; log `logging/runs/pilot_c/run.log`, PID `run.pid`
- Raw results: `idea-stage/pilot_c.json`
- Null control: `logging/runs/pilot_c/null_permuted.json`; synthetic smoke:
  `logging/runs/pilot_c/smoke_synthetic.json`
- Derived caches: `data/OCR/HateMM/pilot_c_typed_blocks.npz` (overlay/scene blocks + typing stats),
  `data/OCR/HateMM/frame_dims_train.json` (per-video frame W×H)

## Data boundary and guard

HateMM **train only, 744 videos** (298 hateful / 446 not). `dev_seen` and `test` were never
opened. An explicit guard was armed at process start and logged as the first line of every stage:

```
[08:38:34] GUARD ARMED: paths containing ['dev_seen', 'test'] and ids containing ['test', 'dev'] HALT the run
```

Every file open in the pilot goes through `guard_path()` (HALT if the path contains `dev_seen` or
`test`) and every id list through `guard_ids()` (HALT if an id contains `test` or `dev`). The
`ocr_windows_K30.jsonl` cache covers 851 videos (744 train + 107 val); val rows are discarded at
parse time, before any other step. `ocr_windows_K30.jsonl` SHA-256 verified at load against
`data/OCR/SHA256SUMS.json`:
`783de1524e4a81d8e2cb91643643a3eddddb24ef3cf3cfc6bdce83acd935b670` — match.

## The OCR cache schema (as it actually is, inspected before writing the typing rule)

`data/OCR/HateMM/ocr_windows_K30.jsonl`, one JSON object per (video, window):

```json
{"video_id": "hate_video_100", "window_k": 0, "t_mid": 2.153,
 "texts": [{"text": "S", "conf": 0.474, "bbox": [[0,0],[597,0],[600,480],[0,480]]}],
 "engine": "paddleocr",
 "engine_version": "paddleocr-3.7.0+paddle-3.3.1+PP-OCRv6_medium_det+PP-OCRv6_medium_rec+en"}
```

Fields: `video_id`, `window_k` (0..29), `t_mid` (seconds), `texts` = list of detections with
`text`, `conf` (float), `bbox` = 4-point polygon `[[x,y],[x,y],[x,y],[x,y]]` in **absolute source
pixels**, plus `engine` / `engine_version`. The geometry required by the frozen typing rule
(box position, per-detection confidence) **is present**. Frame width/height is **not** stored —
see deviation D1.

## Frozen decision rule (transcribed unedited from `PILOT_FREEZE_2026-08-09.md`, section P-C)

> **Provenance typing rule (frozen, unsupervised, no label access).** Group detections across the
> 30 windows into tracks by normalised box-centre proximity (≤ 0.05 in both axes) and text
> similarity (token Jaccard ≥ 0.6). A track is **overlay-like** if it persists in ≥ 50 % of the
> windows that contain any text **and** its box-centre standard deviation is ≤ 0.05 in both axes.
> All other detections are **scene-like**.
>
> **Arms** (identical folds / head / seeds to the OCR fusion pilot; three seeds):
>
> | arm | input |
> |---|---|
> | 0 | baseline `[l2(img) ‖ l2(txt)]`, 1792-d |
> | 1 | + untyped mean-pooled OCR block (replicates the +0.0094 result), 2560-d |
> | 1c | **parameter-matched control**: the untyped OCR block duplicated into two blocks, 3328-d |
> | 2 | + **typed**: overlay-mean block ‖ scene-mean block, 3328-d |
>
> Arm 1c exists so that arm 2 − arm 1c isolates provenance typing from the added dimensionality —
> the capacity confound that the A0 ±OCR pilot could not separate.
>
> **Endpoints.**
> - **O1** — descriptive: share of OCR text mass classified overlay vs scene; coverage per class.
> - **O2** — AUROC of overlay-text-presence and of scene-text-presence against the video label,
>   separately (descriptive, non-gating).
> - **O3 (gating)** — OOF macro-F1 over the 744 train videos, seed-paired deltas.
>
> **Frozen decision rule** (primary quantity = seed-mean `arm2 − arm1c`):
> - **GO** — `≥ +0.010` and positive on 3/3 seeds.
> - **AMBIGUOUS** — `+0.003 … +0.010`, or mixed sign with a positive mean.
> - **NO-GO** — `≤ +0.003`.
>
> **Reading.** NO-GO means on-screen text's weakness in this pipeline is not a typing problem, and
> slot #7 closes. GO means provenance is a real axis and is worth a mechanism-level design.

## O1 — provenance split: coverage and text mass

Typing operates on the same detections the fusion pilot used: the frozen `conf ≥ 0.5,
len(text.strip()) ≥ 2` filter, all 30 windows.

| quantity | overlay | scene | total |
|---|---|---|---|
| videos with ≥1 detection of this class | **401** (53.9 %) | **448** (60.2 %) | 594 with any text |
| detections | 42 105 | 47 812 | 89 917 |
| share of detections | **46.8 %** | 53.2 % | |
| characters | 616 148 | 891 134 | 1 507 282 |
| share of characters | **40.9 %** | 59.1 % | |
| tracks | 1 751 | 26 214 | 27 965 |

Coverage of the split across the 744 train videos:

| class membership | videos | share |
|---|---|---|
| both overlay and scene | **255** | 34.3 % |
| overlay only | **146** | 19.6 % |
| scene only | **193** | 25.9 % |
| neither (no OCR text at all in 30 windows) | **150** | 20.2 % |

**Neither class is nearly empty.** 401 videos carry an overlay block and 448 carry a scene block;
343 videos get an all-zero overlay block and 296 an all-zero scene block. So the O3 endpoint is
*not* underpowered by a degenerate split — but it is diluted: for 339 videos (146 + 193) exactly
one of arm 2's two OCR blocks is a zero vector, and for 150 videos both are.

Structure of the split: overlay is carried by few, long tracks (1 751 tracks holding 42 105
detections, ≈24 detections/track) while scene is many short ones (26 214 tracks holding 47 812
detections, ≈1.8 detections/track). That is the shape the rule was meant to produce.

## O2 — presence indicators vs the label (descriptive, non-gating)

AUROC of a binary presence indicator against the HateMM binary label, 744 train videos:

| indicator | positive rate | AUROC |
|---|---|---|
| overlay-text present | 0.539 | **0.4927** |
| scene-text present | 0.602 | **0.4568** |
| any OCR text present | 0.798 | 0.4834 |

All three sit at or below 0.5. **The mere presence of on-screen text — of either provenance —
carries essentially no signal about the label**, and scene-text presence is mildly
*anti*-correlated with hatefulness. Whatever OCR contributes in O3 comes from text *content*, not
from the fact that text is there.

## O3 — OOF macro-F1 over the 744 HateMM-train videos (gating)

Same frozen 5 folds, same `nn.Linear(d,1)` head, same AdamW (lr 1e-3, wd 1e-2), batch 64, same
inner-4-fold lockstep epoch/threshold selection, same three seeds as
`idea-stage/OCR_FUSION_PILOT_FREEZE.md`.

| arm | dim | seed 20260810 | seed 20260811 | seed 20260812 | **mean ± std** |
|---|---|---|---|---|---|
| 0 baseline | 1792 | 0.8077 | 0.8143 | 0.8092 | **0.8104 ± 0.0035** |
| 1 untyped OCR-30 | 2560 | 0.8155 | 0.8205 | 0.8235 | **0.8198 ± 0.0041** |
| 1c untyped ×2 (control) | 3328 | 0.8125 | 0.8168 | 0.8110 | **0.8134 ± 0.0030** |
| 2 typed overlay ‖ scene | 3328 | 0.8138 | 0.8192 | 0.8205 | **0.8178 ± 0.0036** |

Seed-paired contrasts:

| contrast | per seed | mean ± std | sign |
|---|---|---|---|
| **arm2 − arm1c (gating)** | +0.0013, +0.0024, +0.0095 | **+0.0044 ± 0.0045** | 3/3 positive |
| arm2 − arm1 | −0.0017, −0.0013, −0.0030 | **−0.0020 ± 0.0009** | 0/3 positive |
| arm1 − arm0 | +0.0077, +0.0061, +0.0144 | **+0.0094 ± 0.0044** | 3/3 positive |
| arm1c − arm1 | −0.0030, −0.0037, −0.0126 | −0.0064 ± 0.0053 | 0/3 positive |
| arm1c − arm0 | +0.0047, +0.0024, +0.0018 | +0.0030 ± 0.0015 | 3/3 positive |
| arm2 − arm0 | +0.0060, +0.0048, +0.0113 | +0.0074 ± 0.0035 | 3/3 positive |

### Arm-1 reproduction check — exact

`idea-stage/OCR_FUSION_PILOT_RESULT.md` reports arm 0 = 0.8077 / 0.8143 / 0.8092 (mean 0.8104) and
OCR-30 = 0.8155 / 0.8205 / 0.8235 (mean 0.8198), delta **+0.0094** (+0.0077, +0.0061, +0.0144).
This pilot's arm 0 and arm 1 reproduce **every one of those six numbers bit-for-bit**, and the
paired deltas match to the last reported digit. The harness is the comparator's harness — folds,
head, optimiser, selection rule, seed streams and the untyped OCR block (read from the prior run's
`pilot_ocr_blocks.npz`, SHA- and id-checked) are the same objects. No discrepancy; proceeded.

## Verdict

Primary quantity, per the frozen rule: seed-mean `arm2 − arm1c` = **+0.0044**, positive on 3/3
seeds. The rule: GO if `≥ +0.010` **and** 3/3 positive; AMBIGUOUS if `+0.003 … +0.010`, or mixed
sign with a positive mean; NO-GO if `≤ +0.003`.

## → **AMBIGUOUS**

+0.0044 lands inside the AMBIGUOUS band. It is not a GO: it is less than half the +0.010 bar.

## Null control

Label-permuted run (permutation seed 12345, applied after features and after O1/O2 were computed,
so the OCR blocks are untouched real features paired with scrambled labels). The expectation was
declared in the script **before** the null was executed: all-arm OOF macro-F1 within [0.40, 0.60]
and `|arm2 − arm1c| < 0.010`.

| arm | seed 20260810 | seed 20260811 | seed 20260812 | mean |
|---|---|---|---|---|
| 0 | 0.5085 | 0.5461 | 0.5153 | 0.5233 |
| 1 | 0.5226 | 0.5129 | 0.5250 | 0.5202 |
| 1c | 0.4960 | 0.5136 | 0.4876 | 0.4990 |
| 2 | 0.5030 | 0.4788 | 0.5037 | 0.4952 |

Null contrasts: `arm2 − arm1c` = **−0.0039**, `arm2 − arm1` = −0.0250, `arm1 − arm0` = −0.0031,
`arm2 − arm0` = −0.0281. Both declared conditions hold (all arms in [0.40, 0.60];
|−0.0039| < 0.010), so the pilot is not void.

**But read the null honestly**: under scrambled labels the gating contrast still moves by 0.0039 in
magnitude — the same order as the +0.0044 it produced on real labels. See caveat 1.

## Caveats

Written against this result, not for it.

1. **The effect is barely larger than the harness's own noise.** The gating contrast is +0.0044
   with a seed std of 0.0045 (per-seed +0.0013, +0.0024, +0.0095 — the mean is carried by one
   seed), and the *same contrast under permuted labels* is −0.0039. A quantity whose null-run
   magnitude is 90 % of its real-run magnitude is not a demonstrated effect; three seeds cannot
   separate them, and no bootstrap CI was pre-registered. The frozen rule returns AMBIGUOUS, and
   that verdict should be read as "consistent with zero", not as "small positive".
2. **The typed arm is worse than the untyped arm at equal information.** `arm2 − arm1` = −0.0020,
   negative on 3/3 seeds. Splitting the OCR text into two provenance blocks *loses* accuracy
   relative to pooling it into one 768-d mean. The gating contrast is positive only because
   arm 1c — duplicating a block — is itself worse than arm 1 (−0.0064, 0/3 positive): under weight
   decay, an exactly duplicated block splits its weight across two copies and is effectively
   regularised differently, so 1c is not a neutral capacity control, it is a *handicapped* one.
   Part of the +0.0044 is arm 1c's handicap rather than arm 2's typing. This is a real weakness of
   the pre-registered control design, and it inflates the primary quantity in the candidate's
   favour. The unconfounded reading is the negative `arm2 − arm1`.
3. **Presence carries nothing (O2).** Overlay-presence AUROC 0.4927 and scene-presence 0.4568 are
   at/below chance, so the provenance split cannot be helping via a coarse "this video has captions"
   cue; if typing helps at all it must be by separating *content* distributions, and the O3 numbers
   give no evidence that it does.
4. **The typing rule degenerates on low-text videos.** A track seen in one window trivially
   satisfies "persists in ≥ 50 % of the windows that contain any text" when the video has only one
   or two text windows, and its centre std is 0 by construction, so it is labelled overlay. All 38
   videos with exactly one text window and all 62 with ≤2 are labelled overlay-bearing. The text
   mass affected is small (56 of the 401 overlay-bearing videos have only singleton overlay tracks,
   holding 101 of 42 105 overlay detections, 0.24 %), so this does not drive the endpoint — but it
   does mean "has overlay text" as a *video-level* flag is partly an artefact of low text volume.
5. **Blocks are zero for a large share of videos.** 150/744 (20.2 %) have no OCR text at all; 343
   have an all-zero overlay block and 296 an all-zero scene block. Arm 2's two blocks are therefore
   simultaneously informative for only 255 videos (34.3 %). The endpoint averages the typing effect
   over a population where two thirds of items exercise at most one of the two typed blocks. Even a
   real typing effect would be attenuated here; equally, this pilot has not measured typing on the
   subpopulation where it could matter.
6. **The typing rule is unvalidated against ground truth.** No human ever checked whether the
   tracks labelled "overlay" are burned-in captions/watermarks and the "scene" ones are text in the
   world. The rule is a geometric heuristic frozen from a description; a 0.4681 overlay share of
   detections is plausible but unverified. A wrong split would look exactly like a null result.
7. **Bounds one fusion, not the concept.** Two mean-pooled 768-d blocks into a linear head is the
   cheapest possible use of provenance. Nothing here speaks to provenance-conditioned routing,
   attention over typed windows, or typed retrieval keys.
8. **Absolute level.** arm 0 = 0.8104 sits below the ~0.820–0.823 A0 figure for the full pipeline,
   because this ladder fixes lr/wd with no grid (inherited from the comparator freeze). All arms
   pay the same penalty; only the paired deltas are meaningful.

## Deviations from the brief, all decided before the run

- **D1 — frame normalisation.** Frame width/height is not stored in the OCR cache. Rather than fall
  back to per-video max bbox extent (which underestimates the frame whenever no detection touches
  an edge, and would inflate normalised coordinates), the true dimensions were read from the source
  video **headers** with `ffprobe -select_streams v:0 -show_entries stream=width,height`. This
  succeeded for **744/744** videos (`frame dims: {"ffprobe": 744}` in the log), so the max-extent
  fallback — which is implemented and would have been logged per video — was never used. This reads
  container metadata only, no pixels, from local files under the raw-video-stays-local policy.
  Cached at `data/OCR/HateMM/frame_dims_train.json` as `[W, H, source]`.
- **D2 — seed-scope tags.** The comparator derives its RNG streams from `(seed, arm_index, fold,
  inner)`. To make the reproduction check *exact* rather than merely close, arm 0 and arm 1 reuse
  the comparator's tags (`0` and `2` — the prior pilot's OCR-30 arm was index 2); the two new arms
  get distinct tags `1c` and `2t`. The gating contrast is between two new tags, so this cannot
  favour either side of it; it only makes the reproduction bit-exact.
- **D3 — typing-rule details the freeze left open**, all fixed in code before any real number was
  produced: tokens are `\w+` on the lower-cased string; token Jaccard of two empty sets is 1.0 and
  of one empty set 0.0; box centre = mean of the 4 polygon vertices; tracks are built greedily over
  windows in ascending `k` and detections in file order, a detection matching against a track's
  **last** member and against at most one track per window, ties broken by smallest Chebyshev
  centre distance then lowest track index; centre std uses `ddof=0`; the persistence denominator is
  the number of windows containing ≥1 *kept* detection.
- **D4 — null-control bar.** The freeze registers an explicit null bar for P-A only. The P-C bar
  (all arms in [0.40, 0.60]; `|arm2 − arm1c| < 0.010`) was written into the script before the
  permuted run was executed, and is reported above whether or not it flatters the result.
- **D5 — encoding device.** The comparator encoded window texts on GPU where available; here the
  4 069 newly-required typed window texts were encoded on **CPU** (the run is CPU-only). 3 551 of
  the 7 620 unique typed window texts were reused verbatim from
  `data/OCR/HateMM/pilot_ocr_window_vecs.npz`. The untyped arm-1 block was not re-encoded at all —
  it was loaded from the comparator's `pilot_ocr_blocks.npz`, which is why arm 1 reproduces exactly.

## What this licenses

- **Not** a GO. Provenance typing does not clear the +0.010 bar, is negative against the
  information-matched arm 1 on 3/3 seeds, and its positive gating contrast rests on a control arm
  that is itself handicapped.
- The slot is not formally closed by the frozen rule (AMBIGUOUS, not NO-GO), but the evidence
  inside this pilot points the same way a NO-GO would: at the linear-fusion level, on-screen text's
  weakness is **not** a typing problem.
- If the slot is pursued anyway, the cheapest informative next step is *not* another fusion arm. It
  is (a) human verification of the overlay/scene split on a sample, and (b) restricting the
  endpoint to the 255 videos that actually have both classes, where the mechanism could act.
  Both would need their own pre-registration; nothing here is a verdict on them.
