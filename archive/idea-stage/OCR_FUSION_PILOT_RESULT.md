# OCR three-stream fusion pilot — results

Run 2026-08-09. Protocol, arms, seeds and the decision rule were frozen **before** any candidate
number was computed, in `idea-stage/OCR_FUSION_PILOT_FREEZE.md`. Single submission: all three arms
× three seeds in one process; no re-run, no tuning after seeing numbers.

- Script: `scripts/ocr_cache/ocr_fusion_pilot.py`
- Log: `logging/runs/ocr_fusion_pilot/run.log` (PID file `run.pid`)
- Raw results: `idea-stage/ocr_fusion_pilot.json`
- OCR block cache (derived, reusable): `data/OCR/HateMM/pilot_ocr_blocks.npz`
- Data boundary: HateMM-train only, 744 videos (298 hateful). `dev_seen` and `test` never opened.
  `ocr_windows_K30.jsonl` SHA-256 verified against `data/OCR/SHA256SUMS.json` at load time.

## Arms

| arm | name | input | dim |
|---|---|---|---|
| 0 | baseline | `[l2(CLIP img) ‖ l2(CLIP txt)]` | 1792 |
| 1 | OCR-3 | baseline ‖ mean-pooled CLIP-text embedding of OCR windows `k ∈ {5,15,25}` | 2560 |
| 2 | OCR-30 | baseline ‖ mean-pooled CLIP-text embedding of all 30 OCR windows | 2560 |

Arms 1 and 2 have identical dimensionality, so arm1-vs-arm2 is a pure OCR-window-budget effect.
Same frozen 5 folds, same head (`nn.Linear(d,1)`, BCEWithLogits, AdamW lr=1e-3 wd=1e-2, batch 64),
same inner-4-fold lockstep epoch/threshold selection, same three seeds for every arm.

OCR coverage after the frozen `conf>=0.5, len>=2` filter: 6565 unique window texts; **255/744
videos (34.3%) have no usable OCR text inside the 3-window budget**, **150/744 (20.2%) have none
across all 30 windows** — those get the all-zero OCR block (neutral under a linear head).

## Numbers — OOF macro-F1 over the 744 HateMM-train videos

| arm | seed 20260810 | seed 20260811 | seed 20260812 | **mean ± std** |
|---|---|---|---|---|
| 0 baseline | 0.8077 | 0.8143 | 0.8092 | **0.8104 ± 0.0035** |
| 1 OCR-3 | 0.8131 | 0.8190 | 0.8164 | **0.8161 ± 0.0029** |
| 2 OCR-30 | 0.8155 | 0.8205 | 0.8235 | **0.8198 ± 0.0041** |

Seed-paired deltas (same seed, same folds, same head init scheme):

| contrast | per-seed | mean ± std |
|---|---|---|
| arm1 − arm0 | +0.0054, +0.0046, +0.0072 | **+0.0057 ± 0.0013** |
| arm2 − arm0 | +0.0077, +0.0061, +0.0144 | **+0.0094 ± 0.0044** |
| arm2 − arm1 | +0.0024, +0.0015, +0.0071 | **+0.0037 ± 0.0031** |

Both OCR arms beat the baseline on **every one of the three seeds**; the paired-delta sign is
consistent, and the paired std is smaller than the across-arm spread.

## Verdict (frozen rule, unedited)

Primary quantity = seed-mean `arm2 − arm0` = **+0.0094 macro-F1**.

Frozen rule: `>= +0.015` → GO; `+0.005 … +0.015` → AMBIGUOUS; `< +0.005` → NO-GO.

## → **AMBIGUOUS**

OCR as a third frozen stream is a real but **sub-threshold** gain at the head level. It is not
noise — it is positive on 3/3 seeds and the paired mean is ~2.2× the baseline seed std — but it
does not clear the +0.015 bar that would have made it an unconditional GO.

## Dose relationship (arm1 vs arm2, reported, not gating)

- 3 windows out of 30 (10% of the OCR read budget) recover **+0.0057**, i.e. **61% of the OCR-30
  gain**, at 1/10 the OCR cost.
- Going from 3 to 30 windows adds only **+0.0037** more.
- The curve is strongly concave: most of what mean-pooled OCR text contributes to a *linear* head
  is already present in a 3-window sample. The extra 27 windows mainly (a) rescue the 105 videos
  that had no text in the 3-window budget but do have text somewhere in the 30, and (b) denoise the
  mean. Neither is worth a 10× OCR budget on this evidence.

Read together with the I5 redundancy gate (`data/OCR/HateMM/i5_redundancy_gate.json`,
`ov@10 = 0.048` vs chance 0.017 → COMPLEMENTARY), the picture is consistent: the OCR key retrieves
a genuinely *different* neighbourhood from the transcript key, but a **mean-pooled, whole-video,
linearly-fused** OCR vector converts only a small part of that complementarity into accuracy.

## Caveats

1. **Baseline anchor.** arm 0 lands at 0.8104, slightly below the ~0.820–0.823 A0 figure quoted for
   the existing pipeline. That is expected and was accepted in the freeze: this pilot fixes
   `lr=1e-3, wd=1e-2` for every arm with **no** hyperparameter grid, whereas Gate-0's A0 selects
   over an 8-point lr×wd grid per fold. The comparison here is internally paired (all three arms
   pay the same penalty), so the deltas are the meaningful quantity, not the absolute level.
2. **Aggregation is the weakest link.** A single unweighted mean over windows, fused linearly, is
   the cheapest possible use of OCR. The AMBIGUOUS verdict bounds *this* fusion, not OCR as a
   modality. Attention/selection over windows, or an OCR-conditioned routing/retrieval design, is
   not measured here and is not excluded by this result.
3. **20% of videos have no usable OCR at all**, and 34% have none in the 3-window budget. Whatever
   the ceiling is, it is being averaged over a population where a fifth of the items contribute a
   zero vector.
4. Three seeds only; no bootstrap CI was pre-registered for this pilot, so the paired std above is
   descriptive, not an inference.

## What this licenses

- **Not** a GO for promoting OCR-as-third-stream into a registered candidate on its own strength.
- It **does** justify keeping the OCR cache in the pipeline as a cheap, non-negative input, and it
  identifies the 3-window budget as the efficient operating point if OCR is carried along.
- The interesting follow-up is not "more windows" (flat) but "better use of the windows the
  complementarity is actually in" — i.e. selection/attention over OCR windows, or OCR-keyed
  retrieval, which the I5 COMPLEMENTARY finding points at. Any such follow-up needs its own
  pre-registration; nothing here is a verdict on it.
