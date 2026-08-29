# A0 ± OCR end-to-end — results

Run 2026-08-09. Protocol, arms, seeds, injection mechanism and the decision rule were frozen
**before** any candidate number was computed, in `idea-stage/A0_OCR_E2E_FREEZE.md`. Single
submission: 2 arms × 3 seeds in one background process, no re-run, no tuning after seeing
numbers. Implementation was validated only on synthetic random-feature caches.

- Runner: `scripts/ocr_cache/run_a0_ocr_e2e.sh`
- Log: `logging/runs/a0_ocr_e2e/run.log` (PID file `run.pid`); per-run trainlogs in
  `logging/runs/a0_ocr_e2e/trainlogs/arm{A,B}_seed{0,1,2}.trainlog`
- Collector (frozen rule, written before results existed):
  `scripts/ocr_cache/collect_a0_ocr_e2e.py`
- Raw results: `idea-stage/a0_ocr_e2e.json`
- OCR archive caches: `data/OCR/HateMM/rac_ocrmean30_{train,dev_seen,test_seen}.pt`
  (builder `scripts/ocr_cache/build_ocr_rac_cache.py`)
- Wall clock: 17 min 33 s for all 6 runs (budget was 2 h; `--epochs 30` unchanged).

## Data boundary

`--val_only_eval True` (new flag in `src/run_rac.py`, default False, inert when off). Every
run logged `[val_only_eval] TEST FIREWALL ON: dropped 215 test_seen rows`. The HateMM test
split never entered a dataloader, a FAISS index, a metric, or checkpoint selection. Train 744
/ dev_seen 107. At the time of this run the `rac_ocrmean30_test_seen.pt` file was a placeholder
carrying the dev rows; HateMM test OCR was never encoded.

> Update 2026-08-09 (after this run, does not affect any number above): the user unsealed test
> *inputs*, the HateMM test OCR cache (215 videos) was built with the identical extractor, and
> `rac_ocrmean30_test_seen.pt` now holds REAL 215-row test features. The placeholder used by
> this run is preserved byte-for-byte as `rac_ocrmean30_test_seen_PLACEHOLDER.pt`.

## Arms

| arm | description | trainable params |
|---|---|---|
| A | current A0 baseline, `classifier_hateClipper`, unchanged | 4,986,881 |
| B | A + OCR third stream, `classifier_hateClipperArchive` (`--archive_feats … --archive_mode stream`) | 6,822,913 (**+1.84 M, +36.8 %**) |

Arm B routes the 768-d OCR-30 mean vector through the pipeline's **pre-existing** third-stream
path: own `archive_proj: Linear(768,1024)+Dropout(0.2)`, L2-normalised, concatenated onto the
fused `img ⊙ text` vector, fusion-MLP input dim `1024 → 2048`. No new model code. Optimiser,
RGCL triplet + BCE hybrid loss, FAISS hard-negative mining, kNN readout, warmup-5 selection —
byte-identical between arms.

OCR vector = arm 2 ("OCR-30") of the frozen-space pilot, unchanged. Train rows reused verbatim
from `pilot_ocr_blocks.npz['o30']` (re-encoding check on 8 videos: `max|Δ| = 1.0e-6`); dev rows
encoded with the identical recipe, **13/107 (12.1 %)** with no usable OCR text (all-zero row),
against 150/744 (20.2 %) on train.

## Numbers — val (dev_seen, 107 videos), at the epoch the pipeline's own rule selected

Selection = best epoch ≥ warmup(5) by `Val_Retrieval acc`, tie-broken by roc — the pipeline's
existing rule, untouched.

### val macro-F1 (primary)

| arm | seed 0 | seed 1 | seed 2 | **mean ± std** |
|---|---|---|---|---|
| A baseline | 0.8365 | 0.8445 | 0.8432 | **0.8414 ± 0.0043** |
| B +OCR | 0.8160 | 0.8070 | 0.8274 | **0.8168 ± 0.0102** |
| **B − A (paired)** | **−0.0205** | **−0.0375** | **−0.0158** | **−0.0246 ± 0.0114** |

### val accuracy (secondary)

| arm | seed 0 | seed 1 | seed 2 | **mean ± std** |
|---|---|---|---|---|
| A baseline | 0.8411 | 0.8505 | 0.8505 | **0.8474 ± 0.0054** |
| B +OCR | 0.8224 | 0.8131 | 0.8318 | **0.8224 ± 0.0094** |
| **B − A (paired)** | **−0.0187** | **−0.0374** | **−0.0187** | **−0.0249 ± 0.0108** |

Selected epochs: A = 21 / 24 / 27, B = 26 / 29 / 24 (all runs completed 30 epochs, no crashes).

## Verdict (frozen rule, unedited)

Primary quantity = seed-mean `B − A` on val macro-F1 = **−0.0246**.

Frozen rule: `≥ +0.010` **and** 3/3 positive seeds → GO; `+0.003 … +0.010` or mixed-sign with
positive mean → AMBIGUOUS; `≤ +0.003` → NO-GO.

## → **NO-GO**

Adding OCR as a learned third stream **hurts**, by 2.5 macro-F1 points, on **3/3 seeds**. The
paired delta is negative on every seed and its magnitude is ~5.7× the baseline seed std
(0.0043). This is not a null result — it is a consistent regression.

## Secondary readout: amplification vs the frozen space (non-gating)

Frozen-space linear-head result was `+0.0094` (`idea-stage/OCR_FUSION_PILOT_RESULT.md`).
Learning-space result is `−0.0246`, i.e. **ratio −2.6×**.

Frozen classification: **SHRUNK** (< 0.5× band). More precisely, the sign flipped. The
hypothesis "letting the fusion MLP and the contrastive loss learn how to use OCR amplifies the
OCR gain" is **dead** on this evidence. The +0.0094 seen in frozen space did not merely fail to
grow — it inverted once the same vector had to be learned through.

## Non-gating observation: ranking improved while decisions got worse

At the selected epoch, retrieval ROC went the *other* way: A mean 0.8821 (0.8572 / 0.8997 /
0.8895) vs B mean 0.9008 (0.9030 / 0.8943 / 0.9052), i.e. **+0.019 for B**. So the OCR stream
made the learned-space kNN similarity ranking modestly *better* while the thresholded kNN vote
got *worse*. This is recorded as an observation only: epochs were selected by acc, not roc, so
the two arms are read at different epochs and this comparison is selection-confounded. It is
not a result and it does not modify the NO-GO.

## Caveats and what this does / does not license

1. **Confounded with capacity.** Arm B adds 1.84 M parameters (+36.8 %) to a head trained on
   744 videos, on top of adding OCR. A parameter-matched control (e.g. a same-shaped third
   stream fed a zero or shuffled OCR block) was **not** pre-registered and was not run, so this
   experiment cannot separate "OCR is harmful here" from "this head cannot absorb +37 % params
   on 744 samples". Either way, the operational conclusion for the A0 pipeline is the same:
   this injection does not pay.
2. **Bounds this injection, not the modality.** As with the frozen-space pilot, the verdict
   bounds *mean-pooled whole-video OCR through the archive third-stream path*. Window
   selection/attention over OCR, or OCR-keyed retrieval (which the I5 COMPLEMENTARY finding
   points at), is untested here and unexcluded.
3. **Small val set.** 107 dev videos; one video ≈ 0.9 acc points. The per-seed deltas
   (−0.019, −0.037, −0.019) correspond to roughly 2, 4 and 2 videos. The direction is
   consistent across seeds but the effect size should not be over-read.
4. **Three seeds, no bootstrap CI** (none was pre-registered); the std above is descriptive.
5. Val numbers are **not** comparable to any previously reported test-split A0 figure — this
   experiment never touched test.

## Consequence

- OCR-as-learned-third-stream in the RGCL pipeline is **closed**. Do not promote it to a
  registered candidate.
- The OCR cache stays useful as a cheap frozen-space input (+0.0094 at head level) and as a
  retrieval-analysis artifact, but the "learning space will amplify it" route is now falsified
  and should not be re-attempted in this form.
- The next OCR question worth GPU time, if any, is selection/attention over windows or
  OCR-keyed retrieval — and it needs its own pre-registration, including a parameter-matched
  control, given caveat 1.
