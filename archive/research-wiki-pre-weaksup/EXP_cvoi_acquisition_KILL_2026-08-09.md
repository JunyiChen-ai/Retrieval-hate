# CVoI acquisition — KILL record (2026-08-09)

- Direction: **CVoI — costed, set-conditioned evidence acquisition for hateful video detection**
- Pre-registration: [`EXP_cvoi_acquisition_prereg.md`](EXP_cvoi_acquisition_prereg.md) (frozen content unmodified; a single pointer line was added to its top STATUS block)
- Idea node: [`ideas/pay-for-evidence-typed-acquisition.md`](ideas/pay-for-evidence-typed-acquisition.md)
- Status: **CLOSED — killed by user, 2026-08-09**

## 1. Ruling

The user ruled on 2026-08-09 that this direction is killed. The direction is closed. **C6
(measured per-action cost registry) and every subsequent gate will not be executed.** No further
work is authorized under `EXP_cvoi_acquisition_prereg.md`.

Stated reasons:

1. **The direction does not make sense as posed.** The premise was that a classifier should learn
   which evidence is worth acquiring *before* paying the acquisition cost.
2. **The cost constraint that justified the framework no longer exists.** The full OCR cache has
   been built (1246 videos: HateMM 851 + HateClipSeg 395, K=30 windows each). OCR is now a
   pre-computed, already-paid, universally available input. An acquisition policy that spends a
   budget to decide *whether to run OCR on a window* optimizes a cost that the project no longer
   pays. Without a live per-action cost there is no accuracy–cost Pareto frontier to move, and the
   central claim of the pre-registration has no object.

## 2. State at time of kill

| Item | State |
| --- | --- |
| Pre-registration stage | **DESIGN ONLY / PRE-REGISTRATION DRAFT** — never executed, never hash-frozen for execution |
| CVoI candidate metric | **Never computed** (zero results) |
| Test-set contact | **Zero** — HateMM test remained sealed throughout |
| Confirmation set | **Not consumed by this direction** — still unspent |
| C6 cost numbers | **None ever produced** |
| Registered deviations | D1, D2, D3 (all registered before any candidate metric existed) |

Deviations already on record, retained as-is:

- **D1** (`EXP_cvoi_acquisition_deviation_D1_2026-08-08.md`) — permanent retirement of the old K=30
  caches; `old_cache_comparability=FAIL` at tolerance `5e-5` (train replay 744/744,
  `max_abs=1.0419e-4`, 24 failures).
- **D2** (`EXP_cvoi_acquisition_deviation_D2_2026-08-09.md`) — C6 timings are binding only inside a
  GPU-exclusive window.
- **D3** (`EXP_cvoi_acquisition_deviation_D3_2026-08-09.md`) — C6 preflight review v1 was superseded
  by a code edit made two minutes after the review was written (3 of 4 pinned source digests
  drifted); review v2 was issued.

### C6 daemon disposition

`scripts/cvoi_acq/run_c6_exclusive_daemon.sh` was waiting for a GPU-exclusive window and never
measured anything. Its last log lines are `PINNED_SOURCE_OK` then `WAIT gpu_not_exclusive`.

- PID file `logging/runs/cvoi_c6_cost/run.pid` held PID `2656871`; the process was **already dead**
  at kill time (verified via `ps -p`; no lingering `cvoi`/`cost_*` processes). No signal was needed.
- `logging/runs/cvoi_c6_cost/status` now reads `HALT_C6_PINNED_SOURCE_DRIFT` followed by the
  appended line `KILLED_BY_USER_2026-08-09`. No existing file in that run directory was deleted or
  rewritten.

## 3. Assets retained (this kill does not retire them)

The direction dies; its infrastructure does not. Nothing below is deleted.

1. **OCR cache — `data/OCR/`.** 1246 videos (HateMM 851 / 25530 windows; HateClipSeg 395 / 11850
   windows), K=30 windows per video, PaddleOCR, SHA256-manifested in `data/OCR/SHA256SUMS.json`
   with statistics in `data/OCR/ocr_cache_stats.json`. This is a **general-purpose model input**
   under the 2026-08-08 OCR ruling, independent of CVoI. It stays and remains usable by any future
   direction.
2. **Frozen grouping logic.** The group/fold construction used for cross-fitting (leak-safe video
   grouping, fold iteration) is direction-independent and remains valid for reuse.
3. **Code — `scripts/cvoi_acq/`.** Retained, not deleted. The cost-measurement, preflight,
   fixture, audit and fold-iteration modules are reusable; they are simply unowned by any active
   experiment. No new run may be launched from them under this pre-registration.

## 4. Red-line accounting

All four hard red lines held through the life of this direction, and the kill does not disturb any
of them:

1. **Zero test-set contact** — held; the direction never reached a stage that could touch test.
2. **Decision rules frozen before results** — vacuously held; there were no results.
3. **Blindness** — held; no candidate metric was ever computed during design or implementation.
4. **Single-submission formal run** — vacuously held; no formal run was ever submitted.

The confirmation split is **not spent** by this direction and remains available to a future
pre-registration.

## 5. What a successor direction would have to establish

Recorded so the closure is not re-litigated by accident. Any revival would need, at minimum:

- A cost that the project **actually pays at inference time** and cannot pre-compute away — the
  full OCR cache removed the original one.
- Evidence that the acquisition decision beats simply consuming all cached evidence, on accuracy,
  not merely on a budget that is now hypothetical.

No such successor is proposed here.
