# CTE-RGCL C0: full-bank tangent numerical/support screen

**Date:** 2026-07-11  
**Outcome:** **STOP at C0; C1 and all teacher stages locked**  
**Global final-performance target:** not met

## Question

CTE proposed to let a future label-blind, train-only MLLM supervise whole-modality withholding relations through a fixed, train-supported tangent of the exact epoch-refreshed full-bank retrieval margin. Before any teacher spend, C0 asked whether that exact vectorized kernel was numerically faithful, supported on MHC-EN and MHC-ZH train folds, and inexpensive enough to run.

C0 is not an MLLM result or a theoretical upper bound. It used only video-level binary labels and existing train-only checkpoint fixtures. No segment/timestamp/span/localization gold exists or was assumed.

## Integrity and execution

An independent code review closed four HIGH issues, including incomplete radius coverage and fail-closed partition/provenance checks. Final review: `0 CRITICAL / 0 HIGH`.

| Job | Purpose | Result |
|---:|---|---|
| 12715 | SLURM CPU syntax/data/fixture sanity | GO |
| 12716 | canonical config/code/input freeze | GO |
| 12717 | MHC five-fold GPU C0 | completed |
| 12718 | MHC-ZH five-fold GPU C0 | completed |
| 12719 | independent dual-dataset decision | **STOP** |

Both GPU jobs were initially `JobHeldUser` and cleared automatically. They were not manually released. C0/C1 call counters are MLLM 0, OCR 0, teacher reads/writes 0/0; val/test endpoint counts are 0.

## Result

| Dataset | margin error | T error | cost error | gradient error | min joint support | peak GPU | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| MHC | 2.689e-7 | **8.053e-5** | **2.223e-5** | 7.145e-7 | 0.9681 | 0.0958 GiB | STOP |
| MHC-ZH | 1.572e-7 | **1.010e-4** | **2.191e-5** | 3.709e-7 | 0.9438 | 0.0963 GiB | STOP |

The frozen limits were margin `1e-5`, T/cost `2e-5`, relative gradient `1e-3`, min norm `1e-4`, at least one supported adjacent pair per fold and peak memory `24 GiB`. Every fold selected `(a1,a2)=(0.20,0.30)`, and support/resource/finite/margin/gradient gates passed. Tangent and interval-cost parity failed on both datasets.

## Decision and anti-repeat

`artifacts/cte/v1/C0_DECISION.json` independently reconstructed exactly five folds and 32 numerical cases per fold and wrote `C0_DECISION=STOP`, `C1_unlocked=false`. Therefore no nested C1 training, no C2 pilot and no teacher call was launched.

Do not rescue this exact route by relaxing `2e-5`, changing precision after seeing the result, changing anchors/radii or retuning the grid. A future route must be newly preregistered and must continue to use only video-level binary gold. This result says only that the exact frozen FP32 CTE tangent implementation missed its own numerical gate; it does not rule out all shared-representation motion or all MLLM integration.

Canonical detailed record: `refine-logs/cte/EXPERIMENT_RESULTS.md`.
