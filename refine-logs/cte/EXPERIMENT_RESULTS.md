# CTE-RGCL Initial Experiment Results

**Date:** 2026-07-11  
**Plan:** `refine-logs/cte/EXPERIMENT_PLAN.md`  
**Authorized/executed:** M0 + C0 only  
**Final route decision:** **C0 STOP; C1 and C2--C4 locked**

## Scope and supervision

C0 tested the registered vectorized full-bank tangent kernel, numerical parity, supported fixed-anchor paths and resource feasibility. It was a zero-teacher implementation/cost screen, not an MLLM experiment and not a theoretical upper bound on other representation-learning methods.

- Only gold supervision: parent-video binary label.
- Segment/timestamp/span/localization gold: nonexistent and unused.
- K4 labels: mechanically inherited parent-video labels, never segment annotations.
- MLLM calls: 0; OCR calls: 0; teacher cache reads/writes: 0/0.
- Validation/test endpoints and teacher/view artifacts: 0.

## Implementation integrity

The first independent review found four HIGH issues: incomplete C0 case coverage, minibatch-only control matching, pre-C1 rather than post-C1 anchor freeze, and incomplete independent partition/provenance verification. All were repaired. A same-reviewer re-review confirmed `CRITICAL: 0`, `HIGH: 0` in `refine-logs/cte/CTE_C0_C1_CODE_REVIEW.md` before execution.

SLURM sanity job 12715 passed Python/sbatch syntax, frozen train caches/folds, K4 parent-label inheritance, checkpoint fixtures and the zero-call contract. Job 12716 published immutable `CONFIG_FREEZE.json` with implementation SHA `09efb69080cc18fc51ba67262631dcf7a8b9d8199ecf4681e926b3ffe5edc198`.

## SLURM runs

| Job | Run | State | Elapsed | Result |
|---:|---|---|---:|---|
| 12715 | CTE CPU sanity | COMPLETED 0:0 | 00:00:03 | SANITY_GO |
| 12716 | `CTE-CONFIG-FREEZE-v1` | COMPLETED 0:0 | 00:00:03 | M0 GO |
| 12717 | `CTE-C0-MICRO-MHC-S0-v1` | COMPLETED 0:0 | 00:05:36 | producer bundle complete |
| 12718 | `CTE-C0-MICRO-MHC_zh-S0-v1` | COMPLETED 0:0 | 00:06:08 | producer bundle complete |
| 12719 | `CTE-C0-DECISION-v1` | COMPLETED 0:0 | 00:00:03 | independent C0 STOP |
| 12720--12721 | documentation/state + frozen-lineage verifier | COMPLETED 0:0 | 00:00:01 each | payload/state/no-C1/all-frozen-input verification GO |

The GPU jobs initially entered `JobHeldUser`, were automatically released by the scheduler, and were never manually released. All jobs used `HateVideo`, four CPUs and the frozen memory/GPU resources; no script set `--time`.

## C0 results

The decision independently required exactly five distinct folds and, per fold, exactly 32 unique cases (`2 tau x 2 modalities x 4 radii x 2 sMin`), then recomputed maxima from case rows rather than trusting producer summaries.

| Gate / measurement | MHC | MHC-ZH | Frozen gate |
|---|---:|---:|---:|
| max absolute margin error | 2.6889e-7 | 1.5718e-7 | <=1e-5 |
| max absolute tangent T error | **8.0526e-5** | **1.0099e-4** | **<=2e-5** |
| max absolute interval-cost error | **2.2231e-5** | **2.1911e-5** | **<=2e-5** |
| max relative directional-gradient error | 7.1451e-7 | 3.7093e-7 | <=1e-3 |
| minimum pre-normalization norm | 1.9906e-2 | 3.0427e-2 | >=1e-4 |
| batch-32 peak allocated GPU memory | 0.0958 GiB | 0.0963 GiB | <=24 GiB |
| kernel median / p95 | 18.74 / 19.53 ms | 19.36 / 19.53 ms | record |
| all-fold adjacent pair | `(0.20,0.30)` | `(0.20,0.30)` | at least one/fold |
| minimum fold joint video support | 0.9681 | 0.9438 | >=0.80 initial |

Completeness, finite values, stable shifted LSE, margin parity, gradient parity, norm, support, no-OOM, memory and timing-count gates passed. Both datasets failed the tangent and cost parity thresholds. The joint decision is therefore STOP, with `C1_unlocked=false`.

## Interpretation and disposition

This is a numerical implementation/cost failure of the exact frozen FP32 vectorized tangent family. The fixed margin errors are small, but division by the smallest registered `a * max(MAD,sMin)` amplifies them enough for T/cost parity to miss the preregistered tolerance. That observation does not authorize a precision change or threshold relaxation.

Per the plan:

- do not tune the numerical thresholds, anchors, radii, tau/lambda/sMin grid or kernel precision;
- do not submit C1 inner/outer runs;
- do not call an MLLM, OCR path or teacher cache;
- do not interpret C0 STOP as evidence that all shared-representation or MLLM methods are impossible.

No C1 accuracy/macro-F1 result exists, and the global two-dataset, three-seed `+0.030 accuracy/+0.030 macro-F1` target remains unmet.

## Canonical evidence

- `artifacts/cte/v1/CONFIG_FREEZE.json`
- `artifacts/cte/v1/c0/{MHC,MHC_zh}/{numerics,support,microbenchmark,manifest}.json`
- `artifacts/cte/v1/C0_DECISION.json` (payload SHA `5f4e520ceab735fe98e68d195a1dcf9aef6748f7e2cf28e6fae77d2530823975`)
- `slurm/logs/cte_c0_{cpu,gpu}_{12715,12716,12717,12718,12719}.out`

Ready for `/auto-review-loop`: **NO**. The registered route terminated before C1/MLLM evidence.
