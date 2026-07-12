# CTE-RGCL Experiment Tracker

**Executed authorization:** C0 only reached execution. Independent `C0=STOP`, so C1 and C2--C4 are locked; teacher calls remain forbidden.  
**Status vocabulary:** `LOCKED / TODO / SUBMITTED / RUNNING / VERIFY / GO / STOP / INVALID`.

| Run ID | Milestone | Purpose | Dataset/fold | Outputs | Resource | Priority | Status | Dependency / binding note |
|---|---|---|---|---|---|---|---|---|
| CTE-CONFIG-FREEZE-v1 | M0 | canonical config/code/data/split freeze | both | `CONFIG_FREEZE.json` | CPU 4/16G | MUST | GO | job 12716; independent review 0 CRITICAL/0 HIGH; zero teacher |
| CTE-C0-MICRO-MHC-S0-v1 | M1 | vectorized full-bank numerics/runtime/support | MHC/F0..4 fixtures | c0 MHC JSON bundle | GPU1 CPU4/32G | MUST | STOP | job 12717 completed; support/resource/margin/gradient pass, T/cost tolerance fail |
| CTE-C0-MICRO-MHC_zh-S0-v1 | M1 | vectorized full-bank numerics/runtime/support | MHC_zh/F0..4 fixtures | c0 MHC_zh JSON bundle | GPU1 CPU4/32G | MUST | STOP | job 12718 completed; support/resource/margin/gradient pass, T/cost tolerance fail |
| CTE-C0-DECISION-v1 | M1 | independent C0 reconstruction/decision | both | `C0_DECISION.json` | CPU 4/16G | MUST | STOP | job 12719 independently recomputed all cases; `C1_unlocked=false` |
| CTE-C1-INNER-MHC-F0-S0-v1 | M2 | nested probe/grid | MHC/F0 | inner bundle | GPU1 CPU4/32G | MUST | LOCKED | C0 GO |
| CTE-C1-INNER-MHC-F1-S0-v1 | M2 | nested probe/grid | MHC/F1 | inner bundle | GPU1 CPU4/32G | MUST | LOCKED | C0 GO |
| CTE-C1-INNER-MHC-F2-S0-v1 | M2 | nested probe/grid | MHC/F2 | inner bundle | GPU1 CPU4/32G | MUST | LOCKED | C0 GO |
| CTE-C1-INNER-MHC-F3-S0-v1 | M2 | nested probe/grid | MHC/F3 | inner bundle | GPU1 CPU4/32G | MUST | LOCKED | C0 GO |
| CTE-C1-INNER-MHC-F4-S0-v1 | M2 | nested probe/grid | MHC/F4 | inner bundle | GPU1 CPU4/32G | MUST | LOCKED | C0 GO |
| CTE-C1-INNER-MHC_zh-F0-S0-v1 | M2 | nested probe/grid | MHC_zh/F0 | inner bundle | GPU1 CPU4/32G | MUST | LOCKED | C0 GO |
| CTE-C1-INNER-MHC_zh-F1-S0-v1 | M2 | nested probe/grid | MHC_zh/F1 | inner bundle | GPU1 CPU4/32G | MUST | LOCKED | C0 GO |
| CTE-C1-INNER-MHC_zh-F2-S0-v1 | M2 | nested probe/grid | MHC_zh/F2 | inner bundle | GPU1 CPU4/32G | MUST | LOCKED | C0 GO |
| CTE-C1-INNER-MHC_zh-F3-S0-v1 | M2 | nested probe/grid | MHC_zh/F3 | inner bundle | GPU1 CPU4/32G | MUST | LOCKED | C0 GO |
| CTE-C1-INNER-MHC_zh-F4-S0-v1 | M2 | nested probe/grid | MHC_zh/F4 | inner bundle | GPU1 CPU4/32G | MUST | LOCKED | C0 GO |
| CTE-C1-SELECT-v1 | M2 | paired cross-dataset minimax tuple/fold | both/F0..4 | selection bundle | CPU 4/16G | MUST | LOCKED | all 10 INNER valid; no outer/dev/test result |
| CTE-C1-OUTER-MHC-F0-S0-v1 | M3 | strict OOF arms | MHC/F0 | outer ledgers | GPU1 CPU4/32G | MUST | LOCKED | SELECT valid |
| CTE-C1-OUTER-MHC-F1-S0-v1 | M3 | strict OOF arms | MHC/F1 | outer ledgers | GPU1 CPU4/32G | MUST | LOCKED | SELECT valid |
| CTE-C1-OUTER-MHC-F2-S0-v1 | M3 | strict OOF arms | MHC/F2 | outer ledgers | GPU1 CPU4/32G | MUST | LOCKED | SELECT valid |
| CTE-C1-OUTER-MHC-F3-S0-v1 | M3 | strict OOF arms | MHC/F3 | outer ledgers | GPU1 CPU4/32G | MUST | LOCKED | SELECT valid |
| CTE-C1-OUTER-MHC-F4-S0-v1 | M3 | strict OOF arms | MHC/F4 | outer ledgers | GPU1 CPU4/32G | MUST | LOCKED | SELECT valid |
| CTE-C1-OUTER-MHC_zh-F0-S0-v1 | M3 | strict OOF arms | MHC_zh/F0 | outer ledgers | GPU1 CPU4/32G | MUST | LOCKED | SELECT valid |
| CTE-C1-OUTER-MHC_zh-F1-S0-v1 | M3 | strict OOF arms | MHC_zh/F1 | outer ledgers | GPU1 CPU4/32G | MUST | LOCKED | SELECT valid |
| CTE-C1-OUTER-MHC_zh-F2-S0-v1 | M3 | strict OOF arms | MHC_zh/F2 | outer ledgers | GPU1 CPU4/32G | MUST | LOCKED | SELECT valid |
| CTE-C1-OUTER-MHC_zh-F3-S0-v1 | M3 | strict OOF arms | MHC_zh/F3 | outer ledgers | GPU1 CPU4/32G | MUST | LOCKED | SELECT valid |
| CTE-C1-OUTER-MHC_zh-F4-S0-v1 | M3 | strict OOF arms | MHC_zh/F4 | outer ledgers | GPU1 CPU4/32G | MUST | LOCKED | SELECT valid |
| CTE-C1-DECISION-v1 | M3 | independent dual-dataset C1 gate | both | `C1_DECISION.json` | CPU 4/16G | MUST | LOCKED | +.05/+.05, correction, churn, controls, support/drift all binding |
| CTE-C1-FULLFREEZE-MHC-S0-v1 | M4 | post-GO full-train checkpoint/anchors | MHC | checkpoint+anchors+manifest | GPU1 CPU4/32G | MUST | LOCKED | only if C1 GO; still zero teacher |
| CTE-C1-FULLFREEZE-MHC_zh-S0-v1 | M4 | post-GO full-train checkpoint/anchors | MHC_zh | checkpoint+anchors+manifest | GPU1 CPU4/32G | MUST | LOCKED | only if C1 GO; still zero teacher |
| CTE-C1-FREEZE-VERIFY-v1 | M4 | independent teacher-before-call lock | both | `C1_FREEZE_VERIFY.json` | CPU 4/16G | MUST | LOCKED | completion permits planning C2, not teacher call itself |
| CTE-C2-PILOT-TBD | M5 | ≤128/dataset teacher ordinal transfer | both | TBD | TBD | MUST | LOCKED | no run ID/calls until separate plan+authorization |
| CTE-C3-SEED0-TBD | M5 | seed0 control separation | both/dev | TBD | TBD | MUST | LOCKED | C2 GO; val no teacher/view |
| CTE-C4-FINAL-TBD | M5 | 2 datasets × seeds 0/1/2 final | both/test | TBD | TBD | MUST | LOCKED | C3 GO; ordinary full-video kNN only |

## Decision ledger

| Gate | Required evidence | Current |
|---|---|---|
| M0 implementation | config/code hashes + independent audit | GO; jobs 12715--12716, final review 0 CRITICAL/0 HIGH |
| C0 | numerics/support/runtime on both datasets | STOP; T and interval-cost FP32/scalar parity exceeded frozen `2e-5` on both datasets |
| C1 | strict OOF +.05 acc/+.05 mF1, corrections, churn, controls | LOCKED by C0 STOP; no C1 job submitted |
| C2 | class-specific two-radius teacher transfer, ≤128/dataset | LOCKED; no calls authorized |
| C3 | seed0 clean beats every control on both metrics/datasets | LOCKED |
| C4 | final +.03/+.03, 2 datasets, 3 seeds, bootstrap/Holm | LOCKED |

## Persistent supervision audit

- `only_gold_supervision = video_level_binary_label`
- `segment_gold_exists = false`
- `segment_gold_used = false`
- C0/C1 `mllm_call_count = 0`
- C0/C1 `val_endpoint_count = test_endpoint_count = 0`
- all validation/test paths: `teacher_or_neutralized_view_count = 0`

## C0 execution ledger (2026-07-11)

| Job | Run | State / exit | Elapsed | Log |
|---:|---|---|---:|---|
| 12715 | CPU sanity | COMPLETED / 0:0 | 00:00:03 | `slurm/logs/cte_c0_cpu_12715.out` |
| 12716 | `CTE-CONFIG-FREEZE-v1` | COMPLETED / 0:0 | 00:00:03 | `slurm/logs/cte_c0_cpu_12716.out` |
| 12717 | `CTE-C0-MICRO-MHC-S0-v1` | COMPLETED / 0:0 | 00:05:36 | `slurm/logs/cte_c0_gpu_12717.out` |
| 12718 | `CTE-C0-MICRO-MHC_zh-S0-v1` | COMPLETED / 0:0 | 00:06:08 | `slurm/logs/cte_c0_gpu_12718.out` |
| 12719 | `CTE-C0-DECISION-v1` | COMPLETED / 0:0 | 00:00:03 | `slurm/logs/cte_c0_cpu_12719.out` |
| 12720--12721 | documentation/state + frozen-lineage verifier | COMPLETED / 0:0 | 00:00:01 each | `slurm/logs/cte_docs_verify_{12720,12721}.out` |

Initial `JobHeldUser` states for GPU jobs cleared automatically; no job was manually released. No C1/C2/MLLM/OCR/teacher job was submitted.

### Independently recomputed C0 cells

| Dataset | max margin err | max T err | max cost err | rel grad err | min norm | peak GiB | selected adjacent pair / joint support | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| MHC | 2.689e-7 | 8.053e-5 | 2.223e-5 | 7.145e-7 | 1.991e-2 | 0.0958 | `(0.20,0.30)`; fold min 0.9681 | STOP |
| MHC-ZH | 1.572e-7 | 1.010e-4 | 2.191e-5 | 3.709e-7 | 3.043e-2 | 0.0963 | `(0.20,0.30)`; fold min 0.9438 | STOP |

Frozen gates were `margin<=1e-5`, `T/cost<=2e-5`, relative gradient `<=1e-3`, minimum norm `>=1e-4`, all-fold support, batch32 no OOM and peak `<=24 GiB`. Thus support, completeness, finite, margin, gradient, norm, runtime and memory gates passed, but both T and cost parity gates failed on both datasets. Per the preregistered rule, this exact CTE route stops without changing thresholds, kernel precision, anchor IDs, radii or other hyperparameters.
