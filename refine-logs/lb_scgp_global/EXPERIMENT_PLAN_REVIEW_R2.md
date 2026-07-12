# Experiment Plan Review R2: LB-SCGP Global

Verdict: **READY_FOR_IMPLEMENTATION**

Severity counts: Critical 0, High 0, Medium 0, Low 0.

Explicit decision: **0 Critical and 0 High: YES**. The revised plan is actionably implementable as a plan. This is not experimental success and does not authorize any success claim before the planned gates run.

Review constraints honored: read-only review of proposal, revised plan/tracker/machine/hash artifacts, and R1 review artifacts; no subagents, external model/MLLM calls, experiments, SLURM submissions, GPU jobs, API calls, or plan artifact edits.

## R1 Closure Checks

| R1 ID | R2 status | Closure evidence |
|---|---|---|
| H1 comparator freeze | CLOSED | Mandatory `LBSCGP-GLOBAL-M2-COMPARATOR-FREEZE-v1` precedes M2 validation; candidate fields, eligible/rejected ledgers, train/OOF+validation-only selection, tie-breaks, and zero final-test/adaptive-query counters are specified in `EXPERIMENT_PLAN.md` lines 49-72 and machine lines 81-88, 184. |
| H2 DIRECT-MOMENT | CLOSED | F3 now explicitly contains `DIRECT-MOMENT`, `DIRECT-CERT-FEATURE`, and `SCALAR-PROPENSITY`; controls are distinct in `EXPERIMENT_PLAN.md` lines 39-42 and 119-125, and machine lines 61-65, 89-96. |
| H3 control leakage | CLOSED | SHUFFLE/scalar construction is train/OOF or preregistered validation-frozen only; final-test labels/predictions/margins/errors are evaluator-only with construction counters fixed to zero in `EXPERIMENT_PLAN.md` lines 120-124 and 201, and machine lines 89-96, 316-323. |
| H4 control resources | CLOSED | M4 is split into 36 per-arm control runs, each with separate CPU/GPU/storage budgets; aggregate budgets are mechanically consistent: 704 must CPU-h, 684 must GPU-h, 786 GB must storage; total with NICE is 720 CPU-h, 684 GPU-h, 791 GB. |
| H5 statistics | CLOSED | M5 defines four tests, paired unit, seed/video hierarchy, stratified video bootstrap, macro-F1 recomputation, one-sided 5th percentile lower bound, bootstrap p-value, Holm order, and joint delta `>= +0.030` plus lower `>0` enforcement in `EXPERIMENT_PLAN.md` lines 128-139 and machine lines 97-109. |
| M1 budget arithmetic | CLOSED | Machine sums match declared budgets and tracker header: 64 MUST, 1 NICE, 65 total; 704/720 CPU-h, 684 GPU-h, 4512 base API, 9024 retry cap, 786/791 GB storage. |
| M2 isolation counters | CLOSED | Validation/test cache, certificate, compiler-target, auxiliary-head, reranker, key-selector, teacher-artifact counters are explicit in `EXPERIMENT_PLAN.md` lines 20 and 201 and machine lines 292-323. |
| L1 M3 decision chain | CLOSED | `LBSCGP-GLOBAL-M3-PAIRED-MAIN-DECISION-v1` is a mandatory M3 decision before M4 in `EXPERIMENT_PLAN.md` line 112, tracker line 37, and machine terminal chain lines 246-255. |
| L2 val/test hashes | CLOSED | Hash policy marks validation/test hashes as provenance-only/evaluator-only and forbids cache, G0, control construction, adaptive tuning, comparator final-test selection, or MLLM use in `EXPERIMENT_PLAN.md` lines 43-47 and machine lines 30-35. |

## Claim Coverage Matrix

| Claim | Evidence plan | R2 decision |
|---|---|---|
| C1 executable, isolated, certifiable, encoder-realizable global geometry | B0/G0 covers common Q, closed strong convexity, PSD/unit diagonal, trust constraints, KKT-only payload, independent verifier, rank-tail `<= d`, nondegeneration, REMOVE/null parity, replay/hash/isolation, resource preflight, robust low coverage disables safety only, local v7 cannot pass. | Covered for implementation. |
| C2 final ordinary-kNN performance and attribution | Frozen comparator, two datasets, seeds 0/1/2, paired FULL vs comparator, ordinary top20 kNN, +0.030 accuracy and macro-F1, positive seed deltas, hierarchical bootstrap lower >0, Holm over four tests, and per-arm controls REMOVE/SHUFFLE/NOISE/DIRECT-MOMENT/DIRECT-CERT-FEATURE/SCALAR. | Covered for implementation. |

## Core Audits

- Leakage/gold boundary: PASS. Only `parent_video_binary_label` is gold; no segment/frame/timestamp/span/localization/stance/target/mechanism/rationale/fragment gold exists or is used.
- Run order/DAG: PASS. 65 machine runs; 64 MUST and 1 NICE; run order matches tracker; dependencies are acyclic with no unknown dependencies or order violations.
- First three G0: PASS. `LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1`, `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1`, `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1`.
- Resource rules: PASS. Every run is within 16 CPU / 128 GB / 2 GPU, uses `HateVideo`, and has `no_time_flag=true`. Cited SLURM scripts contain no active `--time` directive.
- Statistics: PASS. The M5 bootstrap/Holm gate is implementable and enforces both observed delta `>= +0.030` and one-sided lower bound `>0`.
- Tracker/machine parity: PASS. Tracker has 65 rows in the exact machine order; parsed per-run CPU/GPU/storage/SLURM resources match machine JSON.
- Hashes: PASS. `EXPERIMENT_PLAN_HASHES.sha256` verifies for revised plan, tracker, and machine JSON.
- Forbidden routes: PASS. No sample weighting, key selection, pair/triplet/SupCon, segment route, local v8, test teacher/head/rerank, or local v7 PASS reuse is planned.

## Plan Counts and Budgets

- MUST-RUN: 64.
- NICE-TO-HAVE: 1.
- Total runs: 65.
- MUST budget: 704 CPU-h, 684 GPU-h, 4512 base API calls, 9024 retry-cap API calls, 786 GB cumulative storage.
- Total including NICE: 720 CPU-h, 684 GPU-h, 4512 base API calls, 9024 retry-cap API calls, 791 GB cumulative storage.

Implementation readiness: **READY_FOR_IMPLEMENTATION** with no remaining nonblocking R2 findings.
