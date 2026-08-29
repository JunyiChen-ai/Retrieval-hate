# Experiment Plan Review: LB-SCGP Global

Verdict: **REVISE**

Severity counts: Critical 0, High 5, Medium 2, Low 2.

Explicit decision: **0 Critical and 0 High: NO**. The plan is not ready for implementation until the High findings are fixed and the machine/tracker artifacts are regenerated.

Review constraints honored: read-only review of the five requested plan artifacts; no subagents, external model calls, experiments, SLURM submissions, or edits to plan/proposal/tracker/history artifacts.

## Top Findings

| ID | Severity | Finding | Exact locations | Required fix |
|---|---:|---|---|---|
| H1 | High | The "moving strongest same-protocol non-MLLM comparator" is not actionably frozen. The plan requires comparator-before-FULL, but it does not define the candidate set, source ledgers, selection criterion, or no-test rule that makes "strongest" reproducible. | `EXPERIMENT_PLAN.md` Claim Map/Baselines/B2-B3 lines 26-39, 75-92; tracker runs 7, 9, 12-23 lines 18-34; machine `baseline_families.F1` lines 75-79 and runs 279-667. | Add a comparator-freeze artifact before M2 that lists eligible non-MLLM systems, hashes, metrics, selection data, tie-breaks, and proves selection used no final test labels or adaptive held/query labels. |
| H2 | High | The decisive direct-control set is narrower than the proposal. The final proposal requires DIRECT-MOMENT and DIRECT-CERT-FEATURE / scalar controls, but the experiment plan and machine JSON only name one direct certificate distillation arm plus scalar. This leaves a direct-moment replaceability route untested. | `FINAL_PROPOSAL.md` controls lines 564-571 and 586-590; `EXPERIMENT_PLAN.md` Baseline Families/B4 lines 37-39 and 98-100; machine `baseline_families.F3` lines 84-87 and M4 control arms lines 679, 703, 727, 751, 775, 799. | Add an explicit DIRECT-MOMENT arm or state, with schema-level detail, that the direct arm covers both moment and certificate-feature routes. Update Table 2, run arms, artifact schemas, gates, and budgets. |
| H3 | High | Control matching can leak or become test-adaptive because SHUFFLE is matched on label/prediction/margin/error-propensity bins while M4 scope is final train/test. The plan does not state that all bins, propensities, and matching manifests are train-only or pre-final OOF/validation artifacts. | `EXPERIMENT_PLAN.md` Baseline Families and Matching Rules lines 37-39 and 97-100; tracker M4 runs 24-29 lines 35-40; machine M4 runs 670-812. | Specify that matching bins and scalar/error-propensity estimates are computed only from train/OOF or preregistered validation artifacts frozen before final test evaluation. Add counters proving final test labels/predictions/margins are evaluator-only and never used for control construction. |
| H4 | High | M4 control bundles hide multiple trained arms under an implausibly low GPU budget. Each bundle contains REMOVE, SHUFFLE, NOISE, DIRECT, and SCALAR; direct/scalar controls are said to use the same split/init/schedule/inference constraints as FULL, but each bundle budgets only 36 GPU-h while a single M3 comparator or FULL run budgets 18 GPU-h. Four trained controls would imply about 72 GPU-h per bundle before overhead. | `EXPERIMENT_PLAN.md` B4 and cost table lines 94-120; tracker M4 runs 24-29 lines 35-40; machine M4 runs 670-812 and budget totals lines 933-935. | Split M4 controls into per-arm runs or itemize per-arm shared-work assumptions. Recompute GPU/CPU/storage totals and concurrency after adding DIRECT-MOMENT. |
| H5 | High | The hierarchical paired bootstrap is named but not defined. The plan lacks the resampling hierarchy, uncertainty unit, paired statistic, macro-F1 recomputation rule, one-sided lower-bound definition, and Holm p-value source. This is not enough to implement the final C2 statistical gate. | `EXPERIMENT_PLAN.md` B3/B4/Table lines 91, 101, 167; tracker run 31 line 42; machine run 31 lines 837-858 and schema line 974. | Define the dataset x metric test exactly: paired prediction unit, seed cluster handling, video resampling within seed/dataset, bootstrap replicate statistic, confidence/lower-bound calculation, p-values, Holm order over four tests, and how +0.030 and lower>0 are jointly enforced. |
| M1 | Medium | Aggregate budget arithmetic is inconsistent with the run matrix. Summing machine runs gives 488 must-run CPU-h and 504 total CPU-h, not 504/520; must-run storage sums to 578 GB and total to 583 GB, not 575/580. GPU/API totals do match. | `EXPERIMENT_PLAN.md` Compute and Data Budget lines 124-131; machine `budget_ranges` lines 933-935; tracker row budgets lines 12-43. | Regenerate aggregate totals from the machine run list after fixing M4 controls. |
| M2 | Medium | Isolation counters are not schema-complete for final inference. The text forbids validation/test cache, compiler target, teacher, head, reranker, key selector, and certificate-feature loads, but the machine `zero_access_counters` only records cache/teacher/MLLM call counters. | `EXPERIMENT_PLAN.md` Immutable Contract and B2/B3 lines 17-19, 81-83, 89-91; machine `zero_access_counters` lines 953-964 and schemas lines 971-973. | Add explicit zero counters to target/eval/control schemas for validation/test cache reads, certificate reads, compiler-target reads, auxiliary head loads, reranker loads, key-selector loads, and teacher artifacts. |
| L1 | Low | The machine `terminal_decision_chain` skips the M3 paired-main completion node. The actual dependency DAG is acyclic and M4/M5 dependencies recover this, but the decision-chain summary is incomplete. | machine `dependency_dag.terminal_decision_chain` lines 906-916. | Add an explicit M3 paired-main-complete decision before M4 controls decision. |
| L2 | Low | Machine `source_hashes` include validation/test JSONL files without clarifying that these hashes are provenance-only and not experimental access for cache, matching, hyperparameter selection, or G0. | machine `source_hashes` lines 14-19; `EXPERIMENT_PLAN.md` Immutable Contract lines 17-19. | Mark val/test hashes as provenance-only, or move them to final evaluator provenance artifacts generated after freeze. |

## Claim Coverage Matrix

| Claim | Required evidence | Plan coverage | Review result |
|---|---|---|---|
| C1: executable, isolated, certifiable, encoder-realizable global geometry | First G0 gates, closed strong convexity, common Q, serialized H-metric normal-cone/KKT-only acceptance, rank-tail <= d, nondegeneration, REMOVE/null parity, replay/hash/isolation, resource preflight, robust low coverage disables only safety claim, local v7 not accepted as PASS. | Mostly covered in `EXPERIMENT_PLAN.md` B0 lines 51-60, first-three G0 lines 105-111 and 187-191, artifact contract lines 146-163; machine G0 runs 140-208. | Covered in design, but implementation readiness is blocked by H3/M2 leakage-counter completeness and M1/H4 resource realism. |
| C2: final ordinary-kNN performance and attribution | MHC and MHC-ZH, seeds 0/1/2, FULL vs strongest same-protocol non-MLLM comparator, accuracy and macro-F1 each >= +0.030, all paired seed deltas positive, hierarchical paired bootstrap lower >0, Holm over four dataset-metric claims, FULL beats REMOVE/SHUFFLE/NOISE/direct/scalar controls under ordinary kNN. | Main run structure exists in B2-B4 lines 75-103, run order lines 113-120, and machine M2-M5 runs 279-880. | Not ready: comparator selection is not frozen (H1), direct-moment attribution is missing (H2), control construction can be test-adaptive (H3), controls are under-budgeted (H4), and the bootstrap/Holm procedure is underspecified (H5). |

No unsupported success claim is emitted by the plan; it repeatedly states planning readiness is not success. The risk is implementation readiness, not claimed experimental success.

## Leakage / Gold-Boundary Audit

Pass:
- Only `parent_video_binary_label` is named as gold; segment/frame/timestamp/span/localization/stance/target/mechanism/rationale/fragment gold is explicitly false or forbidden in `EXPERIMENT_PLAN.md` lines 14-20 and machine lines 32-37.
- Cache is train-only and label-blind, with labels entering after seal only, in `EXPERIMENT_PLAN.md` lines 17-18 and B1 lines 62-73.
- Final endpoint is ordinary full-video train-memory top20 kNN with no MLLM/cache/teacher/head/rerank/key selection in `EXPERIMENT_PLAN.md` lines 18-20.

Blocking concerns:
- H3: M4 matching on label/prediction/margin/error-propensity bins is not explicitly train-only or pre-final OOF/validation-only.
- M2: access counters do not yet enumerate all forbidden validation/test artifact loads.

## Run-Order / DAG Audit

Pass:
- First three runs are exactly G0: `LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1`, `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1`, `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1`.
- JSON run count is 33 total, 32 MUST and 1 NICE; run order matches run IDs.
- Dependency check is acyclic, with no unknown dependencies and no dependency ordered after its consumer.
- Comparator runs precede paired FULL runs in M2 and M3.
- No validation/test performance run appears before G0 and M1 cache seal gates.

Concern:
- L1: terminal decision-chain summary omits M3 paired-main completion.

## Resource / Budget Arithmetic Audit

Pass:
- Per-run SLURM requests are within 16 CPU / 128 GB / 2 GPU, use `HateVideo`, and set `no_time_flag=true`.
- Planned concurrency caps avoid oversubscription if followed: at most two 1-GPU jobs, G0 serial, M1 CPU parallel max 8 CPU/64 GB.
- API call formula is coherent: base 4512, retry cap 9024.

Blocking concerns:
- H4: M4 control bundles under-budget multiple trained arms.
- M1: declared CPU/storage totals do not match the machine run matrix.

Computed from machine JSON:
- Must-run CPU-h: 488, declared 504.
- Total CPU-h with NICE: 504, declared 520.
- Must-run GPU-h: 480, declared 480.
- Must-run API base/retry cap: 4512/9024, declared 4512/9024.
- Must-run storage: 578 GB, declared 575 GB.
- Total storage with NICE: 583 GB, declared 580 GB.

## Statistics Audit

Pass:
- Two datasets and seeds 0/1/2 are present.
- Accuracy and macro-F1 are both required.
- Mean deltas >= +0.030, all seed deltas positive, hierarchical bootstrap lower >0, and Holm over four dataset-by-metric tests are required.
- Test labels are described as evaluator-only.

Blocking concern:
- H5: the bootstrap/Holm protocol is not implementable without a defined resampling hierarchy and paired statistic.

## Teacher / Cache / Control Audit

Pass:
- M1 API formula is exact and seed-independent: `4 * (U_MHC + U_MHC_zh) + R_MHC + R_MHC_zh`.
- Deduplication, retry cap, fallback to all-unresolved, seal, Merkle root, and cost uncertainty are specified.
- Certificate-field predictivity is diagnostic only.
- REMOVE/SHUFFLE/NOISE/direct/scalar controls are represented at the family level.

Blocking concerns:
- H2: direct-moment control from the final proposal is missing or not explicitly covered.
- H3: control matching inputs need train-only/OOF provenance.
- H4: control bundles need realistic per-arm budgets.

## Machine / Tracker Parity Audit

Pass:
- `EXPERIMENT_PLAN.machine.json` validates as JSON.
- Every machine run has dataset/split/system/seed/dependencies/metrics/artifacts/schemas/resources/budget/gate/failure/parallel/priority/status fields.
- Tracker and machine run IDs, counts, ordering, dependencies, and per-run budgets are aligned.
- `EXPERIMENT_PLAN_HASHES.sha256` matches the three plan artifacts.
- `global_pivot_validated=false`, `planning_only=true`, `segment_gold_exists=false`, and `segment_gold_used=false` are explicit.

Concerns:
- M1 aggregate totals mismatch run sums.
- L1 terminal decision-chain summary incomplete.
- L2 val/test source hashes need provenance clarification.

## First-Three-Runs Audit

Result: PASS.

The first three tracker and machine runs are all G0, are serial, and contain no performance metrics:
1. `LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1`
2. `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1`
3. `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1`

## Prioritized Author Revision Checklist

1. Freeze the moving strongest comparator candidate set and selection ledger before any FULL validation/final run.
2. Add DIRECT-MOMENT or prove the direct-control arm covers both direct moment and direct certificate-feature routes.
3. Make all SHUFFLE/scalar matching inputs train-only or pre-final OOF/validation-only; add explicit final-test construction-access counters.
4. Split or itemize M4 control arms and recompute resource budgets, storage, run count if needed, and concurrency.
5. Define the hierarchical paired bootstrap and Holm protocol precisely enough to implement.
6. Regenerate machine JSON/tracker parity, aggregate totals, access-counter schemas, and hash manifest.

Implementation readiness: **not ready** until all High findings are resolved and the regenerated artifacts pass parity/hash checks.
