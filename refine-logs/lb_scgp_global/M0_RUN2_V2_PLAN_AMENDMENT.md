# M0 Run2-v2 Plan Amendment: LB-SCGP Global-R2

Date: 2026-07-12

Author thread: fresh exact GPT-5.5 xhigh prospective plan-amendment author in `/data/jehc223/RGCL`. No subagent, workflow, external model, code execution, SLURM submission, experiment, API, MLLM, OCR, GPU, training, or performance work was performed.

## Verdict

This is a prospective planning amendment only. It is ready for independent amendment review. It does not authorize v2 implementation or v2 execution.

If independent amendment review passes, the only next boundary it can authorize is a v2 implementation audit. Execution requires completed v2 implementation, static contract matrix and negative tests, fresh independent v2 code review with 0 Critical / 0 High, exact hashes and no-clobber check, and separate execution authorization.

Run3 and all later runs, including MLLM/cache work, remain locked until Run2-v2 PASS and fresh independent v2 artifact review.

## Preserved Run2-v1 Evidence

Run2-v1 remains exactly one historical planned run record:

`LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1`

It is FAIL_STOP infrastructure evidence only, with zero remaining budget. Both attempted SLURM jobs stay bound to the v1 run ID and v1 namespace.

| Job | Resources | Elapsed | State | Failure | Artifact |
|---|---:|---:|---|---|---|
| `12902` | 8 CPU / 64 GB / 0 GPU | 4 seconds | FAILED `1:0` | older producer path, `KeyError: finite_vi_diagnostic` | none |
| `12904` | 8 CPU / 64 GB / 0 GPU | 1 second | FAILED `1:0` | newer validator path, `KeyError: payload_schema` | none |

Both failures occurred before publish. They are not scientific, numerical, KKT, rank, factor, mechanism, dataset, MLLM/OCR, GPU, training, validation/test, or performance evidence. They must never be deleted, overwritten, reused as v2, or called PASS.

Actual diagnostic spend from both v1 attempts: 2 jobs, 5 wall-seconds total, 40 allocated CPU-seconds = 0.0111111111 CPU-hours, 0 GPU-hours, 0 API calls, and no scientific artifact storage.

## New Prospective Run

Add exactly one new MUST run:

`LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2`

Planned config: `configs/lb_scgp_global_r2/m0_synth_kkt_v2.json`.

Planned artifact namespace: `artifacts/lb_scgp_global/v2/m0/synth_kkt/`.

All schemas, locks, source manifests, access ledgers, payloads, and publish locks must be v2-lineage bound. Unreviewed partial v2 files currently present in the workspace are unauthorized and are not implementation evidence or execution permission.

Dependencies and gates include Run1 FROZEN, preserved v1 failure evidence, approved amendment after independent amendment review, static contract matrix and negative tests, fresh independent v2 code review with 0 Critical / 0 High, exact hashes/no-clobber check, and separate execution authorization.

## Science Freeze

The v2 repair is only interface-key/contract alignment across config, schema, validator, wrapper, producer, verifier, and artifact paths.

Unchanged from v1:

- resource request: 8 CPU / 64 GB / 0 GPU, `HateVideo`, no `--time`;
- planned budget: 32 CPU-h, 0 GPU-h, 0 API calls, 5 GB;
- schema intent, thresholds, fixtures, expected decisions, KKT tolerances;
- movement/nondegeneration target, rank/factor rules, solver/math/verifier logic;
- intended claim and failure transition.

Forbidden: solver fix, tuning, tolerance change, fixture change, rescue, or post-hoc scientific change.

## DAG Amendment

Run order prefix after amendment:

1. `LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1`
2. `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1`
3. `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2`
4. `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1`

Run3 `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1` now depends on v2 PASS and fresh independent v2 artifact review, not v1. All other run IDs remain unchanged and downstream ordering remains valid.

## Budget Accounting

Conventions:

- v1 is one historical planned run record with failed infrastructure evidence and zero remaining budget.
- v2 is one new prospective MUST record with the original 32 CPU-h / 5 GB planned envelope.
- The two v1 SLURM attempts are actual diagnostic spend under v1, not two extra planned scientific runs.

Original approved R2 paper-plan envelope before v2: 64 MUST plus 1 NICE; MUST 704 CPU-h / 684 GPU-h / 4512 base API / 9024 retry-cap API / 786 GB; total with NICE 720 CPU-h / 684 GPU-h / 4512 base API / 9024 retry-cap API / 791 GB.

Substitution paper-plan view: replacing exhausted v1 with v2 keeps the aggregate paper-plan envelope unchanged at the same totals above.

Lifetime lineage envelope after retaining v1 and adding v2: 65 MUST plus 1 NICE, 66 records total; MUST 736 CPU-h / 684 GPU-h / 4512 base API / 9024 retry-cap API / 791 GB; total with NICE 752 CPU-h / 684 GPU-h / 4512 base API / 9024 retry-cap API / 796 GB.

Remaining prospective budget after Run1 FROZEN and v1 FAIL_STOP: MUST 700 CPU-h / 684 GPU-h / 4512 base API / 9024 retry-cap API / 785 GB; total with NICE 716 CPU-h / 684 GPU-h / 4512 base API / 9024 retry-cap API / 790 GB. Current execution-authorized remaining budget is 0.

## Gold And Isolation

Only gold supervision remains `parent_video_binary_label`.

`segment_gold_exists=false` and `segment_gold_used=false`. No segment, frame, timestamp, span, localization, stance, target, mechanism, rationale, or fragment gold exists or is used.

All relevant forbidden access/call counters remain zero, including MLLM/OCR/API calls, validation/test/held/cache/certificate/compiler-target/teacher/head/reranker/key-selector access, query label/embedding access, and final-test construction access.

No science tuning is authorized.
