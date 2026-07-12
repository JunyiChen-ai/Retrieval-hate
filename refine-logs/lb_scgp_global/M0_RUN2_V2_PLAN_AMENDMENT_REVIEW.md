# M0 Run2-v2 Plan Amendment Review: LB-SCGP Global-R2

Date: 2026-07-12

Reviewer boundary: fresh independent local amendment review only. I did not resume or rely on failed thread `019f53f3-419d-7ad1-9626-192916e054c4`. No subagents, workflows, SLURM submission/monitoring, experiment code, Python, ML, API, MLLM, OCR, GPU work, data processing, implementation, code/config/schema/log/artifact modification, or execution was performed. This review creates only the three allowed reviewer artifacts.

## Verdict

Verdict: **AUTHORIZE V2 IMPLEMENTATION AUDIT**

Authorization boundary:

- `execution_authorized=false`
- `run3_authorized=false`
- `v2_execution_authorized=false`
- `v2_implementation_audit_authorized=true`

This verdict authorizes only the next review boundary: a v2 implementation audit. It does not authorize v2 execution, Run3, MLLM/cache work, realbank resource work, GPU/training/performance work, validation/test work, or any downstream run. Before any v2 execution, the plan still requires completed v2 implementation, static contract matrix and negative tests, fresh independent v2 code review with 0 Critical / 0 High, exact hashes/no-clobber checks, and separate execution authorization.

## Severity Counts

| Severity | Count |
|---|---:|
| Critical | 0 |
| High | 0 |
| Medium | 0 |
| Low | 1 |

## Findings

### Critical

None.

### High

None.

### Medium

None.

### Low

L1: Pre-existing v2 workspace files are present but unreviewed.

Evidence: current workspace contains `configs/lb_scgp_global_r2/m0_synth_kkt_v2.json`, v2 schemas, v2 analysis scripts, v2 wrapper, and v2 SLURM script. The amendment and plan correctly state that these files are unauthorized, not implementation evidence, and not execution permission. This is a residual audit risk only: the future v2 implementation audit must hash-bind exactly reviewed files and enforce no-clobber lineage before any separate execution authorization.

## Audit Results

- No post-hoc scientific change found. The amendment preserves v1 science, thresholds, fixtures, expected decisions, KKT tolerances, movement/nondegeneration target, rank/factor rules, solver/math/verifier logic, intended claim, failure transition, resource request 8 CPU / 64 GB / 0 GPU, and planned envelope 32 CPU-h / 5 GB / 0 GPU/API. The only allowed repair is interface-key/contract alignment across config, schema, validator, wrapper, producer, verifier, and artifact paths.
- Run2-v1 is preserved exactly once as `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1`, status `FAIL_STOP`, with jobs `12902` and `12904` bound to the same v1 run record. No v1 artifact is accepted, no numerical/KKT/rank/factor scientific evidence is produced, and v1 is never called PASS.
- Run2-v2 has unique ID `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2`, planned config `configs/lb_scgp_global_r2/m0_synth_kkt_v2.json`, and planned namespace `artifacts/lb_scgp_global/v2/m0/synth_kkt/`.
- Run3 `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1` depends on v2 PASS and fresh independent v2 artifact review. Run3 and all later runs remain locked. Downstream IDs and order are unchanged after the v2 insertion.
- Counts and budgets are internally consistent: 65 MUST, 1 NICE, 66 total; lifetime MUST 736 CPU-h / 684 GPU-h / 4512 base API / 9024 retry-cap API / 791 GB; lifetime total 752 CPU-h / 684 GPU-h / 4512 base API / 9024 retry-cap API / 796 GB. Original/substitution envelope remains 704/720 CPU-h as stated; remaining prospective budget is 700/716 CPU-h; execution-authorized remaining budget is 0.
- Actual v1 diagnostic spend is exactly 2 jobs, 5 wall-seconds, 40 allocated CPU-seconds = 0.0111111111 CPU-hours, 0 GPU-hours, 0 API calls, and no scientific artifact storage.
- Gold/isolation contract is preserved: only `parent_video_binary_label` is gold; `segment_gold_exists=false`; `segment_gold_used=false`; no segment/frame/timestamp/span/localization/stance/target/mechanism/rationale/fragment gold exists or is used. The amendment performs no data/model/API access, tuning, or rescue.
- Existing v2 workspace files are explicitly unreviewed and unapproved. They confer no execution authority.
- The stale report `M0_RUN2_RESULT_TO_CLAIM_REVIEW.md` is treated as untrusted/stale provenance only. The authoritative current result-to-claim report is `M0_RUN2_RESULT_TO_CLAIM_REVIEW_FRESH.md`.

## Mechanical Self-Checks

| Check | Result |
|---|---|
| `jq empty` on plan/amendment machine JSON | PASS |
| `sha256sum -c EXPERIMENT_PLAN_HASHES.sha256` | PASS |
| `sha256sum -c M0_RUN2_V2_PLAN_AMENDMENT_HASHES.sha256` | PASS |
| Tracker rows | 66 |
| Machine `run_order` length / `runs` length | 66 / 66 |
| MUST / NICE counts | 65 / 1 |
| Tracker order vs machine `run_order` | PASS, `diff` clean |
| DAG unknown dependencies / order violations | 0 / 0 |
| v1 count in run order | 1 |
| v2 count in run order | 1 |
| v2 status | `LOCKED_PENDING_AMENDMENT_REVIEW` |
| Run3 dependency | `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2` |
| v1 synth artifact namespace | Absent |
| v2 synth artifact namespace | Absent |
| Plan zero-access counters | 48 counters, all zero |
| Job `12902` stderr/out hashes | Match amendment; stdout empty |
| Job `12904` stderr/out hashes | Match amendment; stdout empty |

## Inspected Input Hashes

| Path | SHA256 |
|---|---|
| `AGENTS.md` | `e6aaf5d66399cdbbe7fcc2c811931277b0ed4a24b592ffa5cbb60315b29ea23c` |
| `refine-logs/lb_scgp_global/EXPERIMENT_PLAN.md` | `8ef1417fe16f8a556c82394705259f13447e7747d600c0f8f8061e237ecb8994` |
| `refine-logs/lb_scgp_global/EXPERIMENT_TRACKER.md` | `d0c7e6ff291b6ae4ba6d5661b2e8ddc2bb8a80ac7f2f6d1e37ab8494782df483` |
| `refine-logs/lb_scgp_global/EXPERIMENT_PLAN.machine.json` | `838370c8eee68f568b6d133f80b305fed23bc14342d4c4bf08df976fd4d73d07` |
| `refine-logs/lb_scgp_global/EXPERIMENT_PLAN_HASHES.sha256` | `0839260bbe8c046a292762dff2cbc73e8623b70b3626d991ee17eecbe10c2a15` |
| `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT.md` | `95b4b839017479bad4d2ed7a48455864c765e8f91997adab0fa4a7ca66aa57fb` |
| `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT.machine.json` | `56603297b2a1f6bcf12e941d96ce95911bb2ab8bfd0de967586430d38959f3d8` |
| `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT_HASHES.sha256` | `a88fc1c75c10b653deebf03c1792733d2bb412f13dc021567b221d009cc38390` |
| `refine-logs/lb_scgp_global/M0_RUN2_RESULT_TO_CLAIM_REVIEW_FRESH.md` | `b02e5a0f7839e8b215c15197f6f120e3bebfe498ab89d2120c26b50585707a0c` |
| `refine-logs/lb_scgp_global/M0_RUN2_RESULT_TO_CLAIM_REVIEW.md` | `12f42df893d90d6ef4a7759dbbb86e8f8d820375be9aae3a0b94010fda1335ef` |
| `refine-logs/lb_scgp_global/M0_SYNTH_KKT_EXECUTION.md` | `31481cb10808e9a6ce81754c47d18351079bf84b602088618b0ecdd341c80333` |
| `refine-logs/lb_scgp_global/M0_SYNTH_KKT_IMPLEMENTATION.md` | `1e248d4f7b8432bf5d2f4f5e04c4a465f4d8b54c8fe72ddc4105a68e108484aa` |
| `refine-logs/lb_scgp_global/EXPERIMENT_PLAN_REVIEW.md` | `21051175a7cc3301c79b995dc222b9e82f041f1d5c214b39896e03e60db27f86` |
| `refine-logs/lb_scgp_global/EXPERIMENT_PLAN_REVIEW_R2.md` | `d952989742c9402cfd38de935deece73b2933e7ab8f70883ca8e0bea40a4bd46` |
| `refine-logs/lb_scgp_global/EXPERIMENT_PLAN_REVIEW_R2.machine.json` | `8c222ff6db3a38752851fbc113c9e1988845ca424ce8fc51d48cfdea0c51f472` |
| `refine-logs/lb_scgp_global/M0_CONTRACT_FREEZE_EXECUTION.md` | `71a502375ec4f99224e8a8cd55aa61ea9f96cff6bf81743646c59e10d1efeebb` |
| `refine-logs/lb_scgp_global/M0_CONTRACT_FREEZE_INDEPENDENT_REVIEW.md` | `37dcf8825ef9cf3af9df47a0346e647272c19d70110a1247cc6ed865beee34ed` |
| `artifacts/lb_scgp_global/v1/m0/contract_freeze.json` | `09b78682389f1c9774c9dffc43c759bceeec9d7f44eca1ce4cd626d0cd6d12da` |
| `artifacts/lb_scgp_global/v1/m0/contract_freeze.json.publish.lock` | `c6fbb49c3c8aba942da3f254c3c5f65d037002548fafb961154e6fa79a3e4ea7` |
| `slurm/logs/lbscgp_global_r2_run2_synth_kkt_12902.err` | `77ba892b49d7c6262bc0a6165188c173b8f0619a7cbbfe9fdcc881fc8ec5f71c` |
| `slurm/logs/lbscgp_global_r2_run2_synth_kkt_12902.out` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `slurm/logs/lbscgp_global_r2_run2_12904.err` | `93e8515cad0d89ec65d3a1844d497694324143b9d836bc304b732765f1ead306` |
| `slurm/logs/lbscgp_global_r2_run2_12904.out` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

## Final Boundary Statement

Because Critical=0 and High=0, the only allowed verdict is issued: **AUTHORIZE V2 IMPLEMENTATION AUDIT**. Execution remains unauthorized, and Run3 remains unauthorized.
