# LB-SCGP G0 Sanitizer Physical Artifact Review

**Date:** 2026-07-11  
**Scope:** physical sanitizer artifacts, locks, logs, scheduler records, JSON payload hashes, and formal no-segment/no-teacher counters only. This is not a G0 freeze, not a G0 PASS decision, not a performance claim, and not authorization for G1 or teacher stages.

## Verdict

CRITICAL: **0**  
HIGH: **0**

Remaining C1 status: **closed at artifact level**.

Meaning of closure: the Round3 physical blocker "the train-only sanitizer artifacts do not exist and cannot be hash-bound" is now closed by physical evidence from jobs `12738` and `12739`. This does not close later G0 freeze, code-audit, synthetic, realfold, replay, decision, G1, or teacher gates.

## Evidence Commands

Scheduler:

```text
sacct -j 12737,12738,12739 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End,AllocCPUS,ReqMem,SubmitLine%90 -P
```

Hashes:

```text
sha256sum artifacts/lb_scgp/inputs/MHC_zh/fold4/outer_train_features.pt artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json slurm/logs/lbscgp_sanitize_12739.out artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json slurm/logs/lbscgp_sanitize_12738.out slurm/logs/lbscgp_sanitize_12737.out configs/lb_scgp/lb_scgp_v1.json configs/lb_scgp/lb_scgp_sanitizer_sources.json
```

Payloads:

```text
jq -cS 'del(.payload_sha256)' artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json | tr -d '\n' | sha256sum
jq -cS 'del(.payload_sha256)' artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json | tr -d '\n' | sha256sum
jq -cS 'del(.payload_sha256)' artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json | tr -d '\n' | sha256sum
```

No segment/subclip artifact:

```text
find artifacts/lb_scgp -type f -iname '*segment*'
find artifacts/lb_scgp -type f -iname '*subclip*'
```

## Physical Hash Ledger

```text
ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496  artifacts/lb_scgp/inputs/MHC_zh/fold4/outer_train_features.pt
b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007  artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json
055dffed9b61053293741ec5ba0ce3577daf458ceef4a3f81143a81d937c684b  artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json
40f333bb82884e9ad412fc87a20165e1640238815c3c6b6951d003e1cb0ec247  slurm/logs/lbscgp_sanitize_12739.out
172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954  artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json
3314b3a7ea4f4b602cb28357258a5edf716cfa0abcd1bc8440d80fc0f222c978  slurm/logs/lbscgp_sanitize_12738.out
80d071490eb9f799ed119dce04d9004aad30a5fd2189ccc6b2c09d7d4fa4b4d7  slurm/logs/lbscgp_sanitize_12737.out
82e32bfa9fa552744106fd8a5b9c2e07de8e55ab0ad29c35cb5a6907ca744b43  configs/lb_scgp/lb_scgp_v1.json
ceb1676bfe81ad62373911458570a681d05b632beebe3b5b757c9c7e83eb5aae  configs/lb_scgp/lb_scgp_sanitizer_sources.json
```

Embedded payload rechecks:

```text
37b9221aee1cb570c2790228854f1889d539148a10d84bea5b9a98b1fca61996  sanitized_provenance payload without payload_sha256
576a682da04ebd992d3be2a091404b97e5de8f36b1f74809f05b067ccc728dea  quarantine manifest payload without payload_sha256
8685c805995f97ad0513658c1345b6320dc794e2bad02b907d9a2d07ff16cb1f  sanitizer decision payload without payload_sha256
```

## Scheduler And Logs

- `12737`: `FAILED`, exit `1:0`, `00:00:04`, `TASK=build`. Log preserves the relative-path failure before artifacts.
- `12738`: `COMPLETED`, exit `0:0`, `00:00:05`, `TASK=build`. Log emits `status=SANITIZED` and the feature hash `ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496`.
- `12739`: `COMPLETED`, exit `0:0`, `00:00:05`, `TASK=verify`. Log emits `status=PASS`.

## Lock And Artifact Presence

All four payloads and all four persistent publish locks exist:

```text
artifacts/lb_scgp/inputs/MHC_zh/fold4/outer_train_features.pt
artifacts/lb_scgp/inputs/MHC_zh/fold4/outer_train_features.pt.publish.lock
artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json
artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json.publish.lock
artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json
artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json.publish.lock
artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json
artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json.publish.lock
```

The `.publish.lock` files are part of the non-clobber contract and are expected to remain.

## Formal Contract Checks

The sanitizer decision has:

```text
status: PASS
slurm_job_id: 12739
feature_cache_sha256: ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496
sanitized_provenance_sha256: b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007
memory_id_count: 464
query_id_sentinel_count: 115
safe_contract_sha256: 69b789d094539583640dd5a7909dc05248236ac37db199c659a226aea051fc47
```

Every decision gate is true:

```text
exact_id_order
feature_schema_whitelist
formal_provenance_sanitized
labels_inherited_from_memory_labels
no_clobber_locks
no_network_external_calls
no_segment_artifact
no_segment_objective
no_teacher_mllm_ocr_calls
pre_freeze_disclosure_record_present
zero_overlap_with_query_ids
```

The formal provenance and decision both record:

```text
segment_cache_path=null
segment_cache_sha256=null
segment_artifact_created=false
segment_objective_allowed=false
teacher_mllm_ocr_calls=0
network_external_calls=0
formal_query_z_read_count=0
formal_query_labels_read_count=0
```

The only gold supervision remains:

```text
only_gold_supervision=parent_video_binary_label
segment_gold_exists=false
segment_gold_used=false
```

No `segment` or `subclip` files exist under `artifacts/lb_scgp`.

## Findings

CRITICAL: none.

HIGH: none.

LOW / residual process note: G0 has not been frozen or decided. The formal code-audit artifact and the registered freeze/synthetic/realfold/replay/decision stages remain future gates. This is outside the physical sanitizer-artifact review scope.

## Post-Update Validation Ledger

JSON syntax command:

```text
jq empty TARGET_STATE.json configs/lb_scgp/lb_scgp_v1.json configs/lb_scgp/lb_scgp_sanitizer_sources.json artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json
```

Result: exit code `0`, no output.

Decision consistency command:

```text
jq -r '[.status,.payload_sha256,.feature_cache_sha256,.sanitized_provenance_sha256,.memory_id_count,.query_id_sentinel_count,(.gates|to_entries|map(select(.value!=true))|length),.segment_artifact_created,.segment_objective_allowed,.teacher_mllm_ocr_calls] | @tsv' artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json
```

Result:

```text
PASS	8685c805995f97ad0513658c1345b6320dc794e2bad02b907d9a2d07ff16cb1f	ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496	b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007	464	115	0	false	false	0
```

Updated record hashes, excluding this self-referential review file:

```text
ddea5a0b872b2fac8cc79031179898fc46905bebdfa6e0856defdcffdea3660b  refine-logs/lb_scgp/G0_SANITIZER_EXECUTION.md
6f89fd78880219466ee88022ee3e6f44d264c981738d8dc6ea906d17725d0a42  refine-logs/lb_scgp/EXPERIMENT_TRACKER.md
274f119f6f1ca20c1b96360790893863e2d075b9c7e48387e07103d110d72ecf  refine-logs/lb_scgp/G0_IMPLEMENTATION_HANDOFF.md
4e82ce7e3f5468cb86d1d2fba9dfb5b971a4d8514a758d3bd0c87749f61c790f  refine-logs/lb_scgp/G0_ROUND3_FIX_HANDOFF.md
a86f5ae999eb0bf299c56f69d7c8254092b689d0e6a3cbd7bf8cc19197282ce1  TARGET_STATE.json
734c5f8d38774f11fa2e02889bdd93b0a1d3f72988de77093decd6d2c96f06b9  TARGET_FINDINGS.md
b0387f49c7dab9686da2baa10ace3c60cbd27e0dc422810974baca652dd49a63  TARGET_LOOP.md
2ca63ace4322c9aa8fcf5450e40101acdcc813c4af53c81557bcdf8cf53d0345  refine-logs/lb_scgp/G0_SANITIZER_PATH_FIX_REVIEW.md
9a5c7ac6449a7e9c967feed203b08e26a2252d38b08e378bc8886d30eb3ecf1b  refine-logs/lb_scgp/G0_INDEPENDENT_REVIEW_ROUND3.md
```

## Final Physical Review Decision

Artifact-level C1 is **closed**: the required train-only whole-video feature artifact, sanitized provenance, quarantine manifest, verifier decision, and locks physically exist; their hashes and embedded payload hashes are consistent; the verifier decision is PASS; feature IDs are 464; held-query sentinels are 115; there is no segment artifact/objective; teacher/MLLM/OCR/network counts remain zero.

G0 PASS is **not** claimed. G1 and teacher remain locked.
