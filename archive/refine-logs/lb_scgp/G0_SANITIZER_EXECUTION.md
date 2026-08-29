# LB-SCGP Pre-G0 Sanitizer Execution

**Date:** 2026-07-11  
**Worker mode:** static artifact inspection/documentation only for this update. No Python, no SLURM submission, no GPU work, no G0 freeze, no synthetic/realfold/replay/decision work, no G1, no teacher, no MLLM, no OCR, no network.

## Scope

This record covers the complete pre-G0 sanitizer history now present in the repository:

- job `12737`: first sanitizer build attempt, failed before artifact creation.
- job `12738`: sanitizer build rerun, completed and produced physical artifacts.
- job `12739`: independent sanitizer verifier, completed and wrote a PASS decision.

This closes the Round3 physical C1 blocker at artifact level only. It does **not** claim G0 PASS, does **not** run or unlock G0 freeze, and does **not** unlock G1 or teacher stages. The sole gold supervision remains the parent-video binary label. No segment-level gold is assumed or used.

## Scheduler Evidence

Command:

```text
sacct -j 12737,12738,12739 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End,AllocCPUS,ReqMem,SubmitLine%90 -P
```

Evidence:

```text
12737|lbscgp_sanitize|FAILED|1:0|00:00:04|2026-07-11T10:20:43|2026-07-11T10:20:47|4|32G|sbatch --export=ALL,TASK=build scripts/slurm/lb_scgp_sanitize_inputs.sbatch
12738|lbscgp_sanitize|COMPLETED|0:0|00:00:05|2026-07-11T10:43:32|2026-07-11T10:43:37|4|32G|sbatch --export=ALL,TASK=build scripts/slurm/lb_scgp_sanitize_inputs.sbatch
12739|lbscgp_sanitize|COMPLETED|0:0|00:00:05|2026-07-11T10:44:39|2026-07-11T10:44:44|4|32G|sbatch --export=ALL,TASK=verify scripts/slurm/lb_scgp_sanitize_inputs.sbatch
```

No `scontrol release`, cancel, requeue, or bypass command is recorded in this inspection. `scontrol show job 12738` and `scontrol show job 12739` no longer have live records, so `sacct`, logs, and physical artifact hashes are the binding evidence.

## Job 12737 Failure History

Job `12737` failed before sanitizer artifact creation during config loading:

```text
ValueError: 'configs/lb_scgp/lb_scgp_v1.json' is not in the subpath of '/data/jehc223/RGCL' OR one path is relative and the other is absolute.
```

Log hash:

```text
80d071490eb9f799ed119dce04d9004aad30a5fd2189ccc6b2c09d7d4fa4b4d7  slurm/logs/lbscgp_sanitize_12737.out
```

Static root cause was `AccessLedger.hash_file` attempting to record a relative config path against absolute `ROOT`. This was a path-normalization implementation failure, not a sanitizer data-contract PASS/FAIL decision. The repair review in `refine-logs/lb_scgp/G0_SANITIZER_PATH_FIX_REVIEW.md` reported 0 Critical / 0 High and authorized a fresh sanitizer build under SLURM.

## Job 12738 Build

Log:

```text
{"feature_sha256":"ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496","run_id":"LBSCGP-G0-SANITIZE-MHC_zh-F4-v1","status":"SANITIZED"}
```

Physical artifacts and hashes:

```text
ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496  artifacts/lb_scgp/inputs/MHC_zh/fold4/outer_train_features.pt
b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007  artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json
055dffed9b61053293741ec5ba0ce3577daf458ceef4a3f81143a81d937c684b  artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json
3314b3a7ea4f4b602cb28357258a5edf716cfa0abcd1bc8440d80fc0f222c978  slurm/logs/lbscgp_sanitize_12738.out
```

Sanitized provenance embedded fields:

```text
slurm_job_id: 12738
feature_cache_sha256: ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496
memory_id_count: 464
query_id_sentinel_count: 115
segment_cache_path: null
segment_cache_sha256: null
segment_artifact_created: false
segment_objective_allowed: false
teacher_mllm_ocr_calls: 0
network_external_calls: 0
formal_query_z_read_count: 0
formal_query_labels_read_count: 0
payload_sha256: 37b9221aee1cb570c2790228854f1889d539148a10d84bea5b9a98b1fca61996
```

The quarantine manifest is not a formal G0 input. It records the pre-freeze mixed-cache source disclosure and confirms `formal_g0_input=false`, `subclip_source_opened=false`, and `subclip_output_created=false`.

## Job 12739 Verifier

Log:

```text
{"run_id":"LBSCGP-G0-SANITIZER-VERIFY-MHC_zh-F4-v1","status":"PASS"}
```

Physical decision and log hashes:

```text
40f333bb82884e9ad412fc87a20165e1640238815c3c6b6951d003e1cb0ec247  slurm/logs/lbscgp_sanitize_12739.out
172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954  artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json
```

Decision embedded fields:

```text
status: PASS
slurm_job_id: 12739
payload_sha256: 8685c805995f97ad0513658c1345b6320dc794e2bad02b907d9a2d07ff16cb1f
feature_cache_sha256: ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496
sanitized_provenance_sha256: b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007
memory_id_count: 464
query_id_sentinel_count: 115
segment_cache_path: null
segment_cache_sha256: null
segment_artifact_created: false
segment_objective_allowed: false
teacher_mllm_ocr_calls: 0
network_external_calls: 0
formal_query_z_read_count: 0
formal_query_labels_read_count: 0
```

All verifier gates are true:

```text
exact_id_order=true
feature_schema_whitelist=true
formal_provenance_sanitized=true
labels_inherited_from_memory_labels=true
no_clobber_locks=true
no_network_external_calls=true
no_segment_artifact=true
no_segment_objective=true
no_teacher_mllm_ocr_calls=true
pre_freeze_disclosure_record_present=true
zero_overlap_with_query_ids=true
```

## Locks And No-Clobber

Required persistent publish locks exist:

```text
artifacts/lb_scgp/inputs/MHC_zh/fold4/outer_train_features.pt.publish.lock
artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json.publish.lock
artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json.publish.lock
artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json.publish.lock
```

Artifact sizes and mtimes:

```text
artifacts/lb_scgp/inputs/MHC_zh/fold4/outer_train_features.pt 3341216 bytes 2026-07-11 10:43:36.602496536 +1200
artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json 1472 bytes 2026-07-11 10:43:36.609496470 +1200
artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json 1702 bytes 2026-07-11 10:44:44.534856947 +1200
artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json 3563 bytes 2026-07-11 10:43:36.610496461 +1200
```

## Payload Rechecks

Shell/jq payload-hash checks:

```text
jq -cS 'del(.payload_sha256)' artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json | tr -d '\n' | sha256sum
37b9221aee1cb570c2790228854f1889d539148a10d84bea5b9a98b1fca61996  -

jq -cS 'del(.payload_sha256)' artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json | tr -d '\n' | sha256sum
576a682da04ebd992d3be2a091404b97e5de8f36b1f74809f05b067ccc728dea  -

jq -cS 'del(.payload_sha256)' artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json | tr -d '\n' | sha256sum
8685c805995f97ad0513658c1345b6320dc794e2bad02b907d9a2d07ff16cb1f  -
```

The recomputed decision payload matches the embedded payload `8685c805995f97ad0513658c1345b6320dc794e2bad02b907d9a2d07ff16cb1f`.

## Segment And Supervision Audit

Artifact scan:

```text
find artifacts/lb_scgp -type f -iname '*segment*'
find artifacts/lb_scgp -type f -iname '*subclip*'
```

Both commands returned no files. Formal provenance and decision both state:

```text
segment_cache_path=null
segment_cache_sha256=null
segment_artifact_created=false
segment_objective_allowed=false
```

The formal config supervision contract remains:

```text
only_gold_supervision: parent_video_binary_label
segment_gold_exists: false
segment_gold_used: false
```

## Outcome

Status: **SANITIZER_BUILD_AND_VERIFY_COMPLETED_ARTIFACT_LEVEL_C1_CLOSED**.

The remaining Round3 physical C1 is closed at artifact level: the train-only whole-video feature artifact exists, has the expected hash, has persistent no-clobber lock, is bound by sanitized provenance, and is independently verified by job `12739`.

This is not G0 PASS and not a performance claim. G0 freeze, formal code-audit artifact creation, synthetic, realfold, replay, G0 decision, G1, and all teacher stages remain unrun/locked. The next registered gate is G0 freeze/formal audit preparation under the existing DAG, not G1 or teacher.
