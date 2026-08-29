# LB-SCGP G0 v5 Freeze Execution Record

**Date:** 2026-07-11
**Run ID:** `LBSCGP-G0-FREEZE-v5`
**Stage:** G0 v5 freeze only.
**Config:** `configs/lb_scgp/lb_scgp_v5.json`
**Namespace:** `artifacts/lb_scgp/v5`

## Preflight

- v4 is operationally `FAIL/BLOCKED` because producer
  `_load_freeze_and_audit` raised `NameError: name 'git_state' is not defined`
  after v4 publication.
- v4 formal artifacts remain immutable prior-lineage evidence only, not valid
  downstream unlock evidence.
- v5 repairs only the missing producer import and v5 lineage/path/schema
  support.
- No scientific method, threshold, supervision, data protocol, numerical logic,
  or evaluation behavior is changed.
- The only gold is `parent_video_binary_label`; `segment_gold_exists=false`,
  `segment_gold_used=false`.
- v5 formal review, review-record, audit-publish, synthetic, realfold, replay,
  decision, G1, teacher, MLLM, OCR, validation, test, and performance stages
  are not authorized in this repair.

## Static Shell Checks

Completed before formal freeze submission:

```text
jq empty configs/lb_scgp/lb_scgp_v5.json
bash -n scripts/slurm/lb_scgp_g0_audit_publish.sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch
bash -n refine-logs/lb_scgp/runtime/v5_repair_checks/lb_scgp_v5_schema_regression.sbatch
git diff --check -- <v5 touched code/config/docs>
rg scans for v4 hard-coded producer/publisher/consumer surfaces
```

## Audit-Only Regression

Final focused regression:

```text
job_id=12820
state=COMPLETED
exit=0:0
elapsed=00:00:03
alloc=2 CPU / 4G
result_json=refine-logs/lb_scgp/runtime/v5_repair_checks/regression_result_12820.json
result_sha256=6db954838549e8d2d5e8b4315925136480ac08f9955845858c6c37a2b7e9bff1
log=slurm/logs/lbscgp_v5_schema_regression_12820.out
log_sha256=6d2e76e2a944ed4e1465d5084ec72a8ee48267bd7efe13379ce2c6677047c17d
```

Cases:

- valid v5 strict schema: PASS
- wrong review-record hash: fail-closed
- wrong dirty hash: fail-closed
- wrong code-audit run ID: fail-closed
- wrong review/publication path: fail-closed

Runtime fixture root was removed. `artifacts/lb_scgp/v5` was not created by
regression.

## Exact Freeze Submission

Submitted command:

```text
CONFIG=configs/lb_scgp/lb_scgp_v5.json TASK=freeze RUN_ID=LBSCGP-G0-FREEZE-v5 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch
```

No `--time` is set. This must be the only formal workload in the v5 repair.

## Post-Freeze

```text
job_id=12823
state=COMPLETED
exit=0:0
elapsed=00:00:02
alloc=8 CPU / 64G
start=2026-07-11T20:59:30
end=2026-07-11T20:59:32
```

The job initially entered `PENDING (JobHeldUser)` and was released
automatically. No manual release, requeue, cancel, or `--time` was used.

Outputs:

```text
freeze=artifacts/lb_scgp/v5/CONFIG_FREEZE.json
freeze_sha256=254e45afe9c0355892824c0c26bc73b4b0854cb20c67c3703982762fad010931
payload_sha256=d89f7cd4ad43c8ef83a04b7530ec19186ec1cb0e96958d239f1ce5c2b146bb4d
lock=artifacts/lb_scgp/v5/CONFIG_FREEZE.json.publish.lock
lock_sha256=54dc06d236d5fc1f3ac96400f1a81faeb1d2c0c8e5af075065d1964260de98a9
log=slurm/logs/lbscgp_g0_cpu_12823.out
log_sha256=c43a2b16fc8c95bdfafb5c48c674fa4b778dd9c3d3c3764e24fe50ed038a0526
```

Freeze-bound hashes:

```text
dirty_diff_sha256=1c8284781fb57e90714b390fdbef362e978b70789632b19df3d8161dfe8827b7
config_canonical_sha256=4a45fb6c66884b6b8aa4571961dff3ef7751c2b9f97e2df1584521cfe1eb3dba
implementation_sha256=939acffbafbd9204fc654972cd73f174393c8466c61f4af045b1c20948a6b687
independent_verifier_sha256=f0f49f41de4efee9abf2267b27b75be440f0020583baef565955c3d0c2988b2d
access_ledger_sha256=ce6898035cf25dbe53f4b258a7b792796e4388a282384185f13a84926397ea0f
```

Supervision/counter facts in the freeze:

```text
only_gold_supervision=parent_video_binary_label
segment_gold_exists=false
segment_gold_used=false
G1_G4_locked=true
outer_held_labels_opened=false
outer_held_content_opened=false
protected_storage_read=false
mllm_call_count=0
ocr_call_count=0
teacher_cache_read_count=0
teacher_cache_write_count=0
outer_held_label_read_count=0
outer_held_content_read_count=0
val_content_read_count=0
test_content_read_count=0
val_test_teacher_artifact_count=0
formal_model_optimizer_evaluator_outer_held_read_count=0
```

Explicit non-claims: no v5 review, no v5 review record, no v5 formal PASS,
no v5 audit-publish, no synthetic, no realfold, no replay, no decision, no G1,
no teacher/MLLM/OCR, no held/validation/test access, and no performance result.
