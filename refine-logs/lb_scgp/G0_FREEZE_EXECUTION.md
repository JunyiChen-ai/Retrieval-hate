# LB-SCGP G0 Freeze Execution Record

**Date:** 2026-07-11  
**Worker scope:** delegated execution worker only; no subagent spawned.  
**Run ID:** `LBSCGP-G0-FREEZE-v1`  
**Stage:** G0 freeze only.

## Preflight Eligibility

- Registered next action in `refine-logs/lb_scgp/EXPERIMENT_TRACKER.md`: `LBSCGP-G0-FREEZE-v1` with status `NEXT_AUTHORIZED_NOT_RUN`.
- Physical sanitizer review: `refine-logs/lb_scgp/G0_SANITIZER_PHYSICAL_REVIEW.md`, 0 Critical / 0 High, C1 closed at artifact level only.
- Sanitizer decision: `PASS`, all gates true, `memory_id_count=464`, `query_id_sentinel_count=115`.
- Formal namespace preflight: `artifacts/lb_scgp/v1` was absent before submission.
- No segment/subclip artifacts were found under `artifacts/lb_scgp`.
- Persistent sanitizer publish locks were present for `outer_train_features.pt`, `sanitized_provenance.json`, `sanitizer_decision.json`, and `sanitizer_manifest.json`.

## Exact Submission

```text
TASK=freeze RUN_ID=LBSCGP-G0-FREEZE-v1 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch
```

No `--time` was set. No synthetic, realfold, replay, decision, G1, teacher, MLLM, or OCR job was submitted.

## Scheduler Result

```text
JobID|JobName|State|ExitCode|Elapsed|Start|End|AllocCPUS|ReqMem|SubmitLine
12742|lbscgp_g0_cpu|COMPLETED|0:0|00:00:03|2026-07-11T11:07:43|2026-07-11T11:07:46|8|64G|sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch
12742.batch|batch|COMPLETED|0:0|00:00:03|2026-07-11T11:07:43|2026-07-11T11:07:46|8||
```

The job reached a terminal `COMPLETED` state. No manual release, cancel, or requeue was used.

## Log And Artifact Hashes

```text
70cd194ed6a811be6956644e832055907f94750ccec6876352a8c1d6b5e98628  slurm/logs/lbscgp_g0_cpu_12742.out
b6697472b61a61706c694a67b21618d618fcad6e7f59265d8696aee79dc46889  artifacts/lb_scgp/v1/CONFIG_FREEZE.json
34a05bde46775bcb75384c1c853cef7985f5c05efcdf3afec5fad6d85ef57d8d  artifacts/lb_scgp/v1/CONFIG_FREEZE.json.publish.lock
```

Freeze payload:

```text
b3b33090b39b3b975c2cf213aab669041b345c6ef3a3f7c200366a506bcebfd5  CONFIG_FREEZE payload without payload_sha256
```

Job log terminal JSON:

```json
{"payload_sha256":"b3b33090b39b3b975c2cf213aab669041b345c6ef3a3f7c200366a506bcebfd5","run_id":"LBSCGP-G0-FREEZE-v1","status":"FROZEN"}
```

## Freeze Artifact Summary

- `status=FROZEN`
- `stage=G0_FREEZE`
- `slurm_job_id=12742`
- `conda_env=HateVideo`
- `config_canonical_sha256=2c990e9d63fd3048e0836db59e4103af701da5eb175ca78424e078604c93918d`
- `implementation_sha256=29cdc2bcf514f88fcde91a0e9bfdd5352f11cf2367e47c8608705a0493ce5df8`
- `independent_verifier_sha256=7acdf3f9e24e6c42a0cae2bfd66986927723099ef73f1afec8904b5ad1a01ec9`
- `sanitized_provenance_sha256=b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007`
- `sanitizer_decision_sha256=172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954`
- `fold_ids_sha256=2c436c0ff1c441392ec6804f8168d2542f61ef30fcfe0e4206563ce2f497d95d`
- `checkpoint_sha256=c135924c87d1b12218f332ed8b955795cc34d6da972a576c011820572c3d0a39`
- `input_files` count: 15

## Supervision And Call Contract

- Only gold supervision: `parent_video_binary_label`.
- `segment_gold_exists=false`
- `segment_gold_used=false`
- No segment/subclip artifact exists under `artifacts/lb_scgp`.
- `mllm_call_count=0`
- `ocr_call_count=0`
- `teacher_cache_read_count=0`
- `teacher_cache_write_count=0`
- `outer_held_label_read_count=0`
- `outer_held_content_read_count=0`
- `val_content_read_count=0`
- `test_content_read_count=0`
- `val_test_teacher_artifact_count=0`
- `formal_model_optimizer_evaluator_outer_held_read_count=0`
- Outer-held IDs are exclusion sentinels only; held labels/content were not opened.

## Explicit Non-Claims

- This is not G0 PASS.
- This is not a formal code audit.
- This is not a synthetic, realfold, replay, decision, G1, or teacher result.
- This is not a performance result.
- G1 remains locked.
- Teacher/MLLM/OCR stages remain locked and at zero calls.
- No segment-level gold exists or is assumed.

## Post-Freeze Audit Note

`CONFIG_FREEZE.json` includes hashes for mutable intermediate records, including `refine-logs/lb_scgp/EXPERIMENT_TRACKER.md`, `TARGET_LOOP.md`, and `TARGET_STATE.json`, as they existed before the post-run audit-trail update required by this task. Later numerical code rehashes `input_files` during predecessor verification, so the independent formal code audit must explicitly assess whether these mutable record hashes create a frozen-input drift hazard before any synthetic/realfold execution. This worker does not patch or self-certify that issue.

Post-update comparison:

```text
refine-logs/lb_scgp/EXPERIMENT_TRACKER.md frozen=6f89fd78880219466ee88022ee3e6f44d264c981738d8dc6ea906d17725d0a42 current=7dc61732427b7cff8fc5915803092165cba93a895009302e8e4afd57b97ed1f9 match=no
TARGET_LOOP.md frozen=b0387f49c7dab9686da2baa10ace3c60cbd27e0dc422810974baca652dd49a63 current=116498365eec75c68d181e71a637381aa881c5402b528fb45c1efe7b56cd5c8c match=no
TARGET_STATE.json frozen=a86f5ae999eb0bf299c56f69d7c8254092b689d0e6a3cbd7bf8cc19197282ce1 current=b04ee698e53b9044f47a961973431b739b846d719cef0f5c6a728a43ca310641 match=no
```

## Next Gate

The next registered gate is `LBSCGP-G0-CODE-AUDIT-v1`: an independent formal implementation audit artifact with 0 Critical / 0 High. No later stage is unlocked by this freeze.
