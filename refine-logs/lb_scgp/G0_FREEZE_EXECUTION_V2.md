# LB-SCGP G0 v2 Freeze Execution Record

**Date:** 2026-07-11  
**Run ID:** `LBSCGP-G0-FREEZE-v2`  
**Stage:** G0 v2 freeze only.  
**Config:** `configs/lb_scgp/lb_scgp_v2.json`  
**Namespace:** `artifacts/lb_scgp/v2`

## Preflight

- v2 namespace was absent before submission.
- v1 freeze artifact and lock were preserved unchanged:
  - `artifacts/lb_scgp/v1/CONFIG_FREEZE.json` SHA256 `b6697472b61a61706c694a67b21618d618fcad6e7f59265d8696aee79dc46889`
  - `artifacts/lb_scgp/v1/CONFIG_FREEZE.json.publish.lock` SHA256 `34a05bde46775bcb75384c1c853cef7985f5c05efcdf3afec5fad6d85ef57d8d`
- No segment/subclip artifact was found under `artifacts/lb_scgp`.
- Existing sanitizer records satisfied the dedicated pre-freeze schema and persistent no-clobber locks were present.

## Exact Submission

```text
CONFIG=configs/lb_scgp/lb_scgp_v2.json TASK=freeze RUN_ID=LBSCGP-G0-FREEZE-v2 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch
```

No `--time` was set. No synthetic, realfold, replay, decision, G1, teacher, MLLM, or OCR job was submitted.

## Scheduler Result

```text
JobID|JobName|State|ExitCode|Elapsed|Start|End|AllocCPUS|ReqMem|SubmitLine
12746|lbscgp_g0_cpu|COMPLETED|0:0|00:00:03|2026-07-11T11:34:16|2026-07-11T11:34:19|8|64G|sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch
12746.batch|batch|COMPLETED|0:0|00:00:03|2026-07-11T11:34:16|2026-07-11T11:34:19|8||
```

The job initially appeared as `PENDING (JobHeldUser)`, released automatically, and reached terminal `COMPLETED`. No manual release, cancel, requeue, or broad repair loop was used.

## Artifact Hashes

```text
e95d167af054b91582e1d1f8fbf66fb57a3b5cd67e298883940d385a65ccf563  slurm/logs/lbscgp_g0_cpu_12746.out
4c6f0199a429bbf766cb284b9ecaedd9dcaf38733834dd54574245ab86b633ae  artifacts/lb_scgp/v2/CONFIG_FREEZE.json
22426f693ba3f72d52928b0a86d08fe439d7e7fd43f9c74c61aedb710443e211  artifacts/lb_scgp/v2/CONFIG_FREEZE.json.publish.lock
```

Freeze payload:

```text
bda18d7dfda00ab5808595afec04e5b925d3a4920f55fe963dec5e9d99795c0b  CONFIG_FREEZE payload without payload_sha256
```

Job log terminal JSON:

```json
{"payload_sha256":"bda18d7dfda00ab5808595afec04e5b925d3a4920f55fe963dec5e9d99795c0b","run_id":"LBSCGP-G0-FREEZE-v2","status":"FROZEN"}
```

## Freeze Summary

- `status=FROZEN`
- `stage=G0_FREEZE`
- `lineage_version=v2`
- `slurm_job_id=12746`
- `conda_env=HateVideo`
- `config_canonical_sha256=3c7e391ca8e35fffa76ebcfc76a1c9c6e7c76c92bcc2dc08faa7e9a72be7cb1b`
- `implementation_sha256=51fc1cee40f489e98e82c4aac93799015ac0ad7918e2847c6dbb7e0596890aef`
- `independent_verifier_sha256=8ab99bad45daea1963dd030c24f91c28c87b46631d46e6fcafa4b3e3e102a4f6`
- `sanitized_provenance_sha256=b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007`
- `sanitizer_decision_sha256=172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954`
- `pre_freeze_sanitizer_contract_sha256=cef4c63dd8a9407bd19f14de6e0e81f7816166113889091b670b5eb30ac5e70b`
- `input_files` count: `13`

v2 frozen input paths:

```text
artifacts/lb_scgp/inputs/MHC_zh/fold4/outer_train_features.pt
artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json
artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json
artifacts/ssr/v1/oof/MHC_zh/fold4/checkpoint_epoch28.pt
artifacts/ssr/v1/oof/MHC_zh/fold4/embeddings.npz
artifacts/ssr/v1/oof/MHC_zh/fold4/train.jsonl
configs/lb_scgp/lb_scgp_v2.json
refine-logs/lb_scgp/EXPERIMENT_PLAN.md
refine-logs/lb_scgp/FINAL_PROPOSAL.md
refine-logs/lb_scgp/PROBLEM_ANCHOR.md
refine-logs/lb_scgp/REFINEMENT_REPORT.md
refine-logs/lb_scgp/REVIEW_SUMMARY.md
refine-logs/lb_scgp/v2/PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json
```

Explicitly excluded mutable records:

```text
refine-logs/lb_scgp/EXPERIMENT_TRACKER.md
TARGET_LOOP.md
TARGET_STATE.json
TARGET_FINDINGS.md
TARGET_REVIEW_RAW.md
refine-logs/lb_scgp/G0_V2_REPAIR_HANDOFF.md
refine-logs/lb_scgp/G0_FREEZE_EXECUTION_V2.md
```

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

## Initial Frozen Input Rehash

Every file row and every allowed NPZ member row matched immediately after freeze. The matched file hashes included:

```text
configs/lb_scgp/lb_scgp_v2.json eec778811cfd2cf72a21dbf55af1c768ac6f849234350d958e900f871e41154f
artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007
artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json 172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954
refine-logs/lb_scgp/v2/PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json cef4c63dd8a9407bd19f14de6e0e81f7816166113889091b670b5eb30ac5e70b
artifacts/ssr/v1/oof/MHC_zh/fold4/checkpoint_epoch28.pt c135924c87d1b12218f332ed8b955795cc34d6da972a576c011820572c3d0a39
artifacts/ssr/v1/oof/MHC_zh/fold4/train.jsonl 970b4c55319c83bffc4659976d885ecf89ff9043d659898fa175156a3736a9f6
artifacts/lb_scgp/inputs/MHC_zh/fold4/outer_train_features.pt ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496
refine-logs/lb_scgp/EXPERIMENT_PLAN.md 9eb1f8c69d0a0e1b7c967b658b9a6d11af38b7653348bb38e2b7c2c6b25c2bc7
refine-logs/lb_scgp/PROBLEM_ANCHOR.md 254f4c68fdf578239b952222f4e72378b8a110df21e85158c32b3cbc3b90200d
refine-logs/lb_scgp/FINAL_PROPOSAL.md 94d7b6e9305e8c6095e0a9f20351bb4cafff042f7aa048f2a934ac6d1a3a0a0c
refine-logs/lb_scgp/REVIEW_SUMMARY.md 1eb8cf3c03fc168b42962d923ca3de2978895f5d968263664f984c543487d0df
refine-logs/lb_scgp/REFINEMENT_REPORT.md 4144fa344c385211d25aae697edbf5744de4e2b10f90c2cd4cede0f3b5b56ef7
```

Allowed NPZ members matched:

```text
memory_ids 2db3c268e6efc0ad75ef6a483a7fb97fb992e0ec5b28e8b27dc7963d9ed28193
memory_labels 4666d718b4e2ea1dd66c7f55aa9c69da49dac5f8f64f83f52594656547cc92c5
memory_z af336d46c81de46c6a88863fa95469dead690f0e5da8d6baefb4a0355041e2de
query_ids d12166e5900780c81561996f83baa1c82cc79e4ebc7d0ae8ee12be5c7087aff7
```

## Explicit Non-Claims

- This is not G0 PASS.
- This is not a formal code audit.
- This is not a synthetic, realfold, replay, decision, G1, or teacher result.
- This is not a performance result.
- G1 remains locked.
- Teacher/MLLM/OCR stages remain locked and at zero calls.
- No segment-level gold exists or is assumed.

## Next Gate

The next gate is `LBSCGP-G0-CODE-AUDIT-v2`: a fresh independent formal code audit of the v2 freeze lineage. No later stage is unlocked by this freeze.

## Post-Documentation Frozen Input Proof

After updating `EXPERIMENT_TRACKER.md`, `TARGET_LOOP.md`, `TARGET_STATE.json`, `TARGET_FINDINGS.md`, this handoff, and this execution record, every v2 frozen immutable input was rehashed again with shell tools. All rows still matched.

File rows:

```text
FILE configs/lb_scgp/lb_scgp_v2.json expected=eec778811cfd2cf72a21dbf55af1c768ac6f849234350d958e900f871e41154f actual=eec778811cfd2cf72a21dbf55af1c768ac6f849234350d958e900f871e41154f match=yes
FILE artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json expected=b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007 actual=b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007 match=yes
FILE artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json expected=172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954 actual=172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954 match=yes
FILE refine-logs/lb_scgp/v2/PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json expected=cef4c63dd8a9407bd19f14de6e0e81f7816166113889091b670b5eb30ac5e70b actual=cef4c63dd8a9407bd19f14de6e0e81f7816166113889091b670b5eb30ac5e70b match=yes
FILE artifacts/ssr/v1/oof/MHC_zh/fold4/checkpoint_epoch28.pt expected=c135924c87d1b12218f332ed8b955795cc34d6da972a576c011820572c3d0a39 actual=c135924c87d1b12218f332ed8b955795cc34d6da972a576c011820572c3d0a39 match=yes
FILE artifacts/ssr/v1/oof/MHC_zh/fold4/train.jsonl expected=970b4c55319c83bffc4659976d885ecf89ff9043d659898fa175156a3736a9f6 actual=970b4c55319c83bffc4659976d885ecf89ff9043d659898fa175156a3736a9f6 match=yes
FILE artifacts/lb_scgp/inputs/MHC_zh/fold4/outer_train_features.pt expected=ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496 actual=ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496 match=yes
FILE refine-logs/lb_scgp/EXPERIMENT_PLAN.md expected=9eb1f8c69d0a0e1b7c967b658b9a6d11af38b7653348bb38e2b7c2c6b25c2bc7 actual=9eb1f8c69d0a0e1b7c967b658b9a6d11af38b7653348bb38e2b7c2c6b25c2bc7 match=yes
FILE refine-logs/lb_scgp/PROBLEM_ANCHOR.md expected=254f4c68fdf578239b952222f4e72378b8a110df21e85158c32b3cbc3b90200d actual=254f4c68fdf578239b952222f4e72378b8a110df21e85158c32b3cbc3b90200d match=yes
FILE refine-logs/lb_scgp/FINAL_PROPOSAL.md expected=94d7b6e9305e8c6095e0a9f20351bb4cafff042f7aa048f2a934ac6d1a3a0a0c actual=94d7b6e9305e8c6095e0a9f20351bb4cafff042f7aa048f2a934ac6d1a3a0a0c match=yes
FILE refine-logs/lb_scgp/REVIEW_SUMMARY.md expected=1eb8cf3c03fc168b42962d923ca3de2978895f5d968263664f984c543487d0df actual=1eb8cf3c03fc168b42962d923ca3de2978895f5d968263664f984c543487d0df match=yes
FILE refine-logs/lb_scgp/REFINEMENT_REPORT.md expected=4144fa344c385211d25aae697edbf5744de4e2b10f90c2cd4cede0f3b5b56ef7 actual=4144fa344c385211d25aae697edbf5744de4e2b10f90c2cd4cede0f3b5b56ef7 match=yes
```

Allowed NPZ member rows:

```text
MEMBER artifacts/ssr/v1/oof/MHC_zh/fold4/embeddings.npz:memory_ids expected=2db3c268e6efc0ad75ef6a483a7fb97fb992e0ec5b28e8b27dc7963d9ed28193 actual=2db3c268e6efc0ad75ef6a483a7fb97fb992e0ec5b28e8b27dc7963d9ed28193 match=yes
MEMBER artifacts/ssr/v1/oof/MHC_zh/fold4/embeddings.npz:memory_labels expected=4666d718b4e2ea1dd66c7f55aa9c69da49dac5f8f64f83f52594656547cc92c5 actual=4666d718b4e2ea1dd66c7f55aa9c69da49dac5f8f64f83f52594656547cc92c5 match=yes
MEMBER artifacts/ssr/v1/oof/MHC_zh/fold4/embeddings.npz:memory_z expected=af336d46c81de46c6a88863fa95469dead690f0e5da8d6baefb4a0355041e2de actual=af336d46c81de46c6a88863fa95469dead690f0e5da8d6baefb4a0355041e2de match=yes
MEMBER artifacts/ssr/v1/oof/MHC_zh/fold4/embeddings.npz:query_ids expected=d12166e5900780c81561996f83baa1c82cc79e4ebc7d0ae8ee12be5c7087aff7 actual=d12166e5900780c81561996f83baa1c82cc79e4ebc7d0ae8ee12be5c7087aff7 match=yes
```

Excluded mutable records:

```text
excluded_from_input_files refine-logs/lb_scgp/EXPERIMENT_TRACKER.md
excluded_from_input_files TARGET_LOOP.md
excluded_from_input_files TARGET_STATE.json
excluded_from_input_files TARGET_FINDINGS.md
excluded_from_input_files TARGET_REVIEW_RAW.md
excluded_from_input_files refine-logs/lb_scgp/G0_V2_REPAIR_HANDOFF.md
excluded_from_input_files refine-logs/lb_scgp/G0_FREEZE_EXECUTION_V2.md
```

This proves the mandatory post-freeze documentation updates did not change any v2 frozen immutable input.
