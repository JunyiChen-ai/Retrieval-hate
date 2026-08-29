# LB-SCGP G0 v3 Freeze Execution Record

**Date:** 2026-07-11  
**Run ID:** `LBSCGP-G0-FREEZE-v3`  
**Stage:** G0 v3 freeze only.  
**Config:** `configs/lb_scgp/lb_scgp_v3.json`  
**Namespace:** `artifacts/lb_scgp/v3`

## Preflight

- v3 namespace was absent before submission.
- v1/v2 freeze artifacts and locks were preserved unchanged.
- v2 formal code audit failed only on C1: mandatory post-freeze review records were not excluded from dirty-state predecessor checks.
- v3 config explicitly binds exact dirty-state exclusions for the mandatory mutable audit trail and exact formal artifact namespace exclusions for `v1`, `v2`, and `v3`.
- No segment/subclip artifact was found under `artifacts/lb_scgp`.
- Existing sanitizer records and no-clobber locks were reused as immutable inputs; sanitizer was not rerun.

## Exact Submission

```text
CONFIG=configs/lb_scgp/lb_scgp_v3.json TASK=freeze RUN_ID=LBSCGP-G0-FREEZE-v3 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch
```

No `--time` was set. No synthetic, realfold, replay, decision, G1, teacher, MLLM, OCR, or cache task was submitted.

## Scheduler Result

```text
JobID|JobName|State|ExitCode|Elapsed|Start|End|AllocCPUS|ReqMem|SubmitLine
12748|lbscgp_g0_cpu|COMPLETED|0:0|00:00:03|2026-07-11T12:00:07|2026-07-11T12:00:10|8|64G|sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch
12748.batch|batch|COMPLETED|0:0|00:00:03|2026-07-11T12:00:07|2026-07-11T12:00:10|8||
```

The job reached terminal `COMPLETED`. No manual release, cancel, requeue, or repair loop was used.

## Artifact Hashes

```text
f60d2301e7460ab91b25f6c323c578e49a00001b46c67a590f9f3c3d58abf545  slurm/logs/lbscgp_g0_cpu_12748.out
9fba7f1649dd67d4bb0fcc193e555d8246d7a4966307732e87d5e9fca7346dd9  artifacts/lb_scgp/v3/CONFIG_FREEZE.json
9c32d07c524e466ad06c06fc2a472829764cd1facff22390eac8b2879d329b8f  artifacts/lb_scgp/v3/CONFIG_FREEZE.json.publish.lock
```

Job log terminal JSON:

```json
{"payload_sha256":"352ec2215e2225b1768a13f39f96ef935b91966606d8db874d6de4410b1a9f3d","run_id":"LBSCGP-G0-FREEZE-v3","status":"FROZEN"}
```

## Freeze Summary

- `status=FROZEN`
- `stage=G0_FREEZE`
- `lineage_version=v3`
- `slurm_job_id=12748`
- `conda_env=HateVideo`
- `git_head=a1b1922bc970bb831526b4d21c911380ec871248`
- `dirty_diff_sha256=91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1`
- `config_canonical_sha256=84227b68eaa496da6e307ce5c5ef3469e1b7c68e350f0d62d1677d01f07645bf`
- `implementation_sha256=b8759436a6c5e2a67bf7125cbd1ab57cb05187e764e837373abfdf1a92916e75`
- `independent_verifier_sha256=d1e50057b4c166a71426f89474b6526e3eab11da5547e15368112b6620dbf5ce`
- `access_ledger_sha256=3db4b94900a9d9b807ab495be869a5ef87a3894f987eef03ea1e948030abdc72`
- `payload_sha256=352ec2215e2225b1768a13f39f96ef935b91966606d8db874d6de4410b1a9f3d`
- `sanitized_provenance_sha256=b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007`
- `sanitizer_decision_sha256=172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954`
- `pre_freeze_sanitizer_contract_sha256=cef4c63dd8a9407bd19f14de6e0e81f7816166113889091b670b5eb30ac5e70b`
- `input_files` count: `13`

v3 formal artifact exclusions:

```text
artifacts/lb_scgp/v1/
artifacts/lb_scgp/v2/
artifacts/lb_scgp/v3/
```

v3 dirty-state excluded exact paths:

```text
refine-logs/lb_scgp/EXPERIMENT_TRACKER.md
TARGET_LOOP.md
TARGET_STATE.json
TARGET_FINDINGS.md
TARGET_REVIEW_RAW.md
refine-logs/lb_scgp/G0_V3_REPAIR_HANDOFF.md
refine-logs/lb_scgp/G0_FREEZE_EXECUTION_V3.md
refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V3.md
```

v3 dirty-state excluded prefix:

```text
refine-logs/lb_scgp/runtime/
```

## Frozen Inputs

```text
FILE configs/lb_scgp/lb_scgp_v3.json a480c9b9bf56c938667b4f8e2f3d07882b84843627233b613d864764c02eaf47
FILE artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007
FILE artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json 172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954
FILE refine-logs/lb_scgp/v2/PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json cef4c63dd8a9407bd19f14de6e0e81f7816166113889091b670b5eb30ac5e70b
FILE artifacts/ssr/v1/oof/MHC_zh/fold4/checkpoint_epoch28.pt c135924c87d1b12218f332ed8b955795cc34d6da972a576c011820572c3d0a39
FILE artifacts/ssr/v1/oof/MHC_zh/fold4/train.jsonl 970b4c55319c83bffc4659976d885ecf89ff9043d659898fa175156a3736a9f6
MEMBER artifacts/ssr/v1/oof/MHC_zh/fold4/embeddings.npz memory_ids 2db3c268e6efc0ad75ef6a483a7fb97fb992e0ec5b28e8b27dc7963d9ed28193
MEMBER artifacts/ssr/v1/oof/MHC_zh/fold4/embeddings.npz memory_labels 4666d718b4e2ea1dd66c7f55aa9c69da49dac5f8f64f83f52594656547cc92c5
MEMBER artifacts/ssr/v1/oof/MHC_zh/fold4/embeddings.npz memory_z af336d46c81de46c6a88863fa95469dead690f0e5da8d6baefb4a0355041e2de
MEMBER artifacts/ssr/v1/oof/MHC_zh/fold4/embeddings.npz query_ids d12166e5900780c81561996f83baa1c82cc79e4ebc7d0ae8ee12be5c7087aff7
FILE artifacts/lb_scgp/inputs/MHC_zh/fold4/outer_train_features.pt ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496
FILE refine-logs/lb_scgp/EXPERIMENT_PLAN.md 9eb1f8c69d0a0e1b7c967b658b9a6d11af38b7653348bb38e2b7c2c6b25c2bc7
FILE refine-logs/lb_scgp/PROBLEM_ANCHOR.md 254f4c68fdf578239b952222f4e72378b8a110df21e85158c32b3cbc3b90200d
FILE refine-logs/lb_scgp/FINAL_PROPOSAL.md 94d7b6e9305e8c6095e0a9f20351bb4cafff042f7aa048f2a934ac6d1a3a0a0c
FILE refine-logs/lb_scgp/REVIEW_SUMMARY.md 1eb8cf3c03fc168b42962d923ca3de2978895f5d968263664f984c543487d0df
FILE refine-logs/lb_scgp/REFINEMENT_REPORT.md 4144fa344c385211d25aae697edbf5744de4e2b10f90c2cd4cede0f3b5b56ef7
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

## Explicit Non-Claims

- This is not G0 PASS.
- This is not a formal code audit.
- This is not a synthetic, realfold, replay, decision, G1, or teacher result.
- This is not a performance result.
- G1 remains locked.
- Teacher/MLLM/OCR stages remain locked and at zero calls.
- No segment-level gold exists or is assumed.

## Next Gate

The next gate is `LBSCGP-G0-CODE-AUDIT-v3`: a fresh independent formal code audit of the v3 freeze lineage. No later stage is unlocked by this freeze.

## Post-Documentation Frozen Input And Dirty-Hash Proof

After updating `EXPERIMENT_TRACKER.md`, `TARGET_LOOP.md`, `TARGET_STATE.json`, `TARGET_FINDINGS.md`, `G0_V3_REPAIR_HANDOFF.md`, and this execution record, the v3 dirty hash was recomputed with the same v3 policy used by producer/common and the independent verifier.

```text
current_dirty_sha256=91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1
frozen_dirty_sha256=91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1
match=yes
```

Every v3 frozen immutable input was rehashed after documentation updates. All rows still matched:

```text
FILE configs/lb_scgp/lb_scgp_v3.json match=yes
FILE artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json match=yes
FILE artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json match=yes
FILE refine-logs/lb_scgp/v2/PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json match=yes
FILE artifacts/ssr/v1/oof/MHC_zh/fold4/checkpoint_epoch28.pt match=yes
FILE artifacts/ssr/v1/oof/MHC_zh/fold4/train.jsonl match=yes
FILE artifacts/lb_scgp/inputs/MHC_zh/fold4/outer_train_features.pt match=yes
FILE refine-logs/lb_scgp/EXPERIMENT_PLAN.md match=yes
FILE refine-logs/lb_scgp/PROBLEM_ANCHOR.md match=yes
FILE refine-logs/lb_scgp/FINAL_PROPOSAL.md match=yes
FILE refine-logs/lb_scgp/REVIEW_SUMMARY.md match=yes
FILE refine-logs/lb_scgp/REFINEMENT_REPORT.md match=yes
MEMBER artifacts/ssr/v1/oof/MHC_zh/fold4/embeddings.npz:memory_ids match=yes
MEMBER artifacts/ssr/v1/oof/MHC_zh/fold4/embeddings.npz:memory_labels match=yes
MEMBER artifacts/ssr/v1/oof/MHC_zh/fold4/embeddings.npz:memory_z match=yes
MEMBER artifacts/ssr/v1/oof/MHC_zh/fold4/embeddings.npz:query_ids match=yes
```

The mandatory future review path was then tested with a temporary controlled file. The file was created, the dirty hash was recomputed, its content was updated, the dirty hash was recomputed again, and the file was deleted. The path is excluded exactly in v3 policy, so the frozen dirty hash remained stable:

```text
dirty_before_review_probe=91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1
dirty_after_review_create=91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1
dirty_after_review_update=91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1
dirty_after_review_delete_restore=91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1
review_probe_left=no
```

No `G0_FORMAL_CODE_AUDIT_REVIEW_V3.md` file was left behind by this proof, and no formal PASS/code-audit PASS artifact was created.
