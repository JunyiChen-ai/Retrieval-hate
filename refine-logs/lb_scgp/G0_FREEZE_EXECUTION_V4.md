# LB-SCGP G0 v4 Freeze Execution Record

**Date:** 2026-07-11
**Run ID:** `LBSCGP-G0-FREEZE-v4`
**Stage:** G0 v4 freeze only.
**Config:** `configs/lb_scgp/lb_scgp_v4.json`
**Namespace:** `artifacts/lb_scgp/v4`

## Preflight

- v4 namespace must be absent before submission.
- v1/v2/v3 freeze artifacts and locks are preserved unchanged.
- v4 repairs only the tooling-lineage gap identified in `G0_FORMAL_CODE_AUDIT_REVIEW_V3.md`: no authorized formal code-audit PASS producer.
- v4 does not change scientific method, numerical thresholds, fixtures, or performance logic.
- v4 binds an independent-verifier `audit-publish` path, exact formal artifact schema, strict machine review record, and downstream consumers.
- No segment/subclip artifact may be used. Parent-video binary labels are the only gold.
- No formal audit PASS, v4 review, or strict review record is created by this repair executor.

## Exact Freeze Submission

```text
CONFIG=configs/lb_scgp/lb_scgp_v4.json TASK=freeze RUN_ID=LBSCGP-G0-FREEZE-v4 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch
```

No `--time` is set. This is the only new SLURM workload authorized for this repair.

## Scheduler Result

```text
JobID|JobName|State|ExitCode|Elapsed|Start|End|AllocCPUS|ReqMem|SubmitLine
12759|lbscgp_g0_cpu|COMPLETED|0:0|00:00:03|2026-07-11T12:48:12|2026-07-11T12:48:15|8|64G|sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch
12759.batch|batch|COMPLETED|0:0|00:00:03|2026-07-11T12:48:12|2026-07-11T12:48:15|8||
```

The job initially entered `PENDING (JobHeldUser)` and was allowed to release automatically. No manual release, cancel, requeue, or repair loop was used.

## Artifact Hashes

```text
6203cab3eded38f22638980c4828020a10ddbd8421819a0d5f5059eca6faa6da  slurm/logs/lbscgp_g0_cpu_12759.out
dcf65eceba04e7c4f08145b2012653705f7347c6e96ebc8b2b769280dff48fd0  artifacts/lb_scgp/v4/CONFIG_FREEZE.json
09003ce9e741d7c0310045f854479deb8fecff74bfddd33f6b9d80dc6df9572a  artifacts/lb_scgp/v4/CONFIG_FREEZE.json.publish.lock
59804c09f63f923a67eb276325ae6be9ce124fd9a3aceb64c46c3809ffdd85b0  configs/lb_scgp/lb_scgp_v4.json
```

Job log terminal JSON:

```json
{"payload_sha256":"92301ad95870b8a1af41e7e69e45054a2e63fe80a021cf4c1c22906aea0872bf","run_id":"LBSCGP-G0-FREEZE-v4","status":"FROZEN"}
```

## Freeze Summary

```text
status=FROZEN
stage=G0_FREEZE
lineage_version=v4
slurm_job_id=12759
conda_env=HateVideo
git_head=a1b1922bc970bb831526b4d21c911380ec871248
dirty_diff_sha256=8ca10aec315f800959e7869beb200f4bbc5f5d27841d8c307d896f1644803e7a
config_canonical_sha256=9e99cba37486e2511b0e37fb7d2c3b59053fbac8aca577ba05b36c138aa67c56
implementation_sha256=c7e9371494f991d88a7ab93cc64769fa1e6a92913df3afd2f647201d0eef1bf1
independent_verifier_sha256=03a78a89867d3cea468b5319463ccabcefa4b4a589a61863bffd3e14c9df5402
access_ledger_sha256=ef67ad3b6521a9b8e9b73dd27260917c531e4ec72a84e04c52afed0c34ba72a7
payload_sha256=92301ad95870b8a1af41e7e69e45054a2e63fe80a021cf4c1c22906aea0872bf
sanitized_provenance_sha256=b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007
sanitizer_decision_sha256=172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954
pre_freeze_sanitizer_contract_sha256=cef4c63dd8a9407bd19f14de6e0e81f7816166113889091b670b5eb30ac5e70b
input_files_count=13
```

Implementation file hashes at freeze:

```text
f1c95add65d59c6bc692682b4a91daf8f5912a9deb6a19998b412b2e404282bb  scripts/analysis/lb_scgp_common.py
215dfca8d13340a13da8cca505dbd9701c2218a6a9af57f768495b626ca5b4f8  scripts/analysis/lb_scgp_sanitize_inputs.py
6cd649d60cd0555aa34e94561a45e3147408354cd8d0c0798d2a9c8ea283351e  scripts/analysis/lb_scgp_verify_sanitizer.py
f73005dcb8057ef417abbad8b79703d6b53c1f6a85cbc01d382cd7439c655f4a  scripts/analysis/lb_scgp_g0.py
2fba75be6ad341e18114631deeed6612f7bfda032f2f46fce87186d8a4d7938b  scripts/analysis/lb_scgp_real_replay.py
03a78a89867d3cea468b5319463ccabcefa4b4a589a61863bffd3e14c9df5402  scripts/analysis/lb_scgp_independent_verify.py
350b9b946470c9afbe4fcbbfb4b742d9f24a984e2c983089276d2ec5f48fff17  scripts/slurm/lb_scgp_sanitize_inputs.sbatch
8722f7d71070b9711a45aa4074c128b516e08e53c556940660ab1ac980229241  scripts/slurm/lb_scgp_g0_cpu.sbatch
055d47a006c1e22d4f5d2d4cf1c0e739cec7950d134d01b39ad435149fe839f4  scripts/slurm/lb_scgp_g0_gpu.sbatch
2a5eeff5238ecb6c303f927cfde11afa96a2ff919b8582332847655e1c1317c1  scripts/slurm/lb_scgp_g0_audit_publish.sbatch
```

## Static Checks Before Freeze

All checks were shell-only; no local Python/import/data/model execution was used.

```text
jq empty configs/lb_scgp/lb_scgp_v1.json configs/lb_scgp/lb_scgp_v2.json configs/lb_scgp/lb_scgp_v3.json configs/lb_scgp/lb_scgp_v4.json TARGET_STATE.json
bash -n scripts/slurm/lb_scgp_g0_cpu.sbatch scripts/slurm/lb_scgp_g0_gpu.sbatch scripts/slurm/lb_scgp_g0_audit_publish.sbatch scripts/slurm/lb_scgp_sanitize_inputs.sbatch
git diff --check -- <LB-SCGP v4 touched code/config/docs>
rg source scan for strict schema identifiers in producer and verifier
find artifacts/lb_scgp/v4 -maxdepth 5 -print
```

`artifacts/lb_scgp/v4` was absent before submission.

## Supervision And Forbidden Access Contract

Binding expected values after freeze:

```text
only_gold_supervision=parent_video_binary_label
segment_gold_exists=false
segment_gold_used=false
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

## Post-Freeze Proofs

After updating `EXPERIMENT_TRACKER.md`, `TARGET_LOOP.md`, `TARGET_STATE.json`, `TARGET_FINDINGS.md`, `G0_V4_AUDIT_PUBLISHER_DESIGN.md`, `G0_V4_TOOLING_REPAIR_HANDOFF.md`, and this execution record, the v4 frozen inputs and dirty state were rechecked by shell only.

Frozen file inputs all matched:

```text
FILE configs/lb_scgp/lb_scgp_v4.json 59804c09f63f923a67eb276325ae6be9ce124fd9a3aceb64c46c3809ffdd85b0
FILE artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007
FILE artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json 172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954
FILE refine-logs/lb_scgp/v2/PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json cef4c63dd8a9407bd19f14de6e0e81f7816166113889091b670b5eb30ac5e70b
FILE artifacts/ssr/v1/oof/MHC_zh/fold4/checkpoint_epoch28.pt c135924c87d1b12218f332ed8b955795cc34d6da972a576c011820572c3d0a39
FILE artifacts/ssr/v1/oof/MHC_zh/fold4/train.jsonl 970b4c55319c83bffc4659976d885ecf89ff9043d659898fa175156a3736a9f6
FILE artifacts/lb_scgp/inputs/MHC_zh/fold4/outer_train_features.pt ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496
FILE refine-logs/lb_scgp/EXPERIMENT_PLAN.md 9eb1f8c69d0a0e1b7c967b658b9a6d11af38b7653348bb38e2b7c2c6b25c2bc7
FILE refine-logs/lb_scgp/PROBLEM_ANCHOR.md 254f4c68fdf578239b952222f4e72378b8a110df21e85158c32b3cbc3b90200d
FILE refine-logs/lb_scgp/FINAL_PROPOSAL.md 94d7b6e9305e8c6095e0a9f20351bb4cafff042f7aa048f2a934ac6d1a3a0a0c
FILE refine-logs/lb_scgp/REVIEW_SUMMARY.md 1eb8cf3c03fc168b42962d923ca3de2978895f5d968263664f984c543487d0df
FILE refine-logs/lb_scgp/REFINEMENT_REPORT.md 4144fa344c385211d25aae697edbf5744de4e2b10f90c2cd4cede0f3b5b56ef7
```

Allowed NPZ members were streamed by exact member name only:

```text
MEMBER artifacts/ssr/v1/oof/MHC_zh/fold4/embeddings.npz:memory_ids 2db3c268e6efc0ad75ef6a483a7fb97fb992e0ec5b28e8b27dc7963d9ed28193
MEMBER artifacts/ssr/v1/oof/MHC_zh/fold4/embeddings.npz:memory_labels 4666d718b4e2ea1dd66c7f55aa9c69da49dac5f8f64f83f52594656547cc92c5
MEMBER artifacts/ssr/v1/oof/MHC_zh/fold4/embeddings.npz:memory_z af336d46c81de46c6a88863fa95469dead690f0e5da8d6baefb4a0355041e2de
MEMBER artifacts/ssr/v1/oof/MHC_zh/fold4/embeddings.npz:query_ids d12166e5900780c81561996f83baa1c82cc79e4ebc7d0ae8ee12be5c7087aff7
```

The central directory lists `query_z.npy` and `query_labels.npy`, but those members were not opened or hashed.

Dirty-state shell reproduction:

```text
frozen_dirty_sha256=8ca10aec315f800959e7869beb200f4bbc5f5d27841d8c307d896f1644803e7a
current_dirty_sha256=8ca10aec315f800959e7869beb200f4bbc5f5d27841d8c307d896f1644803e7a
match=yes
```

Future review/report sidecar and formal namespace dry-run proof:

```text
refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.md -> exact_dirty_path_excluded
refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.record.json -> exact_dirty_path_excluded
artifacts/lb_scgp/v4/g0/code_audit/review.md -> formal_artifact_prefix_excluded
artifacts/lb_scgp/v4/g0/code_audit/review_record.json -> formal_artifact_prefix_excluded
artifacts/lb_scgp/v4/g0/code_audit/audit.json -> formal_artifact_prefix_excluded
artifacts/lb_scgp/v4/g0/code_audit/publication_index.json -> formal_artifact_prefix_excluded
artifacts/lb_scgp/v4/g0/code_audit/audit.json.publish.lock -> formal_artifact_prefix_excluded
simulated_dirty_hash_after_hypothetical_review_record_and_formal_outputs=8ca10aec315f800959e7869beb200f4bbc5f5d27841d8c307d896f1644803e7a
```

This was an exact shell simulation only. No fake review file, fake review-record file, or formal code-audit namespace was created. After the proof:

```text
ABSENT refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.md
ABSENT refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.record.json
ABSENT artifacts/lb_scgp/v4/g0/code_audit
```

`artifacts/lb_scgp/v4` contains only:

```text
artifacts/lb_scgp/v4/CONFIG_FREEZE.json
artifacts/lb_scgp/v4/CONFIG_FREEZE.json.publish.lock
```

## Explicit Non-Claims

- This is not G0 PASS.
- This is not a formal code audit.
- This is not a formal code-audit PASS artifact.
- This is not a synthetic, realfold, replay, decision, G1, teacher, MLLM, OCR, or performance result.
- G1 and teacher remain locked.
