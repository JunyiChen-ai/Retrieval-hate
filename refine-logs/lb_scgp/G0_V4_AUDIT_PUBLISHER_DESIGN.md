# LB-SCGP G0 v4 Audit Publisher Design

**Date:** 2026-07-11
**Scope:** minimal tooling-lineage repair for the v3 gap: no authorized formal code-audit PASS producer.

## Trust Boundary

The repair executor does not create a v4 independent review report, review sidecar, or formal PASS artifact.

A later fresh independent auditor is trusted to supply only:

- `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.md`
- `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.record.json`

The free-form report is not parsed as the trust root. The strict machine record must bind the report SHA256 and assert the reviewer/process identity, scope, 0 Critical, 0 High, no-segment-gold PASS, and formal PASS authorization. The publisher independently recomputes all repository facts before publishing.

Required review-record schema:

```text
schema_version=1
record_type=LB_SCGP_G0_CODE_AUDIT_INDEPENDENT_REVIEW_RECORD_V4
run_id=LBSCGP-G0-CODE-AUDIT-v4
stage=G0_CODE_AUDIT
status=PASS
lineage_version=v4
config_path=configs/lb_scgp/lb_scgp_v4.json
artifact_namespace=artifacts/lb_scgp/v4
freeze_run_id=LBSCGP-G0-FREEZE-v4
freeze_path=artifacts/lb_scgp/v4/CONFIG_FREEZE.json
freeze_file_sha256=<actual v4 freeze file hash>
freeze_payload_sha256=<actual v4 freeze payload hash>
config_canonical_sha256=<actual v4 canonical config hash>
implementation_sha256=<actual v4 implementation hash>
independent_verifier_sha256=<actual verifier hash>
review_report_path=refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.md
review_report_sha256=<actual report file hash>
reviewer_identity=<nonempty independent reviewer id>
review_process_identity=fresh_independent_gpt_5_5_xhigh
review_scope=LB-SCGP G0 v4 formal code audit for LBSCGP-G0-FREEZE-v4
critical=0
high=0
important=<integer >= 0>
no_segment_gold_pass=true
formal_pass_authorized=true
independent_reviewer=true
repair_executor_created=false
only_gold_supervision=parent_video_binary_label
segment_gold_exists=false
segment_gold_used=false
payload_sha256=<canonical JSON hash excluding payload_sha256>
```

The publisher rejects missing or additional fields.

## Publisher

Task: `audit-publish` in `scripts/analysis/lb_scgp_independent_verify.py`, invoked only through:

```text
CONFIG=configs/lb_scgp/lb_scgp_v4.json TASK=audit-publish RUN_ID=LBSCGP-G0-CODE-AUDIT-v4 REVIEW=refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.md REVIEW_RECORD=refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.record.json sbatch scripts/slurm/lb_scgp_g0_audit_publish.sbatch
```

Wrapper properties:

- CPU-only: 2 CPU / 4G.
- No `--time`.
- `conda activate HateVideo`.
- Offline environment flags.
- Strict `TASK`, `RUN_ID`, `CONFIG`, `REVIEW`, and `REVIEW_RECORD` checks before Python.
- Shell-quoted arguments.

## Recomputed Gates

The publisher recomputes or validates:

- exact v4 config path, namespace, run IDs, freeze identity, freeze payload hash, freeze file hash, freeze lock hash;
- canonical config hash, implementation hash, independent verifier hash, current git head and dirty hash equality with frozen dirty hash;
- all frozen input file hashes and allowed NPZ member hashes using only `memory_ids`, `memory_labels`, `memory_z`, and `query_ids`;
- forbidden `query_z` and `query_labels` are not opened;
- v1-v3 freeze and lock hashes match the values embedded in v4 config;
- v4 `g0/code_audit` namespace is absent before publish;
- only parent-video binary gold, `segment_gold_exists=false`, `segment_gold_used=false`;
- all MLLM/OCR/teacher/cache/held-label/held-content/val/test counters are zero;
- exact v4 dirty-policy lists and narrow future review/report sidecar exclusions;
- audit-publish wrapper contract and run-ID authorization;
- strict review-record schema and report hash binding.

If any check fails before commit, the task exits nonzero without formal PASS files or publish locks.

## Publication Transaction

After all checks pass, the publisher builds a temporary transaction directory under `artifacts/lb_scgp/v4/g0/`, writes all formal files and persistent no-clobber locks inside it, fsyncs, then atomically renames the directory to:

```text
artifacts/lb_scgp/v4/g0/code_audit/
```

Published files:

- `review.md`
- `review_record.json`
- `audit.json`
- `publication_index.json`
- one persistent `.publish.lock` per file

The primary `audit.json` schema is `LB_SCGP_G0_CODE_AUDIT_PASS_ARTIFACT_V4`, with exact top-level fields and canonical `payload_sha256`. `publication_index.json` binds output file hashes and lock hashes. The producer and decision verifier consume the same exact schema and reject missing/additional/drifted fields.

## Deferred Dynamic Test Matrix

The later independent auditor should test, through SLURM, that `audit-publish` fails closed for:

- missing review report;
- missing review record;
- malformed JSON review record;
- wrong review report hash;
- wrong run ID;
- wrong config path;
- wrong freeze path/hash/payload;
- wrong implementation hash;
- dirty-state drift;
- nonzero Critical or High;
- `no_segment_gold_pass=false`;
- segment-gold fields not false/false;
- nonzero forbidden counters;
- preexisting output file, lock, or `code_audit` directory;
- partial/foreign formal output directory;
- wrong wrapper `TASK`;
- wrong wrapper `RUN_ID`;
- wrong review or review-record path.

No dynamic publish-path execution is authorized in this repair task.
