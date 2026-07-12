# LB-SCGP G0 v4 Tooling Repair Handoff

**Date:** 2026-07-11
**Executor:** sole v4 repair executor; no subagent, sidecar, reviewer, dynamic workflow, nested Codex process, or other model was used.
**Scope:** minimal tooling-lineage repair only.

## Repair Summary

- Added `configs/lb_scgp/lb_scgp_v4.json` with namespace `artifacts/lb_scgp/v4`.
- Registered exact v4 run IDs:
  - `LBSCGP-G0-FREEZE-v4`
  - `LBSCGP-G0-CODE-AUDIT-v4`
  - `LBSCGP-G0-SYNTH-v4`
  - `LBSCGP-G0-REAL-MHC_zh-F4-S0-v4`
  - `LBSCGP-G0-REAL-REPLAY-MHC_zh-F4-S0-v4`
  - `LBSCGP-G0-DECISION-v4`
- Added strict future review paths to the v4 dirty-policy exact exclusions:
  - `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.md`
  - `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.record.json`
- Added exact v4 mutable tooling records to the dirty-policy exclusions:
  - `G0_V4_AUDIT_PUBLISHER_DESIGN.md`
  - `G0_V4_TOOLING_REPAIR_HANDOFF.md`
  - `G0_FREEZE_EXECUTION_V4.md`
- Added v4 formal artifact exclusion prefix `artifacts/lb_scgp/v4/`, preserving v1-v3 prefixes.
- Embedded v1-v3 freeze/lock hashes in v4 config for later non-overwrite verification.
- Added `scripts/slurm/lb_scgp_g0_audit_publish.sbatch`, a CPU-only 2 CPU / 4G wrapper with no `--time`.
- Added `audit-publish` to `scripts/analysis/lb_scgp_independent_verify.py`.
- Tightened the v4 code-audit consumer contract in both:
  - `scripts/analysis/lb_scgp_g0.py::_load_freeze_and_audit`
  - `scripts/analysis/lb_scgp_independent_verify.py::decide`

## Formal Audit Schema

The formal audit artifact type is:

```text
LB_SCGP_G0_CODE_AUDIT_PASS_ARTIFACT_V4
```

The strict machine review record type is:

```text
LB_SCGP_G0_CODE_AUDIT_INDEPENDENT_REVIEW_RECORD_V4
```

Both schemas reject missing and additional top-level fields. The publisher will not infer PASS from free-form report text. It requires the strict review record and independently recomputes all frozen checks before publication.

Published formal namespace, after a later independent audit only:

```text
artifacts/lb_scgp/v4/g0/code_audit/
```

Expected files:

```text
review.md
review_record.json
audit.json
publication_index.json
*.publish.lock
```

## Later Invocation

The next independent auditor may invoke only after creating the real v4 review and strict machine record:

```text
CONFIG=configs/lb_scgp/lb_scgp_v4.json TASK=audit-publish RUN_ID=LBSCGP-G0-CODE-AUDIT-v4 REVIEW=refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.md REVIEW_RECORD=refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.record.json sbatch scripts/slurm/lb_scgp_g0_audit_publish.sbatch
```

## Non-Claims

- No v4 formal audit PASS was created by this repair.
- No `G0_FORMAL_CODE_AUDIT_REVIEW_V4.md` was created.
- No v4 strict review machine record was created.
- No audit-publish task was run.
- No synthetic, realfold, replay, decision, G1, teacher, MLLM, OCR, held-label, held-content, val, test, or performance task was run.
- The only gold remains the parent-video binary label.
- `segment_gold_exists=false` and `segment_gold_used=false` remain binding.

## Next Boundary

After v4 freeze completes, a fresh independent GPT-5.5 xhigh auditor must:

1. Review the v4 frozen lineage.
2. Create `G0_FORMAL_CODE_AUDIT_REVIEW_V4.md`.
3. Create `G0_FORMAL_CODE_AUDIT_REVIEW_V4.record.json` with exact schema and payload hash.
4. Invoke the audit-publish wrapper.
5. Validate the published artifacts before any synthetic authorization.
