# LB-SCGP G0 v5 NameError Repair Design

**Date:** 2026-07-11
**Scope:** minimal no-clobber lineage repair for the sole v4 operational High.

## Operational Truth

v4 formal review files and published code-audit artifacts exist and are
internally consistent, but v4 remains **FAIL/BLOCKED**. The producer consumer
path failed after publication:

```text
NameError: name 'git_state' is not defined
```

The failure occurs in
`scripts/analysis/lb_scgp_g0.py::_verify_v4_code_audit_schema` while
`_load_freeze_and_audit` verifies the strict code-audit schema. Therefore v4
formal outputs are evidence for a failed operational lineage and are not valid
downstream unlock evidence.

## Repair Boundary

The v5 repair changes only lineage/tooling behavior needed to consume a strict
audit schema without the missing symbol and to fork a no-clobber namespace.

No scientific method, numerical threshold, supervision rule, data protocol,
projector, Dykstra logic, ranking logic, Farkas/cone logic, factorization,
fit/rollback behavior, cost formula, or evaluation behavior is changed.

## Implementation Plan

- Import `git_state` into the G0 producer from `lb_scgp_common`.
- Make the existing strict code-audit schema contract lineage-aware for `v4`
  and `v5` while preserving the exact top-level schema and fail-closed checks.
- Keep strict record/artifact/index types lineage-specific:
  - `LB_SCGP_G0_CODE_AUDIT_INDEPENDENT_REVIEW_RECORD_V5`
  - `LB_SCGP_G0_CODE_AUDIT_PASS_ARTIFACT_V5`
  - `LB_SCGP_G0_CODE_AUDIT_PUBLICATION_INDEX_V5`
- Extend the independent verifier publisher/decision consumer to the same
  v4/v5 contract.
- Make the audit-publish wrapper config-driven for `v4` or `v5`; it still
  rejects any other lineage and still requires exact config review paths.
- Add `configs/lb_scgp/lb_scgp_v5.json` with namespace
  `artifacts/lb_scgp/v5`, exact v5 run IDs, exact future review/record paths,
  and narrow dirty exclusions for this v5 repair documentation plus future v5
  sidecars.

## v5 Run IDs

```text
freeze   LBSCGP-G0-FREEZE-v5
audit    LBSCGP-G0-CODE-AUDIT-v5
synth    LBSCGP-G0-SYNTH-v5
real     LBSCGP-G0-REAL-MHC_zh-F4-S0-v5
replay   LBSCGP-G0-REAL-REPLAY-MHC_zh-F4-S0-v5
decision LBSCGP-G0-DECISION-v5
```

Only the freeze run is authorized in this repair.

## Prior-Lineage Evidence

v5 embeds exact no-clobber hashes for:

- v1-v4 `CONFIG_FREEZE.json` and `.publish.lock` files;
- all v4 formal code-audit artifacts and locks under
  `artifacts/lb_scgp/v4/g0/code_audit/`.

Those v4 formal files are prior-lineage no-clobber evidence only. They are not
accepted as v5 audit artifacts and cannot unlock v5 numerical stages.

## Regression

Focused audit-only SLURM regression uses temporary runtime fixtures under
`refine-logs/lb_scgp/runtime/v5_repair_checks/`, not v4 formal artifacts as
valid unlock evidence. It exercises:

- valid v5 strict-schema producer consumption;
- wrong review-record hash;
- wrong dirty hash;
- wrong code-audit run ID;
- wrong publication path.

All Python/import/runtime checks must run under SLURM with conda `HateVideo`.

## Non-Claims

- No v5 review, review record, formal PASS, synthetic, realfold, replay,
  decision, G1, teacher, MLLM, OCR, held-label/content, validation, test, or
  performance job is authorized by this repair.
- The only gold remains `parent_video_binary_label`.
- `segment_gold_exists=false` and `segment_gold_used=false` remain binding.
