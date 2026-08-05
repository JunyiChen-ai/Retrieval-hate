# C01 A0 v4 typed-audit repair record

**Status:** `V4_READY_NOT_RUN_PENDING_REVIEW`

This record is prospective. No Python execution, SLURM submission, result,
decision, metric, CONTINUE/KILL verdict, retry, C02 action, or test-cache access
is authorized or claimed.

## Frozen predecessor and failure evidence

- V3 remains byte-frozen:
  - config `configs/c01/c01_a0_v3.json`, SHA256
    `4ddb0f6f322de06316ea014a77c732b1a593c0fae5d926558d6c64a1be21cda5`;
  - analysis `scripts/analysis/c01_policy_contrast_a0_v3.py`, SHA256
    `40b35eee2fb6fdbdb21fe9b4acfdcebf003c121c76492b898fbd2ea9b8c34dfb`;
  - wrapper `scripts/slurm/c01_a0_cpu_v3.sbatch`, SHA256
    `e61b99620622d4161e0baded335e6172bd55b606e425668843cd9d370489af99`;
  - record `refine-logs/C01_A0_V3_RECORD.md`, SHA256
    `3af07f73155ba7b6857879e2f8b408028cf9a3a2edf34f99f16a82bebb5138fb`.
- Job `13735` remains a fail-closed historical run, not a scientific result.
  Its stdout/stderr SHA256 values are respectively
  `cf2a95043ca98139756f42a93693869184c111c577c58168ba5c7987435c9124`
  and
  `9271e642fb6f0fd85265cf9fd4633432647c5ca49659a4cb2eb431f950c92cf6`.
  The v3 namespace is still empty and contains no result or decision.
- V4 exact-binds `TARGET_REVIEW_RAW.md` lines 1340–1363, heading
  `C01 A0 v3 scoped static re-review and runtime correction (2026-07-29)`,
  section SHA256
  `3a7c4fdad2e740b2d224d86d982fb5972ace55374df0fb976cf577e4e915af1d`.

## V4 identity and acyclic SHA chain

- Run/schema: `C01-A0-v4`, `c01_a0_result_v4`,
  `c01_a0_decision_v4`.
- Config: `configs/c01/c01_a0_v4.json`, SHA256
  `2d9488e6f9af6be00d500d1c2f13912fd4be0ab9439608d33b0857178efe7ca6`.
- Analysis: `scripts/analysis/c01_policy_contrast_a0_v4.py`, SHA256
  `3c545eed876f97aa05f3e85375430bedf8e63226c70f3ee8ea12da02e9bf5514`.
- Wrapper: `scripts/slurm/c01_a0_cpu_v4.sbatch`, SHA256
  `9ae7f10370114647063f5ab18d97e6080ff8f90543a6e74c7b4fc24494bbf107`.
- Exclusive namespace:
  `artifacts/c01_policy_contrastive/v4/a0/C01-A0-v4`; it is absent at
  preparation time.
- The source pins the config and frozen v3 source. The wrapper pins v4
  source/config and frozen v3 source/config. This immutable record pins the
  wrapper. TARGET records pin this record after its final bytes are frozen.
  No file attempts to pin its own SHA256.

## Exact schema repair

The v3 numerical-equivalence contract is copied byte-for-JSON-value into the
v4 config and compared for exact equality to the frozen v3 config before any
runtime work. V4 adds one HALT-only discriminated audit union:

- `REGISTERED_NULL_RETRIEVAL`: a real retrieval audit. Its
  `registered_null_top20_count` must have exact Python type `int` (not `bool`)
  and value `0`, with semantics
  `DIRECT_COUNT_FROM_TOP20_NEIGHBOR_IDS`.
- `DERIVED_AVERAGE_SCORE_CONTROL`: the `avg_score` control is not a retrieval.
  Its explicit `registered_null_top20_count` must be JSON/Python null/`None`.
  Its source proof must name exactly `endpoint_std` and `endpoint_ow`, and
  both linked source audits must be direct-retrieval variants whose explicit
  integer counts are zero. No direct count is fabricated for the average.
- `NO_REGISTERED_NULL`: the MHC-ZH path has an explicit integer occurrence
  count `0` and semantics `NOT_APPLICABLE_NO_REGISTERED_NULL`. It is dispatched
  by `audit_kind`, not inferred from a missing field.

All three variants require exact top-level key sets and exact types/values for
the aggregation fields. Evaluation/reference booleans used by public guards are
also exact-key checked and must be literal true booleans. Unknown variants,
missing fields, surplus fields, `bool` masquerading as `int`, a non-null derived
count, missing/wrong source arms, or nonzero source counts fail closed with
`HALT_AUDIT_SCHEMA_V4`.

The public and final HALT-only aggregators consume only the typed summaries.
They do not use permissive `dict.get` defaults. V4's wrapper around the frozen
shuffle implementation validates every producer audit before the legacy
shuffle counter sees it, then exact-key/type validates the returned shuffle
aggregate.

## Fail-closed self-test

The source contains a pure schema self-test that is required after SLURM
runtime enforcement and before namespace creation. It is persisted in both
result and decision artifacts and requires these six exact cases:

1. valid HateMM direct retrieval plus derived average-score linkage passes;
2. HateMM average-score missing the explicit count is rejected;
3. HateMM average-score carrying an integer direct count is rejected;
4. HateMM average-score with a nonzero source-arm count is rejected;
5. valid MHC-ZH `NO_REGISTERED_NULL` passes;
6. MHC-ZH `NO_REGISTERED_NULL` missing the count is rejected.

Negative fixtures count as passing only when they raise the exact v4 schema
HALT label. An unexpected exception is re-raised, and acceptance of an invalid
fixture itself HALTs.

## Unchanged science and execution boundary

V4 SHA-imports v3 and changes no representation, arm, retrieval operation,
rotation, shuffle draw, bootstrap, Holm family, displacement rule, metric,
threshold, gain, net-fix rule, finite numerical envelope, signed-zero rule, or
binary64 reference. The runtime diff from v3 is restricted to run/output
identity, result/decision schema versions, and the additional HALT-only audit
schema guard.

The wrapper remains CPU-only at 8 CPU / 32 GB, uses conda `HateVideo`, requests
no GPU, and has no `--time`, dependency, array, singleton, chain, force, or
release path. Only JSON parsing, Bash syntax, source/config hash consistency,
textual schema inspection, namespace absence, frozen-v3 hash checks, and
`git diff --check` are performed during preparation. The embedded Python
self-test is not executed on the login node. Execution requires fresh
independent static review and separate explicit authorization.
