# M0 Run2-v4 Plan Amendment: LB-SCGP Global-R2

Date: 2026-07-13

Author thread: fresh **Claude Opus 4.8** in the **v4-prep role** in `/data/jehc223/RGCL`. This
role is deliberately separate from — and does not perform the work of — the later roles required
by `M0_RUN2_V3_RESULT_TO_CLAIM_REVIEW.md` §5(a)/(c)/(d): the independent v4 amendment reviewer,
the fresh 0C/0H v4 static code reviewer, the independent execution authorizer, and the
independent executor. No subagent, workflow, external model, API/MLLM/OCR call, GPU/training,
validation/test, or performance work was performed. No project Python was executed; `jq -e .`,
`sha256sum`, `sed`, `diff`, and `grep` were used only as read-only well-formedness / hash / text
tools. No SLURM job was submitted and nothing was committed to git.

## Verdict

This is a prospective planning amendment plus a byte-exact v4 clone/freeze. It is **ready for
independent v4 amendment review**. It does **not** authorize v4 execution.

If independent amendment review passes, the only next boundary it can authorize is a **fresh
independent 0C/0H v4 code review** — which must include the mandatory runtime cross-check
static-simulation table (`M0_RUN2_V3_RESULT_TO_CLAIM_REVIEW.md` §5(c), every row PASS) and the
dependency-availability evidence item (§5(d)). Execution then additionally requires exact
hashes / no-clobber review and a separate execution authorization. Run3 (`REALBANK-RESOURCE-v1`)
and all later runs, including MLLM/cache work, remain locked until Run2-v4 PASS and a fresh
independent v4 artifact review.

## Motivation — the v3 death and why a plan amendment is a hard precondition

Per `M0_RUN2_V3_RESULT_TO_CLAIM_REVIEW.md` (fresh 0C/0H result-to-claim review), job `12974`
FAILED with a `RuntimeError` at `verify_machine_run2`:

```
machine run order[2] drift: expected 'LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v3',
got 'LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2'
```

Classification: `infrastructure_plan_document_code_drift` (NOT science), fail-closed, zero
false-positive, zero science leak. Root cause: the v3 clone advanced the **code** to v3
(`RUN2`/`PAYLOAD_SCHEMA_ID`) but performed **no plan-amendment ceremony**, so the authoritative
`EXPERIMENT_PLAN.machine.json` still described **v2** at index `[2]`. The v3 single-submit
budget is spent and the v3 lineage is **CLOSED**. The v2 lineage was already closed (job `12971`,
missing `jsonschema`). This amendment opens a brand-new **v4** lineage and writes it into the
authoritative plan **first**, which is the specific defect that killed v3.

## Preserved Run2-v1 / v2 / v3 evidence (closed lineages)

Prior-version failure evidence is preserved in the refine-logs lineage (result-to-claim reviews +
execution/clone records), **not** as run-array entries, per the v3 verdict §5(a):

| Lineage | Job | Failure | Status |
|---|---|---|---|
| v1 | `12902` / `12904` | interface `KeyError` (finite_vi_diagnostic / payload_schema) | FAIL_STOP; retained as `runs[1]` historical record |
| v2 | `12971` | missing `jsonschema` dependency | closed; evidence in `M0_RUN2_V2_*` + `M0_RUN2_V2_RESULT_TO_CLAIM_REVIEW.md` |
| v3 | `12974` | machine `run_order[2]` v2/v3 plan-document/code drift | consumed and CLOSED; evidence in `M0_RUN2_V3_*` + `M0_RUN2_V3_RESULT_TO_CLAIM_REVIEW.md` (sha256 `fcdb7ec4…f45a7`), death log `…run2_v3_12974.err` (sha256 `3482445c…f4adc`) |

None of these may be deleted, overwritten, reused as v4, or called PASS.

## Placement semantics — REPLACE at index `[2]`, not INSERT (diverges from the v2 precedent, with cause)

The v2 amendment **inserted** `…-v2` at index 2 and shifted realbank 2→3, because v2's code was a
rewrite whose `verify_machine_run2` moved to `run_order[2]`. **v4 is a byte-clone of v3**, so its
`verify_machine_run2` is pinned to `machine["runs"][2]` / `run_order[2]`
(`…v4_common.py:820-821`, a numeric literal a token-sed cannot move). Therefore this amendment
makes **index `[2]` itself become v4**, i.e. it **replaces the index-2 record in place** (v2
content → v4 content), keeping array length and every downstream index unchanged (realbank stays
at `[3]`; the entire `[3..65]` tail is unchanged). It does **not** insert. The prior v1→v2 INSERT
history is preserved verbatim in `dependency_dag.pre_post_run_order_diff.inserted_run_id` /
`inserted_after`; the v2→v4 REPLACE is recorded in the new
`dependency_dag.pre_post_run_order_diff.v4_replace_record` block and in the
`supplement_amendment` block rather than by silently rewriting v1→v2 history.

> The independent amendment review must explicitly ratify this REPLACE semantics as the correct
> byte-clone analogue of the v2 INSERT.

## Machine-JSON change list (Group A / B / C — before → after)

Applied to `refine-logs/lb_scgp_global/EXPERIMENT_PLAN.machine.json` (backup of the pre-amendment
file at `EXPERIMENT_PLAN.machine.json.pre_v4_amendment.bak`, sha256
`6caa5c2e78961fc7d60f9f970e319534aa6af3b6cc4a033b76b83f2ab18a0492`).

### Group A — CODE-VALIDATED at index `[2]` (hard runtime asserts; each flipped to v4)

| # | JSON path | before (v2) | after (v4) | runtime assert |
|---|---|---|---|---|
| A1 | `run_order[2]` | `…SYNTH-KKT-v2` | `…SYNTH-KKT-v4` | `…v4_common.py:821` (the v3 killer) |
| A2 | `runs[2].run_id` | `…-v2` | `…-v4` | `:822` |
| A3 | `runs[2].artifact_paths[0]` | `…/v2/…/manifest.json` | `…/v4/…/manifest.json` | `:823` (== `cfg.run.artifact_path`) |
| A4 | `runs[2].artifact_schema_ids[0]` | `…payload_v2` | `…payload_v4` | `:824` (== `PAYLOAD_SCHEMA_ID`) |

### Group A′ — code-validated but MUST NOT change (confirmed unchanged)

| # | JSON path | value (unchanged) | assert |
|---|---|---|---|
| A5 | `runs[2].slurm` | `{cpu:8, ram_gb:64, gpu:0, env:"HateVideo", no_time_flag:true}` | `:825` (== `cfg.run.slurm`) |
| A6 | `runs[2].dependencies` | `["…CONTRACT-FREEZE-v1", "…SYNTH-KKT-v1"]` | `:826` (== `[RUN1, RUN2_V1]`) — v4 depends on synth-kkt-**v1**, not v2/v3 |

Confirmed post-edit by `jq`: `runs[2].slurm` and `runs[2].dependencies` are byte-identical to the
pre-amendment file.

### Group B — plan-coherence (not read by `verify_machine_run2`; flipped to a coherent v4 record)

- `runs[2].purpose` → v4 wording (byte-clone of v3; opened by v4 amendment; no science change).
- `runs[2].planned_config_path` → `configs/lb_scgp_global_r2/m0_synth_kkt_v4.json`.
- `runs[2].artifact_namespace` → `artifacts/lb_scgp_global/v4/m0/synth_kkt/`.
- `runs[2].lineage_bound_outputs[0..7]` → `v4_config … v4_manifest`.
- `runs[2].science_freeze_assertions.solver_math_verifier_logic_change_scope` → v4 wording
  (byte-clone of v3; `solver_math_verifier_logic_unchanged=false` stated relative to v1; the
  v1→v2 fix2 reconciliation is inherited unchanged; **fixture/threshold/rank-gate/scope
  unchanged**).
- `runs[2].authorization_requirements` → the two `…_v2_…` requirement keys renamed to `…_v4_…`;
  added `requires_runtime_crosscheck_static_simulation_table_all_pass` and
  `requires_dependency_availability_evidence` (v3-verdict §5(c)/(d) items).
- `runs[2].implementation_fix_freeze` (v2 fix/fix2 review history) → **replaced** by a
  `runs[2].clone_freeze` block that truthfully states v4's state: byte-clone of v3, clone-freeze
  complete, amendment authored, and **all** reviews/authorizations PENDING (`false`). This
  removes the v2-specific completed-review claims (`fresh_fix_review_failed 0C/2H/1M/1L`,
  `fix2_freeze_complete`) that would be **false for v4** — an anti-fabrication requirement, and
  aligned with §5(a) ("prior-version evidence preserved in refine-logs, not run-array entries").
- `runs[2].gate` / `runs[2].failure_transition` / `runs[2].status` → v4 wording; status
  `FIX2_COMPLETE_FRESH_REVIEW_LOCKED` → `V4_CLONE_FROZEN_AMENDMENT_PENDING_INDEPENDENT_REVIEW`.
- `runs[3].dependencies[0]` (realbank) → `…SYNTH-KKT-v4` (the downstream repoint the v2 amendment
  did as v1→v2).
- `runs[3].gate` → references Run2-v4 PASS + fresh independent v4 artifact review; preserves
  v1/v2/v3 as FAIL_STOP / consumed-closed evidence. `runs[3].status`
  `LOCKED_UNTIL_V2_PASS` → `LOCKED_UNTIL_V4_PASS` (names the dependency, now v4).
- `dependency_dag.pre_post_run_order_diff.run3_new_dependency` → `…-v4`; `inserted_run_id` /
  `inserted_after` preserved (v2 history) with an added note; new `v4_replace_record` sub-block.
- `dependency_dag.terminal_decision_chain[2]` → `…-v4`.
- `gates_and_terminal_statuses.v2_authorization_boundary` → renamed `v4_authorization_boundary`
  and made truthful (`amendment_review_passed…=false`, `clone_freeze_complete=true`, v4 execution
  gates) — same anti-fabrication rationale as the `clone_freeze` block.
- `readiness`: `first_three_g0_runs[2]` → v4; `status` → v4; `g0_runs_through_v2_supplement`
  renamed `g0_runs_through_v4_supplement` with `[2]` → v4; the v2 review-status flags rewritten
  to the truthful v4 pending-review set (`carried_static_residuals_pending_fresh_v4_review =
  [M-A, M-B]`, per the v3 verdict's carried residuals).
- `supplement_amendment` block → full v4 rewrite (`id` `M0_RUN2_V4_PLAN_AMENDMENT`, `lineage`,
  `status`, `lineage_model="byte_exact_clone_of_v3_no_fix_cycle"`, `placement_semantics`,
  `supersedes.{v2,v3}`, `author_thread`, `amendment_artifacts`, `clone_freeze_artifact`,
  `planned_config_path`, `artifact_path`, `artifact_namespace`, `run3_realbank_dependency` → v4).

### Group C — registry coherence

- `artifact_schemas`: **added** `scgp_global_synth_kkt_payload_v4` (same key list as
  `…payload_v1`/`…payload_v2`, which are retained). No run2 module reads `artifact_schemas`, so
  this is coherence-only.

## Deliberately preserved v2-vocabulary (flagged for the independent amendment reviewer)

The following fields still contain a `v2`/`fix2` token after the amendment. Each is **either
genuine history, an unrelated schema name, or policy/budget vocabulary** the v3 verdict's §5(a.1)
change-list did **not** enumerate. Numeric values remain correct for v4. They are surfaced here
explicitly so the reviewer can decide whether to extend the amendment; the v4-prep author chose
**not** to rewrite them (avoids rewriting history / over-editing beyond the verdict's mandate):

| JSON path | token | classification |
|---|---|---|
| `schema_version` | `scgp_global_experiment_plan_v2` | plan schema version, not a lineage id |
| `runs[4].artifact_schema_ids` / `runs[5]…` | `scgp_global_cache_replica_v2` | M1 cache schema (different lineage) |
| `artifact_schemas.scgp_global_synth_kkt_payload_v2` | key | registry entry retained (v4 added alongside) |
| `dependency_dag.pre_post_run_order_diff.inserted_run_id` (+ note) | `…-v2` | preserved v2 INSERT history |
| `dependency_dag.pre_post_run_order_diff.v4_replace_record.replaced_run_id` | `…-v2` | names what v4 replaced (history) |
| `dependency_dag.locks.run3_and_later_locked_until_v2_pass` | key | gating vocabulary; value `true` still correct |
| `dependency_dag.locks.mllm_cache_locked_until_v2_pass_and_fresh_independent_artifact_review` | key | gating vocabulary; value `true` |
| `budget_ranges.conventions.lineage_run_records_include_historical_v1_and_prospective_v2` | key | budget convention; value `true` |
| `budget_ranges.conventions.paper_plan_substitution_view_replaces_exhausted_v1_with_v2` | key | budget convention; value `true` |
| `budget_ranges.original_approved_r2_envelope_before_v2` | key | historical budget baseline name |
| `budget_ranges.*.status_aware_counts.must_fix2_complete_fresh_review_locked_v2` | key | run-count-by-status; value `1` still correct (one M0 supplement run, pre-execution) |
| `budget_ranges.*.status_aware_counts.must_downstream_locked_until_v2_pass` (+ `_explicit_`) | key | run-counts `62` / `1` still correct |
| `…isolation…v1_synth_kkt_artifact_reuse_as_v2_count` | key | isolation counter; value `0` still correct |

## Hash cascade (before → after)

Editing the authoritative plan forces a bounded cascade that the runtime asserts **do** check
(`…v4_common.py:840`→`:834` `verify_expected_hashes`; `…v4_independent_verify.py:952-954`).

| # | File / binding | before | after |
|---|---|---|---|
| 1 | `EXPERIMENT_PLAN.machine.json` (M′) | `6caa5c2e…0492` | `42bf49ed…4590a90` |
| 2 | `EXPERIMENT_PLAN.md` | `af1c217c…7b23a` | `a98effc3…5ae3eb` |
| 3 | `EXPERIMENT_TRACKER.md` | `327614bb…00db2` | `4d3c4b8c…9e9da4` |
| 4 | `EXPERIMENT_PLAN_HASHES.sha256` (H′; lines 1–3 updated to #1–#3) | `2e6d731d…c802` | `910f0f64…1568b1` |
| 5 | v4 config `authoritative_inputs[EXPERIMENT_PLAN.machine.json]` | `6caa5c2e…0492` | `42bf49ed…4590a90` (= M′) |
| 6 | v4 config `authoritative_inputs[EXPERIMENT_PLAN_HASHES.sha256]` | `2e6d731d…c802` | `910f0f64…1568b1` (= H′) |
| 7 | v4 config `authoritative_inputs[EXPERIMENT_PLAN.md]` | `af1c217c…7b23a` | `a98effc3…5ae3eb` |
| 8 | v4 config `authoritative_inputs[EXPERIMENT_TRACKER.md]` | `327614bb…00db2` | `4d3c4b8c…9e9da4` |
| 9 | v4 config `authoritative_inputs[M0_RUN2_V4_PLAN_AMENDMENT.md]` | (absent) | added |
| 10 | v4 config `authoritative_inputs[M0_RUN2_V4_PLAN_AMENDMENT.machine.json]` | (absent) | added |
| 11 | v4 config `authoritative_inputs[M0_RUN2_V4_PLAN_AMENDMENT_HASHES.sha256]` | (absent) | added |

Bindings 1–8 are the load-bearing runtime checks: `verify_expected_hashes` reads exactly
`cfg.hash_bindings.authoritative_inputs`, so the plan edit (#1) is only visible to the run if the
config binding (#5) is updated in lock-step; leaving either stale reproduces a Critical
hash-drift death. Bindings 9–11 are additive provenance for the v4 amendment docs. Per the v3
verdict's sequencing note, the v4 amendment **independent review** docs are **not** bound here
(they do not exist yet); the execution authorizer adds those bindings after the review and
re-freezes the config. Exact post-freeze config hash and the nine v4-entity hashes are recorded in
`M0_RUN2_V4_CLONE_FREEZE.md` (the config's own hash is not pinned by any runtime assert). The v2
amendment + review bindings remain in the config unchanged as historical provenance.

## v4 clone (byte-exact of v3)

The nine v4 entities are a byte-for-byte-equivalent clone of the frozen v3 entities, via the
single deterministic substitution
`sed 's/cert_v2/cert__CERTKEEP__/g; s/v3/v4/g; s/cert__CERTKEEP__/cert_v2/g'` (the cert guard is a
no-op for v3→v4 since `cert_v2` contains no `v3`, but is retained for symmetry). Equivalence,
per-entity SHA256, and the §5(c) runtime cross-check static-simulation table are in
`M0_RUN2_V4_CLONE_FREEZE.md`. `scgp_global_cert_v2` is preserved (Run1-frozen shared schema; not
one of the nine cloned entities). No behavioral code byte changed; the index literals
`machine["runs"][2]` / `run_order[2]` stay `[2]`.

## Science freeze

The v4 change is **byte-clone of v3 lineage identity + amendment-driven config hash-binding
cascade only**. Unchanged from v3 (and, for science, from v1): d=3, fixture identities/counts,
thresholds, case roles, expected decisions, KKT tolerances, movement/nondegeneration target,
rank/factor rules, solver/math/verifier logic, resource request (8 CPU / 64 GB / 0 GPU,
`HateVideo`, no `--time`), planned budget (32 CPU-h, 0 GPU-h, 0 API, 5 GB), intended claim, and
failure transition. Forbidden: solver fix, tuning, tolerance change, fixture change, rank-gate
weakening, rescue, or any post-hoc scientific change. Residual static findings **M-A** and **M-B**
carry unchanged to v4 and must be re-adjudicated by the fresh v4 code review.

## DAG amendment

Run order prefix after amendment:

1. `LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1`
2. `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1`
3. `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v4`  (index `[2]`, REPLACED in place from v2)
4. `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1`  (index `[3]`, unchanged; now depends on v4)

All other run IDs and downstream ordering are unchanged.

## Gold and isolation

Only gold supervision remains `parent_video_binary_label`. `segment_gold_exists=false`,
`segment_gold_used=false`; no segment/frame/timestamp/span/localization/stance/target/mechanism/
rationale/fragment gold exists or is used. All forbidden access/call counters remain zero. No
science tuning is authorized. v4 produced no artifact and no counters.

## Authorization boundary

- This amendment authorizes **no** execution.
- `ready_for_independent_amendment_review = true`; `ready_for_execution = false`.
- Remaining ceremony (per `M0_RUN2_V3_RESULT_TO_CLAIM_REVIEW.md` §5): independent v4 amendment
  review (ratifying REPLACE semantics) → fresh independent 0C/0H v4 code review including the
  §5(c) runtime cross-check static-simulation table (all rows PASS) and dependency-availability
  evidence (§5(d)) → exact hashes / no-clobber review → separate execution authorization → single
  executor submit (exactly one `sbatch`, no resubmission) → fresh result-to-claim review.
- The v4-prep role (this document + the clone/freeze) is separate from the amendment-review,
  code-review, execution-authorization, and executor roles.
