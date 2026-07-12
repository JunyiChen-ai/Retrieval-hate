# M0 Run2-v2 Implementation Audit

Date: 2026-07-12

Boundary: implementation audit only. I read `AGENTS.md` first. I did not run Python, submit or monitor SLURM, run experiments, use GPU/training/data/validation/test, call MLLM/OCR/model/network tools, or edit implementation files. Shell use was static/read-only inspection only. This report is the only write.

This audit does not authorize execution.

## Authority

- `M0_RUN2_RESULT_TO_CLAIM_REVIEW_FRESH.md` authorizes only one non-overwriting v2 repair lineage in principle, with no execution authorization, and requires a new review before Run3 consideration (`refine-logs/lb_scgp_global/M0_RUN2_RESULT_TO_CLAIM_REVIEW_FRESH.md:108-121`, `:139-143`).
- `M0_RUN2_V2_AMENDMENT_INDEPENDENT_REVIEW.md` authorizes only this v2 implementation audit; execution remains locked until completed implementation, static contract matrix/negative tests, fresh independent v2 code review with 0 Critical / 0 High, exact hashes/no-clobber, and separate execution authorization (`refine-logs/lb_scgp_global/M0_RUN2_V2_AMENDMENT_INDEPENDENT_REVIEW.md:127-131`, `:148-155`).
- The amended plan inserts `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2` under `artifacts/lb_scgp_global/v2/m0/synth_kkt/`, keeps science frozen, and requires exact hash/no-clobber review before any submission (`refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT.md:32-58`).

## Static Hash Checkpoint

Current implementation stack hashes:

| Path | SHA256 |
|---|---|
| `configs/lb_scgp_global_r2/m0_synth_kkt_v2.json` | `1b490acee733cb85d5bb977c6872ee6cf60d92ed5fb2dc02f5f1988345183312` |
| `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_payload_v2.schema.json` | `c4cbaa229d254e18e0aa1732e48370fc9794cdbf3055a8aaf1c038785db3eae4` |
| `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_case_v2.schema.json` | `df3616ff50c54dc756bb600c6ef26626afafe070678450d10ca73b9ff83ddcac` |
| `scripts/analysis/lb_scgp_global_r2_run2_v2_common.py` | `8589bdb93945e6f48c2b7d201b304c4fd2b2e583f71147371c66fc3b9e8b434e` |
| `scripts/analysis/lb_scgp_global_r2_run2_v2_validate.py` | `48099a580fa91672883320829e22469c069c5500cd38ec411f33ce4768633ebd` |
| `scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py` | `f12036859c0f8894609b9811c3073bdefdc0dae85b7b142b4575a21812c3c84b` |
| `scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py` | `623a8bd386bb300b1ed8775bc7fbd7f85ca0a3119ea844e1a7d91b9517c4b071` |
| `scripts/wrappers/lb_scgp_global_r2_run2_v2.sh` | `97e284043f9109aa0413e60fe711a497c4c463f7451ff8aad98f5c8da8db374b` |
| `scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v2.sbatch` | `f91462303e4b28eb0722d546f135c00c2aeacf56151e27d5421f211520ba94bf` |

Current authority and Run1 hashes:

| Path | SHA256 |
|---|---|
| `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT.md` | `95b4b839017479bad4d2ed7a48455864c765e8f91997adab0fa4a7ca66aa57fb` |
| `refine-logs/lb_scgp_global/M0_RUN2_V2_PLAN_AMENDMENT.machine.json` | `56603297b2a1f6bcf12e941d96ce95911bb2ab8bfd0de967586430d38959f3d8` |
| `refine-logs/lb_scgp_global/EXPERIMENT_PLAN.machine.json` | `838370c8eee68f568b6d133f80b305fed23bc14342d4c4bf08df976fd4d73d07` |
| `refine-logs/lb_scgp_global/EXPERIMENT_TRACKER.md` | `d0c7e6ff291b6ae4ba6d5661b2e8ddc2bb8a80ac7f2f6d1e37ab8494782df483` |
| `artifacts/lb_scgp_global/v1/m0/contract_freeze.json` | `09b78682389f1c9774c9dffc43c759bceeec9d7f44eca1ce4cd626d0cd6d12da` |
| `artifacts/lb_scgp_global/v1/m0/contract_freeze.json.publish.lock` | `c6fbb49c3c8aba942da3f254c3c5f65d037002548fafb961154e6fa79a3e4ea7` |

Old protected scope rehash: 278 files, manifest SHA256 `243e89b69b169b222dd97f9df092d511f823fb26201e91fd89cb581710940462`.

## Severity Summary

| Severity | Count |
|---|---:|
| Critical | 0 |
| High | 4 |
| Medium | 1 |
| Low | 1 |

## Findings

### H1: Stale and partly unenforced hash bindings block exact v2 provenance

Severity: High.

Evidence:

- The v2 config binds stale authoritative hashes: `EXPERIMENT_PLAN.machine.json` expected `8b86061f...` but current is `838370c8...`; `EXPERIMENT_PLAN_HASHES.sha256` expected `72a97e60...` but current is `0839260b...`; `EXPERIMENT_TRACKER.md` expected `2aa355a4...` but current is `d0c7e6ff...`; `M0_RUN2_V2_PLAN_AMENDMENT.md` expected `8f09886d...` but current is `95b4b839...` (`configs/lb_scgp_global_r2/m0_synth_kkt_v2.json:33-45`).
- The config's `v2_declared_source_hashes_without_config` block includes a stale common-module hash: expected `b5154808...`, current `8589bdb9...` (`configs/lb_scgp_global_r2/m0_synth_kkt_v2.json:70-79`).
- `verify_expected_hashes` would fail closed on stale authoritative inputs during source-manifest construction (`scripts/analysis/lb_scgp_global_r2_run2_v2_common.py:715-730`), before artifact publish. That is safe, but it means the stack is not execution-ready.
- The `v2_declared_source_hashes_without_config` block is not itself enforced: the source manifest recomputes implementation rows from `implementation_files` (`scripts/analysis/lb_scgp_global_r2_run2_v2_common.py:742-750`), and the independent verifier later checks only those emitted rows (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:317-319`).

Impact:

Exact hash/no-clobber binding is not satisfied. Future execution would either fail before publication or rely on source hashes that are not the declared config ledger. This is an implementation-blocking provenance issue, not a scientific result.

Repair:

Refresh all authoritative hashes to current reviewed files, include the amendment machine/hash artifacts and the 0C/0H amendment review as explicit bindings, and either enforce or remove the `v2_declared_source_hashes_without_config` block. Bind the config hash through a separate non-circular ledger or source manifest, not by stale self-reference.

### H2: Strict JSON Schema validation of emitted payloads is missing

Severity: High.

Evidence:

- The payload schema requires a strict case matrix whose cases reference the case schema (`schemas/lb_scgp_global_r2/scgp_global_synth_kkt_payload_v2.schema.json:277-284`).
- The case schema requires case-level `finite_vi_diagnostics` and all public case fields (`schemas/lb_scgp_global_r2/scgp_global_synth_kkt_case_v2.schema.json:92-101`, `:229-248`).
- The validator checks only JSON syntax and schema self-strictness via `schema_requires_no_additional_properties`; it does not validate a produced manifest against the payload/case schemas (`scripts/analysis/lb_scgp_global_r2_run2_v2_validate.py:50-57`, `:124-137`).
- The producer also checks only schema self-strictness before publishing (`scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py:214-218`, `:361-367`).
- The independent verifier uses handwritten top-level key checks and checks only the FULL case key set (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:340-370`). It does not schema-validate every case in `case_matrix`.
- The verifier checks top-level finite-VI non-acceptance (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:347-350`), but it does not check each case's `finite_vi_diagnostics` fields required by the case schema.

Impact:

MCF-M1 is not closed. A schema-invalid payload, especially in non-FULL cases or case-level finite-VI diagnostics, could pass the handwritten semantic path if `payload_sha256` were recomputed and the FULL-case semantic checks passed.

Repair:

Run real JSON Schema validation in the producer before publish and in the independent verifier before PASS. Validate the top-level manifest and every case against the v2 schemas with the external case-schema reference resolved. Add explicit checks that every case has non-accepting `finite_vi_diagnostics`.

### H3: Hash and injection checks can reject only by payload hash, not by semantic hash binding

Severity: High.

Evidence:

- The producer emits top-level hash declarations for config, payload schema, Run1 artifact/lock, implementation, operator, case matrix, source manifest, and access ledger (`scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py:349-359`).
- Case-level hashes are emitted for operator, primal, and case payload (`scripts/analysis/lb_scgp_global_r2_run2_v2_common.py:949-973`).
- The independent verifier checks source/access file hashes and recomputes current Run1/authoritative/implementation file rows (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:290-337`), but it does not compare the top-level `hashes.config_sha256`, `payload_schema_sha256`, `run1_artifact_sha256`, `run1_lock_sha256`, `implementation_sha256`, `operator_hash`, or `case_matrix_sha256` fields to recomputed values.
- The `perturbed_artifact_source_operator_hash` injection mutates only `manifest["hashes"]["operator_hash"]` (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:458-460`). Because the injection does not recompute `payload_sha256`, rejection can occur at the generic payload-hash check (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:343-344`) rather than through an operator-hash semantic check.

Impact:

The negative injection matrix overstates the strength of hash binding. A tampered manifest with internally recomputed `payload_sha256` but false top-level hash declarations could pass the current verifier for fields that are not independently recomputed and compared.

Repair:

In the independent verifier, recompute and assert every top-level and case-level hash field, including operator, primal, case payload, case matrix, config, schema, Run1 artifact/lock, and implementation hashes. Mutations in `run_injections` should recompute `payload_sha256` after tampering so each injection proves the intended semantic rejection path.

### H4: Serialized KKT normal-family fields are not fully semantically verified

Severity: High.

Evidence:

- The schema requires the serialized normal families and KKT status blocks: `affine_normals`, `box_coordinate_normals`, `soc_normals`, `psd_normal`, `halfspace_normals`, `stationarity`, `dual_feasibility`, and `complementarity` (`schemas/lb_scgp_global_r2/scgp_global_synth_kkt_payload_v2.schema.json:553-575`).
- The producer emits those families (`scripts/analysis/lb_scgp_global_r2_run2_v2_common.py:1055-1115`).
- The verifier checks PSD sign and recomputes core stationarity/PSD/structural relations (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:351-352`, `:390-420`), but it does not assert the serialized `dual_feasibility.status`, `complementarity.status`, per-family complementarity entries, normal block `present` flags, sign conventions, or normal residual fields.
- The `incomplete_cone_family` injection removes `soc_normals` (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:467-469`), but there is no injection for malformed present normal-family content with recomputed `payload_sha256`.

Impact:

The verifier is closer to a recomputed FULL primal/KKT check than a strict verifier of the serialized H-metric normal-cone certificate. KKT-only acceptance is partially implemented, but the serialized certificate fields can drift without being rejected.

Repair:

Assert every serialized normal-family block and KKT status/residual field against recomputed values or exact expected zero blocks, including dual feasibility, complementarity per family, sign conventions, and normal presence. Add recomputed-hash injections for wrong dual/complementarity/status/normal-family content.

### M1: Dirty/no-clobber lock allowlist and partial publish order are inconsistent

Severity: Medium.

Evidence:

- The config allows only four JSON output files after run and omits their `.publish.lock` files (`configs/lb_scgp_global_r2/m0_synth_kkt_v2.json:19-27`).
- The plan requires v2-bound locks/publish locks/no-clobber locks (`refine-logs/lb_scgp_global/EXPERIMENT_PLAN.machine.json:558-566`).
- `exclusive_publish_json` creates persistent `.publish.lock` files on successful publish (`scripts/analysis/lb_scgp_global_r2_run2_v2_common.py:204-230`), and the verifier does the same for semantic verification (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:503-522`).
- The wrapper removes all v2 outputs and locks only when `COMPLETE != 1` (`scripts/wrappers/lb_scgp_global_r2_run2_v2.sh:13-28`, `:65`).
- The producer publishes child artifacts before the manifest (`scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py:365-367`), so a final manifest publish failure depends on wrapper cleanup to remove already-published children.

Impact:

The path is fail-closed under normal wrapper exit, but the declared dirty policy does not match successful lock outputs, and partial child output cleanup depends on the wrapper. This can block retry/no-clobber review or make a failed attempt look partially published.

Repair:

Add an explicit allowed lock-output list or include locks in `allowed_new_files_after_run`. Prefer an all-or-none publish sequence or a staged directory finalized only after the manifest and semantic verification decision are both ready. Add signal cleanup for failure paths where practical.

### L1: Tracker/machine status is conservative but stale after amendment review

Severity: Low.

Evidence:

- The tracker still says Run2-v2 is `LOCKED_PENDING_AMENDMENT_REVIEW` (`refine-logs/lb_scgp_global/EXPERIMENT_TRACKER.md:4`, `:16`).
- The machine plan still labels the v2 run `LOCKED_PENDING_AMENDMENT_REVIEW` (`refine-logs/lb_scgp_global/EXPERIMENT_PLAN.machine.json:617-621`).
- The amendment independent review has already passed 0C/0H and authorized only this implementation audit (`refine-logs/lb_scgp_global/M0_RUN2_V2_AMENDMENT_INDEPENDENT_REVIEW.md:148-155`).

Impact:

This is conservative and does not allow execution. It can, however, make future boundary tracking ambiguous unless the next fix/freeze pass records that amendment review passed while implementation/code review and execution remain locked.

Repair:

In the next authorized planning/freeze update, move the v2 status to a post-amendment, implementation-fix/review-locked state while preserving `ready_for_execution=false`.

## Passed Static Checks

- Config run ID, schema ID, v2 artifact path, path keys, SLURM policy fields, and v2 namespace are present (`configs/lb_scgp_global_r2/m0_synth_kkt_v2.json:92-123`).
- Producer and schemas agree on case-level `finite_vi_diagnostics` naming, and producer emits expected injection keys (`schemas/lb_scgp_global_r2/scgp_global_synth_kkt_case_v2.schema.json:92-101`; `scripts/analysis/lb_scgp_global_r2_run2_v2_common.py:941-946`; `scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py:318-328`).
- The independent verifier does not import producer/common; its imports are standalone standard-library plus NumPy (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:1-20`).
- Producer executes `orth_cap` from certificate encodings and constructs `M_Q=Q^T(G-I)Q/N`; verifier replays `orth_cap` for the FULL case (`scripts/analysis/lb_scgp_global_r2_run2_v2_common.py:451-475`, `:847-858`, `:908-920`; `scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:371-377`).
- FULL movement/nondegeneration gates exist in producer and verifier (`scripts/analysis/lb_scgp_global_r2_run2_v2_common.py:897-905`; `scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:384-399`).
- Rank/factor/null failure semantics exist, including rank-failure null probe and verifier checks (`scripts/analysis/lb_scgp_global_r2_run2_v2_common.py:586-640`, `:867-885`, `:1123-1132`; `scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:422-434`).
- Run1 artifact and publish lock hashes are still current; old protected scope is still 278 files with manifest SHA256 `243e89b69b169b222dd97f9df092d511f823fb26201e91fd89cb581710940462`.
- SLURM requests exactly 8 CPU and 64 GB, has no GPU directive, activates `HateVideo`, and has no `--time` (`scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v2.sbatch:2-13`). The wrapper runs validator, producer, and independent verifier inside the future job (`scripts/wrappers/lb_scgp_global_r2_run2_v2.sh:49-63`).
- The target stack keeps zero forbidden route declarations and only `parent_video_binary_label` gold (`configs/lb_scgp_global_r2/m0_synth_kkt_v2.json:3-17`; `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_payload_v2.schema.json:345-354`; `scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py:329-334`).
- The amended DAG still blocks execution and Run3: execution-authorized remaining budget is zero, Run3 depends on v2, and execution requires separate authorization (`refine-logs/lb_scgp_global/EXPERIMENT_PLAN.machine.json:638-640`, `:3761-3772`, `:3793-3798`, `:4215-4222`).
- Current v2 outputs are absent: `artifacts/lb_scgp_global/v2` does not exist. Only v1 contract-freeze artifacts are present under `artifacts/lb_scgp_global/`.

## Repair List

1. Refresh and enforce v2 authoritative/source hash bindings, including amendment machine/hash files and the 0C/0H amendment review.
2. Add actual JSON Schema validation for the produced manifest and every case in both producer and independent verifier.
3. Recompute and assert all top-level and case-level hash fields in the independent verifier.
4. Strengthen negative injections by recomputing `payload_sha256` after mutations so rejection proves the targeted semantic guard.
5. Semantically verify every serialized normal-family, dual-feasibility, complementarity, PSD sign, finite-VI diagnostic, and KKT status field.
6. Align dirty/no-clobber output allowlists with persistent publish locks and reduce partial-publish cleanup dependence.
7. Update tracker/machine status only in a separate authorized planning/freeze pass, keeping execution locked.

## Decision

- Critical: 0
- High: 4
- Medium: 1
- Low: 1
- implementation_audit_result: FAIL_BLOCKED_FOR_EXECUTION
- execution_authorization: not_authorized
- run3_unlock: not_authorized
- next_boundary: implementation-fix/freeze pass only, followed by a fresh independent implementation/code review. No SLURM submission, Python validation, experiments, data processing, MLLM/OCR/cache work, GPU/training/evaluation, Run3, or realbank unlock is authorized by this audit.

Report SHA256: compute after file creation; not embedded to avoid self-referential drift.
