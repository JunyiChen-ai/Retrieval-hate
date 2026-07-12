# LB-SCGP Global-R2 Run2 Result-to-Claim Review

Date: 2026-07-12

Reviewer boundary: independent local result-to-claim review only. No subagents, no Python, no SLURM submission, no implementation/config/artifact edits. This file is the only written output.

## Structured Verdict

- intended_claim: Run2 synthetic closed-convex global projection and serialized H-metric normal-cone/KKT are executable and unlock Run3.
- claim_supported: no
- route: supplement
- confidence: high
- executable_decision: DO NOT UNLOCK RUN3. Permit at most one prospective, non-overwriting supplement lineage, `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2`, under the boundaries below.

## What Supports

- The approved global proposal and R2 plan define a valid intended C1 gate for synthetic KKT before Run3.
- Run1 `LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1` is frozen and independently reviewed as authorizing Run2 only, not Run3.
- Job `12902` was submitted through the Run2 SLURM path and failed clearly with exit `1:0`; this supports only the procedural fact of a fail-closed attempt.
- Tracker and execution notes record Run2 as `FAIL_STOP`, no second Run2 job, and no published artifact.

None of this supports the scientific or executable KKT claim itself.

## What Does Not Support

- `sacct -j 12902` reports `FAILED`, exit `1:0`, elapsed `00:00:04`.
- `slurm/logs/lbscgp_global_r2_run2_synth_kkt_12902.err` contains only the Python traceback ending in `KeyError: 'finite_vi_diagnostic'`.
- `slurm/logs/lbscgp_global_r2_run2_synth_kkt_12902.out` is empty.
- `artifacts/lb_scgp_global/v1/m0/synth_kkt/` does not exist.
- Therefore there is no Run2 `manifest.json`, no publish lock, no payload hash, no access ledger, no case matrix, no verified KKT payload, no injection results, no rank-tail/factor/Procrustes evidence, and no independent verification artifact.
- Local preflight or post-run Python results are excluded from PASS evidence because they violated the current AGENTS.md SLURM discipline.

Run2 cannot be interpreted as a numerical solver failure, MLLM failure, OCR failure, dataset failure, or scientific failure. It failed before manifest publication because of a producer/verifier field assembly bug.

## Root Cause

Primary root cause:

- Active producer `scripts/analysis/lb_scgp_global_r2_synth_kkt.py` builds each case with `case_payload["finite_vi_diagnostic"]`.
- `verify_manifest` then checks `manifest["finite_vi_diagnostic"]["used_for_acceptance"]` at top level before the manifest exposes that field.
- The active manifest assembly copies many FULL-case blocks to top level but omits `finite_vi_diagnostic`, causing the top-level `KeyError` before publication.

Secondary contract risk:

- The active wrapper calls `lb_scgp_global_r2_synth_kkt_validate.py` and `lb_scgp_global_r2_synth_kkt.py`.
- The active producer manifest shape is not aligned with `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_payload_v1.schema.json`: for example schema expects `schema_version=lb_scgp_global_r2_synth_kkt_manifest_v1` and `terminal_state in {PRODUCED_PENDING_INDEPENDENT_VERIFY, FAIL}`, while the active producer assembles `schema_version=scgp_global_synth_kkt_payload_v1` and `terminal_state=RUN2_SYNTH_KKT_GO`.
- The unused `lb_scgp_global_r2_run2_producer.py` appears closer to that schema, but it is not wired by the active Run2 wrapper/config and cannot be treated as Run2 evidence.

## Missing Evidence

The following evidence is missing and blocks Run3:

- A published non-overwriting Run2 v2 manifest and lock.
- Canonical payload hash and access ledger hash.
- SLURM-only validation/provenance for the exact v2 source/config/schema/wrapper.
- Case results for FULL, REMOVE/null, SHUFFLE, NOISE, and required injection failures.
- Serialized H-metric normal-cone/KKT verification with finite-VI marked diagnostic only.
- Common-basis `orth_cap` and `M_Q=Q^T(G-I)Q/N` replay evidence.
- Rank-tail `<= d`, factor, and Procrustes replay.
- Zero counters for MLLM/OCR/held/validation/test/query/cache/certificate/compiler/teacher/head/reranker/key routes.
- Explicit closure of the three Run1 Medium findings from `M0_CONTRACT_FREEZE_INDEPENDENT_REVIEW.md`: semantic validation beyond schema-only acceptance, dirty-state binding, and actual `orth_cap`/`M_Q` execution.
- Prospective plan/tracker amendment and independent code review before any v2 submission.
- Immutable discipline record for non-SLURM local Python commands.

## Lineage Decision

`LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2` is allowed once, with restrictions:

- It must be a new lineage and new artifact path; it must not overwrite or remove v1 failure evidence, job `12902` logs, tracker row, execution notes, or hashes.
- Run1 contract freeze may be reused if v2 is limited to non-scientific repair of the Run2 implementation contract and binds the frozen Run1 artifact/hash/lock plus new v2 source/config/schema/wrapper/dirty-tree hashes.
- A new contract freeze is required if v2 changes scientific thresholds, fixtures, expected outcomes, solver semantics, data scope, authorization scope, or gold/leakage boundaries.
- The tracker and machine/plan records must be amended prospectively before v2 execution; post-hoc tracker repair is not enough.
- A fresh independent code review must approve v2 before submission.
- If v2 fails, it must enter a new result-to-claim review. No automatic retry or parameter-tuning rerun is authorized.

## Minimal Repair Boundary

Allowed:

- Fix field assembly order and location so finite-VI diagnostic fields are consistently case-level or top-level by contract, and verified before publish.
- Make producer, semantic verifier, wrapper, config, and JSON schemas describe the same artifact shape.
- Replace login-node preflight with an authoritative SLURM-only preflight/check job, or move all executable validation into the submitted job.
- Bind v2 dirty/source/config/schema/wrapper hashes and Run1 frozen hashes in the v2 manifest.
- Close the three Run1 Medium findings with v2 SLURM-published evidence.
- Add immutable discipline records for the four known local Python commands.

Forbidden:

- Do not change scientific thresholds, fixtures, expected case outcomes, solver target, rank thresholds, KKT tolerance, coverage threshold, or controls to pass.
- Do not reinterpret finite-VI probes or solver traces as acceptance.
- Do not open held/validation/test, MLLM, OCR, query labels, query embeddings, cache, teacher/head/reranker/key-selector artifacts.
- Do not use local Python results as PASS support.

## Discipline Audit

Current AGENTS.md requires compute through SLURM and states the main conversation itself must not execute chores. Run2 execution used four direct login-node Python commands according to the audit prompt:

1. preflight validator, first run;
2. preflight validator, second run;
3. post-run `py_compile`;
4. post-run `git_dirty_hash` import.

Execution documentation records one direct preflight command and its hash, and mentions `py_compile` only as a covered check. I found no complete execution-doc record for the second preflight validator, the post-run `py_compile`, or the post-run `git_dirty_hash` import.

Severity:

- High: direct login-node Python execution violated current AGENTS.md and cannot be used to support PASS.
- Medium: execution documentation is incomplete for the full set of local Python commands and must be supplemented with an immutable record: exact command, cwd, user, timestamp, stdout/stderr hash, produced file hash if any, reason, and explicit exclusion from scientific evidence.

## Gold And Scope Audit

The result-to-claim decision preserves the approved scope:

- only gold: `parent_video_binary_label`;
- no segment/frame/timestamp/span/localization/stance/target/mechanism/rationale gold;
- no MLLM/OCR/cache construction;
- no held/validation/test/query label/query embedding access;
- no teacher/head/reranker/key-selector route;
- no performance claim.

## Severity Register

| ID | Severity | Finding | Executable ruling |
|---|---|---|---|
| C1 | Critical | Run2 v1 published no artifact and no KKT/case evidence, so the intended claim is unsupported. | Run3 remains locked. |
| H1 | High | Four login-node Python executions violated the current SLURM discipline; local outputs cannot support PASS. | Require immutable discipline record; exclude local results from claim evidence. |
| M1 | Medium | `finite_vi_diagnostic` is assembled at case level but checked at manifest top level before publish. | Fix field assembly/order only; do not tune science. |
| M2 | Medium | Active producer/verifier/schema/wrapper are not contract-consistent. | Align v2 producer, semantic verifier, schemas, config, and wrapper before submission. |
| M3 | Medium | Run1 Medium findings are not closed by v1 because no verified Run2 artifact exists. | Close MCF-M1/MCF-M2/MCF-M3 with SLURM-published v2 evidence. |
| M4 | Medium | Execution documentation is incomplete for local Python discipline events. | Add immutable discipline supplement before v2 review. |
| L1 | Low | Tracker header still says current status is Run1 frozen, while row 2 records Run2 `FAIL_STOP`. | Prospective tracker amendment should make Run2 fail-stop status unambiguous. |

Severity counts: Critical 1, High 1, Medium 4, Low 1.

## Next Action

Proceed with `route=supplement` only:

1. Preserve all v1 failure evidence.
2. Prospectively amend plan/tracker for `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2`.
3. Add immutable discipline record for local Python violations.
4. Submit v2 only after independent code review confirms the minimal repair boundary.
5. After v2 completes or fails, run a new result-to-claim review before Run3.

