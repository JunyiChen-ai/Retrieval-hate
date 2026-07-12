# LB-SCGP Global-R2 M0 Run2 Fresh Result-to-Claim Review

Date: 2026-07-12

Reviewer boundary: fresh independent local result-to-claim review only. No subagents, workflows, model/API calls, SLURM submissions, experiments, GPU/training/performance work, validation/test work, MLLM/OCR work, Run2 repair, Run3, or code/config/schema/wrapper/artifact edits were performed. This file is the only new write.

Stale prior report provenance:

- `refine-logs/lb_scgp_global/M0_RUN2_RESULT_TO_CLAIM_REVIEW.md`
- SHA256: `12f42df893d90d6ef4a7759dbbb86e8f8d820375be9aae3a0b94010fda1335ef`
- Status: untrusted/stale provenance only. It is not used as the fresh judgment because it covers job `12902` but predates or omits the later job `12904` evidence.

## Structured Verdict

- intended_claim: The synthetic global projection / serialized H-metric normal-cone KKT gate is executable and independently verifiable for LB-SCGP Global-R2.
- claim_supported: no
- route: infrastructure_repair
- confidence: high
- repair_authorization: authorized
- repair_authorization_scope: authorized in principle only for one new non-overwriting v2 lineage; execution is not authorized by this report.
- execution_authorization: not_authorized
- plan_amendment_required: yes

## What Results Support

The evidence supports only these narrow procedural facts:

- Run1 `LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1` remains frozen, with artifact SHA256 `09b78682389f1c9774c9dffc43c759bceeec9d7f44eca1ce4cd626d0cd6d12da` and lock SHA256 `c6fbb49c3c8aba942da3f254c3c5f65d037002548fafb961154e6fa79a3e4ea7`.
- The old protected LB-SCGP scope independently recomputes to 278 files and manifest SHA256 `243e89b69b169b222dd97f9df092d511f823fb26201e91fd89cb581710940462`.
- Job `12902` was a v1 Run2 attempt using job name `lbscgp_global_r2_run2_synth_kkt`; `sacct` reports `FAILED`, exit `1:0`, elapsed `00:00:04`, 8 CPU, 64G, batch MaxRSS `5388K`.
- Job `12904` was a later v1 Run2 attempt using job name `lbscgp_global_r2_run2`; `sacct` reports `FAILED`, exit `1:0`, elapsed `00:00:01`, 8 CPU, 64G, batch MaxRSS `5420K`.
- Both failures are fail-closed infrastructure/interface failures before any published Run2 artifact or independent verification result.
- Current `artifacts/lb_scgp_global/v1/m0/synth_kkt` is absent; no v1 synth_kkt child artifacts or locks currently exist.

## What Results Do Not Support

The evidence does not support the intended executability claim:

- No v1 Run2 `manifest.json` was published.
- No v1 Run2 source manifest, access ledger, semantic verifier decision, publish lock, payload hash, case matrix, KKT payload, rank-tail audit, factor replay, Procrustes replay, injection matrix, or robust coverage report exists.
- No producer/verifier output from `12904` exists because it failed in validator preflight with `KeyError: 'payload_schema'`.
- No independently verifiable scientific output from `12902` exists because it failed before publication with `KeyError: 'finite_vi_diagnostic'`.
- Code presence is not evidence of KKT, rank, factor, solver, or mechanism behavior.

Neither `12902` nor `12904` may be interpreted as a scientific, optimization, numerical, KKT, rank, factor, mechanism, MLLM, OCR, dataset, or performance failure.

## Job Lineage Classification

### Job 12902

- Run ID intended: `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1`.
- Job name: `lbscgp_global_r2_run2_synth_kkt`.
- Log hashes:
  - stderr: `77ba892b49d7c6262bc0a6165188c173b8f0619a7cbbfe9fdcc881fc8ec5f71c`
  - stdout: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Failure point: older producer path `scripts/analysis/lb_scgp_global_r2_synth_kkt.py` reached `build_manifest`, then `verify_manifest`, and failed on missing top-level `manifest["finite_vi_diagnostic"]`.
- Classification: infrastructure artifact-shape/interface failure before publish.

### Job 12904

- Run ID intended: `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1`.
- Job name: `lbscgp_global_r2_run2`.
- Log hashes:
  - stderr: `93e8515cad0d89ec65d3a1844d497694324143b9d836bc304b732765f1ead306`
  - stdout: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Failure point: newer validator path `scripts/analysis/lb_scgp_global_r2_run2_validate.py` failed on config path key `cfg["paths"]["payload_schema"]`.
- Classification: infrastructure preflight config-schema-validator path-key/interface failure before producer or independent verifier.

### 12902 vs 12904

The two jobs are distinct attempts under the same v1 Run2 run ID and same intended v1 artifact namespace. Job `12902` preceded `12904`; the former used the older `run2_synth_kkt` path, while the latter used the newer `run2` path. Their coexistence violates, or at minimum consumes and ambiguates, the single-submit v1 Run2 authorization. The v1 Run2 lineage must not be retried again.

## Missing Evidence

Missing evidence required for the intended claim:

- Published non-overwriting manifest under a new reviewed lineage.
- Source manifest and access ledger.
- Publish locks.
- Independent semantic verification decision.
- Strict payload/schema validation result.
- Static contract matrix and negative tests before submission.
- Case metrics for FULL, REMOVE/null, SHUFFLE, NOISE, ambiguity/coverage, and required injection failures.
- Serialized H-metric normal-cone/KKT payload with stationarity, dual feasibility, complementarity, PSD sign convention, and finite-VI diagnostic-only status.
- Rank-tail `<= d`, factor, Procrustes, and nondegeneration evidence.
- Common-basis `orth_cap` and `M_Q=Q^T(G-I)Q/N` replay evidence.
- Zero-access counters in a published artifact.
- Fresh result-to-claim review after any v2 completion or failure.

## Failure Classification

failure_classification: infrastructure_preflight_and_artifact_interface_failure

Details:

- `12902`: artifact field placement/interface mismatch, `finite_vi_diagnostic` case-level vs top-level access before publication.
- `12904`: config/schema/validator/wrapper interface mismatch, `.paths.schema` and older wrapper fields present while validator expected `.paths.payload_schema`, `.paths.case_schema`, `.paths.cert_schema`, and newer wrapper path.
- No scientific threshold, fixture, KKT tolerance, solver, numerical result, rank audit, factorization, or mechanism result was tested to completion.

## Suggested Claim Revision

Do not claim Run2 executability or independent verifiability.

Supported revision:

> Run2-v1 was attempted twice under the same intended run ID and failed fail-closed at infrastructure/interface preflight or pre-publication stages. No synthetic KKT artifact or independent verification evidence was produced. Run1 remains frozen and old protected scope remains unchanged.

## Repair Authorization

repair_authorization: authorized

This is authorization in principle only, not execution authorization. Exact constraints:

1. Preserve all v1 evidence immutably and separately addressably, including stale report, `12902` logs, `12904` logs, `M0_SYNTH_KKT_IMPLEMENTATION.md`, `M0_SYNTH_KKT_EXECUTION.md`, tracker row, and all hashes.
2. Do not rerun or overwrite v1. The repair must use a new run ID, new config, new artifact namespace, new locks, and new lineage, e.g. `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2` and `artifacts/lb_scgp_global/v2/m0/synth_kkt/`.
3. The minimal permitted fix is limited to aligning config, schema, validator, wrapper, SLURM, and artifact-shape interfaces: path keys, wrapper path, schema path keys, finite-VI field location/name, and producer/verifier/schema expectations.
4. Independent static contract matrix and negative tests must pass before any repair SLURM submission.
5. Scientific thresholds, fixtures, expected decisions, KKT tolerances, movement/nondegeneration target, resource request, and intended claim must remain unchanged.
6. Solver/math/verifier semantics and hyperparameters must remain frozen for a pure infrastructure repair. Any scientific change requires a different prospectively reviewed lineage.
7. Current partial v2 scaffolding in the worktree, if present, is not reviewed execution evidence and is not execution authorization.
8. If v2 fails or completes, a new result-to-claim review is required before any Run3 consideration.

## Plan Amendment Required

plan_amendment_required: yes

Before execution, `EXPERIMENT_PLAN.machine.json` and `EXPERIMENT_TRACKER.md` must be prospectively amended to introduce the v2 run ID, namespace, config, locks, dependencies, provenance, and preservation of both v1 failed attempts. That amendment must receive a fresh independent plan/code review before any SLURM submission. This report does not authorize execution.

## No-Segment / Zero-Access Audit

Pass for the available evidence, with the limitation that no Run2 artifact counters were published.

- Only gold supervision remains `parent_video_binary_label`.
- No segment, frame, timestamp, span, localization, stance, target, mechanism, rationale, or fragment gold exists or is used.
- Jobs `12902` and `12904` failed before any train, held, validation, test, teacher, cache, MLLM, OCR, network/model, GPU, `query_z`, or `query_labels` access could become artifact evidence.
- Run1 access counters remain zero in the frozen contract artifact.
- The v1 synth_kkt namespace is absent, so there are no nonzero Run2 access counters or data-route artifacts to classify.

## Explicit Next Boundary

Next boundary: prospectively amend plan-machine/tracker for one new v2 infrastructure repair lineage, then obtain a fresh independent plan/code review. Only after that review may a separate executor consider a v2 SLURM submission. The current report is not execution authorization.

Run3 remains locked. M1 cache, MLLM/OCR, GPU, training, validation/test, performance, realbank resource, and all later runs remain locked.

## Normalized Decision

- claim_supported: no
- route: infrastructure_repair
- confidence: high
- 12902_classification: v1 infrastructure artifact-shape/interface failure before publish
- 12904_classification: v1 infrastructure config-schema-validator preflight failure before producer/verifier
- repair_boundary: v2 authorized in principle only; execution not authorized until prospective plan-machine/tracker amendment and fresh independent review
- Run3: locked
