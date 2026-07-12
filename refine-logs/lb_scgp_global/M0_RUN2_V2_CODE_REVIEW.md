# M0 Run2-v2 Fresh Static Code Review

Date: 2026-07-12

Reviewer boundary: fresh independent static code review only. I read `AGENTS.md` first. I did not run Python, Python imports, `py_compile`, tests, data/model code, conda commands, SLURM, experiments, MLLM/OCR/API/model calls, GPU/training/evaluation, or validation/test data/cache inspection. Shell use was limited to static reads and allowed checks: `rg`, `sed`/`nl`, `jq`, `awk`, `bash -n`, `sha256sum`, `find`, `ls`, `wc`, `git status`, and `git diff`. The only file write is this report.

This review does not authorize SLURM execution. It decides only whether the implementation is safe to request a separate execution-authorization review for exactly one future SLURM validation job.

## Verdict

FAIL_BLOCKED_FOR_EXECUTION

Severity counts:

| Severity | Count |
|---|---:|
| Critical | 0 |
| High | 2 |
| Medium | 1 |
| Low | 1 |

0 Critical / 0 High is not achieved.

## Executive Summary

Run2-v2 is not safe to advance to a separate execution-authorization request. The hash, schema, no-clobber, wrapper, and top-level KKT-verifier fixes are substantially improved, but the current producer is statically inconsistent with its own rank gate: it constructs positive-definite FULL/non-REMOVE Gram matrices with `N > d` while `d=3`, so the rank/factor audit must fail before publication. This is a deterministic implementation blocker, not a runtime uncertainty.

There is also an independent-verifier gap: case-level serialized residual, movement, rank, factor, and robust-coverage fields are schema/hash-checked but not semantically recomputed and compared for every case. Top-level FULL KKT normal-family blocks are now strongly checked, but the case matrix can still carry internally rehashed semantic drift in several serialized fields.

No performance evidence exists. No segment/frame/timestamp/span/localization/stance/target/mechanism/rationale/fragment gold is assumed; only synthetic binary labels are used in Run2-v2 fixtures, while the project-level only gold remains `parent_video_binary_label`. Run3/M1 remain locked. This review itself does not authorize SLURM.

## Findings

### H5: Producer rank contract is statically impossible for the FULL case

Severity: High.

Evidence:

- The config fixes `d=3` for the rank/factor contract (`configs/lb_scgp_global_r2/m0_synth_kkt_v2.json:109-110`).
- The producer builds the primary FULL case with 10 replicas and passes `d` from the config (`scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py:137-145`, `:277-278`).
- For non-REMOVE cases, the common helper sets `g0 = I_N` and calls `structural_forced_solution` (`scripts/analysis/lb_scgp_global_r2_run2_v2_common.py:890-899`).
- `structural_forced_solution` returns only after `eig_min >= 1e-5`, so the resulting `G_star` is positive definite (`scripts/analysis/lb_scgp_global_r2_run2_v2_common.py:624-638`). For the FULL case, this is a 10x10 positive-definite matrix, hence rank 10.
- The rank/factor helper counts rank by eigenvalues above an approximately `1e-7` threshold and returns failure immediately when `rank > d` (`scripts/analysis/lb_scgp_global_r2_run2_v2_common.py:677-687`).
- Case `kkt_status` is `PASS` only if `rank_audit["status"] == "PASS"` (`scripts/analysis/lb_scgp_global_r2_run2_v2_common.py:991-992`), and the producer refuses to publish a failing candidate (`scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py:381-384`).

Impact:

A future authorized SLURM run is expected to fail before publishing the manifest. This blocks execution readiness even though the failure is fail-closed. The same problem applies to other non-REMOVE cases with `N > d` (`scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py:156-191`).

Required fix:

Make the synthetic fixtures and rank contract mathematically consistent without changing science post hoc. For example, construct a PSD/unit-diagonal `G_star` with rank `<= d` and nontrivial movement, or set fixture sizes/contracts according to the already approved science boundary after an authorized plan/fix pass. A fresh static review is required afterward.

### H6: Independent verifier does not semantically recompute all serialized case-matrix fields

Severity: High.

Evidence:

- The verifier replays consensus/operator/primal hashes for each case, but `verify_case_hashes` only checks operator, primal, and case-payload hashes (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:489-511`).
- The per-case loop checks finite-VI, acceptance path, `kkt_status`, replay shape/operator fields, and hashes (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:847-853`). It does not recompute and compare each case's serialized `movement_metrics`, `primal_residuals`, `rank_audit`, `factor_replay`, or `robust_coverage`.
- The top-level movement block is only checked for equality to the FULL case's serialized block (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:589-590`), while actual movement is used only for threshold gating (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:868-883`). The serialized movement values can drift if the FULL case and top-level block are changed together and hashes are recomputed.
- The verifier recomputes rank/factor for the top-level FULL matrix (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:887-892`), but it does not compare the recomputed audit to the serialized `full_case["rank_audit"]` or every other case's audit/factor fields.
- The rank-failure probe only checks `factor_returned_null` and status (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:896-899`), not the serialized eigenvalue/audit fields required by the schema.

Impact:

After recomputing `case_payload_sha256`, `case_matrix_sha256`, and `payload_sha256`, a tampered manifest could preserve KKT acceptance while carrying false case-level metrics/audits. This weakens the case matrix as independent evidence and does not satisfy the requested "every serialized block" verification standard.

Required fix:

For every case, recompute and compare serialized `movement_metrics`, `primal_residuals`, `rank_audit`, `factor_replay`, and `robust_coverage` against independently reconstructed arrays and labels. Also compare the rank-failure probe's serialized audit values to an independent recomputation. Add targeted injections with refreshed hashes for these fields.

### M2: `jsonschema` availability remains a static environment uncertainty, but fail-closed

Severity: Medium.

Evidence:

- Producer-side real schema validation imports `jsonschema` and raises if unavailable (`scripts/analysis/lb_scgp_global_r2_run2_v2_common.py:176-185`), then validates the payload and each case before publish (`scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py:381-382`).
- The independent verifier repeats local schema validation with local case-schema ref resolution and raises if `jsonschema` is unavailable (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:165-201`, `:814-815`).
- This review was not authorized to run Python imports, so dependency availability in `HateVideo` remains untested.

Impact:

Missing `jsonschema` cannot produce a false PASS; both producer and verifier fail closed. It can, however, consume the future single SLURM validation attempt with an environment failure.

Required fix:

Before any future execution request, either document `jsonschema` as installed in the `HateVideo` environment through an authorized SLURM preflight path, or keep the risk explicitly accepted as fail-closed infrastructure risk.

### L2: `EXPERIMENT_PLAN.md` still contains conservative pre-fix wording

Severity: Low.

Evidence:

- The tracker states amendment review passed, fix/freeze is complete, and Run2-v2 remains locked for code review/execution authorization (`refine-logs/lb_scgp_global/EXPERIMENT_TRACKER.md:4`, `:16`, `:99-104`).
- The machine plan also records implementation fix/freeze complete, `ready_for_execution=false`, and `ready_for_fresh_independent_v2_code_review=true` (`refine-logs/lb_scgp_global/EXPERIMENT_PLAN.machine.json:612-624`, `:4171-4180`, `:4198-4235`).
- The Markdown plan still says Run2-v2 is locked pending independent amendment review and describes the amendment as authorizing only amendment review (`refine-logs/lb_scgp_global/EXPERIMENT_PLAN.md:7`, `:141-156`, `:248-250`).

Impact:

This is conservative and does not grant execution. It can confuse future boundary review unless the next authorized documentation update aligns the Markdown plan with tracker/machine state.

Required fix:

In a separate authorized planning/doc pass, update the Markdown status text while preserving `ready_for_execution=false` and separate execution authorization.

## Closure Matrix For Prior Findings

| Prior finding | Closure status | Evidence |
|---|---|---|
| H1 stale/unenforced hashes | Closed for current reviewed files. Config authoritative hashes verify; amendment machine/hash/review artifacts are bound; config does not self-declare its own fixed hash. | `configs/lb_scgp_global_r2/m0_synth_kkt_v2.json:33-57`, `:82-91`; `scripts/analysis/lb_scgp_global_r2_run2_v2_common.py:779-845`; `scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:699-812`. |
| H2 missing JSON Schema validation | Closed in design, with M2 dependency risk. Producer and verifier both perform real JSON Schema validation and per-case validation, with local ref resolution and fail-closed missing dependency behavior. | `scripts/analysis/lb_scgp_global_r2_run2_v2_common.py:176-218`; `scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py:381-382`; `scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:165-201`, `:814-815`. |
| H3 incomplete hash/injection semantics | Mostly closed. Top-level file/source/config/schema/Run1/implementation/operator/case-matrix hashes are recomputed; injections refresh payload hashes. H6 remains for case-level semantic fields after hashes are recomputed. | `scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:786-812`, `:914-923`, `:925-990`. |
| H4 incomplete serialized KKT checks | Partially closed but not execution safe. Top-level FULL normal families, signs, stationarity, dual feasibility, complementarity, PSD, and finite-VI checks are now strong; case-matrix serialized fields and the rank contract remain blocked by H5/H6. | Top-level: `scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:546-681`. Blockers: H5/H6 above. |
| M1 dirty/no-clobber lock mismatch | Closed with residual unavoidable hard-kill risk. Allowed outputs include lock files; producer cleans partial child publishes; wrapper cleans all prospective outputs unless `decision == PASS`; signal traps exist. | `configs/lb_scgp_global_r2/m0_synth_kkt_v2.json:18-31`; `scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py:230-238`, `:385-396`; `scripts/wrappers/lb_scgp_global_r2_run2_v2.sh:12-31`, `:38-41`, `:75-76`. |
| L1 stale tracker/machine status | Closed for tracker and machine. A new Low doc-only stale wording issue remains in `EXPERIMENT_PLAN.md` as L2. | `refine-logs/lb_scgp_global/EXPERIMENT_TRACKER.md:4`, `:16`; `refine-logs/lb_scgp_global/EXPERIMENT_PLAN.machine.json:612-624`, `:4171-4180`. |

## Static Checks And Observations

- Current reviewed implementation hashes:
  - `configs/lb_scgp_global_r2/m0_synth_kkt_v2.json`: `c9bd928c01dcf49fe385a751ea5eb9bed107fd4c95970ce77015eaca59dc3465`
  - `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_payload_v2.schema.json`: `3d3f9bebba166b7ce0c270490ab11a8def241fbd2f93fc832efe451673219ce3`
  - `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_case_v2.schema.json`: `df3616ff50c54dc756bb600c6ef26626afafe070678450d10ca73b9ff83ddcac`
  - `scripts/analysis/lb_scgp_global_r2_run2_v2_common.py`: `36b043e54568fd286337e4650d0c52d7d7693da77c78ff9286bed7f4e0ad12c8`
  - `scripts/analysis/lb_scgp_global_r2_run2_v2_validate.py`: `48099a580fa91672883320829e22469c069c5500cd38ec411f33ce4768633ebd`
  - `scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py`: `957d02dcc78302e8091583b9dffdf5150d2e25facaa38da537c6011733889aca`
  - `scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py`: `b3effff578b5ccdb0e56f7e4a261eb8c164783b65c0fcbc0f53b08589164c75b`
  - `scripts/wrappers/lb_scgp_global_r2_run2_v2.sh`: `14eb036f17b77037fd89624c0f9f7487432fa476da540b0fbc5f503719320716`
  - `scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v2.sbatch`: `f91462303e4b28eb0722d546f135c00c2aeacf56151e27d5421f211520ba94bf`
- `sha256sum -c` passed for `EXPERIMENT_PLAN_HASHES.sha256`, `M0_RUN2_V2_PLAN_AMENDMENT_HASHES.sha256`, and `M0_RUN2_V2_PLAN_AMENDMENT_REVIEW_HASHES.sha256`.
- Config-bound authoritative inputs and Run1 frozen hashes passed `sha256sum -c`. Declared validation/test provenance hashes were listed but not opened, consistent with the not-opened contract.
- Old protected scope remains 278 files with aggregate `243e89b69b169b222dd97f9df092d511f823fb26201e91fd89cb581710940462`.
- Run1 artifact and lock hashes remain `09b78682389f1c9774c9dffc43c759bceeec9d7f44eca1ce4cd626d0cd6d12da` and `c6fbb49c3c8aba942da3f254c3c5f65d037002548fafb961154e6fa79a3e4ea7`.
- `bash -n` passed for the wrapper and sbatch. No `--time` directive is present; the sbatch requests 8 CPU, 64 GB, 0 GPU, activates `HateVideo`, and calls the wrapper (`scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v2.sbatch:2-24`).
- `artifacts/lb_scgp_global/v2` remains absent. No v2 artifact was created by this review.
- Git status is dirty. The implementation/review files are untracked; the only tracked diff observed is unrelated `slurm/logs/disk_guard.log`. I did not revert or modify unrelated state.

## Negative Injection Review

Static inspection confirms the declared injection keys are present and the verifier recomputes `payload_sha256` after each mutation (`scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py:914-923`, `:925-990`). This fixes the prior stale-hash-only rejection problem. Some mutations are expected to be rejected at the schema layer before reaching later semantic checks, which is acceptable for fields whose schema has a strict `const` or required-key contract. No injection was executed in this review.

H6 remains because there are no refreshed-hash injections for bogus case-level `movement_metrics`, `primal_residuals`, `rank_audit`, `factor_replay`, or robust-coverage content.

## Remaining Risks And Required Fixes

1. Fix the rank/fixture inconsistency in H5. The current future job is statically expected to fail.
2. Extend the independent verifier to recompute and compare every serialized case-level metric/audit field in H6.
3. Add targeted refreshed-hash negative injections for case-level movement, residual, rank, factor, robust coverage, and rank-failure audit drift.
4. Resolve or explicitly accept the fail-closed `jsonschema` dependency uncertainty through an authorized SLURM-only preflight path.
5. Align the Markdown plan status text in a separate authorized doc pass.

## Required Statements

- No performance evidence exists, and no performance claim is possible from this review.
- No segment-level or fragment-level gold is assumed. The only project gold supervision remains `parent_video_binary_label`; Run2-v2 uses synthetic fixture labels only.
- Run3, M1, MLLM/cache work, validation/test work, training, and realbank remain locked.
- This review does not authorize SLURM execution. A pass would only have allowed a separate execution-authorization review; this review fails.

Report SHA256: should be computed externally after file creation; it is not embedded to avoid a self-referential claim.
