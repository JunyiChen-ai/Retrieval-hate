# M0 Run2-v2 Implementation Fix/Freeze

Date: 2026-07-12

Boundary: implementation fix/freeze only. I read `AGENTS.md` and `M0_RUN2_V2_IMPLEMENTATION_AUDIT.md` completely first. I did not run Python, import Python packages, run `py_compile`, submit or monitor SLURM, run experiments, read data/validation/test content, call MLLM/OCR/model/network tools, use GPU/training/performance, or produce artifacts under `artifacts/lb_scgp_global/v2/`.

Execution remains locked. `ready_for_execution=false`; execution-authorized budget remains 0.

## Changed Files

| Path | SHA256 after fix |
|---|---|
| `configs/lb_scgp_global_r2/m0_synth_kkt_v2.json` | `c9bd928c01dcf49fe385a751ea5eb9bed107fd4c95970ce77015eaca59dc3465` |
| `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_payload_v2.schema.json` | `3d3f9bebba166b7ce0c270490ab11a8def241fbd2f93fc832efe451673219ce3` |
| `scripts/analysis/lb_scgp_global_r2_run2_v2_common.py` | `36b043e54568fd286337e4650d0c52d7d7693da77c78ff9286bed7f4e0ad12c8` |
| `scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py` | `957d02dcc78302e8091583b9dffdf5150d2e25facaa38da537c6011733889aca` |
| `scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py` | `b3effff578b5ccdb0e56f7e4a261eb8c164783b65c0fcbc0f53b08589164c75b` |
| `scripts/wrappers/lb_scgp_global_r2_run2_v2.sh` | `14eb036f17b77037fd89624c0f9f7487432fa476da540b0fbc5f503719320716` |
| `refine-logs/lb_scgp_global/EXPERIMENT_TRACKER.md` | `b0f0dfc9bffc51fd3fcd80eccefda2144e24be2d09d22d4d6a26050c7df471dd` |
| `refine-logs/lb_scgp_global/EXPERIMENT_PLAN.machine.json` | `b14d9b92bfb60aa28c18477afa505b048306d5c710991d3b59ae9399a0ae9574` |
| `refine-logs/lb_scgp_global/EXPERIMENT_PLAN_HASHES.sha256` | `997bf908a0c45c5b23c424fd95924e45d305af2132085b7a03f21e047ea20a99` |

Report SHA256 is intentionally not embedded to avoid self-reference; compute after file creation.

## Unchanged Protected Files

| Path | SHA256 |
|---|---|
| `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_case_v2.schema.json` | `df3616ff50c54dc756bb600c6ef26626afafe070678450d10ca73b9ff83ddcac` |
| `scripts/analysis/lb_scgp_global_r2_run2_v2_validate.py` | `48099a580fa91672883320829e22469c069c5500cd38ec411f33ce4768633ebd` |
| `scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v2.sbatch` | `f91462303e4b28eb0722d546f135c00c2aeacf56151e27d5421f211520ba94bf` |
| `artifacts/lb_scgp_global/v1/m0/contract_freeze.json` | `09b78682389f1c9774c9dffc43c759bceeec9d7f44eca1ce4cd626d0cd6d12da` |
| `artifacts/lb_scgp_global/v1/m0/contract_freeze.json.publish.lock` | `c6fbb49c3c8aba942da3f254c3c5f65d037002548fafb961154e6fa79a3e4ea7` |

## Audit Finding Closure

| Finding | Closure |
|---|---|
| H1 stale/unenforced hashes | Refreshed config authoritative hashes for current plan/tracker/machine/hash files, bound amendment md/machine/hash artifacts, amendment review, and implementation audit; removed unenforced `v2_declared_source_hashes_without_config`; config hash is emitted/recomputed at runtime and not self-declared. |
| H2 missing JSON Schema validation | Added fail-closed `jsonschema` validation in producer before publish and independent verifier before PASS, with local v2 case-schema reference resolution and per-case validation. |
| H3 incomplete hash verification | Independent verifier now recomputes/asserts payload, config, payload schema, case schema, Run1 artifact/lock, implementation, source/access, case matrix, top-level operator, and every case operator/primal/case payload hash. Injections recompute `payload_sha256`; case mutations refresh case and case-matrix hashes. |
| H4 incomplete serialized KKT checks | Independent verifier now validates metric, affine/box/SOC/PSD/halfspace normal blocks, stationarity, dual feasibility, complementarity per family, sign conventions, normal presence/residuals, finite-VI diagnostics on every case, and acceptance/status fields. |
| M1 dirty/no-clobber mismatch | Config allowed outputs now include persistent `.publish.lock` files. Wrapper cleans all prospective JSON/lock outputs on failure or signal and marks complete only after semantic verification JSON says PASS. Producer removes outputs it created if a later publish step fails. |
| L1 stale tracker/machine status | Tracker and machine status now record amendment review passed and implementation fix/freeze complete, with fresh independent code review and execution authorization still locked. |

## Static Contract Matrix

| Contract | Static check/future enforcement |
|---|---|
| SLURM-only future execution | Wrapper and sbatch remain the only execution path; sbatch is 8 CPU / 64 GB / 0 GPU / `HateVideo` / no `--time`. |
| No local execution evidence | This pass used no Python, no SLURM, no artifact generation, no data/val/test, no model/network calls. |
| Schema contract | Producer and independent verifier both validate the manifest and every case against strict v2 schemas before publish/PASS. |
| Hash contract | Runtime recomputation binds config, schemas, Run1 artifact/lock, implementation files, source manifest, access ledger, case matrix, and case-level hashes. |
| KKT-only acceptance | `acceptance_path` remains `serialized_h_metric_normal_cone_kkt`; finite-VI fields are non-accepting diagnostics only. |
| Gold isolation | Emitted v2 payload keeps only `parent_video_binary_label`; segment gold fields remain false and all forbidden counters stay zero. |
| Dirty/no-clobber | Future successful JSON files and persistent publish locks are declared; failed/signal exits remove prospective outputs. |
| Downstream locks | Run3/M1 remain locked until v2 PASS plus fresh independent artifact review; execution budget remains 0. |

## Future Single-SLURM Negative Tests

The future single SLURM job must execute `scripts/wrappers/lb_scgp_global_r2_run2_v2.sh`, which runs validator, producer, and independent verifier. The verifier declares and executes these expected rejections with recomputed payload hashes:

- `nan_overflow`
- `perturbed_artifact_source_operator_hash`
- `invalid_extra_missing_schema_fields`
- `wrong_dual_sign`
- `incomplete_cone_family`
- `forbidden_path`
- `rank_failure`
- `finite_vi_only_attempted_acceptance`
- `identity_no_movement_claims_full`
- `malformed_normal_residual`
- `malformed_normal_presence`
- `malformed_complementarity`
- `malformed_dual_status`
- `malformed_stationarity_status`
- `malformed_finite_vi_diagnostic`

No local Python evidence exists for these tests in this fix/freeze pass.

## Pure Shell Checks Run

| Check | Result |
|---|---|
| `jq -e` on config, payload schema, case schema, machine plan | PASS |
| `bash -n` on wrapper and sbatch | PASS |
| `sha256sum -c refine-logs/lb_scgp_global/EXPERIMENT_PLAN_HASHES.sha256` | PASS |
| `sha256sum -c M0_RUN2_V2_PLAN_AMENDMENT_HASHES.sha256` | PASS |
| `sha256sum -c M0_RUN2_V2_PLAN_AMENDMENT_REVIEW_HASHES.sha256` | PASS |
| `rg` stale v2 source-hash/pending-amendment/active forbidden-gold checks | PASS/no stale active v2 match |
| `git diff --check` on touched tracked paths | PASS |
| `git diff --check --no-index /dev/null <file>` spot checks for untracked touched files | PASS/no whitespace diagnostics |

## Hash Design Note

The config does not declare its own SHA256 and does not declare fixed source-file hashes for implementation files. The producer emits the exact config hash by reading the config file at runtime, and both producer/source manifest and independent verifier recompute implementation hashes from `implementation_files`. The fix/freeze report is also excluded from config hash bindings to avoid a circular report-config self-hash. Its SHA256 must be reported externally after file creation.

## Decision

- implementation_fix_freeze_status: COMPLETE
- ready_for_fresh_independent_code_review: true
- ready_for_execution: false
- execution_authorization: not_authorized
- execution_authorized_remaining_budget: 0
- run3_unlock: not_authorized
