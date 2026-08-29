# M0 Run2 Local Python Discipline Record

Date: 2026-07-12

Immutable record policy: append only. Do not edit or remove prior entries from this file. This record is discipline evidence only and is excluded from scientific evidence for C1/C2.

## Binding Hashes

- Fresh result-to-claim review: `refine-logs/lb_scgp_global/M0_RUN2_RESULT_TO_CLAIM_REVIEW_FRESH.md`, SHA256 `b02e5a0f7839e8b215c15197f6f120e3bebfe498ab89d2120c26b50585707a0c`.
- Stale/provenance result-to-claim review: `refine-logs/lb_scgp_global/M0_RUN2_RESULT_TO_CLAIM_REVIEW.md`, SHA256 `12f42df893d90d6ef4a7759dbbb86e8f8d820375be9aae3a0b94010fda1335ef`.
- job12902 stderr: `slurm/logs/lbscgp_global_r2_run2_synth_kkt_12902.err`, SHA256 `77ba892b49d7c6262bc0a6165188c173b8f0619a7cbbfe9fdcc881fc8ec5f71c`.
- job12902 stdout: `slurm/logs/lbscgp_global_r2_run2_synth_kkt_12902.out`, SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- job12904 execution doc: `refine-logs/lb_scgp_global/M0_SYNTH_KKT_EXECUTION.md`, SHA256 `31481cb10808e9a6ce81754c47d18351079bf84b602088618b0ecdd341c80333`.
- job12904 stderr: `slurm/logs/lbscgp_global_r2_run2_12904.err`, SHA256 `93e8515cad0d89ec65d3a1844d497694324143b9d836bc304b732765f1ead306`.
- job12904 stdout: `slurm/logs/lbscgp_global_r2_run2_12904.out`, SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Known Login-Node Python Uses

The result-to-claim review identifies four login-node Python discipline violations. Exact shell history is not fully recoverable from the filesystem. Where exact command/timestamp/stdout/stderr are not recoverable, this record says so explicitly.

### 1. Preflight Validator, Recoverable Temp Record

- Status: violation.
- Exact outer command: unknown. Inferred command class only: local Python invocation of `scripts/analysis/lb_scgp_global_r2_synth_kkt_validate.py` for `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1` preflight.
- CWD: `/data/jehc223/RGCL`.
- User: `jehc223`.
- Recoverable timestamp: `/tmp/lbscgp_global_r2_run2_preflight.json` mtime `2026-07-12 12:46:01.083889792 +1200`.
- Recoverable temp file: `/tmp/lbscgp_global_r2_run2_preflight.json`, size `17935`, SHA256 `9384d6525cdacb0e8369399a6c835c52c3cd8b50db9ff17131eb1df3d2f97cd1`.
- Recoverable JSON fields: schema `lb_scgp_global_r2_run2_synth_kkt_validation_v1`, phase `preflight`, run id `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1`, status `PASS`, validator SHA256 `1511a861ca3fc2ee024d890b21fec29b312e278dab8b3d040107804d894dbdda`.
- Recoverable nested command evidence includes a login-node Python subprocess: `/data/jehc223/miniconda3/envs/HateVideo/bin/python -m py_compile scripts/analysis/lb_scgp_global_r2_common.py scripts/analysis/lb_scgp_global_r2_synth_kkt.py scripts/analysis/lb_scgp_global_r2_synth_kkt_validate.py`.
- Outer stdout/stderr: unknown; not recoverable from the temp JSON.
- Why violation: project policy requires all Python validation/compute to run inside SLURM. This preflight was local-login validation.
- Scientific evidence status: excluded. The PASS preflight record is not used as evidence for C1/C2.

### 2. Second Preflight Validator

- Status: violation identified by `M0_RUN2_RESULT_TO_CLAIM_REVIEW.md`.
- Exact command: unknown; not recoverable from available files.
- CWD: unknown.
- User: unknown.
- Timestamp: unknown.
- Temp file/stdout/stderr/hash: not recovered by `/tmp` scan for Run2-related files.
- Why violation: project policy requires Python validation to run inside SLURM.
- Scientific evidence status: excluded. No result from this command is used as evidence for C1/C2.

### 3. Post-Run `py_compile`

- Status: violation identified by `M0_RUN2_RESULT_TO_CLAIM_REVIEW.md`.
- Exact command: unknown; not recoverable as shell history or standalone stdout/stderr.
- CWD: unknown.
- User: unknown.
- Timestamp: unknown.
- Recoverable related command class: `python -m py_compile` over Run2-v1 implementation files, but the exact post-run command instance is not recoverable.
- Temp file/stdout/stderr/hash: not recovered.
- Why violation: even syntax validation by Python must be inside the authorized SLURM wrapper for this project.
- Scientific evidence status: excluded. It is not used to support any artifact or claim.

### 4. Post-Run `git_dirty_hash` Import

- Status: violation identified by `M0_RUN2_RESULT_TO_CLAIM_REVIEW.md`.
- Exact command: unknown; not recoverable as shell history or standalone stdout/stderr.
- CWD: unknown.
- User: unknown.
- Timestamp: unknown.
- Temp file/stdout/stderr/hash: not recovered.
- Why violation: importing project Python on the login node is still login-node Python execution and is outside the authorized SLURM boundary.
- Scientific evidence status: excluded. It is not used to support any artifact or claim.

## Related Recoverable Non-Evidence Files

The old protected hash manifests were produced before/around Run2-v1 and are recorded only as provenance:

- `/tmp/lb_scgp_global_r2_old_protected_pre.sha256`, mtime `2026-07-12 12:09:45.468039471 +1200`, SHA256 `243e89b69b169b222dd97f9df092d511f823fb26201e91fd89cb581710940462`.
- `/tmp/lb_scgp_global_r2_old_protected_post.sha256`, mtime `2026-07-12 12:17:26.451741451 +1200`, SHA256 `243e89b69b169b222dd97f9df092d511f823fb26201e91fd89cb581710940462`.
- `/tmp/lb_scgp_global_r2_old_protected_final.sha256`, mtime `2026-07-12 12:19:04.749833566 +1200`, SHA256 `243e89b69b169b222dd97f9df092d511f823fb26201e91fd89cb581710940462`.

These files do not convert Run2-v1 into successful evidence.

## Discipline Rule For Run2-v2

All future Python validation, production, semantic verification, source hashing, dirty binding, and artifact checks for `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2` must occur inside the single future SLURM wrapper. Pre-submission checks are limited to pure shell/static checks such as `bash -n`, `jq`, `rg`, `git diff`, and `sha256sum`.
