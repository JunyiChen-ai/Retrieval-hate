# LB-SCGP Global-R2 M0 Contract Freeze Execution

Date: 2026-07-12

## Run

- Run ID: `LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1`
- SLURM job: `12901`
- SLURM terminal status: `COMPLETED`, exit `0:0`
- Run1 state: `FROZEN`
- Command: `sbatch scripts/slurm/lb_scgp_global_r2_m0_contract_freeze.sbatch`
- Resources: 4 CPU, 16 GB, 0 GPU, no `--time`, `HateVideo`

## Artifacts

- Artifact: `artifacts/lb_scgp_global/v1/m0/contract_freeze.json`
- Artifact file SHA256: `09b78682389f1c9774c9dffc43c759bceeec9d7f44eca1ce4cd626d0cd6d12da`
- Payload SHA256: `57f935cfa6ff22f81ec726eba9e0000d76f95bf93575b7539b78ba4d7c5bde53`
- Publish lock: `artifacts/lb_scgp_global/v1/m0/contract_freeze.json.publish.lock`
- Publish lock SHA256: `c6fbb49c3c8aba942da3f254c3c5f65d037002548fafb961154e6fa79a3e4ea7`

## Validation

Run1 validator status: `PASS`.

SLURM-side checks:

- `jq -e` on config, machine plan, and both schemas.
- `bash -n` on wrapper and sbatch.
- `python -m py_compile` on all new Global-R2 Python files.
- `git diff --check -- refine-logs/lb_scgp_global/EXPERIMENT_TRACKER.md`.
- New-file trailing whitespace scan.

## Hashes

- Config SHA256: `5111c2d6d74c745afe35a7067566b531714ce6482df3f5b9469a442486146868`
- Implementation manifest SHA256: `ac73a23cc3d7cae72b17e13831a487491fbe6e4867c9f992e6d789b77eb3e072`
- Validator SHA256: `061cf80ae532e268b623050db5b39c47b9f0858f0a9591d94f8404e11e778fd2`
- Wrapper SHA256: `1eb342475c63df5d16bf570c82de465803652e4a6157444ea02294b58fa9596d`
- SLURM wrapper SHA256: `ccbade355239d3c313b70ed55a2907f7a2d51716a62b809e835ec9ad1441d882`
- Restricted cert schema SHA256: `4d3f1663e633c30ae58e35c0feddaa2fa9bbedba279cdbe6f38ecc35d761f22f`
- Contract schema SHA256: `d6a22233ec2ad028f1fdf8a0315641a30a76e1b3a96249615c48161b3d890105`
- Dirty-tree policy hash in artifact: `e4087426cd2d1d43ab4d7c950d7ef9f4077580766bfae3d03e0b2c17b21524d9`

## Isolation Proof

- Nonzero access counters: none.
- MLLM calls: `0`.
- OCR calls: `0`.
- Held/validation/test/cache/certificate/compiler-target/teacher/head/reranker/key-selector counters: `0`.
- Segment gold exists: `false`.
- Segment gold used: `false`.
- `query_z` reads: `0`.
- `query_labels` reads: `0`.
- Validation/test hashes were recorded as declared plan provenance only and were not opened by Run1.
- Train provenance hashed: `data/gt/MHC/train.jsonl`, `data/gt/MHC_zh/train.jsonl`.

## Protected Old Hashes

Protected old scope: `configs/lb_scgp`, `artifacts/lb_scgp`, `refine-logs/lb_scgp`, old `scripts/analysis/lb_scgp_*.py`, and old `scripts/slurm/lb_scgp_*.sbatch`, excluding new `lb_scgp_global_r2_*`.

- Pre snapshot count: `278`
- Post snapshot count: `278`
- Pre snapshot manifest SHA256: `243e89b69b169b222dd97f9df092d511f823fb26201e91fd89cb581710940462`
- Post snapshot manifest SHA256: `243e89b69b169b222dd97f9df092d511f823fb26201e91fd89cb581710940462`
- Byte comparison: identical.

## Boundary

This is only a contract freeze. It is not M0 success and it is not a performance claim. The next boundary is a fresh independent code+freeze audit.
