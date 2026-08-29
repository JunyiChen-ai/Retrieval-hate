# LB-SCGP Global-R2 M0 Synth KKT Execution

Date: 2026-07-12

## Run

- Run ID: `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1`
- SLURM job: `12904`
- Command: `sbatch scripts/slurm/lb_scgp_global_r2_m0_synth_kkt.sbatch`
- Resources: 8 CPU, 64 GB, 0 GPU, no `--time`, `HateVideo`
- Terminal status: `FAILED`
- Exit code: `1:0`
- Elapsed: `00:00:01`
- Batch MaxRSS: `5420K`

## Failure

Run2 failed in the preflight validator before producer execution:

```text
KeyError: 'payload_schema'
```

The submitted config exposed `.paths.schema` instead of the validator-expected `.paths.payload_schema`. No Run2 artifact directory was created.

## Artifact Audit

- Main artifact `artifacts/lb_scgp_global/v1/m0/synth_kkt/manifest.json`: absent.
- Source manifest: absent.
- Access ledger: absent.
- Semantic verifier decision: absent.
- Run2 publish locks: absent.
- Producer ran: no.
- Independent verifier ran: no.
- KKT/rank/factor/case matrices: not produced.

## Access And Isolation

The job failed before any producer/verifier path that could read data. No train, held, validation, test, teacher, cache, MLLM, OCR, network/model, GPU, `query_z`, or `query_labels` access occurred in Run2. Segment gold remains absent and unused.

## Run1 And Old-Protected Proof

Run1 frozen files remained unchanged:

- Run1 artifact: `09b78682389f1c9774c9dffc43c759bceeec9d7f44eca1ce4cd626d0cd6d12da`
- Run1 lock: `c6fbb49c3c8aba942da3f254c3c5f65d037002548fafb961154e6fa79a3e4ea7`
- Run1 config: `5111c2d6d74c745afe35a7067566b531714ce6482df3f5b9469a442486146868`
- Run1 cert schema: `4d3f1663e633c30ae58e35c0feddaa2fa9bbedba279cdbe6f38ecc35d761f22f`
- Run1 contract schema: `d6a22233ec2ad028f1fdf8a0315641a30a76e1b3a96249615c48161b3d890105`
- Run1 common: `b0461460a71f72c81b611bb060950a459e84d7f5cfe46f62da19625e624c59db`
- Run1 producer: `1c11544e5305c4350b3d985ccf81e88de8f1f31c58e662df819570aaa92bccbc`
- Run1 validator: `061cf80ae532e268b623050db5b39c47b9f0858f0a9591d94f8404e11e778fd2`
- Run1 wrapper: `1eb342475c63df5d16bf570c82de465803652e4a6157444ea02294b58fa9596d`
- Run1 sbatch: `ccbade355239d3c313b70ed55a2907f7a2d51716a62b809e835ec9ad1441d882`

Protected old LB-SCGP scope remained unchanged: 278 files, manifest SHA256 `243e89b69b169b222dd97f9df092d511f823fb26201e91fd89cb581710940462`.

## Decision

Run2 final status: **FAIL**.

No rerun, tuning, or repair was performed after the failed authorized submission. M0 overall is not passed. The next boundary is a fresh independent Run2 code+artifact review or newly authorized repair boundary.
