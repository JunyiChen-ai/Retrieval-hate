# LB-SCGP Global-R2 M0 Synth KKT Implementation

Date: 2026-07-12

## Scope

Run2 attempted only:

`LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1`

No Run3 or later run was submitted. No MLLM, OCR, GPU, training, performance, train-bank, held, validation, test, `query_z`, `query_labels`, teacher, cache, or network/model call was made.

## Run2 Files Created

- `configs/lb_scgp_global_r2/m0_synth_kkt_v1.json`
- `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_payload_v1.schema.json`
- `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_case_v1.schema.json`
- `scripts/analysis/lb_scgp_global_r2_run2_common.py`
- `scripts/analysis/lb_scgp_global_r2_run2_producer.py`
- `scripts/analysis/lb_scgp_global_r2_run2_independent_verify.py`
- `scripts/analysis/lb_scgp_global_r2_run2_validate.py`
- `scripts/wrappers/lb_scgp_global_r2_run2.sh`
- `scripts/slurm/lb_scgp_global_r2_m0_synth_kkt.sbatch`

## Intended Correction

The final pre-submission code path was revised so the intended FULL synthetic fixture no longer used the invalid `G_star=G0` zero-gradient construction:

- FULL intended `G0` as an explicit identity baseline.
- FULL intended `G_star` as a certificate/Q/M_Q-induced off-diagonal movement from `G0`.
- FULL intended nonzero `r_struct`, nonzero structural dual `nu`, diagonal affine duals, and stationarity closure through `grad_G = G_star-G0`, `grad_r = lambda r`, `normal_G = -A_struct^T nu + diag(mu)`, and `normal_r = nu`.
- FULL intended movement metrics as a binding nondegeneration gate.
- REMOVE/null parity remained the only zero-movement replay case.
- Independent verifier included an identity/no-movement FULL rejection injection.

## Actual Implementation State

Run2 did not reach producer or verifier execution. The submitted job failed in the SLURM validator before artifact creation because the on-disk config remained an older partial shape with `.paths.schema` and wrapper `scripts/wrappers/lb_scgp_global_r2_run2_synth_kkt.sh`, while the validator expected `.paths.payload_schema`, `.paths.case_schema`, `.paths.cert_schema`, and `scripts/wrappers/lb_scgp_global_r2_run2.sh`.

Because the single authorized submission failed, the implementation is not accepted and was not tuned or resubmitted.

## Hashes

- Config: `f6edf36c82f541f7fba52d981361a660f8dc91683245c198faffeee8a2ebffaa`
- Payload schema: `501b783285ef7e85d45d6d877590d8aa2c2992469ae3e9fcaef4d47771672e61`
- Case schema: `df3616ff50c54dc756bb600c6ef26626afafe070678450d10ca73b9ff83ddcac`
- Common: `5db2ab20758f1d2339489c321aa54b28d41972df1a564f0c1faf6b744b7642f4`
- Producer: `ecb4b655f137cc9496be84a0300181161198f04a9fda58e53ac4b85201bdac1d`
- Independent verifier: `3a5e1d53b4fd656b9d289eb7cacd0e982cb255d9368d5dd9a5fe25ca8e35e0b1`
- Validator: `bdd2936f1d7e3542cb82e58dac64928ca991ce14da1924f63e01f67eca41c6c5`
- Wrapper: `4a18e5ec1e33cbc3cdd99324a8f1ccd9313d4eec947b2a0376509458400cb201`
- SLURM script: `c6dbcf32826c11aabf1c0e90eab5b816ae192eebd47858ae507007217e9b9b70`

## Boundary

Run2 final status is **FAIL**. The next boundary is a fresh independent Run2 code+artifact review or a newly authorized repair run; this execution does not authorize Run3, M1, cache, MLLM/OCR, GPU, training, validation/test, or performance work.
