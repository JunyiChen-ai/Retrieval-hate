# C01 Zero-Contract Probe

**Status:** `READY_NOT_RUN`. The probe is diagnostic-only and does not authorize a change to A0. No probe job or artifact exists at preregistration time.

## Trigger and diagnosis

SLURM job `13712` failed closed (`FAILED`, exit `1:0`, elapsed `00:00:53`) before producing an A0 result or decision. The exact halt was `HateMM/train/standard/img has 1 rows at/below epsilon 1e-12; first=355`.

The following is `REPORTED_EXTERNAL_NOT_VERIFIED_BY_PROBE`; it motivates the probe but is not a prewritten probe conclusion:

- `slurm/logs/gen_embed_readout_13468.out:996-1002,1023-1026` reports both decoder failures and one zero-vector video in the four HateMM-train readout cells.
- `src/utils/generate_VideoMLLM_embedding_readout_HF.py:423-426` shows the shared guard writing zero image/text tensors to every readout cell.
- `refine-logs/PROVENANCE_AUDIT_2026-07-28.md:187-193` reports row 355 as `hate_video_95`, label 1, zero in historical caches and consumed by the deployed rule.
- `refine-logs/MNTP_S1_RECORD.md:183-185,227-230` reports the same row/ID and a matched historical-arm zero.

This evidence is not substituted for a current-cache check. A0 remains unchanged and fail-closed until the probe verifies the exact eight hash-protected caches.

## Frozen probe

- Analysis: `scripts/analysis/c01_zero_contract_probe.py`
- SLURM: `scripts/slurm/c01_zero_contract_probe.sbatch`
- Run ID: `C01-ZERO-PROBE-v1`
- Exclusive artifact: `artifacts/c01_policy_contrastive/v2/zero_contract_probe/C01-ZERO-PROBE-v1/zero_contract_probe.json`
- Resources: conda `HateVideo`, CPU-only, `1 CPU / 4G`, `CUDA_VISIBLE_DEVICES=""`, all thread variables `1`, no `--time`, no GPU, no `disk_guard`.
- Inputs: exactly the manifest-registered MHC-ZH/HateMM `train`/`dev_seen` standard/one-word L24 caches. Test-like paths are rejected before open. Before any cache open, the whole manifest must exact-match approved SHA256 `083275d39a1026bde3b6583bd5608d41cec5b431da9ffda87ae8ab1046cf2305`. Each cache is then rehashed and must equal its manifest 64-hex SHA256—not merely sha16—before `torch.load`.
- IDs: cache IDs must already be non-empty raw strings; coercion is forbidden, and paired endpoints are compared directly in exact order.
- Output: per dataset/split/policy/modality exact-zero counts, IDs, row indices and labels; nonzero-tiny and non-finite rows with ID/index/label; and exact standard-versus-one-word zero-mask comparisons. For each zero row, the other modality is one neutral enum: `normal_nonzero`, `exact_zero`, `tiny_nonzero`, or `nonfinite`. `exact_zero` means only that every stored component equals zero; the probe does not call it structural. Structural interpretation requires the separately cited external evidence and a fresh review. Feature values are never serialized.
- Publication: small JSON capped at 1 MB, atomic exclusive-create in a fresh namespace, no force or overwrite path.

## C01 v2 repair criterion

Retaining a zero block may be considered only if all of the following hold:

1. the zero is a documented structural null rather than an arbitrary small-norm representation;
2. standard and one-word endpoints have an exact zero-mask match at the same ID and same modality;
3. no nonzero row at or below `1e-12` exists;
4. frozen historical evidence confirms the same baseline protocol consumed that same ID/modality as the structural null.

Any endpoint-only zero, ID/modality mismatch, non-structural tiny row, non-finite row, or missing historical-baseline evidence remains fail-closed. The probe artifact itself sets `allow_zero_block_in_a0=false`; a separate reviewed C01 v2 implementation would be required after the evidence satisfies the criterion.

## Execution state

- Probe: `READY_NOT_RUN`
- A0 job `13712`: failed closed; no result/decision artifact
- A0 science script: unchanged by this preparation
- Probe submission/artifact: none
