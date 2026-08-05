# C04-A0T-SMALL-v1 v5 CPU-Preflight Authority Context

Date: 2026-07-30  
Status: **PROSPECTIVE AUTHORITY SNAPSHOT / NOT SUBMITTED**

## Reviewed predecessor

- Prospective v5 config SHA-256:
  `78e2ade7e91c74446eeba0d2965bc4675a804717857a0a202caabe1f80440a1b`
- v5 implementation record SHA-256:
  `aa49585cc27853793b349868bb6996ab5f051a44cfa1ff39c2fee0e7e1a704a1`
- v5 implementation review verdict: `GO (0C/0H/0I)`
- Frozen design review SHA-256:
  `340ae2c156e7acab8a19dcda9625f883058377ca618bdc4fd59177900738a854`

## Exact authority delta

The authorized config may change only the normalized staged fields:

- `authorization.preflight_materialization_authorized: false -> true`
- `review.code_resource_verdict: PENDING -> GO`
- `review.code_resource_authorization_sha256:
  PENDING_CODE_RESOURCE_REVIEW -> exact authority-manifest SHA-256`

`implementation_authorized` remains true. Every other authorization is exactly
false: teacher, GPU, Slurm-GPU, small tranche, post-job reconciliation, dev,
test, OCR, external API, network, cross-dataset, label-value-before-seal, chain,
release, and resubmit.

The normalized config-contract SHA-256 is
`2bc1971e8b222e874a2000a2fca25b70e4391c41a0ecaf69b7040fdc7cb65f50`.
All fifteen implementation hashes, all fifteen frozen-design hashes, source
paths/hashes/sizes, and model/processor file/tree hashes remain unchanged.

## Pending payload semantics

Before CPU preflight:

- system/A/B/combined prompt hashes are exactly
  `PENDING_CPU_PREFLIGHT_HASH_FREEZE`;
- map expected hashes are exactly
  `PENDING_CPU_PREFLIGHT_AND_FRESH_PAYLOAD_REVIEW`;
- payload-review pin is
  `PENDING_CPU_PREFLIGHT_AND_PAYLOAD_REVIEW`;
- GPU-execution pin is `PENDING_GPU_EXECUTION_REVIEW`;
- resource-reconciliation pin is
  `PENDING_POST_JOB_RECONCILIATION_REVIEW`;
- authority payload binding is exactly `NO_PREFLIGHT_PAYLOAD_YET`.

The namespace `artifacts/c04/a0t_small_v1_impl_v5` was absent at this authority
freeze. No authorization/payload manifest or runtime artifact existed, no
Python/data/model operation ran, and no SLURM job was submitted.

CPU preflight remains blocked until the exact authority manifest and authorized
config snapshot pass a fresh independent unlock review.
