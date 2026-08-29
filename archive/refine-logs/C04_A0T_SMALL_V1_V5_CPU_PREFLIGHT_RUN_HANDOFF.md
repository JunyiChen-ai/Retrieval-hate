# C04 A0T Small V1 v5 — CPU Preflight Run Handoff

## Status

- Runtime authority: `GO (0C/0H/0I)`
- Authorized scope: `ONE_FIXED_CPU_PREFLIGHT_ONLY`
- Execution state: `READY_NOT_SUBMITTED`
- No C04 v5 artifact namespace existed at freeze time.
- No SLURM job has been submitted by the implementation/review agents.

## Fixed SLURM entrypoint

- Absolute path: `/data/jehc223/RGCL/scripts/slurm/c04_a0t_small_v1_v5_preflight.sbatch`
- Repository-relative path: `scripts/slurm/c04_a0t_small_v1_v5_preflight.sbatch`
- SHA-256: `4f30fb3009a6f8e01c38b0fe73b146ed09981a855acc1ab52dd7b12b26893ded`
- Fixed wrapper: `scripts/wrappers/c04_a0t_small_v1_v5_preflight.sh`
- Wrapper SHA-256: `f824e197396feea7bbc370b0581a80fa4e7d42024ab6484cf3b62e30838be7d6`

The independent run agent may submit exactly once with:

```bash
sbatch scripts/slurm/c04_a0t_small_v1_v5_preflight.sbatch
```

The job is CPU-only. The fixed sbatch has no GPU request, `--time`, array, or dependency. If initially held as `JobHeldUser`, wait for automatic release; do not force-release it.

## Frozen runtime authority

- Authorized config: `configs/c04/c04_a0t_small_v1_v5.json`
- Authorized config SHA-256: `5228c08fefcd41c354202dfd7a750b46e1a5d4d66a8b3562b736c9d943f526a6`
- Normalized config-contract SHA-256: `2bc1971e8b222e874a2000a2fca25b70e4391c41a0ecaf69b7040fdc7cb65f50`
- Code/resource authorization manifest: `refine-logs/C04_A0T_SMALL_V1_V5_CODE_RESOURCE_AUTHORIZATION.json`
- Manifest SHA-256: `85a2ddc140ee523fdbdcd6764a736bdbd6b8c1731b7b76439207498d3d74d5a4`
- Manifest closure SHA-256: `b97bc9ad533f01e8dd4a9eee2b047117e68c9cefeee869e96bf6993e63640d3a`
- CPU-preflight unlock record: `refine-logs/C04_A0T_SMALL_V1_V5_CPU_PREFLIGHT_UNLOCK_RECORD.md`
- Unlock-record SHA-256: `04a93da7dbe71f12c709fe55a6571dd16eadb81bd302521cec098f3877c2d918`
- Independent unlock review: `refine-logs/C04_A0T_SMALL_V1_V5_CPU_PREFLIGHT_UNLOCK_REVIEW.md`
- Unlock-review SHA-256: `7b02c1ac67f447abc1f0a9501b1431c0e6d2ed289c311ca471a48327deda501e`

## Final TARGET closure

- `TARGET_STATE.json`: `56d425a4cea4849c7256dd5f5fb6548e98b807c1d5791dd3a8198ca1fb1689c7`
- `TARGET_LOOP.md`: `1241a4e99f65fd259c307b405af3d80fc02785c49744c8fabfc8586e2c737dab`
- `TARGET_FINDINGS.md`: `813c1041ed698d12d60bf45f5789fa3ed93a92aee0cd719552bee315020e9747`
- `TARGET_REVIEW_RAW.md`: `74bc50ba5cd6eea650fd84c0abba124db00eddcbc2530e07cf3532420d0eb924`

## Hard boundary after submission

This authority permits only the fixed CPU preflight self-test and payload freeze. It does **not** authorize:

- teacher inference or teacher payload generation beyond the preflight freeze contract;
- any GPU allocation or GPU computation;
- the small run, reconciliation, dev/test evaluation, OCR, external API/network access, cross-validation, label access, chained submission, release, or resubmission;
- any overwrite or reuse of an existing artifact namespace.

After the preflight terminates, an independent collector/reviewer must inspect the frozen artifacts and issue a fresh payload-review verdict. No GPU or downstream stage becomes authorized merely because the CPU preflight succeeds.
