# LB-SCGP G0 v5 Repair Handoff

**Date:** 2026-07-11
**Executor:** sole v5 repair executor; no subagent, sidecar, dynamic workflow,
nested Codex process, or other model was used.

## Root Cause

v4 is operationally blocked because the producer strict schema verifier called
`git_state(cfg=cfg)` without importing `git_state`. The failure is recorded in
`slurm/logs/lbscgp_v4_pub_verify_12786.out` and
`refine-logs/lb_scgp/runtime/v4_audit_checks/publication_verification.json`.

v4 formal files are internally consistent but cannot unlock downstream work.

## Files Changed

- `scripts/analysis/lb_scgp_g0.py`
  - imports `git_state`;
  - extends the strict code-audit consumer contract to v4/v5;
  - adds exact v5 prior-lineage no-clobber path set.
- `scripts/analysis/lb_scgp_independent_verify.py`
  - extends publisher and decision consumer to the same v4/v5 strict contract;
  - adds exact v5 formal-prefix and prior-lineage path sets.
- `scripts/slurm/lb_scgp_g0_audit_publish.sbatch`
  - remains CPU-only 2 CPU / 4G with no `--time`;
  - now accepts config-driven v4 or v5 lineages and rejects any other lineage.
- `configs/lb_scgp/lb_scgp_v5.json`
  - new no-clobber v5 namespace and run IDs.
- `refine-logs/lb_scgp/runtime/v5_repair_checks/`
  - audit-only regression harness and result files.

## v5 Namespace

```text
config:    configs/lb_scgp/lb_scgp_v5.json
namespace: artifacts/lb_scgp/v5
freeze:    artifacts/lb_scgp/v5/CONFIG_FREEZE.json
audit dir: artifacts/lb_scgp/v5/g0/code_audit/
```

Future v5 sidecars, not created by this repair:

```text
refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.md
refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.record.json
```

## Regression Summary

Audit-only regression job `12820` completed under SLURM with conda `HateVideo`:

```text
state=COMPLETED
exit=0:0
elapsed=00:00:03
alloc=2 CPU / 4G
result=PASS
```

It proved valid v5 producer consumption and fail-closed behavior for wrong
hash, dirty hash, run ID, and path. Fixture directories were removed, and
`artifacts/lb_scgp/v5` was not created.

Earlier audit-only job `12819` also passed before an error-message cleanup; it
is retained as runtime audit residue but not used as the final regression
result.

## Formal Freeze

The only authorized formal v5 workload was:

```text
CONFIG=configs/lb_scgp/lb_scgp_v5.json TASK=freeze RUN_ID=LBSCGP-G0-FREEZE-v5 sbatch scripts/slurm/lb_scgp_g0_cpu.sbatch
```

It was submitted as job `12823`, entered `PENDING (JobHeldUser)`, released
automatically, and completed `0:0` in `00:00:02` on `8 CPU / 64G`.

```text
freeze_sha256=254e45afe9c0355892824c0c26bc73b4b0854cb20c67c3703982762fad010931
payload_sha256=d89f7cd4ad43c8ef83a04b7530ec19186ec1cb0e96958d239f1ce5c2b146bb4d
lock_sha256=54dc06d236d5fc1f3ac96400f1a81faeb1d2c0c8e5af075065d1964260de98a9
log_sha256=c43a2b16fc8c95bdfafb5c48c674fa4b778dd9c3d3c3764e24fe50ed038a0526
dirty_diff_sha256=1c8284781fb57e90714b390fdbef362e978b70789632b19df3d8161dfe8827b7
config_canonical_sha256=4a45fb6c66884b6b8aa4571961dff3ef7751c2b9f97e2df1584521cfe1eb3dba
implementation_sha256=939acffbafbd9204fc654972cd73f174393c8466c61f4af045b1c20948a6b687
independent_verifier_sha256=f0f49f41de4efee9abf2267b27b75be440f0020583baef565955c3d0c2988b2d
access_ledger_sha256=ce6898035cf25dbe53f4b258a7b792796e4388a282384185f13a84926397ea0f
```

## Non-Claims

No v5 formal audit PASS, no v5 review/record, no audit-publish invocation, no
synthetic, no realfold, no replay, no decision, no G1, no teacher/MLLM/OCR, no
held/validation/test access, and no performance result is created by this
handoff.

The only gold is `parent_video_binary_label`; `segment_gold_exists=false` and
`segment_gold_used=false`.
