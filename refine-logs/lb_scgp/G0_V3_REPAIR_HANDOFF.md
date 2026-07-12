# LB-SCGP G0 v3 Repair Handoff

**Date:** 2026-07-11  
**Scope:** narrow repair for the sole v2 formal code audit Critical C1.  
**Prior artifacts preserved:** v1 and v2 freeze artifacts, publish locks, and formal records were not modified.

## Repair Summary

- Added `configs/lb_scgp/lb_scgp_v3.json` with namespace `artifacts/lb_scgp/v3` and exact freeze run ID `LBSCGP-G0-FREEZE-v3`.
- Registered v3 future identities in config only: `LBSCGP-G0-CODE-AUDIT-v3`, `LBSCGP-G0-SYNTH-v3`, `LBSCGP-G0-REAL-MHC_zh-F4-S0-v3`, `LBSCGP-G0-REAL-REPLAY-MHC_zh-F4-S0-v3`, and `LBSCGP-G0-DECISION-v3`.
- Kept the sealed real fixture, train-only feature artifact, sanitizer provenance/decision, and v2 pre-freeze sanitizer contract snapshot as immutable v3 inputs. No sanitizer, synthetic, realfold, replay, decision, G1, teacher, MLLM, OCR, or cache task was run.
- Made dirty-state exclusions config-explicit for v3 and shared by producer/common and the independent verifier. v1/v2 configs retain compatibility with their existing default policy.
- Added v3 formal artifact exclusion prefixes exactly: `artifacts/lb_scgp/v1/`, `artifacts/lb_scgp/v2/`, and `artifacts/lb_scgp/v3/`.
- Added exact dirty-state excluded paths only for mutable audit/progress records expected to change after v3 freeze: `EXPERIMENT_TRACKER.md`, `TARGET_LOOP.md`, `TARGET_STATE.json`, `TARGET_FINDINGS.md`, `TARGET_REVIEW_RAW.md`, `G0_V3_REPAIR_HANDOFF.md`, `G0_FREEZE_EXECUTION_V3.md`, and `G0_FORMAL_CODE_AUDIT_REVIEW_V3.md`.
- Kept the only dirty-state prefix exclusion narrow: `refine-logs/lb_scgp/runtime/`.
- Changed the GPU replay wrapper's replay run-ID guard to parse `lineage.run_ids.replay` from `CONFIG` and fail closed if the field is absent or mismatched. No replay job was submitted.

## File Hashes At v3 Freeze

```text
a480c9b9bf56c938667b4f8e2f3d07882b84843627233b613d864764c02eaf47  configs/lb_scgp/lb_scgp_v3.json
f1c95add65d59c6bc692682b4a91daf8f5912a9deb6a19998b412b2e404282bb  scripts/analysis/lb_scgp_common.py
d82d1d32ca2ae89fe291cae015120d8f350a8784280e922b7be365d779b98600  scripts/analysis/lb_scgp_g0.py
d1e50057b4c166a71426f89474b6526e3eab11da5547e15368112b6620dbf5ce  scripts/analysis/lb_scgp_independent_verify.py
055d47a006c1e22d4f5d2d4cf1c0e739cec7950d134d01b39ad435149fe839f4  scripts/slurm/lb_scgp_g0_gpu.sbatch
```

Freeze-bound hashes:

```text
config_canonical_sha256=84227b68eaa496da6e307ce5c5ef3469e1b7c68e350f0d62d1677d01f07645bf
implementation_sha256=b8759436a6c5e2a67bf7125cbd1ab57cb05187e764e837373abfdf1a92916e75
independent_verifier_sha256=d1e50057b4c166a71426f89474b6526e3eab11da5547e15368112b6620dbf5ce
dirty_diff_sha256=91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1
access_ledger_sha256=3db4b94900a9d9b807ab495be869a5ef87a3894f987eef03ea1e948030abdc72
payload_sha256=352ec2215e2225b1768a13f39f96ef935b91966606d8db874d6de4410b1a9f3d
```

## Static Checks

All checks were shell-only; no local Python/import/data/model execution was used.

```text
jq empty configs/lb_scgp/lb_scgp_v3.json
bash -n scripts/slurm/lb_scgp_g0_cpu.sbatch scripts/slurm/lb_scgp_g0_gpu.sbatch scripts/slurm/lb_scgp_sanitize_inputs.sbatch
git diff --check -- scripts/analysis/lb_scgp_common.py scripts/analysis/lb_scgp_g0.py scripts/analysis/lb_scgp_independent_verify.py scripts/slurm/lb_scgp_g0_gpu.sbatch configs/lb_scgp/lb_scgp_v3.json
```

## Non-Claims

- This handoff is not a formal code audit.
- No formal code-audit PASS artifact was created.
- No synthetic, realfold, replay, decision, G1, teacher, MLLM, OCR, or performance job was run.
- The only gold remains `parent_video_binary_label`; `segment_gold_exists=false`, `segment_gold_used=false`.
- MLLM/OCR/teacher/cache calls and reads/writes are all `0`.
- Outer-held label/content, validation content, and test content reads are all `0`.

## Next Gate

The next allowed gate is an independent `LBSCGP-G0-CODE-AUDIT-v3` review. No later stage is unlocked by the v3 freeze.
