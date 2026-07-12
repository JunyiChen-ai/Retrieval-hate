# LB-SCGP G0 v2 Repair Handoff

**Date:** 2026-07-11  
**Scope:** bounded C1/H1 repair and no-clobber v2 freeze only.  
**Prior artifact preserved:** `artifacts/lb_scgp/v1/CONFIG_FREEZE.json` SHA256 `b6697472b61a61706c694a67b21618d618fcad6e7f59265d8696aee79dc46889`; lock SHA256 `34a05bde46775bcb75384c1c853cef7985f5c05efcdf3afec5fad6d85ef57d8d`.

## Repair Summary

- Added `configs/lb_scgp/lb_scgp_v2.json` with artifact namespace `artifacts/lb_scgp/v2` and exact freeze run ID `LBSCGP-G0-FREEZE-v2`.
- Registered v2 G0 later identities in config only: `LBSCGP-G0-CODE-AUDIT-v2`, `LBSCGP-G0-SYNTH-v2`, `LBSCGP-G0-REAL-MHC_zh-F4-S0-v2`, `LBSCGP-G0-REAL-REPLAY-MHC_zh-F4-S0-v2`, and `LBSCGP-G0-DECISION-v2`. They remain unrun and locked behind a fresh independent v2 formal code audit.
- Made freeze/run identity and formal freeze input keys config-driven in producer/verifier while preserving v1 defaults.
- Excluded mutable audit-trail records from v2 formal `input_files` and dirty-state predecessor checks: `refine-logs/lb_scgp/EXPERIMENT_TRACKER.md`, `TARGET_LOOP.md`, `TARGET_STATE.json`, `TARGET_FINDINGS.md`, `TARGET_REVIEW_RAW.md`, this handoff, and `G0_FREEZE_EXECUTION_V2.md`.
- Added `refine-logs/lb_scgp/v2/PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json` to bind the existing sanitizer physical records under the clarified dedicated pre-freeze schema without exposing protected disclosure locators as formal inputs.
- Amended `refine-logs/lb_scgp/EXPERIMENT_PLAN.md` before v2 freeze to state the dedicated pre-freeze sanitizer schema and that the full generic manifest/decision schema applies from `G0_FREEZE` onward.
- Widened the GPU replay wrapper's locked replay identity guard to accept the explicit future v2 replay run ID; defaults still point to v1 and no GPU/replay job was run.

## File Hashes Before v2 Freeze

```text
eec778811cfd2cf72a21dbf55af1c768ac6f849234350d958e900f871e41154f  configs/lb_scgp/lb_scgp_v2.json
9eb1f8c69d0a0e1b7c967b658b9a6d11af38b7653348bb38e2b7c2c6b25c2bc7  refine-logs/lb_scgp/EXPERIMENT_PLAN.md
cef4c63dd8a9407bd19f14de6e0e81f7816166113889091b670b5eb30ac5e70b  refine-logs/lb_scgp/v2/PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json
60580f3857c13f5fd944653c538bf9ba9daf2218ef5d4683b1f1ef1b26af2e55  scripts/analysis/lb_scgp_common.py
842f427b9e4c4f118ca47d75bdcb93619fe01fc69e5631ffd71809086db0658b  scripts/analysis/lb_scgp_g0.py
8ab99bad45daea1963dd030c24f91c28c87b46631d46e6fcafa4b3e3e102a4f6  scripts/analysis/lb_scgp_independent_verify.py
2fba75be6ad341e18114631deeed6612f7bfda032f2f46fce87186d8a4d7938b  scripts/analysis/lb_scgp_real_replay.py
79324abdd5f7ef243b189a4eba57f037e7fe56d2b6e6b1e888be7bac4e1ca29f  scripts/slurm/lb_scgp_g0_gpu.sbatch
```

## Static Checks

All checks were shell-only; no login-node Python/import/data/model execution was used.

```text
jq empty configs/lb_scgp/lb_scgp_v2.json refine-logs/lb_scgp/v2/PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json TARGET_STATE.json
bash -n scripts/slurm/lb_scgp_g0_cpu.sbatch scripts/slurm/lb_scgp_g0_gpu.sbatch scripts/slurm/lb_scgp_sanitize_inputs.sbatch scripts/disk_guard.sh
git diff --check -- scripts/analysis/lb_scgp_common.py scripts/analysis/lb_scgp_g0.py scripts/analysis/lb_scgp_independent_verify.py scripts/analysis/lb_scgp_real_replay.py scripts/slurm/lb_scgp_g0_gpu.sbatch configs/lb_scgp/lb_scgp_v2.json refine-logs/lb_scgp/EXPERIMENT_PLAN.md refine-logs/lb_scgp/v2/PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json
```

Payload hashes were validated with `jq -cS 'del(.payload_sha256)' | tr -d '\n' | sha256sum` for the v2 snapshot and the three existing sanitizer records.

## Non-Claims

- No PASS audit artifact was created.
- No synthetic, realfold, replay, decision, G1, teacher, MLLM, OCR, or performance job was run.
- The only gold remains `parent_video_binary_label`; `segment_gold_exists=false`, `segment_gold_used=false`.
- No segment/subclip artifact or objective exists under `artifacts/lb_scgp`.
- The next gate is a fresh independent v2 formal code audit; this handoff is not self-certification.
