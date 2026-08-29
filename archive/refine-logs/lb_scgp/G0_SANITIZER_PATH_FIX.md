# LB-SCGP G0 Sanitizer Path-Normalization Fix

**Date:** 2026-07-11  
**Worker mode:** sole GPT-5.5 xhigh repair worker; no subagent, sidecar, dynamic workflow, SLURM submission, Python/import execution, data/model execution, teacher/MLLM/OCR/network call, artifact/cache write, or performance work.

## Lineage

Job `12737` (`LBSCGP-G0-SANITIZE-MHC_zh-F4-v1`) failed before sanitizer artifacts were created. The submitted command was:

```text
sbatch --export=ALL,TASK=build scripts/slurm/lb_scgp_sanitize_inputs.sbatch
```

The wrapper defaulted to relative paths:

```text
CONFIG=configs/lb_scgp/lb_scgp_v1.json
SOURCE_CONFIG=configs/lb_scgp/lb_scgp_sanitizer_sources.json
```

The log `slurm/logs/lbscgp_sanitize_12737.out` shows the failure occurred during config loading:

```text
ValueError: 'configs/lb_scgp/lb_scgp_v1.json' is not in the subpath of '/data/jehc223/RGCL' OR one path is relative and the other is absolute.
```

No `outer_train_features.pt`, `sanitized_provenance.json`, quarantine `sanitizer_manifest.json`, or `sanitizer_decision.json` exists from this run. The sanitizer verifier was not submitted.

## Root Cause

`AccessLedger.hash_file` called:

```text
Path(path).relative_to(ROOT)
```

on `path="configs/lb_scgp/lb_scgp_v1.json"` while `ROOT=/data/jehc223/RGCL` is absolute. `Path.relative_to` rejects relative-vs-absolute comparisons, so config load failed before any sanitizer artifact or decision record could be produced.

## Exact Code Changes

- `scripts/analysis/lb_scgp_common.py`
  - Added one canonical fail-closed helper pair:
    - `canonical_root_path(path, root=ROOT, must_be_under_root=True)`
    - `root_relative_path(path, root=ROOT, must_be_under_root=True)`
  - Relative inputs now resolve under `ROOT`; absolute or symlink-resolved escapes outside `ROOT` raise `RuntimeError`.
  - `AccessLedger.record_file`, `hash_file`, `read_json`, `read_jsonl`, and `record_bank_member` now use the helper and record stable ROOT-relative POSIX paths.
  - `resolve`, `implementation_hash`, `exclusive_publish`, and `hash_npz_members_only` now use the same canonicalization.
- `scripts/analysis/lb_scgp_sanitize_inputs.py`
  - Relative `--source-config` continues to work through shared canonicalization.
  - Source-config loading, source cache selection, quarantine ledger rows, source-lineage paths, and formal output paths now use the shared helper.
- `scripts/analysis/lb_scgp_verify_sanitizer.py`
  - Verifier direct ledger rows and formal decision paths now use shared canonicalization.
- `scripts/analysis/lb_scgp_g0.py`
  - Freeze config path, sanitizer provenance/decision paths, fixture paths, bank paths, train-only feature path, evidence paths, and output manifests now use shared canonicalization.
- `scripts/analysis/lb_scgp_independent_verify.py`
  - Kept verifier independence by adding a local helper with the same semantics instead of importing producer/common code.
  - Config reads, hash reads, real-bank paths, freeze input checks, expected freeze path set, and verifier ledger rows now normalize relative paths under `ROOT`.
- `scripts/analysis/lb_scgp_real_replay.py`
  - Added local canonicalization for relative `--config`, config-derived artifact paths, read paths, hash paths, and publish paths.

The sanitizer wrapper may continue using relative `CONFIG` and `SOURCE_CONFIG` defaults. This is not a one-off absolute-path workaround.

## Static Scan Classification

Command:

```text
rg -n "relative_to\(ROOT\)|Path\([^\n]*\)\.relative_to\(ROOT\)|\.resolve\(" scripts/analysis/lb_scgp_*.py
```

Remaining `.resolve()` sites are classified as safe:

- `scripts/analysis/lb_scgp_common.py`: helper internals only (`ROOT` canonicalization, candidate canonicalization, root drift check).
- `scripts/analysis/lb_scgp_independent_verify.py`: local helper internals plus `_forbidden_path_string` fallback for scanner candidate generation only.
- `scripts/analysis/lb_scgp_real_replay.py`: local helper internals only.

There are no remaining direct `Path(...).relative_to(ROOT)` ledger/config/output callsites outside canonical helpers after this repair.

## Non-Actions

- No SLURM job was submitted or resubmitted.
- No sanitizer artifact, verifier decision, G0 freeze, synthetic, realfold, replay, decision, G1, teacher, MLLM, OCR, network, model, data, cache, or performance action was run.
- No segment path was reintroduced. `segment_gold_exists=false`, `segment_gold_used=false`, `segment_artifact_created=false`, `segment_objective_allowed=false`, `lambda_seg=0`, and `segment_cache=None` remain the contract.
- Protected-path, formal-surface, no-clobber, source-isolation, and allowed-member gates were not loosened.
- This record does not self-certify `0 CRITICAL / 0 HIGH`.

## SHA256 Ledger

```text
3f0121867458553c824ae9e82fc1bfb016e28ddb4391b7ea678098aea5e0e24e  scripts/analysis/lb_scgp_common.py
215dfca8d13340a13da8cca505dbd9701c2218a6a9af57f768495b626ca5b4f8  scripts/analysis/lb_scgp_sanitize_inputs.py
6cd649d60cd0555aa34e94561a45e3147408354cd8d0c0798d2a9c8ea283351e  scripts/analysis/lb_scgp_verify_sanitizer.py
182416fde91967a913baa15ccb403bf0c3da7bae2a9167244700a5930e33deef  scripts/analysis/lb_scgp_g0.py
7acdf3f9e24e6c42a0cae2bfd66986927723099ef73f1afec8904b5ad1a01ec9  scripts/analysis/lb_scgp_independent_verify.py
fb1e3c3b85c84df9fa675dd877fc33879f02ce7f708597f997e2253181ea7cec  scripts/analysis/lb_scgp_real_replay.py
82e32bfa9fa552744106fd8a5b9c2e07de8e55ab0ad29c35cb5a6907ca744b43  configs/lb_scgp/lb_scgp_v1.json
ceb1676bfe81ad62373911458570a681d05b632beebe3b5b757c9c7e83eb5aae  configs/lb_scgp/lb_scgp_sanitizer_sources.json
350b9b946470c9afbe4fcbbfb4b742d9f24a984e2c983089276d2ec5f48fff17  scripts/slurm/lb_scgp_sanitize_inputs.sbatch
80d071490eb9f799ed119dce04d9004aad30a5fd2189ccc6b2c09d7d4fa4b4d7  slurm/logs/lbscgp_sanitize_12737.out
```

## Static Validation

Allowed validation only was used or prepared for final checks: `rg`/`sed` inspection, `jq empty`, `bash -n`, `git diff --check`, and `sha256sum`. No Python/import execution was used.

## Next Step

An independent sole GPT-5.5 xhigh reviewer must review this path-normalization repair before any sanitizer resubmission. If that review accepts the repair, the next operational step is a fresh sanitizer build under SLURM, followed only then by the independent sanitizer verifier if required artifacts exist.
