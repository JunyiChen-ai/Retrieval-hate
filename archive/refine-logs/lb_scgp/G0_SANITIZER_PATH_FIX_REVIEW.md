# LB-SCGP Sanitizer Path-Normalization Repair Review

Scope was read-only static inspection only. I did not edit files, run Python/imports, submit SLURM, use network, create artifacts, or delegate.

**Supersession note:** this review was completed before the fresh sanitizer build/verifier. Its "artifacts absent" observations describe the state immediately after failed job `12737` and before successful jobs `12738` and `12739`. Current physical artifact status is recorded in `refine-logs/lb_scgp/G0_SANITIZER_EXECUTION.md` and `refine-logs/lb_scgp/G0_SANITIZER_PHYSICAL_REVIEW.md`.

## Findings

CRITICAL: none.

HIGH: none.

MEDIUM: none.

LOW: none.

## Evidence

Job 12737 root cause is correctly identified. The log shows `lb_scgp_sanitize_inputs.py` failed during `load_config(args.config, ledger)` and then `AccessLedger.hash_file` called `Path(path).relative_to(ROOT)` on relative `configs/lb_scgp/lb_scgp_v1.json`, raising the relative-vs-absolute `ValueError`: `slurm/logs/lbscgp_sanitize_12737.out:7`, `slurm/logs/lbscgp_sanitize_12737.out:10`, `slurm/logs/lbscgp_sanitize_12737.out:18`, `slurm/logs/lbscgp_sanitize_12737.out:19`, `slurm/logs/lbscgp_sanitize_12737.out:21`.

The repair closes that path class. Relative wrapper defaults remain relative at `scripts/slurm/lb_scgp_sanitize_inputs.sbatch:18` and `scripts/slurm/lb_scgp_sanitize_inputs.sbatch:19`, but `load_config` now reaches `AccessLedger.read_json`, which canonicalizes through `canonical_root_path`: `scripts/analysis/lb_scgp_common.py:81`, `scripts/analysis/lb_scgp_common.py:82`, `scripts/analysis/lb_scgp_common.py:214`, `scripts/analysis/lb_scgp_common.py:216`.

Canonical semantics are fail-closed for relative and absolute paths: relative inputs resolve under `ROOT`, symlink/`..` or absolute escapes raise `RuntimeError`, and stable ROOT-relative POSIX recording is returned: `scripts/analysis/lb_scgp_common.py:145`, `scripts/analysis/lb_scgp_common.py:152`, `scripts/analysis/lb_scgp_common.py:154`, `scripts/analysis/lb_scgp_common.py:155`, `scripts/analysis/lb_scgp_common.py:157`, `scripts/analysis/lb_scgp_common.py:160`, `scripts/analysis/lb_scgp_common.py:164`, `scripts/analysis/lb_scgp_common.py:167`.

AccessLedger hash/read/record behavior is now canonical and stable: `record_file` records `rel.as_posix()` at `scripts/analysis/lb_scgp_common.py:67`, `scripts/analysis/lb_scgp_common.py:69`; `hash_file` opens the canonical filesystem path and records the canonical relative path at `scripts/analysis/lb_scgp_common.py:73`, `scripts/analysis/lb_scgp_common.py:74`, `scripts/analysis/lb_scgp_common.py:76`; JSON/JSONL reads canonicalize before open at `scripts/analysis/lb_scgp_common.py:81`, `scripts/analysis/lb_scgp_common.py:88`. NPZ helpers canonicalize container paths and only open explicit allowed members: `scripts/analysis/lb_scgp_common.py:811`, `scripts/analysis/lb_scgp_common.py:813`, `scripts/analysis/lb_scgp_common.py:823`, `scripts/analysis/lb_scgp_common.py:824`, `scripts/analysis/lb_scgp_common.py:839`, `scripts/analysis/lb_scgp_common.py:842`, `scripts/analysis/lb_scgp_common.py:852`.

No-clobber and nonexistent parent handling are intact. `exclusive_publish` canonicalizes output, creates missing parents inside ROOT, uses `O_EXCL`, refuses existing targets, hardlinks tmp to final, and leaves persistent locks: `scripts/analysis/lb_scgp_common.py:259`, `scripts/analysis/lb_scgp_common.py:261`, `scripts/analysis/lb_scgp_common.py:262`, `scripts/analysis/lb_scgp_common.py:264`, `scripts/analysis/lb_scgp_common.py:271`, `scripts/analysis/lb_scgp_common.py:278`, `scripts/analysis/lb_scgp_common.py:287`.

Sanitizer source flow is isolated. Formal config carries sanitized artifact paths only: `configs/lb_scgp/lb_scgp_v1.json:15`, `configs/lb_scgp/lb_scgp_v1.json:20`, `configs/lb_scgp/lb_scgp_v1.json:22`; source lineage remains quarantine-only: `configs/lb_scgp/lb_scgp_sanitizer_sources.json:3`, `configs/lb_scgp/lb_scgp_sanitizer_sources.json:4`, `configs/lb_scgp/lb_scgp_sanitizer_sources.json:7`. Sanitizer loads source config through canonicalization and writes source details only into the quarantine manifest, not formal provenance: `scripts/analysis/lb_scgp_sanitize_inputs.py:116`, `scripts/analysis/lb_scgp_sanitize_inputs.py:117`, `scripts/analysis/lb_scgp_sanitize_inputs.py:120`, `scripts/analysis/lb_scgp_sanitize_inputs.py:208`, `scripts/analysis/lb_scgp_sanitize_inputs.py:241`, `scripts/analysis/lb_scgp_sanitize_inputs.py:242`, `scripts/analysis/lb_scgp_sanitize_inputs.py:247`.

Sanitizer verifier path flow and decision isolation are preserved. It reads manifest/provenance/feature paths canonically, rejects quarantine manifest as formal input, checks sanitized provenance for forbidden surfaces, and emits a decision containing sanitized paths/hashes only: `scripts/analysis/lb_scgp_verify_sanitizer.py:73`, `scripts/analysis/lb_scgp_verify_sanitizer.py:79`, `scripts/analysis/lb_scgp_verify_sanitizer.py:83`, `scripts/analysis/lb_scgp_verify_sanitizer.py:93`, `scripts/analysis/lb_scgp_verify_sanitizer.py:95`, `scripts/analysis/lb_scgp_verify_sanitizer.py:155`, `scripts/analysis/lb_scgp_verify_sanitizer.py:164`, `scripts/analysis/lb_scgp_verify_sanitizer.py:184`.

G0 freeze/realfold path reachability is preserved. G0 binds sanitized provenance/decision and train-only feature hashes before use: `scripts/analysis/lb_scgp_g0.py:1497`, `scripts/analysis/lb_scgp_g0.py:1511`, `scripts/analysis/lb_scgp_g0.py:1513`, `scripts/analysis/lb_scgp_g0.py:1570`, `scripts/analysis/lb_scgp_g0.py:1575`. Formal protected/quarantine surfaces are rejected: `scripts/analysis/lb_scgp_g0.py:1844`, `scripts/analysis/lb_scgp_g0.py:1845`.

Independent verifier independence and syntax around the requested block are acceptable. It imports none of producer/common code: `scripts/analysis/lb_scgp_independent_verify.py:2`, `scripts/analysis/lb_scgp_independent_verify.py:4`. It has its own canonical helper: `scripts/analysis/lb_scgp_independent_verify.py:70`, `scripts/analysis/lb_scgp_independent_verify.py:74`, `scripts/analysis/lb_scgp_independent_verify.py:76`, `scripts/analysis/lb_scgp_independent_verify.py:80`. The synthetic expected-access block is not duplicated or malformed: `scripts/analysis/lb_scgp_independent_verify.py:991`, `scripts/analysis/lb_scgp_independent_verify.py:994`, `scripts/analysis/lb_scgp_independent_verify.py:1000`, `scripts/analysis/lb_scgp_independent_verify.py:1002`. The final `decide(cfg,args)` appears once: `scripts/analysis/lb_scgp_independent_verify.py:1756`, `scripts/analysis/lb_scgp_independent_verify.py:1765`.

No-segment boundary is intact. Config says parent-video binary labels only and no segment gold: `configs/lb_scgp/lb_scgp_v1.json:10`, `configs/lb_scgp/lb_scgp_v1.json:13`. G0 returns `segment=None`, sets `lambda_seg=0.0`, asserts no segment objective, and calls `compute_loss(... segment_cache=None)`: `scripts/analysis/lb_scgp_g0.py:1917`, `scripts/analysis/lb_scgp_g0.py:1941`, `scripts/analysis/lb_scgp_g0.py:1944`, `scripts/analysis/lb_scgp_g0.py:2000`. Real replay mirrors this: `scripts/analysis/lb_scgp_real_replay.py:210`, `scripts/analysis/lb_scgp_real_replay.py:235`, `scripts/analysis/lb_scgp_real_replay.py:272`, `scripts/analysis/lb_scgp_real_replay.py:297`.

Artifact state is accurate. `find artifacts/lb_scgp -maxdepth 5 -print` returned `No such file or directory`; hidden lock scan found no LB-SCGP locks/sentinels. Docs record the same absence and blocked verifier state: `refine-logs/lb_scgp/G0_SANITIZER_EXECUTION.md:81`, `refine-logs/lb_scgp/G0_SANITIZER_EXECUTION.md:86`, `refine-logs/lb_scgp/G0_SANITIZER_EXECUTION.md:92`, `refine-logs/lb_scgp/G0_SANITIZER_EXECUTION.md:100`, `refine-logs/lb_scgp/EXPERIMENT_TRACKER.md:4`, `refine-logs/lb_scgp/EXPERIMENT_TRACKER.md:10`, `refine-logs/lb_scgp/EXPERIMENT_TRACKER.md:11`.

## Verdicts

- Job-12737 root-cause closure: PASS.
- Build path reachability: PASS.
- Verifier path reachability: PASS.
- Escape/no-clobber safety: PASS.
- Independent verifier syntax/semantics: PASS by static inspection.
- No-segment boundary: PASS.
- Docs/state accuracy: PASS.

Final verdict: PASS, with 0 Critical and 0 High.

A fresh sanitizer build may be resubmitted under SLURM. The sanitizer verifier should remain blocked until that build produces the required train-only feature artifact, sanitized provenance, and quarantine manifest.
