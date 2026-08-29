# LB-SCGP G0 Formal Code Audit Review

Review target: `LBSCGP-G0-FREEZE-v1` after sanitizer physical review and freeze job `12742`.

Reviewer mode: sole independent read-only audit. No subagents, Python, SLURM, GPU/compute, network, or code repair were used. This file is the only artifact written. The formal PASS artifacts under `artifacts/lb_scgp/v1/g0/code_audit/` were not created because the audit is not 0 Critical / 0 High.

## Verdict

**FAIL**

- Critical findings: **1**
- High findings: **1**
- Formal code-audit PASS artifact authorized: **NO**
- Synthetic/realfold/replay/decision authorization: **NO**
- G1 and all teacher/MLLM/OCR stages remain locked.

## Critical Findings

### C1. Freeze hash-binds mutable audit-trail records that already drifted

Severity: **Critical**

`CONFIG_FREEZE.json` is not a valid predecessor for formal synthetic/realfold execution in the current repository state. It freezes mutable progress records and later predecessor checks rehash them as immutable inputs. The required post-freeze audit-trail updates changed those records, so a formal code-audit PASS artifact would still leave `LBSCGP-G0-SYNTH-v1` fail-closed before numerical execution.

Evidence:

- Freeze construction adds mutable records to `input_files`: `experiment_tracker`, `target_loop`, and `target_state` are appended in [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1581).
- Numerical predecessor verification rehashes every frozen `input_files` row and raises `frozen input drift` on mismatch in [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1634) and [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1645).
- The independent G0 decision verifier also requires the freeze dirty hash to equal the current dirty hash and revalidates every frozen input in [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1703), [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1711), and [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1720).
- The freeze execution record explicitly documents the hazard in [G0_FREEZE_EXECUTION.md](/data/jehc223/RGCL/refine-logs/lb_scgp/G0_FREEZE_EXECUTION.md:100).
- The same record shows current mismatches:
  - `EXPERIMENT_TRACKER.md`: frozen `6f89fd...d0a42`, current `7dc617...d1f9` at [G0_FREEZE_EXECUTION.md](/data/jehc223/RGCL/refine-logs/lb_scgp/G0_FREEZE_EXECUTION.md:105).
  - `TARGET_LOOP.md`: frozen `b0387f...f06b9`, current `116498...5c8c` at [G0_FREEZE_EXECUTION.md](/data/jehc223/RGCL/refine-logs/lb_scgp/G0_FREEZE_EXECUTION.md:106).
  - `TARGET_STATE.json`: frozen `a86f5a...82ce1`, current `b04ee6...0641` at [G0_FREEZE_EXECUTION.md](/data/jehc223/RGCL/refine-logs/lb_scgp/G0_FREEZE_EXECUTION.md:107).
- The tracker independently records this as a required audit issue before synthetic/realfold execution in [EXPERIMENT_TRACKER.md](/data/jehc223/RGCL/refine-logs/lb_scgp/EXPERIMENT_TRACKER.md:54).

Impact:

This is not cosmetic provenance drift. The producer's `_load_freeze_and_audit` would reject the frozen predecessor during `synthetic`, and the decision verifier would also reject the freeze. Therefore the frozen code/config/provenance are not safe to authorize a formal `LBSCGP-G0-CODE-AUDIT-v1` PASS artifact or later G0 numerical stages.

Minimal repair:

Create a new freeze lineage, not an in-place edit of `CONFIG_FREEZE.json` because the namespace and publish lock are already committed. The repair must either:

1. remove mutable tracker/state/progress records from the rehashed freeze `input_files` set and bind immutable snapshots instead, or
2. snapshot those records under a frozen namespace before any required post-freeze documentation changes and rehash only those snapshots.

After repair/refreeze, rerun this formal audit before any synthetic/realfold/replay/decision stage.

## High Findings

### H1. Pre-freeze sanitizer lineage records do not satisfy the registered manifest/decision field contract

Severity: **High**

The sanitizer artifacts are physically useful and do not show leakage, but the sanitizer decision/manifest/provenance are not complete under the registered JSON provenance contract. This weakens lineage reproducibility and should not be silently absorbed into a formal PASS audit.

Evidence:

- The experiment plan requires every manifest/decision to include the full metadata and supervision/counter set, including `git_head`, `dirty_diff_sha256`, `conda_env`, runtime versions, config/implementation/verifier hashes, `input_files`, `output_files`, `only_gold_supervision`, `segment_gold_exists`, `segment_gold_used`, and separate forbidden-access counters in [EXPERIMENT_PLAN.md](/data/jehc223/RGCL/refine-logs/lb_scgp/EXPERIMENT_PLAN.md:153).
- `sanitized_provenance.json` lacks `status`, `git_head`, `dirty_diff_sha256`, `conda_env`, runtime versions, config/implementation/verifier hashes, `input_files`, `output_files`, `only_gold_supervision`, `segment_gold_exists`, `segment_gold_used`, and separate `mllm_call_count` / `ocr_call_count` / teacher-cache counters; see the compact record at [sanitized_provenance.json](/data/jehc223/RGCL/artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json:1).
- `sanitizer_decision.json` has `status=PASS` but also lacks the same registered provenance fields and separate zero counters; see [sanitizer_decision.json](/data/jehc223/RGCL/artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json:1).
- The quarantine manifest correctly marks `formal_g0_input=false` but is still a manifest needed for lineage and also lacks the full registered field set; it contains the mixed source path only in quarantine scope at [sanitizer_manifest.json](/data/jehc223/RGCL/artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json:1).
- The sanitizer builder and verifier manually assemble these reduced schemas in [lb_scgp_sanitize_inputs.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_sanitize_inputs.py:208), [lb_scgp_sanitize_inputs.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_sanitize_inputs.py:243), and [lb_scgp_verify_sanitizer.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_verify_sanitizer.py:155).

Impact:

This is not evidence of segment gold or held-label leakage. It is a provenance-contract gap: the formal freeze later carries many of the missing fields, but the pre-freeze sanitizer decision and manifest are the only physical lineage for the quarantine exception. The current records are therefore incomplete for the registered artifact contract.

Minimal repair:

Regenerate the sanitizer lineage under a new no-clobber run/version with the full registered manifest/decision schema, or explicitly amend the experiment plan to define a narrower pre-freeze sanitizer schema before refreezing. Either path requires a new freeze and fresh audit.

## No-Segment-Gold Audit

Status: **PASS**

The hard supervision constraint is respected in the current frozen G0 code paths. The only gold used is the parent-video binary label. I found no live segment/timestamp/span/localization/stance/target/mechanism/rationale gold path.

Evidence:

- The plan states `segment_gold_exists=false`, `segment_gold_used=false`, and forbids G0/G1 subclip artifacts, segment caches, and segment objectives in [EXPERIMENT_PLAN.md](/data/jehc223/RGCL/refine-logs/lb_scgp/EXPERIMENT_PLAN.md:12).
- The frozen config records parent-video-only supervision and no segment gold in [lb_scgp_v1.json](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:10).
- The producer returns `segment=None`, sets `lambda_seg=0.0`, rejects any segment objective, and calls `compute_loss(..., segment_cache=None)` in [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1917), [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1941), [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1944), and [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2000).
- The replay independently enforces the same no-segment path in [lb_scgp_real_replay.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_real_replay.py:206), [lb_scgp_real_replay.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_real_replay.py:210), and [lb_scgp_real_replay.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_real_replay.py:294).
- Sanitizer formal artifacts record `segment_cache_path=null`, `segment_cache_sha256=null`, `segment_artifact_created=false`, and `segment_objective_allowed=false` in [sanitized_provenance.json](/data/jehc223/RGCL/artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json:1) and [sanitizer_decision.json](/data/jehc223/RGCL/artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json:1).

## Other Gate Audits

### Independent Verifier Separation

Status: **PASS by static inspection**

`lb_scgp_independent_verify.py` explicitly imports none of `lb_scgp_common`, `lb_scgp_g0`, SSR solver/projector/ranking/evaluator/factor/rollback implementations in [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1). It independently reconstructs real Dykstra traces, rank-cell objectives, selected trace hashes, exact-vote ledgers, factorization, registered-cone Farkas checks, H10, resource gates, and replay gates.

No self-certifying loophole was found in the reviewed verifier logic, but C1 prevents reaching those stages from the current freeze.

### Formal/Mixed/Quarantine/Protected Path Denial

Status: **PASS for formal G0 surfaces; H1 remains for sanitizer lineage completeness**

Formal scanners define protected prefixes and forbidden key/hash fragments in [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:30), [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:44), and [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:49), and reject formal protected locators in [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:191). G0 scans sanitizer provenance and decision before freeze/use in [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1511). The independent real verifier repeats formal-surface scans in [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1594).

The mixed source path appears only in the quarantine manifest, which is marked `formal_g0_input=false` at [sanitizer_manifest.json](/data/jehc223/RGCL/artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json:1), and not in `CONFIG_FREEZE.json` formal `input_files`.

### Outer-Held Leakage

Status: **PASS by static inspection**

Formal bank access is limited to `memory_ids`, `memory_z`, `memory_labels`, and `query_ids` sentinels; `query_z` and `query_labels` are forbidden in [lb_scgp_v1.json](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:51). Member-only loaders reject forbidden members in [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:811) and [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:839). Realfold checks memory/query sentinel disjointness in [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2105). The independent verifier mirrors the allowed-member policy in [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1022).

### Exact-Vote, Dykstra, Rank-Cell, Farkas, H10, Fit-Replay, and Resource Gates

Status: **PASS by static inspection, not executable from this freeze because of C1**

The frozen code has the expected fail-closed gates:

- `LOCAL_STATIONARY_CERTIFIED` and bounded/REMOVE status controls are registered in [lb_scgp_v1.json](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:78) and [lb_scgp_v1.json](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:87).
- Dykstra projector transitions and correction hashes are emitted by producer code in [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1099) and [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1184), then independently replayed/matched in [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1357) and [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1388).
- Registered cone definition drift raises in producer and verifier at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1368) and [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1456).
- H10 formula is registered in [lb_scgp_v1.json](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v1.json:153), emitted in timings in [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2230), and independently recomputed/gated in [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1670).
- Real replay requires a separate `fit_replay.json` and matches batch order, target-fit steps, realized bank hash, rollback hashes, and no-segment state in [lb_scgp_real_replay.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_real_replay.py:409), [lb_scgp_real_replay.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_real_replay.py:447), and [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1637).
- One-GPU resource gates are emitted in [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2250), checked in [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1666), and the GPU wrapper requests one A100 without `--time` in [lb_scgp_g0_gpu.sbatch](/data/jehc223/RGCL/scripts/slurm/lb_scgp_g0_gpu.sbatch:3) and [lb_scgp_g0_gpu.sbatch](/data/jehc223/RGCL/scripts/slurm/lb_scgp_g0_gpu.sbatch:8).

### No-Clobber and Run Identities

Status: **PASS by static inspection**

Formal publishers use persistent `O_EXCL` publish locks in [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:259), and replay uses the same lock pattern in [lb_scgp_real_replay.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_real_replay.py:72). Run IDs and sealed identities are enforced for freeze, synthetic, realfold, replay, and decision in [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1539), [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1651), [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2089), [lb_scgp_real_replay.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_real_replay.py:402), and [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1691).

The CPU and GPU wrappers activate `HateVideo`, do not specify `--time`, and route formal tasks through SLURM in [lb_scgp_g0_cpu.sbatch](/data/jehc223/RGCL/scripts/slurm/lb_scgp_g0_cpu.sbatch:7), [lb_scgp_g0_cpu.sbatch](/data/jehc223/RGCL/scripts/slurm/lb_scgp_g0_cpu.sbatch:12), [lb_scgp_g0_gpu.sbatch](/data/jehc223/RGCL/scripts/slurm/lb_scgp_g0_gpu.sbatch:8), and [lb_scgp_g0_gpu.sbatch](/data/jehc223/RGCL/scripts/slurm/lb_scgp_g0_gpu.sbatch:13).

## Freeze Hash Binding Summary

`LBSCGP-G0-FREEZE-v1` did freeze the intended code hashes and sanitizer physical artifact hashes:

- config file SHA256: `82e32bfa9fa552744106fd8a5b9c2e07de8e55ab0ad29c35cb5a6907ca744b43`
- `CONFIG_FREEZE.json` file SHA256: `b6697472b61a61706c694a67b21618d618fcad6e7f59265d8696aee79dc46889`
- implementation aggregate SHA256 in freeze: `29cdc2bcf514f88fcde91a0e9bfdd5352f11cf2367e47c8608705a0493ce5df8`
- independent verifier SHA256: `7acdf3f9e24e6c42a0cae2bfd66986927723099ef73f1afec8904b5ad1a01ec9`
- sanitized feature SHA256: `ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496`
- sanitized provenance SHA256: `b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007`
- sanitizer decision SHA256: `172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954`

The blocker is not those stable artifacts. The blocker is that the same freeze also hash-bound mutable audit-trail documents whose current hashes no longer match.

## Minimal Repairs Required Before Re-Audit

1. Replace `LBSCGP-G0-FREEZE-v1` with a new no-clobber freeze lineage whose predecessor checks cannot be invalidated by mandatory progress documentation. Do not edit the existing freeze artifact in place.
2. Fix or explicitly re-specify the sanitizer lineage schema so the pre-freeze sanitizer decision/manifest/provenance satisfy the registered artifact contract or a newly registered narrower pre-freeze contract.
3. Re-run a formal independent code audit after the new freeze. Only if that audit is 0 Critical / 0 High should `artifacts/lb_scgp/v1/g0/code_audit/review.md` and `audit.json` be created.

## Precise Next Authorization

Authorize **repair/refreeze/re-review only**. Do **not** authorize `LBSCGP-G0-CODE-AUDIT-v1` PASS artifacts, `LBSCGP-G0-SYNTH-v1`, `LBSCGP-G0-REAL-MHC_zh-F4-S0-v1`, replay, decision, G1, or any teacher/MLLM/OCR stage from `LBSCGP-G0-FREEZE-v1`.
