# LB-SCGP G0 Formal Code Audit Review V2

Review target: `LBSCGP-G0-FREEZE-v2`, SLURM job `12746`, after v1 formal audit FAIL and v2 repair/freeze.

Reviewer mode: sole independent read-only audit. No subagents, Python execution, SLURM submission, GPU/compute, network, or implementation repair were used. This file is the only artifact written. Because this audit has a Critical finding, no formal PASS artifacts were created under `artifacts/lb_scgp/v2/g0/code_audit/`.

## Verdict

**FAIL**

- Critical findings: **1**
- High findings: **0**
- Formal code-audit PASS artifact authorized: **NO**
- `LBSCGP-G0-SYNTH-v2`, realfold, replay, decision, G1, teacher, MLLM, and OCR authorization: **NO**

## Critical Findings

### C1. v2 dirty-state repair still excludes too little: this mandatory formal review invalidates the downstream decision freeze gate

Severity: **Critical**

The v2 repair correctly removed the original v1 mutable tracker/state records from `input_files` and from the freeze access ledger, but it did not exclude the mandatory v2 formal review record `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V2.md` from the dirty-state predecessor check. The independent decision verifier requires the frozen dirty hash to equal the current dirty hash, and its dirty hash includes untracked/modified files unless they are in the hard-coded formal artifact prefixes, exact mutable audit-trail path list, mutable runtime prefix, or protected prefixes.

That means creating this required review file after freeze changes the dirty state that `LBSCGP-G0-DECISION-v2` will compare against `CONFIG_FREEZE.json`. Even if `artifacts/lb_scgp/v2/g0/code_audit/review.md` and `audit.json` were otherwise valid, the v2 lineage would fail the final independent G0 decision freeze gate before it could authorize G1. This is the same class of predecessor-readiness failure as v1 C1, narrowed but not closed.

Evidence:

- The v2 config excludes only the tracker/state/findings/raw/handoff/execution records at [configs/lb_scgp/lb_scgp_v2.json](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v2.json:36). It does not list `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V2.md`.
- The v2 freeze execution record repeats the same excluded set at [G0_FREEZE_EXECUTION_V2.md](/data/jehc223/RGCL/refine-logs/lb_scgp/G0_FREEZE_EXECUTION_V2.md:191). It does not list the v2 formal audit review record.
- The common dirty-hash implementation excludes only `artifacts/lb_scgp/v1/`, `artifacts/lb_scgp/v2/`, the exact mutable audit-trail paths, and `refine-logs/lb_scgp/runtime/` at [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:59), [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:63), and [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:72).
- The common dirty-hash implementation adds every untracked file that is not excluded or protected into the dirty hash at [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:351) and [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:355).
- The independent decision verifier carries the same narrow exclusion set at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:59), [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:63), and [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:72), and it adds untracked non-excluded files at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:265).
- The independent G0 decision freeze gate requires `freeze.get("dirty_diff_sha256") == current_dirty_hash()` at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1756) and [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1761).
- The frozen v2 dirty hash is already bound in `artifacts/lb_scgp/v2/CONFIG_FREEZE.json` as `65bd94f0b0d351f996272c407bef7fa19835475b61cb46c6302d354e6be7191d` at [CONFIG_FREEZE.json](/data/jehc223/RGCL/artifacts/lb_scgp/v2/CONFIG_FREEZE.json:1). The mandatory review file did not exist at freeze time and is not under `artifacts/lb_scgp/v2/`.

Impact:

This is an audit-readiness blocker, not a numerical-performance issue. The v2 frozen immutable input rows rehash clean, but the downstream independent decision verifier is designed to STOP on dirty drift. A formal PASS artifact would therefore be unsafe because mandatory documentation created after freeze is not guaranteed to preserve the v2 predecessor checks.

Required repair:

Create a new no-clobber freeze lineage. Do not edit the v2 freeze in place. Before refreeze, define and implement a complete documentation policy for all mandatory post-freeze audit/execution records: either exact-list them in the dirty-hash exclusion set or place them under a frozen mutable-audit prefix such as `refine-logs/lb_scgp/runtime/`, with the same policy in producer and independent verifier. Then refreeze and rerun this formal audit.

## High Findings

None.

## Closure Checks

### v1 C1 Closure

Status: **FAIL**

What is closed:

- The v2 formal `input_files` and `access_ledger` omit the v1 mutable tracker/state records. The frozen `input_files` are the v2 config, sanitizer safe records, pre-freeze sanitizer contract snapshot, checkpoint, remove ledger, allowed NPZ member hashes, train-only feature cache, and immutable scientific protocol documents at [CONFIG_FREEZE.json](/data/jehc223/RGCL/artifacts/lb_scgp/v2/CONFIG_FREEZE.json:1).
- The v2 freeze record says the input count is 13 and lists the stable frozen paths at [G0_FREEZE_EXECUTION_V2.md](/data/jehc223/RGCL/refine-logs/lb_scgp/G0_FREEZE_EXECUTION_V2.md:69) and [G0_FREEZE_EXECUTION_V2.md](/data/jehc223/RGCL/refine-logs/lb_scgp/G0_FREEZE_EXECUTION_V2.md:71).
- Shell rehash of all 12 file rows and all 4 allowed NPZ members matched the frozen values: config, sanitizer provenance, sanitizer decision, contract snapshot, checkpoint, remove ledger, train-only feature cache, experiment plan, anchor, final proposal, review summary, refinement report, plus `memory_ids`, `memory_labels`, `memory_z`, and `query_ids`.

What remains open:

- Dirty-state exclusions are not broad enough for the mandatory v2 formal review record, as detailed in Critical C1.

### v1 H1 Closure

Status: **PASS by static and hash inspection**

The v2 repair prospectively defines a dedicated pre-freeze sanitizer schema before v2 freeze and keeps the generic formal schema boundary from `G0_FREEZE` onward.

Evidence:

- `EXPERIMENT_PLAN.md` defines the dedicated pre-freeze sanitizer schema and states that the full generic manifest/decision schema applies from `G0_FREEZE` onward at [EXPERIMENT_PLAN.md](/data/jehc223/RGCL/refine-logs/lb_scgp/EXPERIMENT_PLAN.md:155) through [EXPERIMENT_PLAN.md](/data/jehc223/RGCL/refine-logs/lb_scgp/EXPERIMENT_PLAN.md:164).
- The producer validates required pre-freeze provenance and decision fields, payload hashes, record hash binding, feature hash agreement, zero segment/cache/objective fields, zero teacher/network/formal-query counters, decision gates, and publish locks at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1533) through [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1614).
- The contract snapshot records `registered_before_run_id=LBSCGP-G0-FREEZE-v2`, `dedicated_schema_registered_before_v2_freeze=true`, `full_generic_manifest_schema_applies_from_g0_freeze=true`, all schema-ok flags true, no segment artifact/objective, no held access, zero teacher/MLLM/OCR calls, and protected disclosure locators excluded from formal inputs at [PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json](/data/jehc223/RGCL/refine-logs/lb_scgp/v2/PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json:6), [PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json](/data/jehc223/RGCL/refine-logs/lb_scgp/v2/PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json:85), and [PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json](/data/jehc223/RGCL/refine-logs/lb_scgp/v2/PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json:96).
- The safe sanitizer provenance and decision physically contain the reduced schema fields and zero counters at [sanitized_provenance.json](/data/jehc223/RGCL/artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitized_provenance.json:1) and [sanitizer_decision.json](/data/jehc223/RGCL/artifacts/lb_scgp/inputs/MHC_zh/fold4/sanitizer_decision.json:1).
- The quarantine disclosure record is explicitly `formal_g0_input=false` and carries the protected mixed source lineage only outside formal G0 inputs at [sanitizer_manifest.json](/data/jehc223/RGCL/artifacts/lb_scgp/quarantine/MHC_zh/fold4/sanitizer_manifest.json:1).

Hash evidence:

- `PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json` file SHA256: `cef4c63dd8a9407bd19f14de6e0e81f7816166113889091b670b5eb30ac5e70b`; payload SHA256: `167a1958ed628ba0ac13aa64d149429e701c00f4bf29d158170bbf52ed0ad86c`.
- `sanitized_provenance.json` file SHA256: `b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007`; payload SHA256: `37b9221aee1cb570c2790228854f1889d539148a10d84bea5b9a98b1fca61996`.
- `sanitizer_decision.json` file SHA256: `172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954`; payload SHA256: `8685c805995f97ad0513658c1345b6320dc794e2bad02b907d9a2d07ff16cb1f`.
- `sanitizer_manifest.json` file SHA256: `055dffed9b61053293741ec5ba0ce3577daf458ceef4a3f81143a81d937c684b`; payload SHA256: `576a682da04ebd992d3be2a091404b97e5de8f36b1f74809f05b067ccc728dea`.

### Config-Driven v2 Run Identities

Status: **PASS by static inspection**

- v2 run IDs are registered in config at [lb_scgp_v2.json](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v2.json:4) through [lb_scgp_v2.json](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v2.json:16).
- Producer tasks derive expected run IDs from config at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1512), and freeze/synthetic/realfold reject mismatched run IDs at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1688), [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1814), and [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2251).
- Replay derives the expected replay run ID from config at [lb_scgp_real_replay.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_real_replay.py:400), and the GPU wrapper explicitly admits the v2 replay ID at [lb_scgp_g0_gpu.sbatch](/data/jehc223/RGCL/scripts/slurm/lb_scgp_g0_gpu.sbatch:36).
- The decision verifier derives expected IDs from config and rejects wrong decision IDs at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:223) and [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1746).
- With `CONFIG=configs/lb_scgp/lb_scgp_v2.json`, no v1/v2 cross-lineage acceptance path was found.

### Independent Verifier Separation

Status: **PASS by static inspection**

`lb_scgp_independent_verify.py` states and implements separation from the producer/common modules at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1). It independently verifies synthetic file sets, manifests, Dykstra, rank/exact vote, Farkas, factor and rollback at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1012), and independently verifies the real bank, rank search, exact vote, Farkas/cone, fit replay, resources, cost and no-endpoint gates at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1601).

No self-certifying import path was found.

### Formal, Mixed, Quarantine, Protected, and Outer-Held Denial

Status: **PASS by static inspection**

- Protected prefixes and forbidden formal keys/hash fragments are registered at [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:30), [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:44), and [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:50).
- Formal records reject protected locators at [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:207) and [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:148).
- Formal bank access is restricted to allowed members; forbidden `query_z` and `query_labels` are not opened or hashed by the shared member loader/hashers at [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:835) and [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:863).
- The realfold path enforces memory/held ID disjointness and records no held labels/content access at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2265) and [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2270).
- The independent verifier repeats allowed-member and held-sentinel checks at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1076), [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1662), and [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1675).

### Exact-Vote, Dykstra, Rank, Farkas, H10, Fit-Replay, Resource, and No-Clobber Gates

Status: **PASS by static inspection**

- Synthetic producer gates require expected statuses, Dykstra tolerances, rank-cell fail-closed REMOVE replay, Farkas thresholds, factor thresholds, rollback parity, and finite outputs at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1858) through [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:1917).
- Realfold producer gates require `LOCAL_STATIONARY_CERTIFIED`, exact rank search, no segment objective, factor/Farkas/fit/rollback/resource/cost gates, no endpoint, and strict H10 `<160` at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2427) through [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2451).
- The independent verifier recomputes Dykstra transitions/corrections at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1232), rank-cell objective/hash/trace selection at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1375), exact vote at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1151), registered-cone/Farkas at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1500), and H10/resource/replay gates at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1643), [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1693), and [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1720).
- Formal no-clobber publication uses persistent `O_CREAT|O_EXCL` locks in [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:275) and [lb_scgp_real_replay.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_real_replay.py:69).

## No-Segment-Gold Audit

Status: **PASS**

The hard supervision invariant is respected in the reviewed v2 G0 surfaces. The only gold supervision is the parent-video binary label. I found no live segment/timestamp/span/localization/stance/target/mechanism/rationale gold path, no segment/subclip artifact under `artifacts/lb_scgp`, and no code path that passes a segment cache/objective into the G0 producer or replay.

Evidence:

- The plan states the only gold is the parent-video binary label, `segment_gold_exists=false`, `segment_gold_used=false`, and no segment/timestamp/span/localization/stance/target/mechanism/rationale gold exists or may be assumed at [EXPERIMENT_PLAN.md](/data/jehc223/RGCL/refine-logs/lb_scgp/EXPERIMENT_PLAN.md:12).
- The problem anchor repeats the invariant at [PROBLEM_ANCHOR.md](/data/jehc223/RGCL/refine-logs/lb_scgp/PROBLEM_ANCHOR.md:6) and [PROBLEM_ANCHOR.md](/data/jehc223/RGCL/refine-logs/lb_scgp/PROBLEM_ANCHOR.md:9).
- v2 config records parent-video-only supervision at [lb_scgp_v2.json](/data/jehc223/RGCL/configs/lb_scgp/lb_scgp_v2.json:61).
- The common module declares no segment/timestamp/span/localization/stance/target/mechanism/rationale gold interface at [lb_scgp_common.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_common.py:4).
- The realfold loader returns `segment=None` and a cache contract with `segment_cache_opened=false`, `segment_objective_allowed=false`, and zero held reads at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2080).
- Producer repo args set `lambda_seg=0.0`, `_assert_no_segment_objective` fails closed, and `compute_loss` receives `segment_cache=None` at [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2105), [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2108), and [lb_scgp_g0.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_g0.py:2164).
- Replay independently sets `segment=None`, fails closed on nonzero `lambda_seg`, passes `segment_cache=None`, and emits `segment_cache_used=false` at [lb_scgp_real_replay.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_real_replay.py:210), [lb_scgp_real_replay.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_real_replay.py:235), [lb_scgp_real_replay.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_real_replay.py:294), and [lb_scgp_real_replay.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_real_replay.py:459).
- The independent verifier requires the freeze and audit objects to keep `only_gold_supervision=parent_video_binary_label`, `segment_gold_exists=false`, and `segment_gold_used=false` at [lb_scgp_independent_verify.py](/data/jehc223/RGCL/scripts/analysis/lb_scgp_independent_verify.py:1794).

## Hash Binding Summary

- v2 config file SHA256: `eec778811cfd2cf72a21dbf55af1c768ac6f849234350d958e900f871e41154f`
- v2 `CONFIG_FREEZE.json` file SHA256: `4c6f0199a429bbf766cb284b9ecaedd9dcaf38733834dd54574245ab86b633ae`
- v2 `CONFIG_FREEZE.json` payload SHA256: `bda18d7dfda00ab5808595afec04e5b925d3a4920f55fe963dec5e9d99795c0b`
- v2 `CONFIG_FREEZE.json.publish.lock` SHA256: `22426f693ba3f72d52928b0a86d08fe439d7e7fd43f9c74c61aedb710443e211`
- freeze `config_canonical_sha256`: `3c7e391ca8e35fffa76ebcfc76a1c9c6e7c76c92bcc2dc08faa7e9a72be7cb1b`
- freeze `implementation_sha256`: `51fc1cee40f489e98e82c4aac93799015ac0ad7918e2847c6dbb7e0596890aef`
- freeze `independent_verifier_sha256`: `8ab99bad45daea1963dd030c24f91c28c87b46631d46e6fcafa4b3e3e102a4f6`
- freeze `access_ledger_sha256`: `8621b77bc8ad852634c4325c8c69740e454c7f2366bc03bd175f1d99973348c7`
- sanitizer provenance file SHA256: `b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007`
- sanitizer decision file SHA256: `172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954`
- pre-freeze sanitizer contract snapshot file SHA256: `cef4c63dd8a9407bd19f14de6e0e81f7816166113889091b670b5eb30ac5e70b`
- train-only feature cache SHA256: `ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496`
- allowed NPZ member SHA256 values rehashed and matched:
  - `memory_ids`: `2db3c268e6efc0ad75ef6a483a7fb97fb992e0ec5b28e8b27dc7963d9ed28193`
  - `memory_labels`: `4666d718b4e2ea1dd66c7f55aa9c69da49dac5f8f64f83f52594656547cc92c5`
  - `memory_z`: `af336d46c81de46c6a88863fa95469dead690f0e5da8d6baefb4a0355041e2de`
  - `query_ids`: `d12166e5900780c81561996f83baa1c82cc79e4ebc7d0ae8ee12be5c7087aff7`

## Next Authorization

Authorize **repair/refreeze/re-review only**.

Do **not** authorize `LBSCGP-G0-CODE-AUDIT-v2` PASS artifacts, `LBSCGP-G0-SYNTH-v2`, `LBSCGP-G0-REAL-MHC_zh-F4-S0-v2`, `LBSCGP-G0-REAL-REPLAY-MHC_zh-F4-S0-v2`, `LBSCGP-G0-DECISION-v2`, G1, teacher, MLLM, OCR, or any performance claim from `LBSCGP-G0-FREEZE-v2`.
