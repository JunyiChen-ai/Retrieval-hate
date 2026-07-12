# LB-SCGP G0 Formal Code Audit Review V3

Review target: `LBSCGP-G0-FREEZE-v3`, frozen by SLURM job `12748`.

Reviewer mode: sole independent formal code audit. No subagent, sidecar agent,
dynamic workflow, nested Codex process, implementation repair, config edit,
synthetic run, realfold run, replay run, decision run, G1 run, teacher call,
MLLM call, OCR call, or performance task was used.

## Verdict

**PASS_REVIEW_ONLY**

- Critical findings: **0**
- High findings: **0**
- Important findings / authorization notes: **2**
- No-segment-gold audit: **PASS**
- v2 Critical closure: **CLOSED for v3**
- Formal PASS artifacts created: **NO**
- Reason PASS artifacts are absent: no existing LB-SCGP protocol-authorized
  producer/task was found for `artifacts/lb_scgp/v3/g0/code_audit/{review.md,audit.json}`;
  the repository only consumes that artifact as a downstream predecessor.
- Executable boundary: v3 implementation/freeze lineage is review-passed, but
  later G0 stages remain locked until a real authorized no-clobber audit
  artifact producer is supplied or explicitly authorized. Hand-written PASS
  JSON is not permitted.

## Critical Findings

None.

## High Findings

None.

## Important Findings

### I1. No authorized formal PASS artifact producer exists

The v3 audit has 0 Critical and 0 High, but I did not create
`artifacts/lb_scgp/v3/g0/code_audit/audit.json` or `review.md`.

Evidence:

- `lb_scgp_g0.py` requires `artifacts / "g0/code_audit/audit.json"` before
  `synthetic` or `realfold` can proceed, and checks `run_id`,
  `stage=G0_CODE_AUDIT`, `status=PASS`, `critical=0`, `high=0`,
  implementation hash, and config hash.
- `lb_scgp_independent_verify.py` also consumes the same audit artifact during
  final decision verification.
- Repository search found no LB-SCGP task that produces the code-audit PASS
  artifact through a schema/no-clobber path. Existing references describe the
  artifact as a required gate, not as a producible task.

Impact: not a v3 code defect, but an authorization/tooling boundary. Synthetic,
realfold, replay, decision, G1, teacher, MLLM, and OCR remain locked.

### I2. Frozen scientific protocol text still contains historical v1/v2 wording

The v3 lineage is unambiguous in `configs/lb_scgp/lb_scgp_v3.json`,
`G0_V3_REPAIR_HANDOFF.md`, `G0_FREEZE_EXECUTION_V3.md`, the v3 freeze payload,
and the tracker. Some frozen scientific input documents still contain older
v1/v2 wording because they were intentionally frozen as immutable scientific
protocol inputs.

Impact: not a blocker for v3, because run IDs, namespace, dirty policy,
implementation hashes, freeze inputs, counters, and artifact lineage are
config-driven and bound by `CONFIG_FREEZE.json`. Future documentation should
avoid treating those stale status sentences as current authorization.

## Audit-Only SLURM

- Job `12750`, `lbscgp_audit_dirty_v3`: audit-only hash/canonicalization check.
  It imported `lb_scgp_common` and `lb_scgp_independent_verify` under SLURM in
  `HateVideo`, read the v3 config, and computed producer/verifier dirty hashes.
  State `COMPLETED`, exit `0:0`, elapsed `00:00:01`, allocation `2 CPU / 4G`.
  Log: `/tmp/lbscgp_audit_dirty_v3_12750.out`, SHA256
  `a3143869abd4a222477a00f75efaf9ca91c9a4c0b6cdaca5a6f1edcd859133e6`.
- Job `12751`, `lbscgp_audit_dirty_v3`: audit-only post-report-create
  dirty-hash check. State `COMPLETED`, exit `0:0`, elapsed `00:00:01`,
  allocation `2 CPU / 4G`. Log:
  `/tmp/lbscgp_audit_dirty_v3_12751.out`, SHA256
  `634159a84941c19688330a410500e7e27aa7df6529d363797762fd7426d61f1e`.
- Job `12752`, `lbscgp_audit_dirty_v3`: audit-only post-report-update
  dirty-hash check. State `COMPLETED`, exit `0:0`, elapsed `00:00:01`,
  allocation `2 CPU / 4G`. Log:
  `/tmp/lbscgp_audit_dirty_v3_12752.out`, SHA256
  `3daf7215abbb62b63a238009a07d19578b77577c9bd3891f2b42c6a9819c7fd2`.
- Job `12753`, `lbscgp_audit_dirty_v3`: audit-wrapper attempt for final
  dirty-hash check. State `FAILED`, exit `1:0`, elapsed `00:00:00`,
  allocation `2 CPU / 4G`. It failed before producing an audit value because
  the wrapper called a non-existent helper name. Log:
  `/tmp/lbscgp_audit_dirty_v3_12753.out`, SHA256
  `37d9ec45455a6eb7d3404a8d7158ee5126a9d9fa4d9b90e98ba6be48c23dd893`.
- Job `12754`, `lbscgp_audit_dirty_v3`: audit-wrapper attempt for final
  dirty-hash check. State `FAILED`, exit `1:0`, elapsed `00:00:01`,
  allocation `2 CPU / 4G`. It failed before producing an audit value because
  the wrapper called the producer hash helper by the verifier helper name. Log:
  `/tmp/lbscgp_audit_dirty_v3_12754.out`, SHA256
  `a0ca19220ab6cd134b7bd7126360a4ea6468a80a2dc360b4e4f7bbebb1139013`.
- Job `12755`, `lbscgp_audit_dirty_v3`: audit-wrapper attempt for final
  dirty-hash check. State `FAILED`, exit `1:0`, elapsed `00:00:00`,
  allocation `2 CPU / 4G`. It failed before producing an audit value because
  the wrapper passed a config path where the helper expects the parsed config.
  Log: `/tmp/lbscgp_audit_dirty_v3_12755.out`, SHA256
  `bf06b0f07ead3f957c18e45f284b50996fbf20947c17a914c99f937be23c9069`.
- Job `12756`, `lbscgp_audit_dirty_v3`: audit-only final post-documentation
  dirty-hash check using the parsed v3 config. State `COMPLETED`, exit `0:0`,
  elapsed `00:00:02`, allocation `2 CPU / 4G`. Log:
  `/tmp/lbscgp_audit_dirty_v3_12756.out`, SHA256
  `10f640e8c14c1221c13d91ebad1b12780e83d16d524b8e3b3a6da6940d42b9fc`.

No synthetic, realfold, replay, decision, G1, teacher, MLLM, OCR, or
performance job was submitted.

## Dirty Hash Proof

- Frozen v3 dirty hash:
  `91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1`
- Before report creation, producer dirty hash:
  `91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1`
- Before report creation, verifier dirty hash:
  `91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1`
- After report creation:
  `91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1`
- After report update:
  `91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1`
- Final:
  `91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1`

The pre-report check was executed in SLURM job `12750`. The post-create check
was executed in SLURM job `12751`. The post-update check was executed in SLURM
job `12752`. Final post-documentation recomputation was executed in SLURM job
`12756`. Failed wrapper attempts `12753`--`12755` did not produce audit values
and did not run any experiment stage.

## Lineage And Frozen Input Evidence

v3 is an independent no-clobber namespace:

- Config SHA256:
  `a480c9b9bf56c938667b4f8e2f3d07882b84843627233b613d864764c02eaf47`
- `CONFIG_FREEZE.json` SHA256:
  `9fba7f1649dd67d4bb0fcc193e555d8246d7a4966307732e87d5e9fca7346dd9`
- `CONFIG_FREEZE.json.publish.lock` SHA256:
  `9c32d07c524e466ad06c06fc2a472829764cd1facff22390eac8b2879d329b8f`
- Freeze payload SHA256:
  `352ec2215e2225b1768a13f39f96ef935b91966606d8db874d6de4410b1a9f3d`
- Config canonical SHA256:
  `84227b68eaa496da6e307ce5c5ef3469e1b7c68e350f0d62d1677d01f07645bf`
- Implementation SHA256:
  `b8759436a6c5e2a67bf7125cbd1ab57cb05187e764e837373abfdf1a92916e75`
- Independent verifier SHA256:
  `d1e50057b4c166a71426f89474b6526e3eab11da5547e15368112b6620dbf5ce`
- Access ledger SHA256:
  `3db4b94900a9d9b807ab495be869a5ef87a3894f987eef03ea1e948030abdc72`

v1/v2 were not overwritten by v3. Current hashes match the prior formal audit
records:

- v1 freeze SHA256:
  `b6697472b61a61706c694a67b21618d618fcad6e7f59265d8696aee79dc46889`
- v1 lock SHA256:
  `34a05bde46775bcb75384c1c853cef7985f5c05efcdf3afec5fad6d85ef57d8d`
- v2 freeze SHA256:
  `4c6f0199a429bbf766cb284b9ecaedd9dcaf38733834dd54574245ab86b633ae`
- v2 lock SHA256:
  `22426f693ba3f72d52928b0a86d08fe439d7e7fd43f9c74c61aedb710443e211`

All v3 frozen file inputs were independently rehashed and matched:

- `configs/lb_scgp/lb_scgp_v3.json`:
  `a480c9b9bf56c938667b4f8e2f3d07882b84843627233b613d864764c02eaf47`
- `sanitized_provenance.json`:
  `b921477c2cc8858f2f9dfe9b6da21a0aaea2287fdaf9059c2f1dba08010d8007`
- `sanitizer_decision.json`:
  `172c9db7589c5b80af7fe6f8476dd9866a4eb840bc1b4524a79b6010c2c3c954`
- `PRE_FREEZE_SANITIZER_CONTRACT_SNAPSHOT.json`:
  `cef4c63dd8a9407bd19f14de6e0e81f7816166113889091b670b5eb30ac5e70b`
- `checkpoint_epoch28.pt`:
  `c135924c87d1b12218f332ed8b955795cc34d6da972a576c011820572c3d0a39`
- `train.jsonl`:
  `970b4c55319c83bffc4659976d885ecf89ff9043d659898fa175156a3736a9f6`
- `outer_train_features.pt`:
  `ea5f0ace7fa614b243269e155ef12e44cfa646f7e2063ec7f0d7aaee11d87496`
- `EXPERIMENT_PLAN.md`:
  `9eb1f8c69d0a0e1b7c967b658b9a6d11af38b7653348bb38e2b7c2c6b25c2bc7`
- `PROBLEM_ANCHOR.md`:
  `254f4c68fdf578239b952222f4e72378b8a110df21e85158c32b3cbc3b90200d`
- `FINAL_PROPOSAL.md`:
  `94d7b6e9305e8c6095e0a9f20351bb4cafff042f7aa048f2a934ac6d1a3a0a0c`
- `REVIEW_SUMMARY.md`:
  `1eb8cf3c03fc168b42962d923ca3de2978895f5d968263664f984c543487d0df`
- `REFINEMENT_REPORT.md`:
  `4144fa344c385211d25aae697edbf5744de4e2b10f90c2cd4cede0f3b5b56ef7`

Allowed NPZ member hashes were recomputed by reading only these members:

- `memory_ids`: `2db3c268e6efc0ad75ef6a483a7fb97fb992e0ec5b28e8b27dc7963d9ed28193`
- `memory_labels`: `4666d718b4e2ea1dd66c7f55aa9c69da49dac5f8f64f83f52594656547cc92c5`
- `memory_z`: `af336d46c81de46c6a88863fa95469dead690f0e5da8d6baefb4a0355041e2de`
- `query_ids`: `d12166e5900780c81561996f83baa1c82cc79e4ebc7d0ae8ee12be5c7087aff7`

`query_z.npy` and `query_labels.npy` are present in the archive but were not
opened or hashed.

## Dirty-Exclusion Audit

Producer/common and independent verifier dirty-state logic is identical in
meaning for v3:

- For lineage versions outside `v1`/`v2`, both require explicit
  `formal_artifact_exclude_prefixes`, `dirty_state_excluded_paths`, and
  `dirty_state_excluded_prefixes`.
- Both normalize relative entries lexically, reject absolute paths and parent
  traversal forms, and require `artifact_namespace` to be present in formal
  artifact exclusions.
- Both exclude only the config-bound formal artifact prefixes for tracked,
  staged, and untracked dirty state.
- Both exclude exact dirty paths by equality for untracked files and by exact
  pathspec for tracked/staged diffs.
- Both exclude dirty prefixes only with the config-bound prefix set.
- Both hard-exclude protected storage prefixes and `slurm/logs/**` from dirty
  state.

v3 config values are narrow:

- Formal artifact prefixes:
  `artifacts/lb_scgp/v1/`, `artifacts/lb_scgp/v2/`, `artifacts/lb_scgp/v3/`
- Exact mutable dirty paths:
  `refine-logs/lb_scgp/EXPERIMENT_TRACKER.md`, `TARGET_LOOP.md`,
  `TARGET_STATE.json`, `TARGET_FINDINGS.md`, `TARGET_REVIEW_RAW.md`,
  `refine-logs/lb_scgp/G0_V3_REPAIR_HANDOFF.md`,
  `refine-logs/lb_scgp/G0_FREEZE_EXECUTION_V3.md`,
  `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V3.md`
- Dirty prefix:
  `refine-logs/lb_scgp/runtime/`

No broad `refine-logs/lb_scgp/` or repository-root exclusion was found. The
actual report path is excluded as one exact path, not as a report-name prefix.
Sibling names such as `G0_FORMAL_CODE_AUDIT_REVIEW_V3.md.bak` are not excluded
by the untracked equality check.

## Run-ID And Wrapper Audit

Static checks passed:

- `bash -n` on all three LB-SCGP SLURM wrappers: pass.
- `jq empty` on v1/v2/v3 configs, v3 freeze, sanitizer records, contract
  snapshot, and quarantine manifest: pass.
- `git diff --check` on reviewed LB-SCGP code/config/wrappers: pass.

Run-ID guard status:

- CPU wrapper accepts only `freeze`, `synthetic`, and `decide`; Python task code
  checks the config-derived run ID for each task.
- GPU wrapper accepts only `realfold` and `replay`, rejects non-`MHC_zh` or
  non-fold-4, and quotes all shell variables used as command arguments.
- Replay guard parses `lineage.run_ids.replay` from `CONFIG` inside the SLURM
  job. Missing, malformed, or non-string replay run ID exits nonzero before
  `lb_scgp_real_replay.py` is called. Wrong `RUN_ID` exits before replay.
- No `#SBATCH --time` directive is present; only comments mention no `--time`.

## Schemas, No-Clobber, And Protected-Surface Audit

The v3 freeze payload, sanitizer safe records, and sanitizer contract snapshot
all have valid newline-free canonical payload hashes. The v3 freeze
`access_ledger_sha256` also matches canonical recomputation.

No-clobber is implemented with persistent `O_CREAT|O_EXCL` publish locks for
formal JSON/JSONL and replay paths. Current locks exist for v1/v2/v3 freeze and
sanitizer artifacts.

Formal protected-surface checks reject quarantine/mixed/source/subclip keys,
protected locators, and prohibited hash keys. The quarantine manifest contains
source lineage and mixed-cache evidence, but it is explicitly
`formal_g0_input=false` and is not a v3 formal input. Formal v3 inputs are the
safe sanitizer provenance, safe sanitizer decision, train-only feature cache,
the v2 pre-freeze sanitizer contract snapshot, frozen checkpoint/bank member
hashes, remove ledger, and immutable scientific protocol documents.

## No-Segment-Gold Audit

**PASS.**

The binding invariant remains:

- `only_gold_supervision=parent_video_binary_label`
- `segment_gold_exists=false`
- `segment_gold_used=false`
- inherited parent labels are not segment gold

Evidence:

- The experiment plan and problem anchor explicitly forbid assuming segment,
  timestamp, span, localization, stance, target, mechanism, or rationale gold.
- v3 config and v3 freeze both bind parent-video-only supervision and false
  segment-gold fields.
- Sanitizer safe provenance and decision have `segment_cache_path=null`,
  `segment_cache_sha256=null`, `segment_artifact_created=false`, and
  `segment_objective_allowed=false`.
- No `segment` or `subclip` artifact exists under `artifacts/lb_scgp`.
- Producer realfold loader returns `segment=None`, rejects segment-cache
  metadata, sets `lambda_seg=0.0`, and passes `segment_cache=None`.
- GPU replay independently sets `segment=None`, requires `lambda_seg=0.0`, and
  emits `segment_cache_used=false`.
- Independent verifier requires the same no-segment fields in synthetic/real
  manifests, replay, freeze, and audit objects.

No segment-level gold annotation, segment objective, held-label read,
validation/test read, teacher cache read/write, MLLM call, or OCR call was found
in the reviewed G0 v3 surfaces.

## Numerical-Gate Static Audit

No synthetic, realfold, replay, or decision performance stage was run in this
audit. Static inspection found fail-closed gates for:

- Dykstra persistent correction traces, set order, replayed transitions,
  `LOCAL_STATIONARY_CERTIFIED` thresholds, and bounded/infeasible controls.
- PSD symmetrization, unit diagonal, off-diagonal box, row/class/semantic/slack
  projectors, rank halfspaces, and vote halfspaces.
- Complete top-20 exact-vote semantics with canonical-ID tie handling, self
  exclusion, weighted signed vote, simultaneous ties, orientation over-budget,
  pivot over-budget, unresolved tie map, and incomplete adjacent enumeration
  mapping to REMOVE with direct REMOVE replay hashes.
- Deterministic PSD factorization, negative-eigenvalue reject, repeated/null
  eigenspace basis, and Procrustes alignment.
- Registered-cone/Farkas checks with abstract and realized displacements,
  universe hashes, family witness checks, separation, and duality-gap gates.
- Runtime/resource/cost gates including one GPU, peak GPU/host limits, exact
  H10 formula, finite values, and strict `<160 GPU-hours`.
- Deterministic randomness, rollback restoration of model/optimizer/scheduler/
  scaler/RNG/sampler/epoch cursor, and independent GPU replay of fit/rollback.

REMOVE controls are present in G0 synthetic/real rollback gates. SHUFFLE, NOISE,
and DIRECT controls are specified for later locked stages, not executed or
claimed by v3 G0.

## v2 Critical Closure

The exact v2 Critical was that the mandatory formal review file
`G0_FORMAL_CODE_AUDIT_REVIEW_V2.md` was not excluded from the dirty-state
predecessor check. v3 closes that defect for the new lineage:

- `G0_FORMAL_CODE_AUDIT_REVIEW_V3.md` is listed as an exact dirty-state excluded
  path in v3 config and v3 freeze.
- Producer/common and independent verifier share the same config-driven
  semantics.
- Frozen and pre-report producer/verifier dirty hashes match:
  `91cf2890acc543fdb2f3988f5063461f70d855469df386908436b4273054a4b1`.
- The exclusion is not broad enough to hide arbitrary review-sibling files.

The post-create dirty-hash check passed after this real report was created.
The post-update dirty-hash check also passed after this real report was updated.
The final check remains pending until the required tracker/TARGET documents are
updated.

## Forbidden Runs And Artifacts

No synthetic, realfold, replay, decision, G1, teacher, MLLM, OCR, or performance
job was submitted by this audit. At initial report creation,
`artifacts/lb_scgp/v3` contains only:

- `CONFIG_FREEZE.json`
- `CONFIG_FREEZE.json.publish.lock`

No `artifacts/lb_scgp/v3/g0/`, `G0_DECISION.json`, synthetic, real, replay, or
formal PASS artifact exists.

## Finalization Notes

Final changed files from this audit are limited to:

- `refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V3.md`
- `refine-logs/lb_scgp/EXPERIMENT_TRACKER.md`
- `TARGET_STATE.json`
- `TARGET_FINDINGS.md`
- `TARGET_LOOP.md`

The final report file SHA256 is reported outside this file in the final
response, because embedding this file's own SHA256 would change the file hash.
