# LB-SCGP G0 Formal Code Audit Review V5

Review target: `LBSCGP-G0-FREEZE-v5`.

Reviewer mode: fresh independent LB-SCGP G0 v5 auditor/publisher in the main
thread. No subagent, sidecar agent, dynamic workflow, nested Codex process,
implementation repair, config edit, protocol edit, threshold edit, data edit,
synthetic run, realfold run, replay run, decision run, G1 run, teacher call,
MLLM call, OCR call, held-label read, held-content read, validation read, test
read, or performance task was used. The externally visible auditor session ID
is not exposed to this process; SLURM job IDs are listed below.

## Verdict

**PASS.**

- Critical findings: **0**
- High findings: **0**
- Important findings / authorization notes: **3**
- No-Segment-Gold audit: **PASS**
- Formal PASS authorization: **YES, if and only if the strict sidecar record
  binds this exact review file hash and the frozen publisher succeeds unchanged**
- Authorized next boundary after verified publication and post-verification:
  **synthetic only**
- Still locked after publication: **realfold, replay, decision, G1, G2,
  teacher, MLLM, OCR, validation, test, held-label/content access, and all
  performance claims**

The only gold supervision is `parent_video_binary_label`.
`segment_gold_exists=false` and `segment_gold_used=false` are binding. There is
no segment-level gold, and none may be assumed, synthesized, cached, or used.

## Critical Findings

None.

## High Findings

None.

## Important Findings

### I1. v5 closes the v4 producer NameError on the exact failed path

The v4 operational failure was not a scientific failure claim; it was a
post-publication consumer crash. SLURM log `12786` shows
`_load_freeze_and_audit -> _verify_v4_code_audit_schema` raising
`NameError: name 'git_state' is not defined` while checking the audit dirty
binding.

In v5, `scripts/analysis/lb_scgp_g0.py` imports `git_state` from
`lb_scgp_common`, and the same strict verifier still calls `git_state(cfg=cfg)`
at the dirty check. Independent SLURM job `12825` exercised
`lb_scgp_g0._load_freeze_and_audit` and `_verify_v4_code_audit_schema` on a
valid v5 strict-schema fixture; the producer consumer succeeded. The same job
rejected a v4-type fallback (`artifact_type`, `lineage`, and `record_values`
failed), wrong run ID, wrong dirty hash, wrong review-record hash, schema
extra fields, prior-lineage hash drift, segment-gold drift, and publication
index path drift. Therefore the v4 NameError root cause is closed for the v5
producer path and v5 cannot silently accept a generic or v4 fallback.

### I2. v5 remains a narrow lineage/tooling publication

The v5 repair adds no scientific method, threshold, supervision, data protocol,
projector, Dykstra, rank-cell, factorization, Farkas/cone, fit, rollback,
cost, or evaluation change. The v4 formal bundle is frozen only as failed
prior-lineage no-clobber evidence. It is not PASS authority for v5.

The publisher and consumers are lineage-aware for v4/v5 and fail closed for
other lineages. v5 publication must create exactly
`artifacts/lb_scgp/v5/g0/code_audit/` with `review.md`,
`review_record.json`, `audit.json`, `publication_index.json`, and one
`.publish.lock` per file. No formal v5 code-audit output existed before this
review was finalized.

### I3. Publication unlock is deliberately synthetic-only

A verified v5 code-audit publication authorizes only the synthetic G0 stage.
It does not authorize realfold, replay, decision, G1, teacher, MLLM, OCR,
validation, test, held-label/content access, or any performance claim. A fresh
authorization boundary remains required after synthetic.

## Frozen Lineage Evidence

- Config path: `configs/lb_scgp/lb_scgp_v5.json`
- Artifact namespace: `artifacts/lb_scgp/v5`
- Freeze artifact SHA256:
  `254e45afe9c0355892824c0c26bc73b4b0854cb20c67c3703982762fad010931`
- Freeze lock SHA256:
  `54dc06d236d5fc1f3ac96400f1a81faeb1d2c0c8e5af075065d1964260de98a9`
- Freeze payload SHA256:
  `d89f7cd4ad43c8ef83a04b7530ec19186ec1cb0e96958d239f1ce5c2b146bb4d`
- Config file SHA256:
  `a51981045073e8f5b69da272654d2102ef3f2f5c8739b765d0b161c1f8c75346`
- Config canonical SHA256:
  `4a45fb6c66884b6b8aa4571961dff3ef7751c2b9f97e2df1584521cfe1eb3dba`
- Implementation SHA256:
  `939acffbafbd9204fc654972cd73f174393c8466c61f4af045b1c20948a6b687`
- Independent verifier SHA256:
  `f0f49f41de4efee9abf2267b27b75be440f0020583baef565955c3d0c2988b2d`
- Freeze access ledger SHA256:
  `ce6898035cf25dbe53f4b258a7b792796e4388a282384185f13a84926397ea0f`
- Frozen dirty SHA256:
  `1c8284781fb57e90714b390fdbef362e978b70789632b19df3d8161dfe8827b7`

Protected v1-v4 and v4 formal failed-lineage hashes were rechecked before
publication authorization and matched the v5 config's exact
`prior_lineage_no_clobber_hashes` set.

## Strict Schema and Publication Toolchain

The producer, publisher, and decision-side audit consumer use the same strict
v5 contract:

- strict record type:
  `LB_SCGP_G0_CODE_AUDIT_INDEPENDENT_REVIEW_RECORD_V5`
- strict artifact type:
  `LB_SCGP_G0_CODE_AUDIT_PASS_ARTIFACT_V5`
- strict publication index type:
  `LB_SCGP_G0_CODE_AUDIT_PUBLICATION_INDEX_V5`

The publisher recomputes and binds the freeze file and payload hashes, freeze
lock hash, config file and canonical hashes, implementation hash and file set,
independent verifier hash, git head, current dirty hash, frozen dirty hash,
frozen inputs, allowed NPZ member hashes only, prior-lineage no-clobber hashes,
wrapper contract, strict review-record hash, strict review-record payload hash,
review file hash, no-segment supervision fields, zero counters, and output
locks.

Only the allowed NPZ members are verified: `memory_ids`, `memory_labels`,
`memory_z`, and `query_ids`. The forbidden held arrays `query_z` and
`query_labels` remain unopened.

The transaction path is atomic and no-clobber: the publisher writes a hidden
`.code_audit.publish.tmp.<job>.<pid>` directory, fsyncs every output and lock,
renames it into place, and removes the temp directory on failure. The final
directory and every file/lock path are refused if preexisting.

## Scientific Binding Audit

Static inspection found the G0 scientific contracts fail closed:

- The producer requires SLURM and conda `HateVideo`.
- Dykstra uses persistent corrections, fixed set order, transition hashes,
  local-stationarity thresholds, and bounded fallback statuses.
- PSD projection symmetrizes input, clips numerical negatives only, enforces
  unit diagonal separately, and preserves the off-diagonal box.
- Rank cells use exact top-20 vote semantics, self exclusion, canonical-ID
  tie handling, adjacent inequality checks, boundary orientation limits, pivot
  limits, unresolved tie map fail-closed behavior, and direct REMOVE replay
  hashes.
- Factorization rejects eigenvalues below `-1e-7`, handles repeated/null
  eigenspaces deterministically, and verifies Procrustes alignment.
- Registered-cone/Farkas checks bind singleton, pair, triplet, and SupCon
  families using parent-video binary labels only.
- REMOVE rollback restores model, optimizer, scheduler, scaler, RNGs, sampler
  cursor, and epoch cursor before direct replay comparison.
- Realfold and replay code set `segment=None`, require `lambda_seg=0.0`, pass
  `segment_cache=None`, and reject segment objective/cache metadata.

## No-Segment-Gold Audit

**PASS.**

Evidence:

- Config and freeze bind `only_gold_supervision=parent_video_binary_label`,
  `segment_gold_exists=false`, and `segment_gold_used=false`.
- Freeze counters are all zero:
  `mllm_call_count=0`, `ocr_call_count=0`,
  `teacher_cache_read_count=0`, `teacher_cache_write_count=0`,
  `outer_held_label_read_count=0`, `outer_held_content_read_count=0`,
  `val_content_read_count=0`, `test_content_read_count=0`,
  `val_test_teacher_artifact_count=0`, and
  `formal_model_optimizer_evaluator_outer_held_read_count=0`.
- Sanitized provenance and sanitizer decision bind
  `segment_cache_path=null`, `segment_cache_sha256=null`,
  `segment_artifact_created=false`, `segment_objective_allowed=false`,
  `teacher_mllm_ocr_calls=0`, `network_external_calls=0`,
  `formal_query_z_read_count=0`, `formal_query_labels_read_count=0`, and
  `formal_model_optimizer_evaluator_outer_held_read_count=0`.
- `query_ids` are held-ID exclusion sentinels only; `query_z` and
  `query_labels` are forbidden.
- No segment annotation, subclip objective, timestamp/span/localization/
  stance/target/mechanism/rationale gold, teacher call, MLLM call, OCR call,
  validation/test content read, or formal held-label/content read was found in
  the G0 v5 authorized surfaces.

## Audit-Only SLURM Checks

No audit-only job ran synthetic, realfold, replay, decision, teacher, MLLM,
OCR, validation, test, held-label, held-content, or performance stages.

- Job `12824`, `lbscgp_v5_ind_audit_neg`: FAILED `2:0`, elapsed `00:00:04`.
  This was a harness issue: the temporary fixture intentionally moved
  `paths.artifacts`, which the decision verifier correctly rejected as
  artifact namespace/config path drift, and one wrong-hash mutator was applied
  too early. The job left no formal v5 output.
- Job `12825`, `lbscgp_v5_ind_audit_neg`: COMPLETED `0:0`, elapsed
  `00:00:04`, 2 CPU / 4G. Result:
  `refine-logs/lb_scgp/runtime/v5_independent_audit/negative_checks_12825.json`.

Job `12825` PASS matrix:

- valid v5 strict schema: producer `_load_freeze_and_audit` succeeded
- wrong review-record hash: fail-closed
- wrong dirty hash: fail-closed
- wrong code-audit run ID: fail-closed
- audit extra field: fail-closed
- review-record extra field: fail-closed
- v4 artifact/record/lineage fallback: fail-closed
- prior-lineage hash drift: fail-closed
- segment-gold drift: fail-closed
- publication-index path drift: fail-closed
- wrapper wrong `TASK`: exit 2 before Python publication
- wrapper wrong `RUN_ID`: exit 2 before Python publication
- wrapper wrong review path: exit 2 before Python publication
- transaction existing-output/no-clobber: fail-closed in runtime fixture
- transaction forced rename failure: temp residue cleaned
- transaction positive runtime fixture: exact four files and four locks, then
  fixture removed
- protected v1-v4 hashes: unchanged
- v5 freeze and lock hashes: unchanged
- pre-publication v5 formal residue: absent

## Publication Authorization

The formal publisher may now be invoked only with:

```text
CONFIG=configs/lb_scgp/lb_scgp_v5.json TASK=audit-publish RUN_ID=LBSCGP-G0-CODE-AUDIT-v5 REVIEW=refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.md REVIEW_RECORD=refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.record.json sbatch scripts/slurm/lb_scgp_g0_audit_publish.sbatch
```

If publisher verification fails, the verdict remains unpublished and no PASS
may be claimed. If publisher verification succeeds, a separate SLURM
post-verifier must call both the producer `_load_freeze_and_audit` path and the
decision-side strict audit consumer against the published v5 bundle. Only if
that post-verifier reports `producer_consumer_ok=true`,
`decision_consumer_ok=true`, `all_ok=true`, and current dirty hash equal to the
frozen dirty hash may synthetic G0 be unlocked.
