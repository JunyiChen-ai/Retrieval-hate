# LB-SCGP G0 Formal Code Audit Review V4

Review target: `LBSCGP-G0-FREEZE-v4`.

Reviewer mode: sole fresh independent GPT-5.5 xhigh auditor/publisher. No
subagent, sidecar agent, dynamic workflow, nested Codex process, other model,
implementation repair, config edit, protocol edit, synthetic run, realfold run,
replay run, decision run, G1 run, teacher call, MLLM call, OCR call, held-label
read, held-content read, validation read, test read, or performance task was
used.

## Verdict

**PASS.**

- Critical findings: **0**
- High findings: **0**
- Important findings / authorization notes: **2**
- No-Segment-Gold audit: **PASS**
- Formal PASS authorization: **YES, if and only if the strict sidecar record
  binds this review hash and the frozen publisher succeeds unchanged**
- Authorized next boundary after verified publication: **synthetic only**
- Still locked after publication: **realfold, replay, decision, G1, G2,
  teacher, MLLM, OCR, validation, test, and all performance claims**

The sole gold supervision remains `parent_video_binary_label`. No segment-level
gold exists, and no segment-level gold may be assumed, synthesized, or used.

## Critical Findings

None.

## High Findings

None.

## Important Findings

### I1. v4 repairs only the formal audit-publish tooling gap

The v3 review passed the scientific/code audit but could not create a formal
PASS artifact because no authorized producer existed. v4 adds a narrow
`audit-publish` task, strict review-record schema, atomic publication path, and
downstream consumers for the same formal schema. This is sufficient to remove
the v3 tooling blocker, but it is not a numerical or performance result.

### I2. Publication unlock is deliberately narrow

The v4 dirty policy excludes the formal artifact namespace, so this audit also
requires explicit residue checks around `artifacts/lb_scgp/v4`. Before
publication, that namespace contains only `CONFIG_FREEZE.json` and its lock.
The publisher rejects a preexisting `g0/code_audit` namespace. After a verified
publish, only `g0/code_audit` may exist as the new formal output; this unlocks
synthetic only and does not authorize any downstream real/performance stage.

## Frozen Lineage Evidence

- Config path: `configs/lb_scgp/lb_scgp_v4.json`
- Artifact namespace: `artifacts/lb_scgp/v4`
- Freeze artifact SHA256:
  `dcf65eceba04e7c4f08145b2012653705f7347c6e96ebc8b2b769280dff48fd0`
- Freeze lock SHA256:
  `09003ce9e741d7c0310045f854479deb8fecff74bfddd33f6b9d80dc6df9572a`
- Freeze payload SHA256:
  `92301ad95870b8a1af41e7e69e45054a2e63fe80a021cf4c1c22906aea0872bf`
- Config file SHA256:
  `59804c09f63f923a67eb276325ae6be9ce124fd9a3aceb64c46c3809ffdd85b0`
- Config canonical SHA256:
  `9e99cba37486e2511b0e37fb7d2c3b59053fbac8aca577ba05b36c138aa67c56`
- Implementation SHA256:
  `c7e9371494f991d88a7ab93cc64769fa1e6a92913df3afd2f647201d0eef1bf1`
- Independent verifier SHA256:
  `03a78a89867d3cea468b5319463ccabcefa4b4a589a61863bffd3e14c9df5402`
- Freeze access ledger SHA256:
  `ef67ad3b6521a9b8e9b73dd27260917c531e4ec72a84e04c52afed0c34ba72a7`
- Frozen dirty SHA256:
  `8ca10aec315f800959e7869beb200f4bbc5f5d27841d8c307d896f1644803e7a`

v1-v3 freeze and lock hashes remain unchanged:

- v1 freeze:
  `b6697472b61a61706c694a67b21618d618fcad6e7f59265d8696aee79dc46889`
- v1 lock:
  `34a05bde46775bcb75384c1c853cef7985f5c05efcdf3afec5fad6d85ef57d8d`
- v2 freeze:
  `4c6f0199a429bbf766cb284b9ecaedd9dcaf38733834dd54574245ab86b633ae`
- v2 lock:
  `22426f693ba3f72d52928b0a86d08fe439d7e7fd43f9c74c61aedb710443e211`
- v3 freeze:
  `9fba7f1649dd67d4bb0fcc193e555d8246d7a4966307732e87d5e9fca7346dd9`
- v3 lock:
  `9c32d07c524e466ad06c06fc2a472829764cd1facff22390eac8b2879d329b8f`

## Audit-Publish Toolchain

The v4 publisher is fail-closed in the reviewed code:

- The free-form review is not parsed as authority.
- The strict review record rejects missing and additional fields.
- The strict record binds reviewer/process identity, report path and file hash,
  run ID, freeze identity, freeze payload/file hash, config canonical hash,
  implementation hash, verifier hash, 0 Critical, 0 High, no-segment-gold PASS,
  and formal authorization.
- The publisher independently recomputes freeze payload/file/lock hashes,
  config file and canonical hashes, implementation hash, verifier hash, current
  git head and dirty hash, frozen file hashes, allowed NPZ member hashes,
  prior v1-v3 freeze and lock hashes, dirty policy, wrapper contract, and
  no-segment/zero-counter fields.
- Only allowed NPZ members are hashed: `memory_ids`, `memory_labels`,
  `memory_z`, and `query_ids`. `query_z` and `query_labels` remain forbidden.
- Publication is an atomic directory transaction to
  `artifacts/lb_scgp/v4/g0/code_audit/`, with byte-identical `review.md` and
  `review_record.json`, canonical `audit.json` and `publication_index.json`,
  and one `.publish.lock` per file.
- `_load_freeze_and_audit` and the final decision verifier consume the same v4
  audit, review-record, and publication-index schemas and reject drift.

## Scientific Binding Audit

Static review found the G0 scientific contracts fail-closed:

- Dykstra uses persistent corrections, fixed set order, replayable transition
  hashes, `LOCAL_STATIONARY_CERTIFIED` thresholds, and bounded fallback for
  infeasible/capped cases.
- PSD projection symmetrizes input, clips only numerical negatives, enforces
  unit diagonal separately, and preserves the off-diagonal box.
- Rank cells use exact top-20 vote semantics, self exclusion, canonical-ID
  tie handling, 19 internal plus twentieth-vs-outsider inequalities, boundary
  orientation limits, pivot limits, unresolved tie map fail-closed behavior,
  and direct REMOVE replay hashes.
- Factorization rejects eigenvalues below `-1e-7`, handles repeated and null
  eigenspaces deterministically, and verifies Procrustes alignment with
  `<=1e-6` reconstruction tolerances.
- Registered-cone/Farkas checks bind singleton, pair, triplet, and SupCon
  families using parent-video binary labels only, with universe hashes,
  separation, witness-family checks, and duality-gap gates.
- REMOVE rollback restores model, optimizer, scheduler, scaler, RNGs, sampler
  cursor, and epoch cursor before direct replay comparison.
- SHUFFLE, NOISE, and DIRECT controls are specified for later locked stages and
  are not executed or claimed by this G0 v4 audit.

## No-Segment-Gold Audit

**PASS.**

Evidence:

- Config and freeze bind `only_gold_supervision=parent_video_binary_label`,
  `segment_gold_exists=false`, and `segment_gold_used=false`.
- Sanitizer provenance and decision bind `segment_cache_path=null`,
  `segment_cache_sha256=null`, `segment_artifact_created=false`,
  `segment_objective_allowed=false`, `formal_query_z_read_count=0`,
  `formal_query_labels_read_count=0`, and
  `formal_model_optimizer_evaluator_outer_held_read_count=0`.
- The frozen bank contract permits `query_ids` only as held-ID sentinels and
  forbids `query_z` and `query_labels`.
- Realfold and replay code set `segment=None`, require `lambda_seg=0.0`, pass
  `segment_cache=None`, and reject segment objective/cache metadata.
- The reviewed surfaces contain no segment annotation, subclip objective,
  timestamp/span/localization/stance/target/mechanism/rationale gold, teacher
  call, MLLM call, OCR call, validation/test content read, or formal
  held-label/content read.

## Audit-Only SLURM Checks

No audit-only job ran synthetic, realfold, replay, decision, teacher, MLLM,
OCR, validation, test, held-label, held-content, or performance stages.

- Job `12779`, `lbscgp_v4_audit_checks`: FAILED `2:0`, elapsed `00:00:19`.
  This was a harness expectation bug in the preexisting-directory negative
  test; the publisher rejected the preexisting `code_audit` namespace and the
  cleanup proof showed no formal residue. Log SHA256:
  `d850770e1050026198ba9e4abb21666ab0f2eca913497d78aabd5933e2f83d55`.
- Job `12780`, `lbscgp_v4_audit_checks`: COMPLETED `0:0`, elapsed
  `00:00:18`, 2 CPU / 4G. Result JSON SHA256:
  `3984d970c8b2ef5bb59511ee7770356285d28b79701deaada982ff63f1ce2039`.
  Log SHA256:
  `ae7a4c2e921ff0913e4bb2ff3935875bfc8088d2d6e8c44b520a9f8c878a4f5f`.

Job `12780` PASS matrix:

- missing review report: publisher failed, no formal residue
- missing review record: publisher failed, no formal residue
- malformed review-record JSON: publisher failed, no formal residue
- additional review-record field: publisher failed, no formal residue
- wrong review report hash: publisher failed, no formal residue
- wrong record run ID: publisher failed, no formal residue
- wrong config path in record: publisher failed, no formal residue
- wrong freeze file hash in record: publisher failed, no formal residue
- wrong implementation hash in record: publisher failed, no formal residue
- nonzero Critical: publisher failed, no formal residue
- nonzero High: publisher failed, no formal residue
- `no_segment_gold_pass=false`: publisher failed, no formal residue
- segment-gold fields true: publisher failed, no formal residue
- formal authorization false: publisher failed, no formal residue
- repair executor claimed: publisher failed, no formal residue
- wrong review process identity: publisher failed, no formal residue
- dirty-state drift: publisher failed before publication, no formal residue
- preexisting `g0/code_audit` directory/lock: publisher failed and cleanup
  removed the deliberate fixture
- wrong wrapper `TASK`: wrapper failed before Python with exit 2
- wrong wrapper `RUN_ID`: wrapper failed before Python with exit 2
- wrong wrapper review path: wrapper failed before Python with exit 2
- wrong wrapper config path: wrapper failed before Python with exit 2
- direct wrong review path: publisher failed
- direct wrong config path: publisher failed
- positive pre-publication validation of frozen recomputation plus strict
  record validation passed without invoking the transaction

Final pre-publication residue proof after job `12780`:

- `artifacts/lb_scgp/v4/g0/code_audit`: absent
- `.code_audit.publish.tmp.*`: absent
- temporary source review path: absent
- temporary source record path: absent
- `artifacts/lb_scgp/v4` contains only `CONFIG_FREEZE.json` and
  `CONFIG_FREEZE.json.publish.lock`

## Publication Authorization

The formal publisher may now be invoked only with:

```text
CONFIG=configs/lb_scgp/lb_scgp_v4.json TASK=audit-publish RUN_ID=LBSCGP-G0-CODE-AUDIT-v4 REVIEW=refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.md REVIEW_RECORD=refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.record.json sbatch scripts/slurm/lb_scgp_g0_audit_publish.sbatch
```

If publisher verification fails, the verdict remains unpublished and no PASS may
be claimed. If publisher verification succeeds, the only newly authorized next
stage is synthetic G0 under the frozen v4 run ID; no downstream realfold,
replay, decision, G1, teacher, MLLM, OCR, validation, or test stage is
authorized by this review.
