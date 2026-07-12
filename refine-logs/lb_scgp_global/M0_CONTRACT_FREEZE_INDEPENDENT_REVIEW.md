# M0 Contract Freeze Independent Review

Date: 2026-07-12

Reviewer constraints: fresh independent Run1 code+freeze audit; no subagents, no dynamic workflow, no external model, no SLURM submission, no GPU/training/evaluation/cache/MLLM/OCR work. This report is the only file written.

## Formal Verdict

Run1 `LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1` is accepted as a frozen contract artifact for the next boundary. This is not M0 overall PASS and not experimental or performance success.

Severity counts:

| Severity | Count |
|---|---:|
| Critical | 0 |
| High | 0 |
| Medium | 3 |
| Low | 0 |

Decision rule applied: because Critical=0 and High=0, Run2 only (`LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1`) is authorized. Run3 and all later runs remain locked. MLLM, OCR, GPU, training, validation/test evaluation, performance work, cache construction, and final-test work remain locked.

## Findings

### MCF-M1: Contract artifact schema is shallow; semantic acceptance depends on producer and audit code

Severity: Medium.

Evidence:

- `schemas/lb_scgp_global_r2/scgp_global_contract_freeze_v1.schema.json:3` has `additionalProperties: true`.
- The schema only types major nested payloads as generic objects at `schemas/lb_scgp_global_r2/scgp_global_contract_freeze_v1.schema.json:9-21`; it does not constrain nested zero-counter values, source-hash members, lock/hash fields, no-success flag, or interface payloads.
- The producer does perform semantic checks: run ID, schema ID, artifact path, authorization flags, and gold/segment flags at `scripts/analysis/lb_scgp_global_r2_contract_freeze.py:165-187`, and artifact existence/lock refusal at `scripts/analysis/lb_scgp_global_r2_contract_freeze.py:189-191`.

Consequence:

The emitted artifact is semantically consistent, but a future consumer must not treat JSON-schema validation alone as sufficient. Run2 must use semantic verifier checks for the KKT payload, counters, hashes, and GO/STOP decision, not parse-only or schema-only acceptance.

### MCF-M2: Dirty policy is auditable but not fully fail-closed in the validator

Severity: Medium.

Evidence:

- The declared dirty policy is narrow in config at `configs/lb_scgp_global_r2/m0_contract_freeze_v1.json:21-38`.
- The validator checks JSON syntax, shell syntax, Python compilation, one tracker diff check, and whitespace at `scripts/analysis/lb_scgp_global_r2_validate.py:83-104`; it does not classify the whole dirty tree against the configured allowlist.
- The artifact records `dirty_tree_sha256=e4087426cd2d1d43ab4d7c950d7ef9f4077580766bfae3d03e0b2c17b21524d9`, and current post-run tracker/docs drift is distinguishable: frozen tracker hash in the artifact is `328dc0f8af51c4fd6dc03807c8fd935993a4686c1290f7474ca22dcee7a42dae`, while current `EXPERIMENT_TRACKER.md` is `e2302205b8a9296b68f5642bd26741b18ad04404948768045b7626ec81e7102e`.
- `sha256sum -c refine-logs/lb_scgp_global/EXPERIMENT_PLAN_HASHES.sha256` now fails only on `EXPERIMENT_TRACKER.md`, consistent with the allowed post-run tracker update.

Consequence:

The drift is auditable and not a Run2 lock by itself, but Run2 must explicitly bind the frozen Run1 artifact hashes and the current tracker status. It must not silently rely on the stale plan hash manifest as if it represented current tracker contents.

### MCF-M3: Common-basis Q interface exists, but Run1 fixture does not exercise `orth_cap`

Severity: Medium.

Evidence:

- `orth_cap` is implemented with centering, float64 SVD, thresholding, rank cap, and canonical sign orientation at `scripts/analysis/lb_scgp_global_r2_common.py:374-389`.
- Structural `M_Q(G)=Q^T(G-I)Q/N`, `vech`, and operator summary are executable at `scripts/analysis/lb_scgp_global_r2_common.py:398-419`.
- The Run1 interface fixture uses `q = row_normalize(np.eye(...))` and truncates columns at `scripts/analysis/lb_scgp_global_r2_contract_freeze.py:73-74`, rather than calling `orth_cap` on `Phi`.

Consequence:

Run1 establishes executable scaffolding, not a full common-basis proof. Run2 must exercise `orth_cap(Phi, ids, rank_cap=8)` from certificate encodings before any synthetic KKT GO. This is nonblocking for Run2 because Run2 is the first run expected to prove the synthetic KKT/operator path.

## No-Segment Audit

Pass for Run1.

- Only gold supervision is `parent_video_binary_label`.
- `segment_gold_exists=false`, `segment_gold_used=false`, and `segment_gold_used_for_selection=false` are serialized in the artifact contract.
- `segment_gold_read_count=0`; all validation/test/held/cache/certificate/compiler-target/teacher/head/reranker/key-selector counters are zero.
- Restricted certificate schema `scgp_global_cert_v2` rejects extra keys through `additionalProperties=false` and contains no timestamp, span, localization, target, mechanism, stance, verdict, rationale, segment, or frame-gold field.
- The access ledger opened only authoritative config/schema/plan/review files and train provenance members. It did not open validation/test/held content, `query_z`, or `query_labels`.

Gold boundary remains: the only gold annotation is `parent_video_binary_label`; no segment/frame/timestamp/span/localization/stance/target/mechanism/rationale gold exists or is used.

## Hash Verification

Artifact and payload:

| Item | SHA256 |
|---|---|
| `artifacts/lb_scgp_global/v1/m0/contract_freeze.json` | `09b78682389f1c9774c9dffc43c759bceeec9d7f44eca1ce4cd626d0cd6d12da` |
| Artifact payload excluding `payload_sha256` | `57f935cfa6ff22f81ec726eba9e0000d76f95bf93575b7539b78ba4d7c5bde53` |
| `contract_freeze.json.publish.lock` | `c6fbb49c3c8aba942da3f254c3c5f65d037002548fafb961154e6fa79a3e4ea7` |
| Access ledger canonical hash | `f2437ea61b95782ef9fd79c4ddd856ebf869d6ed7bda8a23d6c93c104b8bc823` |
| Dirty-tree hash recorded by artifact | `e4087426cd2d1d43ab4d7c950d7ef9f4077580766bfae3d03e0b2c17b21524d9` |

Config, schemas, implementation, verifier, wrapper:

| Item | SHA256 |
|---|---|
| `configs/lb_scgp_global_r2/m0_contract_freeze_v1.json` | `5111c2d6d74c745afe35a7067566b531714ce6482df3f5b9469a442486146868` |
| `schemas/lb_scgp_global_r2/scgp_global_cert_v2.schema.json` | `4d3f1663e633c30ae58e35c0feddaa2fa9bbedba279cdbe6f38ecc35d761f22f` |
| `schemas/lb_scgp_global_r2/scgp_global_contract_freeze_v1.schema.json` | `d6a22233ec2ad028f1fdf8a0315641a30a76e1b3a96249615c48161b3d890105` |
| Implementation manifest | `ac73a23cc3d7cae72b17e13831a487491fbe6e4867c9f992e6d789b77eb3e072` |
| `scripts/analysis/lb_scgp_global_r2_common.py` | `b0461460a71f72c81b611bb060950a459e84d7f5cfe46f62da19625e624c59db` |
| `scripts/analysis/lb_scgp_global_r2_contract_freeze.py` | `1c11544e5305c4350b3d985ccf81e88de8f1f31c58e662df819570aaa92bccbc` |
| `scripts/analysis/lb_scgp_global_r2_validate.py` | `061cf80ae532e268b623050db5b39c47b9f0858f0a9591d94f8404e11e778fd2` |
| `scripts/wrappers/lb_scgp_global_r2_run1.sh` | `1eb342475c63df5d16bf570c82de465803652e4a6157444ea02294b58fa9596d` |
| `scripts/slurm/lb_scgp_global_r2_m0_contract_freeze.sbatch` | `ccbade355239d3c313b70ed55a2907f7a2d51716a62b809e835ec9ad1441d882` |

Plan/review bindings:

| Item | SHA256 |
|---|---|
| `AGENTS.md` | `e6aaf5d66399cdbbe7fcc2c811931277b0ed4a24b592ffa5cbb60315b29ea23c` |
| `FINAL_PROPOSAL.md` | `b5ab9409b86407da952d3492bc28b9944b80cd653e032dcf627cfca75cd1a9ff` |
| `EXPERIMENT_PLAN.md` | `a0528dd4242becff1db85f797cca3363e8884bc661d8e992a3c90d87cee9e054` |
| `EXPERIMENT_PLAN.machine.json` | `f0339096270d84f5f14b4a19468bb3168357c4fd4f488c799d2f357d8c619a9e` |
| `EXPERIMENT_PLAN_REVIEW_R2.md` | `d952989742c9402cfd38de935deece73b2933e7ab8f70883ca8e0bea40a4bd46` |
| `EXPERIMENT_PLAN_REVIEW_R2.machine.json` | `8c222ff6db3a38752851fbc113c9e1988845ca424ce8fc51d48cfdea0c51f472` |
| frozen pre-run `EXPERIMENT_TRACKER.md` in artifact | `328dc0f8af51c4fd6dc03807c8fd935993a4686c1290f7474ca22dcee7a42dae` |
| current post-run `EXPERIMENT_TRACKER.md` | `e2302205b8a9296b68f5642bd26741b18ad04404948768045b7626ec81e7102e` |
| current `M0_IMPLEMENTATION_HANDOFF.md` | `f5d28177cf4d1e40c9fbe45c52b08f5dadde01db1a5ba0d3e03d46842c3f6746` |
| current `M0_CONTRACT_FREEZE_EXECUTION.md` | `71a502375ec4f99224e8a8cd55aa61ea9f96cff6bf81743646c59e10d1efeebb` |

Independent recomputation confirmed:

- Payload hash matches the artifact's `payload_sha256`.
- Access ledger hash matches the artifact's `access_ledger_sha256`.
- Implementation manifest hash matches the artifact's `implementation_sha256`.
- Artifact is newline-terminated canonical JSON.

## Protected Snapshot Verification

Pass.

Protected old scope: `configs/lb_scgp`, `artifacts/lb_scgp`, `refine-logs/lb_scgp`, old `scripts/analysis/lb_scgp_*.py`, and old `scripts/slurm/lb_scgp_*.sbatch`, excluding new `lb_scgp_global_r2_*`.

Evidence:

- `/tmp/lb_scgp_global_r2_old_protected_pre.sha256`: 278 rows, SHA256 `243e89b69b169b222dd97f9df092d511f823fb26201e91fd89cb581710940462`.
- `/tmp/lb_scgp_global_r2_old_protected_post.sha256`: 278 rows, SHA256 `243e89b69b169b222dd97f9df092d511f823fb26201e91fd89cb581710940462`.
- `/tmp/lb_scgp_global_r2_old_protected_final.sha256`: 278 rows, SHA256 `243e89b69b169b222dd97f9df092d511f823fb26201e91fd89cb581710940462`.
- `cmp` pre/post and pre/final returned `0`.
- Independent current rehash of the protected scope returned count `278` and manifest SHA256 `243e89b69b169b222dd97f9df092d511f823fb26201e91fd89cb581710940462`.

Conclusion: the old 278 protected files are byte-identical before/after/final and match the current protected scope.

## SLURM And Wrapper Audit

Pass.

- Wrapper fixes and checks `RUN_ID=LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1` and refuses any other run ID at `scripts/wrappers/lb_scgp_global_r2_run1.sh:6-13`.
- Wrapper checks config run ID and exact artifact path at `scripts/wrappers/lb_scgp_global_r2_run1.sh:15-24`.
- SLURM script requests 4 CPUs, 16 GB, and no GPU directive at `scripts/slurm/lb_scgp_global_r2_m0_contract_freeze.sbatch:2-8`.
- SLURM script activates `HateVideo` at `scripts/slurm/lb_scgp_global_r2_m0_contract_freeze.sbatch:11-13`.
- `sacct -j 12901` reports `COMPLETED`, exit `0:0`, `AllocCPUS=4`, `ReqMem=16G`, elapsed `00:00:03`, batch `MaxRSS=3268K`.
- stdout/stderr logs for job `12901` are both zero bytes.

No Run2, Run3, MLLM, OCR, GPU, training, performance, validation/test evaluation, or extra SLURM job was submitted by this audit.

## Access And Provenance Audit

Pass.

Access ledger opened:

- authoritative inputs: `AGENTS.md`, final proposal, approved plan, machine plan, R2 review, R2 machine review, hash manifests, and frozen pre-run tracker;
- schemas/config;
- train provenance files only: `data/gt/MHC/train.jsonl` and `data/gt/MHC_zh/train.jsonl`.

Access ledger did not open:

- `data/gt/MHC/val.jsonl`, `data/gt/MHC/test.jsonl`, `data/gt/MHC_zh/val.jsonl`, or `data/gt/MHC_zh/test.jsonl`;
- held content;
- `query_z` or `query_labels`;
- MLLM/OCR/cache/certificate/compiler-target/teacher/head/reranker/key-selector artifacts.

Validation/test hashes appear only as declared provenance strings in config/artifact and were not content-opened by Run1.

## Interface Audit

Run1 provides substantive executable scaffolding for the next boundary:

- restricted certificate validation and encoding: `scripts/analysis/lb_scgp_global_r2_common.py:293-339`;
- replica consensus: `scripts/analysis/lb_scgp_global_r2_common.py:342-363`;
- common-basis helper `orth_cap`: `scripts/analysis/lb_scgp_global_r2_common.py:374-389`;
- `M_Q` structural moment and `vech` interface: `scripts/analysis/lb_scgp_global_r2_common.py:392-419`;
- global projection contract with strong-convexity guard and trust constraints: `scripts/analysis/lb_scgp_global_r2_common.py:431-466`;
- rank-tail/null/no-rescue audit and PSD factor path: `scripts/analysis/lb_scgp_global_r2_common.py:469-523`;
- Procrustes interface: `scripts/analysis/lb_scgp_global_r2_common.py:526-533`.

Limit: Run1 is a contract freeze, not a KKT solver run. It records the serialized H-metric normal-cone/KKT-only GO rule and finite-VI diagnostic-only rule, but it does not prove KKT payload correctness. That proof belongs to Run2.

## Executable Decision

Decision: AUTHORIZE RUN2 ONLY.

Allowed next boundary:

- `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1` only, through the approved SLURM path and resource policy.

Still locked:

- Run3 and all later runs;
- M1 cache and all MLLM/OCR work;
- GPU/training/performance runs;
- validation/test/held evaluation or content access;
- any sample weighting, key selection, pair/triplet/SupCon, segment route, teacher/head/rerank, local-v7 reuse, local-v8, or direct attribution shortcut outside the approved future controls.

Run2 must close the Medium findings by semantic verifier behavior: strict payload validation beyond shallow schema, explicit dirty-state binding, and actual `orth_cap`/`M_Q` execution before any synthetic KKT GO.
