# M0 REALBANK-RESOURCE-v2 Independent Artifact (Receiving) Review

Date: 2026-07-13

Reviewer: fresh, zero-context, independent **artifact receiving reviewer** for the
`lb_scgp_global_r2` M0 **REALBANK-RESOURCE-v2** run (run_id `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v2`,
machine `runs[3]`, job `12996`; artifacts published in the deliberately-reused v1 namespace
`artifacts/lb_scgp_global/v1/m0/realbank_resource/`). This role is deliberately **separate** from the
realbank-prep/freeze author, the merged amendment/code-review reviewer, the full-chain static auditor,
the execution authorizer, and the executor. I re-derived every check below from files, artifacts, and
frozen hashes read directly in this session; I did not rely on the producer's or executor's
self-report for any load-bearing claim.

## Verdict

**ARTIFACT_ACCEPTED.** The v2 realbank static/resource microbenchmark artifact set is complete,
schema-conformant, internally and externally hash-consistent, and satisfies every conjunct of the frozen
GO decision rule under independent recomputation. The M0/B0 realbank-resource block is **complete**; the
**M1 gate is unlocked** (this acceptance authorizes routing to the M1 planning/freeze chain — it does
**not** itself authorize any M1 execution). Acceptance is conditioned on the obligations in §5, chiefly
the `is_science=false` NON-SCIENCE placeholder science-owner override.

---

## 1. Completeness & binding (PASS)

**Artifact set present (4 payloads + 4 publish locks).** `decision.json`, `source_manifest.json`,
`access_ledger.json`, `semantic_verification.json`, each with its `.publish.lock`. The lock files carry
the writer PIDs: `391836` (producer — decision/source_manifest/access_ledger) and `393975` (verifier —
semantic_verification), consistent with the two-process producer↔independent-verifier architecture.

**Artifact file hashes recomputed on disk and cross-checked (all MATCH):**

| file | recomputed sha256 | bound where | verdict |
|---|---|---|---|
| `decision.json` | `0ef617fd…674dd8` | `semantic_verification.manifest_file_sha256` | ✓ |
| `source_manifest.json` | `3b103a06…b5a1f3d` | `decision.hashes` & `verif.metrics.source_manifest_sha256` | ✓ |
| `access_ledger.json` | `8c380d7c…a504d5` | `decision.hashes` & `verif.metrics.access_ledger_sha256` | ✓ |
| `semantic_verification.json` | `925397c8…7ec689` | terminal (nothing binds it downstream) | ✓ n/a |

(The `access_ledger.json` internal field `access_ledger_sha256=1e755515…` is a payload-only hash of the
ledger array and is distinct **by construction** from the whole-file hash `8c380d7c…`; not a defect.)

**Payload / tree cross-process agreement.** `decision.payload_sha256 = 6ddb5dc7…` byte-equals
`semantic_verification.manifest_payload_sha256` — the independent verifier recomputed the canonical
payload digest and it matched. `relevant_tree_sha256 = c123a6d9…` is identical across `decision`,
`source_manifest`, and `semantic_verification`.

**Frozen-entity no-drift.** I re-`sha256sum`'d all **8** v2 entities; all match
`REALBANK_RESOURCE_V2_CLONE_FREEZE.md §1` byte-for-byte: config `1d69b961…`, schema `4d95d128…`,
common `f90f153f…`, validate `ea703b3e…`, producer `e5e9a06a…`, verify `7ffa860b…`, wrapper `348b056b…`,
sbatch `d7ab1e75…`.

**Bank binding (549/579).** Both allowlisted train **feature** banks match their declared sha256:
MHC `deea74ff…` and MHC_zh `929571f8…`. The real-bank row counts propagate into `rank_tail`
(`positive_eigenmass = n = 549 / 579`) and `resource_peak.per_dataset.n = 549 / 579`.

**Run1 dependency binding.** `decision.hashes.run1_artifact_sha256 = 09b78682…` and
`run1_lock_sha256 = c6fbb49c…` match `artifacts/lb_scgp_global/v1/m0/contract_freeze.json{,.publish.lock}`
on disk exactly (SYNTH-KKT-v4 lineage dependency intact).

**Schema conformance.** `schema.required` (23 keys) equals `decision.json` top-level keys (23) as a set
(permutation only). `artifact_schema_id = scgp_global_realbank_resource_v2`; all `schema_version` fields
carry the `…_v2` suffix.

**Lineage / science flags vantage-invariant.** `run_id = LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v2` in
`decision`, `semantic_verification`, and `source_manifest`; `structural_placeholder.is_science = false`;
`authorized_boundary = {m1_cache_and_later_locked: true, train_bank_static_replay_only: true}`;
`terminal_state = PRODUCED_PENDING_INDEPENDENT_VERIFY` (this review **is** that pending independent
verify). `runs[3]` in the machine plan pins run_id/schema-id v2, slurm `{16,96,0,HateVideo,no_time}`,
deps `[…SYNTH-KKT-v4]`, artifact path in the intentionally-preserved v1 namespace.

## 2. GO criterion — each conjunct independently re-checked (PASS)

Frozen rule: `GO iff job_peak_rss_bytes ≤ 96 GiB AND rank_eps ≤ d (all ds) AND in-job replay match
(all ds) AND all isolation injections REJECT`. `decision = GO`, `producer_status = PASS_CANDIDATE`,
verifier `decision = PASS`.

1. **Peak RSS ≤ 96 GiB — PASS (arithmetic self-checked).** `cap_bytes = 103,079,215,104` = `96·1024³`
   (verified equal). `job_peak_rss_bytes = 679,878,656` = **648.38 MiB = 0.6332 GiB**; `within_cap = true`.
   Headroom **95.37 GiB**; peak = **0.66 %** of cap. Per-dataset peaks: MHC `672,751,616` (641.6 MiB),
   MHC_zh `679,878,656` (648.4 MiB); the job peak is the max of the two. Independent verifier's own
   process peaked at `648,015,872` (618.0 MiB), a separate figure and also far under cap.

2. **rank_eps ≤ d (both ds) — PASS.** MHC `rank_eps=549 ≤ d=1792`; MHC_zh `579 ≤ 1792`. Residuals at
   machine epsilon (`3.68e-15`, `1.92e-15`), `negative_eigenmass = 0`, `tail_ratio = 0`,
   `lambda_dplus1 = 0` → full-rank Gram (rank = N < d), per-dataset `status = PASS`, `all_rank_le_d = true`.

3. **In-job replay bit-identical (both ds) — PASS.** `run1 == run2` digests byte-equal: MHC
   `d5d01542…`, MHC_zh `5da06f1c…`; `all_match = true`. The independent verifier additionally recomputed
   `rank_eps_by_dataset = {MHC:549, MHC_zh:579}` and `job_peak_rss_bytes = 679,878,656` byte-identically
   → the R-2 cross-process LAPACK-determinism residual did **not** materialize.

4. **Isolation injections all REJECT — PASS (11/11 counted).** `isolation_injection_results.cases` has
   exactly **11** keys, **0** non-REJECT: validation label/content, test label/content, held content,
   cache artifact, query_z, query_labels, teacher artifact, non-allowlisted train bank, mutated-hash
   train-bank open. `all_reject = true`.

**Verifier tamper suite — 15/15 REJECT (counted).** `semantic_verification.injection_results` has
exactly **15** keys, **0** non-REJECT (coverage_safety_enabled, decision_flipped_to_stop,
extra_top_level_key, forbidden_source_path, injection_case_not_reject, nonzero_forbidden_counter,
placeholder_claims_science, rank_eps_tampered, rank_le_d_false, replay_digest_tampered, replay_match_false,
resource_over_cap, stale_payload_sha256, train_bank_hash_tampered, within_cap_flipped_false). Notably
`placeholder_claims_science → REJECT` enforces `is_science` cannot flip to `true`.

**Zero-counters — 47/47 == 0 (counted, set-equal across two artifacts).** Both
`decision.gold_isolation.zero_counters` and `access_ledger.zero_counters` hold exactly **47** counters,
all `0`, and the two key-sets are identical. This includes `train_labels_opened=false`,
`train_label_read_count=0`, `gpu_device_count=0`, `mllm_call_count=0`, `model_call_count=0`,
`training_call_count=0`, `performance_evaluation_count=0`, and every validation/test/held/cache
content-and-label read counter.

**robust_coverage fail-open — handled correctly.** `robust_constraints_enabled = false`,
`fail_open = true`; both datasets `coverage_gate_pass = false` (MHC 3.6 %, MHC_zh 7.6 % below gate) with
`safety_claim = disabled` and `class_stratification = deferred_train_labels_not_opened`. Because
robustness is **not** a GO conjunct and the run deliberately declines to open train labels for class
stratification, sub-gate coverage is **reported, not fatal** — the run makes no robustness/safety claim
rather than opening labels to chase coverage. This is the correct fail-open disposition.

## 3. Access ledger / gold isolation (PASS)

Actual reads (from `access_ledger.access_ledger`): exactly **2** `authorized_train_bank`
`train_bank_feature_read` entries (MHC, MHC_zh; `authorized_train_bank_read_count = 2` in both
`decision.allowed_reads` and the ledger). Other ledger rows are hash-only provenance:
`authoritative_input` file-hashes (plan machine-json + tracker), `schema_or_source` file-hashes
(config + schema), and four `declared_provenance_not_opened` rows for `data/gt/{MHC,MHC_zh}/{val,test}.jsonl`
— declared identity carried, **content never opened** (corroborated by the 47 zero-counters). **Zero**
validation/test/held/cache content or label access; **no** accuracy / macro-F1 / margin / prediction /
performance figure appears anywhere in the artifact set. `only_gold_supervision =
parent_video_binary_label`; `segment_gold_exists = false`; `segment_gold_used = false`.

I mirrored this discipline as reviewer: I hashed only the two allowlisted banks, the artifacts, the 8
frozen entities, and the Run1 dependency artifact. I did **not** open or hash any val/test/held/cache
content.

## 4. Runtime / log cross-check

`slurm/logs/lbscgp_global_r2_realbank_resource_v2_12996.{out,err}` are both **0 bytes** — consistent
with a clean run whose payloads go to the artifact JSONs and whose wrapper `jq -e '.decision=="PASS"'`
gate passed with no stderr spew (execution record: sacct `COMPLETED / 0:0`, elapsed 14 s). The sacct
`12996.batch MaxRSS 3504K` is the batch shell only; the authoritative pipeline peak is the producer's
`getrusage` figure independently corroborated by the verifier (§2.1). `source_manifest.authoritative_inputs`
binds **11** documents (incl. `REALBANK_FULLCHAIN_STATIC_AUDIT.md`); the code-review doc is intentionally
**not** bound — matching the execution authorization's no-post-binding finding.

## 5. Obligations attached to acceptance

1. **`is_science=false` NON-SCIENCE placeholder — science-owner override REQUIRED before any scientific
   claim.** `b_struct = vech(M_Q(G0))` from a deterministic label-blind Φ seed (per-dataset L2 =
   `[0.00195, 0.00315]`, `rank_cap_r = 8`, `m_scale = 36`) exists **only** to open the
   orth_cap/structural-moment/adjoint code path at real N. It certifies nothing. The real cache `b_struct`
   arrives at M1. No downstream scientific claim may rest on this placeholder until the science owner
   overrules or replaces it.
2. **M1 and all downstream remain LOCKED.** `m1_cache_and_later_locked = true`. This acceptance completes
   the M0/B0 block and **unlocks the M1 gate** (i.e., permits the M1 planning/freeze/review/authorization
   chain to begin) but authorizes **no** M1 execution. M1 requires its own clone-freeze → code-review →
   authorization → execution → artifact-review chain.
3. **No performance/robustness claim.** The run emits no accuracy / macro-F1 and does no training or kNN;
   `safety_claim = disabled`. Downstream must not cite this as a performance or robustness result — it is
   a train-bank static/resource-and-integrity microbenchmark only.
4. **Bookkeeping (non-blocking).** Machine `runs[3].status` still reads
   `V2_CLONE_FROZEN_AFTER_V1_TMPDIR_BURN_PENDING_INDEPENDENT_REVIEW`; advancing it to a
   completed/GO-accepted state is a separate plan-bookkeeping step. This does not affect artifact
   acceptance (the artifact is self-describing and hash-bound).

## 6. Reviewer boundary & required statements

- **Read-only.** My only write is this document. I ran no project Python, imports, `py_compile`,
  `conda`, `sbatch`/`squeue`/`sacct`-mutating, experiment, MLLM/OCR/API/model/network/GPU/training/
  evaluation. Tools used were read-only: `sha256sum`, `jq`, `diff`, `ls`, and `python3` for arithmetic.
  I opened no validation/test/held/cache content; I hashed only the two allowlisted train **feature**
  banks, the four artifacts + locks, the eight frozen v2 entities, and the Run1 dependency artifact.
- **No performance evidence exists and none is claimed.** The only project gold is
  `parent_video_binary_label`; no segment/frame/timestamp/span/localization/stance/target/mechanism/
  rationale/fragment gold was assumed or introduced. Train **features** were opened (allowlisted +
  hash-checked, count == 2); train **labels** were never read (count == 0).
- **Model-binding declaration (precedent: v1/v2 authorizations & code review).** Project discipline
  (`CLAUDE.md`) binds subagents to **Opus 4.8**; the harness environment reports this session's model as
  Opus 4.8 (1M-context variant, `claude-opus-4-8[1m]`), which satisfies that binding. The `AGENTS.md`
  "GPT-5.5 xhigh" cross-model reviewer backend is unavailable this session, so this receiving review is
  **same-family (Opus 4.8)**, not cross-model; independence is instead enforced by fresh, zero-context
  re-derivation of every load-bearing check from on-disk artifacts and frozen hashes. Documented process
  fact, not a defect.

---

**Final: ARTIFACT_ACCEPTED — M0/B0 realbank-resource block complete; M1 gate unlocked; obligations §5
(esp. `is_science=false` science-owner override, M1-remains-locked) attached.**
