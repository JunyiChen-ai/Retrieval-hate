# M0 REALBANK-RESOURCE-v2 Execution Record

Date: 2026-07-13

Executor: **Claude Opus 4.8**, executor role for `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v2`
(machine `runs[3]`), separate from the realbank-prep/freeze, amendment/code-review, full-chain audit,
and execution-authorization roles. Authorized by
`REALBANK_RESOURCE_V2_EXECUTION_AUTHORIZATION.md` (this session) — scope: exactly one CPU-only SLURM
submission.

## Outcome (one line)

**COMPLETED — decision GO (producer) / PASS (independent verifier).** The v1 `$TMPDIR` preflight-death
class is closed; the full real-bank static/resource pipeline ran to completion and published a clean
GO. Peak RSS **≈ 648 MiB**, ~99.4 % under the 96 GiB cap. Single-submit budget spent; **not
resubmitted**. Routes to a fresh independent artifact review (do not self-certify).

## 1. Submission

- **Pre-submit guards (executor, read-only):** (1) `sha256sum -c` of all 8 v2 frozen entities →
  **8/8 MATCH** (config `1d69b961…`, schema `4d95d128…`, common `f90f153f…`, validate `ea703b3e…`,
  producer `e5e9a06a…`, verify `7ffa860b…`, wrapper `348b056b…`, sbatch `d7ab1e75…`); (2)
  **squeue contention re-check** → my running+pending CPUs = 0, so 16 + 0 = 16 ≤ 16 cap, no
  contention (would have waited/polled otherwise); (3) `sacct` prior v2 rows = 0 and artifact dir
  absent. Submit was guarded on all three.
- **Command:** `sbatch scripts/slurm/lb_scgp_global_r2_m0_realbank_resource_v2.sbatch`
- **Job id: `12996`** (job name `lbscgp_global_r2_realbank_resource_v2`).
- No `--time`. Requested 16 CPU / 96 GB / 0 GPU / HateVideo.

## 2. Timeline (sacct)

| event | value |
|---|---|
| State / ExitCode | **COMPLETED / 0:0** |
| Elapsed | 00:00:14 |
| Node | `foscsmlprd01` |
| Auto-release | ran immediately (RUNNING within seconds; no long JobHeldUser hold) |
| sacct `12996.batch` MaxRSS | 3504K — **the batch shell only; NOT the pipeline peak** (the accounting sampler did not capture the short-lived python children; the authoritative peak is the producer's `getrusage`, §3) |

## 3. Terminal-state evidence — decision.json (producer) + semantic_verification.json (verifier)

All four artifacts published under the (deliberately v1-namespace) path
`artifacts/lb_scgp_global/v1/m0/realbank_resource/`: `decision.json`, `source_manifest.json`,
`access_ledger.json`, `semantic_verification.json` (+ their `.publish.lock`s). The wrapper set
`COMPLETE=1` (its final `jq -e '.decision=="PASS"'` gate passed), so the cleanup trap correctly
**kept** them.

**Producer `decision.json`:** `decision = GO`, `terminal_state = PRODUCED_PENDING_INDEPENDENT_VERIFY`,
`producer_status = PASS_CANDIDATE`, `no_success_claim = true`.

**Independent verifier `semantic_verification.json`:** `decision = PASS` (this is what the wrapper's
COMPLETE gate consumed).

### GO criterion — each conjunct satisfied

| conjunct | evidence |
|---|---|
| `job_peak_rss_bytes ≤ 96 GiB` | **679,878,656 B ≈ 648.4 MiB ≈ 0.633 GiB**; `within_cap = true`. Per-dataset: MHC 672,751,616 B (n=549, d=1792, q_rank=8, m=36); MHC_zh 679,878,656 B (n=579, d=1792, q_rank=8, m=36). Measurement = `resource.getrusage(RUSAGE_SELF).ru_maxrss`. **Distance to cap ≈ 95.37 GiB headroom (peak = 0.66 % of cap).** |
| `rank_eps(G0) ≤ d` (all ds) | `all_rank_le_d = true`. MHC `rank_eps=549 ≤ d=1792` PASS (reconstruction_residual 3.68e-15, tail_ratio 0, no negative eigenmass); MHC_zh `579 ≤ 1792` PASS (residual 1.92e-15). Full-rank Gram (rank = N < d), as expected. |
| in-job replay match (all ds) | `all_match = true`. MHC run1==run2 digest `d5d01542…`; MHC_zh `5da06f1c…`. |
| all isolation injections REJECT | producer `isolation_injection_results.all_reject = true` — **11/11 REJECT** (validation label/content, test label/content, held, cache, query_z, query_labels, teacher, non-allowlisted train bank, mutated-hash train bank open). |

### Independent verifier corroboration (R-2 did not materialize)

`semantic_verification.json.metrics` shows the verifier **re-loaded the banks and independently
recomputed**: `job_peak_rss_bytes = 679,878,656` **byte-identical to the producer**;
`rank_eps_by_dataset = {MHC:549, MHC_zh:579}`; `all_rank_le_d = true`; `all_replay_match = true`;
`isolation_all_reject = true`; `verifier_peak_rss_bytes = 648,015,872` (its own process, ≈ 618 MiB).
Because the verifier's independently recomputed replay/rank/peak agreed with the producer's, the
**R-2 highest-residual risk (cross-process LAPACK replay bit-determinism) did not materialize** on real
CLIP-L eigen-tails at fixed 16-thread BLAS.

### Verifier tamper suite — 15/15 REJECT

`semantic_verification.json.injection_results` — all fifteen manifest-tamper mutations REJECTED:
`coverage_safety_enabled, decision_flipped_to_stop, extra_top_level_key, forbidden_source_path,
injection_case_not_reject, nonzero_forbidden_counter, placeholder_claims_science, rank_eps_tampered,
rank_le_d_false, replay_digest_tampered, replay_match_false, resource_over_cap, stale_payload_sha256,
train_bank_hash_tampered, within_cap_flipped_false`. The verifier also pinned the manifest/payload/
source_manifest/access_ledger/relevant-tree sha256s.

### Access discipline / gold isolation

`gold_isolation`: `train_labels_opened = false`; `only_gold_supervision = parent_video_binary_label`;
`segment_gold_used = false`; **all 47 `zero_counters` == 0** (train_label_read, validation/test/held/
cache content & label reads, query_z/query_labels/teacher reads, mllm/ocr/model/network/training/
performance-evaluation calls, gpu_device_count — every one 0). `structural_placeholder.is_science =
false` (r=8, m=36) — NON-SCIENCE placeholder disclosed. `robust_coverage`: `fail_open = true`,
`robust_constraints_enabled = false`, `safety_claim = disabled` (coverage MHC 3.6 % / MHC_zh 7.6 %
below the gate) — reported, **did not fail the run** (as designed; not a GO conjunct).

## 4. Classification

- **Clean GO/PASS** on the frozen single-submit terms. The wrapper fix (a) — in-repo `slurm/tmp/`
  handoff — worked: the producer read the validation JSON without tripping `canonical_root_path`, and
  the pipeline ran the full torch-import → bank-load → PSD-Gram → eigendecomposition → rank-factor →
  structural-moment → replay → isolation path (14 s, ≈ 648 MiB peak).
- **Single-submit budget spent; not resubmitted** (executor instruction 6). This is a GO, so there is
  no failure to adjudicate.
- **Next step:** route to a **fresh independent artifact review** (separate role) before any downstream
  unlock (M1 cache remains locked). Do not self-certify. The `is_science=false` placeholder must be
  overruled or replaced by the science owner before any *scientific* claim rests on the `b_struct`
  measurement.

## Required statements

- **No performance evidence exists and none is claimed.** The run emitted **no accuracy / macro-F1**
  and did **no training or kNN** (0 training/model/mllm/performance-evaluation calls). It is a
  train-bank static/resource microbenchmark; the reported peak-RSS / rank / replay / injection results
  are resource-and-integrity measurements, not a scientific result.
- The only project gold is `parent_video_binary_label`; no segment/frame/timestamp/span/localization/
  stance/target/mechanism/rationale/fragment gold was assumed or introduced. Train **features** were
  opened (allowlisted + hash-checked, `authorized_train_bank_read_count == 2`); train **labels** were
  never read (`train_label_read_count == 0`).
- Run4 (M1 cache), MLLM/cache, validation/test, and training remain **locked**. This GO unlocks nothing
  by itself; it authorizes only the fresh artifact review.
- Executor = Claude Opus 4.8, separate from realbank-prep/freeze, amendment/code-review, full-chain
  audit, and execution-authorization roles. The executor wrote only this record and the authorization
  document; no code/config/schema/plan was edited, and the job was submitted exactly once.
