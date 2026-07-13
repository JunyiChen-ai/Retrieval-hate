# M1 CACHE v2 Execution Authorization (post v1 double-burn; symlink-tolerant guard)

Date: 2026-07-13

Authorizer: **Claude Opus 4.8**, **independent M1 v2 execution-authorization + executor role** —
separate from m1-prep (v2 author/implementer/freezer), the v2 amendment reviewer, and the fresh
0C/0H v2 code reviewer. Derived only from the frozen v2 evidence + read-only re-verification this
session; execution proceeds strictly within §7.

Discipline (authorization phase): read-only (`sha256sum`, `ls`, `grep`, `jq`, `sacct`, `squeue`).
No SLURM submission during authorization, no Python compute, no GPU/MLLM/OCR/network/model run, no
validation/test or train-label read. Only this document is written in this phase.

Model declaration: ran as Claude Opus 4.8 (`claude-opus-4-8`, 1M context), per CLAUDE.md. No
deviation.

Context: the v1 cache runs **double-burned** (jobs 13003/13004, `M1_CACHE_EXECUTION_RECORD.md`) at
Stage-1 — train mp4s are in-repo **symlinks whose targets escape the repo**, and v1's
`canonical_root_path.resolve()` fired at the video site before any model load or write (fail-closed,
zero artifacts). v2 replaces run_ids v1→v2 and swaps the two video sites to a symlink-tolerant
`canonical_video_path`, with zero other isolation weakening.

---

## 0. Verdict

**AUTHORIZED.** The M1 v2 cache block (`runs[4]` CACHE-MHC-v2 / `runs[5]` CACHE-MHC_zh-v2 →
`runs[6]` CACHE-SEAL-v1, executed via the v2 seal entities) is cleared for execution under §7. All
six gates PASS.

| gate | result |
|---|---|
| 1. Review credential (DELTA-2: RATIFIED v2 + PASS, C0/H0) | **PASS** |
| 2. Freeze integrity (14 v2 entities + machine sha256) | **PASS — 14/14 + machine match** |
| 3. Dependencies + runtime evidence (smoke2 13009) | **PASS — 6/6 present; smoke2 COMPLETED** |
| 4. Single-submission ledger (3 v2 names zero; v1 burn separate; namespace absent) | **PASS** |
| 5. Resource envelope (≤ 2 GPU / 16 CPU / 128 GB; squeue clear) | **PASS — 2 GPU / 8 CPU / 64 GB** |
| 6. Risk transcription (guard-fix scope, determinism, extrapolation) | **PASS — smoke2-settled** |

---

## 1. Gate 1 — Review credential (DELTA-2)

`M1_CACHE_CODE_REVIEW.md` DELTA-2 VERDICT (line 547) reads verbatim: **"Critical = 0, High = 0 →
`AMENDMENT_RATIFIED` (v2) + `PASS_STATIC_REVIEW`."** DD1 confirms the v2 amendment diff is exactly
in-scope (run_id v1→v2 + burn provenance + seal dependency re-point; runs[6].run_id stays SEAL-v1;
runs length 66=66); DD2 confirms guard-fix semantics (tolerate input-symlink location only, zero
other isolation weakening, `followed_target` audit-only, `video_sha256` retained); DD3 the 14 SHAs +
3-config rebind; DD4 the readlink topology re-derivation; DD5 smoke2 on the real frozen
`build_dataset_packs`. The two v2 review items are CLOSED. Execution authorization is the separate
role this document discharges.

## 2. Gate 2 — Freeze integrity (exact-hashes)

Recomputed sha256 (this session, read-only); every value equals `M1_CACHE_V2_FREEZE.md` §1:

| # | entity | recomputed (prefix) | freeze | match |
|---|---|---|---|---|
| 1 | `…_m1_cache_v2_common.py` | `56fbb403…` | `56fbb403…` | ✓ |
| 2 | `…_m1_evidence_pack_v2.py` | `54a0a97e…` | `54a0a97e…` | ✓ |
| 3 | `…_m1_cache_producer_v2.py` | `ee34eb9a…` | `ee34eb9a…` | ✓ |
| 4 | `…_m1_cache_seal_v2.py` | `62d1d100…` | `62d1d100…` | ✓ |
| 5 | `m1_cache_mhc_v2.json` | `4e42013d…` | `4e42013d…` | ✓ |
| 6 | `m1_cache_mhc_zh_v2.json` | `ce9bcab2…` | `ce9bcab2…` | ✓ |
| 7 | `m1_cache_seal_v2.json` | `6d6b4faf…` | `6d6b4faf…` | ✓ |
| 8 | `…_m1_cache_v2.sh` | `4c165fe2…` | `4c165fe2…` | ✓ |
| 9 | `…_m1_cache_seal_v2.sh` | `17f47f60…` | `17f47f60…` | ✓ |
| 10 | `…_m1_cache_mhc_v2.sbatch` | `ac986d8b…` | `ac986d8b…` | ✓ |
| 11 | `…_m1_cache_mhc_zh_v2.sbatch` | `8be1f454…` | `8be1f454…` | ✓ |
| 12 | `…_m1_cache_seal_v2.sbatch` | `e8f8e706…` | `e8f8e706…` | ✓ |
| 13 | `scgp_global_cache_replica_v2.schema.json` | `4bfcfea2…` | `4bfcfea2…` (carried) | ✓ |
| 14 | `scgp_global_cache_seal_v1.schema.json` | `f4605bb7…` | `f4605bb7…` (carried) | ✓ |
| — | `EXPERIMENT_PLAN.machine.json` | `ab0a06fb…` | `ab0a06fb…` | ✓ |

**14/14 + machine match.** All four task-mandated spot-checks confirm (common `56fbb403`,
evidence_pack `54a0a97e`, producer `ee34eb9a`, seal `62d1d100`; configs `4e42013d`/`ce9bcab2`/
`6d6b4faf`; schemas `4bfcfea2`/`f4605bb7`; machine `ab0a06fb`). The two artifact schemas are
correctly carried byte-unchanged (renaming contract-versioned ids would desync the machine
`artifact_schema_ids`). v2 config run_ids verified: MHC-v2 / MHC_zh-v2 / **SEAL-v1** (seal run_id
intentionally retained; its SLURM job name is `…_seal_v2`).

## 3. Gate 3 — Dependencies + runtime evidence (smoke2)

Dependencies unchanged from v1 and re-confirmed present: `torch` 2.6.0, `transformers` 4.49.0,
`numpy` 1.26.4, `jsonschema` 4.26.0, `decord` 0.6.0, `av` 17.0.0. `dependency_check()` fails closed.

Runtime evidence: `M1_SMOKE2_RECORD.md` — smoke2 job **13009** (`m1_smoke2_realpath`, COMPLETED,
elapsed 00:04:38) imports and calls the **frozen** `build_dataset_packs` + `canonical_video_path`
on real repo-escaping symlinked HateMM mp4 (the exact v1 burn surface, not a re-implementation):
744/744 symlinked mp4 processed with **no raise**, all followed-targets recorded external, all
forbidden zero-counters 0; 16-frame decode via followed link, R=4 determinism byte-identical, cert_v2
validate, offline load 5.94 s, GPU guard PASS (`CUDA_VISIBLE_DEVICES="0"`), peak 52.71 GiB. This
empirically settles the v1 burn and releases simulation rows 2/7/8/10 on the real cluster.

## 4. Gate 4 — Single-submission ledger

`sacct --starttime 2020-01-01 --name=<3 v2 names>` returns **zero rows** for
`lbscgp_global_r2_m1_cache_mhc_v2`, `…_mhc_zh_v2`, `…_seal_v2` — no prior submission of any v2 job.
The v1 burn jobs `lbscgp_global_r2_m1_cache_mhc_v1` (13003) / `…_mhc_zh_v1` (13004) are the
**separate v1 lineage** (both FAILED) and do not consume any v2 run's single submit. smoke2 (13009,
`m1_smoke2_realpath`) is a non-lineage independent job name. Artifact namespace
`artifacts/lb_scgp_global/v1/m1/` is **absent** (v1 burn wrote nothing; v2 has not run). The v2
single-submit ceremony is intact for all three runs.

## 5. Gate 5 — Resource envelope

Per-user cap: 16 CPU / 128 GB / 2 GPU. v2 sbatch headers (verified):

| run | job name | run_id | gres | cpus | mem |
|---|---|---|---|---|---|
| runs[4] CACHE-MHC-v2 | `lbscgp_global_r2_m1_cache_mhc_v2` | …-MHC-v2 | `gpu:a100:1` | 4 | 32G |
| runs[5] CACHE-MHC_zh-v2 | `lbscgp_global_r2_m1_cache_mhc_zh_v2` | …-MHC_zh-v2 | `gpu:a100:1` | 4 | 32G |
| runs[6] CACHE-SEAL-v1 | `lbscgp_global_r2_m1_cache_seal_v2` | …-SEAL-v1 | (none — CPU) | 4 | 32G |

Two caches concurrently → **2 GPU / 8 CPU / 64 GB**, within cap and equal to
`concurrency.m1_cache_parallel_max2`. Seal is CPU-only, runs after both caches. `squeue -u $USER` is
**empty** — no conflict. All three set no `--time` and `HF_HUB_OFFLINE=1`.

## 6. Gate 6 — Risk transcription (accepted)

- **Guard-fix scope (the v2 change):** `canonical_video_path` relaxes containment **only** for the
  mp4 leaf under `data/video/<ds>/All/` (parent must resolve in-repo; `..`/absolute/non-video-root
  still raise fail-closed); allowlist/forbidden/output checks byte-unchanged; `followed_target`
  audit-only; `video_sha256` retained. Only the two video sites changed. Confirmed by DD2 + smoke2.
- **R=4 determinism / `sigma_cache ≈ 0`:** greedy decoding + byte-identical input; smoke2 R=4
  byte-identical. M2 gate trivially satisfied.
- **Parse-failure fallback → canonical unresolved**, no rescue; no numeric parse-rate floor exists
  (DELTA-1 D4). Nonzero unresolved is not a failure.
- **Runtime extrapolation:** ~8 GPU-h/dataset (v1 smoke 50.67 s/video R=4; MHC 549 → ~7.7 h, MHC_zh
  579 → ~8.2 h), concurrent wall ~8 h; matches the pinned `estimated_gpu_hours=8`.
- **Two benign warnings** (temperature=1e-06 under greedy; `use_fast` deprecation) — cosmetic.

All risks benign, fail-closed, or fail-open-to-unresolved-by-contract. None can corrupt a sealed
artifact.

---

## 7. Authorization scope (binding)

The authorized action set is **exactly**:

1. **`runs[4]` CACHE-MHC-v2** (`sbatch scripts/slurm/lb_scgp_global_r2_m1_cache_mhc_v2.sbatch`) —
   **exactly one** submission.
2. **`runs[5]` CACHE-MHC_zh-v2** (`sbatch scripts/slurm/lb_scgp_global_r2_m1_cache_mhc_zh_v2.sbatch`)
   — **exactly one** submission. **May run in parallel** with runs[4].
3. **`runs[6]` CACHE-SEAL-v1** (`sbatch scripts/slurm/lb_scgp_global_r2_m1_cache_seal_v2.sbatch`) —
   **exactly one** submission, permitted **only after** both runs[4] and runs[5] reach `COMPLETED`
   **and** their artifacts are in place (`cache.jsonl` line count `== 4·U_D`, `cache_manifest.json`
   + `access_ledger.json` present, `zero_counters` all 0). CPU-only.

**Out of scope / prohibited:** any resubmission of a FAILED v2 lineage job (halt, collect evidence,
report `main`, await result-to-claim — do **not** resubmit); any edit to a frozen entity; any forced
release of a JobHeldUser hold; any GPU/MLLM/OCR/network run outside these three jobs; any
validation/test or label read; any re-run of smoke2 (already spent). If runs[4] or runs[5] fails,
runs[6] is **not** submitted.

Expected artifact targets (monitoring): MHC `cache.jsonl` → **2196** records (4×549 train);
MHC_zh → **2316** (4×579 train). Seal → `cache_seal_decision.json` (GO/STOP + recomputed Merkle
matched to each manifest).

## 8. Required statements

- No performance evidence exists or is claimed; this authorization is derived from static v2 freeze
  evidence + smoke2's runtime record. No accuracy / macro-F1, no training, no kNN.
- The only project gold is `parent_video_binary_label`; the M1 chain opens no train label and no
  validation/test content or label (labels enter only after the seal). The mp4 read is authorized
  train evidence; the followed external symlink target is recorded for audit only.
- M2, validation/test, and training remain **locked**; this authorization unlocks nothing downstream
  of the seal.
- Authorizer = Claude Opus 4.8, independent M1 v2 execution-authorization role. This is the only file
  written in the authorization phase; no code/config/schema/plan/machine JSON was edited; no job was
  submitted during authorization; no Python compute was run.
