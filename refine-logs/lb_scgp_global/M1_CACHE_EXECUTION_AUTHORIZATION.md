# M1 CACHE Execution Authorization

Date: 2026-07-13

Authorizer: **Claude Opus 4.8**, **independent M1 execution-authorization role** — separate from
m1-prep (amendment author + implementer + freezer), the independent amendment reviewer, the fresh
0C/0H static code reviewer, and (nominally) the executor. In this session the same agent instance
holds authorizer + executor, but the authorization below is derived **only** from the frozen
evidence and re-verified read-only checks; execution proceeds strictly within the scope this
document fixes.

Discipline for the authorization phase: read-only. Shell limited to `sha256sum`, `ls`, `grep`,
`sacct`, `squeue`, `sbatch --help`-class introspection. **No** SLURM submission during
authorization; **no** Python; **no** GPU/MLLM/OCR/network/model run; **no** validation/test or
train-label read; the only file written in this phase is this document.

Model declaration: ran as Claude Opus 4.8 (`claude-opus-4-8`, 1M context), per CLAUDE.md's
subagent-model requirement. No model deviation.

---

## 0. Verdict

**AUTHORIZED.** The M1 cache block (`runs[4]` CACHE-MHC-v1 / `runs[5]` CACHE-MHC_zh-v1 →
`runs[6]` CACHE-SEAL-v1) is cleared for execution under the exact scope in §7. All six
authorization gates below PASS.

| gate | result |
|---|---|
| 1. Review credential (RATIFIED + PASS, C0/H0) | **PASS** |
| 2. Freeze integrity (14 entities + machine sha256) | **PASS — 14/14 + machine match** |
| 3. Dependencies + runtime evidence (smoke 13002) | **PASS — 6/6 present; smoke COMPLETED** |
| 4. Single-submission ledger (3 lineage names zero; namespace absent) | **PASS — 0 prior submits** |
| 5. Resource envelope (≤ 2 GPU / 16 CPU / 128 GB; squeue clear) | **PASS — 2 GPU / 8 CPU / 64 GB** |
| 6. Risk transcription (R=4 determinism, fallback, extrapolation, warnings) | **PASS — all benign / fail-closed** |

---

## 1. Gate 1 — Review credential

`refine-logs/lb_scgp_global/M1_CACHE_CODE_REVIEW.md` carries the fresh 0C/0H independent review plus
a post-fix **DELTA REVIEW** section. The DELTA VERDICT (M1_CACHE_CODE_REVIEW.md:450) reads
verbatim: **"Critical = 0, High = 0 → `AMENDMENT_RATIFIED` + `PASS_STATIC_REVIEW`."** Both HIGH
must-fixes (ruling ① historical-snapshot rollback; §4.5 GPU-guard idiom swap) are recorded as
applied in-scope and cascade-consistent (M1_CACHE_CODE_REVIEW.md:387–447, D1–D4). The two review
items (amendment ratification + fresh static code review) are CLOSED. The review explicitly notes
that "exact-hashes / no-clobber review and execution authorization" are a separate role — this
document discharges that separate role.

## 2. Gate 2 — Freeze integrity (exact-hashes)

Recomputed sha256 (this session, read-only) for all 14 frozen entities and the machine plan; every
value equals `M1_CACHE_FREEZE.md` §1 and the §FIX cascade:

| # | entity | recomputed sha256 (prefix) | freeze | match |
|---|---|---|---|---|
| 1 | `scgp_global_cache_replica_v2.schema.json` | `4bfcfea2…` | `4bfcfea2…` | ✓ |
| 2 | `scgp_global_cache_seal_v1.schema.json` | `f4605bb7…` | `f4605bb7…` | ✓ |
| 3 | `…_m1_cache_v1_common.py` | **`601d61e2…`** | `601d61e2…` (FIX-2) | ✓ |
| 4 | `…_m1_evidence_pack_v1.py` | `ca9d94ec…` | `ca9d94ec…` | ✓ |
| 5 | `…_m1_cache_producer_v1.py` | `c82b87d6…` | `c82b87d6…` | ✓ |
| 6 | `…_m1_cache_seal_v1.py` | `399e7956…` | `399e7956…` | ✓ |
| 7 | `m1_cache_mhc_v1.json` | **`23c777de…`** | `23c777de…` (FIX-1) | ✓ |
| 8 | `m1_cache_mhc_zh_v1.json` | **`57bf435d…`** | `57bf435d…` (FIX-1) | ✓ |
| 9 | `m1_cache_seal_v1.json` | **`94147df7…`** | `94147df7…` (FIX-1) | ✓ |
| 10 | `…_m1_cache_v1.sh` | `a364a7df…` | `a364a7df…` | ✓ |
| 11 | `…_m1_cache_seal_v1.sh` | `81c1ccac…` | `81c1ccac…` | ✓ |
| 12 | `…_m1_cache_mhc_v1.sbatch` | `4516223e…` | `4516223e…` | ✓ |
| 13 | `…_m1_cache_mhc_zh_v1.sbatch` | `b340efaa…` | `b340efaa…` | ✓ |
| 14 | `…_m1_cache_seal_v1.sbatch` | `5a42c3e7…` | `5a42c3e7…` | ✓ |
| — | `EXPERIMENT_PLAN.machine.json` | **`7638ac78…`** | `7638ac78…` | ✓ |

**14/14 entities + machine plan match.** The four task-mandated spot-checks (common.py `601d61e2`,
configs `23c777de`/`57bf435d`/`94147df7`, machine `7638ac78`) all confirm — the FIX-1 cascade and
FIX-2 idiom swap are present in the on-disk bytes that will run.

## 3. Gate 3 — Dependencies + runtime evidence

Site-packages (`/data/jehc223/miniconda3/envs/HateVideo/lib/python3.11/site-packages`), read-only:
`torch` 2.6.0, `transformers` 4.49.0, `numpy` 1.26.4, `jsonschema` 4.26.0, `decord` 0.6.0, `av`
17.0.0 — **all present** (the frame loader uses decord-first / av-fallback; both present). The
producer's `dependency_check()` fails closed before the model loads if any is missing.

Runtime evidence: `M1_CACHE_FREEZE.md` §2 / `M1_SMOKE_RECORD.md` — the approved non-lineage smoke
(job **13002**, COMPLETED, elapsed 00:08:45, real `--gres=gpu:a100:1` alloc) empirically exercised
the exact frozen code path (offline model load 5.97 s, `load_video_frames` 10/10 decode, processor
`videos=[frames]` accepted, generate + `parse_certificate`, R=4 byte-identical determinism, GPU
peak 52.71 GiB < 80 GiB). This releases the three DEFERRED-TO-RUNTIME simulation rows and the FIX-2
GPU-guard HIGH on the real cluster.

## 4. Gate 4 — Single-submission ledger

`sacct -u $USER --starttime 2020-01-01 --name=<3 lineage names>` returns **zero rows** for all of
`lbscgp_global_r2_m1_cache_mhc_v1`, `lbscgp_global_r2_m1_cache_mhc_zh_v1`,
`lbscgp_global_r2_m1_cache_seal_v1` — no prior submission of any lineage job. A broad
`sacct … | grep m1_cache` over July returns NONE. The smoke job `lbscgp_global_r2_m1_smoke`
(13002, COMPLETED) is present and is an **independent, non-lineage job name** that consumes no run
entry's single submit (per code-review ruling ③). Artifact namespace `artifacts/lb_scgp_global/v1/m1/`
is **absent** (only `…/v1/m0/` exists). No `slurm/logs/*cache_mhc*|*cache_seal*` files exist. The
single-submit ceremony is intact and unburned for all three runs.

## 5. Gate 5 — Resource envelope

Per-user cap (CLAUDE.md): **16 CPU / 128 GB / 2 GPU**. sbatch headers (verified this session):

| run | job name | gres | cpus | mem |
|---|---|---|---|---|
| runs[4] CACHE-MHC-v1 | `lbscgp_global_r2_m1_cache_mhc_v1` | `gpu:a100:1` | 4 | 32G |
| runs[5] CACHE-MHC_zh-v1 | `lbscgp_global_r2_m1_cache_mhc_zh_v1` | `gpu:a100:1` | 4 | 32G |
| runs[6] CACHE-SEAL-v1 | `lbscgp_global_r2_m1_cache_seal_v1` | (none — CPU) | 4 | 32G |

Two caches run concurrently → **2 GPU / 8 CPU / 64 GB**, within the cap and equal to the plan's
`concurrency.m1_cache_parallel_max2` (`max_gpu_total=2, max_cpu_total=8, max_ram_total_gb=64`). The
seal is CPU-only (0 GPU) and runs after both caches complete, so peak concurrency never exceeds the
two-cache footprint. `squeue -u $USER` is currently **empty** — no running/pending job of mine to
conflict. All three sbatch set no `--time` (per project rule) and `HF_HUB_OFFLINE=1`.

## 6. Gate 6 — Risk transcription (accepted)

- **R=4 determinism / `sigma_cache ≈ 0`:** replicas use byte-identical evidence-pack input + greedy
  decoding (`do_sample=False, num_beams=1, torch.manual_seed(0)`); the smoke showed 10/10 videos
  byte-identical across all four replicas. The M2 `sigma_cache` gate is trivially satisfied. Pinned
  fact, not a runtime risk.
- **Parse-failure fallback:** any strict-JSON failure → `canonical_unresolved_observables()` with
  `parse_flags`, no re-prompt / schema rescue (contract-correct; the smoke's `hate_video_104`
  exercised this deterministically). No numeric parse-rate floor exists in the plan or FINAL_PROPOSAL
  (DELTA D4), so a nonzero unresolved count is **not** a failure condition.
- **Runtime extrapolation:** smoke per-video wall ≈ 50.67 s (R=4) → MHC 549 videos ≈ **7.73 GPU-h**,
  MHC_zh 579 videos ≈ **8.15 GPU-h**, concurrent wall ≈ **~8 h**; matches the amendment's pinned
  `estimated_gpu_hours=8` per cache run. Any real dedup only reduces this. Long-running, no `--time`,
  JobHeldUser auto-release expected.
- **Two benign warnings:** `.err` "`do_sample` is False but `temperature`=1e-06" (greedy ignores it;
  determinism proven) and the transformers `use_fast` deprecation notice. Neither touches a frozen
  entity or correctness; **not blocking**.

All risks are benign, fail-closed, or fail-open-to-unresolved-by-contract. None can corrupt a
sealed artifact.

---

## 7. Authorization scope (binding)

The authorized action set is **exactly**:

1. **`runs[4]` CACHE-MHC-v1** (`sbatch scripts/slurm/lb_scgp_global_r2_m1_cache_mhc_v1.sbatch`) —
   **exactly one** submission.
2. **`runs[5]` CACHE-MHC_zh-v1** (`sbatch scripts/slurm/lb_scgp_global_r2_m1_cache_mhc_zh_v1.sbatch`)
   — **exactly one** submission. **May run in parallel** with runs[4] (`m1_cache_parallel_max2`).
3. **`runs[6]` CACHE-SEAL-v1** (`sbatch scripts/slurm/lb_scgp_global_r2_m1_cache_seal_v1.sbatch`) —
   **exactly one** submission, permitted **only after** both runs[4] and runs[5] reach `COMPLETED`
   **and** their artifacts are in place (`cache.jsonl` line count `== 4·U_D`, `cache_manifest.json`
   + `access_ledger.json` present, `zero_counters` all 0). The seal is CPU-only.

**Out of scope / prohibited:** any resubmission of a FAILED lineage job (on any FAILED cache, halt,
collect evidence, report to `main`, await result-to-claim — do **not** resubmit); any edit to a
frozen entity, config, schema, plan, or machine JSON; any forced release of a JobHeldUser hold
(await auto-release); any GPU/MLLM/OCR/network run outside these three jobs; any validation/test or
label read. The smoke (13002) is already spent and is **not** re-run. If runs[4] or runs[5] fails,
runs[6] is **not** submitted.

Expected artifact targets (for monitoring, from freeze §3–4 / smoke): MHC `cache.jsonl` →
**2196** records (4×549 at full dedup); MHC_zh → **2316** (4×579). Seal → `cache_seal_decision.json`
with a GO/STOP decision + recomputed Merkle root matched to each manifest.

## 8. Required statements

- No performance evidence exists or is claimed; this authorization is derived from static freeze
  evidence + the prior smoke's runtime record. No accuracy / macro-F1, no training, no kNN.
- The only project gold is `parent_video_binary_label`; the M1 chain opens no train label and no
  validation/test content or label (labels enter only after the seal decision). The train-ID
  allowlist is split membership, not gold.
- M2 (comparator freeze), validation/test, and training remain **locked**; this authorization
  unlocks nothing downstream of the seal.
- Authorizer = Claude Opus 4.8, independent M1 execution-authorization role. This document is the
  only file written in the authorization phase; no code / config / schema / plan / machine JSON was
  edited; no job was submitted during authorization; no Python was executed.
