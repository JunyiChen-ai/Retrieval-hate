# M0 REALBANK-RESOURCE-v2 Independent Execution Authorization

Date: 2026-07-13

Authorizer: **Claude Opus 4.8**, fresh independent **execution authorizer** for the
`lb_scgp_global_r2` M0 **REALBANK-RESOURCE-v2** run (run_id
`LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v2`, machine `runs[3]`). This role is deliberately **separate**
from the realbank-prep author (v2 clone + wrapper fix + freeze, `REALBANK_RESOURCE_V2_CLONE_FREEZE.md`),
the merged independent amendment-ratification + fresh 0C/0H static code-review role
(`REALBANK_RESOURCE_V2_CODE_REVIEW.md`), the v1 full-chain static auditor / result-to-claim reviewer,
and the executor (§9). I did not author, review, or implement the v2 entities; I independently
re-verified the authorization preconditions from files and cluster state read directly in this session.

## Authorizer boundary

Read-only authorization. **No mandated write obligation exists for this run.** The v2 config's
`hash_bindings.authoritative_inputs` is already complete at **11** entries (it additively binds
`REALBANK_FULLCHAIN_STATIC_AUDIT.md`) and does **not** bind the code-review document (verified:
`grep -c REALBANK_RESOURCE_V2_CODE_REVIEW` on the config = 0), so there is no post-review config
re-binding step; the executor verifies the **frozen** config hash `1d69b961…`. Apart from writing this
authorization document, I wrote nothing. I did **not** run project Python, imports, `py_compile`,
tests, `conda`, `sbatch`, an experiment, or any MLLM/OCR/API/model/network/GPU/training/evaluation, and
I touched no validation/test/held/cache/query content (only the two allowlisted train **feature** banks
and the authoritative/plan hashes were `sha256sum`-checked, read-only). No artifact under
`artifacts/lb_scgp_global/v1/m0/realbank_resource/` was created (confirmed absent, §5).

**Model-binding divergence declaration** (precedent: v1/v4 authorizations, v2 code review §Opus 4.8):
project discipline (`CLAUDE.md`) binds subagents to **Opus 4.8**; `AGENTS.md`'s "GPT-5.5 xhigh"
cross-model backend is unavailable this session, so this authorization runs on Opus 4.8. Independence
is enforced by fresh-context re-derivation of every check from on-disk / cluster state. Documented
process fact, not a defect.

---

## Verdict

**AUTHORIZED — exactly one CPU-only SLURM submission.**

All five authorization checks pass, the v1 `$TMPDIR` preflight-death class is independently confirmed
**closed** (§3), and there is no post-review binding obligation. Authorization scope = **exactly one**
invocation of:

```
sbatch scripts/slurm/lb_scgp_global_r2_m0_realbank_resource_v2.sbatch
```

The residual runtime risks (R-2 cross-process replay bit-determinism = highest residual, fail-closed;
R-1 `torch.load(weights_only=True)`; the two documentation-precision Low-1 deviations, byte-identical
v1→v2 and verified decision-inert) are non-blocking under §7. A fail-closed `decision:FAIL`/non-GO is a
**known, consciously-accepted STOP outcome, not grounds for re-submission** — the single-submit budget
is spent on submission.

---

## 1. Check 1 — Static-review credential (PASS)

`refine-logs/lb_scgp_global/REALBANK_RESOURCE_V2_CODE_REVIEW.md` exists and reads:

- **Verdict:** **`AMENDMENT_RATIFIED`** (the machine-plan v1→v2 run-id/schema-id REPLACE-in-place at
  `runs[3]` across all six run-id + two schema-id sites is minimal and faithfully landed; array length
  66 and downstream indices unchanged; four-file hash cascade exact; every diff hunk maps to the
  declared change list) **and `PASS_STATIC_REVIEW`** (the eight v2 entities are a proven byte-clone of
  the frozen v1 set modulo the lineage version token, plus exactly the audit §5.4 three-line wrapper
  fix; handoff row-A defect closed and re-derived to PASS; three-way interface alignment 23/47/11,
  `runs[3]` index-pin, dependency set, resource policy, fail-closed plumbing all hold on v2 tokens).
- **Severity counts:** **Critical = 0, High = 0, Medium = 0, Low = 2** (both documentation-precision,
  inherited from the v1 review; non-blocking). Pass criterion (Critical = 0 ∧ High = 0) met.
- The reviewer is a fresh 0C/0H role, separate from prep/audit/authorization/executor, who
  independently re-derived the 11-row full-chain handoff table (row A now PASS) and the 21-row runtime
  cross-check table (no row FAILs) on the v2 tokens.

**Result: PASS.**

## 2. Check 2 — Freeze-state no-drift (PASS)

I re-`sha256sum`'d all **eight** v2 entities this session and compared to
`REALBANK_RESOURCE_V2_CLONE_FREEZE.md` §1. **All eight match byte-for-byte:**

| # | entity | SHA256 (freeze §1) | on-disk |
|---|---|---|---|
| 1 | `configs/lb_scgp_global_r2/m0_realbank_resource_v2.json` | `1d69b961…cbf26b9d` | ✓ |
| 2 | `schemas/lb_scgp_global_r2/scgp_global_realbank_resource_v2.schema.json` | `4d95d128…545fb0271` | ✓ |
| 3 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v2_common.py` | `f90f153f…35a00fd8` | ✓ |
| 4 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v2_validate.py` | `ea703b3e…8bab3a91` | ✓ |
| 5 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v2_producer.py` | `e5e9a06a…ce3e0dab` | ✓ |
| 6 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v2_independent_verify.py` | `7ffa860b…b8b5d8e1` | ✓ |
| 7 | `scripts/wrappers/lb_scgp_global_r2_realbank_resource_v2.sh` | `348b056b…431c1419` | ✓ |
| 8 | `scripts/slurm/lb_scgp_global_r2_m0_realbank_resource_v2.sbatch` | `d7ab1e75…63153833` | ✓ |

**Machine plan pin:** `EXPERIMENT_PLAN.machine.json` on-disk =
`f4d54b78501b02253c14da2b42b2c04431f2f1c80f56c0f27b3420adf830fc9b` = the amendment target `f4d54b78…`;
the config's machine-plan binding internally holds the same `f4d54b78…` value (verified). **Plan pin
MATCH.**

**`runs[3]` content ↔ config field-for-field:** `jq '.runs[3]'` confirms `run_id` =
`LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v2`; `artifact_schema_ids` = `[scgp_global_realbank_resource_v2]`;
`artifact_paths` = `[artifacts/lb_scgp_global/v1/m0/realbank_resource/decision.json]` (**the v1 namespace
is intentionally preserved** — lineage version ≠ artifact-path version; the wrapper's hard-coded literal
at line 54 is the same v1 path, so the `CONFIG_ARTIFACT` gate passes with no drift); `slurm` =
`{cpu:16, ram_gb:96, gpu:0, env:"HateVideo", no_time_flag:true}`; `dependencies` =
`[LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v4]`; `status` =
`V2_CLONE_FROZEN_AFTER_V1_TMPDIR_BURN_PENDING_INDEPENDENT_REVIEW`. `run_id`/`schema_id`/`RUN3` move in
lock-step to v2 (v3-death / code↔plan-drift prevention). The v1 burn entities are **preserved on disk**
as evidence (not deleted).

**Result: PASS.** Zero drift; eight entities, machine plan pin, and `runs[3]`-vs-code lock-step all hold.

## 3. Check — v1 `$TMPDIR` preflight-death class CLOSED (independently re-derived)

The v1 single submit (job `12994`) died preflight because the wrapper wrote the validator→producer
handoff JSON to `$TMPDIR=/data/jehc223/home/tmp` (out-of-repo), which the producer's in-repo-only
`read_json → canonical_root_path` guard rejected. The v2 wrapper applies audit fix (a): it writes the
handoff to the **in-repo** `/data/jehc223/RGCL/slurm/tmp/` (lines 64–66:
`REALBANK_TMPDIR="/data/jehc223/RGCL/slurm/tmp"; mkdir -p; mktemp "$REALBANK_TMPDIR/…v2_validation.XXXXXX.json"`).

I independently re-verified the load-bearing facts this session:

- **`slurm/` is a real in-repo directory, not a symlink** (`ls -ld slurm` → `drwxr-x…`; `readlink slurm`
  → not a symlink), and `realpath -m slurm/tmp` = `/data/jehc223/RGCL/slurm/tmp`, which resolves **under
  ROOT** `/data/jehc223/RGCL`. Therefore `canonical_root_path`'s `resolved.relative_to(root)` returns
  `slurm/tmp/…json` with **no** `ValueError` → the producer read **passes** (the exact v1 failure is
  fixed).
- The v2 code review §3 additionally confirmed `slurm/tmp` lies outside all five `old_protected` roots
  (`slurm/` ≠ `scripts/slurm/`) and every no-clobber / git-diff / relevant_git_status pathspec, that the
  cleanup trap (pre-armed) `rm -f`s the transient, `mkdir -p` is idempotent, and no second
  ambient-environment-derived path remains in the chain. This authorizer accepts that re-derivation.

**Result: the v1 preflight-death class is closed.**

## 4. Check 3 — Dependency-availability evidence (PASS) — env is not frozen

Read-only `ls` of the `HateVideo` interpreter tree the sbatch's `conda activate HateVideo` resolves
(`/data/jehc223/miniconda3/envs/HateVideo/lib/python3.11/site-packages/`, `bin/python -> python3.11`)
shows the full required third-party set present: **`numpy` 1.26.4**, **`torch` 2.6.0** (used
function-level via `torch.load(..., weights_only=True)` in both bank loaders; torch 2.6 defaults to and
supports `weights_only=True` — R-1 resolved, §7), **`jsonschema` 4.26.0** (+ transitive
`jsonschema_specifications`/`referencing`/`rpds_py` cp311). Runtime-import corroboration:
`M0_ENV_REPAIR_RECORD.md` records `import jsonschema`/`numpy` runtime-OK on the HateVideo absolute
interpreter and the exact deferred `jsonschema` triple reproduced OK; the sbatch's
`source conda.sh; conda activate HateVideo` + the `require_slurm_realbank` `CONDA_DEFAULT_ENV=="HateVideo"`
gate guarantee the job runs in exactly that tree. The SLURM validator's `python_dependency_check`
(`find_spec` over `{numpy, torch, jsonschema}`, `sys.exit(1)` on any miss, **inside** the job **before**
the producer) is the v2-death (missing dep) prevention, fail-closed.

**Result: PASS.**

## 5. Check 4 — Single-submit ledger (PASS)

- **`sacct` all-time for the v2 job name is empty.**
  `sacct -u jehc223 --name=lbscgp_global_r2_realbank_resource_v2 --starttime=2000-01-01` → **zero rows**
  (the v2 job has never been submitted; the v1 burn was a **different** job name,
  `lbscgp_global_r2_realbank_resource_v1`).
- **`artifacts/lb_scgp_global/v1/m0/realbank_resource/` does not exist** (`ls` → "No such file or
  directory"). Note the product namespace is the **v1 path** (lineage v2 reuses the v1 artifact
  namespace by design); it was never populated (the v1 burn produced no artifact). No `decision.json` /
  manifest / access-ledger / publish-lock exists.
- **`slurm/logs/` has no v2 output:** `find slurm/logs -name '*realbank_resource_v2*'` → 0 files.

**Result: PASS.** The v2 single-submit budget is intact; this authorizes the first and only v2
submission.

## 6. Check 5 — Resources (PASS)

- **sbatch request** (`lb_scgp_global_r2_m0_realbank_resource_v2.sbatch`, re-read this session):
  `--cpus-per-task=16`, `--mem=96G`, **no** `--gres`/`--gpus` (GPU = 0), **no `--time`** (line 8
  comment), `--job-name=lbscgp_global_r2_realbank_resource_v2`, `--partition=slurmpartition`, and
  `source conda.sh; conda activate HateVideo` with `OMP/MKL/OPENBLAS_NUM_THREADS=16` (the R-2
  determinism precondition). Matches machine `runs[3].slurm` `{cpu:16, ram_gb:96, gpu:0, env:"HateVideo",
  no_time_flag:true}` exactly. Within per-user caps (16 CPU / 128 GB / 2 GPU): 16 ≤ 16, 96 ≤ 128, 0 ≤ 2.
- **`squeue -u jehc223` is empty at authorization time** — **no** running or pending jobs of mine
  (running CPUs held = 0). The lead flagged a possible B-line 8 CPU + 1 GPU job; it is **not** in the
  queue now, so there is **no** CPU/GPU contention and the 16-CPU request consumes the cap exactly
  (16 + 0 = 16 ≤ 16). **Executor pre-submit re-check required:** if a competing CPU job appears in
  `squeue` before the submit (such that running+requested CPU would exceed 16), **wait for it to finish
  and poll — do not submit over-cap.** (SLURM would in any case hold the v2 job PENDING rather than
  over-allocate, but the executor must not knowingly submit into a cap violation.)
- **`sinfo`:** partition `slurmpartition` up, node `foscsmlprd01` state `mix`,
  `CPUS(A/I/O/T)=36/220/0/256` → **220 idle CPUs** (≫ 16), `RealMemory 1000000MB`. A CPU-only
  16-core / 96 GB job is schedulable; initial `PENDING (JobHeldUser)` is expected and must auto-release
  (do not force).

**Result: PASS** (with the executor pre-submit contention re-check noted above).

## 7. Risk resolutions (recorded and released)

- **R-2 (cross-process replay bit-determinism) — ACCEPT (highest residual, fail-closed).** In-job
  run1==run2 is same-process → bit-identical. The producer↔independent-verifier cross-check additionally
  requires the verifier's independently recomputed replay digest to byte-match the producer's, relying
  on LAPACK `eigvalsh`/`svd` determinism at fixed thread count (`OMP/MKL/OPENBLAS=16`, set in the sbatch,
  shared by both processes) with `floatify` absorbing last-ULP noise. A miss → `decision:FAIL`
  (**STOP**), never a false GO. (The v2 audit notes `require_slurm_realbank` does not *assert* the
  thread vars == 16 — an optional future hardening, not this bug; the sbatch does export them.) Risk
  acceptor = coordination session; this authorizer releases with knowledge.
- **R-1 (`torch.load(weights_only=True)`) — ACCEPT (Low, fail-closed).** Precedented by
  `lb_scgp_sanitize_inputs.py` / `dataset.py`; installed **torch 2.6.0** supports and defaults to
  `weights_only=True`; a rejected payload raises cleanly (no artifact), never a silent wrong result.
- **Two Low-1 decision-inert deviations from the accepted v4 math — ACCEPT (byte-identical v1→v2).**
  `factor_from_psd_gram` drops v4's non-PASS early-return (but the GO gate independently requires
  `rank_audit["status"]=="PASS"` per dataset → cannot manufacture a GO) and `orth_cap` drops an unused
  `singular_values` field (`q_info` never read). Both provably GO-inert; documentation-precision only.
- **NON-SCIENCE placeholder (R-4) — disclosed end-to-end.** `b_struct` is a deterministic, label-blind
  placeholder (`is_science=false`, verifier REJECTS `is_science=True`); certifies nothing; science owner
  retains the overrule right before any downstream scientific claim. This run emits **no accuracy /
  macro-F1** and does no training or kNN.

## 8. Ledger snapshot (authorization-time)

| field | value |
|---|---|
| v2 job name | `lbscgp_global_r2_realbank_resource_v2` |
| prior v2 submissions (sacct, all-time) | **0** |
| authorized submissions remaining | **exactly 1** |
| `artifacts/lb_scgp_global/v1/m0/realbank_resource/` (v1-namespace) | absent |
| v2 slurm logs | none |
| `squeue -u jehc223` | **empty** — 0 running/pending; no contention (executor re-checks before submit) |
| partition / node | `slurmpartition` up / `foscsmlprd01` mix / 220 idle CPUs / 1000000 MB |
| resource request | 16 CPU / 96 GB / 0 GPU / no `--time` / HateVideo |
| config hash to verify | `1d69b961c68163163ececb9e75ded4630661d2e276bbcc63302d7f76cbf26b9d` (frozen; no post-binding move) |
| machine plan pin | `f4d54b78501b02253c14da2b42b2c04431f2f1c80f56c0f27b3420adf830fc9b` |
| GO criterion | `peak_rss ≤ 103079215104 (96 GiB) ∧ rank_eps(G0) ≤ d (all ds) ∧ in-job replay match (all ds) ∧ all injections REJECT` |
| v1 preflight-death class | **closed** (fix (a), in-repo `slurm/tmp/` handoff, re-derived §3) |

---

## 9. Executor instructions

1. **Verify the eight frozen hashes first**, especially
   `sha256sum configs/lb_scgp_global_r2/m0_realbank_resource_v2.json` == `1d69b961…cbf26b9d` (frozen; no
   post-review binding move). The other seven entities must match §2. Do **not** edit
   `REALBANK_RESOURCE_V2_CODE_REVIEW.md` if it is under runtime hash validation — but note the v2 config
   binds `REALBANK_FULLCHAIN_STATIC_AUDIT.md`, not the code review, so the runtime-frozen doc set is the
   11 `authoritative_inputs`.
2. **Re-check `squeue -u jehc223` immediately before submit.** If a competing CPU job would push
   running+requested CPU over 16, **wait and poll** — do not submit over-cap. Empty queue at
   authorization time.
3. Submit **exactly once**: `sbatch scripts/slurm/lb_scgp_global_r2_m0_realbank_resource_v2.sbatch`. Do
   not set `--time`.
4. **Record the returned job id** and the resulting
   `slurm/logs/lbscgp_global_r2_realbank_resource_v2_<jobid>.{out,err}` paths.
5. Expect initial `PENDING (JobHeldUser)`; **wait for auto-release** — do not force. At N=549/579 the
   pure-CPU pipeline should finish in minutes once running.
6. **After any terminal state, do not resubmit** regardless of outcome. A fail-closed
   `decision:FAIL`/non-GO (R-2 replay mismatch, rank/cap miss, or an injection failing to REJECT) is an
   **authorized, consciously-accepted STOP** (§7), not grounds for re-submit.
7. **On GO/PASS** (verifier stamps `decision:PASS`, producer publishes manifest + counters): route to a
   **fresh independent artifact review** (separate role) before any downstream unlock — do not
   self-certify. Collect `decision.json` (GO/STOP), `resource_peak` (peak RSS) and its distance to the
   96 GiB cap, `rank_tail`, the replay-hash match, and the 11 injection + 15 tamper REJECTs.
8. **On FAIL / fail-closed non-publish**: route to a **fresh result-to-claim review** with the job log
   as evidence. If it dies at a **preflight class** again, that would re-trigger the pause + full-chain
   audit escalation; a numeric-section fail-closed is the pre-accepted informative class.
9. The executor role is separate from realbank-prep/freeze, amendment/code-review, full-chain audit, and
   this authorization role.

---

## Required statements

- No performance evidence exists and none is claimed; the realbank run emits **no accuracy / macro-F1**
  and does **no training or kNN** — it is a train-bank static/resource microbenchmark (`resource_peak`,
  `rank_tail ≤ d`, in-job replay determinism, `robust_coverage` fail-open, isolation-injection
  defenses).
- The only project gold is `parent_video_binary_label`. No segment/frame/timestamp/span/localization/
  stance/target/mechanism/rationale/fragment gold is assumed or introduced; the realbank code opens
  train **features** (allowlisted + hash-checked) and never reads train **labels**.
- Run4 (M1 cache), MLLM/cache, validation/test, and training remain **locked**. This authorization
  covers only one CPU-only static/resource microbenchmark SLURM job. The `is_science=false` placeholder
  (R-4) must be overruled or replaced by the science owner before any *scientific* claim rests on it.
- Authorization scope = exactly one `sbatch scripts/slurm/lb_scgp_global_r2_m0_realbank_resource_v2.sbatch`.
- Authorizer = Claude Opus 4.8 fresh independent execution-authorization role, separate from the
  realbank-prep/freeze, merged amendment/code-review, full-chain audit, and executor roles. This
  authorization made no write beyond this document (no config re-binding obligation applies to v2).
