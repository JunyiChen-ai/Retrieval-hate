# M0 Run2-v2 Execution Authorization

Authorizer: Claude Opus 4.8, fresh / zero-context execution-authorization reviewer.
Role separation: this authorizer is a **distinct role** from the static code
reviewer (who wrote `M0_RUN2_V2_CODE_REVIEW_FIX2.md`) and from the future executor
(who will run `sbatch`). This authorizer performed **read-only** checks only and did
**not** submit, and is not permitted to submit, any SLURM job.

Authorization timestamp: 2026-07-12T15:52:46Z (local NZ date 2026-07-13).
Repo HEAD at authorization: `386265f0306c9455cd0f863ca31e49d1c5520fc2`.

Tools used were limited to read-only inspection (`rg`/`sed`/`nl`/`jq`/`awk`/
`bash -n`/`sha256sum`/`find`/`ls`/`wc`/`git status`/`git diff`/`squeue`/`sacct`/
`sinfo`). The only file written is this authorization document.

## Verdict

**AUTHORIZED** — exactly one SLURM submission of the v2 synthetic-KKT validator.

All four gate checks passed. No blockers.

---

## 1. Static-review credential — PASS

`refine-logs/lb_scgp_global/M0_RUN2_V2_CODE_REVIEW_FIX2.md` exists (28,603-byte
file present in the directory listing). Its verdict is **PASS_STATIC_REVIEW**.
Severity counts: **Critical = 0, High = 0** (Medium = 3, Low = 2). The gate
condition (exists ∧ PASS_STATIC_REVIEW ∧ Critical=0 ∧ High=0) is satisfied.

Advisory (does not block, per the gate as written): the static review carries a
conditional escalation **M-A** — *if* a future authority rules that the synthetic
fixture's `G0` must itself be a realizable rank-`<=d` PSD `Z0 Z0^T` (not merely
symmetric + unit-diagonal), M-A would become High and block. No such ruling exists
in the record; the code reviewer independently adjudicated the `G0` baseline change
as legitimate implementation repair against the frozen `FINAL_PROPOSAL.md` text
(`G0 ∈ S^N` = symmetric; "no hard rank constraint"). This authorizer does not
overturn that ruling and is not the authority contemplated by M-A. Two fail-closed
residuals (**M-B** rank-deficient-construction convergence not statically provable;
**M-C** `jsonschema` availability in `HateVideo` unproven statically) cannot cause a
false PASS but could consume the single authorized attempt with an
environment/convergence failure — flagged to the executor below.

## 2. Freeze-state drift — PASS (no drift)

Re-computed SHA256 of all 9 v2 entities; every hash equals the value declared in
`M0_RUN2_V2_IMPLEMENTATION_FIX2_FREEZE.md`.

| # | Path | SHA256 (recomputed) | Freeze-doc match |
|---|---|---|---|
| 1 | `configs/lb_scgp_global_r2/m0_synth_kkt_v2.json` | `5545826820cd86f588fb43fd73b4070767fb9f904ea683fc2245f0903d48f700` | yes |
| 2 | `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_payload_v2.schema.json` | `735250091f6a92ef787d9eadccca3c438379cc07f2418439401916270eced9a2` | yes |
| 3 | `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_case_v2.schema.json` | `df3616ff50c54dc756bb600c6ef26626afafe070678450d10ca73b9ff83ddcac` | yes (parity) |
| 4 | `scripts/analysis/lb_scgp_global_r2_run2_v2_common.py` | `5ef8f677e13c0586675a3072b85774b90b7eb5e6ba8da91188f128d3a6d6e24f` | yes |
| 5 | `scripts/analysis/lb_scgp_global_r2_run2_v2_validate.py` | `4389c4a1d1cbe21ec516ec414c472ebff075bbfb4a8041939163f478bdc3fc36` | yes |
| 6 | `scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py` | `8c4cc842aa53e3d8293744449101282a33abb7715691653bb934b2e563b1cd51` | yes |
| 7 | `scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py` | `795b56f852534c2cfb4264c9dec2f43dd4abc75a4655282465b6546d39111ef1` | yes |
| 8 | `scripts/wrappers/lb_scgp_global_r2_run2_v2.sh` | `14eb036f17b77037fd89624c0f9f7487432fa476da540b0fbc5f503719320716` | yes (parity) |
| 9 | `scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v2.sbatch` | `f91462303e4b28eb0722d546f135c00c2aeacf56151e27d5421f211520ba94bf` | yes (parity) |

Git tracking note: entities 1–7 and 9 are tracked and unmodified
(`git status --porcelain` clean); entity 8 (the wrapper) is untracked (`??`),
consistent with the session's start-of-run git snapshot. Tracking state does not
affect drift: drift is defined by hash equality to the freeze document, which holds
exactly for all nine. `bash -n` on the sbatch and wrapper: OK.

## 3. Single-submit ledger — PASS (zero prior v2 submissions)

- **sacct** (`sacct -u jehc223 --starttime 2026-07-12`, and a wider
  `--starttime 2026-06-01` scan): the only `lbscgp_global_r2` jobs are
  `12901 lbscgp_global_r2_run1` (COMPLETED, Run1 contract freeze),
  `12902 lbscgp_global_r2_run2_synth_kkt` (FAILED, v1 lineage), and
  `12904 lbscgp_global_r2_run2` (FAILED, v1 lineage). The v2 sbatch's job name is
  `lbscgp_global_r2_run2_v2` (sbatch `:5`); a targeted search
  `sacct … | grep global_r2_run2_v2` returned **no match** (exit 1). The other `v2`
  hits in sacct (`gen_archive_v2`, `consv2_*`, `arcv2_*`, `genV2_*`) belong to
  unrelated experiment families, not this lineage. **Zero v2 submissions in history.**
- **Artifact directory**: `artifacts/lb_scgp_global/v2/m0/synth_kkt/` does not exist;
  `artifacts/lb_scgp_global/` contains only `v1`. (The `find … -iname "*v2*"` hit
  `artifacts/lb_scgp/v2/CONFIG_FREEZE.json` is a different top-level lineage —
  `lb_scgp`, not `lb_scgp_global` — and is unrelated.)
- **slurm/logs**: no `lbscgp_global_r2_run2_v2_*.{out,err}` output file exists.

## 4. Resource & environment — PASS

sbatch `scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v2.sbatch`:
`--cpus-per-task=8`, `--mem=64G`, **no `--time`** (comment only, `:8`), **no GPU**
(no `--gres`/`--gpus`), `conda activate HateVideo` (`:13`). These match the config
`run.slurm` block exactly: `{cpu:8, ram_gb:64, gpu:0, env:"HateVideo",
no_time_flag:true}`. Request is well within the per-user cap (16 CPU / 128 GB /
2 GPU).

Cluster state at authorization:
- `squeue -u jehc223`: **empty** — this user has no queued or running jobs.
- `sinfo`: partition `slurmpartition` up, node `foscsmlprd01` MIXED,
  `CPUS(A/I/O/T)=52/204/0/256` → **204 idle CPUs** (≫ 8 requested);
  `RealMemory=1000000MB`, `AllocMem=409600MB` → ~590 GB free (≫ 64 GB requested).

Ample headroom; no contention.

---

## Authorization scope

This document authorizes **exactly one** invocation:

```
sbatch scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v2.sbatch
```

Nothing else is authorized. No re-submission, no variant, no other config, no manual
Python/producer/verifier run outside this sbatch, no GPU job, no Run3/M1 unlock.

## Ledger snapshot at time of authorization

`squeue -u jehc223`:

```
JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)
(no rows — user has no jobs)
```

`sinfo -o "%P %a %D %t %C %m"`:

```
slurmpartition* up 1 mix 52/204/0/256 1000000   (CPUS A/I/O/T; MEMORY MB)
```

`sacct -u jehc223 …` (lb_scgp_global_r2 lineage only; no v2 present):

```
12901  lbscgp_global_r2_run1            COMPLETED  2026-07-12T12:16:10   (Run1 contract freeze)
12902  lbscgp_global_r2_run2_synth_kkt  FAILED     2026-07-12T12:46:12   (v1 lineage)
12904  lbscgp_global_r2_run2            FAILED     2026-07-12T12:55:48   (v1 lineage)
        lbscgp_global_r2_run2_v2         <absent>   —                    (zero v2 submissions)
```

## Executor instructions (须知)

1. Submit **exactly once**: `sbatch scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v2.sbatch`.
   Do not set `--time`. Expect the job to begin as `PENDING (JobHeldUser)` and
   auto-release; do not force-release.
2. **Record the job id** returned by `sbatch`, and the resulting
   `slurm/logs/lbscgp_global_r2_run2_v2_<jobid>.{out,err}` paths.
3. Once the job reaches any terminal state (COMPLETED / FAILED / CANCELLED /
   TIMEOUT / OOM / node failure), **do not resubmit under any outcome**. The
   single-submit budget is then spent. A further attempt requires a new
   execution-authorization pass, not this document.
4. Be aware the run is **fail-closed** on two environmental/numeric residuals
   (static-review M-B, M-C): a missing `jsonschema` in `HateVideo` or a
   non-converging rank-deficient construction will make the job exit non-zero
   without a false PASS — but it still consumes the one authorized attempt. Also
   note L-A: do not perform an unauthorized re-run, as the wrapper's exit-trap
   cleanup could delete a prior successful run's artifacts.
5. On a genuine wrapper PASS (`decision == "PASS"` gate), the produced artifacts
   under `artifacts/lb_scgp_global/v2/m0/synth_kkt/` must go through a **separate,
   fresh artifact review** before any claim is made. This authorization covers
   submission only; it is not artifact acceptance and not performance evidence.

## Required statements

- This authorization is not performance evidence and makes no performance claim.
- Run3, M1, MLLM/cache, validation/test, training, and realbank remain locked.
- Authorizer role (this document) is separate from the static-code-review role and
  from the executor role. This authorizer did not and may not submit any job.
