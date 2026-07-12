# M0 Run2-v3 Independent Execution Record

Date: 2026-07-13

Executor: **Claude Opus 4.8** (`claude-opus-4-8`), independent **executor** role for the
`lb_scgp_global_r2` M0 Run2 **v3** lineage. This role is deliberately **separate** from the
v3-setup/freeze role (`M0_RUN2_V3_CLONE_FREEZE.md`), the static code-review role
(`M0_RUN2_V3_CODE_REVIEW.md`), and the execution-authorization role
(`M0_RUN2_V3_EXECUTION_AUTHORIZATION.md`). I did not create, review, or authorize the v3 entities;
I performed exactly one authorized SLURM submission, monitored it to a terminal state, and recorded
the read-only evidence below.

**Model-binding divergence declaration** (precedent: `M0_RUN2_V3_EXECUTION_AUTHORIZATION.md`
§Model-binding): `AGENTS.md:15` binds the main-dialogue subagent to "GPT-5.5 xhigh"; that backend
is unavailable this session, so the executor runs on the `CLAUDE.md`-bound **Opus 4.8**. Documented
process fact, not a defect; the executor performs no scientific judgement.

**Scope discipline:** the only file written by this executor role is this record. No code, config,
schema, environment, cache, validation/test data, or artifact was created or modified. The single
authorized v3 submission is **spent**; **the v3 lineage will not be resubmitted regardless of
outcome** (per `M0_RUN2_V3_EXECUTION_AUTHORIZATION.md` §8.4).

---

## Verdict

**FAILED — fail-closed non-publish. Decision = FAIL. No artifact produced.**

Root cause = a **statically foreseeable infrastructure/consistency failure**: a document-vs-code
`run_id` drift between the authoritative experiment-machine plan (still `…-SYNTH-KKT-v2`) and the
code-side `RUN2` constant (already `…-SYNTH-KKT-v3`). This is **not** the M-B rank-construction
convergence risk and **not** a scientific/numeric failure; execution died in the producer's very
first authoritative-input check, **before** any rank-deficient construction ran, so **M-B was never
reached**.

---

## 1. Pre-flight verification (all PASS — submission gated on these)

| # | Check | Result |
|---|---|---|
| 1 | `M0_RUN2_V3_EXECUTION_AUTHORIZATION.md` exists and reads `AUTHORIZED — exactly one CPU-only SLURM submission` (§Verdict) | PASS |
| 2 | `sha256sum` of the three spot-checked entities matches the authorization doc byte-for-byte | PASS |
| 3 | `squeue -u jehc223` had no same-named job at submit time | PASS |
| 4 | `artifacts/lb_scgp_global/v3/` did not exist at submit time | PASS |

SHA256 spot-check (re-hashed at execution time = authorization doc §2):

- `scripts/wrappers/lb_scgp_global_r2_run2_v3.sh` → `8d9123e9f4eec357a91bd94cbf6c292a3bb188496011845c706f3e34b72d66d3` ✓
- `scripts/analysis/lb_scgp_global_r2_run2_v3_producer.py` → `6ef3a4a8146ec9b2a2a94236c1e40f0ebf27aa862b89d27a60e92499b21f5114` ✓
- `scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v3.sbatch` → `4495ec3c49970b7024ebf443d845ec5f325013b2190116c547827c6f5de6b3d9` ✓

---

## 2. Submission and timeline

| field | value |
|---|---|
| submit command | `sbatch scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v3.sbatch` |
| submit time | **2026-07-13 04:30:04 NZST** (UTC `2026-07-12T16:30:04Z`) |
| returned job id | **12974** (name `lbscgp_global_r2_run2_v3`) |
| initial state | `PENDING (JobHeldUser)` — expected; left to auto-release (no `scontrol release`, no cancel) |
| held duration | ~30 min (submit 04:30:04 → release ~05:00:19 NZST) |
| auto-release + run + die | `Start = End = 2026-07-13T05:00:19` (`sacct`); ran and failed instantly |
| terminal state | **FAILED**, `Elapsed 00:00:00`, `ExitCode 1:0` (`12974` and `12974.batch` both FAILED) |
| detected terminal at | 05:00:33 NZST (background poller) |

`sacct -j 12974` (terminal):
```
12974            lbscgp_global_r2_run2_v3   FAILED  2026-07-13T05:00:19  2026-07-13T05:00:19  00:00:00  1:0  0:0
12974.batch                        batch    FAILED  2026-07-13T05:00:19  2026-07-13T05:00:19  00:00:00  1:0
```

---

## 3. Evidence (read-only, with file:line)

### 3.1 Stage reached — died in stage 2 (producer), validate (stage 1) passed

Wrapper `scripts/wrappers/lb_scgp_global_r2_run2_v3.sh` (`set -euo pipefail`) runs three stages in
sequence: `validate.py` (lines 60–63) → `producer.py` (lines 65–68) → `independent_verify.py`
(lines 70–73), then a `jq` PASS gate (line 75) and `COMPLETE=1` (line 76).

- **Wrapper self pre-checks PASSED:** the `RUN_ID` guard (line 43, exit 2), config `run_id` match
  (line 50, exit 2), and config `artifact_path` match (line 54, exit 2) all passed — none of their
  stderr messages appear in the `.err`; the config side is correctly v3
  (`configs/lb_scgp_global_r2/m0_synth_kkt_v3.json`: `run.run_id = LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v3`,
  `run.artifact_path = artifacts/lb_scgp_global/v3/m0/synth_kkt/manifest.json`).
- **Stage 1 (validate) passed:** under `set -e`, `producer.py` (line 65) could only be invoked if
  `validate.py` (lines 60–63) exited 0. The 0-byte `.out` is consistent, not contradictory:
  `validate.py` writes its result to the `--json-out` temp file (`VALIDATION_JSON`, line 59), not
  to stdout, and `cleanup_on_exit` (lines 23–32, `rm -f "$VALIDATION_JSON"`) removes that temp on
  exit. No validate PASS lines are expected in `.out`.
- **Stage 2 (producer) is where it died** — see 3.2.

### 3.2 Root cause — machine-plan (v2) vs code-constant (v3) run_id drift

`slurm/logs/lbscgp_global_r2_run2_v3_12974.err` (12 lines, full):
```
lines 5–6:   File ".../lb_scgp_global_r2_run2_v3_producer.py", line 271, in main
                 machine_summary = verify_machine_run2(cfg, ledger)
lines 8–9:   File ".../lb_scgp_global_r2_run2_v3_common.py", line 821, in verify_machine_run2
                 assert_equal(machine["run_order"][2], RUN2, "machine run order[2]")
line 12:     RuntimeError: machine run order[2] drift: expected
             'LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v3', got 'LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2'
```

Primary-source confirmation of the drift (read-only, this session):

- **Code-side constant** — `scripts/analysis/lb_scgp_global_r2_run2_v3_common.py:25`:
  `RUN2 = "LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v3"`.
- **Authoritative machine plan** — `cfg["paths"]["experiment_machine"]` resolves to
  `refine-logs/lb_scgp_global/EXPERIMENT_PLAN.machine.json`. In that file
  **`run_order[2] = "LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2"`** and
  **`runs[2].run_id = "LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2"`** — the plan carries **no v3 entry** for
  the synth-KKT run; it was authored for the v2 attempt and never advanced to v3.
- `verify_machine_run2` (`common.py:816–827`) reads that plan and, at line 821, asserts
  `run_order[2] == RUN2`. Plan says `v2`, code says `v3` → `RuntimeError` via `assert_equal`
  (`common.py:387`). This is the **first substantive check** in the producer's `main()`
  (`producer.py:271`), executed **before** the rank-deficient construction, so the M-B
  convergence risk (authorization §6, `common.py:643–700`) was **never touched**.

### 3.3 Fail-closed non-publish confirmed

- `artifacts/lb_scgp_global/v3/` **does not exist** (`ls` → "No such file or directory"): no
  `manifest.json`, `source_manifest.json`, `access_ledger.json`, `semantic_verification.json`, or
  publish-lock was written. The wrapper's `cleanup_on_exit` (`COMPLETE != 1` branch, lines 28–30)
  removes any prospective outputs on the non-zero exit.
- `slurm/logs/lbscgp_global_r2_run2_v3_12974.out` = **0 bytes**.
- Decision = **FAIL** (fail-closed); **no performance/scientific claim is possible or made** — a
  byte-exact clone that refused at an authoritative-input consistency check produced no counters.

---

## 4. Qualitative assessment

- **Failure class:** statically foreseeable **infrastructure / document-code consistency** failure
  — the code lineage advanced v2 → v3 (`common.py:25` `RUN2`, config, wrapper, sbatch, artifact
  paths all v3) but the authoritative experiment-machine plan JSON was **not** updated from its v2
  synth-KKT entry. A static cross-check of `EXPERIMENT_PLAN.machine.json` `run_order`/`runs[2]`
  against the code-side `RUN2` constant would have caught this before submission.
- **NOT M-B:** distinct from the accepted M-B rank-construction convergence risk (authorization
  §6). Execution never reached the numpy rank construction; this is a plain equality assert on a
  plan document, not a numerical non-convergence.
- **Fail-closed behaved correctly:** no false PASS, no partial artifact, clean non-publish. The
  single authorized attempt was spent on this fail-closed refusal (zero science), which is the
  quota-burn outcome the authorization released with knowledge of — though triggered by this more
  basic drift rather than M-B.
- **Next step (executor-neutral):** per authorization §8.6, route to a **fresh result-to-claim
  review** to adjudicate the failure and decide a v4, with this record + the `.err` as evidence.
  The natural v4 fix is a data-only update of `EXPERIMENT_PLAN.machine.json` `run_order[2]`/
  `runs[2]` to the v3 run_id (with its dependency set `[RUN1, RUN2_V1]` per `common.py:826`) — but
  that is a setup/review decision, **outside the executor role**, and is not performed here.

---

## Required statements

- No performance evidence exists and no performance claim is made or possible: the job refused at an
  authoritative-input consistency check and produced no artifact and no counters.
- The only project gold is `parent_video_binary_label`. No segment/frame/timestamp/span/
  localization/stance/target/mechanism/rationale/fragment gold is assumed or introduced; v3 produced
  no artifact.
- Run3, M1, MLLM/cache, validation/test, training, and realbank remain **locked**. This record
  covers only the one authorized CPU-only synthetic-certificate self-test SLURM job (job 12974).
- **The v3 lineage single-submit budget is spent and will not be resubmitted** regardless of
  outcome. A v4 (if any) is a separate setup/review/authorization decision.
- Executor = Claude Opus 4.8, executor role, separate from the v3-setup/freeze, static-code-review,
  and execution-authorization roles.
