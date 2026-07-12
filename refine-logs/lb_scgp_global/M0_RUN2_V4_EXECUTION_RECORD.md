# M0 Run2-v4 Independent Execution Record

Date: 2026-07-13

## Role declaration

Author = **Claude Opus 4.8**, fresh independent **executor** for the `lb_scgp_global_r2` M0 Run2
**v4** lineage. This role is deliberately **separate** from the v4-prep / clone-freeze role
(`M0_RUN2_V4_CLONE_FREEZE.md`), the merged amendment-ratification + code-review role
(`M0_RUN2_V4_CODE_REVIEW.md`), and the execution-authorization role
(`M0_RUN2_V4_EXECUTION_AUTHORIZATION.md`). I did not author, review, ratify, or authorize any v4
entity. I performed only the executor obligations: verify preconditions from files + cluster state,
submit **exactly once**, monitor to terminal state, collect evidence, and write this record.

**Model-binding divergence declaration** (precedent: v3 execution record, v4 authorization §Authorizer
boundary): `AGENTS.md` binds the main-dialogue subagent to "GPT-5.5 xhigh"; that backend is
unavailable, so this execution runs on the `CLAUDE.md`-bound **Opus 4.8** (`claude-opus-4-8`).
Documented process fact, not a defect.

**Write scope:** this file only. I made no edit to any plan / machine.json / `_HASHES.sha256` /
config / schema / script / wrapper / sbatch / authorization / review / freeze document. The one
active mutation was the single authorized `sbatch` submission.

---

## Verdict (as of 07:56, 2026-07-13)

**SUBMITTED — job 12978. Non-terminal: `PENDING (JobHeldUser)` for 61+ minutes, awaiting SLURM
auto-release.** No terminal decision (PASS / FAIL) exists yet; no v4 artifact and no `.out`/`.err`
have been produced. Per the authorization's single-submit budget, the one authorized submission is
**spent** and there will be **no resubmission regardless of outcome**.

---

## 1. Preconditions re-verified (all PASS) — submission gate

Re-verified directly from files + live cluster state this session (not transcribed from the
authorization doc):

| # | Check | Result |
|---|---|---|
| 1 | Authorization present & AUTHORIZED | PASS — `M0_RUN2_V4_EXECUTION_AUTHORIZATION.md:35` "AUTHORIZED — exactly one CPU-only SLURM submission." |
| 2a | Config hash == post-binding target | PASS — `sha256sum configs/lb_scgp_global_r2/m0_synth_kkt_v4.json` = `e67fc6f79073b8a425ce24834db966bd5b61a2282e2247a5ef87e396056844d1` (equals authorization §3 executor target; **not** pre-binding `118afadf…`) |
| 2b | Wrapper hash == CLONE_FREEZE §1 | PASS — `scripts/wrappers/lb_scgp_global_r2_run2_v4.sh` = `0ad33ba4c3e43e52800d5d1a79316e0b1ebb84d9fc65aac4a4de087e1c65d161`, matches `M0_RUN2_V4_CLONE_FREEZE.md:49` |
| 2c | Sbatch hash == CLONE_FREEZE §1 | PASS — `scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v4.sbatch` = `8e1359ac259fd9e54181d94f77e7e29c55d8c1b44c012b1081ecffc792145427`, matches `M0_RUN2_V4_CLONE_FREEZE.md:50` |
| 3a | No same-named job in squeue | PASS — `squeue -u jehc223` showed only `12976 tarc_g2pred`; no `lbscgp_global_r2_run2_v4` present pre-submit |
| 3b | `artifacts/lb_scgp_global/v4/` absent | PASS — `ls` → "No such file or directory" |
| 4 | Resource coexistence | PASS — only non-`label_*` job was `12976 tarc_g2pred` (8 CPU / 64 GB, PENDING). v4 (8 CPU / 64 GB / 0 GPU) + tarc = exactly 16 CPU / 128 GB = per-user cap; no other jobs consuming resources → normal submit |

I did **not** edit `M0_RUN2_V4_CODE_REVIEW.md` (hash-frozen at `41650dce…`; any byte change would
fail the runtime replay).

## 2. Submission

- Command (executed **exactly once**): `sbatch scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v4.sbatch`
- Returned: **`Submitted batch job 12978`**
- `scontrol show job 12978`: `SubmitTime=2026-07-13T06:54:33`,
  `StdOut=/data/jehc223/RGCL/slurm/logs/lbscgp_global_r2_run2_v4_12978.out`,
  `StdErr=/data/jehc223/RGCL/slurm/logs/lbscgp_global_r2_run2_v4_12978.err`.

## 3. Monitoring timeline (sacct + squeue polling, 30 s cadence)

Job stayed `PENDING`, reason `(JobHeldUser)`, `Elapsed=00:00:00` throughout (never started):

| wall-clock | elapsed since submit | state / reason |
|---|---|---|
| 06:54:44 (first poll) | ~00:00 | PENDING (JobHeldUser) |
| 07:05:43 | ~11 min | PENDING (JobHeldUser) |
| 07:15:00 | ~20 min | PENDING (JobHeldUser); 12976 also PENDING (JobHeldUser) |
| 07:24:15 | ~30 min | PENDING (JobHeldUser) |
| 07:33:25 | ~39 min | PENDING (JobHeldUser) |
| 07:42:35 | ~48 min | PENDING (JobHeldUser) |
| 07:52:46 | ~58 min | PENDING (JobHeldUser) |
| 07:55:49 | ~61 min | PENDING (JobHeldUser) |

- `scontrol show job 12978` at 07:55: `JobState=PENDING Reason=JobHeldUser Dependency=(null)`,
  **`EligibleTime=Unknown`** — the held job has not yet been released by the scheduler.
- Companion job `12976 tarc_g2pred` (different GPU lineage, 1×A100) remained `PENDING (JobHeldUser)`
  the whole time; both v4 (CPU) and tarc (GPU) are held together, consistent with a user-level held
  queue awaiting the periodic auto-release, not a resource-contention block (202 idle CPUs on the
  node per authorization §6).
- I did **not** force-release (`scontrol release`) and did **not** cancel — per the `CLAUDE.md`
  policy "等自动放行即可,不要强行释放" and the authorization §9 items 4–5.

## 4. Evidence state at record time

- `slurm/logs/lbscgp_global_r2_run2_v4_12978.{out,err}` — **not yet created** (SLURM writes them only
  once the job starts; `ls` → "No such file or directory").
- `artifacts/lb_scgp_global/v4/` — **still absent** (`ls` → "No such file or directory"); no manifest,
  no counters, no `decision` field.
- `sacct -j 12978`: `PENDING 0:0 Submit=2026-07-13T06:54:33 Elapsed=00:00:00`.

No producer nine-step evidence (`verify_machine_run2`, rank / KKT construction, `independent_verify`,
`decision`), no validate-16, and no `decision` PASS/FAIL exist yet — the job has not executed.

## 5. Single-submit ledger — SPENT

- Prior v4 submissions before this session: **0** (authorization §5, re-confirmed).
- This session: **1** submission (job 12978).
- Authorized submissions remaining: **0**.
- **绝不重提 / no resubmission:** regardless of the eventual terminal outcome — including a fail-closed
  M-B non-convergence (`rank_deficient_structural_solution`), which the coordination session
  consciously accepted (authorization §7) — this lineage will **not** be resubmitted. A failure feeds
  a separate v5 decision, not a v4 retry.

## 6. Next step (deferred — job not yet terminal)

The job is still held-pending past the 60-minute reporting threshold, so terminal routing cannot be
executed yet. When 12978 reaches a terminal state:
- **On PASS** (producer publishes v4 manifest + counters): route to a **fresh independent artifact
  review** (separate role) — do not self-certify.
- **On FAILED** (job error or fail-closed non-publish): route to a **fresh result-to-claim** review
  with the job log as evidence. An M-B convergence-window death (past preflight) is the known,
  accepted class and does **not** trigger the preflight-class full-chain-audit escalation
  (authorization §7).

## Required statements

- No performance evidence exists and no performance claim is made — the v4 job has produced no
  artifact and no counters (it never started).
- The only project gold is `parent_video_binary_label`; the v4 fixtures are synthetic; no
  segment/frame/timestamp/localization gold is assumed.
- Run3, M1, MLLM/cache, validation/test, training, and realbank remain **locked**.
- Executor = Claude Opus 4.8, fresh, execution role only, separate from v4-prep/freeze,
  amendment/code-review, and execution-authorization roles. Write scope = this file only; the only
  mutation was the single authorized `sbatch`.
