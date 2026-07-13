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

## Verdict (TERMINAL, updated 2026-07-13 ~11:30)

**COMPLETED — job 12978, ExitCode `0:0`, `decision == "PASS"`.** The held queue auto-released at
`2026-07-13T11:28:35` (after ~4h34m held) and the job ran 6 seconds
(`Start=11:28:35 End=11:28:41 Elapsed=00:00:06`). The producer published the v4 certificate and the
independent semantic verifier returned **PASS**; the M-B movement-window / rank-KKT construction
converged (did **not** fail-closed). The single-submit budget is **spent** — **no resubmission**.
Full terminal evidence is in §7bis below. (The pre-terminal snapshot that follows in §§1–6 is
retained as the submission-time record.)

---

## 0. Terminal result — §7bis (decision = PASS)

See the detailed terminal-evidence section **§7bis** at the end of this document (added on the
COMPLETED notification). Headline: `semantic_verification.json.decision = "PASS"`, KKT stationarity
normalized residual `0.0` / z*-Gram residual `1.07e-15`, 21/21 adversarial injections REJECT, all
access-ledger counters `0`, and M1/M2/M3 Medium findings closed.

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

---

## 7bis. TERMINAL EVIDENCE (COMPLETED, decision = PASS) — added on the 11:28 terminal notification

**Note on §§1–6 above:** those sections are the *submission-time* snapshot (07:56, still held). They
are left intact for the record. This §7bis supersedes their "not yet terminal" statements: the job
subsequently auto-released and completed PASS.

### 7bis.1 SLURM terminal state (`sacct -j 12978`)

| field | value |
|---|---|
| State | **COMPLETED** |
| ExitCode | **0:0** |
| Submit / Start / End | `2026-07-13T06:54:33` / `2026-07-13T11:28:35` / `2026-07-13T11:28:41` |
| Elapsed | `00:00:06` (6 s) |
| held duration | ~4h34m `PENDING (JobHeldUser)` then auto-released (never force-released) |

`12978.batch` also `COMPLETED 0:0`. The background poller (`v4_12978_poll.log`) recorded held at
iters 1–4 (08:11–08:56); release happened during the 15-min gap and was caught by the team-lead's
sacct check → poller stopped, redundant.

### 7bis.2 Console logs — fail-closed-silent (expected)

- `slurm/logs/lbscgp_global_r2_run2_v4_12978.out` = **0 bytes** (empty).
- `...12978.err` = 544 bytes, **only** the known-benign
  `independent_verify.py:167 DeprecationWarning: jsonschema.RefResolver is deprecated` (lines 1–2).
  **No traceback, no error.** This is the L-C/forward-risk warning the authorization §4 flagged as
  fail-closed-safe; it did not raise. The DeprecationWarning line is itself positive evidence that
  `independent_verify.py` executed.

The three pipeline scripts print nothing to stdout by design (raise-or-succeed); all step evidence is
in the four JSON artifacts, not the console. `validate.py`'s 16-item output goes to a `mktemp`
`VALIDATION_JSON` that the wrapper's `cleanup_on_exit` trap deletes (wrapper lines 26, 59–63), so the
per-item validate detail is not persisted — but `set -euo pipefail` (wrapper line 2) guarantees
validate + producer both exited 0 (else the run would have aborted before `independent_verify`).

### 7bis.3 Wrapper gate — artifacts persist ⟺ decision==PASS

`scripts/wrappers/lb_scgp_global_r2_run2_v4.sh` runs, in order (lines 60→73):
`validate.py` → `producer.py` → `independent_verify.py`, then `jq -e '.decision=="PASS"'` (line 75)
→ `COMPLETE=1` (line 76). The `cleanup_on_exit` trap **removes all prospective outputs unless
`COMPLETE==1`** (lines 28–30). Therefore the presence of all four persisted artifacts **plus** exit 0
is a structural proof that `decision=="PASS"`.

### 7bis.4 Semantic verifier decision (`semantic_verification.json`)

- **`decision` = `"PASS"`** ; `acceptance_path` = `"serialized_h_metric_normal_cone_kkt"` ;
  `finite_vi_acceptance` = `false` (the finite-VI path correctly cannot accept; the serialized
  H-metric normal-cone KKT path is the accepting one).
- `medium_findings_closed`: `M1_strict_schema_semantic_verifier=true`,
  `M2_dirty_binding_run1_run2_relevant_tree=true`,
  `M3_orth_cap_and_M_Q_executed_with_rank_cap_cases=true` — **all three closed.** (M2 = the
  `verify_machine_run2` dirty-binding run1/run2 relevant-tree check the authorization flagged as
  "this time should pass" — it passed.)
- `metrics`: `stationarity_normalized_residual = 1.756e-16`, `zstar_gram_residual = 1.069e-15`,
  `movement_fro = 0.03638`, `movement_offdiag_max = 0.008683`, `objective_value = 0.05919`,
  `rank_eps = 3`, `structural_dual_l2 = 0.34214`.
- `injection_results`: **21/21 adversarial injections `REJECT`** (schema-tamper, forbidden-path,
  wrong-dual-sign, malformed stationarity/complementarity/dual, no-movement claim, finite-VI-only
  acceptance, rank-failure-not-null, operator-hash perturbation, NaN/overflow, …) — the fraud-
  resistance battery is fully effective.
- Chain integrity: `semantic_verification.manifest_payload_sha256` ==
  `manifest.payload_sha256` = `81aae9837a5baa3061308606365bfbae1305b982e54f07474032047e3020acfc`
  (the verifier judged the exact produced manifest).

### 7bis.4a Manifest KKT numeric block (complementarity / dual / duality-gap / primal residuals)

- `complementarity`: `status=PASS`, `max_abs=0.0`; per-family all `0.0`
  (`box_coordinate`, `soc`, `psd`, `halfspace`, `structural_band`).
- `dual_feasibility`: `status=PASS`, `linear_multiplier_min=0.0`, `psd_dual_lambda_min=0.0`,
  `soc_cone_residual_max=0.0`, `affine_unrestricted=true`.
- `duality_gap`: `dual_objective_materialized=false`, `gap_pass_claimed=false`,
  note = "No dual objective pass is claimed; stationarity plus valid normal decomposition is the
  acceptance path." **The certificate deliberately does not claim a duality-gap pass** — acceptance
  rests on stationarity (residual 0.0) + valid normal-cone decomposition, an honest scope.
- `primal`: `objective_value=0.0591923`; `residual_summaries` all `0.0` (`symmetry_fro`,
  `unit_diag_inf`, `psd_min_violation`, `offdiag_box_violation`, `coordinate/row/class_trust_violation`,
  `structural_equality_l2`, `structural_band_violation`); `G0` and `G_star` are the 10×10 unit-diag
  Gram matrices (`G_star ≠ G0`, the movement of §7bis.5).
- `injection_results_expected` (manifest): **21 cases, every one `REJECT`** — exactly matching the
  21/21 actual `REJECT` in `semantic_verification.injection_results` (expected == observed, the
  fraud-battery contract is met case-for-case).

### 7bis.5 M-B convergence — PASSED SMOOTHLY (the key untested runtime risk)

The authorization §7 flagged **M-B** (`rank_deficient_structural_solution`, the 30-step geometric
shrink `scale*=0.7` seeking the movement/`r_abs_max` window, `v4_common.py:643-700`) as the first
genuinely-untested assert v4 would reach; a window-infeasibility there would fail-closed (no publish).
**It did not fail.** `manifest.movement_metrics`:
`fro_norm_G_star_minus_G0 = 0.036379`, `max_abs_offdiag_change = 0.008683`,
`positive_threshold = 0.005` → `max_abs_offdiag_change (0.008683) ≥ positive_threshold (0.005)`, i.e.
the shrink found a feasible movement window on the one authorized run. `manifest.stationarity`:
`status=PASS`, `normalized_residual=0.0`, `residual_norm=0.0`. `metric.H_positive_definite=true`,
`lambda_struct=1.0`. The accepted G* is rank-realizable at machine precision
(`zstar_gram_residual=1.069e-15`). **No `rank_deficient_structural_solution` raise; the consciously-
accepted quota-burn risk did not materialize.**

`manifest.rank_failure_probe` is a **deliberate negative control** (`case_id
RANK_FAILURE_RETURNS_NULL`, `expected_status=ENCODER_RANK_GATE_FAIL`, got
`status=ENCODER_RANK_GATE_FAIL`, `factor_returned_null=true`, `reconstruction_residual=1e99`) — it
confirms the rank gate correctly returns null (fails closed, no truncation rescue) on a genuinely
rank-deficient input. This is the gate working on a bad fixture, **not** the main construction
failing.

### 7bis.6 Boundary / gold-isolation / no-claim

- `manifest.no_success_claim = true`; `authorized_boundary`: `run3_or_later_locked=true`,
  `synthetic_only=true`, `run_id="LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v4"`.
- `gold_isolation`: `only_gold_supervision="parent_video_binary_label"`, `segment_gold_exists=false`,
  `segment_gold_used=false`.
- `access_ledger.zero_counters` and `gold_isolation.zero_counters`: **every counter = 0** —
  `gpu_device_count=0`, `network_call_count=0`, `mllm_call_count=0`, `model_call_count=0`,
  `training_call_count=0`, `performance_evaluation_count=0`, all `test_*`/`validation_*`/`held_*`
  label & content read counts `0`, `forbidden_path_read_count=0`, `run3_or_later_attempt_count=0`.
  Confirms the CPU-only synthetic self-test touched no validation/test/gold/cache/MLLM/network/GPU.
- `case_matrix.status = "PASS"` over 6 synthetic cases (`FULL_SYNTH_KKT`, `REMOVE_NULL_PARITY`,
  `SHUFFLE_SYNTH_CONTROL`, `NOISE_SYNTH_CONTROL`, `AMBIGUOUS_COVERAGE_LOW`,
  `ROBUST_COVERAGE_REPORTED`). `schema_fixture_results.unresolved_values.schema_status = "PASS"`,
  3 invalid-schema fixtures all `REJECT`.
- `manifest.terminal_state = "PRODUCED_PENDING_INDEPENDENT_VERIFY"` is the **producer's** manifest
  state; the independent verifier then ran and issued `decision=PASS` in `semantic_verification.json`.

### 7bis.7 M-A residual (unchanged, non-blocking)

The M-A conditional-High (synthetic `G0` not verified PSD / rank-`≤d` realizable, authorization §7)
is **not** dissolved by this run — the verifier still does not check the *input* `G0`'s PSD-ness (the
`zstar_gram_residual` proves the *accepted solution* is realizable, a different object). M-A stays
Medium, non-blocking; it must be resolved (ruling/amendment) before any *scientific* claim rests on
the synthetic `G0` fixture. `no_success_claim=true`, so no such claim exists here.

### 7bis.8 Artifacts produced (all under `artifacts/lb_scgp_global/v4/m0/synth_kkt/`)

`manifest.json` (+`.publish.lock`), `source_manifest.json` (+lock), `access_ledger.json` (+lock),
`semantic_verification.json` (+lock). `payload_sha256 = 81aae983…3020acfc`;
`access_ledger_sha256 = 3cf91480…41a12ddd`; `relevant_tree_sha256 = 63d0623a…78ba2f0`.

### 7bis.9 Ledger + next step (terminal)

- Single-submit budget: **spent** (1/1). **No resubmission** — respected.
- Routing per authorization §9 item 6: PASS → **fresh independent artifact review** (separate role);
  do not self-certify. This executor does not adjudicate the artifact — it only records the terminal
  fact and hands off. (This §7bis is an evidence transcription, not a certification.)
