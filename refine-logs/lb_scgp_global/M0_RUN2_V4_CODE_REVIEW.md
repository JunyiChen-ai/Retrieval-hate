# M0 Run2-v4 Merged Static Review (Amendment-Ratification + Fresh Code Review)

Date: 2026-07-13

Reviewer: **Claude Opus 4.8**, fresh, zero-context, zero-history (0C/0H) independent reviewer for
the `lb_scgp_global_r2` M0 Run2 **v4** lineage. This document performs a **merged** review as
directed: (i) ratification of the v4 plan-amendment (`M0_RUN2_V3_RESULT_TO_CLAIM_REVIEW.md`
§5(a), including the mandatory REPLACE-semantics ruling) and (ii) the fresh 0C/0H static code
review with the mandatory runtime cross-check static-simulation table (§5(c)) and the
dependency-availability item (§5(d)).

## Reviewer boundary

Read-only static review only. I did **not** participate in any prior round (v1/v2/v3, the v4-prep
amendment authoring, or the v4 clone/freeze); every ruling below is grounded in files I read and
commands I ran **in this session**, not in any prior-round conclusion. I explicitly **re-ran**
the clone-equivalence diff, the 13-row runtime simulation, and the old-protected manifest
reconstruction myself rather than copying the freeze document's tables. I did **not** run project
Python, imports, `py_compile`, tests, `conda`, SLURM, `sbatch`, `squeue`, experiments,
MLLM/OCR/API/model/network/GPU/training/evaluation, or validation/test data/cache inspection.
Shell was limited to the allowed static tools: `rg`/`grep`, `sed`/`nl`, `jq`, `awk`, `bash -n`,
`diff`, `sha256sum`, `find`, `ls`, `wc`, `git status`, `git diff`. The only file I wrote is this
report. No artifact under `artifacts/lb_scgp_global/v4` was created (its absence is confirmed
below). This review authorizes **no** SLURM execution.

### Model-binding divergence declaration (precedent: `M0_RUN2_V2_CODE_REVIEW_FIX2.md`, `M0_RUN2_V3_CODE_REVIEW.md`)

`AGENTS.md:15` binds the main-dialogue subagent to **"GPT-5.5 xhigh"**. That backend is not
available for this session's subagent, so this review runs on the `CLAUDE.md`-bound **Opus 4.8**
(`claude-opus-4-8`). This is a documented `AGENTS.md`↔`CLAUDE.md` divergence, recorded for
transparency; it is a process/documentation fact, not a code defect, and does not affect any
ruling below.

---

## Verdict

**PASS_STATIC_REVIEW**  — and the v4 plan-amendment is **RATIFIED** (REPLACE-at-index-`[2]`
semantics affirmed as the correct byte-clone analogue of the v2 INSERT).

| Severity | Count | Items |
|---|---:|---|
| Critical | 0 | — |
| High | 0 | — |
| Medium | 3 | M-A (carried), M-B (carried; now the next-reachable untested runtime risk), **M-C-doc (new)** |
| Low | 3 | L-A (carried), L-B (carried), L-C (carried) |

Critical = 0 and High = 0 → the static-review pass criterion is met, and (per the task's own
gate) authorization may proceed to the next ceremony boundary. All 13 §5(c) simulation rows
independently evaluate **PASS** (§4). Medium/Low items do not block but must be understood before
execution authorization (§8).

---

## 1. Amendment ratification — REPLACE-at-index-`[2]` and the declared change list

### 1.1 REPLACE (not INSERT) is the only byte-clone-compatible placement — RATIFIED

I read `verify_machine_run2` directly (`…v4_common.py:816-827`). It hardcodes the numeric index
literal `[2]`:

```
820  run = machine["runs"][2]
821  assert_equal(machine["run_order"][2], RUN2, "machine run order[2]")
822  assert_equal(run["run_id"], RUN2, ...)
823  assert_equal(run["artifact_paths"], [cfg["run"]["artifact_path"]], ...)
824  assert_equal(run["artifact_schema_ids"], [PAYLOAD_SCHEMA_ID], ...)
825  assert_equal(run["slurm"], cfg["run"]["slurm"], ...)
826  assert_equal(run["dependencies"], [RUN1, RUN2_V1], ...)
```

with code constants (`…v4_common.py:23-28`) `RUN2 = "…SYNTH-KKT-v4"`,
`PAYLOAD_SCHEMA_ID = "…payload_v4"`, `RUN1 = "…CONTRACT-FREEZE-v1"`, `RUN2_V1 = "…SYNTH-KKT-v1"`,
`CERT_SCHEMA_ID = "…cert_v2"`. Because v4 is a **byte-clone** of v3, the index literal `[2]`
cannot move (a token-sed cannot rewrite a numeric literal). Therefore the amendment must make
**index `[2]` itself become v4**. An INSERT (the v2 precedent, valid there because v2 was a code
*rewrite* that moved its own reader to `[2]`) would push v2 to `[3]` and require the code to read
`[3]` — impossible for a byte-clone whose reader is pinned to `[2]`. **REPLACE-in-place at `[2]`
is the unique placement under which byte-clone code passes.** Confirmed on disk: `run_order` and
`runs` arrays are **length 66 in both** the pre-amendment backup and the current plan (no
insertion), realbank stays at index `[3]` (`REALBANK-RESOURCE-v1`, now `dependencies=[…-v4]`), and
the `[3..65]` tail is unshifted. **REPLACE semantics ratified.**

### 1.2 Actual-change audit — `.bak` vs current plan, every hunk mapped to a declared item

I diffed `EXPERIMENT_PLAN.machine.json.pre_v4_amendment.bak` (sha256 `6caa5c2e…0492`, re-hashed
here) against the current `EXPERIMENT_PLAN.machine.json` (sha256 `42bf49ed…4590a90` = M′,
re-hashed here) and mapped **every** hunk to the amendment's declared Group A/B/C list. There are
**zero undeclared changes**. On-disk values (via `jq`):

**Group A — code-validated at `[2]` (each flipped v2→v4; hard runtime assert):**

| # | path | on-disk value | assert | verdict |
|---|---|---|---|---|
| A1 | `run_order[2]` | `…SYNTH-KKT-v4` | `common:821` | ✓ |
| A2 | `runs[2].run_id` | `…SYNTH-KKT-v4` | `common:822` | ✓ |
| A3 | `runs[2].artifact_paths[0]` | `artifacts/lb_scgp_global/v4/m0/synth_kkt/manifest.json` | `common:823` | ✓ (== `cfg.run.artifact_path`) |
| A4 | `runs[2].artifact_schema_ids[0]` | `scgp_global_synth_kkt_payload_v4` | `common:824` | ✓ (== `PAYLOAD_SCHEMA_ID`) |

**Group A′ — code-validated, MUST NOT change (confirmed unchanged on disk):**

- `runs[2].slurm` = `{cpu:8, ram_gb:64, gpu:0, env:"HateVideo", no_time_flag:true}` — not in the
  diff; `jq` deep-equal vs `cfg.run.slurm` returns **true** (`common:825`). ✓
- `runs[2].dependencies` = `["…CONTRACT-FREEZE-v1", "…SYNTH-KKT-v1"]` = `[RUN1, RUN2_V1]` — not in
  the diff (`common:826`). v4 correctly depends on synth-kkt-**v1**, not v2/v3. ✓

**Group B — plan-coherence (not read by `verify_machine_run2`):** all present and confined to the
declared fields — `runs[2].purpose`, `planned_config_path` (→ `…v4.json`), `artifact_namespace`
(→ `…/v4/…`), `lineage_bound_outputs[0..7]` (`v2_*`→`v4_*`),
`science_freeze_assertions.solver_math_verifier_logic_change_scope`,
`authorization_requirements` (two `…_v2_…` keys renamed `…_v4_…` + two new §5(c)/(d) keys),
`implementation_fix_freeze`→`clone_freeze` (anti-fabrication, §2.2 below), `gate`,
`failure_transition`, `status` (→ `V4_CLONE_FROZEN_AMENDMENT_PENDING_INDEPENDENT_REVIEW`),
`runs[3].dependencies[0]` (realbank → `…-v4`), `runs[3].gate`, `runs[3].status`
(`LOCKED_UNTIL_V2_PASS`→`…_V4_PASS`), `dependency_dag.pre_post_run_order_diff` (added
`inserted_run_id_note`, `run3_new_dependency`→`…-v4`, added `v4_replace_record` block),
`dependency_dag.terminal_decision_chain[2]` (→`…-v4`),
`gates_and_terminal_statuses.v2_authorization_boundary`→`v4_authorization_boundary` (§2.2),
`readiness` flags + `first_three_g0_runs[2]`(→v4) + `g0_runs_through_v2_supplement` renamed
`…through_v4_…` with `[2]`→v4 + review-status set replaced by
`carried_static_residuals_pending_fresh_v4_review=[M-A,M-B]`, and the `supplement_amendment` block
full v4 rewrite. Every one appears in the amendment's declared list.

**Group C — registry coherence:** `artifact_schemas` gained `scgp_global_synth_kkt_payload_v4`
(same 12-key list as `…_v1`/`…_v2`, both retained). Coherence-only; no run2 module reads
`artifact_schemas` (§2.3). ✓

**Undeclared-change count: 0.** No behavioral/structural key outside the declared list changed;
the index literals `machine["runs"][2]`/`run_order[2]` are untouched in the code.

---

## 2. The five v4-prep pending rulings (待裁点) — adjudicated

**① REPLACE-semantics ratify — YES.** See §1.1. Unique byte-clone-compatible placement; array
length and downstream indices verified unchanged on disk. Ratified.

**② Anti-fabrication rewrite (`implementation_fix_freeze`→`clone_freeze`;
`v2_authorization_boundary`→`v4_…`; readiness flags) — FAITHFUL, not overreach.** Decisive
criterion (per the task): do runtime asserts **read** these fields? I read every machine-plan
reader (§2.3): only `verify_machine_run2` reads plan **content**, and it reads **only**
`runs[2].{run_id, artifact_paths, artifact_schema_ids, slurm, dependencies}` and `run_order[2]`.
None of the rewritten fields (`clone_freeze`, `authorization_requirements`, `gate`, `status`,
`gates_and_terminal_statuses.*`, `readiness.*`, `supplement_amendment`) are at a code-read site,
so the rewrite carries **zero runtime risk** and **high truth-value benefit**: leaving v2's
`implementation_fix_freeze` (`fresh_fix_review_failed_zero_critical_two_high_…`,
`fix2_freeze_complete`, `amendment_review_passed_zero_critical_zero_high=true`) on the **index-2
v4 run** would assert a **false** already-reviewed/completed state for v4. I verified the
replacements are **truthful for the current pre-review state**: `clone_freeze` has
`v4_amendment_independent_review_passed=false`, `fresh_independent_v4_code_review_passed=false`,
`runtime_crosscheck_static_simulation_table_all_pass_confirmed_by_reviewer=false`,
`carried_static_residuals=[M-A,M-B]`; `v4_authorization_boundary` has
`amendment_review_passed_zero_critical_zero_high=false`, `clone_freeze_complete=true`. This is
exactly aligned with verdict §5(a) ("prior-version evidence preserved in refine-logs, **not** as
run-array entries"). Ratified.

**③ Nine preserved v2/fix2 tokens — acceptable; code reads none of them.** I confirmed (§2.3)
that the plan is read only by `verify_machine_run2`, which touches only the `runs[2]` subset +
`run_order[2]`. The preserved tokens — `schema_version` (plan-schema version), `runs[4]/runs[5]`
`scgp_global_cache_replica_v2` (M1 cache schema, different lineage), the retained
`artifact_schemas.…payload_v2` **key**, `dependency_dag.pre_post_run_order_diff.inserted_run_id`
and `v4_replace_record.replaced_run_id` (genuine v2 INSERT/REPLACE history),
`dependency_dag.locks.*_v2_*` **keys**, `budget_ranges.*` convention/baseline/status-count
**keys**, and the isolation counter **key** — are each genuine history, an unrelated schema name,
or policy/budget vocabulary whose numeric value is still correct, and **none is at a code-read
site**. Rewriting them would risk corrupting genuine history (the v1→v2 INSERT record) or an
unrelated M1 schema. Preservation ratified; no extension of the amendment is required.

**④ Review-doc binding deferral + two-phase freeze — SOUND; not a Critical trap.** The concern is
that the execution authorizer must later add the amendment-**review** doc bindings and re-freeze
the config, changing the config's own hash away from the clone-freeze §1 target `118afadf…3bf0f`.
I verified this is safe: (a) the v4 config is **not** self-bound in `authoritative_inputs` (grep:
`m0_synth_kkt_v4.json` absent from the 27 keys); (b) the config's own hash is **not** pinned by
any runtime assert — `…v4_independent_verify.py:1016` compares
`manifest["hashes"]["config_sha256"]` to a **runtime-recomputed** `sha256_file(config)`, i.e. a
producer↔verifier internal-consistency check on the same file, not against a frozen constant.
Therefore re-freezing the config (adding review-doc bindings) does not break any assert. Crucially
there is **no plan-drift window**: the authoritative plan is frozen now at M′ and the config binds
M′; the authorizer adds **only** review-doc bindings and must **not** re-edit the plan, so M′, H′,
and the three v4-amendment-doc bindings stay valid. This is documented in clone-freeze §4 and
matches the v2 precedent (config re-frozen after review). **Authorizer obligations** (recorded for
§8): add only the review-doc bindings + re-freeze the config; do **not** re-edit the plan or the
existing amendment docs; **re-run** the full `authoritative_inputs` replay (now including the
review docs) against on-disk after re-freeze; update the config's nine-entity freeze hash (it will
move off `118afadf…`).

**⑤ Human-doc append-only — SUFFICIENT.** `git diff` shows `EXPERIMENT_PLAN.md` (+4 lines) and
`EXPERIMENT_TRACKER.md` (+7 lines) are **pure insertions (11 insertions, 0 deletions)** — each a
new "## Run2-v4 Amendment Note (2026-07-13)" section that accurately states v2/v3 closure, the
REPLACE-at-`[2]` semantics, realbank staying at `[3]` repointed to v4, evidence preservation, and
the remaining locks. Prior v1/v2/v3 records are untouched. Neither human doc is parsed by any
runtime assert (validate `jq -e .`s only the config, the `.machine.json`, the three schemas, and
the run1 contract-freeze JSON — not the markdown); they are hash-bound only. Append-only is the
correct history-preserving choice and is sufficient.

### 2.3 Machine-plan reader enumeration (supports ②/③)

`grep` for `experiment_machine`/`machine[`/`verify_machine_run2` across the four v4 modules: the
plan **content** is read at exactly one site, `…v4_common.py:819` (`machine = read_json(...)`)
inside `verify_machine_run2`, which reads only `machine["runs"][2]` (a 5-field subset) and
`machine["run_order"][2]`. `…v4_validate.py:138` passes `cfg["paths"]["experiment_machine"]` only
as a path to a `jq -e .` well-formedness check (`:144`) — no field read. `…v4_producer.py:271`
merely calls `verify_machine_run2`. No other machine field is read anywhere.

---

## 3. Clone equivalence — independently re-proven

**Source integrity:** all nine frozen v3 source entities re-hash exactly to
`M0_RUN2_V3_CLONE_FREEZE.md` §0 (config `e6d33b5d…`, payload `1d6f93a1…`, case `df3616ff…`, common
`9de62f6d…`, validate `2e0bb00b…`, producer `6ef3a4a8…`, independent_verify `4025dbf0…`, wrapper
`8d9123e9…`, sbatch `4495ec3c…`), so the clone derives from the true frozen v3 bytes. All nine v4
entities re-hash exactly to `M0_RUN2_V4_CLONE_FREEZE.md` §1.

**Protective-transform diff (per pair):**
`diff <(sed 's/cert_v2/cert__CERTKEEP__/g; s/v3/v4/g; s/cert__CERTKEEP__/cert_v2/g' <v3>) <v4>`:

- **Eight entities EMPTY** (byte-exact clones, equal byte length): payload, case, common, validate,
  producer, independent_verify, wrapper, sbatch. In each, residual `v3`=0 and residual
  `v2`==`cert_v2` count (the only surviving `v2` is the frozen shared cert). Cert-preservation
  counts match the freeze doc (case 1, common 1, independent_verify 2, others 0).
- **Config NON-EMPTY — expected and correct.** The config is the one class-3 entity (the amendment
  edits the authoritative plan, forcing a hash-binding cascade). Its diff is confined to
  `hash_bindings.authoritative_inputs`: **4 values updated** (`EXPERIMENT_PLAN.md`,
  `…machine.json`=M′, `EXPERIMENT_PLAN_HASHES.sha256`=H′, `EXPERIMENT_TRACKER.md`) + **3 entries
  added** (`M0_RUN2_V4_PLAN_AMENDMENT.md`/`.machine.json`/`_HASHES.sha256`). I proved the deviation
  is confined by diffing the two configs with the `authoritative_inputs` block deleted after a
  cert-preserving `v3→v4` normalization — the remainder is **byte-identical (rename-equivalent)**,
  i.e. no behavioral/structural/logic change. The byte-length delta (8709→9129 = +420) is fully
  explained by the three added binding lines. Every extra config difference is a declared binding
  update (verdict §5(b) satisfied).

**Cert-schema (`scgp_global_cert_v2`) preservation** is correct: rewriting to `cert_v4` would
point at a nonexistent schema and break the frozen binding `4d3f…22f`. The file exists and its
hash matches the config `run1_frozen` binding.

---

## 4. §5(c) runtime cross-check static-simulation table — independently re-run (all 13 PASS)

Every runtime assertion reading a frozen external document, statically evaluated against the
on-disk amended state by my own re-derivation (not copied):

| Row | Assert (site) | Reads | Independent evidence | PASS? |
|---|---|---|---|---|
| 1 | `run_order[2]==RUN2` (`common:821`) | plan | `jq`: `run_order[2]`=`…-v4` == `RUN2`=`…-v4` | **PASS** |
| 2 | `runs[2].run_id==RUN2` (`common:822`) | plan | `jq`: `…-v4` == `…-v4` | **PASS** |
| 3 | `runs[2].artifact_paths==[cfg.run.artifact_path]` (`common:823`) | plan+cfg | string-equal test: both `[…/v4/…/manifest.json]` → EQUAL | **PASS** |
| 4 | `runs[2].artifact_schema_ids==[PAYLOAD_SCHEMA_ID]` (`common:824`) | plan | `jq`: `[…payload_v4]` == `[…payload_v4]` | **PASS** |
| 5 | `runs[2].slurm==cfg.run.slurm` (`common:825`) | plan+cfg | `jq` deep-equal → **true** (order-independent, as Python dict `==`) | **PASS** |
| 6 | `runs[2].dependencies==[RUN1,RUN2_V1]` (`common:826`) | plan | `jq`: `[freeze-v1, synth-kkt-v1]` | **PASS** |
| 7 | `verify_expected_hashes(authoritative_inputs)` (`common:840`→`834`) | cfg-bound hashes | **replayed 27/27 vs on-disk, fail=0** (incl. machine=M′ `42bf49ed…`, PLAN_HASHES=H′ `910f0f64…`, three v4-amendment-doc bindings) | **PASS** |
| 8 | `verify_expected_hashes(run1_frozen)` (`common:846`) | run1 frozen | **replayed 10/10 vs on-disk, fail=0** (incl. cert `4d3f…22f`); run1 untouched | **PASS** |
| 9 | `old_protected manifest/count` (`common:858-861`) | old lb_scgp tree | **independently reconstructed** the scope fn (`common:766-794`): manifest `243e89b…0462`, count **278** — **exact match** to config `old_protected_pre_snapshot`; this round's files (`refine-logs/lb_scgp_global/*`, `lb_scgp_global_r2_run2_v4_*`) are out of scope | **PASS** |
| 10 | `resource_and_run_check` (`validate:110-122`) | cfg internal | reads run_id/schema_id/artifact_path(literal `…/v4/…`)/slurm/authorized_run_ids — all v4 on disk | **PASS** |
| 11 | `authoritative_inputs unchanged` (`independent_verify:952-954`) | cfg-bound hashes | same 27 as row 7; fail=0 | **PASS** |
| 12 | `manifest.artifact_schema_id/run_id==SCHEMA_ID/RUN2` (`independent_verify:1039`) | produced manifest | producer writes `PAYLOAD_SCHEMA_ID`/`RUN2`=v4 (`producer:297-298`); verifier `SCHEMA_ID`/`RUN2`=v4 (`:24-25`) | **PASS (by construction)** |
| 13 | `manifest.authorized_boundary=={run_id:RUN2,synthetic_only:True,run3_or_later_locked:True}` (`independent_verify:1043`) | produced manifest | producer writes exactly that with `RUN2`=v4 (`producer:300-304`) | **PASS (by construction)** |

**Row 9 is stronger than the freeze doc's scope-exclusion argument:** I re-ran the actual manifest
computation (five roots, prefix rule OR `lb_scgp_`-non-`lb_scgp_global_r2_` `.py/.sbatch` name
rule, sort by relpath, `"{hash}  {rel}\n"` rows) and got `243e89b…0462` / 278 byte-for-byte — so
the frozen set has **not** drifted since Run1 for any reason. This matters because the v3 death was
at step 7 (`verify_machine_run2`), one step **before** `build_source_manifest` (step 8) where rows
7/8/9 live; with the plan-drift fixed, rows 7/8/9 are the newly-reachable asserts, and all three
provably PASS on disk.

**Load-bearing insight (verdict):** rows 7/8/11 (hash layer) verify only "the document I read is
the document I froze" and are blind to rows 1–4 (code-constant vs frozen-content). In v3 the hash
layer PASSED while rows 1–2 FAILED. Here the amendment made the plan content v4 in lock-step with
the code constants, so rows 1–4 **and** the hash layer both PASS.

---

## 5. Dependency availability (§5(d))

Full import enumeration (top-level **and** in-function) of the four v4 modules gives third-party
set = **`{numpy, jsonschema}`**, exactly as required:

- `common.py`: `numpy` (L19) + **deferred** `from jsonschema import …` (**L182-183**),
  `try/except → RuntimeError("… refusing to validate Run2-v4 payload")` (fail-closed, correctly
  v4-labelled).
- `validate.py`: no third-party top-level; `jsonschema` presence is a `find_spec` subprocess
  preflight (L101-106), fail-closed.
- `producer.py`: `numpy` (L16); local `…_v4_common`.
- `independent_verify.py`: `numpy` (L20) + **deferred** `from jsonschema import …` (**L167-168**),
  `→ RuntimeError("… independent verifier refuses PASS")`.

The deferred sites are at the **same line numbers as v3** (byte-clone leaves them untouched).
Read-only `ls` of `HateVideo` site-packages confirms `jsonschema 4.26.0` + full transitive set
(`jsonschema_specifications`, `referencing 0.37.0`, `rpds_py 2026.6.3`) and `numpy 1.26.4` are
installed in the exact interpreter tree the sbatch's `conda activate HateVideo` resolves.
**Limitation (→ L-C):** a listing proves *installed*, not that they *import* cleanly at runtime,
and the env is not frozen; any breakage is fail-closed. The execution authorizer must re-confirm
`{numpy, jsonschema}` present at authorization time (read-only or authorized SLURM-only preflight,
never a login-node interpreter run).

---

## 6. Residual re-adjudication — M-A / M-B (carried, byte-identical to v3)

Because `producer.py`/`common.py`/`independent_verify.py` are byte-identical to v3 (§3, empty
protective diffs, equal length), the underlying numeric logic is unchanged. I re-read the loci
myself:

- **M-A (Medium; conditional-High retained, untriggered).** `G0` is derived from `G_star` with the
  diagonal forced to 1 (`common:~420`); the verifier checks `G0` shape, symmetry
  (`np.allclose(g0,g0.T)`) and unit diagonal (`independent_verify:1090-1093`) but performs **no**
  `G0 ⪰ 0` / `rank(G0)≤d` check (targeted scan returns none). `FINAL_PROPOSAL.md:129` writes
  `G0=Z0 Z0^T` (a stronger property than the fixture honors) — a correctness-neutral **fidelity**
  gap (the KKT certificate is valid for any symmetric unit-diagonal anchor). Medium, non-blocking.
  Escalates to **High** only if a science authority rules `G0` must be a realizable rank-≤d PSD
  `Z0 Z0^T`; no such ruling exists in the materials I read.
- **M-B (Medium, fail-closed; the top single-submit residual).**
  `rank_deficient_structural_solution` (`common:643-700`) runs a 30-step geometric shrink
  (`scale *= 0.7`, `common:697`) seeking a `scale` satisfying
  `0.005 < movement_off_max ≤ 0.018 and movement_fro > 0.005 and r_abs_max ≤ 0.20`
  (`common:693`). Iter-0 is designed in-window but window feasibility is **not statically
  provable** for all fixtures; on failure it raises → producer refuses to publish (**fail-closed**,
  never a false PASS). It has **never executed to completion** in any lineage (v1 `KeyError`, v2
  `jsonschema`, v3 plan-drift all died before it) and is **not dissolvable by any allowed
  read-only means**. With rows 1–13 now provably PASS on disk, **M-B is the next real runtime
  risk** if v4 runs: a non-convergence burns the one authorized attempt for zero science. Medium,
  non-blocking; the authorizer must accept the convergence-burn risk consciously or obtain an
  authorized SLURM-only producer dry-run first.

Carried Lows (byte-identical wrapper/logic): **L-A** wrapper `cleanup_on_exit` footgun
(non-actionable under single-submit; v4 tree absent), **L-B** fixture assumes rather than
demonstrates a rank-≤d projection optimum (Run3 scope), **L-C** dependency-by-listing +
`RefResolver` deprecation forward-risk (fail-closed; §5).

---

## 7. M-C-doc (new, Medium) — clone-freeze §3 equivalence table misreports the config pair

`M0_RUN2_V4_CLONE_FREEZE.md` §3 lists **all nine** pairs as "blanket-sed diff **empty**" with
"byte length v3 == v4" and totals "**9 empty / all equal**", including the **config** row as
"empty / 8709 == 8709". This is **factually wrong for the config** and **self-contradicts** the
same document's §2 ("this cascade is **non-empty** and is runtime-checked") and §4 (which tabulates
the 7 config binding changes): the on-disk v4 config is **9129 bytes** (not 8709) and its
protective-transform diff is **non-empty** (the 4-updated + 3-added `authoritative_inputs`
cascade). The most likely cause is a stale copy of the v3 clone-freeze §3 table (where the config
*was* a pure clone at 8709==8709 because v3 held the plan constant with zero recompute), not
re-checked against the v4 config — the same "transcribe without re-reading the source" hazard the
project's numeric-provenance discipline warns against.

Severity **Medium (documentation-accuracy), non-blocking.** It is **not** an undeclared change
(the config cascade is fully and correctly declared in §2/§4 and in the amendment's hash-cascade
table), and it carries **no runtime or false-PASS risk** (the config's own hash is not
runtime-pinned; I independently verified the config's only deviation is the declared cascade with
**no** behavioral/structural change). But §3 is the document's byte-clone "equivalence proof," and
as written it would mislead a future auditor who trusts its "9 empty / all equal" summary about the
config. **Fix:** correct the §3 config row to show the config is the single class-3 entity
(non-empty diff confined to the declared `authoritative_inputs` cascade; 8709→9129), and the total
to "8 empty + 1 config cascade." The substance is correct; only the §3 summary needs the edit.
Because Critical=0/High=0, this does not change the PASS.

---

## 8. Notes for the execution authorizer

1. **M-B is the top residual and is not statically dissolvable** — fail-closed only (burned
   attempt), never false-PASS, convergence untested end-to-end. With the plan-drift fixed and rows
   1–13 provably PASS, M-B is the first genuinely-untested runtime assert v4 would reach. Accept the
   single-submit burn risk consciously, or run an authorized SLURM-only producer dry-run first.
2. **M-A conditional escalation is live but untriggered.** Confirm no science authority requires
   `G0 = Z0 Z0^T` (PSD rank-≤d); such a ruling makes M-A High and blocks.
3. **L-C mandatory item:** re-confirm `{numpy, jsonschema}` present in `HateVideo` at authorization
   time (read-only or authorized SLURM-only preflight — never a login-node interpreter run). Env not
   frozen; `RefResolver` deprecation forward-risk noted.
4. **Two-phase config freeze (待裁点④):** when adding the amendment-**review** doc bindings, edit
   **only** those bindings and re-freeze the config; do **not** re-edit the plan or existing
   amendment docs; **re-replay** all `authoritative_inputs` (now incl. review docs) vs on-disk
   after re-freeze; the config's nine-entity freeze hash will move off `118afadf…3bf0f` (expected,
   per clone-freeze §4 and the v2 precedent).
5. **Escalation rule still in force (verdict §6.3):** if v4 also dies at a *preflight-class* miss,
   pause the per-version ceremony and run a one-time full-chain static-execution audit before any
   v5. (M-B lives in the numeric section, past preflight — a death there is a different, more
   legitimate class, still fail-closed.)

---

## 9. Static checks performed (record)

- `sha256sum` on the plan (M′ `42bf49ed…`), the `.bak` (`6caa5c2e…`), the three amendment docs
  (`8428b7f8…`/`30221b10…`/`3bbfd910…`), `EXPERIMENT_PLAN.md` (`a98effc3…`),
  `EXPERIMENT_TRACKER.md` (`4d3c4b8c…`), `EXPERIMENT_PLAN_HASHES.sha256` (H′ `910f0f64…`), and all
  nine v4 entities (= clone-freeze §1) and all nine v3 sources (= clone-freeze §0).
- `diff` `.bak` vs current plan — every hunk mapped to a declared Group A/B/C item; **0 undeclared
  changes**. `jq` array lengths 66==66 (REPLACE, not INSERT); `runs[2]`/`runs[3]` inspected.
- Nine-pair protective-transform `diff` — 8 EMPTY, config non-empty = declared cascade only
  (proved via `authoritative_inputs`-deleted rename-normalized diff = byte-identical remainder).
- 13-row §5(c) table independently re-derived: rows 1–6/10 via `jq` on-disk; rows 7/8/11 via
  27+10-key hash replay (fail=0); row 9 via independent reconstruction of `old_protected_hash_manifest`
  (`243e89b…`/278 exact); rows 12/13 by reading producer/verifier constants.
- Full import enumeration (incl. in-function) of the four modules → `{numpy, jsonschema}`;
  read-only `ls` of `HateVideo` site-packages (`jsonschema 4.26.0` + transitive + `numpy 1.26.4`).
- Machine-plan reader enumeration; M-A/M-B loci re-read; `git diff` of the two human docs (+11/-0).
- Amendment `_HASHES.sha256` self-consistency (lists .md/.machine.json, both match on-disk);
  amendment `.machine.json` `jq -e .` well-formed; `artifacts/lb_scgp_global/v4` **absent** (only
  `v1`). No Python/import/`py_compile`/test/conda/SLURM executed.

---

## Required statements

- No performance evidence exists and no performance claim is made or possible from this static
  review of a byte-exact clone + plan amendment.
- The only project gold is `parent_video_binary_label`. No segment/frame/timestamp/span/
  localization/stance/target/mechanism/rationale/fragment gold is assumed or introduced; the v4
  fixtures are synthetic and v4 produced no artifact and no counters.
- Run3, M1, MLLM/cache, validation/test, training, and realbank remain **locked**.
- This review **RATIFIES** the v4 plan amendment (REPLACE-at-`[2]` affirmed) and returns
  **PASS_STATIC_REVIEW**. It authorizes **no** SLURM execution. Remaining ceremony (verdict §5):
  exact-hashes/no-clobber review → separate independent execution authorization (carrying the L-C
  dependency item, the two-phase-freeze obligations of §8.4, and conscious acceptance of M-B) →
  single executor submit (exactly one `sbatch`, no resubmission) → fresh result-to-claim review.
- This reviewer role is separate from the v4-prep/amendment-authoring, clone-freeze,
  execution-authorization, and executor roles.

Report SHA256 is to be computed externally after this file is written; it is not embedded to
avoid a self-referential hash.
