# LB-SCGP Global-R2 M0 Run2-v3 Fresh Result-to-Claim Review

Date: 2026-07-13

Reviewer: **Claude Opus 4.8**, fresh / zero-context, zero-history (0C/0H) independent
result-to-claim reviewer. Role separation: this reviewer is a **distinct role** from the v3
static code reviewer (`M0_RUN2_V3_CODE_REVIEW.md`), the v3 clone-freezer
(`M0_RUN2_V3_CLONE_FREEZE.md`), the v3 execution-authorizer
(`M0_RUN2_V3_EXECUTION_AUTHORIZATION.md`), and the v3 executor.

Reviewer boundary: read-only adjudication only. No subagents, workflows, model/API calls,
SLURM submissions, experiments, GPU/training/performance work, validation/test work, MLLM/OCR
work, environment mutation, package installs, or code/config/schema/wrapper/artifact edits were
performed. Python was invoked **only** in read-only mode to parse and walk the already-frozen
plan JSON (no interpreter run touched the target run2 modules, wrote artifacts, or reached any
network/model/GPU path); every quoted assert line and every quoted document value was
established by direct read of the on-disk file. This file is the only new write.

Precedent binding: this review follows and is consistent with
`M0_RUN2_RESULT_TO_CLAIM_REVIEW_FRESH.md` (v1-lineage verdict: `claim_supported=no`,
`route=infrastructure_repair`; v1 lineage closed after two fail-closed `KeyError` attempts,
jobs `12902`/`12904`) and `M0_RUN2_V2_RESULT_TO_CLAIM_REVIEW.md` (v2-lineage verdict:
`claim_supported=no`, `route=infrastructure_repair`; v2 lineage closed after one fail-closed
missing-`jsonschema` attempt, job `12971`). This document adjudicates the single consumed **v3**
attempt (job `12974`) and specifies the conditions under which a **v4** lineage may open.

---

## Structured Verdict

- intended_claim: The v3 synthetic global-projection / serialized H-metric normal-cone KKT
  gate is executable and independently verifiable for LB-SCGP Global-R2.
- claim_supported: **no**
- route: **infrastructure_repair** (protocol / plumbing — authoritative-plan amendment)
- failure_classification: **infrastructure_plan_document_code_drift** (NOT science)
- confidence: **high**
- v3_lineage_disposition: **single-submit consumed → lineage CLOSED; must not be re-proposed**
- science_information_leak: **none**
- false_positive_risk: **none** (fail-closed; died before rank/KKT construction; no `decision`
  emitted, no artifact published)
- repair_authorization: authorized **in principle only** for one new non-overwriting **v4**
  lineage; execution is **not** authorized by this report.
- execution_authorization: **not_authorized**
- plan_amendment_required: **yes** (a v4 plan-amendment ceremony is a hard precondition — this
  is the specific defect that killed v3)
- new_mandatory_review_item: **yes** — a runtime cross-check static-simulation table (§5c)
- Run3 / M1 / MLLM-cache / validation-test / training / realbank: **locked**

---

## 1. Failure classification — infrastructure, not science (independently verified)

Job `12974` FAILED with a `RuntimeError` I read directly at
`slurm/logs/lbscgp_global_r2_run2_v3_12974.err:12`:

```
RuntimeError: machine run order[2] drift: expected 'LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v3',
got 'LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2'
```

The traceback is exact and shallow. It is raised by `assert_equal`
(`lb_scgp_global_r2_run2_v3_common.py:387`) from **`verify_machine_run2`**
(`…v3_common.py:821`), which is called from **`producer.py:271`**, which is the **seventh** of
nine ordered steps inside `producer.main()`:

1. `require_slurm_run2()` — passed (job ran under SLURM).
2. `assert_equal(args.run_id, RUN2)` (`producer.py:249`) — passed (`RUN2` = `…-v3`; wrapper
   passed the v3 run-id).
3. `read_json(config)` + `verify_config_and_schema(cfg)` (`:251-252`) — passed.
4. Four no-clobber existence refusals (`:254-262`) — passed (no v3 artifact pre-existed).
5. `assert_equal(validation["run_id"], RUN2)` and `validation["status"] == "PASS"`
   (`:266-267`) — passed. **`validate.py` had already emitted `status:PASS`**: the missing-
   `jsonschema` residual that killed v2 (job `12971`) was genuinely repaired, so the v3
   validator preflight cleared. This is the one thing v3 advanced past v2.
6. `ledger.hash_file(config)` (`:270`) — passed.
7. **`verify_machine_run2(cfg, ledger)` (`:271`) — DIED** on the very first assert inside it
   (`…v3_common.py:821`, `machine["run_order"][2] == RUN2`).

Steps 8–9 (`build_source_manifest` at `:272`, then all rank/factor/KKT/movement construction at
`:277-409`) **never ran**. Because the death is at step 7, it is **earlier** than the two
statically-flagged science-adjacent residuals from the v3 code review — M-B (rank-deficient
construction convergence, `M0_RUN2_V3_CODE_REVIEW.md:295`) and M-A (synthetic `G0` PSD/rank
realizability, `:266`) — both of which live in the step-8-onward numeric section that was never
reached.

Terminal facts I independently confirm:

- No `artifacts/lb_scgp_global/v3/` tree exists; no manifest, source_manifest, access_ledger,
  semantic_verification, or publish lock was written (the producer's writes are at `:400-405`,
  well past the death point, and are `try`-guarded with `cleanup_created_outputs`).
- No `decision`/PASS gate, no KKT tolerance, stationarity, dual-feasibility, complementarity,
  rank-tail, factor-replay, movement, or any numeric/solver result was computed.

**Zero false-positive risk, zero science leak.** The gate is fail-closed and refused to proceed;
a plan/code drift cannot manufacture a false PASS. This is not a scientific, optimization,
numerical, KKT, rank, factor, solver, mechanism, MLLM, OCR, dataset, or performance result of
any kind. It is a **pure infrastructure failure** at a plan-vs-code consistency assert, before
any scientific computation — the same fail-closed, zero-return character as v1 (interface
`KeyError`) and v2 (missing dependency), one layer further out (**authoritative-document
content vs code constant**).

## 2. Root cause — authoritative-plan / code version drift (statically preventable)

The v3 clone advanced the **code side** to v3: `RUN2 = "…SYNTH-KKT-v3"` and
`PAYLOAD_SCHEMA_ID = "…payload_v3"` (`…v3_common.py:25-26`), and `cfg.run.run_id` /
`cfg.authorization.authorized_run_ids` are v3. But the **authoritative plan document** —
`refine-logs/lb_scgp_global/EXPERIMENT_PLAN.machine.json`, which `verify_machine_run2` reads via
`cfg["paths"]["experiment_machine"]` and cross-checks at fixed index `[2]` — still describes
**v2** at that index:

- `machine["run_order"][2]` = `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2` (the exact assert that fired).
- `machine["runs"][2]["run_id"]` = `…-v2` (the next assert that would have fired,
  `…v3_common.py:822`).

The reason the plan still says v2 is structural: v2 got itself written into the authoritative
plan through a **plan-amendment ceremony** — `M0_RUN2_V2_PLAN_AMENDMENT.md` +
`M0_RUN2_V2_PLAN_AMENDMENT.machine.json` (whose `pre_post_run_order_diff` INSERTED
`…SYNTH-KKT-v2` into `run_order` exactly once and repointed realbank's dependency v1→v2), passed
through its own independent review (`M0_RUN2_V2_AMENDMENT_INDEPENDENT_REVIEW.md`). **v3 performed
no corresponding amendment.** The clone freeze and code review treated the plan JSON as an
immutable input to hold constant ("keep original values unchanged during clone") — correct for a
byte-clone, but it left the plan pinned to v2 while the code moved to v3. The result is a
deterministic, **statically foreseeable** document-code drift.

Why the v3 review did not catch it: `M0_RUN2_V3_CODE_REVIEW.md` §1 proved byte-clone equivalence
of the **nine cloned entities**, §3 proved that references **internal to those nine entities**
resolve to v3 (cert stays v2), and §3 (`:188`) noted "none of the nine cloned entities has its
hash bound in the config." That scope is code-internal. The machine JSON is **not** one of the
nine cloned entities — it is an external authoritative input that the code **reads and asserts
against** at runtime. No step of the v3 review statically simulated `verify_machine_run2`
executing against the on-disk plan (i.e. "does `machine.run_order[2]` on disk equal the code's
`RUN2` constant?"). It does not — and could not, without either the amendment or the simulation.

### The three-lineage failure signature (one shared root)

| Lineage | Job | Failure | Class | Fail-closed? | False-pos risk | Statically preventable? |
|---|---|---|---|---|---|---|
| v1 | 12902 / 12904 | interface `KeyError` | infrastructure (interface) | yes | none | yes |
| v2 | 12971 | missing `jsonschema` dependency | infrastructure (environment) | yes | none | yes |
| v3 | 12974 | plan `run_order[2]` v2/v3 drift | infrastructure (plan document) | yes | none | yes |

All three burned the **sole authorized single-submit quota** at preflight, returned **zero
science**, and were each **predictable by static means**. The common root (see §6) is that every
review to date verified code-internal consistency but never statically evaluated the code's
**runtime assertions against frozen external state** (interface contract / environment / plan
document).

## 3. What the results support / do not support

Support only these narrow procedural facts:

- Run1 `…CONTRACT-FREEZE-v1` remains frozen and untouched; the v3 attempt wrote nothing.
- The v2 missing-`jsonschema` residual is genuinely repaired: v3's `validate.py` reached
  `status:PASS`, and the producer advanced through six of its own preflight steps.
- v3 job `12974` fail-closed at the authoritative-plan consistency assert (`run_order[2]` v2/v3
  drift) and published no artifact.

Do **not** support: any claim of v3 executability or independent verifiability; any scientific,
numeric, KKT, rank, factor, mechanism, or performance claim; any Run3 / M1 / realbank unlock.

## 4. v3 lineage disposition — CLOSED

Per the single-submit regime carried from the v1 and v2 verdicts and instantiated by the v3
execution authorization, the v3 budget is now **spent**: job `12974` reached a terminal state
(FAILED). **The v3 lineage is closed and must not be re-proposed, re-run, or resubmitted under
any outcome.** Any further attempt requires a brand-new **v4** lineage and a brand-new
execution-authorization pass; the existing v3 authorization document is void for future use.

## 5. Route judgment and v4-lineage opening conditions

route: **infrastructure_repair** (protocol/plumbing — plan amendment), consistent with the v1
and v2 precedents. v4 may be opened **only** after **all** of the following are satisfied, in
order.

### (a) Plan-amendment ceremony — write v4 into the authoritative plan FIRST

Mirror the v2 precedent (`M0_RUN2_V2_PLAN_AMENDMENT.md` + its
`M0_RUN2_V2_AMENDMENT_INDEPENDENT_REVIEW.md`): author `M0_RUN2_V4_PLAN_AMENDMENT.md` +
`.machine.json` (+ `_HASHES.sha256`), edit the authoritative
`EXPERIMENT_PLAN.machine.json` (and the human `EXPERIMENT_PLAN.md` / `EXPERIMENT_TRACKER.md` for
coherence), recompute the affected hashes, and pass a **fresh independent amendment review**.

**Placement semantics — REPLACE at index `[2]`, not INSERT (mandatory; diverges from the v2
precedent with cause).** The v2 amendment *inserted* `…-v2` at index 2 and shifted realbank
2→3, because v2's code was a rewrite whose `verify_machine_run2` moved from `run_order[1]`
(v1 code) to `run_order[2]`. v4 is a **byte-clone** of v3, so its `verify_machine_run2` is fixed
at `machine["runs"][2]` / `run_order[2]` (`…v4_common.py:820-821`, a numeric literal a token-sed
cannot move). Therefore the v4 amendment must make **index `[2]` itself become v4**, i.e.
**replace the index-2 record in place** (v2 content → v4 content), keeping array length and every
downstream index (realbank at `[3]`, and the entire `[3..65]` tail) unchanged. Do **not** insert
(inserting v4 at `[2]` pushes v2 to `[3]` and realbank to `[4]`, is temporally backwards, and
shifts every downstream index against the still-uncoded realbank run). Prior-version (v1/v2/v3)
failure evidence is preserved in the refine-logs lineage (result-to-claim reviews + execution
records), not as run-array entries. The amendment's independent review **must explicitly ratify
this REPLACE semantics** as the correct byte-clone analogue of the v2 INSERT.

#### (a.1) Machine-JSON field change list — the construction blueprint

Every `v2`-lineage string in `EXPERIMENT_PLAN.machine.json` was enumerated (33 locations). They
split into three groups. **Group A is Critical** — these are the fields
`verify_machine_run2` actually reads and asserts against code constants; leaving any at v2 =
the exact v3 death.

**Group A — CODE-VALIDATED at index `[2]` (must flip to v4; each is a hard runtime assert):**

| # | JSON path | current (v2) | required (v4) | assert |
|---|---|---|---|---|
| A1 | `run_order[2]` | `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2` | `…-v4` | `…v4_common.py:821` (**the killer**) |
| A2 | `runs[2].run_id` | `…-v2` | `…-v4` | `:822` |
| A3 | `runs[2].artifact_paths[0]` | `artifacts/lb_scgp_global/v2/m0/synth_kkt/manifest.json` | `…/v4/…` | `:823` (must equal `cfg.run.artifact_path`) |
| A4 | `runs[2].artifact_schema_ids[0]` | `scgp_global_synth_kkt_payload_v2` | `…payload_v4` | `:824` (must equal `PAYLOAD_SCHEMA_ID`) |

**Group A′ — code-validated but MUST NOT change (guard against over-editing):**

| # | JSON path | value | assert | note |
|---|---|---|---|---|
| A5 | `runs[2].slurm` | `{cpu:8, ram_gb:64, gpu:0, env:"HateVideo", no_time_flag:true}` | `:825` (== `cfg.run.slurm`) | identical v2/v4; **leave unchanged** |
| A6 | `runs[2].dependencies` | `["…CONTRACT-FREEZE-v1", "…SYNTH-KKT-v1"]` | `:826` (== `[RUN1, RUN2_V1]`) | v4 depends on **synth-kkt-v1**, not v2/v3; **leave unchanged** |

**Group B — PLAN-COHERENCE (v2-token, NOT read by `verify_machine_run2`; flip for internal /
DAG consistency and to avoid a stale v2 record):** `runs[2].purpose`,
`runs[2].planned_config_path` (→ `m0_synth_kkt_v4.json`), `runs[2].artifact_namespace` (→
`…/v4/…`), `runs[2].lineage_bound_outputs[0..7]` (`v2_*` → `v4_*`),
`runs[2].science_freeze_assertions.solver_math_verifier_logic_change_scope`, `runs[2].gate`,
`runs[2].status`; **`runs[3].dependencies[0]`** (realbank: `…SYNTH-KKT-v2` → `…-v4`, the
downstream repoint the v2 amendment did as v1→v2), `runs[3].gate`;
`dependency_dag.pre_post_run_order_diff.run3_new_dependency` (→ v4) and its historical
`inserted_run_id`/`inserted_after` (record the v2→v4 REPLACE in the amendment doc /
`supplement_amendment` block rather than silently rewriting v1→v2 history — an amendment-review
decision), `dependency_dag.terminal_decision_chain[2]` (→ v4);
`readiness.first_three_g0_runs[2]` (→ v4), `readiness.status`,
`readiness.g0_runs_through_v2_supplement[2]` (→ v4 **and rename the key** `…through_v2_…` →
`…through_v4_…`); the entire `supplement_amendment` block (`lineage`, `status`,
`fix2_freeze_artifact` → the v4 freeze doc, `planned_config_path`, `artifact_path`,
`artifact_namespace`, `run3_realbank_dependency` → v4).

**Group C — REGISTRY coherence (no v2 literal at a code-read site, but incomplete):**
`artifact_schemas` is a dict currently holding `scgp_global_synth_kkt_payload_v1` and `…_v2`
only — **register `scgp_global_synth_kkt_payload_v4`**. No run2 module reads
`machine["artifact_schemas"]`, so this is coherence-only, not a Group-A Critical; but the
amendment should add it so the plan is self-consistent.

#### (a.2) Hash recompute cascade — THIS lineage HAS recompute points (unlike the v3 clone)

The v3 clone had **no** hash recompute (pure token-sed; plan held constant). Editing the plan in
(a.1) forces a bounded cascade that the runtime asserts **do** check:

1. Edit `EXPERIMENT_PLAN.machine.json` (Groups A/B/C) → its SHA256 changes from the current
   `6caa5c2e…0492` to a new `M′`.
2. `config.hash_bindings.authoritative_inputs["…/EXPERIMENT_PLAN.machine.json"]` := `M′`.
   **Checked at runtime** by `verify_expected_hashes` (`…v4_common.py:840`→`:834`, in
   `build_source_manifest`) **and** by `independent_verify.py:952-954`. If the plan is edited but
   this binding is not updated (or vice versa) → Critical hash-drift death.
3. `EXPERIMENT_PLAN_HASHES.sha256`: update its `EXPERIMENT_PLAN.machine.json` line to `M′` (and
   its `EXPERIMENT_PLAN.md` / `EXPERIMENT_TRACKER.md` lines if those are edited). This changes
   that file's own SHA256 from `2e6d731d…c802` to `H′`.
4. `config.hash_bindings.authoritative_inputs["…/EXPERIMENT_PLAN_HASHES.sha256"]` := `H′`
   (same two runtime checks as step 2).
5. If `EXPERIMENT_PLAN.md` / `EXPERIMENT_TRACKER.md` are edited for coherence, update their
   `authoritative_inputs` bindings to the new hashes (same two runtime checks).
6. **Add** `authoritative_inputs` entries for the new v4 amendment documents
   (`M0_RUN2_V4_PLAN_AMENDMENT.md` / `.machine.json` and its independent review), mirroring how
   the config already binds the v2 amendment + review docs — so provenance covers v4, not just
   up-to-v2.
7. Steps 2/4/5/6 change the **config** → its own SHA256 changes (from `e6d33b5d…b7d5`). The
   config's own hash is **not** pinned by any runtime assert against a frozen expected
   (`independent_verify.py:1016` compares it to the manifest's self-recorded value, computed at
   runtime from the same file), so the cascade **terminates** at the config — no infinite loop.
   But the new config hash **is** a ceremony freeze target: re-freeze it in the v4 clone-freeze
   doc + `FILE_HASHES.sha256`.

Sequencing note (from the v2 precedent): the config binds the amendment's **independent review**
doc, so the order is amendment authored → amendment independently reviewed → `authoritative_inputs`
updated to bind the v4 amendment + review → config re-frozen. Do not freeze the config against
an unwritten review.

### (b) v4 = byte-exact clone of the v3 code + amendment-driven binding updates

v4 source must be a **byte-for-byte-equivalent clone** of the frozen v3 nine entities, changes
limited strictly to: file names, internal self-references (`v3` → `v4` for module names /
`run_id` / namespace / the run's own artifact paths), **cert_v2 preserved** (it is a permanent
shared-schema name, not a lineage marker — see `M0_RUN2_V3_CODE_REVIEW.md:131-158`), and the
config `hash_bindings` fields that the (a.2) cascade touches (steps 2/4/5/6). **No behavioral
code change is permitted** — in particular the index literals `machine["runs"][2]` /
`run_order[2]` must stay `[2]` (that is precisely why (a) must REPLACE at index `[2]`). Any
behavioral edit voids clone status and forces a full v4 implementation audit.

### (c) NEW MANDATORY review item — runtime cross-check static-simulation table

The v4 static code review **must** add a section that enumerates **every** runtime assertion in
`validate.py` / `producer.py` / `independent_verify.py` (and the `common.py` helpers they call)
that reads a **frozen external document** (plan JSON, config, run1-frozen artifacts, schema
files, authoritative-input hashes), and **statically evaluates each one against the on-disk
amended state**, producing an explicit PASS/FAIL verdict per row. **Any row that does not
provably evaluate to PASS = Critical.** This is the check that structurally did not exist for v3.
The table must cover at minimum (line numbers are v3's; carry to v4):

| Row | Assert (site) | Reads frozen | Static verdict must show |
|---|---|---|---|
| 1 | `run_order[2] == RUN2` (`common:821`) | plan | plan `run_order[2]` on disk == code `RUN2` (=v4) |
| 2 | `runs[2].run_id == RUN2` (`common:822`) | plan | == v4 |
| 3 | `runs[2].artifact_paths == [cfg.run.artifact_path]` (`common:823`) | plan+cfg | both `…/v4/…manifest.json` |
| 4 | `runs[2].artifact_schema_ids == [PAYLOAD_SCHEMA_ID]` (`common:824`) | plan | both `…payload_v4` |
| 5 | `runs[2].slurm == cfg.run.slurm` (`common:825`) | plan+cfg | dicts equal (unchanged) |
| 6 | `runs[2].dependencies == [RUN1, RUN2_V1]` (`common:826`) | plan | `[freeze-v1, synth-kkt-v1]` (unchanged) |
| 7 | `verify_expected_hashes(authoritative_inputs)` (`common:840`→`834`) | cfg-bound hashes | every bound hash == on-disk file hash **after** the (a.2) cascade, esp. `machine.json`=`M′`, `PLAN_HASHES`=`H′` |
| 8 | `verify_expected_hashes(run1_frozen)` (`common:846`) | run1 frozen | unchanged (amendment does not touch run1) |
| 9 | `old_protected manifest/count` (`common:860-861`) | old lb_scgp tree | unchanged — note the old-protected scope is `refine-logs/lb_scgp/` and `lb_scgp_*` (non-`lb_scgp_global_r2_*`) scripts, so the amendment (in `refine-logs/lb_scgp_global/`) and v4 code (`lb_scgp_global_r2_run2_v4_*`) are **excluded** → PASS |
| 10 | `resource_and_run_check` (`validate:112-120`) | cfg internal | `run_id`/`schema_id`/`authorized_run_ids` all v4 |
| 11 | `authoritative_inputs unchanged` (`independent_verify:952-954`) | cfg-bound hashes | same as row 7 |
| 12 | `manifest.artifact_schema_id/run_id == SCHEMA_ID/RUN2` (`independent_verify:1039`) | produced manifest | producer writes v4 constants |
| 13 | `manifest.authorized_boundary == {run_id:RUN2,…}` (`independent_verify:1043`) | produced manifest | v4 |

The load-bearing insight this table operationalizes: **rows 7/8/11 (the hash layer) verify only
"the document I read is the document I froze"; they are structurally blind to rows 1–4 (code
constant vs frozen-content). In v3 the hash layer would have PASSED** (the config faithfully
bound the v2-content plan and the plan was v2-content) **while rows 1–2 FAILED** (code=v3 vs
content=v2). Hash-consistency ⊥ semantic-consistency; only the code-constant-vs-content rows
catch version drift.

### (d) Dependency-availability evidence + remaining ceremony (carry v2 + v3 lessons)

Retain the v2 lesson: the v4 execution authorizer must, by read-only means, re-confirm every
third-party import of the four v4 modules (`{numpy, jsonschema}`, including the **deferred /
in-function** `jsonschema` import) is present in `HateVideo` **before** authorizing (the
`M0_ENV_REPAIR_RECORD.md` install must still hold; env state is not frozen). Then the rest of the
ceremony as v3: **freeze** the v4 entities with exact SHA256 bindings → **fresh 0C/0H static code
review** including the (c) static-simulation table, re-adjudicating residuals M-A / M-B (still
carried) → **independent execution authorization** (with the dependency-availability item **and**
a confirmation that the amendment review passed and rows 1–13 evaluate PASS) → **independent
executor single submit** (exactly one `sbatch`, no resubmission under any outcome) → fresh
result-to-claim review of the outcome.

## 6. Process lesson (recorded)

The three consecutive single-submit burns (v1 `KeyError`, v2 missing dependency, v3 plan drift)
share one root: **every review verified code-internal consistency but never statically evaluated
the code's runtime assertions against frozen external state** — the interface contract (v1), the
environment (v2), and the authoritative plan document (v3). Each residual was fail-closed and
zero-false-positive, yet each consumed the sole authorized quota at preflight for zero science,
and each was predictable by static means.

Corrective rules, in force for all future single-submit lineages:

1. **(c) is mandatory**: the static-simulation table above must accompany every code review; any
   runtime assert against a frozen document that is not shown to evaluate PASS gates
   authorization.
2. **Dependency-availability evidence** (v2 rule) remains a hard authorization gate.
3. **Escalation**: if the v4 attempt **also** dies at a preflight-class failure (any fail-closed
   miss before the numeric section), **pause the per-version amendment/clone ceremony** and
   perform a **one-time full-chain static-execution audit** — a line-by-line dry-run of
   `validate → producer → independent_verify` against the complete frozen on-disk state, front to
   back — **before** opening a v5. Three identical-class burns indicate the ceremony's review
   scope, not the individual fix, is the defect.

---

## Required statements

- This review is not performance evidence and makes no performance claim; none is possible from
  a preflight-failed run that computed no numeric result.
- The only project gold is `parent_video_binary_label`. No segment/frame/timestamp/span/
  localization/stance/target/mechanism/rationale/fragment gold is assumed or introduced; v3
  produced no artifact and no counters.
- Run3, M1, MLLM/cache, validation/test, training, and realbank remain locked.
- The v3 single-submit budget is spent and the v3 lineage is closed; this report authorizes no
  execution. A v4 lineage requires the full ceremony in §5 — plan amendment (a) + independent
  amendment review, byte-clone (b), the mandatory static-simulation table (c), and the
  dependency-availability + remaining ceremony (d) — before any submission.
- Reviewer role (this document) is separate from the static-code-review, clone-freeze,
  execution-authorization, and executor roles.
