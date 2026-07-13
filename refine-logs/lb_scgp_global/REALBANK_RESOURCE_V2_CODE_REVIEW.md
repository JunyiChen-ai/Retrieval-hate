# M0 REALBANK-RESOURCE-v2 — Merged Independent Review (Amendment Ratification + Pre-Execution Static Code Review)

Date: 2026-07-13

Reviewer: **Claude Opus 4.8**, fresh / zero-context, zero-history (0C/0H) independent reviewer.
This role is distinct from the realbank-prep author (v2 clone + wrapper fix + freeze), the full-chain
static auditor, the v1 result-to-claim reviewer, and the later execution authorizer and executor. This
document performs both (i) the independent amendment review (ratify the additive/REPLACE-in-place
`runs[3]` run-id bump + the wrapper fix) and (ii) the fresh 0C/0H static code review of the v2 clone
with **independently re-derived** handoff and runtime-simulation tables. It authorizes no execution.

## Reviewer boundary & required statements

- **Read-only static analysis only.** Shell limited to `rg`/`sed`/`nl`/`jq`/`awk`/`bash -n`/`diff`/
  `sha256sum`/`find`/`ls`/`wc`/`git status`/`git diff`/`realpath`/`grep`. No Python/import/`py_compile`/
  conda/SLURM/`sbatch`/`squeue` was run. No experiment, MLLM/OCR/API/network/model/GPU/training/
  evaluation. No validation/test/held/cache/query **content** was opened (only the two allowlisted train
  **feature** banks and the val/test **provenance hashes** were checksummed, read-only, to verify
  bindings). The only file written is this report.
- **No performance evidence exists and none is claimed.** This is a static review of an unexecuted run;
  the realbank run itself emits no accuracy/macro-F1 and does no training or kNN.
- **Opus 4.8 deviation declaration (precedent as before).** Project discipline (`CLAUDE.md`) constrains
  subagents to Opus 4.8, and `AGENTS.md`'s "GPT-5.5 xhigh" cross-model backend is unavailable this
  session, so this "fresh independent" review is performed by the same model family as the implementer
  rather than a cross-model reviewer. Independence is enforced by 0C/0H context reset and by re-deriving
  every binding/table row from on-disk state (never trusting the freeze predemonstration). This mirrors
  the v1/v2/v3/v4 review precedent. Documented process fact, not a defect.
- The only project gold is `parent_video_binary_label`; no segment/frame/span/localization/stance/
  target/mechanism/rationale/fragment gold is assumed or introduced; train **labels** are not opened by
  the realbank code. Run4 (M1 cache), MLLM/cache, validation/test, and training remain locked.

---

## VERDICT

- **AMENDMENT_RATIFIED** — the machine-plan v1→v2 run-id/schema-id bump is a minimal, faithfully landed
  REPLACE-in-place at `runs[3]` (plus the two consumer deps and the four global run-id/schema-id
  registries); the array length and all downstream indices are unchanged; the four-file hash cascade is
  exact; every diff hunk maps to the declared change list with **no undeclared change**.
- **PASS_STATIC_REVIEW** — the eight v2 entities are a proven byte-clone of the frozen v1 set modulo the
  realbank-lineage version token, plus exactly the audit-confirmed three-line wrapper fix; the handoff
  chain's one defect (row A, `$TMPDIR` escape) is closed and independently re-derived to PASS; the
  three-way interface alignment, `runs[3]` index-pin, dependency set, resource policy, and fail-closed
  plumbing all hold on the v2 tokens.
- **Grade tally: Critical = 0, High = 0, Medium = 0, Low = 2** (both documentation-precision, both
  inherited from the v1 review; non-blocking). Since Critical = 0 **and** High = 0, the gate criterion
  for `AMENDMENT_RATIFIED + PASS_STATIC_REVIEW` is met.
- `ready_for_execution` remains **false**: still required before any single submit — the execution
  authorizer's dependency-availability re-confirmation (env is not frozen), an exact-hashes/no-clobber
  authorizer check, and a separate execution authorization.

---

## 1. Amendment ratification (machine plan v1 → v2)

### 1.1 Hash cascade — each `sha256sum`-recomputed from on-disk this session

| file | freeze-declared | on-disk (this session) | verdict |
|---|---|---|---|
| `EXPERIMENT_PLAN.machine.json` | `f4d54b78…` | `f4d54b78501b02253c14da2b42b2c04431f2f1c80f56c0f27b3420adf830fc9b` | **MATCH** |
| `EXPERIMENT_PLAN.md` | `a3325f9d…` | `a3325f9d2de21b7be21b3000c7a5953426908dd0444a995bb8f699a5ecb3dd8c` | **MATCH** |
| `EXPERIMENT_TRACKER.md` | `51367c17…` | `51367c176ce62979fefa37542b6a53809ff43143d819f300c1d807cacd407618` | **MATCH** |
| `EXPERIMENT_PLAN_HASHES.sha256` | `d3251b0b…` | `d3251b0b335d9806b7c14c08efc00d98243b0284cb3a599dee7c963f9f59947c` | **MATCH** |
| `…pre_realbank_v2_amendment.bak` | `d5023b62…` | `d5023b621afca08eed940d2853551c872da8151378c4a83075744e422cb18fdb` | **MATCH** |

`EXPERIMENT_PLAN_HASHES.sha256` internally lists plan.md / tracker / machine.json at the **new** hashes
(`a3325f9d` / `51367c17` / `f4d54b78`) — self-consistent. The pre-amendment machine hash `d5023b62`
recorded in the `.bak` is exactly the value the pre-v2 config *used* to bind (see §2.3).

### 1.2 `.bak → current` machine-plan diff — every hunk maps to the declared change list

`diff -u` yields a small, fully bounded change. Independent classification of every hunk:

| hunk | JSON path | change | declared? |
|---|---|---|---|
| 1 | `run_order[3]` | run-id `…RESOURCE-v1` → `-v2` | run-id site 1/6 |
| 2 | `runs[3].run_id` | `…RESOURCE-v1` → `-v2` | run-id site 2/6 |
| 3 | `runs[3].artifact_schema_ids[0]` | `scgp_global_realbank_resource_v1` → `_v2` | schema-id site 1/2 |
| 4 | `runs[3]` body | **+`v1_burn_and_v2_clone`** description key; `realbank_protocol.decision_record` and `.decided_by` extended (add audit + clone-freeze refs) | status/说明块 |
| 5 | `runs[3].status` | `GATE_OPEN_PENDING_REALBANK_IMPLEMENTATION_AND_REVIEW` → `V2_CLONE_FROZEN_AFTER_V1_TMPDIR_BURN_PENDING_INDEPENDENT_REVIEW` | status |
| 6 | `runs[4].dependencies[0]` (M1-CACHE-MHC) | `…RESOURCE-v1` → `-v2` | run-id site 3/6 (runs[4] deps) |
| 7 | `runs[5].dependencies[0]` (M1-CACHE-MHC_zh) | `…RESOURCE-v1` → `-v2` | run-id site 4/6 (runs[5] deps) |
| 8 | `dependency_dag.terminal_decision_chain[3]` | `…RESOURCE-v1` → `-v2` | run-id site 5/6 |
| 9 | `artifact_schemas.scgp_global_realbank_resource_v1` (registry key) | → `_v2` | schema-id site 2/2 |
| 10 | `readiness.g0_runs_through_v4_supplement[3]` | `…RESOURCE-v1` → `-v2` | run-id site 6/6 |

**6 run-id sites + 2 schema-id sites + 1 status + 1 description block. No line outside these changed.**

Structural invariants, independently proven this session:
- `jq '.runs|length'` = **66** on both `.bak` and current; `run_order|length` = **66** both.
- `diff <(jq -S 'del(.runs[3])' bak) <(jq -S 'del(.runs[3])' cur)` reduces to **exactly** the six
  external references (1 schema registry + run_order + terminal_decision_chain + g0_runs_through_v4 +
  runs[4].deps + runs[5].deps) — i.e. everything except `runs[3]` is byte-identical apart from the
  declared cross-references.
- `runs[0]`,`runs[1]`,`runs[2]` **byte-identical** (`.bak` vs current); `runs[6:]` **byte-identical**
  (M1-cache-SEAL onward untouched); `runs[4]`,`runs[5]` differ **only** in `dependencies`.
- Whole-file count: current machine has **0** `REALBANK-RESOURCE-v1` / **6** `-v2` run-id refs and
  **0** / **2** schema-id refs; the `.bak` is the exact mirror (6/0, 2/0). Every v1 token flipped, none
  missed, none spurious.

### 1.3 Artifact namespace kept v1 — code assertions independently confirmed consistent

The freeze intentionally keeps the artifact path at the **v1** namespace while bumping the lineage id to
v2 (lineage version ≠ artifact-path version). I confirmed all four owners agree:

- machine `runs[3].artifact_paths` = `["artifacts/lb_scgp_global/v1/m0/realbank_resource/decision.json"]`
- config `run.artifact_path` = `artifacts/lb_scgp_global/v1/m0/realbank_resource/decision.json`
- wrapper `ARTIFACT_ROOT` (line 9) = `artifacts/lb_scgp_global/v1/m0/realbank_resource`
- wrapper assertion (line 54) hard-codes `…/v1/m0/realbank_resource/decision.json`

Because the config field and the wrapper's hard-coded literal are **both** the v1 path, the wrapper's
`if CONFIG_ARTIFACT != <v1 literal> → exit 2` gate **passes** (no drift). A naive lineage bump that had
also moved the artifact namespace to `/v2/` while leaving the literal at `/v1/` (or vice-versa) would
`exit 2`; that failure mode is absent. **Consistent.**

`run_id` / `schema_id` move in lock-step: machine `runs[3].run_id` = config `run.run_id` = code
`RUN3` (both `common.py` and `independent_verify.py`) = `…REALBANK-RESOURCE-v2`; machine
`artifact_schema_ids` = config `schema_id` = `scgp_global_realbank_resource_v2`. This is the **v3-death
(code↔plan drift) prevention**: the code constant and the machine content bumped together.

---

## 2. Clone-equivalence static review (eight v2 entities)

### 2.1 Entity SHA256 — all 8 recomputed == FREEZE table

`1d69b961…` config, `4d95d128…` schema, `f90f153f…` common, `ea703b3e…` validate, `e5e9a06a…`
producer, `7ffa860b…` verify, `348b056b…` wrapper, `d7ab1e75…` sbatch. **8/8 MATCH the CLONE_FREEZE.**
The eight v1 baseline entities also recompute to the audit's frozen SHAs (`9c4ecc05` sbatch, `f80b41ea`
wrapper, `b2bbec02` validate, `dc38d5c3` producer, `49cc2d9a` verify, `46e1f3fe` common, `c436c3dd`
config, `db79cdd3` schema), so the diffs below are against the correct frozen baseline.

### 2.2 Five code/schema files = version-token rename, byte-verified

I diffed each v1→v2 file directly and inspected **every** changed line; all are realbank-lineage token
renames with **zero** behavioral change:
- **common.py** (7 lines): docstring run-id, `RUN3`, `SCHEMA_ID`, `MANIFEST_/SOURCE_MANIFEST_/
  SEMANTIC_VERIFICATION_SCHEMA_VERSION`, `CONFIG_PATH` — all `…_v1` → `…_v2` (load-bearing constants,
  moved in lock-step).
- **validate.py** (3 lines): common-import module name, `validation` schema_version, `validator_sha256`
  self-path.
- **producer.py** (1 line): common-import module name.
- **independent_verify.py** (6 lines): `RUN3`, `SCHEMA_ID`, `MANIFEST_SCHEMA_VERSION`, `CONFIG_PATH`,
  two `semantic_verification` schema_version constants.
- **schema.json** (4 lines): `schema_version` const, `artifact_schema_id` const, `run_id` const (×2).

Guard preservation independently confirmed — the rename correctly did **not** touch the reused-v1
tokens: artifact namespace `lb_scgp_global/v1/m0/realbank_resource` is preserved (config 12/12,
wrapper 2/2, producer 1/1); run1 `contract_freeze` tokens preserved (common 2/2); `scgp_global_cert_v2`
preserved (config 1/1, not bumped to v3); the `_config_v1` schema-version format suffix preserved
(`…realbank_resource_v2_config_v1`). A blanket `s/v1/v2/g` would have corrupted all of these; the guard
did not.

### 2.3 Config (6th entity) = token rename + required hash cascade + one additive binding, all verified

The config is **not** claimed byte-identical (the freeze scopes "byte-identical" to the five code/schema
files, correctly). Its diff is: (i) the 8 implementation-file paths, `paths.{payload_schema,wrapper,
slurm_script}`, `run.{run_id,schema_id}`, and `schema_version` v1→v2 renames; (ii) the **required**
plan-hash cascade — the four `authoritative_inputs` plan hashes updated from the pre-amendment values
(`10fd5232`/`d5023b62`/`a8360a2a`/`d226abfe`) to the post-amendment on-disk values (`a3325f9d`/
`f4d54b78`/`d3251b0b`/`51367c17`); (iii) one **additive** binding for `REALBANK_FULLCHAIN_STATIC_AUDIT.md`
(`ab436cc4…`). All bindings recomputed against on-disk this session:
- `authoritative_inputs`: **11/11 MATCH** (incl. the new audit-doc binding and the 4 cascade hashes;
  the bound machine hash `f4d54b78` equals the on-disk machine plan → config↔machine consistent).
- `run1_frozen`: **10/10 MATCH** (contract_freeze artifact + lock, run1 config/schemas/common/
  contract_freeze.py/validate.py/sbatch/run1.sh).
- `train_banks`: **2/2 MATCH** — MHC `deea74ff…`-class `.pt` at n=549, MHC_zh at n=579 (feature banks
  under the preserved v1 namespace).
- `old_protected_pre_snapshot`: `243e89b6…` / 278 paths — **unchanged** from v1 (R-3).
- `declared_validation_test_provenance_not_opened`: 4 entries (val/test hashes; files **not** opened).

### 2.4 Wrapper = renames + exactly the audit §5.4 three-line fix; sbatch = renames only

Wrapper diff = `RUN_ID`/`CONFIG`/`EXPECTED` + three script-path renames, **plus** the sole behavioral
change, verbatim the audit §5.4 spec (direction (a)):
```
REALBANK_TMPDIR="/data/jehc223/RGCL/slurm/tmp"
mkdir -p "$REALBANK_TMPDIR"
VALIDATION_JSON=$(mktemp "$REALBANK_TMPDIR/lbscgp_global_r2_realbank_resource_v2_validation.XXXXXX.json")
```
(the removed v1 line was `mktemp "${TMPDIR:-/tmp}/…v1_validation.XXXXXX.json"`), plus a 5-line rationale
comment. The validation-temp filename token correctly follows the lineage rename to `…_v2_validation…`.
sbatch diff = job-name + `RUN_ID` + `CONFIG` + wrapper-path renames only; **no** resource/env change
(still `--cpus-per-task=16 --mem=96G`, no `--gres`, no `--time`, `OMP/MKL/OPENBLAS=16`). `bash -n` is
clean on both.

---

## 3. Fix (a) validity — independent A-row derivation (the load-bearing check)

I re-derived the handoff-row-A fix from the actual v2 code (not the audit).

**Producer-read acceptance.** The wrapper mints `$VALIDATION_JSON` under the absolute in-repo dir
`/data/jehc223/RGCL/slurm/tmp/…`. The producer consumes it via `read_json → canonical_root_path`
(`common.py:130-139`):
```
root = ROOT.resolve()                              # ROOT = Path("/data/jehc223/RGCL")  (common:35)
candidate = raw (raw.is_absolute() → candidate = raw)
resolved  = candidate.resolve()
rel = resolved.relative_to(root)                   # ValueError → RuntimeError("path escapes …")
```
`slurm/` is a **real directory, not a symlink** (`readlink slurm` → not a symlink; `ls -ld` → real
dir), and `realpath -m slurm/tmp` = `/data/jehc223/RGCL/slurm/tmp`, so `resolved` stays under `root`,
`relative_to` returns `slurm/tmp/…json`, **no** `ValueError` is raised, and the producer read **PASSES**.
(Under v1, `$TMPDIR=/data/jehc223/home/tmp` is not under ROOT, so `relative_to` raised → the v1 preflight
death. Fixed.) Because both `ROOT.resolve()` and `candidate.resolve()` canonicalize the shared
`/data/jehc223/RGCL` prefix identically, the check is robust even if a parent component were a symlink.

**`old_protected` non-membership.** The five roots hashed by `old_protected_hash_manifest`
(`common.py:766-771`) are `configs/lb_scgp`, `artifacts/lb_scgp`, `refine-logs/lb_scgp`,
`scripts/analysis`, `scripts/slurm`. The temp lives under the **top-level** `slurm/`, which is **not**
`scripts/slurm/` — so `slurm/tmp/…` falls under **none** of the five `rglob` roots and cannot perturb the
manifest hash. (This is the exact `slurm/` ≠ `scripts/slurm/` distinction; the fix would have been unsafe
under any dir inside those five roots.)

**No-clobber / git checks do not cover it.** `slurm/tmp` appears in **no** gate list: it is not in the
config `dirty_policy.allowed_new_files_after_run` (the 8 v1-namespace artifact/lock paths; 0 slurm/tmp),
not in the wrapper `PROSPECTIVE_OUTPUTS` (the same 8 artifacts), and not in any `git diff --check` /
`relevant_git_status` bounded pathspec (grep of all v2 `.py` for `slurm/tmp` → none). The transient file
is therefore invisible to every clobber/diff/status gate.

**`mkdir -p` idempotency + cleanup-trap coverage.** `mkdir -p "$REALBANK_TMPDIR"` is idempotent
(`slurm/` exists and is user-writable; `slurm/tmp` is currently absent and is created at runtime; `-p`
no-ops if present). `VALIDATION_JSON=""` is initialised (line 11) and the `EXIT`/HUP/INT/TERM traps are
**armed (line 38) before** the temp is minted (line 66); `cleanup_on_exit` does `rm -f "$VALIDATION_JSON"`
on every exit path (line 26), removing the in-repo temp. The empty `slurm/tmp/` dir is left behind but is
git-invisible (audit's optional `rmdir` intentionally not bundled). Covered.

**No second landmine.** The only `TMPDIR`/`/tmp` references remaining in the whole v2 chain are the fix
itself and its comment; the v1 `${TMPDIR:-/tmp}` fallback is gone. Both Python `tempfile.mkstemp` sites
(`common.py:195`, `independent_verify.py:796`) pass an explicit in-repo `dir=str(fs_path.parent)` and
never consult `$TMPDIR`. After fix (a) the chain has **zero** ambient-environment-derived file paths.

---

## 4. Two-table independent re-derivation (v2 tokens)

### 4.1 Full-chain handoff table (11 rows) — re-derived

Row **A PASS** (derived in §3, was FAIL). Rows B–K PASS: config/machine/schema each parse and are read
in-repo through `jq`/`canonical_root_path`/`root_path`; train banks are allowlisted + sha-checked; the
run1 contract-freeze artifact + lock, and the producer/verifier-published `source_manifest`/
`access_ledger`/`decision`/`semantic_verification` are all in-repo under the v1 artifact namespace; the
atomic `mkstemp` tempfiles use explicit in-repo `dir=`. **Tally 11/11 PASS**, 0 out-of-repo path.

### 4.2 Runtime cross-check simulation table (21 rows) — independently re-verified on v2

| row | assertion | independent verdict on v2 |
|---|---|---|
| 1 | wrapper `RUN_ID==EXPECTED`; config run_id/artifact | **PASS** — default==EXPECTED==`…RESOURCE-v2`; config artifact = v1-namespace `…/decision.json` |
| 2 | `require_slurm_realbank` 16/96/0 | **PASS (env at runtime)** — sbatch `--cpus-per-task=16 --mem=96G`, no `--gres`, no `--time`; guard fail-closed |
| 3 | validate `jq -e .` config/machine/schema/run1 | **PASS** — all four parse (`jq -e .`) |
| 4 | validate `schema_strict_check` | **PASS** — 21 `additionalProperties:false`; 23 top-level `required` |
| 5 | validate `bash -n` wrapper + sbatch | **PASS** — both clean |
| 6 | dependency numpy/torch/jsonschema | **PASS (runtime)** — validate builds `names=['numpy','torch','jsonschema']`, `find_spec`, `sys.exit(1 if missing)`; all three present in `…/envs/HateVideo/lib/python3.11/site-packages` |
| 7 | validate `py_compile` the 4 `.py` | **DEFERRED-TO-RUNTIME (fail-closed)** — login-node compile forbidden; runs in SLURM validator |
| 8 | validate run1 hashes + old_protected | **PASS (run1 10/10) / runtime (old_protected `243e89b6…`/278)** |
| 9 | validate `verify_authoritative_hashes` | **PASS** — 11/11 == on-disk (incl. new audit-doc binding) |
| 10 | validate `verify_train_bank_bindings` | **PASS** — 2/2 sha == on-disk; n=549/579 |
| 11 | validate `no_clobber_check` | **PASS** — `artifacts/lb_scgp_global/v1/m0/realbank_resource/` absent |
| 12 | validate `resource_and_run_check` | **PASS** — run_id/schema_id/artifact_path/slurm all match |
| 13 | producer `verify_config_and_schema` | **PASS** — 10 authorization flags `false` + `train_bank_read_allowed=true`; `authorized_run_ids=[…-v2]`; schema strict |
| 14 | producer `verify_machine_realbank` `runs[3]` | **PASS** — `jq`: run_order[3]/run_id/artifact_paths/schema_ids/slurm/deps=[SYNTH-KKT-v4]/banks == config |
| 15 | producer manifest schema validation | **PASS (by construction)** — 23 required keys, strict-validated pre-publish |
| 16 | verifier `set(manifest)==TOP_KEYS` | **PASS** — verifier `TOP_KEYS`(23) == schema.required(23), set-identical |
| 17 | verifier `zero_counters` set | **PASS** — common == verifier == schema, all 47, set-identical |
| 18 | verifier `verify_machine` `runs[3]` | **PASS** — independent code path, same result as row 14 |
| 19 | verifier injection recompute == manifest | **PASS (by construction)** — 11 cases (common == schema); FORBIDDEN_TOKENS byte-identical v1→v2 (`query_z,query_labels,teacher,cache,held,certificate`) |
| 20 | verifier authoritative/run1 on-disk == config | **PASS** — == rows 8–9 |
| 21 | verifier GO consistency | **PASS (expected)** — N=549/579 O(N³) peak ≪ 96 GiB; `rank_eps(G0)≤N≤d`; in-process replay deterministic; injections REJECT; cross-process replay = the one R-2 fail-closed dependency |

**No row FAILs.** The runtime-deferred rows (2 env, 6 dependency, 7 py_compile, 8 old_protected, 21
cross-process replay) are fail-closed: any miss raises → `FAIL`/non-GO → wrapper cleanup → no artifact,
no false PASS. The interface-alignment counts (23 top-keys, 47 zero-counters, 11 injection cases) were
extracted directly from the v2 entities this session, and the two `common.py`/`verify.py` decision-inert
deviations from the accepted v4 code (`factor_from_psd_gram` non-PASS early-return omission; `orth_cap`
unused `singular_values`) are **byte-identical v1→v2** — the v1 review's Low-1 grading carries over
verbatim (both provably GO-inert).

---

## 5. Findings & grading

- **Low-1 (inherited, documentation precision).** The "byte-faithful copy of the v4 code" framing remains
  slightly overstated: `factor_from_psd_gram` and `orth_cap` carry two documented, **decision-inert**
  deviations (confirmed byte-identical v1→v2 this session). Non-blocking; recommend the "faithful reuse
  with two documented decision-inert deviations" wording. Same finding graded in the v1 review.
- **Low-2 (inherited, out of scope, documentation coherence).** The two stale v2-era summary lines flagged
  by the v1 review (`EXPERIMENT_TRACKER.md` "Status-aware counts"; `EXPERIMENT_PLAN.md` milestone rollup
  under the `## G0 Runs Through Run2-v2 Supplement` header) were not in this amendment's declared scope
  (the v2 amendment touches only `runs[3]` + the two deps + the four registries) and persist. No code
  reads them; the authoritative machine plan is correct. Non-blocking; recommend a follow-up doc pass.
- **Observations (not findings).** (i) The config legitimately carries the plan-hash cascade + one
  additive audit-doc binding beyond the pure token rename — required and verified all-MATCH on-disk; the
  freeze correctly scopes "byte-identical" to the five code/schema files only. (ii) The wrapper leaves an
  empty `slurm/tmp/` directory after the run (git-invisible; the audit's optional `rmdir` was
  intentionally not bundled to keep the blast radius to the wrapper).

**Critical = 0, High = 0, Medium = 0, Low = 2.**

---

## 6. Conclusion

The REALBANK-RESOURCE-v2 machine amendment is **RATIFIED** (minimal REPLACE-in-place `runs[3]` run-id/
schema-id bump across all six run-id and two schema-id sites, exact four-file hash cascade, array length
66 and downstream indices unchanged, artifact namespace deliberately and consistently kept at v1) and the
eight-entity v2 clone **PASSES STATIC REVIEW** (proven byte-clone of the frozen v1 set modulo the lineage
token, plus exactly the audit §5.4 three-line wrapper fix; all bindings recompute == on-disk; three-way
interface alignment 23/47/11 re-verified; `runs[3]` index-pin and code↔plan lock-step intact). The v1
`$TMPDIR` preflight-death class is **closed**: I independently re-derived that the in-repo `slurm/tmp/`
handoff resolves under ROOT (so the producer's `canonical_root_path` accepts it), sits outside all five
`old_protected` roots (`slurm/` ≠ `scripts/slurm/`) and every no-clobber / git pathspec, is idempotently
provisioned by `mkdir -p`, and is removed by the pre-armed cleanup trap — with no second ambient-env
landmine anywhere in the chain.

Two inherited, documentation-only Low findings; **no Critical, no High** → the criterion for
`AMENDMENT_RATIFIED + PASS_STATIC_REVIEW` is satisfied. Execution remains **unauthorized**: it still
requires the execution authorizer's dependency-availability re-confirmation (env is not frozen), the
exact-hashes/no-clobber check, and a separate execution authorization before the single CPU-only
`sbatch`; any fail-closed non-GO is again a consciously-accepted STOP, not grounds for a second submit.

Reviewer = Claude Opus 4.8, fresh independent 0C/0H amendment-review + code-review role, separate from
realbank-prep/freeze, full-chain audit, v1 result-to-claim, execution authorization, and executor. Wrote
only this document; edited no code/config/schema/plan; submitted no job.
