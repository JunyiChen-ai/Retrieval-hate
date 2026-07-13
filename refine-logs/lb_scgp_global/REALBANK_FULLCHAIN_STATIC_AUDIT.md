# REALBANK Full-Chain Static Execution Audit

Date: 2026-07-13

Auditor: **Claude Opus 4.8**, fresh independent full-chain static-audit role, triggered by the
v4-authorization **§7 escalation rule** (preflight-class death → pause ceremony, run full-chain audit
before v-next). Read-only except this document + the companion
`REALBANK_RESOURCE_V1_RESULT_TO_CLAIM_REVIEW.md`. This audit performs a **line-by-line execution-path
calculus** of the entire realbank entity set — I trace it as if I am the kernel running it — and
statically judges every file `open/read/write`, every `Path`/`canonical_root_path` check, every
`os.environ` read, every `subprocess`, and every `assert`'s left/right provenance as PASS / FAIL /
UNPROVABLE. It confirms (or would overturn) the coordination-ruled fix direction **(a)** and specifies
the exact diff.

Entities audited (frozen SHA per `REALBANK_RESOURCE_V1_FREEZE.md`):
1. `scripts/slurm/lb_scgp_global_r2_m0_realbank_resource_v1.sbatch` (`9c4ecc05…`)
2. `scripts/wrappers/lb_scgp_global_r2_realbank_resource_v1.sh` (`f80b41ea…`)
3. `scripts/analysis/lb_scgp_global_r2_realbank_resource_v1_validate.py` (`b2bbec02…`)
4. `scripts/analysis/lb_scgp_global_r2_realbank_resource_v1_producer.py` (`dc38d5c3…`)
5. `scripts/analysis/lb_scgp_global_r2_realbank_resource_v1_independent_verify.py` (`49cc2d9a…`)
6. `scripts/analysis/lb_scgp_global_r2_realbank_resource_v1_common.py` (`46e1f3fe…`)
7. `configs/lb_scgp_global_r2/m0_realbank_resource_v1.json` (`c436c3dd…`)
8. `schemas/lb_scgp_global_r2/scgp_global_realbank_resource_v1.schema.json` (`db79cdd3…`)

Repo root `ROOT = /data/jehc223/RGCL`. Runtime environment fact established from
`slurm/logs/…_12994.err`: **`$TMPDIR = /data/jehc223/home/tmp`** (out-of-repo).

---

## Verdict (summary)

**One defect, single point of failure, fully proven; fix direction (a) CONFIRMED (not overturned).**

- The chain contains **exactly one** ambient-environment-derived path: the wrapper's
  `mktemp "${TMPDIR:-/tmp}/…"` (line 59). It writes the validator→producer handoff JSON **out-of-repo**.
- The **validator writes it unguarded** (`validate.py:186` raw `Path.write_text`) → succeeds.
- The **producer reads it guarded** (`producer.py:119 → common.py:143 → canonical_root_path`,
  in-repo-only) → `RuntimeError: path escapes repository root` → job dies at its first action.
- **Both Python `tempfile.mkstemp` call sites** (`common.py:195`, `independent_verify.py:796`) pass an
  **explicit `dir=<in-repo artifact parent>`**, so they **never consult `$TMPDIR`** — there is **no
  second temp-path landmine**. Every other path in the chain is in-repo.
- Handoff table (§4): **1 row FAIL** (the defect), **10 rows PASS**, **0 UNPROVABLE**. After fix (a)
  the FAIL row becomes PASS and the table is clean.
- The failure isolates cleanly: the traceback died at `producer.py:119`, which is **after**
  `require_slurm_realbank`, `read_json(config)`, `verify_config_and_schema`, `verify_machine_realbank`,
  and the no-clobber loop — so **all of those asserts executed and passed at runtime**. The config ↔
  machine-plan ↔ code-constant bindings are therefore not merely statically consistent but
  **runtime-proven consistent** up to line 119.

---

## 1. sbatch line-by-line (`…_m0_realbank_resource_v1.sbatch`)

| line | statement | execution-path judgment |
|---|---|---|
| 1 | `#!/usr/bin/env bash` | shebang. — |
| 2 | `#SBATCH --partition=slurmpartition` | valid partition (authz §5 `sinfo` up). PASS |
| 3 | `#SBATCH --cpus-per-task=16` | → `SLURM_CPUS_PER_TASK=16`; matches `require_slurm_realbank` (16). PASS |
| 4 | `#SBATCH --mem=96G` | → `SLURM_MEM_PER_NODE` ∈ allowed `{98304,96000,96}`. **Runtime-proven PASS** (validator's `require_slurm_realbank` passed in job 12994). |
| 5 | `--job-name=lbscgp_global_r2_realbank_resource_v1` | single-submit ledger key. PASS |
| 6-7 | `--output/--error=…/%x_%j.{out,err}` | ABSOLUTE in-repo `slurm/logs/`. Wrote `…_12994.{out,err}`. PASS |
| 8 | no `--time` (comment) | project policy. PASS |
| 9 | `set -euo pipefail` | fail-closed. PASS |
| 11 | `cd /data/jehc223/RGCL` | CWD = ROOT (absolute). Anchors all later relative paths in-repo. PASS |
| 12-13 | `source …/conda.sh; conda activate HateVideo` | sets `CONDA_DEFAULT_ENV=HateVideo` (guarded downstream). Runtime-proven PASS. |
| 14-19 | `export PYTHONUNBUFFERED/HF_HUB_OFFLINE/WANDB_MODE/OMP_/MKL_/OPENBLAS_NUM_THREADS` | thread vars are the **R-2 determinism precondition** (16). All are **set**, not read from ambient. `HF_HUB_OFFLINE=1` is inert here (no HF/network call in this run). PASS |
| 21-22 | `RUN_ID=…; CONFIG=configs/…v1.json` | RUN_ID = `RUN3` literal; CONFIG = relative in-repo (resolves under CWD=ROOT). PASS |
| 24 | `RUN_ID=… CONFIG=… bash scripts/wrappers/…v1.sh` | passes RUN_ID/CONFIG as env into the wrapper; relative wrapper path in-repo. PASS |

**sbatch verdict: PASS.** `$TMPDIR` is **not** set/exported by the sbatch — it is inherited from the
SLURM job environment (`/data/jehc223/home/tmp`). No env var here builds a path except CWD (=ROOT,
correct). The thread vars are inherited by the wrapper's Python children (R-2 precondition holds).

## 2. Wrapper line-by-line (`…_realbank_resource_v1.sh`)

| line | statement | absolute-path ownership / judgment |
|---|---|---|
| 2 | `set -euo pipefail` | fail-closed; `-u` requires all vars defined. PASS |
| 4 | `cd /data/jehc223/RGCL` | CWD = ROOT. PASS |
| 6-7 | `RUN_ID=${RUN_ID:-…}; CONFIG=${CONFIG:-configs/…}` | inherited from sbatch; defaults are `RUN3`/in-repo. PASS |
| 8 | `EXPECTED=…REALBANK-RESOURCE-v1` | code constant. — |
| 9 | `ARTIFACT_ROOT=artifacts/lb_scgp_global/v1/m0/realbank_resource` | **relative in-repo** (under CWD=ROOT). PASS |
| 11 | `VALIDATION_JSON=""` | init empty → trap guard `[[ -n … ]]` skips rm if unset. PASS |
| 12-21 | `PROSPECTIVE_OUTPUTS=("$ARTIFACT_ROOT/…" ×8)` | 4 artifacts + 4 `.publish.lock`, all relative in-repo. PASS |
| 23-32 | `cleanup_on_exit()` | on EXIT: `rm -f "$VALIDATION_JSON"` (if non-empty), and if `COMPLETE≠1` `rm -f "${PROSPECTIVE_OUTPUTS[@]}"`. Removes the temp handoff + any prospective output. **Verified post-mortem: worked** (temp gone, artifact dir absent). PASS |
| 34-41 | `signal_exit`; `trap … EXIT/HUP/INT/TERM` | traps armed before any output. PASS |
| 43-46 | `if RUN_ID != EXPECTED → exit 2` | assert L=env RUN_ID, R=code constant. PASS |
| 48 | `CONFIG_RUN_ID=$(jq -r '.run.run_id' "$CONFIG")` | **reads CONFIG (in-repo)** via jq (no repo guard needed; in-repo by construction). PASS |
| 49 | `CONFIG_ARTIFACT=$(jq -r '.run.artifact_path' "$CONFIG")` | in-repo read. PASS |
| 50-53 | `if CONFIG_RUN_ID != RUN_ID → exit 2` | assert L=config field, R=env RUN_ID. PASS |
| 54-57 | `if CONFIG_ARTIFACT != "artifacts/…/decision.json" → exit 2` | assert L=config field, R=code literal. PASS |
| **59** | **`VALIDATION_JSON=$(mktemp "${TMPDIR:-/tmp}/lbscgp_…_validation.XXXXXX.json")`** | **DEFECT ORIGIN.** `mktemp` template's directory = `$TMPDIR` (`/data/jehc223/home/tmp`) → the created path is **OUT-OF-REPO**. The fallback `/tmp` is **also** out-of-repo, so the path escapes the repo under *any* `$TMPDIR`. This is the **only** ambient-env-derived path in the whole chain. **FAIL** |
| 60-63 | `python …_validate.py --config "$CONFIG" --run-id "$RUN_ID" --json-out "$VALIDATION_JSON"` | validator writes to the out-of-repo `$VALIDATION_JSON`; args are in-repo config + RUN3 + out-of-repo json-out. Validator **succeeded** (unguarded write, §3). |
| 65-68 | `python …_producer.py --config "$CONFIG" --run-id "$RUN_ID" --validation-json "$VALIDATION_JSON"` | producer **reads** the out-of-repo `$VALIDATION_JSON` via guarded `read_json` → **RuntimeError** (§3). Job dies here. **FAIL propagates** |
| 70-73 | `python …_independent_verify.py --config "$CONFIG" --manifest "$ARTIFACT_ROOT/decision.json" --out "$ARTIFACT_ROOT/…semantic_verification.json"` | (never reached) all args in-repo relative. Static PASS |
| 75 | `jq -e '.decision == "PASS"' "$ARTIFACT_ROOT/…json"` | (never reached) in-repo read. Static PASS |
| 76 | `COMPLETE=1` | (never reached) → trap kept `COMPLETE=0` → prospective outputs cleaned. PASS |

**Wrapper verdict: FAIL at line 59** (single defect). Every other wrapper path is in-repo. The
three-stage call ordering, the exit gates, and the cleanup trap are all correct; the trap's fail-closed
behavior is post-mortem-verified.

## 3. Python modules — every path op, env read, subprocess, assert

### 3a. `validate.py` (ran to completion, exit 0)

- `sys.path.insert(0, str(ROOT/"scripts/analysis"))` (L13): absolute in-repo import path. PASS.
- `require_slurm_realbank()` (L152, in common): env reads = `SLURM_JOB_ID`, `CONDA_DEFAULT_ENV`,
  `SLURM_CPUS_PER_TASK`, `SLURM_MEM_PER_NODE|_PER_CPU`, `SLURM_GPUS*` — **all are guards** (compared to
  expected), **none builds a path**. Runtime-proven PASS (validator exited 0).
- `if args.config != CONFIG_PATH: raise` (L153): assert L=CLI arg, R=code constant. PASS.
- `cfg = read_json(args.config)` (L155): `read_json → canonical_root_path` in-repo. PASS.
- `run_command(["jq","-e",".",rel])` ×4 (L164) over `[config, machine, schema, contract_freeze]`,
  `cwd=ROOT`: in-repo relative. PASS. `bash -n wrapper/slurm` (L167): in-repo. PASS.
  `python -m py_compile files` (L169): in-repo. PASS. `git diff --check -- files + tracker` (L170):
  **bounded pathspec** = implementation files + tracker. PASS.
- `schema_strict_check → read_json(schema)` (L56): in-repo. PASS.
- `scan_trailing_whitespace → open(ROOT/rel)` (L46): in-repo files. PASS.
- `verify_run1_hashes` (L66,69): `sha256_file(ROOT/rel)` + `old_protected_hash_manifest()`: in-repo.
  PASS. `verify_authoritative_hashes` (L80): `sha256_file(ROOT/rel)`: in-repo. PASS.
  `verify_train_bank_bindings` (L91): `sha256_file(ROOT/bank["path"])` — hashes the `.pt` **bytes**
  (not labels), in-repo. PASS.
- `no_clobber_check` (L106): `canonical_root_path` on the 4 artifact paths + `allowed_new_files`:
  in-repo. PASS.
- `sha256_file("scripts/analysis/…validate.py")` (L183): relative, cwd=ROOT. PASS.
- `os.environ.get("SLURM_JOB_ID","")` (L184): record only, not a path. PASS.
- **`Path(args.json_out).write_text(canonical_json(result)+"\n")` (L186): RAW path, NO
  `canonical_root_path`.** Writes to `$VALIDATION_JSON` = out-of-repo → **SUCCEEDS unguarded**. This is
  the **write half of the asymmetry**: the validator will happily write anywhere, so it did not catch
  the bad path; the producer's guarded read is what failed. Not itself a safety breach (it writes its
  own output), but it is why the defect surfaced one process late. After fix (a) this path is in-repo,
  making the write/read symmetric. **Noted (optional hardening in §5.4).**

### 3b. `producer.py` (died at L119)

- `require_slurm_realbank()` (L101): env guards only (as 3a). Runtime PASS.
- `assert_equal(args.run_id, RUN3)` (L102): L=CLI, R=constant. Runtime PASS.
- `cfg = read_json(args.config)` (L104): `canonical_root_path` in-repo. Runtime PASS.
- `assert_equal(args.config, CONFIG_PATH)` (L105): L=CLI, R=constant. Runtime PASS.
- `verify_config_and_schema(cfg)` (L106): asserts every `authorization.*` flag (L=config, R=`False`
  except `train_bank_read_allowed=True`), run/schema/artifact literals, and schema-strictness via
  `read_json(schema)` (in-repo). **Runtime PASS** (reached L119 afterward).
- `verify_machine_realbank(cfg)` (L107): `read_json(machine)` + `sha256_file(canonical_root_path(machine))`
  (in-repo); asserts `run_order[3]`,`runs[3].run_id`,`artifact_paths`,`artifact_schema_ids`,`slurm`,
  `dependencies`, and per-dataset bank `path/sha256/train_n` — each L=machine-plan field, R=config field
  or `EXPECTED_N` constant. **Runtime PASS** (reached L119). This is strong evidence the machine↔config
  bindings are internally consistent.
- no-clobber loop (L109-117): `canonical_root_path` on 4 artifact paths (in-repo). Runtime PASS.
- **`validation = read_json(args.validation_json)` (L119): `canonical_root_path($TMPDIR/…json)` →
  `resolved.relative_to(ROOT)` raises `ValueError` → wrapped `RuntimeError: path escapes repository
  root`. THE DEFECT (read half). FAIL.** Died here; nothing below executed.
- *(below L119, static-only, never reached):* `ledger.open_train_bank` (L133) `canonical_root_path`
  +allowlist+sha, `load_bank_features → torch.load(fs_path)` (L134, in-repo canonical path);
  `build_source_manifest` (L192) `hash_source_file`(refuses `data/`)/`old_protected_hash_manifest`(rglob
  in-repo roots)/`implementation_hashes`(`canonical_root_path`)/`relevant_git_status`(bounded pathspec);
  `sha256_file(canonical_root_path(…))` hashes (L271-277); `exclusive_publish_json` ×3 (L289-293)
  `canonical_root_path` + `tempfile.mkstemp(dir=in-repo parent)`. **All in-repo → static PASS.**
- `os.environ.get("SLURM_JOB_ID","")` (L214): record only. PASS.

### 3c. `independent_verify.py` (never reached; self-contained, no import of common/producer)

- own guard `root_path()` (L96-101): `resolved.relative_to(ROOT.resolve())` — equivalent in-repo-only
  hardening. Every `read_json`/`sha256_file`/`publish_json` routes through it.
- `if not os.environ.get("SLURM_JOB_ID"): raise` (L818): env guard, not a path. PASS.
- `read_json(args.config)` (L820), `read_json(args.manifest)` (L821, `args.manifest`=
  `ARTIFACT_ROOT/decision.json`, in-repo relative from wrapper): `root_path` in-repo. PASS.
- `derive_datasets` (L473-476): `root_path(bank["path"])` + `sha256_file` + `load_bank_features →
  torch.load`: in-repo. PASS. `verify_machine` (L482): `read_json(machine)`: in-repo. PASS.
- `verify_source_and_access` (L664-709): `sha256_file(root_path(…))` on source/access/run1/
  authoritative/impl/schema: in-repo. PASS.
- **`publish_json(args.out)` (L783-808): `root_path(out)` + `tempfile.mkstemp(prefix=…,
  dir=str(fs_path.parent))`** — `args.out`=`ARTIFACT_ROOT/…semantic_verification.json` (in-repo), and
  `dir=` is the in-repo artifact parent → **`mkstemp` does NOT consult `$TMPDIR`.** PASS. (This is the
  direct contrast that proves the wrapper's `mktemp` is the *only* TMPDIR-sensitive site.)
- `sha256_file(root_path(args.manifest))` (L831,845): in-repo. PASS.

### 3d. `common.py` shared primitives

- `canonical_root_path(path)` (L130-139): `root=ROOT.resolve()`; `candidate = raw if absolute else
  root/raw`; `resolved=candidate.resolve()`; `resolved.relative_to(root)` — **in-repo-only guard**; on
  `ValueError` raises `RuntimeError("path escapes repository root")`. This is the guard that fired.
  Correct and load-bearing (it underpins the access-ledger/isolation "never reads outside repo"
  property). **Do not weaken it** (see §5.2).
- `read_json` (L142-145) = `canonical_root_path` then `open`. All producer/common reads are in-repo.
- `exclusive_publish_json` (L182-208): `canonical_root_path` + `tempfile.mkstemp(dir=str(fs_path.parent))`
  → **explicit in-repo dir, no `$TMPDIR` consult.** PASS.
- `require_slurm_realbank` (L416-430): env reads are guards only; **none builds a path**. PASS.
- `old_protected_hash_manifest` (L765-793): `rglob("*")` over roots `configs/lb_scgp`,
  `artifacts/lb_scgp`, `refine-logs/lb_scgp`, `scripts/analysis`, `scripts/slurm`. **Load-bearing for
  the fix**: a transient temp file living under any of these roots during
  `build_source_manifest` would perturb the manifest hash → `assert_equal(old_hash,…)` fail. The fix's
  temp dir must avoid all five (see §5.1). PASS as-is (the current out-of-repo temp does not perturb it;
  but that is accidental, not designed).
- `relevant_git_status` (L804-812): `git status --porcelain -- <relevant_paths>` — **bounded pathspec**
  = implementation + authoritative + run1 files. A temp file elsewhere is invisible. PASS.

**Env-read inventory (whole chain):** the complete set of `os.environ`/`getenv` reads is
`SLURM_JOB_ID`, `CONDA_DEFAULT_ENV`, `SLURM_CPUS_PER_TASK`, `SLURM_MEM_PER_NODE`, `SLURM_MEM_PER_CPU`,
`SLURM_GPUS`, `SLURM_GPUS_ON_NODE`, `SLURM_STEP_GPUS`, `SLURM_JOB_GPUS` — **all guards** — plus the
shell's `$TMPDIR` (wrapper L59) and `$RUN_ID`/`$CONFIG` (wrapper L6-7). **Only `$TMPDIR` derives a
path.** (Confirmed by `grep -nE 'os\.environ|getenv' + mktemp|tempfile|TMPDIR|gettempdir` over all 6
executable entities.)

## 4. Full-chain handoff table

Every inter-process file handoff: step → writer(site) → write path → write-guard → reader(site) →
read-guard → verdict. (In-repo = under `/data/jehc223/RGCL`.)

| # | artifact | writer @ site | write path | write-guard | reader @ site | read-guard | verdict |
|---|---|---|---|---|---|---|---|
| **A** | **validation JSON (handoff)** | validate.py:186 `Path.write_text` | **`$TMPDIR/…json` OUT-OF-REPO** | **none (raw)** | producer.py:119 `read_json` | `canonical_root_path` (in-repo only) | **FAIL** → after (a): in-repo → **PASS** |
| B | config json | frozen (prep) | `configs/…v1.json` (in-repo) | — | wrapper:48-49 jq; validate:155; producer:104; verify:820 | jq(in-repo)/`canonical`/`root_path` | PASS (runtime-proven to producer:119) |
| C | machine plan | frozen | `refine-logs/…machine.json` | — | producer:107; verify:482 | `canonical`/`root_path` | PASS (runtime-proven at producer:107) |
| D | payload schema | frozen | `schemas/…v1.schema.json` | — | validate:165; producer:77; verify:453 | `canonical`/`root_path` | PASS (runtime to producer:106) |
| E | train banks ×2 (`.pt`) | frozen data | `data/CLIP_Embedding/…` | — | producer:133-134; verify:473-476 | `canonical`+allowlist+sha / `root_path`+sha | PASS (static; not reached in run) |
| F | run1 artifact + lock | frozen (Run1) | `artifacts/…/contract_freeze.json(.lock)` | — | producer:276-277; validate:66-69; verify:706-708 | `canonical`/`ROOT`-join/`root_path` | PASS |
| G | source_manifest.json | producer:289 `exclusive_publish_json` | `artifacts/…/source_manifest.json` | `canonical`+`mkstemp(dir=in-repo)` | verify:664,668 | `root_path` | PASS (static) |
| H | access_ledger.json | producer:291 `exclusive_publish_json` | `artifacts/…/access_ledger.json` | `canonical`+`mkstemp(dir=in-repo)` | verify:666,669 | `root_path` | PASS (static) |
| I | decision.json (manifest) | producer:293 `exclusive_publish_json` | `artifacts/…/decision.json` | `canonical`+`mkstemp(dir=in-repo)` | wrapper:72(arg); verify:821 `read_json` | `root_path` | PASS (static) |
| J | semantic_verification.json | verify:837 `publish_json` | `artifacts/…/semantic_verification.json` | `root_path`+`mkstemp(dir=in-repo)` | wrapper:75 jq -e | jq(in-repo) | PASS (static) |
| K | producer/verifier atomic tempfiles | common:195 / verify:796 `mkstemp` | `dir=<in-repo artifact parent>` | **`dir=` explicit → `$TMPDIR` NOT consulted** | internal (`os.link`→`unlink`) | — | PASS (proves A is the only TMPDIR site) |

**Tally: 1 FAIL (row A), 10 PASS, 0 UNPROVABLE.** The one runtime-environment fact that static review
originally lacked (`$TMPDIR`) is now known from the `.err`, so no row is unprovable. After fix (a), row
A is PASS and the whole table is clean.

## 5. Fix specification — direction (a), CONFIRMED

Fix (a) = write the validation handoff JSON into a **repo-internal** temp directory; **do not** loosen
`canonical_root_path`/`root_path`. Confirmed correct below, with the exact diff and a gate-by-gate
compatibility proof for the chosen directory.

### 5.1 Chosen in-repo temp dir: `slurm/tmp/` (verified against every gate)

Requirements: must (i) be **outside** the five `old_protected` roots (else a transient temp file
perturbs the manifest hash), (ii) not be in any `no_clobber` / `allowed_new_files` / forbidden list,
(iii) not fall inside the `git diff --check` / `relevant_git_status` bounded pathspecs, (iv) not start
with `data/` and contain no forbidden token, (v) already exist or be cheap to create.

`slurm/tmp/` (distinct from the `scripts/slurm` old-protected root) satisfies all of them:

| gate | mechanism | `slurm/tmp/` result |
|---|---|---|
| `old_protected_hash_manifest` (common:765) | rglob over `configs/lb_scgp`,`artifacts/lb_scgp`,`refine-logs/lb_scgp`,`scripts/analysis`,`scripts/slurm` | **not under any** (`slurm/` ≠ `scripts/slurm/`) → manifest hash unperturbed. PASS |
| `no_clobber_check` (validate:98) + producer no-clobber (producer:109) | explicit 4 artifact paths + `allowed_new_files_after_run` | `slurm/tmp/…` in **none**. PASS |
| `git diff --check -- files+tracker` (validate:170) | bounded pathspec = impl files + tracker | `slurm/tmp/…` not in pathspec; untracked temp not diffed. PASS |
| `relevant_git_status` (common:804) | `git status --porcelain -- <relevant_paths>` | `slurm/tmp/…` not in pathspec → invisible. PASS |
| `forbidden_reason` / FORBIDDEN_TOKENS | substring `query_z/query_labels/teacher/cache/held/certificate`, and `data/` prefix | validation temp is never passed to the ledger; path has no token and rel ≠ `data/…`. N/A → PASS |
| `.gitignore` | patterns incl. `artifacts`, `data`, `*.out` | no pattern matches `slurm/tmp/`; but the file is `rm`'d on exit and git ignores empty dirs → **no git pollution**. (Optional: add `slurm/tmp/` to `.gitignore` as belt-and-suspenders.) PASS |

(Alternative equally-safe dir: a repo-root `.tmp_realbank/`. `slurm/tmp/` is recommended because
`slurm/` already exists and co-locates with `slurm/logs/`.)

### 5.2 Why NOT (b) loosen the path hardening — rejected, with reason

(b) = permit `read_json`/`canonical_root_path` to read the specific out-of-repo validation temp path.
**Rejected.** `canonical_root_path` (common) and `root_path` (verifier) are the load-bearing invariant
behind the whole access-ledger / isolation-injection story ("this run never reads outside the repo").
Punching a hole — even a narrow, path-specific one — (1) creates a bypass surface that every future
auditor must re-reason about, (2) must be duplicated **consistently across two modules** (common +
verifier) or they diverge, and (3) trades a one-line plumbing fix for a weakening of the exact
safety property the ceremony exists to protect. (a) keeps the invariant fully intact and only moves the
handoff file inside the repo. **(a) strictly dominates (b).**

### 5.3 Why NOT (c) stream the handoff / (d) export global `$TMPDIR` — rejected

- (c) pass validation JSON via stdout/pipe (no temp file): changes the validator↔producer **interface**
  across 2-3 entities → larger re-review/re-freeze surface for no additional safety. Rejected as
  over-scoped.
- (d) `export TMPDIR=<in-repo>` in the wrapper and keep `mktemp "${TMPDIR:-/tmp}/…"` unchanged: works,
  but redirects `$TMPDIR` for the **entire** Python subprocess tree (incl. torch's scratch/inductor
  paths) — a broader blast radius than the single handoff. Rejected in favor of the targeted path.

### 5.4 Exact diff (blast radius = wrapper entity #7 only)

`mktemp` does **not** create parent directories, so the fix is the one-line template change **plus** a
provisioning `mkdir -p`. Replace wrapper line 59:

```diff
-VALIDATION_JSON=$(mktemp "${TMPDIR:-/tmp}/lbscgp_global_r2_realbank_resource_v1_validation.XXXXXX.json")
+REALBANK_TMPDIR="/data/jehc223/RGCL/slurm/tmp"
+mkdir -p "$REALBANK_TMPDIR"
+VALIDATION_JSON=$(mktemp "$REALBANK_TMPDIR/lbscgp_global_r2_realbank_resource_v1_validation.XXXXXX.json")
```

- **Cleanup trap already covers it:** `cleanup_on_exit` does `rm -f "$VALIDATION_JSON"` (line 26),
  which removes the file on every exit path; the empty `slurm/tmp/` dir is git-invisible. (Optional:
  append `rmdir "$REALBANK_TMPDIR" 2>/dev/null || true` to the trap to remove the dir when empty —
  harmless, single-job so no race.)
- **No config edit forced:** the config lists the wrapper in `implementation_files` (paths, not
  hashes) and in `paths.wrapper`, but binds **no frozen hash** of the realbank wrapper anywhere
  (`hash_bindings` covers Run1 entities, authoritative docs, and the old-protected manifest — not this
  wrapper; and `old_protected` does not scan `scripts/wrappers/`). The pipeline hashes
  `implementation_files` at **runtime** and cross-checks producer↔verifier self-consistently, so it
  hard-codes no wrapper hash. Only the **freeze ceremony** records the wrapper SHA, so re-freeze
  updates exactly one value (entity #7).
- **Optional defense-in-depth (do NOT bundle into v2):** route `validate.py:186`'s `--json-out` write
  through `canonical_root_path` so writer and reader enforce the same invariant. This would change
  entity #4 too, widening the re-freeze/re-review surface; keep v2 minimal (wrapper only) and track
  this separately.

**After fix (a):** wrapper line 59-region yields an **in-repo** `$VALIDATION_JSON`; validate.py writes
it in-repo; producer's `read_json → canonical_root_path` resolves it under ROOT → **PASS**. Handoff
row A flips to PASS; the chain has **no** remaining out-of-repo path and **no** ambient-env-derived
path.

## 6. v2-clone inherited environment-interaction audit (other TMPDIR-class mines)

Beyond `$TMPDIR`, I enumerated every other ambient-environment interaction the v2 clone inherits, to
confirm no second landmine hides behind the first:

| env / interaction | where consulted | can it move a file / break the run? | verdict |
|---|---|---|---|
| `$TMPDIR` | wrapper:59 `mktemp` | **YES — the defect.** | **FIXED by §5.4** |
| Python `tempfile` default dir (`gettempdir` → `$TMPDIR`) | common:195, verify:796 `mkstemp` | **No** — both pass explicit `dir=<in-repo>`; `$TMPDIR` never consulted | PASS (no change needed) |
| `HOME` / `XDG_CACHE_HOME` | torch/numpy/jsonschema import; `torch.load(weights_only=True)` of a local `.pt` | No file handoff depends on it; no compilation; env proven importable (`M0_ENV_REPAIR_RECORD.md`) | PASS |
| locale (`LC_ALL`/`LANG`) | JSON `sort_keys=True`, `sorted(key=…)`, `np.argsort(kind="mergesort")`, jq | Python/np string order is codepoint-based (locale-independent); jq does no locale sort here | PASS (no determinism exposure) |
| thread vars `OMP/MKL/OPENBLAS_NUM_THREADS` | set in sbatch:17-19; inherited by Python | **Not a path**, but the **R-2 replay-determinism precondition**. Set correctly. *Residual:* `require_slurm_realbank` does **not** assert they equal 16, so a future clone that drops the sbatch exports would still pass preflight yet risk a cross-process replay `FAIL`. Pre-accepted R-2 risk, **not** this bug; optional future hardening = assert thread vars in `require_slurm_realbank`. | PASS (flagged, non-blocking) |
| `HF_HUB_OFFLINE`, `WANDB_MODE`, `PYTHONUNBUFFERED` | sbatch exports | inert this run (no HF/wandb/network call) | PASS |
| `CONDA_DEFAULT_ENV`, `SLURM_*` | `require_slurm_realbank` guards | guards only; never build a path | PASS |
| `PYTHONPATH` | validate/producer `sys.path.insert(ROOT/"scripts/analysis")` | absolute in-repo, not ambient-derived | PASS |
| CWD | sbatch:11 + wrapper:4 `cd ROOT` | anchors relative paths in-repo; set explicitly | PASS |

**Conclusion:** after fix (a) the realbank chain has **zero** ambient-environment-derived file paths
and **one** flagged non-path determinism dependency (thread vars, pre-accepted R-2). The
"runtime file-handoff path + environment variable" class that burned four ceremonies is, for realbank,
fully modeled and closed.

---

## Required statements

- No performance evidence exists and none is claimed; this audit is static and read-only. It reproduces
  no accuracy/macro-F1 and runs no training/kNN/model/MLLM/OCR/network/GPU.
- The only project gold is `parent_video_binary_label`; no segment/frame/span/localization/stance/
  target/mechanism/rationale/fragment gold is assumed or introduced. No train label is read anywhere in
  the chain (the banks are opened as feature `.pt` files, hash-checked; `load_bank_features` explicitly
  does not read `labels`). The `is_science=false` structural placeholder certifies nothing.
- Run4 (M1 cache), MLLM/cache, validation/test, and training remain **locked**; this audit unlocks
  nothing. v2 opens only via the six-step gate in the companion result-to-claim review.
- Auditor = Claude Opus 4.8, fresh independent full-chain-audit role, separate from realbank-prep/
  freeze, amendment/code-review, authorization, executor, and result-to-claim roles. Wrote only this
  document and the companion result-to-claim review; edited no code/config/schema/plan; ran no job.
