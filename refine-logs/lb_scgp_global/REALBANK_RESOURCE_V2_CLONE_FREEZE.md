# M0 REALBANK-RESOURCE-v2 Clone / Freeze (v1 $TMPDIR-burn fix)

Date: 2026-07-13

Author: **Claude Opus 4.8**, **realbank-prep role only** — separate from the independent amendment
reviewer, the fresh 0C/0H realbank code reviewer, the execution authorizer, and the executor. This
document clones the eight v1 realbank entities to v2, applies the audit-confirmed fix (a), and freezes
it. It authorizes no execution.

Discipline: static clone + one targeted wrapper fix + freeze. No project Python was executed, no
`py_compile`, no import, no conda/SLURM/`sbatch`/`squeue`, no experiment, no MLLM/OCR/API/network/
model/GPU/training/evaluation, and no validation/test/cache content was touched. `jq`, `bash -n`,
`sha256sum`, `sed`/`diff`/`grep`, `git status` were read-only (`sed` only to generate/prove the
byte-transform). No artifact under `artifacts/lb_scgp_global/v1/m0/realbank_resource/` was created
(absence confirmed below). Nothing was committed to git; no SLURM job was submitted.

---

## 0. Why v2

The realbank **v1** single submit (job `12994`) died **preflight** at `producer.py:119`:
`REALBANK_FULLCHAIN_STATIC_AUDIT.md` proves the sole defect is the wrapper's
`mktemp "${TMPDIR:-/tmp}/…"` (line 59), which wrote the validator→producer handoff JSON **out-of-repo**
(`$TMPDIR=/data/jehc223/home/tmp`); the producer reads it through the in-repo-only
`canonical_root_path` guard → `RuntimeError: path escapes repository root`. The v4 producer used a
plain `open()` there; the realbank producer hardened it, which is what tripped. No artifact was
published; the burn is preserved as refine-logs evidence, not a run-array entry.

Fix direction **(a)** (audit §5, CONFIRMED, strictly dominates (b)/(c)/(d)): keep the load-bearing
in-repo guard intact and move the handoff file **inside** the repo to `slurm/tmp/`. Audit §5.1 verified
`slurm/tmp/` is outside all five `old_protected` roots and every `no_clobber` / `git diff --check` /
`relevant_git_status` bounded pathspec, so the transient (trap-cleaned) file perturbs no gate.

Per the coordination ruling and the lead's decision tree — `runs[3].run_id` **contains** `-v1` and the
code asserts it (`RUN3`, `verify_machine_realbank`, producer, wrapper) — v2 is a synchronized run-id
bump (v3-lesson: code constant and machine content move in lock-step), matching the synth-kkt
v1→…→v4 convention.

## 1. v2 entities (8) — SHA256

| # | entity | SHA256 |
|---|---|---|
| 1 | `configs/lb_scgp_global_r2/m0_realbank_resource_v2.json` | `1d69b961c68163163ececb9e75ded4630661d2e276bbcc63302d7f76cbf26b9d` |
| 2 | `schemas/lb_scgp_global_r2/scgp_global_realbank_resource_v2.schema.json` | `4d95d128e6f7369f8484983a50acac7d4a57dcc3265456f183bc5bc545fb0271` |
| 3 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v2_common.py` | `f90f153faf10bc6a2e14bbb8bdab1835bec2e8bfafbe8bc65287f72a35a00fd8` |
| 4 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v2_validate.py` | `ea703b3e97e7d99f080d6398f79a7d66dbbe8780bdca65eb949c27b38bab3a91` |
| 5 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v2_producer.py` | `e5e9a06a2297561b5a898ff99cb30baf1f97ba1b7ce20e22408d9c8dce3e0dab` |
| 6 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v2_independent_verify.py` | `7ffa860b8f5260c881586121a7686cb569f55c2d5a0522954f158639b8b5d8e1` |
| 7 | `scripts/wrappers/lb_scgp_global_r2_realbank_resource_v2.sh` | `348b056b1d12a1cd8f1940c90edd417603ae79afe63150610fa54cde431c1419` |
| 8 | `scripts/slurm/lb_scgp_global_r2_m0_realbank_resource_v2.sbatch` | `d7ab1e75a1d9e58f28651ae4479bd02b2d29485836d418c0eaf0633963153833` |

Machine amendment (REPLACE-in-place at `runs[3]`; array length 66 and downstream indices unchanged):
run_id and `artifact_schema_ids` v1→v2 across all six run-id sites (run_order[3], runs[3].run_id,
runs[4]/runs[5] deps, terminal_decision_chain, g0_runs_through_v4_supplement) + the two schema-id sites
(runs[3], artifact_schemas registry); status → `V2_CLONE_FROZEN_AFTER_V1_TMPDIR_BURN_PENDING_INDEPENDENT_REVIEW`.
Plan cascade: machine `d5023b62…`→`f4d54b78…`; `EXPERIMENT_PLAN.md` `10fd5232…`→`a3325f9d…`;
`EXPERIMENT_TRACKER.md` `d226abfe…`→`51367c17…`; `EXPERIMENT_PLAN_HASHES.sha256`→`d3251b0b…`.
Pre-v2 plan backup: `EXPERIMENT_PLAN.machine.json.pre_realbank_v2_amendment.bak` (`d5023b62…`).
v1 entities are **not** deleted (burn evidence). Artifact namespace `lb_scgp_global/v1/m0/realbank_resource/`
and `train_banks`, A/B/C protocol are **unchanged**.

## 2. Diff vs v1 = (version-token rename) + (3-line wrapper fix), proven

- **Five code/schema files (schema, common, validate, producer, verify):** byte-identical to a guarded
  `v1→v2` token substitution — `diff <(guarded_sed(v1)) v2` is **empty** for all five (verified this
  session). Guard preserves the artifact namespace `lb_scgp_global/v1/`, the run1 `CONTRACT-FREEZE-v1`
  / `contract_freeze_v1` tokens, `SYNTH-KKT-v4`, `scgp_global_cert_v2`, and the `config_v1` format
  suffix; every realbank-lineage `-v1`/`_v1` (run_id, `scgp_global_realbank_resource_v1`,
  `…manifest_v1`, `…source_manifest_v1`, `…semantic_verification_v1`, `…validation_v1`, file/module
  names, `CONFIG_PATH`) → v2. **Zero behavioral change** in these five.
- **sbatch:** diff = job-name + `CONFIG` + wrapper path token renames only. No resource/env change
  (still 16 CPU / 96 GB / 0 GPU, `OMP/MKL/OPENBLAS=16`, no `--time`).
- **wrapper:** diff = `RUN_ID`/`CONFIG`/`EXPECTED` + three script-path token renames, **plus** the sole
  behavioral change — the audit §5.4 fix (a):

  ```diff
  -VALIDATION_JSON=$(mktemp "${TMPDIR:-/tmp}/lbscgp_global_r2_realbank_resource_v1_validation.XXXXXX.json")
  +REALBANK_TMPDIR="/data/jehc223/RGCL/slurm/tmp"
  +mkdir -p "$REALBANK_TMPDIR"
  +VALIDATION_JSON=$(mktemp "$REALBANK_TMPDIR/lbscgp_global_r2_realbank_resource_v2_validation.XXXXXX.json")
  ```
  (plus a 5-line explanatory comment). The temp handoff now lives at `slurm/tmp/…` (in-repo).

## 3. Full-chain handoff table (audit §4, post-fix — row A flips to PASS)

| # | artifact | writer @ site | write path | reader @ site | read-guard | verdict |
|---|---|---|---|---|---|---|
| **A** | **validation JSON (handoff)** | validate:186 `write_text` | **`slurm/tmp/…json` IN-REPO (fixed)** | producer:119 `read_json` | `canonical_root_path` | **PASS** (was FAIL) |
| B | config json | frozen (prep) | `configs/…v2.json` | wrapper:48-49 jq; validate:155; producer:104; verify:820 | jq / `canonical` / `root_path` | PASS |
| C | machine plan | frozen | `refine-logs/…machine.json` | producer:107; verify:482 | `canonical` / `root_path` | PASS |
| D | payload schema | frozen | `schemas/…v2.schema.json` | validate:165; producer:77; verify:453 | `canonical` / `root_path` | PASS |
| E | train banks ×2 (`.pt`) | frozen data | `data/CLIP_Embedding/…` | producer:133-134; verify:473-476 | `canonical`+allowlist+sha / `root_path`+sha | PASS |
| F | run1 artifact + lock | frozen (Run1) | `artifacts/…/contract_freeze.json(.lock)` | producer; validate; verify | `canonical` / `ROOT`-join / `root_path` | PASS |
| G | source_manifest.json | producer `exclusive_publish_json` | `artifacts/…/source_manifest.json` | verify | `root_path` | PASS |
| H | access_ledger.json | producer `exclusive_publish_json` | `artifacts/…/access_ledger.json` | verify | `root_path` | PASS |
| I | decision.json (manifest) | producer `exclusive_publish_json` | `artifacts/…/decision.json` | wrapper(arg); verify `read_json` | `root_path` | PASS |
| J | semantic_verification.json | verify `publish_json` | `artifacts/…/semantic_verification.json` | wrapper jq -e | jq | PASS |
| K | producer/verifier atomic tempfiles | common/verify `mkstemp` | `dir=<in-repo artifact parent>` | internal (`os.link`→`unlink`) | `dir=` explicit → `$TMPDIR` NOT consulted | PASS |

**Tally: 11/11 PASS.** The one out-of-repo path (row A) is now in-repo; no ambient-environment-derived
file path remains (audit §6: no second landmine; thread-vars R-2 pre-accepted, non-blocking).

## 4. Runtime cross-check static-simulation table (21 rows, re-verified for v2)

Identical in structure to `REALBANK_RESOURCE_V1_FREEZE.md` §3, re-verified against the v2 tokens this
session — all PASS or DEFERRED-TO-RUNTIME. Re-confirmed rows: v2 config valid JSON, hash
`1d69b961…`; **no trailing whitespace** on all 8 v2 entities; **schema.required == verifier `TOP_KEYS`
== producer manifest** (23 keys, v2); **`zero_counters` 47/47 set-equal** schema==common==verify;
**machine `runs[3]` == v2 config** (run_id v2, `artifact_schema_ids`=[`scgp_global_realbank_resource_v2`],
artifact path in the unchanged `…/v1/m0/realbank_resource/` namespace, slurm 16/96/0, deps
[`…SYNTH-KKT-v4`]); `run_order[3]`=v2; schema strict (21 objects / 21 `additionalProperties:false`);
`bash -n` clean on v2 wrapper + sbatch; artifact dir absent (no-clobber). `py_compile` + the
`numpy`/`torch`/`jsonschema` dependency check remain DEFERRED-TO-RUNTIME (login-node execution
forbidden; the SLURM validator runs them fail-closed).

**Load-bearing insight:** the run_id bump moved the code constant (`RUN3`=v2) and the machine content
(`runs[3].run_id`=v2, `artifact_schema_ids`=v2) in lock-step, so the machine verifier's asserts stay
consistent (no v3-style code↔content drift). The fresh code review must independently re-derive §3–§4.

## 5. Residuals for the fresh code review (unchanged from v1 freeze §4, non-blocking)

R-1 `torch.load(weights_only=True)` (precedented, fail-closed); R-2 eigendecomposition determinism
(same job, `OMP/MKL/OPENBLAS=16`, floatify — audit §6 flags that `require_slurm_realbank` does not
assert the thread vars equal 16, an optional future hardening, not this bug); R-3 `old_protected`
bound `243e89b…`/278 (v2 files are `lb_scgp_global_r2_*` → excluded; validator recomputes, fail-closed);
R-4 structural placeholder `is_science=false` (disclosed, verifier-checked; science-owner override
right). The audit's optional defense-in-depth (route `validate.py:186`'s write through
`canonical_root_path`) was **intentionally NOT bundled** into v2 (audit §5.4) to keep the blast radius
to the wrapper; track separately.

## Status flags

- `ready_for_review = true` — ready for the independent amendment review (ratify the additive run-id
  REPLACE + the wrapper fix) and the fresh 0C/0H realbank **v2** code review (independently re-derive
  §3–§4 and re-adjudicate R-1…R-4).
- `ready_for_execution = false` — execution remains unauthorized. Independent amendment review,
  dependency-availability evidence, fresh 0C/0H code review with the §3–§4 tables all-PASS,
  exact-hashes/no-clobber review, and separate execution authorization are all still required before any
  single executor submit.

## Role separation & required statements

- The realbank-prep role (this document + the v2 clone + the plan/hash edits) is separate from the
  independent amendment-review, fresh code-review, execution-authorization, and executor roles. This
  document authorizes no execution.
- No performance evidence exists and none is claimed; none is possible from a static clone/fix, and the
  realbank run itself emits no accuracy/macro-F1 and does no training or kNN.
- The only project gold is `parent_video_binary_label`; no segment/frame/timestamp/span/localization/
  stance/target/mechanism/rationale/fragment gold is assumed or introduced; train labels are not opened
  (train **features** are, hash-checked).
- Run4 (M1 cache), MLLM/cache, validation/test, and training remain locked. This freeze unlocks neither
  realbank execution nor anything downstream. The v1 single-submit burn is preserved as evidence; v2 is
  a fresh single-submit lineage.
