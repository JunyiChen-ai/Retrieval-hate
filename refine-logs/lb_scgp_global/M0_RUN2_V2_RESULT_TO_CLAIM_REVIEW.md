# LB-SCGP Global-R2 M0 Run2-v2 Fresh Result-to-Claim Review

Date: 2026-07-13

Reviewer: **Claude Opus 4.8**, fresh / zero-context, zero-history (0C/0H) independent
result-to-claim reviewer. Role separation: this reviewer is a **distinct role** from the
static code reviewer (`M0_RUN2_V2_CODE_REVIEW_FIX2.md`), the execution-authorization
reviewer (`M0_RUN2_V2_EXECUTION_AUTHORIZATION.md`), and the executor
(`M0_RUN2_V2_EXECUTION_RECORD.md`).

Reviewer boundary: read-only adjudication only. No subagents, workflows, model/API calls,
SLURM submissions, experiments, GPU/training/performance work, validation/test work,
MLLM/OCR work, environment mutation, package installs, or code/config/schema/wrapper/
artifact edits were performed. No Python was executed and no import was run in the target
environment; the environment fact below was established by a **read-only site-packages
directory listing**, not by executing the interpreter. This file is the only new write.

Precedent binding: this review follows and is consistent with
`refine-logs/lb_scgp_global/M0_RUN2_RESULT_TO_CLAIM_REVIEW_FRESH.md` (the v1-lineage
verdict: `claim_supported=no`, `route=infrastructure_repair`, v1 lineage closed after two
fail-closed `KeyError` attempts, jobs `12902`/`12904`). v2 is the "authorized-in-principle"
single non-overwriting repair lineage that that verdict permitted; this document adjudicates
its single consumed attempt (job `12971`).

---

## Structured Verdict

- intended_claim: The v2 synthetic global-projection / serialized H-metric normal-cone KKT
  gate is executable and independently verifiable for LB-SCGP Global-R2.
- claim_supported: **no**
- route: **infrastructure_repair** (environment dependency provisioning)
- failure_classification: **infrastructure_environment_dependency_failure** (NOT science)
- confidence: **high**
- v2_lineage_disposition: **single-submit consumed → lineage CLOSED; must not be re-proposed**
- science_information_leak: **none**
- false_positive_risk: **none** (fail-closed; PASS gate never reached, `decision` never emitted)
- repair_authorization: authorized **in principle only** for one new non-overwriting **v3**
  lineage; execution is **not** authorized by this report.
- execution_authorization: **not_authorized**
- plan_amendment_required: **yes**
- Run3 / M1 / MLLM-cache / validation-test / training / realbank: **locked**

---

## 1. Failure classification — infrastructure, not science (independently verified)

I independently re-read the validator report at
`slurm/logs/lbscgp_global_r2_run2_v2_12971.err` and traced its `checks` array item by item.
The report contains **16 checks; exactly ONE is FAIL, the other 15 are PASS**:

| # | Check | Status |
|---|---|---|
| 1 | `jq -e .` config `m0_synth_kkt_v2.json` | PASS |
| 2 | `jq -e .` `EXPERIMENT_PLAN.machine.json` | PASS |
| 3 | `jq -e .` payload schema | PASS |
| 4 | `jq -e .` case schema | PASS |
| 5 | `jq -e .` cert schema | PASS |
| 6 | `jq -e .` `v1/m0/contract_freeze.json` | PASS |
| 7 | schema-load errors (`{"errors":{}}`) | PASS |
| 8 | `bash -n` wrapper | PASS |
| 9 | `bash -n` sbatch | PASS |
| 10 | **`python_dependency_jsonschema`** (`returncode 1`, `missing:["jsonschema"]`) | **FAIL** |
| 11 | `py_compile` of the four v2 python modules | PASS |
| 12 | `git diff --check` (nine entities + tracker) | PASS |
| 13 | new-file whitespace scan (`bad_lines:[]`) | PASS |
| 14 | hash mismatch scan (`mismatches:[]`) | PASS |
| 15 | no-clobber forbidden-outputs scan (`existing_forbidden_outputs:[]`) | PASS |
| 16 | final errors scan (`errors:[]`) | PASS |

Overall report `status: FAIL` (`…12971.err:170`), driven solely by check #10. The single
failing check is a **missing Python package in the `HateVideo` conda environment**
(`…12971.err:101-112`): `args` invoke the job Python `find_spec('jsonschema')`, `returncode 1`,
`stdout {"checked":["jsonschema"],"missing":["jsonschema"]}`. This is the environment residual
**M-C** that the static review (`…CODE_REVIEW_FIX2.md:319-324`) and the execution authorization
(`…EXECUTION_AUTHORIZATION.md:40-42, 149-153`) both pre-flagged as fail-closed.

**Producer and independent_verify never ran; zero science information was computed or leaked.**
`validate.py` is the first of the three ordered wrapper segments
(`validate → producer → independent_verify`, wrapper `:60-73`) under `set -euo pipefail`.
`validate.py` emitted the FAIL report and exited non-zero, so `set -e` aborted before the
producer (`:65-68`) or the verifier (`:70-73`) or the `jq -e '.decision == "PASS"'` gate
(`:75`) could be reached; `COMPLETE` stayed `0` and the EXIT trap swept prospective outputs
(none existed). I independently confirm the terminal facts:

- `artifacts/lb_scgp_global/v2/` **does not exist** (only `v1` is present) — zero artifacts:
  no `manifest.json`, `source_manifest.json`, `access_ledger.json`, `semantic_verification.json`,
  or publish lock.
- stdout `…12971.out` is **0 bytes**.
- `sacct`: job `12971` FAILED, `ExitCode 1:0`, `Elapsed 00:00:01`.
- The `decision` field / PASS gate was **never produced**. No KKT tolerance, stationarity,
  dual-feasibility, complementarity, rank-tail, factor-replay, movement, or any numeric/solver
  result was computed to completion.

**Zero false-positive risk.** The validator is fail-closed and correctly refused to proceed;
a missing dependency cannot manufacture a false PASS (the same property the static review
established at `…CODE_REVIEW_FIX2.md:201-209`). Code presence is not behavioral evidence.

Therefore this failure is **not** a scientific, optimization, numerical, KKT, rank, factor,
solver, mechanism, MLLM, OCR, dataset, or performance result of any kind. It is a pure
environment-dependency (infrastructure) failure at validator preflight, before any scientific
computation. This is the same character as the v1-lineage failures (interface `KeyError`s),
one layer out (environment rather than interface).

## 2. What the results support / do not support

Support only these narrow procedural facts:

- Run1 `LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1` remains frozen and untouched; the v2 attempt
  wrote nothing.
- v2 job `12971` reached the validator, which fail-closed on the missing `jsonschema`
  dependency and published no artifact.
- Fifteen of sixteen static/validator preflight checks passed; the sole blocker is
  environmental.

Do **not** support:

- Any claim of v2 executability or independent verifiability.
- Any scientific, numeric, KKT, rank, factor, mechanism, or performance claim.
- Any Run3 / M1 / realbank unlock.

## 3. v2 lineage disposition — CLOSED

Per the execution authorization (`…EXECUTION_AUTHORIZATION.md:105, 146-148`) and the executor
record (`…EXECUTION_RECORD.md:8-10, 116-119, 124-125`), the single-submit v2 budget is now
**spent**. Job `12971` reached a terminal state (FAILED). **The v2 lineage is closed and must
not be re-proposed, re-run, or resubmitted under any outcome.** Any further attempt requires a
brand-new lineage (v3) and a brand-new execution-authorization pass; the existing v2
authorization document is void for future use.

## 4. Route judgment and v3-lineage opening conditions

route: **infrastructure_repair** (consistent with the v1 precedent verdict).

v3 may be opened **only** after **all** of the following are satisfied, in order:

### (a) Environment repair with a verifiable record

Install `jsonschema` into the `HateVideo` conda environment
(`/data/jehc223/miniconda3/envs/HateVideo`). Produce a checkable repair record that captures
the install command, the resolved version, and a **read-only** post-install confirmation
(e.g. the presence of `…/envs/HateVideo/lib/python3.11/site-packages/jsonschema*`, or an
authorized SLURM-only preflight — never a login-node interpreter run outside SLURM). Record
the record's path and hash.

### (b) Full dependency audit of the four v2 `.py` modules — ALL import sites, not only top-level

Enumerate every import of
`scripts/analysis/lb_scgp_global_r2_run2_v2_{common,validate,producer,independent_verify}.py`
and prove each **third-party** package is importable in `HateVideo`. This reviewer's read-only
enumeration (grep of all import lines, plus a read-only site-packages check) already establishes
the picture; the v3 authorizer must re-confirm it at authorization time:

- **Third-party, top-level:** `numpy` — imported top-level in `common.py:19`, `producer.py:16`,
  `independent_verify.py:20`. **Read-only check: PRESENT** in `HateVideo` site-packages.
- **Third-party, DEFERRED (in-function, not top-level):** `jsonschema` — `from jsonschema import
  Draft7Validator, RefResolver` and `from jsonschema.exceptions import SchemaError` at
  `common.py:182-183` and `independent_verify.py:167-168`. **Read-only check: ABSENT** →
  this is exactly what killed job `12971`. To be provisioned per (a).
- **Local intra-package:** `validate.py` and `producer.py` import
  `lb_scgp_global_r2_run2_v2_common` via `sys.path` insertion (not a PyPI package).
- **Stdlib only** otherwise: `argparse, copy, hashlib, json, math, os, subprocess, sys,
  tempfile, pathlib, typing, __future__`.

**Load-bearing lesson for the audit method:** the dependency that burned the attempt
(`jsonschema`) is a **deferred / in-function import**, invisible to any audit that inspects
only module-top import statements. `validate.py` itself has **no** top-level third-party import,
yet it triggers the jsonschema preflight and imports `common` (which carries the deferred
jsonschema import). The v3 dependency audit **must** therefore grep **every** import line across
the modules (top-level and in-function) — not the module header — and prove availability of the
full third-party set `{numpy, jsonschema}`. After (a), that set is satisfied (numpy present,
jsonschema installed); v3 must still re-verify, because environment state is not frozen.

### (c) v3 = byte-exact clone of the v2 code

v3 source must be a **byte-for-byte-equivalent clone** of the frozen v2 entities, with changes
limited strictly to: file names, internal self-references (module names / `run_id` /
namespace / paths from `v2` to `v3`), and hash bindings. **No behavioral code change is
permitted.** Any behavioral edit voids the clone status and forces a **full v3 implementation
audit** (not a mechanical re-hash). The nine v2 entities to clone are those enumerated in
`M0_RUN2_V2_IMPLEMENTATION_FIX2_FREEZE.md` (config, payload/case/cert schemas, the four python
modules, wrapper, sbatch).

### (d) Full ceremony re-run for v3

1. **Freeze** the v3 entities with exact SHA256 bindings (fresh freeze document).
2. **Fresh 0C/0H static code review** (Critical=0, High=0 required), independently
   re-adjudicating the residual findings — note M-A (conditional escalation: synthetic `G0`
   realizability), M-B (rank-deficient construction convergence not statically provable) still
   apply and carry to v3.
3. **Independent execution authorization**, with a **new mandatory checklist item:
   "dependency-availability evidence"** — the authorizer must, by read-only means (per (a)/(b)),
   confirm every third-party import of the four modules is present in `HateVideo` **before**
   authorizing, so the single-submit quota is not spent on a foreseeable environment miss.
4. **Independent executor single submit** — exactly one `sbatch`, no resubmission under any
   outcome, followed by a fresh result-to-claim review of the outcome.

## 5. Process lesson (recorded)

**A static-review fail-closed Medium (the M-B / M-C class) is, under a single-submit regime,
equivalent to a high-probability burn of the only authorized quota.** Job `12971` is the
concrete proof: M-C (`jsonschema` availability unproven) was flagged **three times** — in the
static review (`…CODE_REVIEW_FIX2.md:319-324`), the execution authorization
(`…EXECUTION_AUTHORIZATION.md:40-42`), and the executor instructions
(`…EXECUTION_AUTHORIZATION.md:149-153`) — yet the attempt was authorized and submitted anyway,
and the residual burned the quota in one second exactly as predicted, with zero scientific
return.

Corrective rule for future single-submit lineages: **fail-closed environmental / dependency
Medium findings must be dissolved before authorization using read-only means** (e.g. listing
the conda env's `site-packages` directory, or an authorized SLURM-only preflight), not carried
into submission as "flagged but non-blocking." A residual that can only fail (never falsely
pass) but can consume the sole attempt should gate authorization, because the cost of a wasted
single-submit is a full new ceremony. The v3 authorization checklist item (d.3) operationalizes
this rule. The deferred-import subtlety in §4(b) is the second half of the lesson: a
dependency audit must cover in-function imports, or it recreates exactly this failure.

---

## Required statements

- This review is not performance evidence and makes no performance claim; none is possible from
  a preflight-failed run.
- The only project gold is `parent_video_binary_label`. No segment/frame/timestamp/span/
  localization/stance/target/mechanism/rationale/fragment gold is assumed or introduced; v2
  produced no artifact and no counters.
- Run3, M1, MLLM/cache, validation/test, training, and realbank remain locked.
- The v2 single-submit budget is spent and the v2 lineage is closed; this report does not
  authorize any execution. A v3 lineage requires the full ceremony in §4 before any submission.
- Reviewer role (this document) is separate from the static-code-review, execution-authorization,
  and executor roles.
