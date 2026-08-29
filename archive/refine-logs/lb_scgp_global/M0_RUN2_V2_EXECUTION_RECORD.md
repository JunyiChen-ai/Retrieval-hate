# M0 Run2-v2 Execution Record

Executor: **Claude Opus 4.8** (`claude-opus-4-8`), fresh execution role.
Role separation: this executor is a **distinct role** from the static code reviewer
(`M0_RUN2_V2_CODE_REVIEW_FIX2.md`) and from the execution-authorization reviewer
(`M0_RUN2_V2_EXECUTION_AUTHORIZATION.md`). This executor performed exactly the one
authorized `sbatch` submission plus read-only monitoring/evidence collection, and
wrote only this record. No code was modified. **This is the sole v2-lineage
submission; per the single-submit budget, no resubmission will occur under any
outcome.**

## Outcome (one line)

Job **12971** → terminal **FAILED** (ExitCode `1:0`, Elapsed `00:00:01`).
Wrapper PASS gate (`decision == "PASS"`) **NOT satisfied**; `decision` field
**未产出 / not produced** (the `independent_verify` segment that emits it never ran).
Root cause: **`jsonschema` missing in the `HateVideo` conda env** — the pre-flagged
static-review residual **M-C**. Fail-closed: **no false PASS, no artifacts written.**

---

## 1. Pre-flight verification (all PASS; would have aborted otherwise)

1. Authorization doc `M0_RUN2_V2_EXECUTION_AUTHORIZATION.md` present; verdict
   **AUTHORIZED** (its line 18).
2. sha256 spot-check of the 3 key entities — all match the authorization doc:
   - `scripts/wrappers/lb_scgp_global_r2_run2_v2.sh` =
     `14eb036f17b77037fd89624c0f9f7487432fa476da540b0fbc5f503719320716` (doc :58) ✓
   - `scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py` =
     `8c4cc842aa53e3d8293744449101282a33abb7715691653bb934b2e563b1cd51` (doc :56) ✓
   - `scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v2.sbatch` =
     `f91462303e4b28eb0722d546f135c00c2aeacf56151e27d5421f211520ba94bf` (doc :59) ✓
3. `squeue -u jehc223` empty (no same-name job running);
   `artifacts/lb_scgp_global/v2/m0/synth_kkt/` did not exist. ✓

## 2. Submission

- Command: `sbatch scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v2.sbatch`
- Submit timestamp: **2026-07-12T15:55:36Z** (NZ local ≈ 2026-07-13T03:55:36).
- sbatch reply: `Submitted batch job 12971`.
- Job id: **12971**. Job name: `lbscgp_global_r2_run2_v2`.

## 3. State timeline

| Event | Time | Source |
|---|---|---|
| Submitted | 2026-07-12T15:55:36Z (local ~03:55:36) | `sbatch` reply |
| Start (RUNNING) | 2026-07-13T03:55:41 (local) | `sacct` Start |
| End (FAILED) | 2026-07-13T03:55:42 (local) | `sacct` End |
| First executor poll already observed terminal | 2026-07-12T15:55:46Z | foreground `sacct` poll |

Auto-release was near-immediate: no `JobHeldUser` dwell was observed (the first poll,
~10 s after submit, already showed `FAILED`). No `scontrol release` was issued.
Elapsed `00:00:01`; ExitCode `1:0`; DerivedExitCode `0:0`.

`sacct -j 12971`:
```
12971         lbscgp_global_r2_run2_v2  FAILED  Start 2026-07-13T03:55:41  End 03:55:42  Elapsed 00:00:01  ExitCode 1:0
12971.batch   batch                     FAILED  Start 2026-07-13T03:55:41  End 03:55:42  Elapsed 00:00:01  ExitCode 1:0
```

## 4. Terminal state & decision result

- Terminal state: **FAILED**.
- Wrapper PASS判定 `decision == "PASS"` (`scripts/wrappers/lb_scgp_global_r2_run2_v2.sh:75`):
  **NOT satisfied** — the gate was never reached.
- `decision` field / `semantic_verification.json`: **未产出 (not produced)**. No
  `manifest.json` or any output exists under
  `artifacts/lb_scgp_global/v2/m0/synth_kkt/` — the directory
  `artifacts/lb_scgp_global/v2/` does not exist; only `v1` is present.

## 5. Evidence excerpts (read-only)

Log files:
- stdout `slurm/logs/lbscgp_global_r2_run2_v2_12971.out` — **0 bytes (empty)**.
- stderr `slurm/logs/lbscgp_global_r2_run2_v2_12971.err` — 29,303 bytes; this is the
  `validate.py` JSON report (validator_sha256
  `4389c4a1dc...b3dfc36`, matching entity 5 in the authorization doc).

Which segment failed — the **validate** segment (the FIRST of the three
validate → producer → independent_verify). The wrapper runs
`lb_scgp_global_r2_run2_v2_validate.py` at
`scripts/wrappers/lb_scgp_global_r2_run2_v2.sh:60-63` under `set -euo pipefail`
(`:2`). The producer (`:65-68`), independent_verify (`:70-73`), and the
`jq -e '.decision == "PASS"'` gate (`:75`) were never reached, so `COMPLETE`
stayed `0` and the exit trap (`:23-32`, `:38`) swept prospective outputs (none
existed) and propagated exit `1`.

Failing check (root cause), from the stderr validate report:
- `slurm/logs/lbscgp_global_r2_run2_v2_12971.err:101-112` — check
  `python_dependency_jsonschema`:
  ```
  args: [.../envs/HateVideo/bin/python, -c, "import importlib.util,...['jsonschema']..."]
  returncode: 1
  status:    "FAIL"
  stdout:    {"checked": ["jsonschema"], "missing": ["jsonschema"]}
  ```
- `...12971.err:170` — overall `"status": "FAIL"`;
  `...:169` — `"slurm_job_id": "12971"`;
  `...:167` — `"run_id": "LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2"`.
- All other validate checks (jq well-formedness, `bash -n` on wrapper+sbatch,
  `py_compile` of the four v2 python modules, git-diff-check, hash/whitespace/
  no-clobber scans) were `PASS`; the single FAIL was the missing dependency.

Independent read-only confirmation of root cause:
- `/data/jehc223/miniconda3/envs/HateVideo/bin/python -c "importlib.util.find_spec('jsonschema')"`
  → **`jsonschema MISSING`**.

## 6. Root-cause adjudication

The failure is the environment residual **M-C** that the authorization document
pre-flagged (its lines 40-42, 149-153): `jsonschema` is not installed in the
`HateVideo` conda environment. The v2 validator is **fail-closed** and correctly
refused to proceed — it produced **no false PASS and no artifacts**. However, per
the single-submit budget (authorization doc lines 105, 146-148), this consumed the
one and only authorized v2 attempt. Any further attempt requires a **new
execution-authorization pass**, not the existing document, and is outside this
executor's remit. No environment mutation, no code change, and no resubmission were
performed.

## Required statements

- This record is not performance evidence and makes no performance claim.
- The single v2-lineage submission budget is now spent; no resubmission will be made
  under any outcome by this executor.
- Executor role (this document) is separate from the static-code-review role and from
  the execution-authorization role.
