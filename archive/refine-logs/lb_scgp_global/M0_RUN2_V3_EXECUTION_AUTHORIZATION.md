# M0 Run2-v3 Independent Execution Authorization

Date: 2026-07-13

Authorizer: **Claude Opus 4.8**, fresh, zero-context, zero-history independent **execution
authorizer** for the `lb_scgp_global_r2` M0 Run2 **v3** lineage. This role is deliberately
**separate** from the v3-setup/freeze role (`M0_RUN2_V3_CLONE_FREEZE.md`), the fresh static
code-review role (`M0_RUN2_V3_CODE_REVIEW.md`), and the executor role (§4(d.4)). I did not
create, review, or execute the v3 entities; I independently re-verified the authorization
preconditions from files and cluster state read directly in this session.

## Authorizer boundary

Read-only authorization only. I did **not** run Python, imports, `py_compile`, tests, `conda`,
`sbatch`, an experiment, or any MLLM/OCR/API/model/network/GPU/training/evaluation, and I touched
no validation/test data or cache. Shell was limited to the allowed read-only tools:
`rg`/`sed`/`nl`/`jq`/`awk`/`bash -n`/`diff`/`sha256sum`/`find`/`ls`/`wc`/`git status`/`squeue`/
`sacct`/`sinfo`. The **only** file I wrote is this authorization document. No artifact under
`artifacts/lb_scgp_global/v3` was created (confirmed absent, §4).

**Model-binding divergence declaration** (precedent: `M0_RUN2_V2_CODE_REVIEW_FIX2.md`,
`M0_RUN2_V3_CODE_REVIEW.md` §Reviewer boundary): `AGENTS.md:15` binds the main-dialogue subagent
to "GPT-5.5 xhigh"; that backend is unavailable for this session, so this authorization runs on
the `CLAUDE.md`-bound **Opus 4.8** (`claude-opus-4-8`). Documented process fact, not a defect;
does not affect any ruling below.

---

## Verdict

**AUTHORIZED — exactly one CPU-only SLURM submission.**

All six authorization checks pass. Authorization scope = **exactly one** invocation of:

```
sbatch scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v3.sbatch
```

The two residual Medium findings (M-A, M-B) are non-blocking under the risk resolutions recorded
in §6, which the coordination session ruled on and this authorizer releases with knowledge.

---

## 1. Check 1 — Static-review credential (PASS)

`refine-logs/lb_scgp_global/M0_RUN2_V3_CODE_REVIEW.md` exists and reads:

- Verdict (§Verdict, line 36): **`PASS_STATIC_REVIEW`**.
- Severity counts (lines 40–45): **Critical = 0, High = 0**, Medium = 2 (M-A, M-B), Low = 3
  (L-A, L-B, L-C).
- Pass criterion (Critical = 0 ∧ High = 0) is met (line 47).
- The reviewer is a fresh 0C/0H role, separate from setup/authorization/executor, and
  independently re-proved clone equivalence rather than copying the freeze doc (lines 8–23, 64).

**Result: PASS.** The static-review credential is present and satisfies the gate.

## 2. Check 2 — Freeze-state no-drift (PASS)

I re-`sha256sum`'d all nine v3 entities in this session and compared to
`M0_RUN2_V3_CLONE_FREEZE.md` §1. **All nine match byte-for-byte:**

| # | v3 entity | SHA256 (re-hashed now = freeze §1) |
|---|---|---|
| 1 | `configs/lb_scgp_global_r2/m0_synth_kkt_v3.json` | `e6d33b5d3078b12d87e4c0dc70d0f4fe1ee53681543da347f0e4402fedceb7d5` ✓ |
| 2 | `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_payload_v3.schema.json` | `1d6f93a1b0933a24e361de9ed32abda9dc5f180039d2514121b8c6c8caf2d2d3` ✓ |
| 3 | `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_case_v3.schema.json` | `df3616ff50c54dc756bb600c6ef26626afafe070678450d10ca73b9ff83ddcac` ✓ |
| 4 | `scripts/analysis/lb_scgp_global_r2_run2_v3_common.py` | `9de62f6df2f68fc46ad24e1c19e33b4bd3eeba9303db8ae383ed271a29c411c2` ✓ |
| 5 | `scripts/analysis/lb_scgp_global_r2_run2_v3_validate.py` | `2e0bb00b28debf8bf3b2099ac6363664d9d4d776740f3505311e0f7b74ca13a6` ✓ |
| 6 | `scripts/analysis/lb_scgp_global_r2_run2_v3_producer.py` | `6ef3a4a8146ec9b2a2a94236c1e40f0ebf27aa862b89d27a60e92499b21f5114` ✓ |
| 7 | `scripts/analysis/lb_scgp_global_r2_run2_v3_independent_verify.py` | `4025dbf0482877cc03c46434631134fe81f0593f3ff12f779c87e80aeb8523c5` ✓ |
| 8 | `scripts/wrappers/lb_scgp_global_r2_run2_v3.sh` | `8d9123e9f4eec357a91bd94cbf6c292a3bb188496011845c706f3e34b72d66d3` ✓ |
| 9 | `scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v3.sbatch` | `4495ec3c49970b7024ebf443d845ec5f325013b2190116c547827c6f5de6b3d9` ✓ |

`git status --porcelain` on the nine paths shows all as untracked (`??`) and no v2 source entity
modified — consistent with a frozen, uncommitted clone.

**Result: PASS.** Zero drift between the frozen bytes and the current on-disk bytes; the reviewed
and authorized artifacts are the identical bytes.

## 3. Check 3 — Dependency-availability evidence (PASS) — mandatory §4(d.3) item, dissolves L-C

Read-only `ls` of the exact `HateVideo` interpreter tree
(`/data/jehc223/miniconda3/envs/HateVideo/lib/python3.11/site-packages/`) that the sbatch's
`conda activate HateVideo` resolves shows the full required set present:

```
numpy/                             numpy-1.26.4.dist-info/          numpy.libs/
jsonschema/                        jsonschema-4.26.0.dist-info/
jsonschema_specifications/         jsonschema_specifications-2025.9.1.dist-info/
referencing/                       referencing-0.37.0.dist-info/
rpds/                              rpds_py-2026.6.3.dist-info/
```

- **`numpy` 1.26.4** — present.
- **`jsonschema` 4.26.0** — present, with its full transitive set: **`referencing` 0.37.0**,
  **`rpds_py` 2026.6.3** (the ABI-sensitive `cp311` C-extension, matching the py3.11 interpreter),
  **`jsonschema_specifications` 2025.9.1**.

This matches the third-party set `{numpy, jsonschema}` enumerated by the code review §4.1 (deferred
imports at `common.py:182` and `independent_verify.py:167`).

**Runtime-import corroboration (compensating for the read-only limitation):** a directory listing
proves *installed*, not *imports cleanly*. I cannot run the interpreter. The env-repair record
`M0_ENV_REPAIR_RECORD.md` supplies the missing runtime evidence:
- §3 (lines 52–53): `python -c "import jsonschema, importlib.metadata as m; print(...)"` →
  `4.26.0  .../HateVideo/lib/python3.11/site-packages/jsonschema/__init__.py` — **runtime import
  succeeded** on the HateVideo absolute interpreter.
- §4 (lines 84, 87–88): the exact deferred triple
  `from jsonschema import Draft7Validator, RefResolver; from jsonschema.exceptions import
  SchemaError` was reproduced and imported **runtime-OK**; `py_compile` of all four `.py` modules
  passed; numpy imported OK (1.26.4).
- §2 records and reconciles the `source activate` silent-fallback trap: the gate
  `require_slurm_run2()`/`CONDA_DEFAULT_ENV=="HateVideo"` plus the sbatch's
  `source conda.sh; conda activate HateVideo` (lines 12–13) guarantee the job runs in exactly the
  HateVideo tree evidenced above.

**Result: PASS.** The mandatory dependency-availability evidence is satisfied (directory listing
here + runtime-import record cited). This **dissolves L-C**. The residual is fail-closed: any
breakage (incl. the `RefResolver` DeprecationWarning forward-risk, code review §4.3 / env-repair
§7 — no `PYTHONWARNINGS` set in the v3 sbatch, `grep` count 0) raises → refuse, never a false PASS.

## 4. Check 4 — Single-submit ledger (PASS)

- **`sacct` history for the v3 job name is empty.** `sacct -u jehc223
  --name=lbscgp_global_r2_run2_v3 --starttime=2000-01-01` returns **zero rows** — the v3 job has
  **never been submitted**. A broader recent scan (`--starttime=2026-07-01`) shows only
  `12971  lbscgp_global_r2_run2_v2  FAILED` (the spent v2 attempt), **no** `run2_v3` job.
- **`artifacts/lb_scgp_global/v3/` does not exist** (`ls` → "No such file or directory"); only
  `artifacts/lb_scgp_global/v1/` is present. No v3 manifest / source_manifest / access_ledger /
  semantic_verification / publish-lock exists.
- **`slurm/logs/` has no v3 output:** `find slurm/logs -name '*run2_v3*'` returns nothing.
- `git status` of the nine v3 paths: all untracked, none previously run.

**Result: PASS.** The single-submit budget is intact; this authorizes the first and only v3
submission.

## 5. Check 5 — Resources (PASS)

- **sbatch request** (`lb_scgp_global_r2_m0_synth_kkt_v3.sbatch`, re-read this session):
  `--cpus-per-task=8`, `--mem=64G`, **no** `--gres`/`--gpus` line (GPU = 0), **no `--time`**
  (line 8 comment "Intentionally no --time: project policy"), `--job-name=lbscgp_global_r2_run2_v3`,
  and `source conda.sh; conda activate HateVideo` (lines 12–13). Within per-user caps
  (16 CPU / 128 GB / 2 GPU): 8 ≤ 16 CPU, 64 ≤ 128 GB, 0 ≤ 2 GPU. Conforms to
  `CLAUDE.md` (no `--time`, HateVideo, SLURM-submitted).
- **`squeue -u jehc223` is empty** — the user has zero jobs queued/running, so the full per-user
  allotment is free; the 8 CPU / 64 GB / 0 GPU request fits with wide headroom.
- **`sinfo`:** partition `slurmpartition*` is `up`, `infinite` timelimit, node `foscsmlprd01`
  state `mix` (partially allocated, accepting work). A CPU-only 8-core / 64 GB job is schedulable;
  initial `PENDING (JobHeldUser)` is expected and must be left to auto-release (do not force).

**Result: PASS.**

## 6. Risk resolutions (authorization preconditions — recorded and released)

- **M-B (Medium, fail-closed) — rank-deficient construction convergence not statically provable,
  no read-only dissolution.** The `rank_deficient_structural_solution` 30-step geometric shrink
  (code review §5, `common.py:643–700`) may fail the movement/`r_abs_max` window for some
  fixtures; on failure the producer refuses to publish (fail-closed). It is the single most
  material residual under single-submit and, unlike the resolved `jsonschema` item, **cannot be
  dissolved by any allowed read-only means** (requires running the numpy construction).
  **Risk decision (transcribed):** the coordination session ruled to **accept the quota-burn
  risk**, on three grounds — (1) fail-closed guarantees **no false positive** (a non-convergence
  can never manufacture a spurious PASS); (2) a genuine failure yields **legitimate diagnostic
  information constituting a valid basis for a v4 fix**; (3) a SLURM-only producer dry-run would
  reintroduce **v1-style dual-submission semantic ambiguity** — precedent: the v1 lineage was
  sealed for exactly that dual-submit ambiguity, so a dry-run is disfavored. **Risk acceptor =
  coordination session; this authorizer releases with knowledge** that a non-convergent fixture
  would burn the one authorized attempt fail-closed with zero science.
- **M-A (Medium) — synthetic `G0` not verified PSD / rank-`≤ d` realizable.** Carries a
  *conditional* escalation to High **only if** a science authority rules the synthetic `G0` must
  itself be a realizable rank-`≤ d` PSD `Z0 Z0^T` (code review §5 M-A, §7 note 2). **No such
  ruling exists** in the materials reviewed; the KKT self-test's correctness does not depend on
  `G0` PSD-ness (correctness-neutral fidelity gap). Escalation **not triggered** → **maintained
  Medium, non-blocking.** Flagged for the science owner before any *scientific* claim rests on the
  fixture.
- **L-C (Low, fail-closed) — dependency availability was proven by listing, not runtime import;
  env not frozen.** **Dissolved** by Check 3 (directory listing here + cited runtime-import record
  in `M0_ENV_REPAIR_RECORD.md` §3–§4). Residual breakage remains fail-closed.

---

## 7. Ledger snapshot (authorization-time)

| field | value |
|---|---|
| v3 job name | `lbscgp_global_r2_run2_v3` |
| prior v3 submissions (sacct, all-time) | **0** |
| authorized submissions remaining | **exactly 1** |
| `artifacts/lb_scgp_global/v3/` | absent |
| v3 slurm logs | none |
| `squeue -u jehc223` | empty (0 jobs) |
| partition / node | `slurmpartition` up / `foscsmlprd01` mix |
| resource request | 8 CPU / 64 GB / 0 GPU / no `--time` / HateVideo |

---

## 8. Executor instructions (§4(d.4))

1. Submit **exactly once**: `sbatch scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v3.sbatch`.
2. **Record the returned job id** immediately.
3. Expect initial `PENDING (JobHeldUser)`; **wait for auto-release** — do not force-release.
4. **After any terminal state, do not resubmit** regardless of outcome (single-submit budget spent
   on submission). M-B may cause a legitimate fail-closed non-publish; that is an authorized
   outcome, not grounds for a re-submit — it feeds a separate v4 decision.
5. **On PASS** (producer publishes the v3 manifest + counters): route to a **fresh independent
   artifact review** (separate role) — do not self-certify.
6. **On FAILED** (job error or fail-closed non-publish): route to a **fresh result-to-claim
   review** to adjudicate the failure and decide v4, with the job log as evidence.
7. The executor role is separate from setup/freeze, code-review, and this authorization role.

---

## Required statements

- No performance evidence exists and no performance claim is made or possible from this
  authorization of a byte-exact clone.
- The only project gold is `parent_video_binary_label`. No segment/frame/timestamp/span/
  localization/stance/target/mechanism/rationale/fragment gold is assumed or introduced; v3 has
  produced no artifact and no counters at authorization time.
- Run3, M1, MLLM/cache, validation/test, training, and realbank remain **locked**. This
  authorization covers only one CPU-only synthetic-certificate self-test SLURM job.
- Authorization scope = exactly one `sbatch scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v3.sbatch`.
  The M-A conditional escalation must be resolved (ruling or amendment) before any *scientific*
  claim rests on the synthetic `G0` fixture.
- Authorizer = Claude Opus 4.8 fresh 0C/0H agent, execution-authorization role, separate from the
  v3-setup/freeze, static-code-review, and executor roles.

Report SHA256 is to be computed externally after this file is written; it is not embedded to avoid
a self-referential hash.
