# M0 Run2-v4 Independent Execution Authorization

Date: 2026-07-13

Authorizer: **Claude Opus 4.8**, fresh, zero-context, zero-history independent **execution
authorizer** for the `lb_scgp_global_r2` M0 Run2 **v4** lineage. This role is deliberately
**separate** from the v4-prep / clone-freeze role (`M0_RUN2_V4_CLONE_FREEZE.md`), the merged
amendment-ratification + fresh 0C/0H static code-review role (`M0_RUN2_V4_CODE_REVIEW.md`), and the
executor role (§8). I did not author, review, or execute the v4 entities; I independently
re-verified the authorization preconditions from files and cluster state read directly in this
session, and I performed the one mandatory active obligation of this ceremony — the post-review
config binding + re-freeze (§3).

## Authorizer boundary

Read-only authorization **except** the single mandated write obligation (§3: add the review-doc
binding to the v4 config and re-freeze). Apart from that one surgical config edit, the review-doc
binding, this authorization document, and the freeze-document addendum, I wrote nothing. I did
**not** run project Python, imports, `py_compile`, tests, `conda`, `sbatch`, an experiment, or any
MLLM/OCR/API/model/network/GPU/training/evaluation, and I touched no validation/test data or cache.
Shell was limited to the allowed tools: `rg`/`sed`/`nl`/`jq`/`awk`/`sha256sum`/`find`/`ls`/`wc`/
`git status`/`squeue`/`sacct`/`sinfo`/`scontrol`. No artifact under `artifacts/lb_scgp_global/v4`
was created (confirmed absent, §5).

**Model-binding divergence declaration** (precedent: `M0_RUN2_V3_EXECUTION_AUTHORIZATION.md`,
`M0_RUN2_V4_CODE_REVIEW.md` §Reviewer boundary): `AGENTS.md:15` binds the main-dialogue subagent to
"GPT-5.5 xhigh"; that backend is unavailable for this session, so this authorization runs on the
`CLAUDE.md`-bound **Opus 4.8** (`claude-opus-4-8`). Documented process fact, not a defect; does not
affect any ruling below.

---

## Verdict

**AUTHORIZED — exactly one CPU-only SLURM submission.**

All authorization checks pass and the mandatory post-review binding obligation is complete. The
executor must verify the **new** config hash (§3) — not the pre-binding `118afadf…3bf0f`.
Authorization scope = **exactly one** invocation of:

```
sbatch scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v4.sbatch
```

The two residual Medium findings (M-A, M-B) are non-blocking under the risk resolutions recorded in
§7, which the coordination session ruled on and this authorizer releases with knowledge. In
particular, a fail-closed non-convergence at M-B (`rank_deficient_structural_solution`) is a
**known, consciously-accepted** outcome, not grounds for re-submission.

---

## 1. Check 1 — Static-review credential (PASS)

`refine-logs/lb_scgp_global/M0_RUN2_V4_CODE_REVIEW.md` exists and reads:

- Verdict (§Verdict, line 38): **`PASS_STATIC_REVIEW`** — **and the v4 plan-amendment is
  `RATIFIED`** (REPLACE-at-index-`[2]` semantics affirmed as the byte-clone analogue of the v2
  INSERT).
- Severity counts (§Verdict table): **Critical = 0, High = 0**, Medium = 3 (M-A carried, M-B
  carried, M-C-doc new), Low = 3 (L-A, L-B, L-C carried).
- Pass criterion (Critical = 0 ∧ High = 0) is met; the reviewer is a fresh 0C/0H role, separate
  from setup/authorization/executor, who independently re-ran the clone-equivalence diff, the 13-row
  runtime simulation, and the old-protected manifest reconstruction rather than copying the freeze
  doc.
- The merged review both **ratifies** the amendment (§1–§2 of the review) and returns the fresh
  code-review PASS with all 13 §5(c) runtime-cross-check rows independently re-derived to PASS.
- M-C-doc (the new Medium) is a documentation-accuracy fix to `M0_RUN2_V4_CLONE_FREEZE.md` §3; it
  was applied (freeze-doc revision note dated 2026-07-13) and carries no runtime or false-PASS risk.

**Result: PASS.** The static-review credential is present, is a PASS with Critical=0/High=0, and the
amendment is ratified.

## 2. Check 2 — Freeze-state no-drift (PASS)

I re-`sha256sum`'d all nine v4 entities in this session and compared to `M0_RUN2_V4_CLONE_FREEZE.md`
§1. **At the pre-binding snapshot all nine matched byte-for-byte, including the config at the §1
target `118afadf…3bf0f`:**

| # | v4 entity | SHA256 (freeze §1) | pre-binding |
|---|---|---|---|
| 1 | `configs/lb_scgp_global_r2/m0_synth_kkt_v4.json` | `118afadfc18cb493a298eda516160f531abfce982471ea836a2d1c6c35f3bf0f` | ✓ (now moved by §3) |
| 2 | `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_payload_v4.schema.json` | `6c31a7c1c98a63ed5a35bdd7313c504f2b870c11c2218f90776b0b88de8ac9ca` | ✓ |
| 3 | `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_case_v4.schema.json` | `df3616ff50c54dc756bb600c6ef26626afafe070678450d10ca73b9ff83ddcac` | ✓ |
| 4 | `scripts/analysis/lb_scgp_global_r2_run2_v4_common.py` | `c6745d4be3eef3afee28d1b63478323fab57fb9e4b79740b05e5a11f2d62dbae` | ✓ |
| 5 | `scripts/analysis/lb_scgp_global_r2_run2_v4_validate.py` | `7eda5e85d1b7bb87e34307946ad112fdad22f7d51f0f5868fed2110ee4b87ec2` | ✓ |
| 6 | `scripts/analysis/lb_scgp_global_r2_run2_v4_producer.py` | `84439f7c2db1adf5f0046a0acbcd49e8a896e4e71e053c2c8211a18688b5179f` | ✓ |
| 7 | `scripts/analysis/lb_scgp_global_r2_run2_v4_independent_verify.py` | `da827f0a4b2bf4f3bf07cb38497e14dcf9d22c2a7be2f9ba9fdc3bc4ca476060` | ✓ |
| 8 | `scripts/wrappers/lb_scgp_global_r2_run2_v4.sh` | `0ad33ba4c3e43e52800d5d1a79316e0b1ebb84d9fc65aac4a4de087e1c65d161` | ✓ |
| 9 | `scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v4.sbatch` | `8e1359ac259fd9e54181d94f77e7e29c55d8c1b44c012b1081ecffc792145427` | ✓ |

**Result: PASS.** Zero drift at authorization start; entities 2–9 remain identical after the §3
binding (only the config row moves, by design).

## 3. Post-review binding obligation — COMPLETE (config re-frozen)

Per `M0_RUN2_V4_CODE_REVIEW.md` §8 item 4 (待裁点④) and the v2 precedent (config re-frozen after
review), I added the v4 **review-doc** binding to the config's
`hash_bindings.authoritative_inputs`. This is the only permitted write to the config.

- **Bound key** (appended at the chronological end of `authoritative_inputs`, matching the existing
  full-relative-path review-binding key style, e.g. `…/M0_RUN2_V2_CODE_REVIEW.md`):
  `refine-logs/lb_scgp_global/M0_RUN2_V4_CODE_REVIEW.md`
  = `41650dcea19c6abd88b0755195ba9333abb43331d68e12f5a5f0d72b2a82d9dc` (on-disk review-doc hash).
- **Scope discipline:** exactly one key added; **no** plan / `EXPERIMENT_PLAN.machine.json` /
  `_HASHES.sha256` / existing-amendment document was touched (the task's "此后不得再动
  plan/machine/HASHES 任何文档"). `authoritative_inputs` count **27 → 28**.
- **Merged-review note:** the v4 review ceremony is a single merged document; the separately
  anticipated `M0_RUN2_V4_PLAN_AMENDMENT_REVIEW.*` / `…_AMENDMENT_INDEPENDENT_REVIEW.md` were never
  created (v4-prep freeze §4 flagged them as "deferred"), so exactly one review doc exists and
  exactly one review binding is added — the v2 lineage's three separate review docs collapse into
  this one.
- **Config freeze hash** (nine-entity §1 config row moves off `118afadf…`, as §4 of the freeze doc
  anticipated):

  | | SHA256 |
  |---|---|
  | config **before** binding (= §1 target) | `118afadfc18cb493a298eda516160f531abfce982471ea836a2d1c6c35f3bf0f` |
  | config **after** binding (**executor verifies THIS**) | `e67fc6f79073b8a425ce24834db966bd5b61a2282e2247a5ef87e396056844d1` |

- **Post-binding verification (read-only after the single surgical edit):** `jq -e .` valid; the new
  value equals the on-disk review-doc hash; the **full `authoritative_inputs` replay is 28/28 ==
  on-disk, fail=0** — exactly the runtime `verify_expected_hashes(authoritative_inputs)` at
  `…v4_common.py:840`→`:834` and `…v4_independent_verify.py:952-954`; `run1_frozen` is **10/10,
  fail=0**. The config is **not** self-bound (verified), so re-freezing it breaks no runtime assert.
- **Hash-freeze consequence:** with `M0_RUN2_V4_CODE_REVIEW.md` now under runtime hash validation
  (`common:840`), it must **not** change by a single byte from `41650dce…`; any edit would
  fail-closed the executor's runtime replay. The addendum recording this binding was appended to
  `M0_RUN2_V4_CLONE_FREEZE.md`, which is **not** bound in the config (verified), so that append does
  not affect any runtime hash check.

**Result: obligation complete.** From here the authorizer makes no further edit to any
plan/machine/HASHES/amendment/review document.

## 4. Check 3 — Dependency-availability evidence (PASS) — mandatory item, dissolves L-C

Read-only `ls` of the exact `HateVideo` interpreter tree that the sbatch's `conda activate
HateVideo` resolves (`/data/jehc223/miniconda3/envs/HateVideo/lib/python3.11/site-packages/`, with
`.../bin/python -> python3.11`) shows the full required set:

- **`numpy` 1.26.4** — present (`numpy/`, `numpy-1.26.4.dist-info/`).
- **`jsonschema` 4.26.0** — present, with its full transitive set: **`jsonschema_specifications`
  2025.9.1**, **`referencing` 0.37.0**, **`rpds_py` 2026.6.3** (the ABI-sensitive `cp311`
  C-extension matching the py3.11 interpreter).

This matches the third-party set `{numpy, jsonschema}` enumerated by the code review §5 (deferred
in-function imports at `…v4_common.py:182-183` and `…v4_independent_verify.py:167-168`, both
fail-closed to `RuntimeError`).

**Runtime-import corroboration** (a listing proves *installed*, not *imports cleanly*):
`M0_ENV_REPAIR_RECORD.md` supplies the missing runtime evidence — §3 (lines 52–53):
`import jsonschema … → 4.26.0  .../HateVideo/lib/python3.11/site-packages/jsonschema/__init__.py`
(runtime import succeeded on the HateVideo absolute interpreter); §4 (lines 87–88): the exact
deferred triple `from jsonschema import Draft7Validator, RefResolver; from jsonschema.exceptions
import SchemaError` reproduced runtime-OK, `py_compile` of all four `.py` modules passed, numpy
imported OK (1.26.4). §2 records the `source activate` silent-fallback trap and confirms the sbatch's
`source conda.sh; conda activate HateVideo` plus the `CONDA_DEFAULT_ENV=="HateVideo"` gate guarantee
the job runs in exactly the evidenced HateVideo tree.

**Result: PASS.** This dissolves L-C. The residual is fail-closed: any breakage (incl. the
`RefResolver` DeprecationWarning forward-risk — no `PYTHONWARNINGS` set in the v4 sbatch) raises →
refuse, never a false PASS.

## 5. Check 4 — Single-submit ledger (PASS)

- **`sacct` all-time for the v4 job name is empty.** `sacct -u jehc223
  --name=lbscgp_global_r2_run2_v4 --starttime=2000-01-01` returns **zero rows** — the v4 job has
  **never been submitted**. A recent scan (`--starttime=2026-07-01`) shows only the spent
  `12971 lbscgp_global_r2_run2_v2 FAILED` and `12974 lbscgp_global_r2_run2_v3 FAILED` — **no**
  `run2_v4` job.
- **`artifacts/lb_scgp_global/v4/` does not exist** (`ls` → "No such file or directory"); only
  `artifacts/lb_scgp_global/v1/` is present. No v4 manifest / source_manifest / access_ledger /
  semantic_verification / publish-lock exists.
- **`slurm/logs/` has no v4 output:** `find slurm/logs -name '*run2_v4*'` → 0 files.

**Result: PASS.** The single-submit budget is intact; this authorizes the first and only v4
submission.

## 6. Check 5 — Resources (PASS)

- **sbatch request** (`lb_scgp_global_r2_m0_synth_kkt_v4.sbatch`, re-read this session):
  `--cpus-per-task=8`, `--mem=64G`, **no** `--gres`/`--gpus` (GPU = 0), **no `--time`** (line 8
  comment "Intentionally no --time: project policy"), `--job-name=lbscgp_global_r2_run2_v4`, and
  `source conda.sh; conda activate HateVideo` (lines 12–13). Matches the config `run.slurm` block
  `{cpu:8, ram_gb:64, gpu:0, env:"HateVideo", no_time_flag:true}` exactly. Within per-user caps
  (16 CPU / 128 GB / 2 GPU): 8 ≤ 16 CPU, 64 ≤ 128 GB, 0 ≤ 2 GPU. Conforms to `CLAUDE.md`.
- **`squeue -u jehc223`** shows a single pending job `12976 tarc_g2pred PENDING (JobHeldUser)`,
  which `scontrol` confirms requests **8 CPU / 64 GB / 1 A100 GPU** (`gres/gpu:a100:1`). This is a
  different GPU lineage (the target-loop G2 prediction job). **Non-conflicting:** the v4 CPU job
  uses 0 GPU, so there is no GPU contention; and CPU/mem co-residency `8+8 = 16 CPU`,
  `64+64 = 128 GB` sits **exactly at** the per-user cap (16 CPU / 128 GB), with 1 GPU ≤ 2 — both fit.
  No `label_*` long-run jobs are queued at authorization time.
- **`sinfo`:** partition `slurmpartition` up, node `foscsmlprd01` state `mix`,
  `CPUS(A/I/O/T)=54/202/0/256` → **202 idle CPUs** (≫ 8) and `RealMemory 1000000MB` → ample node
  headroom. A CPU-only 8-core / 64 GB job is schedulable; initial `PENDING (JobHeldUser)` is expected
  and must auto-release (do not force).

**Result: PASS.**

## 7. Risk resolutions (authorization preconditions — recorded and released)

- **M-B (Medium, fail-closed) — rank-deficient-construction convergence not statically provable, no
  read-only dissolution; now the next-reachable untested runtime risk.** With the v3 plan-drift fixed
  and rows 1–13 provably PASS on disk (code review §4), `rank_deficient_structural_solution`
  (`…v4_common.py:643-700`, 30-step geometric shrink `scale *= 0.7` seeking the movement/`r_abs_max`
  window at `:693`) is the first genuinely-untested assert v4 would reach. It has never executed to
  completion in any lineage (v1 `KeyError`, v2 `jsonschema`, v3 plan-drift all died upstream). On
  window-infeasibility it raises → producer refuses to publish (**fail-closed**, never a false PASS),
  but it burns the one authorized attempt for zero science. **Risk decision (transcribed):** the
  coordination session ruled to **accept the quota-burn risk**, on three grounds — (1) fail-closed
  guarantees **no false positive**; (2) a genuine failure yields **legitimate diagnostic information
  = a valid basis for a v5 fix**; (3) a SLURM-only producer dry-run would reintroduce **v1-style
  dual-submission semantic ambiguity** (the v1 lineage was sealed for exactly that), so a dry-run is
  disfavored. This ruling was made for v3 and **carries to v4 unchanged** (M-B is byte-identical:
  `producer.py`/`common.py`/`independent_verify.py` are byte-clones of v3). **Risk acceptor =
  coordination session; this authorizer releases with knowledge** that a non-convergent fixture would
  burn the one authorized attempt fail-closed with zero science.
- **M-A (Medium; conditional-High, untriggered) — synthetic `G0` not verified PSD / rank-`≤ d`
  realizable.** The verifier checks `G0` shape, symmetry, and unit diagonal but performs no
  `G0 ⪰ 0` / `rank(G0) ≤ d` check (code review §6). Escalates to **High only if** a science authority
  rules the synthetic `G0` must itself be a realizable rank-`≤ d` PSD `Z0 Z0^T`. **No such ruling
  exists** in the materials reviewed; the KKT self-test's correctness does not depend on `G0`
  PSD-ness (correctness-neutral fidelity gap). Escalation **not triggered** → **maintained Medium,
  non-blocking.** Flagged for the science owner before any *scientific* claim rests on the fixture.
- **M-C-doc (Medium, documentation-accuracy) — resolved.** The freeze-doc §3 mis-statement was
  corrected (freeze-doc revision note, 2026-07-13); no runtime/false-PASS impact.
- **Escalation rule still in force (code review §8 item 5):** M-B lives in the numeric section, past
  preflight — a death there is a legitimate, more-informative class (still fail-closed), and does
  **not** trigger the "died at a preflight-class miss → pause ceremony, run full-chain audit before
  v5" rule. Only a *preflight-class* miss would.

---

## 8. Ledger snapshot (authorization-time)

| field | value |
|---|---|
| v4 job name | `lbscgp_global_r2_run2_v4` |
| prior v4 submissions (sacct, all-time) | **0** |
| authorized submissions remaining | **exactly 1** |
| `artifacts/lb_scgp_global/v4/` | absent |
| v4 slurm logs | none |
| `squeue -u jehc223` | `12976 tarc_g2pred` PENDING (8 CPU / 64 GB / 1 A100) — different GPU lineage, non-conflicting |
| partition / node | `slurmpartition` up / `foscsmlprd01` mix / 202 idle CPUs |
| resource request | 8 CPU / 64 GB / 0 GPU / no `--time` / HateVideo |
| config hash to verify | `e67fc6f79073b8a425ce24834db966bd5b61a2282e2247a5ef87e396056844d1` (post-binding) |
| review doc frozen at | `41650dcea19c6abd88b0755195ba9333abb43331d68e12f5a5f0d72b2a82d9dc` |

---

## 9. Executor instructions

1. **Verify the config hash first.** `sha256sum configs/lb_scgp_global_r2/m0_synth_kkt_v4.json` must
   equal `e67fc6f79073b8a425ce24834db966bd5b61a2282e2247a5ef87e396056844d1` (the **post-binding**
   hash — **not** the pre-binding `118afadf…3bf0f` recorded in freeze §1). The other eight v4
   entities must still match freeze §1. Do **not** edit `M0_RUN2_V4_CODE_REVIEW.md` (now hash-frozen
   at `41650dce…`; any byte change fails the runtime replay, fail-closed).
2. Submit **exactly once**: `sbatch scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v4.sbatch`. Do not
   set `--time`.
3. **Record the returned job id** immediately, and the resulting
   `slurm/logs/lbscgp_global_r2_run2_v4_<jobid>.{out,err}` paths.
4. Expect initial `PENDING (JobHeldUser)`; **wait for auto-release** — do not force-release.
5. **After any terminal state, do not resubmit** regardless of outcome (single-submit budget spent
   on submission). A fail-closed non-publish from M-B is an **authorized, consciously-accepted**
   outcome (§7), not grounds for re-submit — it feeds a separate v5 decision.
6. **On PASS** (producer publishes the v4 manifest + counters): route to a **fresh independent
   artifact review** (separate role) — do not self-certify.
7. **On FAILED** (job error or fail-closed non-publish): route to a **fresh result-to-claim review**
   to adjudicate the failure and decide v5, with the job log as evidence. If the death is the M-B
   convergence window (past preflight), it is the known, accepted class — it does **not** trigger the
   preflight-class full-chain-audit escalation.
8. The executor role is separate from v4-prep/freeze, code-review, and this authorization role.

---

## Required statements

- No performance evidence exists and no performance claim is made or possible from this authorization
  of a byte-exact clone + plan amendment.
- The only project gold is `parent_video_binary_label`. No segment/frame/timestamp/span/
  localization/stance/target/mechanism/rationale/fragment gold is assumed or introduced; the v4
  fixtures are synthetic and v4 has produced no artifact and no counters at authorization time.
- Run3, M1, MLLM/cache, validation/test, training, and realbank remain **locked**. This authorization
  covers only one CPU-only synthetic-certificate self-test SLURM job.
- Authorization scope = exactly one `sbatch scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v4.sbatch`.
  The M-A conditional escalation must be resolved (ruling or amendment) before any *scientific* claim
  rests on the synthetic `G0` fixture.
- Authorizer = Claude Opus 4.8 fresh 0C/0H agent, execution-authorization role, separate from the
  v4-prep/freeze, merged amendment/code-review, and executor roles. The one write beyond this
  document was the mandated post-review config binding (§3) and its freeze-doc addendum.

Report SHA256 is to be computed externally after this file is written; it is not embedded to avoid a
self-referential hash.
