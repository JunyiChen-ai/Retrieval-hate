# M0 REALBANK-RESOURCE-v1 Independent Execution Authorization

Date: 2026-07-13

Authorizer: **Claude Opus 4.8**, fresh independent **execution authorizer** for the
`lb_scgp_global_r2` M0 **REALBANK-RESOURCE-v1** run (lineage: run_id
`LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1`, machine `runs[3]`). This role is deliberately
**separate** from the realbank-prep author (recon + amendment + eight-entity implementation + freeze,
`REALBANK_RESOURCE_V1_FREEZE.md`), the merged independent amendment-ratification + fresh 0C/0H static
code-review role (`REALBANK_RESOURCE_V1_CODE_REVIEW.md`), and the executor (§8). I did not author,
review, or implement the realbank entities; I independently re-verified the authorization
preconditions from files and cluster state read directly in this session.

## Authorizer boundary

Read-only authorization. **No mandated write obligation exists for this run** — unlike the v4
ceremony, the realbank config's `hash_bindings.authoritative_inputs` is already complete at **10**
entries and does **not** bind the code-review document (verified: `grep -c REALBANK_RESOURCE_V1_CODE_REVIEW`
on the config = 0), so there is no post-review config re-binding step and the executor verifies the
**frozen** config hash `c436c3dd…`, not a post-binding hash. Apart from writing this authorization
document, I wrote nothing. I did **not** run project Python, imports, `py_compile`, tests, `conda`,
`sbatch`, an experiment, or any MLLM/OCR/API/model/network/GPU/training/evaluation, and I touched no
validation/test/held/cache/query content (only the two allowlisted train **feature** banks and the
authoritative/plan hashes were `sha256sum`-checked, read-only). No artifact under
`artifacts/lb_scgp_global/v1/m0/realbank_resource/` was created (confirmed absent, §5).

**Model-binding divergence declaration** (precedent: `M0_RUN2_V4_EXECUTION_AUTHORIZATION.md`
§Model-binding, `REALBANK_RESOURCE_V1_CODE_REVIEW.md` §Opus 4.8 deviation): project discipline
(`CLAUDE.md`) binds subagents to **Opus 4.8** (`claude-opus-4-8`); `AGENTS.md` names a "GPT-5.5 xhigh"
backend which is unavailable this session. This authorization therefore runs on Opus 4.8. Independence
is enforced by fresh-context re-derivation of every check from on-disk / cluster state. Documented
process fact, not a defect; does not affect any ruling below.

---

## Verdict

**AUTHORIZED — exactly one CPU-only SLURM submission.**

All five authorization checks pass; there is no post-review binding obligation. Authorization scope =
**exactly one** invocation of:

```
sbatch scripts/slurm/lb_scgp_global_r2_m0_realbank_resource_v1.sbatch
```

The residual runtime risks (R-2 cross-process replay bit-determinism = highest residual, fail-closed;
R-1 `torch.load(weights_only=True)`; and the two documentation-precision Low-1 deviations, both
verified decision-inert) are non-blocking under the resolutions recorded in §7, which the coordination
session ruled on and this authorizer releases **with knowledge**. In particular, a fail-closed
`decision:FAIL`/non-GO (e.g. a cross-process replay mismatch or a rank/cap miss) is a **known,
consciously-accepted STOP outcome, not grounds for re-submission** — the single-submit budget is spent
on submission.

---

## 1. Check 1 — Static-review credential (PASS)

`refine-logs/lb_scgp_global/REALBANK_RESOURCE_V1_CODE_REVIEW.md` exists and reads:

- **Verdict:** **`AMENDMENT_RATIFIED`** (the A/B/C ruling and the additive `runs[3]` edit are correct,
  minimal, faithfully landed; REPLACE-nothing/additive-only discipline holds; hash cascade exact)
  **and `PASS_STATIC_REVIEW`** (eight-entity implementation interface-aligned three ways, index-pinned
  to `runs[3]`, dependency-clean, resource-correct, fail-closed on every runtime assertion).
- **Severity counts:** **Critical = 0, High = 0, Medium = 0, Low = 2** (both documentation-precision:
  Low-1 "byte-faithful" wording, Low-2 two out-of-scope stale v2-era summary lines). Pass criterion
  (Critical = 0 ∧ High = 0) is met.
- The reviewer is a fresh 0C/0H role, separate from prep/authorization/executor, who independently
  re-derived the 21-row runtime cross-check simulation table (no row FAILs), the amendment hash
  cascade, and the three-way interface alignment (23 top-level keys, 47 zero-counters, 11 isolation
  cases) rather than trusting the freeze predemonstration.

**Result: PASS.** The static-review credential is present, is a PASS with Critical=0/High=0, and the
amendment is ratified.

## 2. Check 2 — Freeze-state no-drift (PASS)

I re-`sha256sum`'d all **eight** realbank entities this session and compared to
`REALBANK_RESOURCE_V1_FREEZE.md` §1. **All eight match byte-for-byte:**

| # | entity | SHA256 (freeze §1) | on-disk |
|---|---|---|---|
| 1 | `configs/lb_scgp_global_r2/m0_realbank_resource_v1.json` | `c436c3dd…d06fbcf` | ✓ |
| 2 | `schemas/lb_scgp_global_r2/scgp_global_realbank_resource_v1.schema.json` | `db79cdd3…be7a73d2` | ✓ |
| 3 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v1_common.py` | `46e1f3fe…4f41b8a9` | ✓ |
| 4 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v1_validate.py` | `b2bbec02…51611ded` | ✓ |
| 5 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v1_producer.py` | `dc38d5c3…7241c114` | ✓ |
| 6 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v1_independent_verify.py` | `49cc2d9a…70f54f26` | ✓ |
| 7 | `scripts/wrappers/lb_scgp_global_r2_realbank_resource_v1.sh` | `f80b41ea…72d4b0a1` | ✓ |
| 8 | `scripts/slurm/lb_scgp_global_r2_m0_realbank_resource_v1.sbatch` | `9c4ecc05…0308ab420` | ✓ |

**Machine plan pin:** `EXPERIMENT_PLAN.machine.json` on-disk =
`d5023b621afca08eed940d2853551c872da8151378c4a83075744e422cb18fdb` = the amendment target
`d5023b62…cb18fdb`; the config's `machine_plan` binding internally holds the same `d5023b62…` value
(verified). **Plan pin MATCH.**

**`runs[3]` content ↔ config field-for-field (the v3-death / index-drift class):** `jq '.runs[3]'`
confirms `run_order[3]` = `runs[3].run_id` = `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1`;
`artifact_paths` = `[artifacts/lb_scgp_global/v1/m0/realbank_resource/decision.json]`;
`artifact_schema_ids` = `[scgp_global_realbank_resource_v1]`; `slurm` =
`{cpu:16, ram_gb:96, gpu:0, env:"HateVideo", no_time_flag:true}`; `dependencies` =
`[LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v4]`; `realbank_protocol.A…banks` = MHC (`deea74ff…`, N=549) /
MHC_zh (`929571f8…`, N=579). All match the config and the code's `RUN3` literal.

**Result: PASS.** Zero drift at authorization start; the eight entities, the machine plan pin, and
the `runs[3]`-vs-code lock-step all hold.

## 3. Check 3 — Dependency-availability evidence (PASS) — env is not frozen

Read-only `ls` of the exact `HateVideo` interpreter tree that the sbatch's `conda activate HateVideo`
resolves (`/data/jehc223/miniconda3/envs/HateVideo/lib/python3.11/site-packages/`, with
`.../bin/python -> python3.11`) shows the full required third-party set `{numpy, torch, jsonschema}`
plus jsonschema's ABI-sensitive transitive closure:

- **`numpy` 1.26.4** — present (`numpy/`, `numpy-1.26.4.dist-info/`).
- **`torch` 2.6.0** — present (`torch/`, `torch-2.6.0.dist-info/`). Used function-level in both bank
  loaders (`common.load_bank_features` and the verifier copy) via `torch.load(..., weights_only=True)`;
  torch 2.6 makes `weights_only=True` the default and fully supports it (R-1 resolved, §7).
- **`jsonschema` 4.26.0** — present, with `jsonschema_specifications` 2025.9.1, `referencing` 0.37.0,
  `rpds_py` 2026.6.3 (the `cp311` C-extension matching the py3.11 interpreter). Used function-level in
  both schema validators.

**Runtime-import corroboration** (a listing proves *installed*, not *imports cleanly*):
`M0_ENV_REPAIR_RECORD.md` (2026-07-13, env-repair role) supplies the runtime evidence — the missing
`jsonschema` was installed into the **correct** HateVideo tree (avoiding the documented
`source activate` silent-fallback-to-ExMRD trap, and rolling back a first mis-install into ExMRD);
§3 shows `import jsonschema → 4.26.0 .../HateVideo/.../jsonschema/__init__.py` (runtime import OK),
§4 shows the exact deferred triple `from jsonschema import Draft7Validator, RefResolver;
from jsonschema.exceptions import SchemaError` reproduced runtime-OK, `numpy` imported OK (1.26.4),
and `py_compile` of the four `.py` modules passed. §2 confirms the sbatch's
`source conda.sh; conda activate HateVideo` plus the `require_slurm_realbank` `CONDA_DEFAULT_ENV=="HateVideo"`
gate guarantee the job runs in exactly the evidenced HateVideo tree.

The SLURM validator's `python_dependency_check` (`importlib.util.find_spec` over exactly
`{numpy, torch, jsonschema}`, `sys.exit(1)` on any miss, **inside** the job **before** the producer)
is the specific v2-death (missing `jsonschema`) prevention and is fail-closed.

**Result: PASS.** The env is not frozen and carries the full required set. Any residual breakage
(incl. the `RefResolver` DeprecationWarning forward-risk — no `PYTHONWARNINGS=error` set in the
sbatch) raises → refuse, never a false PASS.

## 4. Check 4 — Single-submit ledger (PASS)

- **`sacct` all-time for the realbank job name is empty.**
  `sacct -u jehc223 --name=lbscgp_global_r2_realbank_resource_v1 --starttime=2000-01-01` returns
  **zero rows** — the realbank job has **never been submitted**.
- **`artifacts/lb_scgp_global/v1/m0/realbank_resource/` does not exist** (`ls` → "No such file or
  directory"). No `decision.json` / manifest / access-ledger / publish-lock exists.
- **`slurm/logs/` has no same-name output:** `find slurm/logs -name '*realbank_resource*'` → 0 files.

**Result: PASS.** The single-submit budget is intact; this authorizes the first and only realbank
submission.

## 5. Check 5 — Resources (PASS)

- **sbatch request** (`lb_scgp_global_r2_m0_realbank_resource_v1.sbatch`, re-read this session):
  `--cpus-per-task=16`, `--mem=96G`, **no** `--gres`/`--gpus` (GPU = 0), **no `--time`** (line 8
  comment "Intentionally no --time: project policy"), `--job-name=lbscgp_global_r2_realbank_resource_v1`,
  `--partition=slurmpartition`, and `source conda.sh; conda activate HateVideo` (lines 12–13) with
  `OMP/MKL/OPENBLAS_NUM_THREADS=16` (the R-2 determinism precondition). Matches machine `runs[3].slurm`
  `{cpu:16, ram_gb:96, gpu:0, env:"HateVideo", no_time_flag:true}` exactly. Within per-user caps
  (16 CPU / 128 GB / 2 GPU): 16 ≤ 16 CPU, 96 ≤ 128 GB, 0 ≤ 2 GPU.
- **`squeue -u jehc223` is empty at authorization time** — **no** running or pending jobs of mine.
  The 16-CPU / 96-GB request therefore consumes the CPU cap exactly (16 + 0 = 16 ≤ 16) with **no
  contention**; there is no competing CPU or GPU lineage to over-subscribe. (Had a competing CPU job
  been present, SLURM would enforce the cap by holding the realbank job PENDING — it never
  over-allocates — but no such job exists now, so immediate scheduling is expected after the
  `JobHeldUser` auto-release.)
- **`sinfo`:** partition `slurmpartition` up, node `foscsmlprd01` state `mix`,
  `CPUS(A/I/O/T)=38/218/0/256` → **218 idle CPUs** (≫ 16) and `RealMemory 1000000MB` → ample node
  headroom. A CPU-only 16-core / 96 GB job is schedulable; initial `PENDING (JobHeldUser)` is expected
  and must auto-release (do not force).

**Result: PASS.**

## 6. No-clobber / hash re-confirmation summary

The 8 frozen entities match (§2), the 10 `authoritative_inputs` and the machine plan pin are bound at
`d5023b62…` (config internal binding verified), the output artifact directory is absent (§4), the env
carries `{numpy 1.26.4, torch 2.6.0, jsonschema 4.26.0 (+transitive)}` (§3), and the resource request
is cap-conformant with zero contention (§5). No post-review binding obligation applies (§Authorizer
boundary). Nothing to clobber; nothing to re-freeze.

## 7. Risk resolutions (authorization preconditions — recorded and released)

- **R-2 (cross-process replay bit-determinism) — ACCEPT (highest residual, fail-closed).** In-job
  run1==run2 is same-process → bit-identical (a GO criterion). The producer↔independent-verifier
  cross-check *additionally* requires the verifier's independently recomputed replay digest to
  byte-match the producer's; this relies on LAPACK `eigvalsh`/`svd` determinism at a fixed thread
  count (`OMP/MKL/OPENBLAS=16`, set in the sbatch and shared by both processes), with `floatify`
  (15 sig figs, `<5e-16 → 0`) absorbing last-ULP noise. On real-CLIP eigen-tails the cross-process
  risk is modestly higher than v4's synthetic data, but a miss yields `decision:FAIL` (**STOP**),
  **not** a false GO. **Risk acceptor = coordination session; this authorizer releases with
  knowledge** that a cross-process replay mismatch would burn the one authorized attempt fail-closed.
- **R-1 (`torch.load(weights_only=True)`) — ACCEPT (Low residual, fail-closed).** Identical to the
  proven `lb_scgp_sanitize_inputs.py` call on the same `{ids,img_feats,text_feats}` bank family;
  `dataset.py` confirms the bank layout. The installed **torch 2.6.0** supports (and defaults to)
  `weights_only=True`; if it rejected the payload it raises cleanly (no artifact), never a silent
  wrong result. Confirmed present in §3.
- **Two Low-1 decision-inert deviations from the accepted v4 math — ACCEPT (documentation-precision
  only).** (i) `factor_from_psd_gram` omits v4's `if audit["status"] != "PASS": return None, audit`
  early-return; but `y`/`zstar_gram_residual`/`nondegenerate` are used **only** inside the replay
  digest, and the GO gate independently requires `rank_audit["status"] == "PASS"` per dataset
  (producer line 182 / verifier line 589), so the omission **cannot manufacture a GO** — it only makes
  the resource measurement marginally more conservative. (ii) `orth_cap` omits an unused
  `singular_values` field from a `q_info` dict that is assigned but **never read**. Both independently
  verified inert by the code review (§2.3); non-blocking.
- **NON-SCIENCE placeholder (R-4) — disclosed end-to-end.** `b_struct` is a deterministic, label-blind
  placeholder (`is_science=false`, re-checked by the verifier, which REJECTS `is_science=True`); it
  certifies nothing and the science owner retains the M-A-analogue right to overrule it before any
  downstream scientific claim rests on it. This run emits **no accuracy / macro-F1** and does no
  training or kNN.

## 8. Ledger snapshot (authorization-time)

| field | value |
|---|---|
| realbank job name | `lbscgp_global_r2_realbank_resource_v1` |
| prior submissions (sacct, all-time) | **0** |
| authorized submissions remaining | **exactly 1** |
| `artifacts/lb_scgp_global/v1/m0/realbank_resource/` | absent |
| realbank slurm logs | none |
| `squeue -u jehc223` | **empty** — no running/pending jobs; no contention |
| partition / node | `slurmpartition` up / `foscsmlprd01` mix / 218 idle CPUs / 1000000 MB |
| resource request | 16 CPU / 96 GB / 0 GPU / no `--time` / HateVideo |
| config hash to verify | `c436c3dd7e5342707a3ee1a16662e4ab0a74cd0fe39442002cabc2fd6d06fbcf` (frozen; **no** post-binding move) |
| machine plan pin | `d5023b621afca08eed940d2853551c872da8151378c4a83075744e422cb18fdb` |
| GO criterion | `peak_rss ≤ 103079215104 (96 GiB) ∧ rank_eps(G0) ≤ d (all ds) ∧ in-job replay match (all ds) ∧ all injections REJECT` |

---

## 9. Executor instructions

1. **Verify the eight frozen hashes first**, especially
   `sha256sum configs/lb_scgp_global_r2/m0_realbank_resource_v1.json` ==
   `c436c3dd7e5342707a3ee1a16662e4ab0a74cd0fe39442002cabc2fd6d06fbcf` (the **frozen** hash — there is
   **no** post-review binding move for realbank). The other seven entities must match §2.
2. Submit **exactly once**: `sbatch scripts/slurm/lb_scgp_global_r2_m0_realbank_resource_v1.sbatch`.
   Do not set `--time`.
3. **Record the returned job id** immediately, and the resulting
   `slurm/logs/lbscgp_global_r2_realbank_resource_v1_<jobid>.{out,err}` paths.
4. Expect initial `PENDING (JobHeldUser)`; **wait for auto-release** — do not force-release. At
   N=549/579 the pure-CPU pipeline is expected to finish in minutes once running.
5. **After any terminal state, do not resubmit** regardless of outcome (single-submit budget spent on
   submission). A fail-closed `decision:FAIL`/non-GO (e.g. R-2 cross-process replay mismatch, a rank
   or cap miss, or an injection failing to REJECT) is an **authorized, consciously-accepted STOP
   outcome** (§7), not grounds for re-submit — it feeds a separate result-to-claim / v-next decision.
6. **On GO/PASS** (verifier stamps `decision:PASS`, producer publishes the manifest + counters): route
   to a **fresh independent artifact review** (separate role) before any downstream unlock — do not
   self-certify. Collect `decision.json` (GO/STOP), `resource_peak` (peak RSS bytes) and its distance
   to the 96 GiB cap, `rank_tail`, the replay-hash match, and the 11 injection + 15 tamper REJECTs.
7. **On FAIL / fail-closed non-publish**: route to a **fresh result-to-claim review** with the job log
   as evidence.
8. The executor role is separate from realbank-prep/freeze, amendment/code-review, and this
   authorization role.

---

## Required statements

- No performance evidence exists and no performance claim is made or possible from this authorization.
  The realbank run itself emits **no accuracy / macro-F1** and does **no training or kNN** — it is a
  train-bank static/resource microbenchmark (`resource_peak`, `rank_tail ≤ d`, in-job replay
  determinism, `robust_coverage` fail-open, isolation-injection defenses).
- The only project gold is `parent_video_binary_label`. No segment/frame/timestamp/span/localization/
  stance/target/mechanism/rationale/fragment gold is assumed or introduced; the realbank code opens
  train **features** (allowlisted + hash-checked) and never reads train **labels**.
- Run4 (M1 cache), MLLM/cache, validation/test, and training remain **locked**. This authorization
  covers only one CPU-only static/resource microbenchmark SLURM job. The `is_science=false` placeholder
  (R-4) must be overruled or replaced by the science owner before any *scientific* claim rests on it.
- Authorization scope = exactly one `sbatch scripts/slurm/lb_scgp_global_r2_m0_realbank_resource_v1.sbatch`.
- Authorizer = Claude Opus 4.8 fresh independent execution-authorization role, separate from the
  realbank-prep/freeze, merged amendment/code-review, and executor roles. This authorization made no
  write beyond this document (no config re-binding obligation applies to realbank).
