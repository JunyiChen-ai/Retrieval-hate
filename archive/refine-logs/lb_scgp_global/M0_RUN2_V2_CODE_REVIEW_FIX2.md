# M0 Run2-v2 FIX2 Fresh Static Code Review

Date: 2026-07-13

Reviewer: Claude Opus 4.8 (fresh 0C/0H). Per `CLAUDE.md` the main-dialogue subagent
model binding is Opus 4.8 (`claude-opus-4-8`). `AGENTS.md:15` instead names
"GPT-5.5 xhigh" as the subagent model; that backend is not available for this
session's subagent, so this review runs on the `CLAUDE.md`-bound Opus 4.8. This
model-binding divergence between `AGENTS.md` and `CLAUDE.md` is recorded here and
is a documentation inconsistency only, not a code defect.

## Reviewer boundary

Fresh, zero-context, zero-history (0C/0H) independent static code review only. I
read `AGENTS.md` first, then `M0_RUN2_V2_IMPLEMENTATION_FIX2_FREEZE.md` and the
prior `M0_RUN2_V2_CODE_REVIEW.md`. I did **not** run Python, Python imports,
`py_compile`, tests, `conda`, SLURM, `sbatch`, `squeue`, experiments, MLLM/OCR/API/
model/network/GPU/training/evaluation, or validation/test data/cache inspection.
Shell was limited to the allowed static tools: `rg`, `sed`/`nl`, `jq`, `awk`,
`bash -n`, `sha256sum`, `find`, `ls`, `wc`, `git status`, `git diff`. I did not
rely on any prior-round conclusion; every ruling below is grounded in files I read
directly. I independently re-adjudicated the prior H5/H6/M2/L2 rather than copying
their dispositions. The only file I wrote is this report. No artifact under
`artifacts/lb_scgp_global/v2` was created (the directory is absent).

This review does not authorize SLURM execution. A pass permits only entry into a
separate execution-authorization step for exactly one future SLURM validation job.

## Verdict

**PASS_STATIC_REVIEW**

Severity counts:

| Severity | Count |
|---|---:|
| Critical | 0 |
| High | 0 |
| Medium | 3 |
| Low | 2 |

0 Critical / 0 High is achieved. Medium/Low items do not block but are listed and
must be understood before execution authorization. One conditional escalation is
flagged (M-A): if a future authority rules that even the *synthetic* fixture's `G0`
must be a realizable rank-`<=d` PSD `Z0 Z0^T`, then M-A becomes High and blocks; a
minimal amendment path is given.

## Hash reconciliation

All nine v2 entities plus the four binding documents hash **exactly** to the values
declared in `M0_RUN2_V2_IMPLEMENTATION_FIX2_FREEZE.md`. No drift.

| Path | SHA256 | Freeze-doc match |
|---|---|---|
| `configs/lb_scgp_global_r2/m0_synth_kkt_v2.json` | `5545826820cd86f588fb43fd73b4070767fb9f904ea683fc2245f0903d48f700` | yes |
| `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_payload_v2.schema.json` | `735250091f6a92ef787d9eadccca3c438379cc07f2418439401916270eced9a2` | yes |
| `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_case_v2.schema.json` | `df3616ff50c54dc756bb600c6ef26626afafe070678450d10ca73b9ff83ddcac` | yes (unchanged parity) |
| `scripts/analysis/lb_scgp_global_r2_run2_v2_common.py` | `5ef8f677e13c0586675a3072b85774b90b7eb5e6ba8da91188f128d3a6d6e24f` | yes |
| `scripts/analysis/lb_scgp_global_r2_run2_v2_validate.py` | `4389c4a1d1cbe21ec516ec414c472ebff075bbfb4a8041939163f478bdc3fc36` | yes |
| `scripts/analysis/lb_scgp_global_r2_run2_v2_producer.py` | `8c4cc842aa53e3d8293744449101282a33abb7715691653bb934b2e563b1cd51` | yes |
| `scripts/analysis/lb_scgp_global_r2_run2_v2_independent_verify.py` | `795b56f852534c2cfb4264c9dec2f43dd4abc75a4655282465b6546d39111ef1` | yes |
| `scripts/wrappers/lb_scgp_global_r2_run2_v2.sh` | `14eb036f17b77037fd89624c0f9f7487432fa476da540b0fbc5f503719320716` | yes (unchanged parity) |
| `scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v2.sbatch` | `f91462303e4b28eb0722d546f135c00c2aeacf56151e27d5421f211520ba94bf` | yes (unchanged parity) |

Binding-document reconciliation:

- `AGENTS.md` on disk = `e6aaf5d66399cdbbe7fcc2c811931277b0ed4a24b592ffa5cbb60315b29ea23c`,
  equal to `config.hash_bindings.authoritative_inputs["AGENTS.md"]`.
- `EXPERIMENT_PLAN.md`, `EXPERIMENT_PLAN.machine.json`, `EXPERIMENT_TRACKER.md`,
  `EXPERIMENT_PLAN_HASHES.sha256` all equal both the freeze-doc values and their
  config bindings (`af1c217c…`, `6caa5c2e…`, `327614bb…`, `2e6d731d…`).
- `sha256sum -c` **OK** for `EXPERIMENT_PLAN_HASHES.sha256`,
  `M0_RUN2_V2_PLAN_AMENDMENT_HASHES.sha256`, and
  `M0_RUN2_V2_PLAN_AMENDMENT_REVIEW_HASHES.sha256`.
- The fresh-review hash bound in config
  (`M0_RUN2_V2_CODE_REVIEW.md` = `be646c3a…`) matches the freeze-doc statement.

No Critical from hash drift.

## Interface-contract re-adjudication (v1 KeyError classes: `finite_vi_diagnostic`, `payload_schema`)

Structurally repaired.

- The field key is uniformly `finite_vi_diagnostics` (plural) in the producer path
  (`…_common.py:1047`), the verifier (`…_independent_verify.py:89` in `CASE_KEYS`,
  `:599` `verify_finite_vi`, `:1216` injection), and both schemas
  (`…_case_v2.schema.json:92,244`). The only singular occurrences are the injection
  *label* string `malformed_finite_vi_diagnostic` (a dict key, not a field access)
  in producer/verifier/payload-schema. No code path accesses a singular
  `finite_vi_diagnostic` field. The v1 KeyError is closed.
- `payload_schema` / `case_schema` / `cert_schema` are always read via
  `cfg["paths"][…]`, which the config defines (`config:98-102`), across all four
  Python modules. The v1 `payload_schema` KeyError is closed.
- Three-stage key contracts align exactly:
  - Case public keys emitted (`…_common.py:1079-1080`, 18 keys) == verifier
    `CASE_KEYS` (`:74-93`, enforced by `set(case) != CASE_KEYS` at `:507`) ==
    case-schema `required` (18 keys).
  - Manifest top-level keys (`…_producer.py:295-386`, 32 keys) == verifier
    `TOP_KEYS` (`:40-73`, enforced at `:1035`) == payload-schema `required`
    (32 keys).
  - Injection set (21, `…_independent_verify.py:1151-1217`) == manifest
    `injection_results_expected` (21, `…_producer.py:331-353`) == payload-schema
    `required` (21), with `set(expected)==set(results)` enforced at `:1227`.

Chain `validate -> producer -> independent_verify`: `validate.py` writes a
validation JSON that the producer requires with `status=="PASS"`
(`…_producer.py:264-267`); the producer publishes the manifest; the verifier
re-derives everything from the serialized manifest. Keys/schema contracts are
consistent at each hop.

## Prior-finding closure (independently re-verified)

### H5 (producer rank contract) — CLOSED as implementation repair; residual → M-A

**(b.i) Mathematical self-consistency: CONFIRMED.**

For non-`REMOVE` cases the producer now calls `rank_deficient_structural_solution`
(`…_common.py:643-698`, byte-identical replay in the verifier
`…_independent_verify.py:385-433`):

- `G_star = rank_d_correlation_target(N,d,…)` = `V Vᵀ` with `V ∈ R^{N×d}`
  row-normalized (`:596-630`). Hence `G_star ⪰ 0`, `diag(G_star)=1`,
  `rank(G_star) <= d`. Off-diagonal box `< 1-1e-4` is enforced (`:628`). This
  directly resolves the prior Welch-bound impossibility: `G_star` is genuinely
  rank-`d`, so `rank_tail_audit`/`factor_from_psd_gram` (`:701-755`) return
  `rank_eps <= d` and `status=PASS` (no longer a statically guaranteed
  `rank>d` failure).
- `S_psd = psd_scale · P_ker(G_star)` with `P_ker` the projector onto `ker(G_star)`
  (`:633-679`). `S_psd ⪰ 0` and `S_psd G_star ≈ 0`, so PSD-cone complementarity is
  exact.
- Affine (structural) normal `-A_structᵀ ν` on `G`, `ν` on `r`; diagonal affine
  dual `diag(A_structᵀ ν)+diag(S_psd)`.
- Movement `= offdiag(A_structᵀ ν) + offdiag(S_psd)`, `G0 = G_star - movement`,
  diagonal reset to 1. By hand, `G`-stationarity
  `movement + (-A_structᵀν + diag(diag adj + diag S) - S_psd) = 0` and
  `r`-stationarity `λ·r + ν = 0` with `r=-ν/λ`, `λ=1` cancel to **exactly 0**
  (float ≈ 1e-16, `floatify`→0). Verifier recomputes both
  (`…_independent_verify.py:787-790`, `:869`).
- Dual feasibility: `S_psd ⪰ 0` (`dual_lambda_min >= -1e-7`), all box/SOC/coord
  multipliers 0 (`zero_normal_block`). Complementarity: `tr(S_psd G_star) ≈ 0`.
- Frozen constants preserved: `d=3`, `lambda_struct=1`, off-diagonal box
  `delta=1e-4`, unit diagonal, PSD on `G`, **coordinate trust `rho_coord=0.02`
  honored** (constructed `|movement|_offdiag <= 0.018 < 0.02`, so
  `coordinate_trust_violation` gates to 0), positive movement `> 0.005`. The
  identity `I_N` still appears only in the centered metric `M_Q(G)=Qᵀ(G−I_N)Q/N`
  and is untouched.

The KKT point satisfies stationarity + dual feasibility + complementarity + primal
feasibility for a strongly convex program → certified global optimum. Math is
self-consistent.

**(b.ii) Semantic verdict: this is repair of an implementation flaw, NOT a change
of frozen scientific semantics.**

Grounds (from the frozen proposal `FINAL_PROPOSAL.md`):

1. `G0` is never the identity in frozen science. `FINAL_PROPOSAL.md:130` defines
   `G0 = Z0 Z0ᵀ ∈ S^N`, a *baseline unit-diagonal Gram*. The notation `S^N` is the
   space of **symmetric** matrices (cf. `:264` "`G in S^N`" with the PSD property
   imposed *separately* as a hard constraint at `:282`). v1's hard-coded `g0=I_N`
   was itself a v1 implementation choice, not a frozen object.
2. The frozen convex program has **no hard rank constraint** (`:291` verbatim:
   "no hard rank constraint"). Rank-`<=d` is a post-hoc encoder-realizability audit
   on the **solution** `G*`, which FIX2 honors exactly (`G_star=VVᵀ`, rank `<=d`).
3. FIX2's `G0` is symmetric and unit-diagonal, consistent with `G0 ∈ S^N`; all
   frozen constants (`d`, `lambda_struct`, `rho_coord`, `delta`, unit diagonal,
   PSD-on-`G`, positive movement) are preserved.
4. Run2 is explicitly a **synthetic KKT self-test**; its contract is the serialized
   certificate structure and `G*` realizability (`FINAL_PROPOSAL.md:334-369`), not
   `G0` provenance. Back-constructing `(G0,G*,duals)` to sit at a KKT point is the
   intended synthetic methodology.

The freeze document itself flagged the opposite reading as a live risk
(`…FIX2_FREEZE.md:71`). Having independently examined the frozen text, I rule the
baseline change (away from `I_N`) is legitimate implementation repair. The residual
fidelity concern is recorded as **M-A** below, with the escalation/amendment path.

### H6 (verifier per-case semantic recompute) — CLOSED

The verifier now, for **every** case, recomputes and compares each serialized block
in `verify_case_serialized_metrics` (`…_independent_verify.py:695-729`), invoked in
the per-case loop (`:1066-1073`):

- `movement_metrics` (`:696-698`), `primal_residuals` (`:699-708`),
  `rank_audit` (`:709-711`), `factor_replay` (`:712-714`),
  `robust_coverage` (`:715-716`), and the recomputed `kkt_status` (`:717-729`)
  are each equality-checked against the serialized values.
- The rank-failure probe's serialized `rank_audit` is recomputed and compared
  (`:1119-1125`), not merely `factor_returned_null`/status.
- Refreshed-hash negative injections now exist for every drift class:
  `bogus_case_movement_metrics`, `bogus_case_primal_residual`,
  `bogus_case_rank_audit`, `bogus_case_factor_replay`,
  `bogus_case_robust_coverage`, `bogus_rank_failure_audit`
  (`:1173-1190`), using `refresh_case_and_matrix` which re-computes
  `case_payload_sha256`, `case_matrix_sha256`, and `payload_sha256`
  (`:1145-1149`). This removes the prior "hash refreshed but semantics
  unchecked" blind spot.

The prior H6 required fix is satisfied.

### M2 (`jsonschema` availability) — remains fail-closed → M-C

`validate.py` now preflights the dependency with the job Python via
`importlib.util.find_spec` and exits non-zero (fails closed) if missing
(`…_validate.py:98-108,148`), before any producer publish. Both producer
(`…_common.py:181-185`) and verifier (`…_independent_verify.py:166-170`) also raise
if `jsonschema` is unavailable. Missing `jsonschema` cannot yield a false PASS. As a
static reviewer I cannot prove the package is installed in `HateVideo`; the risk is
retained as fail-closed environmental uncertainty (M-C).

### L2 (`EXPERIMENT_PLAN.md` stale wording) — CLOSED

`EXPERIMENT_PLAN.md` now states uniformly that fix2 is complete but locked pending a
fresh 0C/0H code review, exact hashes/no-clobber review, and separate execution
authorization (`:7,156,192,249,256`); the machine plan holds `ready_for_execution=false`
and `ready_for_fresh_independent_v2_code_review=true` (`…machine.json:4176-4178,4202`).
The prior conservative wording is aligned.

## Negative-injection review (item d)

All 21 injections trace to a concrete rejection path, covering the required failure
modes:

- Schema-layer rejects (const/`additionalProperties:false`): `wrong_dual_sign`
  (`v_psd=+S_psd` vs const `v_psd=-S_psd`), `incomplete_cone_family` (drops
  `soc_normals`, required), `invalid_extra_missing_schema_fields` (adds key),
  `rank_failure` (`factor_returned_null=false` vs const `true`),
  `malformed_normal_presence` (`present=false` vs const `true`).
- Hash rejects: `perturbed_artifact_source_operator_hash`
  (`operator_hash` vs FULL-case hash, `:1013`).
- Path reject: `forbidden_path` (`source_manifest_path=data/gt/MHC/test.jsonl` vs
  binding, `:1000`).
- Semantic-recompute rejects: `nan_overflow` (top-level `G_star` array mismatch
  under `require_array_close`), the six case-level `bogus_*` and
  `bogus_rank_failure_audit`, `malformed_dual_status`/`malformed_complementarity`
  (dict/per-family mismatch), `malformed_stationarity_status` (`:862`),
  `malformed_normal_residual` (zero-block tol 1e-12, `:748`),
  `malformed_finite_vi_diagnostic` (exact-dict `verify_finite_vi`),
  `finite_vi_only_attempted_acceptance` (`acceptance_path` guard `:1050`),
  `identity_no_movement_claims_full` (FULL no-movement guard `:1088`).

Injection key set matches `injection_results_expected` and is required to be
uniformly `REJECT` (`:1227-1233`). No injection is executed by this review.

## Wrapper and sbatch (items e, f)

Wrapper `scripts/wrappers/lb_scgp_global_r2_run2_v2.sh`:

- `set -euo pipefail`; `cd` to repo root; `RUN_ID`, config `run_id`, and config
  `artifact_path` guards (`:43-57`).
- Ordered `validate -> producer -> independent_verify`, then
  `jq -e '.decision == "PASS"'` gates `COMPLETE=1` (`:60-76`).
- `cleanup_on_exit` (EXIT trap) always removes the temp validation JSON and, when
  `COMPLETE != 1`, removes all eight prospective outputs; HUP/INT/TERM route through
  the EXIT trap (`:23-41`). Because the verifier exits non-zero on FAIL and
  `set -e` fires, failure paths never reach the `jq` gate and outputs are cleaned.
- The eight `PROSPECTIVE_OUTPUTS` equal the config `allowed_new_files_after_run`
  (config `:19-28`). No-clobber is enforced independently by `validate.py`
  (`:79-95`) and the producer pre-check (`…_producer.py:254-262`).
- `bash -n` OK. One robustness footgun recorded as **L-A**.

sbatch `scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v2.sbatch`:

- `--cpus-per-task=8`, `--mem=64G`, **no `--time`** (only a comment), job name and
  `%x_%j.{out,err}` under `/data/jehc223/RGCL/slurm/logs/`, `conda activate
  HateVideo`, **no GPU requested**, calls the wrapper with `RUN_ID`/`CONFIG`
  (`:2-24`). `bash -n` OK.
- Matches config `run.slurm` `{cpu:8, ram_gb:64, gpu:0, env:"HateVideo",
  no_time_flag:true}` and `validate.resource_and_run_check` (`…_validate.py:118`);
  `require_slurm_run2` enforces 8 CPU / 64 GB / 0 GPU / `HateVideo` at runtime
  (`…_common.py:288-302`).

`git diff --check` on the nine entities + tracker: clean (exit 0). No trailing
whitespace flagged. `artifacts/lb_scgp_global/v2` absent.

## Findings

### M-A (Medium): synthetic `G0` is back-constructed and not verified PSD / rank-`<=d` realizable

Evidence: `G0` is produced as `G_star − offdiag(A_structᵀν) − offdiag(S_psd)` with
the diagonal forced to 1 (`…_common.py:687-689`), i.e., derived *from* `G_star`
rather than realized as an embedding Gram `Z0 Z0ᵀ` (`Z0 ∈ R^{N×d}`). The verifier
checks `G0` symmetry and unit diagonal (`…_independent_verify.py:1090-1093`) but
does **not** check `G0 ⪰ 0` (PSD) or `rank(G0) <= d`. Generically `G0` here is
full-rank and may have a slightly negative eigenvalue.

Impact: the fixture's baseline is less faithful to the real-method baseline
(`FINAL_PROPOSAL.md:129-130`, where `G0=Z0Z0ᵀ` is always a rank-`<=d` PSD encoder
Gram). It does **not** break the KKT self-test (the certificate is valid for any
anchor `G0`, and `G0 ∈ S^N` = symmetric is satisfied). Non-blocking under the
stated frozen requirement (symmetric, unit-diagonal).

Conditional escalation: if a future authority rules that the synthetic fixture's
`G0` must itself be a realizable rank-`<=d` PSD `Z0Z0ᵀ` (not merely symmetric
unit-diagonal), this finding becomes **High** and blocks. Minimal amendment path,
any one of:
1. Construct `Z0 ∈ R^{N×d}` deterministically, set `G0=Z0Z0ᵀ`, then set
   `G* = Z*Z*ᵀ` from a small in-`R^{N×d}` rotation/perturbation `Z*` of `Z0`
   within `rho_coord`, and re-verify KKT closure; or
2. Add explicit `G0 ⪰ 0` and `rank(G0) <= d` checks to the independent verifier
   and have the producer construct `G0` to satisfy them; or
3. Formally amend the frozen definition to state that the synthetic-fixture anchor
   `G0` need only be symmetric and unit-diagonal.

### M-B (Medium, fail-closed): rank-deficient construction convergence is not statically guaranteed

Evidence: `rank_deficient_structural_solution` searches over a 30-step geometric
shrink (`scale *= 0.7`) for a `scale` simultaneously satisfying
`0.005 < movement_off_max <= 0.018`, `movement_fro > 0.005`, and
`r_abs_max <= 0.20` (`…_common.py:682-698`). Feasibility of this window depends on
the per-case geometry of `Q` and the structural adjoint; it is not statically
provable for all ~10 constructed fixtures (six case-matrix cases + four
`orth_cap_fixture_matrix` cases). If any case fails to converge it raises and the
producer refuses to publish (`…_producer.py:389-396`) — fail-closed — but consumes
the single authorized SLURM attempt. Iteration-0 is feasible for well-conditioned
`Q`, so convergence is likely; unlike the prior H5 this is not a static
impossibility. Non-blocking, fail-safe.

### M-C (Medium, fail-closed): `jsonschema` availability unproven statically

As in prior M2. Preflight added; missing dependency cannot cause a false PASS but
can consume the single SLURM attempt with an environment failure. Either confirm
`jsonschema` in `HateVideo` through an authorized SLURM-only preflight, or accept
the fail-closed infrastructure risk explicitly.

### L-A (Low): wrapper cleanup can delete a prior successful run's artifacts on re-run

Evidence: `cleanup_on_exit` runs `rm -f "${PROSPECTIVE_OUTPUTS[@]}"` whenever
`COMPLETE != 1` (`…_run2_v2.sh:28-30`) without distinguishing files created by the
current invocation from pre-existing ones. On an (unauthorized) second invocation
after a prior success, `validate`/`producer` correctly refuse via no-clobber
(non-zero exit), but the ensuing EXIT-trap cleanup then deletes the prior
success's `manifest.json`/`source_manifest.json`/`access_ledger.json`/
`semantic_verification.json` (+ locks). Non-blocking: requires an unauthorized
re-run and the artifacts are deterministic and excluded from source binding, but it
is a genuine footgun given project provenance discipline. Consider guarding cleanup
to only remove outputs this invocation created (cf. the producer's own
`cleanup_created_outputs` tracking, `…_producer.py:230-238,398-408`).

### L-B (Low, Run3 scope note): fixture assumes rather than demonstrates a rank-`<=d` projection optimum

The frozen convex program has no hard rank constraint, so its projection solution is
generically full-rank and would then fail the `d`-realizability audit. The synthetic
fixture sidesteps this by *constructing* a rank-`d` `G*` on the PSD boundary and
back-fitting `G0`. This is legitimate for a verifier self-test but does not
demonstrate that the real projection yields a rank-`<=d` solution; that is a Run3 /
real-method question, explicitly out of Run2 synthetic scope. Recorded for Run3.

## Static checks performed

- `sha256sum` on all nine entities + four binding docs: all match freeze doc.
- `sha256sum -c`: `EXPERIMENT_PLAN_HASHES.sha256`,
  `M0_RUN2_V2_PLAN_AMENDMENT_HASHES.sha256`,
  `M0_RUN2_V2_PLAN_AMENDMENT_REVIEW_HASHES.sha256` — all OK.
- `AGENTS.md` on-disk hash == config binding.
- `jq`-parsed and read: config, payload schema, case schema; verified schema
  strictness intent (`additionalProperties:false` throughout;
  `schema_requires_no_additional_properties` is also enforced by producer/validator).
- `bash -n` OK for wrapper and sbatch; no `--time` directive.
- `git diff --check` on the nine entities + tracker: clean (exit 0).
- `artifacts/lb_scgp_global/v2` absent; no v2 artifact created by this review.
- No Python/import/`py_compile`/test/conda/SLURM execution was performed.

## Required statements

- No performance evidence exists and no performance claim is made or possible from
  this review.
- Only project gold is `parent_video_binary_label`. No segment/frame/timestamp/span/
  localization/stance/target/mechanism/rationale/fragment gold is assumed or
  introduced; Run2-v2 fixtures use synthetic labels only.
- Run3, M1, MLLM/cache, validation/test, training, and realbank remain locked.
- This review does not authorize SLURM execution. PASS_STATIC_REVIEW permits only a
  separate execution-authorization review for exactly one future CPU-only SLURM
  validation job. The conditional M-A escalation must be resolved (ruling or
  amendment) if any authority holds that the synthetic `G0` must be realizable
  rank-`<=d` PSD.

Report SHA256 is to be computed externally after this file is written; it is not
embedded to avoid a self-referential hash.
