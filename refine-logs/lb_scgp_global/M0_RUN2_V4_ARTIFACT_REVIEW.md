# M0 Run2-v4 Fresh Independent Artifact Review

Date: 2026-07-13

Reviewer: **Claude Opus 4.8**, fresh, zero-context, zero-history (0C/0H) independent **artifact
acceptance reviewer** for the `lb_scgp_global_r2` M0 Run2 **v4** lineage. This is the ceremony step
**after** the executor submitted the single authorized SLURM job and the producer + independent
semantic verifier published the v4 artifact set: a fresh acceptance review whose PASS unlocks the
next boundary (row-3 `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1`). This role is deliberately
**separate** from the v4-prep/clone-freeze, the merged amendment-ratification + code-review, the
execution-authorization, and the executor roles.

## Reviewer boundary

Read-only acceptance review. The **only** file I wrote is this report. I did **not** run project
Python, imports, `py_compile`, tests, `conda`, `sbatch`/`squeue`/`sacct`, an experiment, or any
MLLM/OCR/API/model/network/GPU/training/evaluation, and I touched no validation/test data or cache
content (I hashed the four declared-not-opened provenance files with `sha256sum` for a binding
check only — their bytes were not parsed into any pipeline and no label was read). I did **not**
recompute the KKT solution, the eigendecomposition, or the payload canonical hash with `numpy`; I
verified the artifact's self-consistency and its bindings to disk using only the allowed static
tools: `jq`, `sha256sum`, `sed`/`grep`/`rg`, `nl`, `wc`, `ls`, `find`. Where a numeric residual
required independent recomputation, I relied on the **independent semantic verifier's** separately
serialized recompute (`semantic_verification.json`) and checked it against the producer manifest,
rather than trusting a single self-report. No artifact byte was mutated; the four `*.publish.lock`
files were read, not written.

### Model-binding divergence declaration (precedent: `M0_RUN2_V4_CODE_REVIEW.md`, `M0_RUN2_V4_EXECUTION_AUTHORIZATION.md`)

`AGENTS.md:15` binds the main-dialogue subagent to **"GPT-5.5 xhigh"**. That backend is unavailable
for this session's subagent, so this review runs on the `CLAUDE.md`-bound **Opus 4.8**
(`claude-opus-4-8`). Documented `AGENTS.md`↔`CLAUDE.md` process divergence, recorded for
transparency; it is not a code or artifact defect and does not affect the conclusion below.

---

## Verdict

**ARTIFACT_ACCEPTED** — the v4 synthetic-KKT artifact set is complete, internally consistent,
correctly bound to the frozen sources, semantically PASS on every pre-registered criterion, and
carries zero gold overreach. **Row-3 REALBANK-RESOURCE-v1 may unlock** per the row-4 (tracker row
`| 4 |`) gate "depends on Run2 PASS plus fresh independent artifact review."

| Severity | Count | Items |
|---|---:|---|
| Critical | 0 | — |
| High | 0 | — |
| Medium | 2 | M-A (carried from code review §6 — untriggered conditional-High), M-B (carried — now **empirically resolved** for this fixture; see §4) |
| Low | 1 | L-obs (new, observability): the M-B shrink-window **step count** is not recorded in the artifact or logs |

No finding blocks acceptance. M-A remains a science-owner flag (not an artifact defect); M-B's
static "convergence-not-provable" risk is now **discharged by evidence** (all six cases produced a
non-degenerate rank-3 factor with in-window movement). L-obs is a note for a future producer, not a
defect in this run.

---

## 1. Completeness & binding (PASS)

**Files present** (8 total): `manifest.json` (203 078 B), `semantic_verification.json`,
`source_manifest.json`, `access_ledger.json`, and one `*.publish.lock` per file. Nothing extra;
the four output files exactly match the config `dirty_policy.allowed_new_files_after_run` set.

**On-disk file hashes bind to the recorded values (I re-`sha256sum`'d each):**

| File | on-disk sha256 | recorded as | source |
|---|---|---|---|
| `manifest.json` | `3761843b…cdafd` | `manifest_file_sha256` | `semantic_verification.json` ✓ |
| `source_manifest.json` | `94eef563…5cd88` | `hashes.source_manifest_sha256` | `manifest.json` ✓ (also `sem_verif.metrics`) |
| `access_ledger.json` | `3cf91480…12ddd` | `hashes.access_ledger_sha256` | `manifest.json` ✓ (also `sem_verif.metrics`) |
| payload | `81aae983…20acfc` | `payload_sha256` == `manifest_payload_sha256` | manifest ↔ sem_verif ✓ (two independent attestations agree) |

`semantic_verification.json` is itself unbound (on-disk `2ecfe5df…`) — expected; it is the terminal
verifier output that nothing downstream references.

**Payload-hash caveat (bounded):** I did **not** re-derive `81aae983…` myself (that needs the
producer's exact canonicalizer, which I did not run). It is attested **twice independently** — the
producer writes it into the manifest and the separate independent verifier recomputes it into
`manifest_payload_sha256`; the two agree byte-for-byte. Combined with the verified whole-file
`manifest_file_sha256`, the payload binding is doubly-attested and disk-anchored.

**Nine-entity source binding (against CLONE_FREEZE §1 + config `e67fc6f7`):** `source_manifest.json`
`run2_implementation_files` lists all nine v4 entities; I re-hashed each on disk:

- Entities 2–9 (payload/case schema, `common`/`validate`/`producer`/`independent_verify`,
  wrapper, sbatch) match `M0_RUN2_V4_CLONE_FREEZE.md` §1 **exactly**.
- Entity 1 = config `e67fc6f79073…56844d1` — the **post-binding** hash from
  `M0_RUN2_V4_EXECUTION_AUTHORIZATION.md` §3, **not** the freeze §1 pre-binding `118afadf…3bf0f`.
  This is correct and expected: the executor was instructed (exec-auth §9.1) to verify the
  post-binding config, and both the manifest (`hashes.config_sha256`) and `source_manifest` record
  `e67fc6f7`, so the run **used the authorized post-review config**. The config's
  `hash_bindings.authoritative_inputs` count is **28** (27→28 = the one added
  `M0_RUN2_V4_CODE_REVIEW.md` = `41650dce…` review-doc binding), matching exec-auth §3.

**Provenance & run1-frozen bindings truthful on-disk:** the four declared-not-opened val/test
`jsonl` files (`MHC`/`MHC_zh` × val/test) hash-match their `access_ledger`/`source_manifest`
bindings; the `run1_frozen` cert schema (`4d3f1663…`) and contract-freeze artifact (`09b78682…`)
match. `source_manifest.relevant_git_status = []` (clean relevant tree);
`dirty_binding.relevant_tree_sha256 = 63d0623a…` agrees across manifest / source_manifest /
sem_verif.

**Lineage identifiers (v4-correct; cert_v2 exception honored):** `run_id =
LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v4`; `artifact_schema_id = scgp_global_synth_kkt_payload_v4`;
`schema_version` = `lb_scgp_global_r2_synth_kkt_manifest_v4` (manifest) /
`…_run2_v4_source_manifest_v1` / `…_run2_v4_semantic_verification_v1`; the shared cert schema is
correctly left at `scgp_global_cert_v2` (the frozen-shared exception). `config_path` /
`source_manifest_path` / `access_ledger_path` all point into `…/v4/…`.

**No gold overreach (PASS).** `access_ledger` entries are only `file_hash` or `declared_not_opened`;
the four val/test files are `kind:"declared_not_opened", scope:"declared_provenance_not_opened"`
(hashed for provenance, never opened for labels). `gold_isolation.only_gold_supervision =
parent_video_binary_label`, `segment_gold_exists=false`, `segment_gold_used=false`. **Every**
zero-counter is 0 — including `segment_gold_read_count`, all `*_label_read_count`,
`query_labels_read_count`, `gpu_device_count`, `mllm_call_count`, `network_call_count`,
`model_call_count`, `ocr_call_count`, `training_call_count`, `run3_or_later_attempt_count`,
`forbidden_path_read_count`. This is a synthetic self-test → **zero gold access**, as required.
`authorized_boundary = {run_id: …-v4, synthetic_only: true, run3_or_later_locked: true}`.

---

## 2. Semantic acceptance (PASS on every pre-registered criterion)

`semantic_verification.json`: `decision = PASS`, `acceptance_path =
serialized_h_metric_normal_cone_kkt`, `finite_vi_acceptance = false`. `case_matrix.status = PASS`
with **all six cases** `kkt_status = PASS` and `rank_audit.status = PASS`:
`FULL_SYNTH_KKT`, `REMOVE_NULL_PARITY`, `SHUFFLE_SYNTH_CONTROL`, `NOISE_SYNTH_CONTROL`,
`AMBIGUOUS_COVERAGE_LOW`, `ROBUST_COVERAGE_REPORTED`.

**Residual magnitudes (top-level = primary FULL case; verifier recompute cross-checked):**

| KKT block | key numbers | tolerance | verdict |
|---|---|---|---|
| stationarity | `status=PASS`; manifest `normalized_residual=0.0`; **verifier recompute `1.756e-16`** | ≤ `1e-6` | PASS (recompute ≈ 0, not blindly copied) |
| complementarity | `max_abs=0.0`; all 5 families (box_coordinate/soc/psd/halfspace/structural_band) = `0.0` | = 0 | PASS |
| dual_feasibility | `status=PASS`; `linear_multiplier_min=0.0`, `psd_dual_lambda_min=0.0`, `soc_cone_residual_max=0.0` | ≥ 0 / = 0 | PASS |
| duality_gap | `gap_pass_claimed=false`, `dual_objective_materialized=false` — no dual-objective PASS is claimed; stationarity + valid normal decomposition **is** the acceptance path | (by design) | PASS |
| primal residuals | all **9** = `0.0` (symmetry_fro, unit_diag_inf, psd_min_violation, offdiag_box, coordinate/row/class trust, structural_equality_l2, structural_band) | = 0 | PASS |
| rank / eig | `rank_eps=3` (= d); `lambda_d=3.077` vs `lambda_dplus1=6.73e-16` (clean rank-3 gap); `eps_rank=3.53e-7`; `reconstruction_residual=8.43e-16`; `zstar_gram_residual=7.74e-16` (verifier `1.069e-15`) | rank thr ~`1e-7` | PASS |
| coordinate-trust offdiag | `max_abs_offdiag_change=0.00868` | ≤ `0.02` (and in shrink window `(0.005, 0.018]`) | PASS |
| strong convexity / eig_min | `metric.H_positive_definite=true`, `lambda_struct=1.0` | ≥ `1e-5` | PASS |

**Verifier↔manifest number agreement** (independent recompute vs producer self-report, all to ≥14
sig figs): `movement_fro 0.03637867…`, `movement_offdiag_max 0.00868327…`, `objective_value
0.05919233…`, `structural_dual_l2 0.34214215…`, `source_manifest_sha256`, `access_ledger_sha256`,
`relevant_tree_sha256` — all match.

**Movement (non-triviality) — PASS.** `movement_metrics.fro_norm_G_star_minus_G0 = 0.0364 > 0.005`
threshold. Across the six cases the FULL/SHUFFLE/NOISE/AMBIGUOUS/ROBUST movements are
`0.0364/0.0362/0.0369/0.0324/0.0387` (all `> 0.005`); `REMOVE_NULL_PARITY` is exactly `0.0` — the
null-parity control, where zero movement is the **intended** behavior.

**rank_failure_probe — behaves correctly.** `case_id=RANK_FAILURE_RETURNS_NULL`,
`expected_status=ENCODER_RANK_GATE_FAIL`, `factor_returned_null=true`, `eps_rank=1e-7`,
`reconstruction_residual=1e99` (sentinel), `failure_policy=return_null_no_truncation_schema_tolerance_rescue`
— the probe deliberately trips the rank gate and returns null rather than silently truncating. ✓

**Injection defenses — every negative injection REJECTED.** Both
`manifest.injection_results_expected` and `semantic_verification.injection_results` carry **21**
bogus cases with **identical key-sets** and **all 21 = REJECT** (0 non-REJECT in either). Categories
span schema tampering (`invalid_extra_missing_schema_fields`, `incomplete_cone_family`,
`malformed_normal_presence`), math corruption (`malformed_complementarity`,
`malformed_stationarity_status`, `wrong_dual_sign`, `malformed_normal_residual` off by `5e-7`,
`nan_overflow`/unit-diagonal), forbidden acceptance paths
(`finite_vi_only_attempted_acceptance`, `identity_no_movement_claims_full`,
`malformed_finite_vi_diagnostic`), binding attacks (`forbidden_path`,
`perturbed_artifact_source_operator_hash`), and the per-case replays
(`bogus_case_{factor_replay,movement_metrics,primal_residual,rank_audit,robust_coverage}`,
`bogus_rank_failure_audit`, `rank_failure`). Additionally, `schema_fixture_results.invalid_schema`
= 3 fixtures (`extra_forbidden_field`, `missing_required_field`, `invalid_confidence`), **all
REJECT**.

**Zero counters** — all 60+ `gold_isolation.zero_counters` = 0 (also mirrored in the access
ledger). **finite-VI diagnostic** present but **non-accepting**:
`finite_vi_diagnostics = {computed:true, attempted_acceptance:false,
acceptance_role:"non_accepting_diagnostic_only", max_probe_violation:0.0}` — it is a reported
diagnostic that provably cannot drive acceptance (and the two finite-VI injection cases confirm any
attempt to accept via it is REJECTED). Not an acceptance item; behavior correct.

**Prior-round Mediums closed** (`semantic_verification.medium_findings_closed`):
`M1_strict_schema_semantic_verifier`, `M2_dirty_binding_run1_run2_relevant_tree`,
`M3_orth_cap_and_M_Q_executed_with_rank_cap_cases` — all `true`.

**Two-stage acceptance flow correct.** Producer wrote the manifest as `terminal_state =
PRODUCED_PENDING_INDEPENDENT_VERIFY`, `acceptance.producer_status = PASS_CANDIDATE`,
`semantic_verifier_required = true`, `no_success_claim = true`; the **separate** independent
semantic verifier then stamped `decision = PASS`. Acceptance is not self-certified by the producer.

---

## 3. M-A residual at the artifact layer (confirmed bounded & disclosed; non-blocking)

Code review §6 flagged M-A (Medium; conditional-High, untriggered): the verifier checks `G0` shape,
symmetry, and unit diagonal but performs **no** `G0 ⪰ 0` / `rank(G0) ≤ d` check. At the **artifact
layer** this is exactly what the manifest shows:

- The **symmetry** and **unit-diagonal** checks on `G0` **are** recorded — `primal.residual_summaries`
  `symmetry_fro = 0.0` and `unit_diag_inf = 0.0`.
- There is **no** `G0`-PSD / `G0`-rank field anywhere in the manifest. The `psd_min_violation = 0.0`
  that does appear is on the **projected solution `G_star`** (the KKT primal), not on the anchor
  `G0`.
- `G0` is **fully disclosed** (`primal.G0`, 10×10). Its off-diagonals reach `-0.994`, `-0.972`,
  `-0.972`, so `G0` is almost certainly **indefinite** (not PSD) — precisely M-A's characterization
  of a synthetic symmetric unit-diagonal anchor that is not a realizable `Z0 Z0^T`. I did not run an
  eigensolver (reviewer boundary), but the disclosed magnitudes make the non-PSD reading plain.

**Scope at the artifact layer:** bounded and disclosed. The KKT certificate is valid for any
symmetric unit-diagonal anchor, so the missing `G0`-PSD check is **correctness-neutral** for this
self-test, and the anchor is auditable in-artifact. **Non-blocking.** The conditional-High
escalation (a science authority ruling `G0` must itself be a rank-`≤ d` PSD `Z0 Z0^T`) remains
**untriggered**; it must be resolved before any *scientific* claim rests on this fixture, but it is
not an acceptance blocker for the synthetic certificate.

---

## 4. M-B post-confirmation — rank construction converged (empirically resolved)

Code review §6 / exec-auth §7 held M-B (Medium, fail-closed) as the top single-submit risk:
`rank_deficient_structural_solution` runs a 30-step geometric shrink (`scale *= 0.7`) seeking
`0.005 < movement_off_max ≤ 0.018 ∧ movement_fro > 0.005 ∧ r_abs_max ≤ 0.20`, with window
feasibility **not statically provable**; on miss it raises → producer refuses to publish
(fail-closed). This lineage had never executed it to completion (v1/v2/v3 all died upstream).

**The published artifact confirms it converged — for all six cases:**

- **Factor built, non-degenerate:** primary case `factor_replay = {factor_returned_null: false,
  nondegenerate: true, gram_reconstruction_residual: 8.43e-16, procrustes_orthogonality_residual:
  0.0, zstar_gram_residual: 7.74e-16}`. A rank-3 factor reconstructs `G_star` to machine precision.
- **Clean rank-3 spectrum:** `rank_audit.rank_eps = 3` (= d), `lambda_d = 3.077 ≫ lambda_dplus1 =
  6.73e-16`, `negative_eigenmass ≈ 9.4e-16`, `status = PASS`.
- **Movement landed in-window:** `movement_fro = 0.0364 (> 0.005)` and `max_abs_offdiag_change =
  0.00868` (inside `(0.005, 0.018]`) — the shrink found a feasible `scale`. The other five cases
  likewise carry in-window (or intentionally-zero null-parity) movement and `rank_audit.status =
  PASS`.

So the statically-unprovable window is **empirically feasible** for the frozen fixtures; M-B's
fail-closed risk did **not** fire, and the single authorized attempt produced a real certificate.

**Execution evidence:** `slurm/logs/lbscgp_global_r2_run2_v4_12978.out` is **empty**;
`…_12978.err` contains **only** the expected `jsonschema.RefResolver` `DeprecationWarning`
(code-review L-C forward-risk) at `independent_verify.py:167` — it did **not** halt the run, and the
job published cleanly (`slurm_policy.job_id = "12978"`, matching the executor ledger).

**L-obs (new, Low — observability):** the **number of shrink steps** actually used (of the 30-step
budget) and the converged `scale` are **not** recorded in the manifest (I searched all scalar paths
for `scale|step|iter|shrink|window` → none) and the `.out` log is empty. Convergence is *proven* by
the in-window movement + successful factorization, but the step count is not recoverable from the
published artifacts. This is a minor observability gap for a future producer, not a defect in this
run.

---

## 5. Conclusion

**ARTIFACT_ACCEPTED.** The v4 synthetic-KKT artifact set is complete, disk-anchored, and internally
triple-consistent (producer manifest ↔ independent verifier ↔ on-disk bytes); bound to the correct
nine frozen entities (entities 2–9 = CLONE_FREEZE §1, config = the authorized post-binding
`e67fc6f7`); semantically **PASS** on every pre-registered criterion (stationarity /
complementarity / dual-feasibility / primal residuals all `0` or `~1e-16`, rank exactly `d=3` with a
clean spectral gap, non-trivial movement, correct rank-failure probe, **all 21 negative injections
REJECTED**, all 3 schema fixtures REJECTED, all zero-counters `0`, finite-VI non-accepting); and
carries **zero gold overreach** (only `parent_video_binary_label` permitted; nothing opened). M-A is
a bounded, disclosed, correctness-neutral science-owner flag; M-B's static convergence risk is
discharged by evidence; L-obs is a note, not a blocker.

Per the row-4 (tracker `| 4 |`) gate, **row-3 `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1` may
unlock**. Two carry-forward obligations remain, neither blocking this acceptance:

1. **M-A conditional-High** must be adjudicated by the science owner before any *scientific* claim
   rests on the synthetic `G0` fixture (it does not affect the certificate's validity).
2. **REALBANK is a different budget** (train banks for `MHC`/`MHC_zh`, no val/test labels) with its
   own single-submit ceremony — this acceptance unlocks its *gate*, not its execution.

---

## Required statements

- No performance evidence exists and no performance claim is made or possible from this artifact: it
  is a synthetic closed-convex-projection / KKT self-test. `no_success_claim = true` in the manifest.
- The only project gold is `parent_video_binary_label`. No segment/frame/timestamp/span/
  localization/stance/target/mechanism/rationale/fragment gold is assumed, introduced, or read; the
  v4 fixtures are synthetic and every gold/label/segment counter is `0`.
- Run3, M1, MLLM/cache, validation/test, training remain **locked**. This review unlocks only the
  **gate** for row-3 REALBANK-RESOURCE-v1; REALBANK carries its own separate authorization ceremony.
- This reviewer role is separate from the v4-prep/clone-freeze, merged amendment/code-review,
  execution-authorization, and executor roles. The only file written is this report.

Report SHA256 is to be computed externally after this file is written; it is not embedded to avoid a
self-referential hash.
