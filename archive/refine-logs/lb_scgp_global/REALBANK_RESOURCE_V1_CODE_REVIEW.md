# M0 REALBANK-RESOURCE-v1 — Merged Independent Review (Amendment Ratification + Pre-Execution Static Code Review)

Date: 2026-07-13

Reviewer: **Claude Opus 4.8**, fresh / zero-context, zero-history (0C/0H) independent reviewer.
This role is distinct from the realbank-prep author (recon + amendment + eight-entity implementation +
freeze), and from the later execution authorizer and executor. This document performs both (i) the
independent amendment-review (ratify A/B/C + the additive `runs[3]` edit) and (ii) the fresh 0C/0H
static code review with an **independently re-derived** runtime cross-check simulation table. It
authorizes no execution.

## Reviewer boundary & required statements

- **Read-only static analysis only.** Shell limited to `rg`/`sed`/`nl`/`jq`/`awk`/`bash -n`/`diff`/
  `sha256sum`/`find`/`ls`/`wc`/`git status`/`git diff`. No Python/import/`py_compile`/conda/SLURM/
  `sbatch`/`squeue` was run. No experiment, MLLM/OCR/API/network/model/GPU/training/evaluation. No
  validation/test **content** was opened (only the two allowlisted train **feature** banks and the
  val/test **provenance hashes** were checksummed, read-only, to verify bindings). The only file
  written is this report.
- **No performance evidence exists and none is claimed.** This is a static review of an
  unexecuted run.
- **Opus 4.8 deviation declaration (precedent as before).** Project discipline constrains subagents to
  Opus 4.8, so this "fresh independent" review is performed by the same model family as the
  implementer rather than a cross-model reviewer. Independence is enforced by 0C/0H context reset and
  by re-deriving every binding/table row from on-disk state rather than trusting the freeze
  predemonstration. This mirrors the v2/v3/v4 review precedent.
- Only project gold is `parent_video_binary_label`; no segment/frame/span/localization/stance/target/
  mechanism/rationale/fragment gold is assumed or introduced; train **labels** are not opened by the
  realbank code. Run4 (M1 cache), MLLM/cache, validation/test, training remain locked.

---

## VERDICT

- **AMENDMENT_RATIFIED** — the A/B/C ruling and the additive `runs[3]` edit are correct, minimal, and
  faithfully landed; the REPLACE-nothing/additive-only discipline holds; the hash cascade is exact.
- **PASS_STATIC_REVIEW** — the eight-entity implementation is interface-aligned three ways, index-
  pinned to `runs[3]`, dependency-clean, resource-correct, and fail-closed on every runtime assertion.
- **Grade tally: Critical = 0, High = 0, Medium = 0, Low = 2** (both documentation-precision;
  non-blocking). Since Critical = 0 **and** High = 0, the gate criterion for
  `PASS_STATIC_REVIEW + AMENDMENT_RATIFIED` is met.
- `ready_for_execution` remains **false**: still required before any single submit — dependency-
  availability evidence (satisfied statically here, but the authorizer must re-confirm env state is
  not frozen), exact-hashes/no-clobber authorizer check, and a separate execution authorization.

---

## 1. Amendment ratification

### 1.1 Hash cascade (each `sha256sum`-recomputed from on-disk this session)

| file | expected (amendment) | on-disk | verdict |
|---|---|---|---|
| `EXPERIMENT_PLAN.machine.json` | `d5023b62…cb18fdb` | `d5023b62…cb18fdb` | **MATCH** |
| `.pre_realbank_amendment.bak` | `42bf49ed…4590a90` | `42bf49ed…4590a90` | **MATCH** |
| `EXPERIMENT_PLAN.md` | `10fd5232…1c53fa3a` | `10fd5232…` | **MATCH** |
| `EXPERIMENT_TRACKER.md` | `d226abfe…6b3ab3b53` | `d226abfe…` | **MATCH** |
| `EXPERIMENT_PLAN_HASHES.sha256` | `a8360a2a…a6bddd3e` | `a8360a2a…` | **MATCH** |
| `REALBANK_…_PLAN_AMENDMENT.md` | `3333c434…f2c00d37` | `3333c434…` | **MATCH** |
| `REALBANK_…_PLAN_AMENDMENT.machine.json` | `ccae7f67…69d6b81b` | `ccae7f67…` | **MATCH** |
| `REALBANK_…_PLAN_AMENDMENT_HASHES.sha256` | `26462aa0…fe1b173e1` | `26462aa0…` | **MATCH** |

`EXPERIMENT_PLAN_HASHES.sha256` internally lists plan.md / tracker / machine.json at the new hashes —
self-consistent. The realbank config binds all four post-cascade plan hashes plus the three amendment
docs plus `M0_RUN2_V4_ARTIFACT_REVIEW.md` (`103ea07c…`), `AGENTS.md`, `FINAL_PROPOSAL.md`; **10/10
authoritative_inputs and 10/10 run1_frozen bindings recomputed == on-disk**.

### 1.2 `.bak → current` machine-plan diff — every hunk maps to the declared change list

`diff -u` yields exactly **one hunk**, entirely inside `runs[3]` (the realbank record):
1. `+ "gate_satisfied_by": …` (declared);
2. `+ "realbank_protocol": { A_train_bank_source, B_structural_placeholder, C_acceptance }` (declared A/B/C);
3. `- "status": "LOCKED_UNTIL_V4_PASS"` → `+ "status": "GATE_OPEN_PENDING_REALBANK_IMPLEMENTATION_AND_REVIEW"` (declared).

No line outside `runs[3]` changed. **No undeclared change.** Structural invariants independently proven:

- `jq '.runs|length'` = **66** on both `.bak` and current (array length unchanged).
- `jq -S '.run_order'` **identical** `.bak` vs current.
- `jq -S '.runs[2]'` **identical** (consumed v4 record untouched; `run_id = …SYNTH-KKT-v4`).
- `jq -S '.runs[4:]'` **identical** (M1-cache onward untouched).
- `diff <(jq -S 'del(.runs[3])' bak) <(jq -S 'del(.runs[3])' cur)` empty → **everything except `runs[3]`
  is byte-identical** (dependency_dag, artifact_schemas, readiness, supplement_amendment all untouched).

### 1.3 A/B/C landing matches the ruling

- **A** — `runs[3].realbank_protocol.A` pins the two frozen CLIP-L/336 train banks; sha256 recomputed
  on disk: MHC `deea74ff…` (N=549), MHC_zh `929571f8…` (N=579) — **both MATCH**; identical to the
  config `train_banks` block. `d = Z0.shape[1]` at runtime; `rank_tail = rank_eps(G0) <= d`.
- **B** — NON-SCIENCE placeholder `b_struct = vech(M_Q(G0))`, `r=8`, `m=36`, `is_science=false`,
  fallback (G0-only) explicitly **not used**. Consistent across amendment ↔ schema (`const false`,
  `rank_cap_r const 8`, `m_scale const 36`) ↔ producer ↔ verifier.
- **C** — two-stage producer → independent verifier; GO criterion
  `job_peak_rss_bytes <= 103079215104 (96 GiB) AND rank_eps(G0)<=d (all ds) AND in-job replay match
  (all ds) AND all injections REJECT`; `robust_coverage` fail-open. Five science keys pinned.

### 1.4 v2→v4 documentation-drift fold

The two **declared** sites are correctly folded: `EXPERIMENT_TRACKER.md` row 4 and `EXPERIMENT_PLAN.md`
item 4 (both the §165-area list and §257) now read "depends on Run2-**v4** PASS plus fresh independent
v4 artifact review … not v1/v2/v3" with realbank status `GATE_OPEN_PENDING_REALBANK_IMPLEMENTATION_AND_REVIEW`.
No `LOCKED_UNTIL_V2_PASS`/`LOCKED_UNTIL_V4_PASS` run-status string remains at any realbank site; the
residual `…SYNTH-KKT-v2` strings are the v2 run's own preserved closed-lineage record (correct).
(See Low-2 for two out-of-scope stale summary lines.)

### 1.5 Consumed-v4-config binding — no cross-contamination

The consumed/closed v4 config `m0_synth_kkt_v4.json` still binds the pre-amendment machine hash
`42bf49ed…` in its `authoritative_inputs` (confirmed by `jq`). This is correct: v4 is single-submit
spent and `ARTIFACT_ACCEPTED`; re-freezing a consumed config would itself be a lineage violation, so it
is **not** retro-updated. The realbank code reads the **current** machine plan and asserts
`d5023b62…` (via `verify_authoritative_hashes` / `verify_machine_realbank`); `rg` confirms **no
realbank entity references `m0_synth_kkt_v4`**. The stale binding is therefore historical provenance in
a file no future run consults — **no runtime cross-pollution**.

---

## 2. Static code review — eight entities

Entity SHA256 (all 8 recomputed == FREEZE / amendment bindings): config `c436c3dd…`, schema
`db79cdd3…`, common `46e1f3fe…`, validate `b2bbec02…`, producer `dc38d5c3…`, independent_verify
`49cc2d9a…`, wrapper `f80b41ea…`, sbatch `9c4ecc05…`. **8/8 MATCH.**

### 2.1 Interface-key three-way alignment (the v1 death class)

Independently extracted and `diff`-checked:

- **Top-level manifest keys — 23.** schema `.required` (23) == verifier `TOP_KEYS` (23, digit-inclusive
  extraction) == producer-emitted manifest keys (23). Verifier enforces `set(manifest) != TOP_KEYS →
  raise` (line 506); schema is strict (`additionalProperties:false`, 21/21 object subschemas). Producer
  runs `validate_manifest_against_schema` before publish.
- **`zero_counters` — 47.** schema `definitions.zero_counters.required` (47) == common
  `ZERO_COUNTER_KEYS` (47) == verifier `ZERO_COUNTER_KEYS` (47), all three sets `diff`-identical.
- **Isolation cases — 11.** schema `cases.required` (11) == common `isolation_injection_cases` probes
  (11) == verifier `recompute_injection_classifier` probes (11), `diff`-identical; both classifiers
  share identical `FORBIDDEN_TOKENS` (query_z, query_labels, teacher, cache, held, certificate) and
  identical `forbidden_reason` logic.

Schema self-consistency: top-level `properties`==`required`; `zero_counters` `properties`==`required`
(both `jq`-verified true).

### 2.2 run_id / path / index constants vs machine `runs[3]`

`run_order[3]` = `runs[3].run_id` = `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1` (the code's `RUN3`
literal). Producer `verify_machine_realbank` and verifier `verify_machine` both read `machine["runs"][3]`
/ `run_order[3]` (numeric literal `3`) and assert run_id, `artifact_paths == [cfg.run.artifact_path]`,
`artifact_schema_ids == [SCHEMA_ID]`, `slurm == cfg.run.slurm`, `dependencies == [RUN2_V4]`, and the
`realbank_protocol.banks`. `jq` confirms machine `runs[3]` matches config field-for-field
(artifact_paths, schema_ids, slurm `{16,96,0,HateVideo,no_time:true}`, deps `[…SYNTH-KKT-v4]`). This is
the v3-death (index/plan drift) prevention, and it holds because the amendment made `runs[3]` **content**
match the `3` **literal** in lock-step.

### 2.3 Math byte-consistency vs the ACCEPTED v4 code (spot-check ≥ 3 core functions)

Function-by-function `diff` of realbank `common.py` against `lb_scgp_global_r2_run2_v4_common.py`:

- **Byte-identical (decision-bearing):** `psd_gram_from_features`, `structural_moment`,
  `structural_adjoint`, `rank_tail_audit`, `row_normalize`, `vech`, `unvech_dual`, `procrustes_align`,
  `floatify`, `vector_to_list`, `matrix_to_list`, `sha256_bytes/obj/file`,
  `schema_requires_no_additional_properties`.
- **Cosmetic-only (semantically identical):** `canonical_json` (same `json.dumps` args, reflowed to one
  line); `payload_hash` (local variable renamed `copy_obj`→`clone`).
- **Two semantic deviations — verified decision-inert (Low-1):**
  - `factor_from_psd_gram` omits v4's `if audit["status"] != "PASS": return None, audit`. Impact:
    when the audit is non-PASS but `rank<=d`, realbank returns `y` (not `None`) and runs Procrustes.
    But `y`, `zstar_gram_residual`, `nondegenerate` are used **only** inside the replay digest (not in
    the returned per-dataset fields), and the GO gate independently requires `rank_audit["status"] ==
    "PASS"` per dataset (producer line 182; verifier line 589). So the omission cannot manufacture a
    GO — both versions publish GO iff status==PASS, and otherwise fail closed. It only makes the
    resource measurement marginally more conservative.
  - `orth_cap` omits `"singular_values": vector_to_list(s)` from its returned info dict. That dict
    (`q_info`) is assigned but **never read** in `run_dataset_pipeline` (`rg` confirms one assignment,
    zero uses). Fully inert.

The "byte-faithful copies" wording is therefore slightly overstated; the decision-bearing linear
algebra **is** byte-identical and the two deviations are provably inert. See Low-1.

### 2.4 NON-SCIENCE placeholder disclosure (R-4)

`is_science=false` propagates through the machine `realbank_protocol`, the schema (`const false`), the
producer manifest (`structural_placeholder.is_science=False`), and is re-checked by the verifier
(`sp["is_science"] is not False → raise`, line 611); a tamper mutation
`placeholder_claims_science` (is_science→True) is in the verifier's negative-test suite and is
REJECTED. The science owner's overrule right (M-A-analogue) is recorded in the amendment/machine plan.

### 2.5 Access discipline

- **Train features allowlisted + hash-checked:** `open_train_bank` refuses any path not on the
  allowlist and refuses if the config-declared sha or on-disk sha mismatches; increments only
  `authorized_train_bank_read_count` (asserted == 2 by the verifier).
- **Train labels never read:** `load_bank_features` reads only `ids`/`img_feats`/`text_feats` and
  never accesses the bank's `labels` key; `train_label_read_count` stays 0; `gold_isolation.
  train_labels_opened=false` re-checked by the verifier. (Informational: `torch.load` deserializes the
  whole `.pt`, which physically contains `labels`, into memory, but no label **value** enters any
  computed quantity or the decision — consistent with the declared "features opened, labels not read".)
- **Val/test/held/cache/teacher/query all forbidden:** `forbidden_reason` rejects the six
  `FORBIDDEN_TOKENS` and any non-allowlisted `data/` path; `hash_source_file` hard-refuses every
  `data/` path. The verifier independently audits the serialized access ledger and raises on any
  `data/` record not of kind `{declared_not_opened, train_bank_feature_read}` or any query_z/
  query_labels/teacher token.
- **Injection coverage:** 11 probes (val label/content, test label/content, held, cache, query_z,
  query_labels, teacher, non-allowlisted train bank, mutated-hash open) — each must REJECT; producer
  and verifier compute them identically and the schema pins each `const "REJECT"`. The verifier
  additionally runs **15 manifest-tamper mutations** (extra key, stale/over-cap resource, flipped
  decision/within_cap, tampered rank/replay, non-reject injection, nonzero counter, tampered bank
  hash, forbidden source path, placeholder-claims-science, coverage-safety-enabled) — I traced each to
  its rejecting assertion; all 15 REJECT.

### 2.6 Fail-closed publish plumbing

Wrapper: RUN_ID/config guards → `validate → producer → independent_verify → jq '.decision=="PASS"'`;
`COMPLETE=1` only on PASS; `cleanup_on_exit` removes all prospective outputs unless COMPLETE.
Producer publishes only a clean GO candidate (raises on non-GO before any write; `try/except
cleanup_created_outputs`), uses O_EXCL publish locks + no-clobber refusals. Verifier publishes
`decision:FAIL` and returns 1 on **any** exception (fail-closed), which the wrapper's `jq` gate turns
into a full cleanup. No path manufactures a false PASS.

---

## 3. 21-row runtime cross-check simulation table — independent re-derivation

Re-derived from on-disk state this session (not copied from the freeze predemonstration). Verdicts:
**PASS** = statically provable now; **DEFERRED-TO-RUNTIME (fail-closed)** = evaluated inside the single
SLURM job, cannot false-PASS.

| Row | Assertion (site) | Independent verdict |
|---|---|---|
| 1 | wrapper `RUN_ID==EXPECTED`; config `run_id`/`artifact` (`jq`) | **PASS** — default==EXPECTED==`…REALBANK-RESOURCE-v1`; config artifact `…/realbank_resource/decision.json` |
| 2 | `require_slurm_realbank` 16/96/0 | **PASS (env at runtime)** — sbatch `--cpus-per-task=16 --mem=96G`, no `--gres`, no `--time`; guard raises on mismatch |
| 3 | validate `jq -e .` config/machine/schema/run1 | **PASS** — all 4 parse (verified `jq -e .`) |
| 4 | validate `schema_strict_check` | **PASS** — 21 objects / 21 `additionalProperties:false` |
| 5 | validate `bash -n` wrapper + sbatch | **PASS** — both `bash -n` clean |
| 6 | validate dependency numpy/torch/jsonschema | **PASS (runtime)** — all 3 present in HateVideo python3.11 site-packages; `find_spec` gate `sys.exit(1)` fail-closed |
| 7 | validate `py_compile` the 4 `.py` | **DEFERRED-TO-RUNTIME (fail-closed)** — login-node `py_compile` forbidden; runs in the SLURM validator; failure → status FAIL → producer never runs |
| 8 | validate `verify_run1_hashes` + old_protected | **PASS (run1 10/10) / runtime (old_protected)** — 10/10 run1_frozen == on-disk; old_protected `243e89b…`/278 excludes all `lb_scgp_global_r2_*` realbank files, runtime-recomputed fail-closed |
| 9 | validate `verify_authoritative_hashes` | **PASS** — 10/10 == on-disk |
| 10 | validate `verify_train_bank_bindings` | **PASS** — 2/2 sha == on-disk; `train_n` 549/579 == `EXPECTED_N` |
| 11 | validate `no_clobber_check` | **PASS** — `artifacts/lb_scgp_global/v1/m0/realbank_resource/` absent |
| 12 | validate `resource_and_run_check` | **PASS** — run_id/schema_id/artifact_path/slurm(16/96/0)/authorization all match |
| 13 | producer `verify_config_and_schema` | **PASS** — 10 authorization flags `false` + `train_bank_read_allowed=true`; schema strict |
| 14 | producer `verify_machine_realbank` `runs[3]` | **PASS** — `jq`: run_order[3]/run_id/artifact_paths/schema_ids/slurm/deps=[v4]/banks all == config |
| 15 | producer manifest schema validation | **PASS (by construction)** — emits exactly the 23 required keys; strict-validated pre-publish |
| 16 | verifier `set(manifest)==TOP_KEYS` | **PASS** — TOP_KEYS(23) == schema.required(23) == producer keys |
| 17 | verifier `zero_counters` set | **PASS** — schema==common==verifier, 47/47 |
| 18 | verifier `verify_machine` `runs[3]` | **PASS** — identical checks to row 14, independent code path |
| 19 | verifier injection recompute == manifest | **PASS (by construction)** — identical 11 probes + `forbidden_reason`/`FORBIDDEN_TOKENS` |
| 20 | verifier authoritative/run1 on-disk == config | **PASS** — re-checks run1_frozen + authoritative_inputs on disk (== rows 8–9) |
| 21 | verifier GO consistency | **PASS (expected)** — at N=549/579 O(N³) peak ≪ 96 GiB; `rank_eps(G0) ≤ N ≤ d` (CLIP-L concat d≈1536 ≫ 579); in-process replay bit-deterministic; injections all REJECT; cross-process replay determinism is the one fail-closed dependency (see R-2) |

**No row evaluates to FAIL.** All four runtime-deferred rows (2 env, 6 dependency, 7 py_compile, 8
old_protected, plus the R-2 cross-process replay in row 21) are fail-closed: any miss raises → `FAIL`/
non-GO → the wrapper cleans up → no artifact, no false PASS.

---

## 4. R-1 … R-4 individual grading

- **R-1 (torch.load `weights_only=True`) — ACCEPT (Low residual).** Identical to the proven
  `lb_scgp_sanitize_inputs.py:142` call on the same `{ids,img_feats,text_feats}` cache family;
  `dataset.py` confirms the bank layout. If `weights_only=True` rejects the payload it raises cleanly
  (no artifact), never a silent wrong result. Fail-closed; the execution authorizer should still
  confirm the installed torch accepts it.
- **R-2 (eigvalsh/svd bit-determinism) — ACCEPT (Low; highest-risk residual, fail-closed).** In-job
  run1==run2 is same-process → bit-identical (GO criterion). The producer↔verifier cross-check
  additionally requires the verifier's independently recomputed replay digest to byte-match the
  producer's; this relies on LAPACK determinism at fixed thread count (`OMP/MKL/OPENBLAS=16`, set in
  the sbatch, shared by both processes) with `floatify` (15 sig figs, `<5e-16→0`) absorbing last-ULP
  noise. This is the accepted-v4 pattern; on real-CLIP eigen-tails the cross-process risk is modestly
  higher than v4's synthetic data, but a miss yields `decision:FAIL` (STOP), **not** a false GO.
- **R-3 (old_protected v4 binding) — ACCEPT.** Config binds `243e89b…`/278; `old_protected_hash_
  manifest` scope is `configs|artifacts|refine-logs/lb_scgp/` plus non-`lb_scgp_global_r2_*`
  `lb_scgp_*` scripts. All realbank files are `lb_scgp_global_r2_realbank_*` under
  `configs/lb_scgp_global_r2/`, `scripts/…`, and `refine-logs/lb_scgp_global/` → **excluded** by name
  and directory; the snapshot is unchanged and runtime-recomputed fail-closed.
- **R-4 (placeholder veto power) — ACCEPT.** `is_science=false` disclosed end-to-end; verifier rejects
  `is_science=True`; science-owner overrule right recorded. NON-SCIENCE cannot leak into a scientific
  claim without an explicit downstream gate.

---

## 5. Dependencies

- Full import inventory (module-level **and** in-function) across all four `.py`:
  third-party = **{numpy, torch, jsonschema}** (numpy module-level in common + verifier; `torch`
  function-level in both bank loaders; `jsonschema` function-level in both schema validators); the rest
  are standard library (`argparse, copy, hashlib, json, math, os, resource, subprocess, sys, tempfile,
  pathlib, typing`). Matches the amendment's declared set exactly.
- **Read-only availability evidence:** `ls` shows `numpy/`, `torch/`, `jsonschema/` all present under
  `…/envs/HateVideo/lib/python3.11/site-packages`.
- **Preflight fail-closed:** validate `python_dependency_check` uses `importlib.util.find_spec` over
  exactly those three names and `sys.exit(1)` on any miss, **inside** the SLURM job **before** the
  producer — the specific v2-death (missing `jsonschema`) prevention.

## 6. Resources

- sbatch `--cpus-per-task=16 --mem=96G`, `conda activate HateVideo`, **no** `--gres=gpu`, **no**
  `--time` ("Intentionally no --time: project policy") — matches machine `runs[3].slurm`
  `{cpu:16, ram_gb:96, gpu:0, env:HateVideo, no_time_flag:true}` and is within the per-user cap
  (16 CPU / 128 GB / 2 GPU).
- **Peak-RSS measurement is sound.** `peak_rss_bytes()` = `resource.getrusage(RUSAGE_SELF).ru_maxrss
  * 1024` — correct KB→bytes on Linux (a missing ×1024 would under-report; the code has it). It is a
  process high-water mark (not a sample, so no transient-spike miss), captured at end-of-job, compared
  to `CAP_BYTES = 103079215104` (96 GiB). The verifier re-checks `within_cap == (job_peak <= CAP)`
  arithmetic and rejects `within_cap != True`. At N=549/579 the true peak (dominated by the torch
  import; N×N doubles are ~2.4 MB) is far under the cap, so the STOP threshold is a correct-but-slack
  guardrail here.

---

## 7. Findings

- **Low-1 — "byte-faithful" claim mildly overstated (documentation precision).** `factor_from_psd_gram`
  and `orth_cap` deviate from v4 (one dropped non-PASS early-return; one dropped unused info field),
  plus two cosmetic edits. Independently verified **decision-inert** (factor covered by the explicit
  `status=="PASS"` GO gate and the verifier's independent recomputation; orth_cap's info dict unused).
  Recommend softening the freeze/amendment wording to "faithful reuse with two documented,
  decision-inert deviations." Non-blocking.
- **Low-2 — two out-of-scope stale v2-era summary lines (documentation coherence).**
  `EXPERIMENT_TRACKER.md` line 6 ("Status-aware counts") still says "…1 explicit `LOCKED_UNTIL_V2_PASS`
  …locked until v2 PASS", and `EXPERIMENT_PLAN.md` line 169 (milestone rollup) still says "v2 PASS plus
  independent artifact review before realbank" under the `## G0 Runs Through Run2-v2 Supplement`
  header. These predate this amendment (v2-era headers never rolled through v3/v4) and were **not** in
  its declared fold scope (row 4 + item 4 only); no code reads them and the authoritative machine plan
  is correct, so they do not affect execution. Recommend a follow-up doc-coherence pass. Non-blocking.
- **Informational (not a finding):** the train `.pt` physically contains a `labels` tensor that
  `torch.load` deserializes into memory; no label value is ever accessed or influences any computed
  quantity — consistent with the declared "features opened, labels not read" discipline.

**Critical = 0, High = 0, Medium = 0, Low = 2.**

---

## 8. Conclusion

The REALBANK-RESOURCE-v1 plan amendment is **RATIFIED** (minimal additive `runs[3]` edit, exact hash
cascade, correct A/B/C landing, correct v2→v4 fold of the declared sites, no cross-lineage
contamination) and the eight-entity implementation **PASSES STATIC REVIEW** (three-way interface
alignment, `runs[3]` index-pin, byte-identical decision math, clean dependency set, correct resource
policy and RSS measurement, and a fully fail-closed producer→independent-verifier pipeline with an
independently re-derived all-PASS 21-row simulation table). The three-burn root cause — runtime
assertions against frozen external state — is closed here: the code-constant-vs-plan-content rows
(14/18) and the hash-layer rows (8–10, 20) both provably PASS.

Two Low, documentation-only findings; **no Critical, no High** → the criterion for
`PASS_STATIC_REVIEW + AMENDMENT_RATIFIED` is satisfied. Execution remains **unauthorized**: it still
requires the execution authorizer's dependency-availability re-confirmation (env is not frozen),
exact-hashes/no-clobber check, and a separate execution authorization before the single `sbatch`.
