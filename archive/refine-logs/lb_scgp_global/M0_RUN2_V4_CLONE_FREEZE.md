# M0 Run2-v4 Byte-Exact Clone / Freeze

Date: 2026-07-13

Creator: **Claude Opus 4.8**, acting in the **v4-prep role only**. This role is deliberately
separate from — and does not perform the work of — the later roles required by
`M0_RUN2_V3_RESULT_TO_CLAIM_REVIEW.md` §5: the independent v4 amendment reviewer, the fresh
0C/0H v4 static code reviewer, the independent execution authorizer, and the independent
executor. This document authors the v4 plan amendment, edits the authoritative plan, runs the
amendment-driven hash cascade, creates the v4 clone, and freezes it. It does **not** review,
authorize, or execute it.

Scope / discipline: static amendment + clone + freeze only. No project Python was executed, no
`py_compile`, no import, no conda/SLURM/`sbatch`/`squeue`, no experiment, no MLLM/OCR/API/network/
model call, no GPU/training/evaluation, and no validation/test data or cache was touched. `jq -e .`
was used only as read-only JSON well-formedness; `sed`/`diff`/`sha256sum`/`grep` were used only to
generate and prove the byte-transform and to re-play (read-only) the runtime hash asserts. No
artifact under `artifacts/lb_scgp_global/v4` was created (its absence is confirmed in §6). Nothing
was committed to git and no SLURM job was submitted.

This report is **not** performance evidence, **not** execution authorization, and **not** a
Run3/M1/MLLM-cache/validation-test/training/realbank unlock. It implements
`M0_RUN2_V3_RESULT_TO_CLAIM_REVIEW.md` §5(a) (plan amendment + hash cascade), §5(b) (byte-clone),
and self-runs §5(c) (runtime cross-check static-simulation table) as a pre-review predemonstration.

---

## 0. Source integrity (what was cloned)

The nine frozen v3 entities were re-hashed immediately before cloning and match the SHA256
recorded in `M0_RUN2_V3_CLONE_FREEZE.md` §1 (config `e6d33b5d…b7d5`, payload `1d6f93a1…d2d3`,
case `df3616ff…dcac`, common `9de62f6d…411c2`, validate `2e0bb00b…13a6`, producer
`6ef3a4a8…5114`, independent_verify `4025dbf0…523c5`, wrapper `8d9123e9…d66d3`, sbatch
`4495ec3c…b3d9`). The clone therefore derives from the exact frozen v3 bytes.

---

## 1. Nine v4 entities — SHA256 manifest

| # | v4 entity | SHA256 |
|---|---|---|
| 1 | `configs/lb_scgp_global_r2/m0_synth_kkt_v4.json` | `118afadfc18cb493a298eda516160f531abfce982471ea836a2d1c6c35f3bf0f` |
| 2 | `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_payload_v4.schema.json` | `6c31a7c1c98a63ed5a35bdd7313c504f2b870c11c2218f90776b0b88de8ac9ca` |
| 3 | `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_case_v4.schema.json` | `df3616ff50c54dc756bb600c6ef26626afafe070678450d10ca73b9ff83ddcac` |
| 4 | `scripts/analysis/lb_scgp_global_r2_run2_v4_common.py` | `c6745d4be3eef3afee28d1b63478323fab57fb9e4b79740b05e5a11f2d62dbae` |
| 5 | `scripts/analysis/lb_scgp_global_r2_run2_v4_validate.py` | `7eda5e85d1b7bb87e34307946ad112fdad22f7d51f0f5868fed2110ee4b87ec2` |
| 6 | `scripts/analysis/lb_scgp_global_r2_run2_v4_producer.py` | `84439f7c2db1adf5f0046a0acbcd49e8a896e4e71e053c2c8211a18688b5179f` |
| 7 | `scripts/analysis/lb_scgp_global_r2_run2_v4_independent_verify.py` | `da827f0a4b2bf4f3bf07cb38497e14dcf9d22c2a7be2f9ba9fdc3bc4ca476060` |
| 8 | `scripts/wrappers/lb_scgp_global_r2_run2_v4.sh` | `0ad33ba4c3e43e52800d5d1a79316e0b1ebb84d9fc65aac4a4de087e1c65d161` |
| 9 | `scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v4.sbatch` | `8e1359ac259fd9e54181d94f77e7e29c55d8c1b44c012b1081ecffc792145427` |

Note the case schema (#3) hashes **identically** to the v3 case schema (`df3616ff…dcac`): its sole
`v2` token is the shared/frozen `scgp_global_cert_v2` reference, which is preserved (see §2), so
its content is byte-for-byte identical to v3 and only its filename changes. (The v3 clone-freeze
recorded the same identity v2↔v3.)

The v4 **config** (#1) hash `118afadf…3bf0f` is post-cascade (it includes the four updated and
three added `authoritative_inputs` bindings of §4). The config's own hash is **not** pinned by
any runtime assert (`…v4_independent_verify.py:1016` compares it to the manifest's self-recorded
value, computed at runtime); it is recorded here as the ceremony freeze target only.

---

## 2. The transform (exactly the three permitted change classes)

v4 was produced from the frozen v3 bytes by a single deterministic substitution:

`sed 's/cert_v2/cert__CERTKEEP__/g; s/v3/v4/g; s/cert__CERTKEEP__/cert_v2/g'`

The cert guard is a **no-op** for v3→v4 (the shared token `cert_v2` contains no `v3`), retained
for symmetry with the v2→v3 clone. This maps onto the three permitted classes and nothing else:

- **(class 1) file names** — the nine files renamed `…v3…`→`…v4…`.
- **(class 2) internal self-references** — module import names
  (`lb_scgp_global_r2_run2_v4_common`), `run_id` (`LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v4`), schema ids
  (`scgp_global_synth_kkt_payload_v4`, `…case_v4`, `lb_scgp_global_r2_synth_kkt_manifest_v4`,
  `…source_manifest_v1` stems), payload `$ref` → `…case_v4.schema.json#`, artifact/config/script
  paths (`artifacts/lb_scgp_global/v4/…`, `configs/…/m0_synth_kkt_v4.json`, the three v4 `.py`
  script paths, the v4 wrapper/sbatch paths, and the **hardcoded config-path literals** at
  `…v4_common.py:854`, `…v4_independent_verify.py:995`/`:1016`, `…v4_validate.py:116`), the
  `mktemp` template, the SLURM `--job-name` (`lbscgp_global_r2_run2_v4`), and lineage-label
  strings in log/error messages.
- **(class 3) hash bindings** — the amendment-driven cascade of §4 (four `authoritative_inputs`
  values updated + three added). Unlike the v3 clone (which had zero recompute because the plan
  was held constant), v4 edits the authoritative plan, so this cascade is non-empty and is
  runtime-checked.

The index literals `machine["runs"][2]` / `run_order[2]` in `…v4_common.py:820-821` are **not**
changed (they stay `[2]`); this is precisely why the amendment REPLACES index `[2]` rather than
inserting. No comment, whitespace, numeric, structural, or logic byte was changed. Byte length is
identical for all nine pairs.

---

## 3. Equivalence proof (per pair)

Method: for each v3/v4 pair, `diff <(sed 's/v3/v4/g' <v3-file>) <v4-file>`. A **blanket**
`v3`→`v4` sed over the v3 file is the maximal-change reference. Because `cert_v2` contains no
`v3`, the blanket sed does **not** over-rewrite it, so at the clone step the diff against the v4
file is **empty for eight of the nine files** (the cert guard changes nothing for v3→v4); the
config is the sole class-3 entity, and in its final frozen form its diff is exactly the §4
binding cascade (4 changed values + 3 added), applied after the clone step.

| v4 entity | blanket-sed diff | residual `v2` in v4 = preserved `cert_v2` | byte length v3 → v4 |
|---|---|---|---|
| config | **class-3 cascade** (sole class-3 entity; diff = declared binding cascade, 4 changed + 3 added) | 2 (`run1_frozen` cert path, `paths.cert_schema`) | 8709 → 9129 |
| payload schema | **empty** | 0 | 26069 == 26069 |
| case schema | **empty** | 1 (`schema_version` const) | 9013 == 9013 |
| common.py | **empty** | 1 (`CERT_SCHEMA_ID`) | 52671 == 52671 |
| validate.py | **empty** | 0 | 6543 == 6543 |
| producer.py | **empty** | 0 | 16398 == 16398 |
| independent_verify.py | **empty** | 2 (cert check + emit) | 60724 == 60724 |
| wrapper.sh | **empty** | 0 | 2215 == 2215 |
| sbatch | **empty** | 0 | 758 == 758 |
| **TOTAL** | **8 empty + 1 config cascade** | **6** | 8 equal; config 8709 → 9129 |

Token audit confirmed independently: every v4 file has **zero** residual `v3` tokens, and its
`v2` count equals its `cert_v2` count (i.e. the only surviving `v2` is the frozen shared cert).
`scgp_global_cert_v2` is a Run1-frozen cross-lineage shared schema, bound (unchanged hash
`4d3f…22f`) in the config `run1_frozen` set and **not** among the nine cloned entities;
preserving it is required (rewriting to `…cert_v4` would point at a nonexistent schema and break
the frozen hash binding). Verified v4 internal cross-references resolve consistently: config
`run.run_id`/`schema_id`/`authorized_run_ids` = `…-v4`/`…payload_v4`; `paths.cert_schema` stays
`…cert_v2…`; wrapper invokes the three v4 `.py` scripts + v4 config; validate/producer import
`lb_scgp_global_r2_run2_v4_common`; sbatch `--job-name=lbscgp_global_r2_run2_v4`, resources
unchanged (`--cpus-per-task=8`, `--mem=64G`, GPU 0, no `--time`, `conda activate HateVideo`).

---

## 4. Amendment-driven binding updates (config class-3 recompute audit)

Editing the authoritative `EXPERIMENT_PLAN.machine.json` forces a bounded cascade that the
runtime asserts **do** check (`…v4_common.py:840`→`:834` `verify_expected_hashes`;
`…v4_independent_verify.py:952-954`). The v4 config `hash_bindings.authoritative_inputs`:

| binding key | before | after |
|---|---|---|
| `…/EXPERIMENT_PLAN.machine.json` | `6caa5c2e…0492` | `42bf49ed…4590a90` (M′) |
| `…/EXPERIMENT_PLAN_HASHES.sha256` | `2e6d731d…c802` | `910f0f64…1568b1` (H′) |
| `…/EXPERIMENT_PLAN.md` | `af1c217c…7b23a` | `a98effc3…5ae3eb` |
| `…/EXPERIMENT_TRACKER.md` | `327614bb…00db2` | `4d3c4b8c…9e9da4` |
| `…/M0_RUN2_V4_PLAN_AMENDMENT.md` | (absent) | `8428b7f8…b42422` (added) |
| `…/M0_RUN2_V4_PLAN_AMENDMENT.machine.json` | (absent) | `30221b10…48d8` (added) |
| `…/M0_RUN2_V4_PLAN_AMENDMENT_HASHES.sha256` | (absent) | `3bbfd910…06aec` (added) |

Underlying file cascade: `EXPERIMENT_PLAN.machine.json` `6caa5c2e…`→`42bf49ed…` (M′);
`EXPERIMENT_PLAN.md` `af1c217c…`→`a98effc3…`; `EXPERIMENT_TRACKER.md` `327614bb…`→`4d3c4b8c…`;
`EXPERIMENT_PLAN_HASHES.sha256` (lines 1–3 updated) `2e6d731d…`→`910f0f64…` (H′). The two runtime
dual-check sites read exactly `cfg.hash_bindings.authoritative_inputs` (`common:840`→`834` and
`independent_verify:952-954`); both were re-played read-only and **all 27 authoritative_inputs +
10 run1_frozen + 4 declared-provenance bindings match on-disk** (`hash_layer_fail=0`). The
`run1_frozen` set is unchanged (the amendment does not touch run1). The v2 amendment + review
bindings remain in the config as historical provenance.

**Deferred (per the v3 verdict sequencing note):** the v4 amendment **independent review** docs
(`M0_RUN2_V4_PLAN_AMENDMENT_REVIEW.*`, `M0_RUN2_V4_AMENDMENT_INDEPENDENT_REVIEW.md`) do not exist
yet and are **not** bound here; the execution authorizer adds those bindings after the review and
re-freezes the config. The config hash `118afadf…3bf0f` recorded in §1 is therefore the
pre-review ceremony freeze target and will change once the review bindings are added — expected,
and consistent with the v2 precedent (config re-frozen after review).

---

## 5. Runtime cross-check static-simulation table (v3 verdict §5(c) — v4-prep pre-run)

Every runtime assertion that reads a frozen external document, statically evaluated against the
on-disk amended state. **All 13 rows PASS** (line numbers carried from v3; v4-verified):

| Row | Assert (site) | Reads | Static verdict | PASS? |
|---|---|---|---|---|
| 1 | `run_order[2] == RUN2` (`common:821`) | plan | plan `run_order[2]`=`…-v4` == `RUN2`=`…-v4` | **PASS** |
| 2 | `runs[2].run_id == RUN2` (`common:822`) | plan | `runs[2].run_id`=`…-v4` == `…-v4` | **PASS** |
| 3 | `runs[2].artifact_paths == [cfg.run.artifact_path]` (`common:823`) | plan+cfg | both `[…/v4/…/manifest.json]` | **PASS** |
| 4 | `runs[2].artifact_schema_ids == [PAYLOAD_SCHEMA_ID]` (`common:824`) | plan | both `[…payload_v4]` | **PASS** |
| 5 | `runs[2].slurm == cfg.run.slurm` (`common:825`) | plan+cfg | dicts equal (order-independent), unchanged | **PASS** |
| 6 | `runs[2].dependencies == [RUN1, RUN2_V1]` (`common:826`) | plan | `[freeze-v1, synth-kkt-v1]`, unchanged | **PASS** |
| 7 | `verify_expected_hashes(authoritative_inputs)` (`common:840`→`834`) | cfg-bound hashes | all 27 == on-disk after cascade (esp. machine=M′, PLAN_HASHES=H′) | **PASS** |
| 8 | `verify_expected_hashes(run1_frozen)` (`common:846`) | run1 frozen | all 10 == on-disk; run1 untouched | **PASS** |
| 9 | `old_protected manifest/count` (`common:858-861`) | old lb_scgp tree | scope = `configs/lb_scgp/`, `artifacts/lb_scgp/`, `refine-logs/lb_scgp/`, and `lb_scgp_*` non-`lb_scgp_global_r2_*` scripts (`…v4_common.py:766-786`); amendment (`refine-logs/lb_scgp_global/`) and v4 code (`lb_scgp_global_r2_run2_v4_*`) are **excluded** → manifest `243e89b…`/count 278 unchanged | **PASS** |
| 10 | `resource_and_run_check` (`validate:110-122`) | cfg internal | `run_id`/`schema_id`/`artifact_path`(literal `…/v4/…`)/`slurm`/`authorized_run_ids` all v4 | **PASS** |
| 11 | `authoritative_inputs unchanged` (`independent_verify:952-954`) | cfg-bound hashes | same set as row 7; all match on-disk | **PASS** |
| 12 | `manifest.artifact_schema_id/run_id == SCHEMA_ID/RUN2` (`independent_verify:1039`) | produced manifest | producer writes `PAYLOAD_SCHEMA_ID`=`…payload_v4` / `RUN2`=`…-v4` (`producer:297-298`); verifier `SCHEMA_ID`/`RUN2` = v4 | **PASS (by construction)** |
| 13 | `manifest.authorized_boundary == {run_id:RUN2, synthetic_only:True, run3_or_later_locked:True}` (`independent_verify:1043`) | produced manifest | producer writes `{run_id:RUN2(v4), synthetic_only:True, run3_or_later_locked:True}` (`producer:300-304`) | **PASS (by construction)** |

Load-bearing insight (from the verdict): rows 7/8/11 (hash layer) verify only "the document I
read is the document I froze"; they are structurally blind to rows 1–4 (code-constant vs
frozen-content). **In v3 the hash layer PASSED while rows 1–2 FAILED** (code=v3 vs plan content=v2).
Here the amendment made the plan content v4 in lock-step with the code constants (`RUN2`=v4), so
rows 1–4 and the hash layer both PASS. This is a v4-prep pre-run; the fresh independent v4 code
review must re-derive this table itself and confirm every row PASS before authorization.

---

## 6. Terminal-state confirmations (read-only)

- `artifacts/lb_scgp_global/v4/` **does not exist** (only `artifacts/lb_scgp_global/v1/` is
  present). No v4 manifest/source_manifest/access_ledger/semantic_verification/publish-lock exists.
- The nine v3 source entities were **not modified** (their hashes still match `M0_RUN2_V3_CLONE_FREEZE.md`);
  the nine v4 entities are new untracked files. The pre-amendment plan is backed up at
  `EXPERIMENT_PLAN.machine.json.pre_v4_amendment.bak` (sha256 `6caa5c2e…0492`).
- No environment mutation, no package install, no interpreter run, no SLURM job, no git commit.
- Residual static findings **M-A** and **M-B** carry unchanged to v4 (they live in the step-8+
  numeric section, downstream of the verify_machine_run2 death point) and must be re-adjudicated
  by the fresh v4 code review.

---

## Status flags

- `ready_for_review = true` — ready for the §5(a) independent v4 amendment review (which must
  ratify the REPLACE-at-index-`[2]` semantics as the byte-clone analogue of the v2 INSERT) and
  the subsequent §5(d) fresh 0C/0H v4 static code review (which must independently re-derive the
  §5(c) table above and re-adjudicate M-A / M-B).
- `ready_for_execution = false` — execution remains unauthorized. The independent amendment
  review, dependency-availability evidence (`jsonschema` incl. the deferred in-function import),
  fresh v4 code review, exact-hashes/no-clobber review, and separate execution authorization are
  all still required before any single executor submit.

## Role separation

The v4-prep role (this document + `M0_RUN2_V4_PLAN_AMENDMENT.md` + `.machine.json` + `_HASHES.sha256`
+ the plan/hash/config edits + the nine-entity clone) is separate from the independent
amendment-review, fresh v4 code-review, execution-authorization, and executor roles. This document
authorizes no execution.

## Required statements

- No performance evidence exists and no performance claim is made; none is possible from a static
  amendment/clone.
- The only project gold is `parent_video_binary_label`. No segment/frame/timestamp/span/
  localization/stance/target/mechanism/rationale/fragment gold is assumed or introduced; v4
  produced no artifact and no counters.
- Run3, M1, MLLM/cache, validation/test, training, and realbank remain locked.
- The v2 and v3 single-submit budgets are spent and both lineages are closed; this document
  authorizes no execution. The full §5 ceremony must complete before any v4 submission.

---

## Revision note

- corrected 2026-07-13 per `M0_RUN2_V4_CODE_REVIEW.md` M-C-doc: §3 config row originally
  mis-stated as "empty diff / 8709 == 8709" (carried from `M0_RUN2_V3_CLONE_FREEZE.md` without
  re-checking the post-cascade state). The config is the **sole class-3 entity**: its diff versus
  a pure `sed(v3)` clone is exactly the declared `authoritative_inputs` binding cascade (4 changed
  values + 3 added), and its byte length is **8709 → 9129**. The §3 config row, the summary row,
  and the section-intro sentence are corrected accordingly; all other content is unchanged.

---

## Post-review binding addendum (2026-07-13)

Added by the **independent execution authorizer** (Claude Opus 4.8, fresh 0C/0H; role separate from
v4-prep, the merged amendment/code reviewer, and the executor) as the mandatory two-phase-freeze
obligation recorded in `M0_RUN2_V4_CODE_REVIEW.md` §8 item 4 (待裁点④) and following the v2
precedent (config re-frozen after review). Only the review-doc binding below was added to the v4
config's `hash_bindings.authoritative_inputs`; **no** plan / `.machine.json` / `_HASHES.sha256` /
existing-amendment document was touched, and the config is **not** self-bound.

**Bound key (appended at the chronological end of `authoritative_inputs`):**

| binding key | review-doc SHA256 | before |
|---|---|---|
| `refine-logs/lb_scgp_global/M0_RUN2_V4_CODE_REVIEW.md` | `41650dcea19c6abd88b0755195ba9333abb43331d68e12f5a5f0d72b2a82d9dc` | (absent) → added |

The v4 review ceremony was a **single merged** document (amendment-ratification + fresh 0C/0H code
review) in `M0_RUN2_V4_CODE_REVIEW.md`; the separately-anticipated
`M0_RUN2_V4_PLAN_AMENDMENT_REVIEW.*` / `M0_RUN2_V4_AMENDMENT_INDEPENDENT_REVIEW.md` files were never
created, so exactly one review doc exists and exactly one review binding is added (the v2 lineage's
three separate review docs collapse into this one). `authoritative_inputs` count **27 → 28**.

**Config freeze hash (nine-entity §1 config row moves off `118afadf…`, as anticipated in §4):**

| | SHA256 |
|---|---|
| config **before** binding (= §1 target) | `118afadfc18cb493a298eda516160f531abfce982471ea836a2d1c6c35f3bf0f` |
| config **after** binding (**executor must verify THIS**) | `e67fc6f79073b8a425ce24834db966bd5b61a2282e2247a5ef87e396056844d1` |

**Post-binding verification (this session, read-only after the single surgical edit):**
`jq -e .` valid; the newly-added value equals the on-disk review-doc hash; the full
`authoritative_inputs` replay is **28/28 == on-disk, fail=0** (this is exactly the runtime
`verify_expected_hashes(authoritative_inputs)` at `…v4_common.py:840`→`:834` and
`…v4_independent_verify.py:952-954`), and `run1_frozen` is **10/10, fail=0**. The other eight §1
entities are unchanged (only the config row moves).

**Hash-freeze consequence:** with `M0_RUN2_V4_CODE_REVIEW.md` now under runtime hash validation
(`common:840`), that review document must **not** change by a single byte from `41650dce…`; any edit
would fail the executor's runtime replay (fail-closed). This addendum is appended to
`M0_RUN2_V4_CLONE_FREEZE.md`, which is **not** bound in the config (verified), so appending here does
not affect any runtime hash check.

Authorizer signature: Claude Opus 4.8 — independent execution-authorization role, v4 lineage,
2026-07-13. Binding obligation complete; from here the authorizer makes no further edit to any
plan / machine / HASHES / amendment / review document.
