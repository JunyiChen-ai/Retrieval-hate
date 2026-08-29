# M0 Run2-v3 Byte-Exact Clone / Freeze

Date: 2026-07-13

Creator: **Claude Opus 4.8**, acting in the **v3-setup role only**. This role is deliberately
separate from — and does not perform the work of — the later roles required by the verdict
§4(d): the fresh 0C/0H static code reviewer, the independent execution authorizer, and the
independent executor. This document creates the v3 clone and freezes it; it does not review,
authorize, or execute it.

Scope / discipline: this was a static clone-and-freeze only. No Python was executed, no
`py_compile`, no import, no `jq`-driven *validation* of payloads, no conda/SLURM/`sbatch`/
`squeue`, no experiment, no MLLM/OCR/API/network/model call, no GPU/training/evaluation, and no
validation/test data or cache was touched. `jq -e .` was used only as a **read-only JSON
well-formedness** check on the three new JSON files (it parses, it does not run project code).
`sed`/`diff`/`sha256sum`/`grep` were used to generate and to *prove* the byte-transform. No
artifact under `artifacts/lb_scgp_global/v3` was created (its absence is confirmed in §5).

This report is **not** performance evidence, **not** execution authorization, and **not** a
Run3/M1/MLLM-cache/validation-test/training/realbank unlock. It implements verdict
`M0_RUN2_V2_RESULT_TO_CLAIM_REVIEW.md` §4 condition **(c)** and the first step of §4(d.1).

---

## 0. Source integrity (what was cloned)

The nine frozen v2 entities were re-hashed immediately before cloning and match the hashes
recorded in `M0_RUN2_V2_IMPLEMENTATION_FIX2_FREEZE.md` (config `5545…f700`, payload
`7352…d9a2`, common `5ef8…e24f`, validate `4389…dc36`, producer `8c4c…1cd51`,
independent_verify `795b…1ef1`, and the three parity files: case `df36…dcac`, wrapper
`14eb…0716`, sbatch `f914…94bf`). The clone therefore derives from the exact frozen v2 bytes.

---

## 1. Nine v3 entities — SHA256 manifest

| # | v3 entity | SHA256 |
|---|---|---|
| 1 | `configs/lb_scgp_global_r2/m0_synth_kkt_v3.json` | `e6d33b5d3078b12d87e4c0dc70d0f4fe1ee53681543da347f0e4402fedceb7d5` |
| 2 | `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_payload_v3.schema.json` | `1d6f93a1b0933a24e361de9ed32abda9dc5f180039d2514121b8c6c8caf2d2d3` |
| 3 | `schemas/lb_scgp_global_r2/scgp_global_synth_kkt_case_v3.schema.json` | `df3616ff50c54dc756bb600c6ef26626afafe070678450d10ca73b9ff83ddcac` |
| 4 | `scripts/analysis/lb_scgp_global_r2_run2_v3_common.py` | `9de62f6df2f68fc46ad24e1c19e33b4bd3eeba9303db8ae383ed271a29c411c2` |
| 5 | `scripts/analysis/lb_scgp_global_r2_run2_v3_validate.py` | `2e0bb00b28debf8bf3b2099ac6363664d9d4d776740f3505311e0f7b74ca13a6` |
| 6 | `scripts/analysis/lb_scgp_global_r2_run2_v3_producer.py` | `6ef3a4a8146ec9b2a2a94236c1e40f0ebf27aa862b89d27a60e92499b21f5114` |
| 7 | `scripts/analysis/lb_scgp_global_r2_run2_v3_independent_verify.py` | `4025dbf0482877cc03c46434631134fe81f0593f3ff12f779c87e80aeb8523c5` |
| 8 | `scripts/wrappers/lb_scgp_global_r2_run2_v3.sh` | `8d9123e9f4eec357a91bd94cbf6c292a3bb188496011845c706f3e34b72d66d3` |
| 9 | `scripts/slurm/lb_scgp_global_r2_m0_synth_kkt_v3.sbatch` | `4495ec3c49970b7024ebf443d845ec5f325013b2190116c547827c6f5de6b3d9` |

Note the case schema (#3) hashes **identically** to the v2 case schema (`df36…dcac`): its sole
`v2` token is the shared/frozen `scgp_global_cert_v2` reference, which is preserved (see §3), so
its content is byte-for-byte identical to v2 and only its filename changes.

---

## 2. The transform (exactly the three permitted change classes)

v3 was produced from the frozen v2 bytes by a single deterministic substitution:

> replace the token `v2` with `v3` **everywhere except inside the shared/frozen `cert_v2`
> token** (`scgp_global_cert_v2`, the Run1-frozen certification schema id/name that is *not* one
> of the nine cloned entities and keeps its permanent `v2` name across all lineages).

Concretely: `sed 's/cert_v2/cert__CERTKEEP__/g; s/v2/v3/g; s/cert__CERTKEEP__/cert_v2/g'`.

This maps precisely onto the three permitted classes and nothing else:

- **(class 1) file names** — the nine files are renamed `…v2…` → `…v3…`.
- **(class 2) internal self-references** — module import names (`lb_scgp_global_r2_run2_v3_common`),
  `run_id` (`LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v3`), schema ids
  (`scgp_global_synth_kkt_payload_v3`, `…case_v3`, `lb_scgp_global_r2_synth_kkt_manifest_v3`,
  `…source_manifest_v1`/`…semantic_verification_v1`/`…validation_v1` lineage stems), the payload
  `$ref` → `scgp_global_synth_kkt_case_v3.schema.json#`, artifact/config/script paths
  (`artifacts/lb_scgp_global/v3/…`, `configs/…/m0_synth_kkt_v3.json`, the three v3 `.py` script
  paths, the v3 wrapper/sbatch paths), the `mktemp` template, the SLURM `--job-name`
  (`lbscgp_global_r2_run2_v3`), and lineage-label strings in log/error messages
  (`"Run2-v3 …"`). The `manifest_v2`→`manifest_v3` retag is confirmed lineage-coupled: the
  v1 lineage used `…manifest_v1`, the v2 lineage `…manifest_v2` (v1 payload L504 / v1 producer
  L283 vs v2 payload L532 / v2 producer L296).
- **(class 3) hash bindings** — **zero change points.** The v2 config binds hashes only of
  external/frozen entities (authoritative docs, `data/gt` provenance, the `old_protected`
  snapshot, and the Run1-frozen set incl. the `cert_v2` schema). **None** of the nine cloned
  entities has its hash bound anywhere in the config, so no binding is recomputed. This was
  verified two ways: (i) the eight non-config entities contain **0** embedded 64-hex literals;
  (ii) `diff` of the config `hash_bindings` block (v2 vs v3, via `jq -S`) is **empty** — all 39
  bound hashes and even the `cert_v2` path key are byte-identical.

No comment, whitespace, numeric, structural, or logic byte was changed. Byte length is identical
for all nine pairs (both `v2` and `v3` are two-character tokens), e.g. config `8709 == 8709`.

---

## 3. Equivalence proof (per pair)

Method: for each v2/v3 pair, `diff <(sed 's/v2/v3/g' <v2-file>) <v3-file>`. A **blanket**
`v2`→`v3` sed over the v2 file is the maximal-change reference; the diff against the actual v3
file therefore surfaces **exactly** the intentional `cert_v2` preservations and nothing else. An
**empty** diff means v3 equals the blanket transform with no exceptions; a **non-empty** diff
must contain only `cert_v3`↔`cert_v2` lines (blanket sed over-rewrites the shared cert token; v3
correctly keeps it). In every file, the count of residual `v2` tokens in the v3 file equals the
count of preserved `cert_v2` tokens — i.e. the *only* surviving `v2` is the frozen cert.

| v3 entity | total v2 in v2-file | changed →v3 | cert_v2 preserved | blanket-sed diff |
|---|---|---|---|---|
| config | 32 | 30 | 2 | 2 cert lines (L75 run1_frozen path, L99 `paths.cert_schema`) |
| payload schema | 5 | 5 | 0 | **empty** |
| case schema | 1 | 0 | 1 | 1 cert line (L15 `schema_version` const); content byte-identical to v2 |
| common.py | 13 | 12 | 1 | 1 cert line (L28 `CERT_SCHEMA_ID`) |
| validate.py | 4 | 4 | 0 | **empty** |
| producer.py | 3 | 3 | 0 | **empty** |
| independent_verify.py | 9 | 7 | 2 | 2 cert lines (L212 check, L234 emit) |
| wrapper.sh | 9 | 9 | 0 | **empty** |
| sbatch | 4 | 4 | 0 | **empty** |
| **TOTAL** | **80** | **74** | **6** | 5 empty / 4 cert-only |

Recorded diff evidence for the four non-empty files (each line is a cert preservation):

```
# config m0_synth_kkt_v3.json
75c75  < …/scgp_global_cert_v3.schema.json …   --->   > …/scgp_global_cert_v2.schema.json …
99c99  < "cert_schema": "…/scgp_global_cert_v3.schema.json"  --->  > "…scgp_global_cert_v2.schema.json"
# case scgp_global_synth_kkt_case_v3.schema.json
15c15  < "schema_version": {"const": "scgp_global_cert_v3", …}  --->  > … "scgp_global_cert_v2" …
# common lb_scgp_global_r2_run2_v3_common.py
28c28  < CERT_SCHEMA_ID = "scgp_global_cert_v3"  --->  > CERT_SCHEMA_ID = "scgp_global_cert_v2"
# independent_verify lb_scgp_global_r2_run2_v3_independent_verify.py
212c212 < if record["schema_version"] != "scgp_global_cert_v3":  --->  > … "scgp_global_cert_v2":
234c234 < out: {…"schema_version": "scgp_global_cert_v3"}  --->  > …"scgp_global_cert_v2"…
```

Why `cert_v2` is preserved and not a behavioral deviation: `scgp_global_cert_v2.schema.json` is a
Run1-frozen, cross-lineage **shared** schema. It is bound (unchanged hash `4d3f…22f`) in the
config `run1_frozen` set and is not among the nine cloned entities. The v1 lineage already
references `scgp_global_cert_v2` (v1 case schema L15, v1 `run2_common` `CERT_SCHEMA_ID`), so the
`v2` in the cert id is a permanent shared-schema name, **not** a lineage marker. Preserving it is
required for v3 to bind the identical frozen cert schema; rewriting it would (a) point at a
nonexistent `scgp_global_cert_v3.schema.json` and (b) break the frozen hash binding — i.e. the
*behavioral* choice is to preserve it. This is the single subtlety that makes a naive blanket
`v2`→`v3` incorrect for the config/case/common/independent_verify files.

Internal v3 cross-references were confirmed to resolve consistently: config `run.run_id` /
`schema_id` / `authorized_run_ids` = `…-v3` / `scgp_global_synth_kkt_payload_v3`; config
`paths.payload_schema`/`case_schema`/`wrapper`/`slurm_script` and `run.artifact_path` point at
the v3 entities while `paths.cert_schema` stays `…cert_v2…`; payload `$ref` → `…case_v3…`;
wrapper invokes the three v3 `.py` scripts and the v3 config; validate/producer import
`lb_scgp_global_r2_run2_v3_common`; independent_verify is (as in v2) intentionally standalone
(imports neither producer nor common); sbatch `--job-name=lbscgp_global_r2_run2_v3` with
resources **unchanged** (`--cpus-per-task=8`, `--mem=64G`, GPU 0, no `--time`, `conda activate
HateVideo`).

### Config hash-binding before/after table (class-3 recompute audit)

No hash-binding field was recomputed. The bound values are byte-identical v2↔v3:

| binding group | # bound hashes | changed? |
|---|---|---|
| `authoritative_inputs` (AGENTS.md, EXPERIMENT_PLAN/TRACKER/REVIEW/AMENDMENT docs, code review) | 24 | no |
| `declared_validation_test_provenance_not_opened` (`data/gt/MHC*`) | 4 | no |
| `old_protected_pre_snapshot.manifest_sha256` | 1 | no |
| `run1_frozen` (v1 artifacts + `cert_v2` schema + contract-freeze schema + `r2_common`/`contract_freeze`/`r2_validate` + contract-freeze sbatch + run1 wrapper) | 10 | no |

The `run1_frozen` cert key path (`…scgp_global_cert_v2.schema.json`) is preserved verbatim, so
the entire `hash_bindings` block is byte-identical (empty `jq -S` diff). Class-3 change points = **0**.

---

## 4. Item-by-item conformance to verdict §4 condition (c)

Condition (c): *"v3 source must be a byte-for-byte-equivalent clone of the frozen v2 entities,
with changes limited strictly to: file names, internal self-references (module names / run_id /
namespace / paths from v2 to v3), and hash bindings. No behavioral code change is permitted. …
The nine v2 entities to clone are those enumerated in `M0_RUN2_V2_IMPLEMENTATION_FIX2_FREEZE.md`
(config, payload/case/cert schemas, the four python modules, wrapper, sbatch)."*

| (c) requirement | status | evidence |
|---|---|---|
| Byte-for-byte-equivalent clone of frozen v2 | **met** | §0 source integrity; §3 blanket-sed diff empty/cert-only; equal byte lengths |
| Change class = file names | **met** | nine files renamed `…v2…`→`…v3…` (§1) |
| Change class = internal self-references (module names / run_id / namespace / paths) | **met** | §2 class-2 list; §3 cross-reference resolution |
| Change class = hash bindings | **met (vacuously)** | §2/§3 class-3 audit: 0 recompute; config binds no cloned-entity hash |
| No behavioral code change | **met** | only 2-char lineage tokens substituted; no comment/whitespace/numeric/logic byte changed; per-file residual `v2` == preserved `cert_v2` |
| Nine entities per FIX2 freeze | **met** | config, payload schema, case schema, common, validate, producer, independent_verify, wrapper, sbatch. (The **cert** schema named in (c)'s parenthetical is the *shared/frozen* `scgp_global_cert_v2.schema.json`; it is a hash-bound dependency, **not** cloned — consistent with the FIX2 freeze's own nine-entity list, which excludes it.) |
| Clone status intact (no behavioral edit forcing a full v3 implementation audit) | **preserved** | mechanical transform + proof; a fresh 0C/0H code review (§4(d.2)) still applies |

This document also discharges §4(d.1) ("Freeze the v3 entities with exact SHA256 bindings") via
§1. Steps §4(d.2) fresh static code review, §4(d.3) independent execution authorization with the
new mandatory dependency-availability evidence item, and §4(d.4) single executor submit remain
**open and unperformed** here; conditions §4(a) environment repair (install `jsonschema` into
`HateVideo`) and §4(b) full deferred-import dependency audit are **outside this role** and remain
prerequisites before any authorization.

---

## 5. Terminal-state confirmations (read-only)

- `artifacts/lb_scgp_global/v3/` **does not exist** (only `artifacts/lb_scgp_global/v1/` is
  present). No v3 manifest/source_manifest/access_ledger/semantic_verification/publish-lock
  exists.
- The nine v2 source entities were **not modified** (their hashes still match FIX2, §0); the nine
  v3 entities are new untracked files. Nothing was committed and no SLURM job was submitted.
- No environment mutation, no package install, no interpreter run.

---

## Status flags

- `ready_for_review = true` — ready for the §4(d.2) fresh 0C/0H static code review (which must
  independently re-adjudicate residual findings M-A and M-B, per the verdict, as they carry to v3).
- `ready_for_execution = false` — execution remains unauthorized. §4(a) environment repair,
  §4(b) dependency audit, §4(d.2) code review, and §4(d.3) execution authorization (with the
  mandatory dependency-availability evidence item) are all still required before any single
  executor submit.

## Required statements

- No performance evidence exists and no performance claim is made; none is possible from a static
  clone.
- The only project gold is `parent_video_binary_label`. No segment/frame/timestamp/span/
  localization/stance/target/mechanism/rationale/fragment gold is assumed or introduced; v3
  produced no artifact and no counters.
- Run3, M1, MLLM/cache, validation/test, training, and realbank remain locked.
- The v2 single-submit budget is spent and the v2 lineage is closed; this document authorizes no
  execution. The full §4 ceremony must complete before any v3 submission.
- The v3-setup role (this document) is separate from the fresh static-code-review,
  execution-authorization, and executor roles.
