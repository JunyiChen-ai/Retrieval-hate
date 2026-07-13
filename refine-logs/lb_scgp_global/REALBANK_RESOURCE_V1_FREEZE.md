# M0 REALBANK-RESOURCE-v1 Implementation Freeze

Date: 2026-07-13

Author: **Claude Opus 4.8**, **realbank-prep role only** — separate from the independent amendment
reviewer, the fresh 0C/0H realbank code reviewer, the execution authorizer, and the executor. This
document freezes the eight-entity realbank implementation and records the runtime cross-check
static-simulation table as a pre-review predemonstration. It authorizes no execution.

Discipline: no project Python was executed, no `py_compile`, no import, no conda/SLURM/`sbatch`/
`squeue`, no experiment, no MLLM/OCR/API/network/model/GPU/training/evaluation, and no validation/
test/cache content was touched. `jq -e .`, `bash -n`, `sha256sum`, `grep`, `find`, `git status` were
read-only. No artifact under `artifacts/lb_scgp_global/v1/m0/realbank_resource/` was created (its
absence is confirmed below). Nothing was committed to git; no SLURM job was submitted.

Spec and decisions are in `REALBANK_RESOURCE_V1_RECON.md` (the blocking spec gap) and
`REALBANK_RESOURCE_V1_PLAN_AMENDMENT.md` (the coordination-session A/B/C ruling). In one line: this
run is a **train-bank static/resource microbenchmark** — it measures `resource_peak`, `rank_tail<=d`,
in-job `replay_hashes` determinism, `robust_coverage` (fail-open), and `isolation_injection_results`
of the real-N dense linear-algebra + eigendecomposition + rank-factor + structural-moment pipeline on
the two frozen CLIP-L/336 train banks, reusing the accepted v4 math; **no training, no kNN, no
performance/accuracy claim**.

---

## 1. Entities frozen (8) — SHA256

| # | entity | SHA256 |
|---|---|---|
| 1 | `configs/lb_scgp_global_r2/m0_realbank_resource_v1.json` | `c436c3dd7e5342707a3ee1a16662e4ab0a74cd0fe39442002cabc2fd6d06fbcf` |
| 2 | `schemas/lb_scgp_global_r2/scgp_global_realbank_resource_v1.schema.json` | `db79cdd36b878ff8286d6471e20638349496bf19230d29c3bd3c2e78be7a73d2` |
| 3 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v1_common.py` | `46e1f3fe36ba93a43d947a166d465a4f06580e535eeb927e49234e374f41b8a9` |
| 4 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v1_validate.py` | `b2bbec0298e6cff75759f52e8f9a4d3d9dcfdecc65ce0188cc96b97051611ded` |
| 5 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v1_producer.py` | `dc38d5c39f109cc3cafb670aecfe5b31f09c18bf33e919f32e0c5c3c7241c114` |
| 6 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v1_independent_verify.py` | `49cc2d9afa12f062ab18e96edaec74e7831f344f702561dda7e507af70f54f26` |
| 7 | `scripts/wrappers/lb_scgp_global_r2_realbank_resource_v1.sh` | `f80b41eaab7b6fa5bb80324f04a8af11b975e1dcdbefe9202323caae72d4b0a1` |
| 8 | `scripts/slurm/lb_scgp_global_r2_m0_realbank_resource_v1.sbatch` | `9c4ecc055bf8f02b4a4c471ad230b0e18b32f85407e61dcd207403d0308ab420` |

Amendment lineage (bound in the config `authoritative_inputs`):
`REALBANK_RESOURCE_V1_PLAN_AMENDMENT.md` `3333c434…f2c00d37`;
`.machine.json` `ccae7f67…69d6b81b`; `_HASHES.sha256` `26462aa0…fe1b173e1`.
Post-amendment plan cascade: `EXPERIMENT_PLAN.machine.json` `42bf49ed…`→`d5023b62…`;
`EXPERIMENT_PLAN.md` `a98effc3…`→`10fd5232…`; `EXPERIMENT_TRACKER.md` `4d3c4b8c…`→`d226abfe…`;
`EXPERIMENT_PLAN_HASHES.sha256`→`a8360a2a…`. Pre-amendment plan backed up at
`EXPERIMENT_PLAN.machine.json.pre_realbank_amendment.bak` (`42bf49ed…4590a90`).

Train banks pinned (A): MHC `deea74ff…dc73e` (N=549), MHC_zh `929571f8…f8f17` (N=579).

## 2. Dependencies (all HateVideo-provided; validator-confirmed at runtime)

- Third-party: `numpy`; `torch` (function-level import in both bank loaders,
  `common.load_bank_features` and the verifier copy); `jsonschema` (function-level in both schema
  validators). The SLURM validator's `python_dependency_check` fails closed if any of
  `numpy`/`torch`/`jsonschema` is missing, before the producer runs.
- Standard library only otherwise: `argparse`, `copy`, `hashlib`, `json`, `math`, `os`, `resource`,
  `subprocess`, `sys`, `tempfile`, `pathlib`, `typing`.

## 3. Runtime cross-check static-simulation table

Every runtime assertion that reads on-disk state, statically evaluated against the frozen on-disk
state. Rows marked PASS were verified read-only this session; DEFERRED rows run inside the single
SLURM job (login-node execution is forbidden by project discipline) and fail closed.

| Row | Assert (site) | Reads | Static verdict | PASS? |
|---|---|---|---|---|
| 1 | wrapper `RUN_ID==EXPECTED`; config `run_id`/`artifact` via `jq` | config | `run_id=…-REALBANK-RESOURCE-v1`, `artifact=…/realbank_resource/decision.json` | **PASS** |
| 2 | `require_slurm_realbank` 16 CPU / 96 GB / 0 GPU | SLURM env | sbatch `--cpus-per-task=16 --mem=96G`, no GPU, no `--time` | **PASS (env at runtime)** |
| 3 | validate `jq -e .` config/machine/schema/run1 | 4 files | all 4 valid JSON | **PASS** |
| 4 | validate `schema_strict_check` | schema | 21 object schemas, 21 `additionalProperties:false` (all strict) | **PASS** |
| 5 | validate `bash -n` wrapper + sbatch | 2 scripts | both `bash -n` clean | **PASS** |
| 6 | validate dependency `numpy/torch/jsonschema` | env | present in HateVideo (torch precedented by `dataset.py`, `lb_scgp_sanitize_inputs.py`; jsonschema by v4) | **PASS (runtime)** |
| 7 | validate `py_compile` the 4 `.py` | scripts | login-node `py_compile` forbidden by discipline; runs in the SLURM validator | **DEFERRED-TO-RUNTIME** |
| 8 | validate `verify_run1_hashes` | run1 frozen + old_protected | 10/10 run1 == on-disk (re-`sha256sum`'d, match v4 bindings); old_protected `243e89b…`/278 runtime-verified (realbank files are `lb_scgp_global_r2_*` → excluded) | **PASS (run1) / runtime (old_protected)** |
| 9 | validate `verify_authoritative_hashes` | 10 authoritative inputs | 10/10 bound == on-disk (each hash computed from on-disk this session) | **PASS** |
| 10 | validate `verify_train_bank_bindings` | 2 banks | 2/2 sha == on-disk (`deea74ff…`, `929571f8…`); `train_n` 549/579 | **PASS** |
| 11 | validate `no_clobber_check` | artifact dir | `artifacts/lb_scgp_global/v1/m0/realbank_resource/` absent | **PASS** |
| 12 | validate `resource_and_run_check` | config | `run_id`/`schema_id`/`artifact_path`/`slurm`(16/96/0)/`authorization` all match | **PASS** |
| 13 | producer `verify_config_and_schema` | config | 10 authorization flags `false` + `train_bank_read_allowed=true` + schema strict | **PASS** |
| 14 | producer `verify_machine_realbank` | machine `runs[3]` | `run_order[3]`/`run_id`/`artifact_paths`/`artifact_schema_ids`/`slurm`/`dependencies=[…-v4]`/`realbank_protocol.banks` all == config | **PASS** |
| 15 | producer manifest schema validation | schema | top-level keys == `schema.required` (23); every nested row emits exactly its `required[]` set | **PASS (by construction)** |
| 16 | verifier `set(manifest)==TOP_KEYS` | manifest | `TOP_KEYS` == `schema.required` == producer keys (23) | **PASS (by construction)** |
| 17 | verifier `zero_counters` set | manifest | schema == common == verifier, 47/47 set-equal | **PASS** |
| 18 | verifier `verify_machine` `runs[3]` | machine | identical to row 14 | **PASS** |
| 19 | verifier injection classifier recompute == manifest | recompute | producer `isolation_injection_cases` and verifier `recompute_injection_classifier` share identical `forbidden_reason` logic and the same 11 probes | **PASS (by construction)** |
| 20 | verifier authoritative/run1 on-disk == config | files | identical to rows 8–9 | **PASS** |
| 21 | verifier GO consistency (`within_cap ∧ rank_le_d ∧ replay ∧ inject`) | recompute | at N=549/579 the O(N³) pipeline peak is far under 96 GiB, `rank_eps(G0) ≤ N ≤ d`, in-process replay is bit-deterministic, isolation guard rejects all probes — GO expected | **PASS (expected)** |

**Load-bearing insight (v4 precedent):** the hash layer (rows 8–10, 20) verifies only "the file I
read is the file I froze"; it is blind to whether code constants match frozen *content* (rows 14/18).
Here the amendment made `runs[3]` content (`…-REALBANK-RESOURCE-v1`, banks, slurm) match the code
constants (`RUN3`, `config.train_banks`, `expected_slurm_block`) in lock-step, so both layers PASS.
The fresh v4-analogue code review must independently re-derive this table.

## 4. Residuals / flags for the fresh code review (non-blocking here)

- **R-1 (torch load semantics).** `load_bank_features` uses `torch.load(..., weights_only=True)`,
  matching `lb_scgp_sanitize_inputs.py`'s proven call on the same `{ids,img_feats,text_feats}` cache
  family. The repo's own `src/data_loader/dataset.py` loads these exact banks with the plain default.
  If `weights_only=True` rejects the payload it fails closed (clean raise, no artifact), not a silent
  wrong result. The reviewer should confirm the load mode against the installed torch.
- **R-2 (determinism).** The replay digest and the producer↔verifier cross-check assume `numpy`
  `eigvalsh`/`svd` are bit-deterministic for a fixed input and thread count (`OMP/MKL/OPENBLAS=16`,
  set in the sbatch and identical for producer and verifier in one job). `floatify` (15 sig figs,
  zero `<5e-16`) absorbs last-ULP noise. This is the same assumption the accepted v4 verifier relies
  on. In-job run1==run2 is bit-identical (same process).
- **R-3 (old_protected).** The config binds the v4-frozen `old_protected` snapshot (`243e89b…`/278),
  unchanged because all realbank files are `lb_scgp_global_r2_*` (excluded) and live under
  `refine-logs/lb_scgp_global/` (not `refine-logs/lb_scgp/`). The SLURM validator recomputes and
  fails closed on any drift.
- **R-4 (structural placeholder).** `b_struct` is a NON-SCIENCE label-blind placeholder
  (`is_science=false`, disclosed in the artifact and re-checked by the verifier). The science owner
  retains the right to overrule it before any downstream scientific claim rests on it (M-A-analogue).

## Status flags

- `ready_for_review = true` — ready for the independent amendment review (ratify A/B/C + the additive
  `runs[3]` edit) and the fresh 0C/0H realbank code review (which must independently re-derive §3 and
  re-adjudicate R-1…R-4).
- `ready_for_execution = false` — execution remains unauthorized. Independent amendment review,
  dependency-availability evidence, fresh 0C/0H code review with the §3 table all-PASS, exact-hashes/
  no-clobber review, and separate execution authorization are all still required before any single
  executor submit.

## Required statements

- No performance evidence exists and none is claimed; none is possible from a static amendment/
  implementation, and the realbank run itself emits no accuracy/macro-F1 and does no training or kNN.
- The only project gold is `parent_video_binary_label`; no segment/frame/timestamp/span/localization/
  stance/target/mechanism/rationale/fragment gold is assumed or introduced, and train labels are not
  opened by realbank (train **features** are; train **labels** are not).
- Run4 (M1 cache), MLLM/cache, validation/test, and training remain locked. This freeze unlocks
  neither realbank execution nor anything downstream.
- The realbank-prep role is separate from the independent amendment-review, fresh code-review,
  execution-authorization, and executor roles. This document authorizes no execution.
