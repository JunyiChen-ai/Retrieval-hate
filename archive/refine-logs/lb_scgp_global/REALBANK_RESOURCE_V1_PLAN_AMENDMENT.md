# M0 REALBANK-RESOURCE-v1 Plan Amendment (A/B/C input protocol)

Date: 2026-07-13

Author: **Claude Opus 4.8**, acting in the **realbank-prep role only**. This role is separate from —
and does not perform the work of — the independent amendment reviewer, the fresh 0C/0H realbank code
reviewer, the execution authorizer, and the executor. This document authors the realbank plan
amendment (pinning decisions A/B/C the frozen plan left undetermined), edits the authoritative plan
additively, runs the amendment-driven hash cascade, implements the full realbank entity set, and
freezes it. It does **not** review, authorize, or execute it.

Discipline: static amendment + implementation + freeze only. No project Python was executed, no
`py_compile`, no import, no conda/SLURM/`sbatch`/`squeue`, no experiment, no MLLM/OCR/API/network/
model call, no GPU/training/evaluation, and no validation/test data or cache was touched. `jq -e .`
was read-only JSON well-formedness; `sha256sum`/`grep`/`find` were read-only. No artifact under
`artifacts/lb_scgp_global/v1/m0/realbank_resource/` was created. Nothing was committed to git and no
SLURM job was submitted.

This amendment is **not** performance evidence, **not** execution authorization, and **not** an
M1/cache/validation-test/training unlock. It resolves the blocking spec gap recorded in
`REALBANK_RESOURCE_V1_RECON.md` by ratifying the coordination-session A/B/C ruling and building the
entities, pending fresh independent review.

---

## 0. Why an amendment was required

`REALBANK_RESOURCE_V1_RECON.md` established that the frozen plan pins *what* realbank measures
(`resource_peak`, `rank_tail<=d`, `replay_hashes`, `robust_coverage`, `isolation_injection_results`;
16 CPU / 96 GB / 0 GPU; `decision.json`; schema `scgp_global_realbank_resource_v1`) but not *what
data the microbenchmark consumes*, and that this is entangled with a run-order paradox: realbank is
`run_order[3]`, but the M1 MLLM cache that supplies `b_struct` (`[4-6]`) and the M2 comparator freeze
that defines `Z0` (`[7]`) both run **after** it. Deciding the input construction is a plan-owner call,
not an implementer call. The coordination session ruled A/B/C (below); this amendment records that
ruling and is itself subject to fresh independent review.

A load-bearing fact behind the design: the accepted Run2-v4 code does **not** contain a convex
solver — it **constructs** synthetic KKT-closed tuples and verifies them. So realbank is not "run the
solver on real data" (there is none); it is a resource/rank/replay/coverage/isolation microbenchmark
of the real-N dense linear-algebra + eigendecomposition + rank-factor + structural-moment pipeline on
the real train banks, reusing the exact accepted v4 math.

## 1. Decision A — train-bank source (`Z0`)

Preregister the frozen CLIP-L/336 pooler train bank as the realbank `Z0` stand-in for the (not-yet-
frozen) M2 comparator: `Z0 = rownorm(concat(img_feats, text_feats))`, `G0 = psd_gram_from_features(Z0)`,
`d = Z0.shape[1]` read at runtime; `rank_tail` is `rank_eps(G0) <= d`. Banks (sha256 pinned into the
machine plan `runs[3].realbank_protocol` and here):

| dataset | path | sha256 | train_n |
|---|---|---|---|
| MHC | `data/CLIP_Embedding/MHC/train_openai_clip-vit-large-patch14-336_HF.pt` | `deea74ff…dc73e` | 549 |
| MHC_zh | `data/CLIP_Embedding/MHC_zh/train_openai_clip-vit-large-patch14-336_HF.pt` | `929571f8…f8f17` | 579 |

Train **content** (features) is opened via a hash-checked allowlist; train **labels** are not opened;
`d`-vs-`N` guarantees `rank_eps(G0) <= min(N,d) <= d`, so the rank gate is a real, passing consistency
check. The eigendecomposition cost is `O(N^3)` and `d`-independent; the semantic identity of `Z0`
does not change the resource conclusion (`N` drives it), and the real comparator `Z0` is frozen only
at M2.

## 2. Decision B — structural placeholder (preferred path)

The sealed MLLM cache does not exist yet, so `b_struct` cannot be the real cache moment. Per the
ruling's **preferred** path, realbank uses a deterministic, label-blind, **NON-SCIENCE** placeholder
`b_struct = vech(M_Q(G0))` built from a deterministic label-blind `Phi` seed and `Q = orth_cap(Phi)`
at the worst-case scale `r = r_max = 8`, `m = r(r+1)/2 = 36` (the `m` scale is supplied by
`FINAL_PROPOSAL.md`'s compute envelope, "even with r<=8, m<=36"). This opens the
orth_cap / structural-moment / structural-adjoint code path (including the dense `N x N` adjoint
allocation) at the real `N` so the measured peak RSS is faithful to an upper bound on the eventual
FULL structural target. It certifies nothing. The fallback (G0-only with a "peak = lower bound,
re-measure after M1" caveat) is therefore **not** used. The placeholder's `is_science=false` status is
disclosed in the artifact (`structural_placeholder`) and re-checked by the independent verifier.

## 3. Decision C — acceptance (two-stage, v4 precedent)

Producer emits a `PRODUCED_PENDING_INDEPENDENT_VERIFY` candidate; a **separate** independent verifier
re-loads the banks, independently recomputes G0 / placeholder / rank-factor / replay digest / coverage
/ injection classifier, runs fail-closed manifest mutations that must all be REJECTED, and stamps
`decision PASS/FAIL`. GO criterion:

> GO iff `job_peak_rss_bytes <= 103079215104` (96 GiB) AND `rank_eps(G0) <= d` for every dataset AND
> the in-job second replay hash matches the first for every dataset AND every isolation injection is
> REJECTED.

`robust_coverage` is reported, fail-open (low coverage disables only the robust safety claim). The
strict schema's five science keys are `resource_peak`, `rank_tail`, `replay_hashes`,
`robust_coverage`, `isolation_injection_results`.

## 4. Plan edits (additive) + hash cascade

- `EXPERIMENT_PLAN.machine.json` `runs[3]`: added `realbank_protocol` (A/B/C) and `gate_satisfied_by`;
  status `LOCKED_UNTIL_V4_PASS` → `GATE_OPEN_PENDING_REALBANK_IMPLEMENTATION_AND_REVIEW`. `runs[2]` and
  every other index untouched; array length unchanged; `run_order` unchanged.
- Folded the v2→v4 documentation drift: `EXPERIMENT_TRACKER.md` row 4 and `EXPERIMENT_PLAN.md` item 4
  now correctly say realbank depends on Run2-**v4** PASS + fresh independent v4 artifact review.

| file | before | after |
|---|---|---|
| `EXPERIMENT_PLAN.machine.json` | `42bf49ed…4590a90` | `d5023b62…cb18fdb` |
| `EXPERIMENT_PLAN.md` | `a98effc3…95ae3eb` | `10fd5232…1c53fa3a` |
| `EXPERIMENT_TRACKER.md` | `4d3c4b8c…4c9e9da4` | `d226abfe…6b3ab3b53` |
| `EXPERIMENT_PLAN_HASHES.sha256` | (recomputed) | `a8360a2a…a6bddd3e` |

**Consumed-v4-config note:** the CONSUMED-and-CLOSED v4 config (`m0_synth_kkt_v4.json`) still binds the
pre-amendment machine hash `42bf49ed…` in its `authoritative_inputs`. v4 is single-submit-spent and
`ARTIFACT_ACCEPTED`; it will not re-run, so that binding is historical provenance and is **not**
retro-updated (re-freezing a consumed artifact's config would itself be a lineage violation). The new
machine hash `d5023b62…` is bound going forward only in the realbank config. This mirrors the v4
amendment's treatment of the consumed v2/v3 lineages.

## 5. Entities implemented (self-contained; no cross-lineage import)

The realbank code is **self-contained**: its pure numerical/serialization helpers are byte-faithful
copies of the accepted Run2-v4 code (so the exact verified linear algebra is reused without a
cross-lineage import), mirroring how the v4 independent verifier is self-contained. New realbank
orchestration: 16/96 SLURM guard, `runs[3]` machine verifier, train-bank-aware access ledger (val/
test/held/cache/teacher/query_z/query_labels all forbidden; only the two allowlisted hash-checked
banks open), real-bank load, NON-SCIENCE structural placeholder, resource/rank/replay/coverage/
isolation pipeline, source manifest.

| # | entity | sha256 |
|---|---|---|
| 1 | `configs/lb_scgp_global_r2/m0_realbank_resource_v1.json` | recorded in `REALBANK_RESOURCE_V1_FREEZE.md` (finalized after binding these amendment docs) |
| 2 | `schemas/lb_scgp_global_r2/scgp_global_realbank_resource_v1.schema.json` | `db79cdd3…be7a73d2` |
| 3 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v1_common.py` | `46e1f3fe…374f41b8a9` |
| 4 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v1_validate.py` | `b2bbec02…611ded` |
| 5 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v1_producer.py` | `dc38d5c3…7241c114` |
| 6 | `scripts/analysis/lb_scgp_global_r2_realbank_resource_v1_independent_verify.py` | `49cc2d9a…70f54f26` |
| 7 | `scripts/wrappers/lb_scgp_global_r2_realbank_resource_v1.sh` | `f80b41ea…72d4b0a1` |
| 8 | `scripts/slurm/lb_scgp_global_r2_m0_realbank_resource_v1.sbatch` | `9c4ecc05…308ab420` |

Third-party imports (all HateVideo-provided; confirmed at runtime by the SLURM validator's dependency
check): `numpy`, `torch` (function-level in the two bank loaders), `jsonschema` (function-level in the
two schema validators). Standard library: `argparse`, `copy`, `hashlib`, `json`, `math`, `os`,
`resource`, `subprocess`, `sys`, `tempfile`, `pathlib`, `typing`.

## 6. Three-burn lessons applied

- **v1 (interface-key mismatch):** the strict schema's key sets are aligned three ways — schema
  `required[]` ↔ producer manifest ↔ verifier `TOP_KEYS`/per-field checks ↔ common `ZERO_COUNTER_KEYS`.
  The static simulation table in `REALBANK_RESOURCE_V1_FREEZE.md` walks the alignment.
- **v3 (index/plan drift):** the code constants pin `runs[3]`/`run_order[3]` and the `runs[3]` content
  is `…-REALBANK-RESOURCE-v1`; the machine verifier asserts both. No numeric index literal is left
  pointing at the wrong run.
- **v2 (missing in-function dependency):** every in-function import (`torch`, `jsonschema`) is listed
  and is validator-checked inside the SLURM job before the producer runs.

## Status flags

- `ready_for_review = true` — ready for the independent amendment review (ratify A/B/C + the additive
  `runs[3]` edit) and the fresh 0C/0H realbank code review (which must independently re-derive the
  static simulation table and re-adjudicate the placeholder disclosure).
- `ready_for_execution = false` — execution remains unauthorized. Independent amendment review,
  dependency-availability evidence, fresh 0C/0H code review with the runtime cross-check static-
  simulation table all-PASS, exact-hashes/no-clobber review, and separate execution authorization are
  all still required before any single executor submit.

## Role separation & required statements

- The realbank-prep role (this document + `.machine.json` + `_HASHES.sha256` + the plan/hash edits +
  the eight-entity implementation + `REALBANK_RESOURCE_V1_FREEZE.md`) is separate from the independent
  amendment-review, fresh code-review, execution-authorization, and executor roles. This document
  authorizes no execution.
- No performance evidence exists and none is claimed; none is possible from a static
  amendment/implementation. The realbank run itself emits no accuracy/macro-F1 and does no training or
  kNN.
- The only project gold is `parent_video_binary_label`; no segment/frame/timestamp/span/localization/
  stance/target/mechanism/rationale/fragment gold is assumed or introduced, and train labels are not
  opened by realbank.
- Run4 (M1 cache), MLLM/cache, validation/test, and training remain locked. This amendment unlocks
  neither realbank execution nor anything downstream.
