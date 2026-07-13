# M0 REALBANK-RESOURCE-v1 — Recon + Spec-Gap / Blocker Report

Date: 2026-07-13

Author: **Claude Opus 4.8**, acting in the **realbank-prep role only** (recon + would-be
implementation + freeze). This role is separate from the later independent amendment reviewer,
code reviewer, execution authorizer, and executor. No project Python was executed, no
`py_compile`/import, no `conda`/SLURM/`sbatch`/`squeue`, no experiment, no MLLM/OCR/API/network/
model/GPU/training/evaluation, and no validation/test data or cache was touched. `jq`/`sha256sum`/
`grep`/`find`/`wc` were used read-only. Nothing was committed to git and no SLURM job was submitted.

**Bottom line:** the gate is **unlocked** (Run2-v4 `ARTIFACT_ACCEPTED`), but the realbank step
**cannot be brought to a review-ready frozen entity set** at this time. The frozen plan pins *what*
realbank must measure and its resources/artifact/schema, but it does **not** determine *what data the
microbenchmark consumes* — and that decision is load-bearing and entangled with a run-order paradox
(realbank runs **before** both the M1 MLLM cache that supplies `b_struct` and the M2 comparator
freeze that defines `Z0`). Building config/schema/scripts now would require me to unilaterally decide
these, which the project discipline forbids. **A plan amendment is required before implementation.**

- `ready_for_review = false` (blocked on amendment; no entities produced)
- `ready_for_execution = false`

---

## 0. Provenance (read-only, this session)

| File | SHA256 | note |
|---|---|---|
| `refine-logs/lb_scgp_global/EXPERIMENT_PLAN.machine.json` | `42bf49ed…4590a90` | authoritative; == the hash bound in the v4 config `authoritative_inputs` |
| `refine-logs/lb_scgp_global/EXPERIMENT_PLAN.md` | `a98effc3…5ae3eb` | == v4-bound |
| `refine-logs/lb_scgp_global/EXPERIMENT_TRACKER.md` | `4d3c4b8c…9e9da4` | == v4-bound |
| `refine-logs/lb_scgp_global/FINAL_PROPOSAL.md` | `b5ab9409…1a9ff` | == plan `source_hashes` |

Gate evidence: `M0_RUN2_V4_ARTIFACT_REVIEW.md` verdict **ARTIFACT_ACCEPTED** (§Verdict, "Row-3
REALBANK-RESOURCE-v1 may unlock"; §5 confirms the row-4 tracker gate "depends on Run2 PASS plus fresh
independent artifact review" is satisfied).

---

## 1. What the frozen plan DOES determine (the spec, verbatim, with locations)

Machine plan `runs[3]` (`EXPERIMENT_PLAN.machine.json:638-688`), tracker row `| 4 |`
(`EXPERIMENT_TRACKER.md:17`), plan §257 / §165 / §76-82, proposal §498-537 / §629-654 / §716-728:

- **run_id**: `LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1`; milestone **M0**, block **B0**, claim **C1**.
- **purpose**: "actual train-bank static/resource microbenchmark plus replay/decision **without
  training or performance claim**".
- **dataset**: `MHC,MHC_zh`.
- **split_scope**: "**train banks and manifests only; no validation/test labels**".
- **system_variant**: `FULL_REALBANK_STATIC_REPLAY`; **baseline_family**: `F1` (REMOVE/null parity),
  `F2` (SHUFFLE / covariance-matched NOISE fixtures).
- **dependencies**: `LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v4` (machine `runs[3].dependencies`, line 654).
- **metrics**: `resource_peak`, `rank_tail_le_d`, `replay_hash`, `robust_coverage`,
  `isolation_injection_failures`.
- **artifact_path**: `artifacts/lb_scgp_global/v1/m0/realbank_resource/decision.json`
  (note the **`v1`** namespace, not v4).
- **artifact_schema_id**: `scgp_global_realbank_resource_v1`. Its frozen top-level keys
  (`artifact_schemas`, `EXPERIMENT_PLAN.machine.json:4087-4093`) are exactly:
  `["resource_peak", "rank_tail", "replay_hashes", "robust_coverage", "isolation_injection_results"]`.
- **slurm**: 16 CPU / 96 GB / 0 GPU, `HateVideo`, no `--time` (CPU-only, as the whole G0 compiler/
  projection/eigendecomposition/KKT/rank/replay path is CPU-only — proposal §519).
- **budget**: 64 CPU-h, 0 GPU-h, 0 API, 10 GB.
- **gate** (line 683): GO iff Run1 frozen, Run2-v1/v2/v3 failure evidence preserved, Run2-v4 semantic
  verification PASS in its single SLURM artifact, and a fresh independent v4 artifact review
  authorizes unlock. **All satisfied** (v4 ACCEPTED).
- **failure_transition**: STOP before M1 cache.
- **access discipline (this is clear):** train banks + manifests only; **zero** val/test label reads;
  **zero** performance/accuracy/macro-F1 metric (no training, no kNN eval); only gold anywhere in the
  project is `parent_video_binary_label`; no segment/frame/timestamp/span/localization/stance/target/
  mechanism/rationale/fragment gold; the robust-coverage rule is **fail-open** — "low robust coverage
  disables only the robust safety claim and robust constraints; it does not fail global geometry"
  (plan §81, proposal §332).
- **acceptance semantics (partially clear):** `rank_tail <= d`, replay-hash parity (determinism),
  resource peak under the STOP cap (proposal §521/§530-537: hard STOP if measured/projected peak RSS
  > 96 GB, CPU > 16, GPU > 2), isolation injections must all be REJECTED, robust coverage reported
  (fail-open). These mirror the synth-kkt acceptance style but on real-sized inputs.

**Machinery already exists (from v4, `ARTIFACT_ACCEPTED`):** the projection / serialized H-metric
normal-cone KKT / rank-tail audit / factor-Procrustes / replay-hash / injection-defense code lives in
`scripts/analysis/lb_scgp_global_r2_run2_v4_{common,producer,independent_verify,validate}.py`
(+ schemas + wrapper + sbatch). Realbank is **not** a byte-clone of v4 — it feeds *real train-bank
sizes* through the same machinery and emits a **different** schema
(`scgp_global_realbank_resource_v1`, 5 resource-focused keys) rather than the full KKT payload.

---

## 2. Entity inventory — NONE exist

`find -iname "*realbank*"` over the repo returns **nothing**. No realbank config, schema, script,
wrapper, or sbatch exists. The machine plan binds **no** realbank implementation-entity hashes (the
only realbank references anywhere in the plan are the run entry, the run_order line, the schema-key
list, and the v4-amendment cross-notes — grep-confirmed). **So there is no missing-vs-bound hash
mismatch**; the blocker is upstream of hashing: the *scientific/interface spec itself* is undetermined.

By lineage convention the full entity set would be (names per the lead's `…_realbank_resource_v1`):
`configs/lb_scgp_global_r2/m0_realbank_resource_v1.json`,
`schemas/lb_scgp_global_r2/scgp_global_realbank_resource_v1.schema.json` (+ any case/input schema),
`scripts/analysis/lb_scgp_global_r2_realbank_resource_v1_{common,validate,producer,independent_verify}.py`,
`scripts/wrappers/lb_scgp_global_r2_realbank_resource_v1.sh`,
`scripts/slurm/lb_scgp_global_r2_m0_realbank_resource_v1.sbatch`. **None written** — see §3 for why.

---

## 3. BLOCKER — the load-bearing spec gaps (why I stopped instead of guessing)

The single most important input to the microbenchmark — **what N×d data builds `G0` and the
structural term** — is not determined by the frozen plan, and it is entangled with a run-order
paradox. The method (proposal §123-135, §189-299) defines:

- `Z0 ∈ R^{N×d}` = "**paired REMOVE/comparator train bank**, row-normalized"; `G0 = Z0 Z0ᵀ`;
  `d` read from `Z0.shape[1]`.
- structural term `r_struct = A_struct·vec(G) − b_struct`, where `b_struct` is derived from the
  **sealed MLLM cache** `K_C` (proposal §189-235).

**Run-order paradox (machine `run_order`, lines 289-297):** realbank is `run_order[3]`, but
- the **M1 MLLM cache** (`…M1-CACHE-*`) that produces `K_C`/`b_struct` is `run_order[4-6]` — **after**
  realbank; and
- the **M2 comparator freeze** (`…M2-COMPARATOR-FREEZE-v1`) that *defines which train bank is `Z0`*
  is `run_order[7]` — **after** realbank.

So at realbank time **neither the real `Z0` (comparator undecided) nor the real `b_struct` (cache
unbuilt) exists.** The plan text does not say how realbank resolves this. Concretely, an implementer
must decide — and the plan does not — the following, each of which materially changes N, d, m, `G0`,
the resource profile, and the scientific meaning of the rank/replay/coverage numbers:

**Gap A — `Z0` source.** Real frozen train banks *do* exist on disk
(`data/CLIP_Embedding/{MHC,MHC_zh}/train_openai_clip-vit-large-patch14-336_HF.pt` and Qwen2.5-VL-7B /
subclip / archive / transcript variants; train N = **549** MHC / **579** MHC_zh, confirmed from
`data/gt/*/train.jsonl`). But the plan does **not** preregister *which* `.pt` file is the realbank
`Z0`, nor whether realbank instead uses **synthetic-at-real-N** feature values (a pure compute-
feasibility probe using only real train *counts*/*ID manifests*). Both readings are internally
coherent and give different implementations and different scientific weight.

**Gap B — structural term with no cache.** Since M1 hasn't run, realbank must either (a) run a
**G0-only / structural-term-disabled** projection, (b) use a **synthetic/placeholder `b_struct`** at
the real `m = r(r+1)/2` dimension, or (c) something else. Unspecified. Note the Block-1 validation
sketch (proposal §637) lists `FULL_GLOBAL_TARGET_REAL_TRAIN → GLOBAL_TARGET_CERTIFIED or
CERTIFIED_NULL`, which *implies* a real-train full target — but that needs the cache, contradicting
the run order. This internal tension must be resolved by the plan owner, not the implementer.

**Gap C — schema sub-structure.** Only the **5 top-level keys** of
`scgp_global_realbank_resource_v1` are frozen; their internal fields, the concrete GO/STOP
**decision rule** written into `decision.json`, the `resource_peak` STOP threshold wiring (96 GB),
the per-dataset `rank_tail` fields (which depend on `d` from Gap A), the **replay** protocol (the
run-twice-and-hash-match determinism procedure is nowhere pinned), the `robust_coverage` fields, and
the `isolation_injection_results` case list/categories/count are all undetermined. There is **no
dedicated realbank protocol block** in the machine plan (grep-confirmed), unlike `comparator_freeze`
and `statistics_protocol` which have their own detailed blocks.

Deciding A/B/C is exactly "自作主张改 plan." Given the v1 death (interface-key mismatch) and the
v3 death (plan/index drift), shipping entities that embed these guesses would very likely die the
same way in review or at runtime. Hence: **stop and request an amendment.**

---

## 4. Secondary finding — stale dependency label in human-readable docs (non-blocking)

The **machine plan is current** (realbank `runs[3].dependencies = …-SYNTH-KKT-v4`, gate references
v4). But two human-readable cells still say realbank depends on **v2**:
- tracker row `| 4 |` (`EXPERIMENT_TRACKER.md:17`): "depends on Run2-v2 PASS plus fresh independent
  v2 artifact review, not v1" and status `LOCKED_UNTIL_V2_PASS`;
- plan §257: "depends on Run2-v2 PASS and fresh independent v2 artifact review, not Run2-v1".

The v4 amendment notes (plan §259-261, tracker §109) *do* correctly state realbank "stays at `[3]`"
and now depends on `…-SYNTH-KKT-v4`, so this is documentation drift left by the v4 cascade, not a
logic error — the gate is de-facto satisfied by v4 PASS. The forthcoming realbank amendment should
fold these two cells forward to v4 for consistency (and flip the status off `LOCKED_UNTIL_V2_PASS`).

---

## 5. Recommendation to the plan owner (options, NOT a decision I am authorized to make)

An amendment (authored by a separate role, then independently reviewed — the v2/v3/v4 pattern) should
pin A/B/C. My non-binding read of the *intent* ("resource microbenchmark", "must pass **measured**
resource preflight", proposal §723; the artifact schema is resource/replay/rank/coverage/isolation,
**not** a KKT payload):

- **A (recommend):** use a **preregistered real frozen train bank** as the realbank `Z0` stand-in —
  the obvious candidate is `train_openai_clip-vit-large-patch14-336_HF.pt` for each dataset (it is the
  established frozen-CLIP comparator family in this repo and is `Z0`-shaped, row-normalizable) —
  precisely so the O(N³) eigendecomposition + dense projection is measured at *real* `N,d`, not a
  synthetic proxy. Bind its sha256 in the config. (If the owner prefers a pure compute probe,
  synthetic-at-real-N is the alternative, but then rank/coverage numbers are not about the real bank.)
- **B (recommend):** run **structural-term-disabled** (G0-only projection: symmetry, unit-diag, PSD,
  off-diag box, coordinate/row/class trust, but `lambda_struct·||r_struct||²` term dropped / `b_struct`
  absent), because the sealed cache does not yet exist and realbank explicitly must not train or make
  a performance claim. Document that the *structural* term's real-train behavior is deferred to a
  post-M1 gate.
- **C:** define the 5-key sub-schema + the explicit `decision.json` GO/STOP rule (resource cap 96 GB,
  `rank_tail<=d`, replay-hash parity, injection all-REJECT, robust-coverage fail-open), reusing the
  v4 producer/verifier two-stage (`PRODUCED_PENDING_INDEPENDENT_VERIFY` → independent `PASS`) pattern.

Once A/B/C are ratified, the entity set in §2 can be implemented and frozen with: (a) a dependency-
availability audit of every third-party import (incl. any in-function imports — the v2 death was a
missing `jsonschema`); (b) static run_id/path/index alignment against `runs[3]` (the v3 death); and
(c) three-way interface-key alignment config↔schema↔producer↔verifier (the v1 death), plus a runtime
cross-check static-simulation table (the v4 precedent).

---

## Role separation & required statements

- The realbank-prep role (this recon) is separate from the independent amendment reviewer, the fresh
  0C/0H code reviewer, the execution authorizer, and the executor. This document authorizes no
  execution and makes no plan edit.
- No performance evidence exists and none is claimed. No entity was created; no artifact, no counters.
- The only project gold is `parent_video_binary_label`; no segment/frame/span/localization/stance/
  target/mechanism/rationale/fragment gold is assumed or introduced.
- Run3+/M1/MLLM-cache/validation-test/training remain locked; this recon does not unlock execution.
