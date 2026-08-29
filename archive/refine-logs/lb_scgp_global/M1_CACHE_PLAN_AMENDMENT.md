# M1 CACHE Plan Amendment (resource fix + model pin + OCR omission)

Date: 2026-07-13

Author: **Claude Opus 4.8**, acting in the **m1-prep role only**. This role is separate from — and does
not perform the work of — the independent amendment reviewer, the fresh 0C/0H M1 code reviewer, the
execution authorizer, and the executor. This document authors the M1 cache plan amendment (pinning the
resource envelope, model, replica semantics, and OCR policy the frozen plan left undetermined for
`runs[4]` / `runs[5]` / `runs[6]`), edits the authoritative plan additively, runs the amendment-driven
hash cascade, implements the full M1 entity set, and freezes it (`M1_CACHE_FREEZE.md`). It does **not**
review, authorize, or execute it.

Discipline: static amendment + implementation + freeze only. No project GPU/MLLM/OCR/network/model run,
no training, no `sbatch`/`squeue`/SLURM submission, no experiment, no validation/test data or label
read, no cache produced. `py_compile` (pure syntax check, no code execution), `jq -e .`, `bash -n`,
`sha256sum`, `grep`, `find`, `git status` were read-only. No artifact under
`artifacts/lb_scgp_global/v1/m1/` was created (absence confirmed in the freeze doc). Nothing was
committed to git and no SLURM job was submitted.

This amendment is **not** performance evidence, **not** execution authorization, and **not** an
M2/comparator/validation-test/training unlock. It resolves the resource contradiction and the
underdetermined model/replica/OCR pins for the already-approved M1 cache run entries, pending fresh
independent review.

---

## 0. Why an amendment was required

The M1 gate is open (M0 fully closed: synth-KKT v4 and realbank v2 both `ARTIFACT_ACCEPTED`,
2026-07-13). The frozen plan pins *what* M1 does (train-only label-blind MLLM cache, `scgp_global_cert_v2`
schema, R=4 replicas, `4*N` calls, Merkle seal) but three items were internally inconsistent or
undetermined:

1. **Resource contradiction (blocking).** `runs[4]`/`runs[5]` requested `slurm.gpu=0` and
   `estimated_gpu_hours=0`, yet each carries `estimated_api_calls` = `4*N` (2196 for MHC, 2316 for
   MHC-ZH). The `zero_access_counters` pin `network_model_api_calls=0` — **no external/provider API is
   allowed**. The only legal way to realize 2196/2316 MLLM inference calls is **local GPU inference**.
   A 0-GPU allocation cannot serve them. This is a resource miscount, not a science change.
2. **Model undetermined.** The plan names the schema but not the MLLM weights.
3. **OCR ambiguity.** The schema text says "ASR/OCR text if available", but `ocr_calls=0` is a forbidden
   counter. These must be reconciled.

Deciding these is a plan-owner call. The coordination session ruled (below); this amendment records the
ruling and is itself subject to fresh independent review. The science lead retains override on the model
pin but must pin the model before cache seal (its `model_processor_hash` is bound into `cache_seal_v1`).

## 1. Decision — resource-contradiction fix (①)

- `runs[4]` / `runs[5]` `slurm.gpu` **0 → 1**; `budget.estimated_gpu_hours` **0 → 8** (est each).
- `estimated_api_calls` **numeric value is unchanged** (2196 / 2316). Only its **semantics** are
  annotated (`budget.api_calls_semantics`): it counts **local `Qwen2.5-VL-7B` `generate()` calls** on
  the allocated GPU under `HF_HUB_OFFLINE=1`; `external_network_model_api_calls` stays **0**. Formula
  `4 * N_train` (R=4 replicas × unique evidence packs): MHC `4*549=2196`, MHC-ZH `4*579=2316`.
- `runs[6]` (seal) **unchanged CPU-only** (`gpu=0`, `estimated_api_calls=0`).
- `by_milestone_must.M1.gpu_hours` **0 → 16** (8 MHC + 8 MHC-ZH; seal CPU).
- All **run-row-summing** GPU-hour aggregates **684 → 700** (the +16 was omitted everywhere): in
  `original_approved_r2_envelope_before_v2`, `paper_plan_substitution_envelope`,
  `remaining_prospective_budget`, `lifetime_lineage_envelope`, `matrix_estimated_must_run`,
  `matrix_estimated_total_with_nice` (both `must_gpu_hours` and `total_gpu_hours`). CPU-h, storage, and
  API figures are unchanged. Post-edit `sum(runs[*].budget.estimated_gpu_hours)=700` matches every
  aggregate (verified). Recorded as `budget_ranges.m1_local_mllm_gpu_correction`.
- `concurrency.parallel_groups.m1_cache_parallel_max2` gains `max_gpu_total=2` — two concurrent 1-GPU
  cache jobs, within the 2-GPU contract cap (`immutable_contract.slurm.max_gpu=2`).

**Erratum framing.** This is the same numeric-provenance discipline as the earlier ERRATUM commits: a
genuine miscount (M1 local-MLLM GPU-hours = 0 is impossible for 4512 local inference calls) is corrected
in *every* view that summed the run rows, with an explicit note. It is not a scope change; the run count
(65 MUST + 1 NICE), claims, gates, datasets, seeds, and forbidden routes are untouched.

## 2. Decision — model pin (②)

Pin **`Qwen/Qwen2.5-VL-7B-Instruct`** (offline weights present at
`/data/jehc223/home/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct`; `HF_HUB_OFFLINE=1`).
Rationale (coordination-session, three reasons):

1. project default MLLM with complete offline weights;
2. the structural observables are **coarse-grained** `{supported,contradicted,unresolved}` states; the
   settled MLLM-method-role campaign (`CAMPAIGN_mllm_method_role.md`) already showed `7B→32B→72B` scale
   does **not** flip the decision variables at this granularity, so a larger model buys no decision
   change here;
3. compute economy versus a 32B/72B ladder.

Authority: **pending fresh review approval**; the **science lead retains override**, but the model
**must be pinned before seal** because `model_processor_hash` is bound into `cache_seal_v1`. Decoding is
greedy (`do_sample=false`, `temperature=0`, `num_beams=1`), fixed across all R=4 replicas.

**Replica semantics pin.** `FINAL_PROPOSAL.md` fixes "four deterministic calls per train video; fixed
decoding/model/processor/prompt/input/schema hashes". The plan did not otherwise define per-replica
variation, so this amendment **pins** it: replicas use **byte-identical evidence-pack input** and fixed
greedy decoding — **no** per-replica prompt/frame/temperature variation. The only admissible cross-replica
difference is hardware-level FP nondeterminism, which the M2 replica-stability gate `sigma_cache` bounds.

## 3. Decision — OCR omission (③)

The evidence pack omits **live OCR** (`ocr_calls=0` contract is honored — no OCR engine is invoked). The
full textual evidence is **ASR + title**:

- **title** = the gt `train.jsonl` `text` field (video title/description metadata), deterministically
  truncated;
- **ASR** = `data/ASR/<DS>/train_asrK4_whisper-large-v3.jsonl` chunk text, concatenated in timestamp
  order and deterministically truncated;
- **frames** = **16** uniform full-video frames (project M=16 discipline).

On-screen text reaches the MLLM only **through the 16 sampled frames** (the VLM reads captions/overlays
visually, as `generate_vision_summary.py` documents). So "ASR/OCR text if available" resolves to
**ASR-only** for this cache, with visual on-screen text carried by the frames rather than a text OCR
call. The evidence pack contains **no** label/split/seed/neighbor/prediction/margin/correctness signal;
only `id + title + asr + frame-sampling-spec` enter `evidence_pack_sha256`.

## 4. Plan edits (additive) + hash cascade

Machine edits (indexes `4`,`5`,`6`; array length unchanged; `run_order` unchanged; every other index
untouched): `runs[4]/[5]` `slurm.gpu 0→1`, `budget.estimated_gpu_hours 0→8`, added
`budget.api_calls_semantics` / `model_pin` / `evidence_pack_protocol` / `replica_protocol` /
`ocr_policy` / `m1_amendment`; `runs[6]` added `seal_protocol` + `m1_amendment`;
`budget_ranges` M1 and aggregate corrections + `m1_local_mllm_gpu_correction`;
`concurrency.m1_cache_parallel_max2.max_gpu_total=2`. Pre-amendment machine plan backed up at
`EXPERIMENT_PLAN.machine.json.pre_m1_amendment.bak` (`f4d54b78…`).

| file | before | after |
|---|---|---|
| `EXPERIMENT_PLAN.machine.json` | `f4d54b78…830fc9b` | `93fdd752…861d5ef6` |
| `EXPERIMENT_PLAN.md` | `a3325f9d…ecb3dd8c` | `ed38f38c…a6918916` |
| `EXPERIMENT_TRACKER.md` | `51367c17…cd407618` | `e19b7f26…e2c24f591` |
| `EXPERIMENT_PLAN_HASHES.sha256` | `a3325f9d/51367c17/f4d54b78` triple | `ed38f38c/e19b7f26/93fdd752` triple |

The new machine plan hash `93fdd752…` is bound going forward in the three new M1 configs' authoritative
inputs (recorded in `M1_CACHE_FREEZE.md`). Consumed/closed M0 configs that still bind the old machine
hash are historical provenance and are **not** retro-updated (realbank precedent).

## 5. Entities implemented (self-contained; no cross-lineage import)

The M1 code reuses byte-faithful copies of the accepted realbank/Run2-v4 pure serialization/hash/guard
helpers (so the exact verified plumbing is reused without a cross-lineage import) and adds M1-specific
orchestration: evidence-pack builder, GPU certificate producer (local `Qwen2.5-VL-7B`, R=4), CPU seal.
SHA256 of all 14 entities is recorded in `M1_CACHE_FREEZE.md` §1. Two schemas, one shared common module,
three stage scripts, three configs, two wrappers, three sbatch.

## 6. Three-burn lessons applied

- **realbank-v1 `$TMPDIR` landmine:** every wrapper temp file is written **in-repo** under
  `slurm/tmp/`; the seal handoff and any producer temp use explicit in-repo `dir=`. No `${TMPDIR:-/tmp}`.
- **v1 interface-key mismatch:** the strict cache/seal schemas' `required[]` are aligned three ways with
  the producer/seal record keys and the common `ZERO_COUNTER_KEYS`; the simulation table in the freeze
  doc walks the alignment.
- **v3 index/plan drift:** the code constants pin `runs[4]`/`runs[5]`/`runs[6]` and `run_order[4..6]`;
  the machine verifier asserts both. No numeric index literal points at the wrong run.
- **v2 missing in-function dependency:** every function-level import (`torch`, `transformers`,
  `jsonschema`, `numpy`, `decord`/`av` via the shared frame loader) is enumerated and dependency-checked
  inside the SLURM job before the model loads.

## Status flags

- `ready_for_review = true` — ready for the independent amendment review (ratify the resource fix + model
  pin + OCR omission + the additive `runs[4..6]` edits) and the fresh 0C/0H M1 code review (which must
  independently re-derive the freeze-doc simulation table, the three handoff tables, and the zero-gold
  self-attestation).
- `ready_for_execution = false` — execution remains unauthorized. Independent amendment review,
  dependency-availability evidence, fresh 0C/0H code review with the simulation table all-PASS,
  exact-hashes/no-clobber review, and separate execution authorization are all still required before any
  single executor submit.

## Role separation & required statements

- The m1-prep role (this document + `.machine.json` + `_HASHES.sha256` + the plan/hash edits + the
  14-entity implementation + `M1_CACHE_FREEZE.md`) is separate from the independent amendment-review,
  fresh code-review, execution-authorization, and executor roles. This document authorizes no execution.
- No performance evidence exists and none is claimed; none is possible from a static
  amendment/implementation. The M1 cache emits no accuracy/macro-F1 and does no training or kNN.
- The only project gold is `parent_video_binary_label`. No segment/frame/timestamp/span/localization/
  stance/target/mechanism/rationale/fragment gold is assumed or introduced. Train **labels are not
  opened** by the evidence-pack builder or the cache producer; they enter only after cache seal. The
  train-ID allowlist is split membership, not gold.
- M2 (comparator freeze), validation/test, and training remain locked. This amendment unlocks neither
  M1 execution nor anything downstream.
