# M1 CACHE Implementation Freeze

Date: 2026-07-13

Author: **Claude Opus 4.8**, **m1-prep role only** — separate from the independent amendment reviewer,
the fresh 0C/0H M1 code reviewer, the execution authorizer, and the executor. This document freezes the
14-entity M1 cache implementation and records the runtime cross-check static-simulation table, three
per-run handoff tables, the zero-gold self-attestation, and the dependency evidence as a pre-review
predemonstration. It authorizes no execution.

Discipline: no project GPU/MLLM/OCR/network/model run, no training, no `sbatch`/`squeue`/SLURM
submission, no experiment, no validation/test data or train-label read, no cache produced. `py_compile`
(pure syntax, no code execution), `jq -e .`, `bash -n`, `sha256sum`, `grep`, `find`, and a light
synthetic-input unit test of the parser/schema/builder in the HateVideo interpreter (no GPU, no dataset,
no label) were the only executions; each is a static verification, not a compute job. No artifact under
`artifacts/lb_scgp_global/v1/m1/` was created (absence confirmed below). Nothing was committed to git;
no SLURM job was submitted.

Spec and decisions are in `M1_CACHE_PLAN_AMENDMENT.md` (+`.machine.json` + `_HASHES.sha256`). In one
line: three stages — a deterministic **label-blind evidence-pack builder**, a **GPU certificate
producer** (local `Qwen2.5-VL-7B`, R=4 replicas, restricted `scgp_global_cert_v2`), and a **CPU seal**
— produce and seal the train-only cache; **no training, no kNN, no accuracy/macro-F1 claim; labels
enter only after the seal decision.**

---

## 1. Entities frozen (14) — SHA256

| # | entity | SHA256 |
|---|---|---|
| 1 | `schemas/lb_scgp_global_r2/scgp_global_cache_replica_v2.schema.json` | `4bfcfea2d4dd38fd8a8125fb803fbde0c5ec05fa12a96d13b908246a6f03f68d` |
| 2 | `schemas/lb_scgp_global_r2/scgp_global_cache_seal_v1.schema.json` | `f4605bb7bd26f730c75e20636841f243f0bf10080b6ea12a70363d9a42790ce1` |
| 3 | `scripts/analysis/lb_scgp_global_r2_m1_cache_v1_common.py` | `601d61e22d6154b2c7023dad5dd19c01ea47d218c7ebb00a4f4833d38970fc24` (FIX-2) |
| 4 | `scripts/analysis/lb_scgp_global_r2_m1_evidence_pack_v1.py` | `ca9d94ec38acbdf74897f42e490409879236f34dee3d81b95ea1fee6bdb0cdd8` |
| 5 | `scripts/analysis/lb_scgp_global_r2_m1_cache_producer_v1.py` | `c82b87d6c1c9d93f5376d5f4d8bbaa975e49814fc2254446cca131ae99d3afd7` |
| 6 | `scripts/analysis/lb_scgp_global_r2_m1_cache_seal_v1.py` | `399e7956429f46af9c0cf5891251e6c2cc487316af5b0087dace6bab9232016d` |
| 7 | `configs/lb_scgp_global_r2/m1_cache_mhc_v1.json` | `23c777de6002eecab2d3f343eef8c8369b0b7acab4b6c9551f8e2159b35ca878` (FIX-1 cascade) |
| 8 | `configs/lb_scgp_global_r2/m1_cache_mhc_zh_v1.json` | `57bf435db9c8ad1b2949e2cfe8d73a4594d394e8e4854202a74c1108171ef742` (FIX-1 cascade) |
| 9 | `configs/lb_scgp_global_r2/m1_cache_seal_v1.json` | `94147df7020ab182b5503f7dcb63af78d08e4c40f6e5e0507b368963f7c2b782` (FIX-1 cascade) |
| 10 | `scripts/wrappers/lb_scgp_global_r2_m1_cache_v1.sh` | `a364a7df678bc48f3a7a8de23f644d259b2e0c847361af4d3b55dc9833a66e29` |
| 11 | `scripts/wrappers/lb_scgp_global_r2_m1_cache_seal_v1.sh` | `81c1ccac1ef3bb2e5b1b368c0eabdb86a5b7dc517db5e739d969eb23b0fda002` |
| 12 | `scripts/slurm/lb_scgp_global_r2_m1_cache_mhc_v1.sbatch` | `4516223e9402f70d129ddbe5f4683d180e66e4177ba28b1ed4a92b57f304cd04` |
| 13 | `scripts/slurm/lb_scgp_global_r2_m1_cache_mhc_zh_v1.sbatch` | `b340efaa615578b843110f2fe0ea8ae5241600cd20815e181aec496269ad105e` |
| 14 | `scripts/slurm/lb_scgp_global_r2_m1_cache_seal_v1.sbatch` | `5a42c3e79d9741faf079a53e85fb6d49b42d5d77345fe4b1a173a3cf0921ed98` |

Amendment lineage (bound in the config `authoritative_inputs`): `M1_CACHE_PLAN_AMENDMENT.md`
`b9505a7f…`; `.machine.json` `c894b829…`; `_HASHES.sha256` `0615b7ee…` (unchanged by the fixes).
Plan cascade (pre-amendment → post-amendment → **post-fix, current**):
`EXPERIMENT_PLAN.machine.json` `f4d54b78…`→`93fdd752…`→**`7638ac78…`**;
`EXPERIMENT_PLAN.md` `a3325f9d…`→`ed38f38c…`→**`e5ec9bc4…`**;
`EXPERIMENT_TRACKER.md` `51367c17…`→`e19b7f26…`→**`f36e3dec…`**;
`EXPERIMENT_PLAN_HASHES.sha256` `5246f208…`→**`9de299fd…`**. The configs' `authoritative_inputs` bind
the post-fix machine/PLAN/TRACKER/HASHES quartet (verified). Pre-amendment plan backed up at
`EXPERIMENT_PLAN.machine.json.pre_m1_amendment.bak` (`f4d54b78…`). See §FIX for the two must-fix diffs.

Run1-frozen contract reused: `schemas/lb_scgp_global_r2/scgp_global_cert_v2.schema.json`
(`4d3f1663…`, unchanged since Run1; the producer cross-validates every replica's observables against it).

## 2. Dependencies (all HateVideo-provided; validator-confirmed at runtime)

Third-party (all present in `/data/jehc223/miniconda3/envs/HateVideo/lib/python3.11/site-packages`,
versions pinned this session): `torch` 2.6.0; `transformers` 4.49.0 (exports
`Qwen2_5_VLForConditionalGeneration` at `transformers/__init__.py:3337` and `AutoProcessor` at `:193`);
`numpy` 1.26.4; `jsonschema` 4.26.0; `decord` 0.6.0 **or** `av` 17.0.0 (the shared frame loader tries
decord first, PyAV fallback — both present). In-repo import: `src/utils/generate_subclip_embedding_HF.py`
`load_video_frames` (`:204`), the exact M=16 sampler used by the accepted `score_segments_mllm.py` /
`generate_vision_summary.py`. **`qwen_vl_utils` is NOT imported** — frames are passed to the processor
directly as `videos=[frames]` (a list of PIL images), mirroring the two accepted scripts. Standard
library only otherwise: `argparse`, `hashlib`, `json`, `os`, `sys`, `tempfile`, `importlib`,
`pathlib`, `typing`. The producer's `dependency_check()` fails closed (before the model loads) if any
hard dependency is missing.

Function-level imports (v2 in-function-dependency lesson): `torch` + `transformers.{AutoProcessor,
Qwen2_5_VLForConditionalGeneration}` + `utils.generate_subclip_embedding_HF.load_video_frames` inside
`producer.main()`; `torch` inside `evidence_pack` is not used (no torch there); `jsonschema.Draft7Validator`
inside `common.validate_against_schema`. All are enumerated in `dependency_check()` / the SLURM env.

## 3. Runtime cross-check static-simulation table

Every runtime assertion that reads on-disk state, statically evaluated against the frozen on-disk state.
Rows marked PASS were verified read-only this session; DEFERRED rows run inside the SLURM job
(login-node GPU/MLLM/SLURM execution is forbidden by project discipline) and fail closed.

| Row | Assert (site) | Reads | Static verdict | PASS? |
|---|---|---|---|---|
| 1 | wrapper `RUN_ID==EXPECTED`; `.run.run_id`/`.run.dataset`/`.run.artifact_path` via `jq` | config | all three configs' `run.*` fields match the wrapper/sbatch constants | **PASS** |
| 2 | `require_slurm_cache` 4 CPU / 32 GB / GPU via `CUDA_VISIBLE_DEVICES` non-empty (FIX-2, accepted idiom); `require_slurm_seal` 4/32/0 GPU | SLURM env | sbatch `--cpus-per-task=4 --mem=32G` + cache `--gres=gpu:a100:1`; seal no gres; no `--time`; exactly-one-GPU enforced by `--gres` + `m1_cache_parallel_max2` cap | **PASS (env at runtime; empirically confirmed by the smoke, M1_SMOKE_RECORD.md)** |
| 3 | `dependency_check` (torch/transformers/numpy/jsonschema/decord|av) | env | all present in HateVideo (§2) | **PASS (runtime; dirs+versions confirmed)** |
| 4 | `bash -n` wrappers + sbatch (this session) | 5 scripts | all clean | **PASS** |
| 5 | `py_compile` the 4 `.py` (this session, HateVideo python) | scripts | all compile | **PASS** |
| 6 | `verify_config` (producer/seal) — 9/8 authorization flags False + the allow flags True + model id | config | all 3 configs pass (verified this session) | **PASS** |
| 7 | `verify_machine_cache`/`verify_machine_seal` runs[4]/[5]/[6] | machine | run_order[i]/run_id/dataset/artifact/schema/slurm/deps/model_pin/frames/replicas/ocr all == config (verified this session) | **PASS** |
| 8 | JSON Schema strict (`schema_requires_no_additional_properties`) | 2 schemas | 0 non-strict nodes; both valid Draft-07 (verified) | **PASS** |
| 9 | replica record validates `scgp_global_cache_replica_v2` | schema | synthetic good/malformed records validate; malformed → canonical unresolved (verified) | **PASS (by construction + synthetic)** |
| 10 | reconstructed cert validates Run1-frozen `scgp_global_cert_v2` | schema | synthetic cert validates against `4d3f1663…` (verified) | **PASS** |
| 11 | seal decision validates `scgp_global_cache_seal_v1` | schema | synthetic GO decision validates (verified) | **PASS** |
| 12 | evidence-pack builder reads only id/title/transcript; deterministic sha; no label | gt/ASR | synthetic gt/ASR (with a label field) → title/id/asr read, label untouched, sha deterministic (verified) | **PASS** |
| 13 | Merkle root determinism (`merkle_root`) | recompute | sorted-leaf root is order-independent (verified); seal recomputes and compares to manifest root | **PASS (producer↔seal by construction)** |
| 14 | `call_count == 4 * unique_pack_count` (producer raise + seal check) | recompute | producer refuses to publish otherwise; seal re-checks; `4*U_D` = 2196 (MHC) / 2316 (MHC_zh) at full dedup | **PASS (expected at runtime)** |
| 15 | no-clobber (`producer`/`seal` + wrapper trap) | artifact dir | `artifacts/lb_scgp_global/v1/m1/` absent this session | **PASS** |
| 16 | model load `Qwen2.5-VL-7B` offline + R=4 greedy generate | weights | offline weights present; `HF_HUB_OFFLINE=1`; symbol exported | **DEFERRED-TO-RUNTIME** |
| 17 | frame decode `load_video_frames(video, 16)` | mp4 | decord/av present; unreadable video → text-only call (fail-open per pack) | **DEFERRED-TO-RUNTIME** |
| 18 | forbidden zero_counters all 0 (seal) | manifest | ledger opens only allowlisted train evidence; every forbidden path raises | **DEFERRED-TO-RUNTIME (fail-closed)** |

**Load-bearing insight (realbank precedent):** the amendment made `runs[4]/[5]/[6]` content
(`slurm 4/32/1|0`, `model_pin`, `evidence_pack_protocol`, `replica_protocol`, `ocr_policy`, dataset,
artifact, schema) match the code constants (`expected_cache_slurm_block`, `MODEL_ID`, `NUM_FRAMES`,
`REPLICAS`, `RUN_INDEX`) in lock-step; `verify_machine_*` asserts both the numeric index and the run_id,
so no index literal points at the wrong run. The fresh code review must independently re-derive this
table.

## 4. Per-run full-chain handoff tables (realbank 11-row methodology)

Every inter-process / on-disk file handoff for each run: writer(site) → write path → write-guard →
reader(site) → read-guard → verdict. In-repo = under `/data/jehc223/RGCL`. All writes use
`exclusive_publish_json[l]` (in-repo `dir=` mkstemp, O_EXCL lock, no `$TMPDIR`); all reads use
`canonical_root_path` (in-repo only). No `${TMPDIR:-/tmp}` anywhere (realbank-v1 landmine closed).

### 4a. `LBSCGP-GLOBAL-M1-CACHE-MHC-v1` (identical shape for MHC_zh, N=579)

| # | artifact | writer @ site | write path | write-guard | reader @ site | read-guard | verdict |
|---|---|---|---|---|---|---|---|
| 1 | config json | frozen (prep) | `configs/…/m1_cache_mhc_v1.json` | — | wrapper `jq`; producer `read_json` | jq(in-repo)/`canonical_root_path` | PASS |
| 2 | machine plan | frozen | `refine-logs/…/EXPERIMENT_PLAN.machine.json` | — | producer `verify_machine_cache` | `canonical_root_path`+`sha256_file` | PASS |
| 3 | replica schema | frozen | `schemas/…/scgp_global_cache_replica_v2.schema.json` | — | producer `validate_against_schema` | `read_json` in-repo | PASS |
| 4 | cert_v2 schema (Run1) | frozen `4d3f1663…` | `schemas/…/scgp_global_cert_v2.schema.json` | — | producer cross-validate | `read_json` in-repo | PASS |
| 5 | gt train.jsonl (title) | frozen data | `data/gt/MHC/train.jsonl` | — | builder `load_titles`/`load_train_ids` (id+text only) | ledger `open_evidence` (allowlist; label untouched) | PASS |
| 6 | ASR train.jsonl | frozen data | `data/ASR/MHC/train_asrK4_whisper-large-v3.jsonl` | — | builder `load_asr` (id+chunks only) | ledger `open_evidence` (allowlist; label untouched) | PASS |
| 7 | train videos (mp4) | frozen data | `data/video/MHC/All/<id>.mp4` | — | builder `sha256_file`; producer `load_video_frames` | ledger `note_video_read` (dir allowlist) | PASS (decode DEFERRED) |
| 8 | Qwen2.5-VL-7B weights | offline HF cache | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct` | — | producer `from_pretrained` | `HF_HUB_OFFLINE=1` | PASS (load DEFERRED) |
| 9 | cache.jsonl | producer `exclusive_publish_jsonl` | `artifacts/…/cache/MHC/cache.jsonl` | O_EXCL lock + in-repo mkstemp | seal `read_cache_records` | `canonical_root_path` | PASS (static) |
| 10 | cache_manifest.json | producer `exclusive_publish_json` | `artifacts/…/cache/MHC/cache_manifest.json` | O_EXCL + in-repo mkstemp | seal `read_json`; wrapper `jq` | `canonical_root_path` | PASS (static) |
| 11 | access_ledger.json | producer `exclusive_publish_json` | `artifacts/…/cache/MHC/access_ledger.json` | O_EXCL + in-repo mkstemp | audit | — | PASS (static) |

**Tally MHC (and MHC_zh): 11 rows, 0 FAIL, 0 out-of-repo, 0 ambient-env path.** No wrapper temp file
(the producer's atomic publish uses explicit in-repo `dir=`); `slurm/tmp/` is provisioned but unused
by the cache path.

### 4b. `LBSCGP-GLOBAL-M1-CACHE-SEAL-v1`

| # | artifact | writer @ site | write path | write-guard | reader @ site | read-guard | verdict |
|---|---|---|---|---|---|---|---|
| 1 | seal config json | frozen (prep) | `configs/…/m1_cache_seal_v1.json` | — | wrapper `jq`; seal `read_json` | jq/`canonical_root_path` | PASS |
| 2 | machine plan | frozen | `refine-logs/…/EXPERIMENT_PLAN.machine.json` | — | seal `verify_machine_seal` | `canonical_root_path`+`sha256_file` | PASS |
| 3 | replica schema | frozen | `schemas/…/scgp_global_cache_replica_v2.schema.json` | — | seal per-record `validate_against_schema` | `read_json` | PASS |
| 4 | seal schema | frozen | `schemas/…/scgp_global_cache_seal_v1.schema.json` | — | seal decision `validate_against_schema` | `read_json` | PASS |
| 5 | MHC cache.jsonl | producer output | `artifacts/…/cache/MHC/cache.jsonl` | — | seal `read_cache_records` | `canonical_root_path` | PASS (needs run4) |
| 6 | MHC cache_manifest.json | producer output | `artifacts/…/cache/MHC/cache_manifest.json` | — | seal `read_json` | `canonical_root_path` | PASS (needs run4) |
| 7 | MHC_zh cache.jsonl | producer output | `artifacts/…/cache/MHC_zh/cache.jsonl` | — | seal `read_cache_records` | `canonical_root_path` | PASS (needs run5) |
| 8 | MHC_zh cache_manifest.json | producer output | `artifacts/…/cache/MHC_zh/cache_manifest.json` | — | seal `read_json` | `canonical_root_path` | PASS (needs run5) |
| 9 | cache_seal_decision.json | seal `exclusive_publish_json` | `artifacts/…/m1/cache_seal_decision.json` | O_EXCL + in-repo mkstemp | wrapper `jq`; M2 compiler (post-seal) | `canonical_root_path` | PASS (static) |

**Tally seal: 9 rows, 0 FAIL, 0 out-of-repo.** The seal depends on both producer outputs (rows 5–8);
it is the gate that lets labels enter afterward. A STOP decision is published (auditable) then fails
closed before the compiler.

## 5. Zero-gold self-attestation

`grep` over the four M1 `.py` files:

- **0** gold-field ACCESS patterns: no `["label"]` / `.get("label")` / `["stance"]` / `["target"]` /
  `["segment…"]`, and no `["split"|"neighbor"|"query_labels"|"query_z"|"prediction"|"margin"]` access.
- Every literal `label` token is a docstring/comment disclaimer, the authorization **flag name**
  `"label_read_allowed"` (never read as a datum), the output boolean `"labels_enter_after…"`, or the
  unrelated error-message **parameter** `label_text` (in `assert_equal`/`validate_against_schema`).
- `seed` appears only as `torch.manual_seed(0)` (a determinism WRITE), the zero-counter key
  `"seed_read_count"` (=0), and comments — no gold-seed read.
- `val`/`test`/`held`/`dev_seen`/`test_seen` appear only in the ledger's `FORBIDDEN_TOKENS` list and a
  comment; the builder opens only `data/gt/<DS>/train.jsonl`, `data/ASR/<DS>/train_asrK4_*.jsonl`, and
  `data/video/<DS>/All/`. The evidence pack fields are `{video_id, dataset, title, asr_transcript,
  num_frames, frame_rule, video_relpath, video_sha256, evidence_pack_sha256}` — **no label/split/seed/
  neighbor**. Verified by a synthetic gt/ASR unit test (label field present in input, absent from pack).

The only project gold is `parent_video_binary_label`; it is not opened by any M1 stage. The
train-ID allowlist (`train_id_allowlist_sha256`) is split membership, not gold.

## 6. Residuals / flags for the fresh code review (non-blocking here)

- **R-1 (replica determinism).** R=4 replicas use byte-identical input + greedy decoding
  (`do_sample=False, num_beams=1`, `torch.manual_seed(0)`). With truly deterministic inference the four
  replicas coincide and the M2 `sigma_cache` gate is trivially satisfied; the replicas exist to bound
  residual FP nondeterminism. This is the amendment's pinned reading of FINAL_PROPOSAL's "four
  deterministic calls … fixed decoding/prompt/input". The reviewer should confirm this is the intended
  replica semantics (vs. seeded temperature diversity, which the plan text does not authorize).
- **R-2 (`evidence_pack_sha256` scope).** The pack hash includes `video_sha256` (mp4 bytes) so dedup is
  content-correct; this makes the builder read every train mp4 once for hashing (in addition to the
  producer's decode). The reviewer may prefer spec-only hashing to avoid the extra read — flagged, not
  blocking; content-addressing is the safer default for correct `U_D`.
- **R-3 (unreadable video / empty evidence).** If `load_video_frames` fails, the replica is a text-only
  call (title+ASR); if a video is missing it is counted (`missing_video_count`) and still produces a
  pack. Fully empty evidence still yields four calls that the strict parser will map to canonical
  unresolved. No fabrication; the reviewer should confirm the fail-open-to-unresolved policy matches the
  contract's "invalid → unresolved, not rescue".
- **R-4 (model pin authority).** `Qwen2.5-VL-7B` is the coordination pin; the science lead retains
  override but must pin before seal (its `model_processor_hash` binds into `cache_seal_v1`). If the
  model changes, `model_processor_hash()` changes and the seal records it.
- **R-5 (`transformers` 4.49.0 video processor).** The producer passes `videos=[frames]` (PIL list) with
  `images=None`, exactly as the accepted `score_segments_mllm.py` / `generate_vision_summary.py`. The
  reviewer should confirm the installed processor accepts this call at runtime (it does for those two
  accepted scripts on the same env).

## FIX — post-review must-fixes (M1_CACHE_CODE_REVIEW.md, AMENDMENT_RATIFIED_PENDING_TWO_FIXES)

The review found Critical=0, High=2. Both are narrow and fail-safe; both are now applied.

**FIX-1 (§3.1, HIGH) — historical-snapshot rollback.** The amendment had corrected *every* run-row
GPU-h aggregate 684→700, including the historical snapshot `original_approved_r2_envelope_before_v2`
(a `…_before_v2` point-in-time record). Ruling ①: historical snapshots keep their as-approved value +
an annotation; only forward views move to 700. Diff (machine JSON, documentation-only, gates no run):

```
budget_ranges.original_approved_r2_envelope_before_v2.must_gpu_hours   700 -> 684
budget_ranges.original_approved_r2_envelope_before_v2.total_gpu_hours  700 -> 684
+ .gpu_hours_note = "historical snapshot retained at 684 …; forward views corrected to 700 …"
budget_ranges.m1_local_mllm_gpu_correction.aggregates_corrected: removed the two
    original_approved_r2_envelope_before_v2.* entries
+ budget_ranges.m1_local_mllm_gpu_correction.historical_snapshot_intentionally_retained_at_684 {…}
```

Forward views (`paper_plan_substitution_envelope`, `remaining_prospective_budget`,
`lifetime_lineage_envelope`, `matrix_estimated_*`, `by_milestone_must.M1`) stay 700/16, matching
`sum(runs[*].estimated_gpu_hours)=700` (verified). `EXPERIMENT_PLAN.md` and `EXPERIMENT_TRACKER.md`
mirror this (original-R2 line → 684 + note; substitution/lifetime/remaining → 700).

**FIX-2 (§4.5, HIGH) — GPU guard brittleness.** `require_slurm_cache` string-compared the last-seen
`SLURM_*_GPUS` var to `"1"`; but `SLURM_JOB_GPUS`/`SLURM_STEP_GPUS` hold device *IDs* (e.g. `"3"`),
not counts, so a correct 1-GPU allocation whose device id ≠ `"1"` would false-fail and burn the single
submit. Replaced with the project's accepted idiom (assert `CUDA_VISIBLE_DEVICES` non-empty, matching
`sq_common.require_runtime(gpu=True)` / `lb_scgp_common.require_slurm(expected_gpu=True)`); the
exactly-one-GPU budget is enforced by the sbatch `--gres=gpu:a100:1` and the `m1_cache_parallel_max2`
cap. Diff (`common.py:require_slurm_cache`): removed the `SLURM_*_GPUS` last-seen-token loop and the
`seen != "1"` raise; added `if not os.environ.get("CUDA_VISIBLE_DEVICES"): raise`. Empirically confirmed
by the non-lineage smoke (`M1_SMOKE_RECORD.md`) on a real 1-GPU allocation.

**Post-fix hash cascade (before → after):**

| file | before (post-amendment) | after (post-fix) |
|---|---|---|
| `EXPERIMENT_PLAN.machine.json` | `93fdd752…` | `7638ac78…` |
| `EXPERIMENT_PLAN.md` | `ed38f38c…` | `e5ec9bc4…` |
| `EXPERIMENT_TRACKER.md` | `e19b7f26…` | `f36e3dec…` |
| `EXPERIMENT_PLAN_HASHES.sha256` | `5246f208…` | `9de299fd…` |
| `…m1_cache_v1_common.py` | `6d2834e9…` | `601d61e2…` |
| `configs/…/m1_cache_mhc_v1.json` | `1f5c0615…` | `23c777de…` |
| `configs/…/m1_cache_mhc_zh_v1.json` | `79506228…` | `57bf435d…` |
| `configs/…/m1_cache_seal_v1.json` | `9e2d487e…` | `94147df7…` |

The three amendment docs (`M1_CACHE_PLAN_AMENDMENT.{md,machine.json}` + `_HASHES.sha256`) are
unchanged by the fixes; their bindings remain valid. All post-fix `verify_config` / `verify_machine_*`
checks re-pass (verified this session). `py_compile` clean on all four modules post-fix.

## Status flags

- `ready_for_review = true` — ready for the independent amendment review (ratify the resource fix +
  model pin + OCR omission + the additive `runs[4..6]` edits) and the fresh 0C/0H M1 code review (which
  must independently re-derive §3, the §4 handoff tables, and §5, and re-adjudicate R-1…R-5).
- `ready_for_execution = false` — execution remains unauthorized. Independent amendment review,
  dependency-availability evidence, fresh 0C/0H code review with the §3 table all-PASS, exact-hashes/
  no-clobber review, and separate execution authorization are all still required before any single
  executor submit.

## Required statements

- No performance evidence exists and none is claimed; none is possible from a static amendment/
  implementation, and the M1 cache run emits no accuracy/macro-F1 and does no training or kNN.
- The only project gold is `parent_video_binary_label`; no segment/frame/timestamp/span/localization/
  stance/target/mechanism/rationale/fragment gold is assumed or introduced, and train **labels are not
  opened** by the evidence-pack builder, the cache producer, or the seal (they enter only after the seal
  decision). The train-ID allowlist is split membership, not gold.
- M2 (comparator freeze), validation/test, and training remain **locked**; this freeze unlocks neither
  M1 execution nor anything downstream.
- The m1-prep role is separate from the independent amendment-review, fresh code-review,
  execution-authorization, and executor roles. This document authorizes no execution.
