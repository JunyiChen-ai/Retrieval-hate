# C04-A0T-SMALL-v1 Prospective Implementation Record

Date: 2026-07-30  
Status: **PROSPECTIVE / CODE-RESOURCE REVIEW REQUESTED / EXECUTION BLOCKED**  
Design authority: `C04_V4_DESIGN_REVIEW.md` = `GO 0C/0H/0I`

## Scope completed

This record covers a prospective small pre-gate implementation only. No Python
program was executed, no dataset artifact was materialized, no model was loaded,
and no SLURM job was submitted or released.

The package implements the bounded amendment for `C04-A0T-SMALL-v1`:

- exactly 200 frozen train IDs from HateMM and exactly 200 from MHC-ZH, using
  `sha256(utf8("C04-A0T-SMALL-v1" + dataset + video_id + "20260729"))`,
  ascending digest and then ascending UTF-8 `video_id` as the tie-break;
- label-blind selection and allowlist/source-manifest seal before any teacher
  access, with the ASR top-level `label` field syntactically skipped rather than
  decoded or materialized;
- one local, pinned, offline Qwen2.5-VL-7B-Instruct snapshot, two fixed prompt
  forms, eight frames, a capped transcript, zero retries, and no OCR/API/network,
  dev/test, cross-dataset, or title input;
- strict prompt and canonical schemas, reliability/fallback semantics, exact
  five semantic gates, role-map/LE3/additive payload generation, provenance,
  Merkle/hash seals, and fail-closed publication;
- checkpoint resume and single-allocation resource-ticket idempotency without
  automatic chaining, release, resubmission, job arrays, or dependencies.

## Prospective files and SHA-256

| File | SHA-256 |
|---|---|
| `scripts/analysis/c04_a0t_small_v1_common.py` | `2152f27fd380849bb52ad7bcf0887e7c4046b2ce9b9013a82a40a376bd0aa14d` |
| `scripts/analysis/c04_a0t_small_v1_preflight.py` | `f1d085d4a711f8113f751a31be565c0f4a60c12717bb26bd04640d9d81bfc7a4` |
| `scripts/analysis/c04_a0t_small_v1_producer.py` | `9c6fde203edc79c757993930b9c3d535b463c4078f363de9db263e4edafbde93` |
| `scripts/wrappers/c04_a0t_small_v1_preflight.sh` | `51cb9edf9c67de554af7953634359facc119e1a70743ea9e57c2521b98cc1d59` |
| `scripts/wrappers/c04_a0t_small_v1.sh` | `7577e9d018638b4461eaef4af252f832e14d3923443e9f0bacc3a61823365739` |
| `scripts/slurm/c04_a0t_small_v1_preflight.sbatch` | `22569b7482b3fc60474ca277f3c58fc17ac16efd6ea1e6aebce3adce510fba47` |
| `scripts/slurm/c04_a0t_small_v1.sbatch` | `ef35504e3fb20362c10c1c8e23631ce8b50e26c7451283a216fd29eb508e461d` |
| `schemas/c04/c04_a0t_small_v1_prompt_record.schema.json` | `ab79ee4da5e879fe60481f999ec7fc40180c10db2064645db2554efecd2cc960` |
| `schemas/c04/c04_a0t_small_v1_canonical_record.schema.json` | `e1f838e3f13431e290fed09590f67f995ea2d8da3bf14a8549e6c2ed3b71142e` |
| `configs/c04/c04_a0t_small_v1.json` | `985bd2a509f215fd93f7d6e7dda3ae75a85e04338c67946861cfcf4dd6275dda` |

The config binds the other nine implementation hashes, all frozen V2/V3/V4
design-history hashes, both train-ASR source hashes, the model snapshot revision,
each required model/processor file hash, and aggregate model/processor tree
hashes. This record is intentionally not self-bound by the config.

## Resource contract

- CPU preflight envelope: 8 CPUs, 64 GB RAM, no GPU.
- Teacher producer envelope: one A100 GPU, 8 CPUs, 64 GB RAM.
- HateMM then MHC-ZH run strictly serially inside one producer process.
- Total small-tranche cap: at most 7,200 GPU-seconds (2 GPU-hours).
- The GPU wrapper captures `/proc/uptime` at allocation entry, deducts startup
  overhead, leaves a 120-second internal reserve, and applies TERM then KILL after
  30 seconds. No `#SBATCH --time` directive is present.

## Static validation performed

- JSON parse/type validation passed for the config and both schemas using `jq`.
- Shell syntax validation passed for both wrappers and both sbatch files using
  `bash -n`.
- The config's implementation hashes and all frozen-design hashes were checked
  against the current files with `sha256sum`.
- Static searches found no `#SBATCH --time`, job array/dependency, `sbatch`, or
  `scontrol` execution in the prospective package.

These are non-compute checks only; runtime/model/data behavior remains untested.

## Hard execution blockers and required next reviews

The package is deliberately non-executable in its current reviewed state:

1. `preflight_materialization_authorized`, `teacher_authorized`,
   `gpu_authorized`, `slurm_authorized`, and
   `small_tranche_execution_authorized` are all `false`.
2. Fixed prompt hashes and deterministic role-map/LE3/additive payload hashes are
   still explicit `PENDING_*` sentinels.
3. `review.code_resource_verdict` is `PENDING`.
4. The required `payload_hash_review.json` does not exist.

After a code/resource GO and separate user authorization, a CPU-only SLURM
preflight may materialize the 200+200 allowlists, manifests, prompt hashes, maps,
and a zero-use resource ticket. Those exact artifacts require a fresh payload
hash review and explicit GO before any GPU authorization or submission. No
prospective file in this package performs chaining, release, or resubmission.
