# C04-A0T-SMALL-v1 Implementation-v4 Record

Date: 2026-07-30  
Status: **PROSPECTIVE / FRESH RE-REVIEW REQUESTED / EXECUTION BLOCKED**  
Scientific tag: `C04-A0T-SMALL-v1`  
Implementation version: `v4_prospective`

## Scope and preservation

This is a new namespace/hash closure responding only to the implementation-v3
verdict `REVISE (0C/1H/0I)`. V3 already closed strict frame-pack resume
validation; its only residual HIGH was that terminal-ledger commit followed by
final-state publication was not operationally recoverable through the fixed
entrypoint when the next CPU Slurm job had a different ID.

The v1, v2, and v3 implementation/config/record snapshots and frozen V2/V3/V4
design history were not modified. No Python program, dataset/model operation,
video decode, SLURM command, or GPU/compute job was run while preparing v4.

## Residual HIGH closure: operationally idempotent publication

V4 preserves the first CPU job that writes the terminal ledger as
`ledger_writer_slurm_job_id`, but no longer requires a later authorized CPU
reconciler's current Job ID to equal that writer.

Every initial, recovery, and completed-state verification call still must:

- be a numeric, CPU-only Slurm job with no GPU, array, or dependency;
- have only implementation and post-job reconciliation authorization true;
- revalidate historical code/payload/GPU GO manifests and the current
  reconciliation GO manifest;
- exactly bind the original GPU job, self-hashed claim, 7,200-second
  reservation, allocation marker, producer seal/provisional usage, reviewed
  pre-ledger hash, config contract, and implementation/design/source/model
  closures;
- validate the terminal ledger's self-hash, unique original GPU job, zero
  reservation, stored authorization/pre-ledger pins, aggregates, and cap;
- query the exact original GPU job through `sacct` again and require exact
  terminal state/elapsed-time equality with the stored ledger.

When `resource_final_state.json` is absent, the successful publisher is recorded
as `final_state_publisher_slurm_job_id`. If publication is recovering an
already-terminal ledger, that CPU job is also recorded as
`recovery_slurm_job_id`; otherwise the field is `NO_RECOVERY_JOB`. A completed
final state may be verified by any later equivalently authorized CPU-only
invocation without rewriting either the terminal ledger or final state.

The fixed wrapper also performs one bounded same-allocation retry only when the
first child failed, the terminal ledger is already present, and final state is
absent. This closes the ledger-write/final-publish interruption window without
submitting, chaining, releasing, resubmitting, or authorizing any GPU. A later
manual CPU-only invocation is independently idempotent.

All v3 frame-pack, single-GPU, 7,200-second cap, terminal resource gate, and
authorization constraints are retained unchanged in the v4 namespace.

## Prospective file hashes

| File | SHA-256 |
|---|---|
| `scripts/analysis/c04_a0t_small_v1_v4_common.py` | `819cbde68b4b77fa4f227d9647f17ca6974ab5ec7ae9408ec258d0b2843987aa` |
| `scripts/analysis/c04_a0t_small_v1_v4_preflight.py` | `a84221ac265c3904614ec39cc468f49b95ed1abdd1431b20fdedb59a3a17e393` |
| `scripts/analysis/c04_a0t_small_v1_v4_gpu_ledger.py` | `08c0f4b2feff8f86334463b6fcc26c375115702fd1dd66a901f892ae35a466ef` |
| `scripts/analysis/c04_a0t_small_v1_v4_producer.py` | `210a9dc34fc44a03ce04075e1b3ab7c69bc560248669cca2c56c847681e23a21` |
| `scripts/wrappers/c04_a0t_small_v1_v4_preflight.sh` | `4ca788d5b8a93eef67a3ff1d6bcce70d2a020e6caa6ddb5c0f8f0ad01a5d97f9` |
| `scripts/wrappers/c04_a0t_small_v1_v4.sh` | `6f5607c9599539c990d18a92eb44efb8f521463d7ef6cb591d68e8d140591da0` |
| `scripts/wrappers/c04_a0t_small_v1_v4_reconcile.sh` | `931610483cc136836dd05d196d53227994f0cafd83193ed92a4c99a67dab57d7` |
| `scripts/slurm/c04_a0t_small_v1_v4_preflight.sbatch` | `f0ca0f7cacbe52f1ca6d7e9c2a41bec1789b2cc528954a2434e6aa9e3851714a` |
| `scripts/slurm/c04_a0t_small_v1_v4.sbatch` | `db9571404bdf2ba515a04f9e3af831f3c6d998795caea6c9c36d36c4b335d210` |
| `scripts/slurm/c04_a0t_small_v1_v4_reconcile.sbatch` | `d6b3d802e1703804a4d68b7c5f9702edf89e8307fecb5467edb7272414d482ea` |
| `schemas/c04/c04_a0t_small_v1_v4_prompt_record.schema.json` | `c4336874236ceaaf6bfa8b7773025f554e88dceb10bd0222105839fb403dc956` |
| `schemas/c04/c04_a0t_small_v1_v4_canonical_record.schema.json` | `d1d85c967c8b652e12133ecad952346bb0bc54a276c941fa2af94497418166d1` |
| `schemas/c04/c04_a0t_small_v1_v4_stage_authorization.schema.json` | `0fcf9b83c86676b02396376085c895e403d1abbe264ba5e46ad54522938d5ec4` |
| `schemas/c04/c04_a0t_small_v1_v4_payload_review.schema.json` | `ced2e3a3aae094daab6ba6613e82f6370e27c96ebb9de7344e47f812d67f5162` |
| `schemas/c04/c04_a0t_small_v1_v4_resource_final_state.schema.json` | `a8c885384f5c873d34d7687532c5edf0f82950b711d26dd38f071ac6af7793fb` |
| `configs/c04/c04_a0t_small_v1_v4.json` | `8939bb4b51fd03cd9a1584d782f8a8d0216924cbe6ad9b7ae087f9b212816edd` |

## Static validation and execution state

- `jq` parsing passed for the config and all five schemas.
- `bash -n` passed for all three wrappers and all three sbatch files.
- The implementation and frozen-design hash closures match.
- Static search found no time directive, array, dependency, `sbatch`, or
  `scontrol` operation.
- The frozen v3 config/record hashes remain
  `a33b3c3c1bb6032f1485f5e9875e475effd6c78733ef1c6c7a86b7acb54d6e02`
  and `c2654f1333b578eaf1d261b07daa215584f7eed1662370ef8d3ec44c323d8d65`.

Runtime behavior is deliberately untested. All materialization, teacher/GPU,
Slurm execution, and reconciliation authorizations are false; every stage
verdict/pin remains pending; no v4 review or runtime artifact exists.
