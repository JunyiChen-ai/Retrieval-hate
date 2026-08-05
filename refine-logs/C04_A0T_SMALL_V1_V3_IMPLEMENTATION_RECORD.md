# C04-A0T-SMALL-v1 Implementation-v3 Record

Date: 2026-07-30  
Status: **PROSPECTIVE / FRESH RE-REVIEW REQUESTED / EXECUTION BLOCKED**  
Scientific tag: `C04-A0T-SMALL-v1`  
Implementation version: `v3_prospective`

## Scope and preservation

This is a new namespace and hash closure responding only to the implementation-
v2 review verdict `REVISE (0C/2H/0I)`. It closes the two remaining findings:
terminal resource reconciliation and strict resume-time frame-pack validation.

The v1 implementation/config/record, v2 implementation/config/record, and
frozen V2/V3/V4 design history were not modified. No Python program, dataset or
model operation, video decode, SLURM command, or GPU/compute job was run while
preparing v3.

## H2 closure: reachable terminal resource reconciliation

V3 adds a separate, fixed CPU-only SLURM entrypoint and
`reconcile-terminal` ledger mode. The mode requires its own strict
`CPU_POST_JOB_RECONCILIATION` GO manifest. Its current authorization snapshot
permits only implementation plus post-job reconciliation; teacher, GPU, SLURM
GPU execution, small-tranche execution, submit/chain/release/resubmit, and all
other evidence permissions remain false.

Before reading `sacct`, the stage exactly binds and rechecks:

- the historical preflight, payload review, and original GPU execution GO;
- the original numeric GPU job, self-hashed allocation claim, 7,200-second
  reservation, allocation-entry marker, producer seal, and provisional GPU
  usage;
- the exact reviewer-pinned pre-reconciliation ledger file hash;
- all implementation/design/source/model/config hash closures.

The CPU stage queries only the exact original job. A nonterminal or ambiguous
`sacct` row halts. A terminal row atomically replaces the 7,200-second
reservation with terminal elapsed GPU seconds, sets reservation to zero, stores
the reconciliation authorization and original/reconciler job identities, and
refuses an actual total above the cap. It neither creates nor authorizes a
second GPU allocation.

Crash recovery is limited to the same CPU reconciliation allocation. If the
ledger was committed before the final-state file, replay requires the stored
authorization/pre-state pins and re-queries terminal `sacct` for exact state
and elapsed-time equality. A strict, self-hashed `resource_final_state.json`
then binds the terminal ledger hash and all prior lineage. The config and
producer seal both require this terminal resource state before any downstream
review.

The reconciliation sbatch file is CPU-only, has no `--time`, array, dependency,
submission chain, release, or resubmission. It is prospective and was not
submitted.

## H4 closure: one strict frame-pack validator on all resume paths

V3 uses one `strict_validate_frame_pack` function for newly created packs,
existing checkpoint resume, completed-seal replay, and post-publication
idempotency verification. For every selected video it requires:

- exactly eight manifest frame rows with indices `0..7` and filenames
  `00.png..07.png`;
- exactly nine directory entries: those eight PNGs plus `manifest.json`;
- a regular, nonsymlink pack directory, manifest, and each PNG, with resolved
  direct-parent containment and file identity checks;
- exact per-frame positive size and SHA-256 plus the manifest self-hash;
- exact run/dataset/video/source/config/model/prompt/allocation lineage;
- exact backend, total-frame count, requested-index vector, and eight decode-
  failure booleans between the checkpoint input and frame manifest.

Thus a partial A/resumed B path and a completed-seal/idempotent path cannot
accept a looser frame-pack reference than initial creation.

## Prospective file hashes

| File | SHA-256 |
|---|---|
| `scripts/analysis/c04_a0t_small_v1_v3_common.py` | `4c126506b9e09d5b43aec7daaf26b3aa52b8dda6b0f5075b0b41285b5e44b73c` |
| `scripts/analysis/c04_a0t_small_v1_v3_preflight.py` | `a72371a4986294a7b3520114e4cda6a946fe773f1f5acde409d4c71e6f8ff92c` |
| `scripts/analysis/c04_a0t_small_v1_v3_gpu_ledger.py` | `2e0ef81a3c06cfb48eccac5240f778cac04cb9f0068b2d5c9b1aa075dba0b91e` |
| `scripts/analysis/c04_a0t_small_v1_v3_producer.py` | `6d2196a8ee4b1debb152f5460ce96e5dfa874f2107836fa8cefe3b710c4f04aa` |
| `scripts/wrappers/c04_a0t_small_v1_v3_preflight.sh` | `f62646fb475eda5d77c27f84187f8c98220db4072c57d23b30194b692c6c83cc` |
| `scripts/wrappers/c04_a0t_small_v1_v3.sh` | `9eacbdb29ac9ae788c04f9ff24aaa775603f5d13803c0215f22d0da273102eea` |
| `scripts/wrappers/c04_a0t_small_v1_v3_reconcile.sh` | `33c7249fb3639b49e49a45b363de8904d584011787e7fb27bfdfcbce0027b161` |
| `scripts/slurm/c04_a0t_small_v1_v3_preflight.sbatch` | `7fd5df83878d3caf23ad3b998e9aae9093abc992eeae01384bfa7287db0bf00c` |
| `scripts/slurm/c04_a0t_small_v1_v3.sbatch` | `e14c61a4673f61f530871acaa0a0390104b855e657e1050c4f3c2dacd00565f5` |
| `scripts/slurm/c04_a0t_small_v1_v3_reconcile.sbatch` | `53ff40d621f1a24ae3f19de673ef6705827224d308e84bcb6f0d580a6671cef1` |
| `schemas/c04/c04_a0t_small_v1_v3_prompt_record.schema.json` | `6b92eaa13df39b01d1d517e106e4c0c2a3f6332c79ae170f7102b8af7b02a42f` |
| `schemas/c04/c04_a0t_small_v1_v3_canonical_record.schema.json` | `0480cef597132996b3a0065b117889d52037bcd77ba38e77b842050ed7212223` |
| `schemas/c04/c04_a0t_small_v1_v3_stage_authorization.schema.json` | `44378721267f259855440d822a1447965fdbd6f986f8a5cc65a2cc27bdefc51d` |
| `schemas/c04/c04_a0t_small_v1_v3_payload_review.schema.json` | `a1ed2a06f927281b539ca0884c31cdb7c4cc0c204e8490fd9cced3e759f4e168` |
| `schemas/c04/c04_a0t_small_v1_v3_resource_final_state.schema.json` | `80933ae26a8dac7b892ffaa1265cc7d3d92107777698e4e23ec0a733a5a1b363` |
| `configs/c04/c04_a0t_small_v1_v3.json` | `a33b3c3c1bb6032f1485f5e9875e475effd6c78733ef1c6c7a86b7acb54d6e02` |

The config binds all fifteen implementation files above; its own hash is
recorded separately for review and is semantically bound by each stage's
config-contract manifest.

## Static validation and execution state

- `jq` parsing passed for the config and all five schemas.
- `bash -n` passed for all three wrappers and all three sbatch files.
- The complete implementation and frozen-design hash closures match.
- Static search found no time directive, array, dependency, `sbatch`, or
  `scontrol` operation.
- The v1 config/record hashes remain
  `985bd2a509f215fd93f7d6e7dda3ae75a85e04338c67946861cfcf4dd6275dda`
  and `77b233e5fdb023df97b29e6dc4cc3ec17bda5ae2c4d35421180e77dcdf27a370`.
- The v2 config/record hashes remain
  `125c75f4eec98039f6c6750aa79cdba72c02e8b6c6c01869b2fea4341fed85ea`
  and `35b702b6291b560bada02159f80bdd77436bc86b0b68b75c1869c3e332c37a69`.

Runtime behavior is deliberately untested. All materialization, teacher/GPU,
SLURM execution, and post-job reconciliation authorizations are false; all
stage verdicts/pins remain pending; no v3 authorization/payload artifact or v3
runtime artifact exists.
