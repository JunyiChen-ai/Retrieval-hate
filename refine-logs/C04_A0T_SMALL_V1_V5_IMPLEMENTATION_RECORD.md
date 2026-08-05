# C04-A0T-SMALL-v1 Implementation-v5 Record

Date: 2026-07-30  
Status: **PROSPECTIVE / FRESH RE-REVIEW REQUESTED / EXECUTION BLOCKED**  
Scientific tag: `C04-A0T-SMALL-v1`  
Implementation version: `v5_prospective`

## Scope and preservation

This new namespace/hash closure responds only to implementation-v4's
`REVISE (0C/1H/1I)`. V4's cross-CPU-job publication recovery was accepted, but
fresh review found that its recovery/completed branch did not uniformly enforce
the hard terminal cap/accounted-field equality, and its payload-review
attestation domain retained a stale v3 suffix.

The v1-v4 implementation/config/record snapshots and frozen V2/V3/V4 design
history were not modified. No Python program, dataset/model operation, video
decode, SLURM command, or GPU/compute job was run while preparing v5.

## HIGH closure: one strict terminal-ledger gate

V5 adds `strict_validate_terminal_ledger`, called unconditionally before either
publishing or verifying `resource_final_state.json`. Every initial terminal
commit, same-allocation retry, cross-CPU-job recovery, and completed-final
verification reaches that one function after a fresh exact `sacct` query for
the original GPU job.

The validator requires:

- exactly one stored job and exactly one stored/live GPU;
- original GPU job identity and exact preflight, payload review, historical GPU
  authorization, config-contract, reconciliation authorization, and reviewed
  pre-ledger lineage;
- terminal ledger/job/live states and exact stored/live `sacct` state equality;
- integer, non-boolean live/stored/job/aggregate values;
- exact equality among live elapsed GPU seconds, stored terminal GPU seconds,
  job accounted seconds, aggregate accounted seconds, and aggregate reconciled
  terminal seconds;
- every such seconds value inside `[0,7200]`, with hard cap exactly 7,200;
- reservation exactly zero and both job/ledger reconciliation flags exactly
  boolean false;
- a numeric first ledger-writer CPU Job ID distinct from the original GPU job.

The final-state schema independently caps terminal and both aggregate seconds at
7,200. The v4 operational recovery design is retained: later authorized
CPU-only jobs may recover or verify publication without matching the first
writer, publisher/recovery IDs remain recorded, and the wrapper's bounded
same-allocation retry remains conditional on terminal-ledger/final-missing.
No path authorizes a second GPU or any submit/chain/release/resubmit action.

## IMPORTANT closure: v5 payload attestation domain

The payload-review attestation now hashes the exact domain
`C04-PAYLOAD-REVIEW-GO-v5\n` followed by the reviewed payload SHA-256.
The v5 common module, payload schema, stage manifests, implementation closure,
and `v5_prospective` version checks bind this domain; no v3/v4 attestation
domain is accepted.

All prior strict frame-pack, source/model/config lineage, single-allocation,
terminal resource gate, and authorization constraints are preserved in the v5
namespace.

## Prospective file hashes

| File | SHA-256 |
|---|---|
| `scripts/analysis/c04_a0t_small_v1_v5_common.py` | `fca7583a33679c543f5f595cf977533502c716afb76a5fd08fc21e75eaa63cdb` |
| `scripts/analysis/c04_a0t_small_v1_v5_preflight.py` | `bc87632d453cebcfcfdc0e708bc44c99df57dc340ef6c82e7bc9583c40181d71` |
| `scripts/analysis/c04_a0t_small_v1_v5_gpu_ledger.py` | `accec35295cb4d54c5cbde78081b7c9353f748cdbbf611f96bd782d0147a6e95` |
| `scripts/analysis/c04_a0t_small_v1_v5_producer.py` | `af21d8bf4194b5b4df90a7b5a53c555fac10be3d015585cedbf435d560ed774c` |
| `scripts/wrappers/c04_a0t_small_v1_v5_preflight.sh` | `f824e197396feea7bbc370b0581a80fa4e7d42024ab6484cf3b62e30838be7d6` |
| `scripts/wrappers/c04_a0t_small_v1_v5.sh` | `f66f275be7b3c7cfad8c497164a7ba49118d21eb8ff8c170f84906f9b79ee71e` |
| `scripts/wrappers/c04_a0t_small_v1_v5_reconcile.sh` | `d69624b8e0ab19ad4c4c9177a01c791a633162558c986b27568b48828ee978f2` |
| `scripts/slurm/c04_a0t_small_v1_v5_preflight.sbatch` | `4f30fb3009a6f8e01c38b0fe73b146ed09981a855acc1ab52dd7b12b26893ded` |
| `scripts/slurm/c04_a0t_small_v1_v5.sbatch` | `54ab8430d0549b4de12a4867ccb188cc70a5edb0a6517db407800924d98ccac3` |
| `scripts/slurm/c04_a0t_small_v1_v5_reconcile.sbatch` | `42afc8a74b27132afcb60df8d6f9a65f7fb541db277413e0c89b560a1b5b830b` |
| `schemas/c04/c04_a0t_small_v1_v5_prompt_record.schema.json` | `59c1b086513e807295572ea2e1328091cf062cbf6d050fb9f917b78b221e59c6` |
| `schemas/c04/c04_a0t_small_v1_v5_canonical_record.schema.json` | `53c7386a7b611560d4b66fa15af4bcc67f208c417cad6c191c429ae34777afd0` |
| `schemas/c04/c04_a0t_small_v1_v5_stage_authorization.schema.json` | `9de93185e58abfd7530d10070c09c8bd92ab05149f6af7565954201125738fc1` |
| `schemas/c04/c04_a0t_small_v1_v5_payload_review.schema.json` | `7bdf8dd0dbd968df7d43cf048f066197692eff9b9de2652aeafd11bdd17cccd6` |
| `schemas/c04/c04_a0t_small_v1_v5_resource_final_state.schema.json` | `3a83f06fcd6f4fcaf5cebc86e2e0e37e7b01a25afb82c06a06811696051367d9` |
| `configs/c04/c04_a0t_small_v1_v5.json` | `78e2ade7e91c74446eeba0d2965bc4675a804717857a0a202caabe1f80440a1b` |

## Static validation and execution state

- `jq` parsing passed for the config and all five schemas.
- `bash -n` passed for all three wrappers and all three sbatch files.
- The implementation and frozen-design hash closures match.
- Static search found no time directive, array, dependency, `sbatch`, or
  `scontrol` operation.
- Frozen v4 config/record hashes remain
  `8939bb4b51fd03cd9a1584d782f8a8d0216924cbe6ad9b7ae087f9b212816edd`
  and `2f602828d0b82b2654410f84fdb90437f7c9ebbbe5069e4ac9891bca88a50dad`.

Runtime behavior is deliberately untested. All materialization, teacher/GPU,
Slurm execution, and reconciliation authorizations are false; every stage
verdict/pin remains pending; no v5 review or runtime artifact exists.
