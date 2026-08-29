# C04-A0T-SMALL-v1 Implementation-v2 Record

Date: 2026-07-30  
Status: **PROSPECTIVE / RE-REVIEW REQUESTED / EXECUTION BLOCKED**  
Scientific tag: `C04-A0T-SMALL-v1`  
Implementation version: `v2_prospective`

## Scope and preservation

This is a new implementation package responding to the first code/resource
review verdict `REVISE (2C/4H/1I)`. The v1 implementation, v1 config, frozen
V2/V3/V4 design history, and the v1 implementation record were not modified.

No Python program, model, video decoder, dataset materialization, GPU job, or
SLURM command was run while preparing v2.

## Review finding closure

### C1: dataset-root escape through video symlinks

V2 hard-codes both the lexical repository roots and the external physical train
roots:

- HateMM: `/data/jehc223/HateMM/video`
- MHC-ZH: `/data/jehc223/Multihateclip/Chinese/video`

Every video locator now requires a lexical dataset-root symlink, a resolved
target directly inside the correct physical train root, a regular-file
lexical/resolved `(device, inode)` identity match, and rejection of dev/test-like
path components. The preflight seals the resolved train-relative path and file
identity; the producer rechecks both before hashing or decoding.

### C2: review/config/hash lineage could be bypassed

The wrappers use fixed config, Python, script, run, and namespace paths; `CONFIG`
is not environment-overridable. V2 uses three strict, exact-hash review stages:

1. code/resource authorization for CPU preflight;
2. payload hash review after CPU preflight;
3. GPU execution authorization binding the exact reviewed payload.

A config-contract hash normalizes only stage authorization and exact review-pin
fields to avoid a circular config/manifest hash. Each strict authorization
manifest separately binds the exact authorization snapshot and review pins.
Together they bind the effective config, GO design review, all code/schema/
wrapper/sbatch hashes, source paths/hashes/sizes, model and processor files/tree
hashes, prompt hashes, preflight outputs, maps, and payload review. Preflight,
allocation claim, producer entry, ticket/claim consumption, checkpoint replay,
and final seal all reverify their applicable lineage.

The current prospective config intentionally contains `PENDING_*` review pins
and false execution authorization. No review manifest is present, so it cannot
pass any execution gate.

### H1: whole-form invalidation polluted slot rates

Only an undecodable JSON value or non-object top level invalidates all four
slots. For a recoverable object, S/P/T/H content and confidence are validated
independently. Missing or malformed fields invalidate only their dependent
slot; `form_valid` remains a diagnostic strict-form flag. A fixture explicitly
checks that an invalid T leaves S/P/H valid.

### H2: GPU ticket and ledger were not persistent/fail-closed

V2 adds a persistent ledger manager and allocation-entry marker. Before model or
data work it:

- records allocation entry;
- reconciles every prior job using exact `sacct` terminal or active-partial
  elapsed time;
- consumes one ticket and creates one allocation claim under a file lock;
- repairs only an interrupted same-job claim transaction;
- reserves the complete remaining 7,200-second cap before work;
- rejects another allocation/resubmission and any aggregate above the cap.

The wrapper installs an EXIT marker before the claim, and records provisional
elapsed time on every exit. A later reconciliation replaces provisional/
reserved time with terminal `sacct` time. The watchdog is measured from wrapper
entry, reserves 120 seconds, and uses TERM then KILL after 30 seconds.

### H3: fallback applicability was counted after materialization

For each dataset/slot, V2 computes all 200 reliability states first, freezes
state counts and applicability, and only then materializes features:

- one observed state: collapse/shuffle/noise =
  `NA_DEGENERATE_EXACT`; the collapse render and serialized slot feature are
  reused from FULL and asserted byte-identical;
- at least two states with any nonempty state count below 10:
  shuffle/noise = `NA_LOW_SUPPORT`, collapse remains `APPLICABLE`;
- otherwise all three controls are `APPLICABLE`.

STATE_ONLY and STATE_BLIND remain mandatory in every case.

### H4: checkpoint and completed-seal resume was under-validated

Each video receives a transactional, lossless eight-frame PNG pack before its
first prompt. A and B, including a partial-A/resumed-B case, must reference the
same frame-pack manifest and eight payload hashes; resume loads the pack instead
of re-decoding.

Checkpoint records are atomic per-record files. On every resume, v2 rechecks
schema, allowlist membership/order, raw-output hash, exact parser replay,
sequence/filename, transcript/video hashes, frame-pack bytes, prompt/model/
processor/preflight/payload/config/allocation provenance, and A/B frame
identity. A completed seal is accepted only after full prompt/canonical schema
and order checks, checkpoint equivalence, output hashes, Merkle roots,
reliability-derived terminal state, access/provisional lineage, claim/job, and
GPU-cap checks. The final seal directory is atomically renamed.

### I1: constant zeros were presented as runtime evidence

V2 removes hard-coded dev/test/OCR/API/cross-dataset zero counters. Train-ASR
and train-video accesses pass through dataset/root guards and create runtime
events. Label-field skipping counts come from the field projector. Absent OCR,
API/network, dev/test, cross-dataset, and SLURM-control entrypoints are labeled
explicitly as static surface assertions, not measured runtime counters.

## Prospective file hashes

| File | SHA-256 |
|---|---|
| `scripts/analysis/c04_a0t_small_v1_v2_common.py` | `408ed88827df55c37fd451185118de4cf3ce887ad937fff5f41e0ae0ea476f18` |
| `scripts/analysis/c04_a0t_small_v1_v2_preflight.py` | `7cdff69f0faabaf6a70571d2e30d76294e5639c94d780404dac591a1ad699693` |
| `scripts/analysis/c04_a0t_small_v1_v2_gpu_ledger.py` | `3025abea0cbb4624474be3c109e85ab750af316ff23749e2fe96659dd39c1f38` |
| `scripts/analysis/c04_a0t_small_v1_v2_producer.py` | `9d5104f2ff117139fdb8500ad78cba46dfdceeca9c4cc2d2765b3b215aa46695` |
| `scripts/wrappers/c04_a0t_small_v1_v2_preflight.sh` | `29c248c6ab30f7dd418ad9d2e6d6cbd7c1adb774ba88296d54df48ac717f5e1d` |
| `scripts/wrappers/c04_a0t_small_v1_v2.sh` | `bd8f975d32b668f88b0d8c6b4ab4fdff9b64b4625bbc24a7f22079cb1db72b8d` |
| `scripts/slurm/c04_a0t_small_v1_v2_preflight.sbatch` | `c40b863b25196516946d37a64db8785e137a723df778cb626a681570451ac784` |
| `scripts/slurm/c04_a0t_small_v1_v2.sbatch` | `cc18189b594217747e923e6c160fda6708b959721af09ee1899ab9bf61f4946d` |
| `schemas/c04/c04_a0t_small_v1_v2_prompt_record.schema.json` | `acef97fa1826151f62e1742a579cbfa974bb945f28a0c1c7568064394d267bb8` |
| `schemas/c04/c04_a0t_small_v1_v2_canonical_record.schema.json` | `b03f32e404184887b69ee3a26371972354ef2191413d7fb2398b6afbd491b100` |
| `schemas/c04/c04_a0t_small_v1_v2_stage_authorization.schema.json` | `8aff343f2b2c69948a21a0f6f6e926be9c9ae04e59d7bbeb2e400fa94de6725a` |
| `schemas/c04/c04_a0t_small_v1_v2_payload_review.schema.json` | `36959d7a4ac1a52042cc8a7f4d392449042b3e6f3788e8f0bc5cf63d90da293f` |
| `configs/c04/c04_a0t_small_v1_v2.json` | `125c75f4eec98039f6c6750aa79cdba72c02e8b6c6c01869b2fea4341fed85ea` |

The config binds the twelve implementation files. Its own SHA-256 is recorded
above and will be bound semantically by the staged config-contract review.

## Static validation and execution state

- JSON parsing passed for the config and four schemas.
- `bash -n` passed for both wrappers and both sbatch files.
- Implementation, frozen-design, and train-ASR hash/size closure passed.
- Static search found no time directive, array, dependency, `sbatch`, or
  `scontrol` execution.
- The frozen v1 config remains
  `985bd2a509f215fd93f7d6e7dda3ae75a85e04338c67946861cfcf4dd6275dda`;
  the frozen v1 implementation record remains
  `77b233e5fdb023df97b29e6dc4cc3ec17bda5ae2c4d35421180e77dcdf27a370`.

Runtime behavior is deliberately untested. Current authorization flags are
false, prompt/map payload hashes remain pending, all three v2 review verdicts
and exact pins remain pending, and no authorization/payload manifest exists.
