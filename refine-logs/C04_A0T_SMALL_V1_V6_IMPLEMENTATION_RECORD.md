# C04-A0T-SMALL-v1 Implementation-v6 Record

Date: 2026-07-30  
Status: **PROSPECTIVE / FRESH RE-REVIEW REQUESTED / EXECUTION BLOCKED**  
Scientific tag: `C04-A0T-SMALL-v1`  
Implementation version: `v6_prospective`

## Scope and preservation

This new namespace/hash closure responds to exactly one defect: the
implementation-v5 CPU preflight could not pass its own static gate. Job `13805`
(`c04_a0t_small_v1_v5_preflight`, submitted `2026-07-30T08:48:50`, held
`JobHeldUser`, started and ended `2026-07-30T16:40:57`) terminated `FAILED`
`1:0` in `00:00:00` with 0 bytes of stdout and 1191 bytes of stderr:

```
File ".../c04_a0t_small_v1_v5_preflight.py", line 152, in verify_static_config
    assert_equal(prompt_hashes(), cfg["prompt_hashes"], "prompt hashes")
RuntimeError: prompt hashes: {'system': '1ffc0675...', ...}
                          != {'system': 'PENDING_CPU_PREFLIGHT_HASH_FREEZE', ...}
```

`configs/c04/c04_a0t_small_v1_v5.json:115-120` carried
`PENDING_CPU_PREFLIGHT_HASH_FREEZE` for all four prompt-hash keys, and
materializing exactly those four hashes is what the CPU preflight exists to do.
The v5 static gate demanded config↔computed equality *before* the freeze that
produces the values, so the freeze run could never start. The identical
equality is asserted by the v5 producer at line 177, so the defect lives in the
shared v5 contract rather than in one entrypoint.

This is an ordering/contract defect. **No metric, result, decision or
CONTINUE/KILL verdict was published by 13805, and none is published here.** The
v1-v5 implementation/config/record snapshots and the frozen V2/V3/V4 design
history were not modified; `artifacts/c04/` still does not exist.

## What v6 changes, and only that

All three wrappers, all three sbatch files and all five JSON schemas are
**byte-identical to their v5 predecessors modulo the `v5`→`v6` version-token
rename** (verified by normalized diff: 0 changed lines each). The scientific
contract — selection,
prompts, frame rule, transcript rule, reliability thresholds, fallback
semantics, role/JL maps, resource cap, watchdog, single-allocation rule,
terminal-ledger gate, payload attestation domain (now
`C04-PAYLOAD-REVIEW-GO-v6`) and every authorization flag — is unchanged in
meaning.

### 1. Stage-scoped prompt-hash contract (`..._v6_common.py`)

New `resolve_prompt_hashes(cfg, freeze_stage)` replaces the unconditional
equality assertion. It computes the four hashes and then, per key:

- value equals the computed hash → accepted as literal binding;
- value equals `PENDING_CPU_PREFLIGHT_HASH_FREEZE` → accepted **only** when
  `freeze_stage` is true **and** `authorization.preflight_materialization_authorized`
  is exactly `True`; otherwise `HALT_PROMPT_HASH_SENTINEL`;
- anything else → `HALT_PROMPT_HASH_CONTRACT`.

`require_exact_keys` restricts the relaxation to exactly
`{system, A, B, combined}`; a foreign or short key set halts. A mixture of
pending and frozen keys halts. The function never relaxes a *value*
comparison — it only decides whether the four keys are permitted to be absent
as literals yet. It returns the computed hashes plus a binding tag
(`SENTINEL_PENDING_CPU_PREFLIGHT_FREEZE` or `LITERAL_BOUND`).

### 2. Literal freeze artifact

`build_prompt_hash_freeze_payload` / `verify_prompt_hash_freeze_payload`
materialize and validate
`artifacts/c04/a0t_small_v1_impl_v6/freeze/prompt_hashes.json`. The payload
always carries the four literal hashes with `downstream_binding=LITERAL_BOUND`;
the sentinel appears only as the recorded *name* of the pre-freeze config
state, never as any key's value. A payload whose keys hold the sentinel is
rejected even when its own `payload_sha256` is internally consistent. The
artifact is staged inside the atomic namespace publication and appears in the
preflight manifest's `staged_output_hashes` and in a new
`prompt_hash_freeze` manifest block.

### 3. Downstream stages are literal-bound

- New `assert_literal_prompt_hashes(mapping, label)` is the single post-freeze
  guard: it rejects the sentinel and any non-computed value. It backs both
  `verify_preflight_manifest` (shared by every post-freeze stage) and the frozen
  artifact verifier.
- The producer calls `resolve_prompt_hashes(cfg, False)` and requires
  `LITERAL_BOUND`, so a downstream consumer reading a config that still holds
  the sentinel HALTs.
- New producer `verify_frozen_prompt_hashes` binds the frozen artifact, the
  preflight manifest and the config to one another before any teacher work.
- The producer's two former `cfg["prompt_hashes"]` reads (record lineage and
  per-form `prompt_sha256`) now read literal computed values, so no sentinel
  can ever reach a sealed record.
- `c04_a0t_small_v1_v6_gpu_ledger.py` gains
  `assert_literal_prompt_hash_binding`, called as the first substantive check in
  both `validate_gpu_environment` and `validate_cpu_reconciliation_environment`,
  i.e. before `claim()` consumes the single-use resource ticket or appends a job
  to the ledger. Without it an unfrozen config would pass the entire claim and
  only then be rejected by the producer: 13805's failure shape displaced one
  stage later, onto a GPU.

  Two boundaries on that claim, stated precisely rather than glossed:
  `scripts/wrappers/c04_a0t_small_v1_v6.sh` arms its `EXIT` trap and writes
  `allocation_entry_marker.json` *before* the first Python call, so the marker
  is written regardless of this gate, and `--mode mark-exit` runs on every
  wrapper exit and is the one ledger mode with no prompt-hash binding gate (it
  is an unconditional accounting trap and must stay reachable).

  A third boundary, operationally the load-bearing one: a claim-time HALT
  preserves the genesis ledger and leaves the resource ticket unconsumed, but
  the namespace is still **not re-runnable**, because that entry marker is
  pinned to `$SLURM_JOB_ID` and to `$C04_ALLOCATION_START_SECONDS` (re-read from
  `/proc/uptime` on every wrapper start, so even the same job cannot match it on
  a second attempt). A later allocation is then refused twice — by the wrapper's
  own `jq -e` guard and by `create_entry_marker` — and `ARTIFACT_ROOT` is a
  hardcoded constant, so recovery needs either manual removal of that marker
  inside the no-clobber namespace or a new implementation version. This is a v5
  property carried forward unchanged (the wrapper is byte-identical), it is
  fail-closed rather than unsafe, and it is out of scope for this repair — but
  it must be closed or explicitly accepted before the GPU stage is authorized.
  It does not touch the CPU preflight, which never creates an entry marker.

- Consequently `mark_exit` gains one early return: when the ledger holds no jobs
  **and** no resource-consumption record exists, it leaves the ledger
  byte-identical. In v5 it fell through and bumped the genesis ledger's revision
  and state even though the `for job in ledger["jobs"]` loop was a no-op, which
  permanently broke the resource ticket's `genesis_gpu_ledger_sha256` pin. Any
  claim-time HALT — including the new binding gate — therefore converted a clean
  pre-claim refusal into a wedged run inside a no-clobber namespace.

  The predicate is the consumption record rather than the job list because
  `claim` publishes `allocation_claim.json` and `resource_ticket_consumed.json`
  before appending the ledger job row: a death inside that window leaves the
  ticket consumed with an empty job list, and that exit must still be recorded.
  So the ledger is left untouched only when nothing was consumed; in every other
  case, including a successful claim followed by a crash, `mark_exit` behaves
  exactly as in v5. The entry marker is updated before the early return either
  way, so the wrapper exit is durably recorded in all cases. These two are the
  ledger's only changes from v5.

### 3b. The config contract must not move when the sentinel is filled in

`config_contract_sha256` now also normalizes `prompt_hashes` to
`<BOUND_BY_PROMPT_HASH_FREEZE_ARTIFACT_AND_IMPLEMENTATION_CLOSURE>`.

This is load-bearing, not cosmetic. The CPU preflight runs on the pre-freeze
config and bakes that contract hash into three immutable places — the
code/resource authorization manifest it verifies, the genesis GPU ledger, and
the resource ticket. Downstream stages require the *post-freeze* config, whose
`prompt_hashes` differ. Hashing that field verbatim would therefore make the
two states mutually unsatisfiable: leaving the config unfrozen burns the GPU
allocation before the producer's HALT, and amending it invalidates the pinned
authorization manifest permanently, since the freeze namespace is no-clobber.
Normalizing the field makes the post-freeze config amendment contract-neutral,
which is precisely what the later payload-hash review stage needs.

No audit coverage is lost: the literal values are pinned in the frozen
prompt-hash artifact and the preflight manifest, every stage re-derives them
through `resolve_prompt_hashes`/`assert_literal_prompt_hashes`, and they are
computed from constants in `..._v6_common.py`, whose SHA-256 is in
`implementation_hashes` and is re-verified by `verify_bound_file_map` at every
stage. A self-test fixture asserts the invariance directly, and the dry
validation additionally confirms the contract hash *does* move when a real
contract field (`small_cap_gpu_seconds`, `teacher_contract.num_frames`) is
tampered with.

### 4. Fail-closed self-test

`prompt_hash_contract_fixtures()` is prepended to `self_test_fixtures()`, so
the wrapper's `--mode self-test` and the freeze run's `run_self_tests` both
enforce thirteen new checks (20 fixtures total), and any `False` raises
`HALT_INVALID_FREEZE`:

| check | requirement |
|---|---|
| `prompt_hash_sentinel_accepted_on_authorized_freeze_run` | sentinel accepted on the freeze run |
| `prompt_hash_sentinel_rejected_without_materialization_authorization` | sentinel rejected when materialization is false |
| `prompt_hash_sentinel_rejected_on_non_freeze_path` | sentinel rejected downstream, with or without authorization |
| `prompt_hash_wrong_value_rejected_on_every_path` | a wrong hash halts on both paths |
| `prompt_hash_mixed_pending_and_frozen_rejected` | no partial freeze |
| `prompt_hash_foreign_key_set_rejected` | only those four keys |
| `prompt_hash_literal_config_accepted_on_both_paths` | `resolve_prompt_hashes` accepts a literal config on either path |
| `prompt_hash_frozen_payload_carries_literal_hashes` | frozen payload is literal and sentinel-free |
| `prompt_hash_sentinel_bearing_payload_rejected` | integrity-valid but sentinel-bearing payload halts |
| `prompt_hash_post_freeze_manifest_guard_accepts_literal` | the shared post-freeze guard accepts literal values |
| `prompt_hash_post_freeze_manifest_guard_rejects_sentinel` | the shared post-freeze guard halts on the sentinel |
| `prompt_hash_post_freeze_manifest_guard_rejects_wrong_value` | the shared post-freeze guard halts on a wrong value or foreign key set |
| `config_contract_invariant_across_prompt_hash_freeze` | filling the sentinel in does not move the config contract, while tampering does |

`_raises_runtime_error` returns `False` for a non-`RuntimeError` escape, so a
fixture cannot pass by raising the wrong exception type.

### 5. Config

`prompt_hashes` still holds the four sentinels — that is now a legal,
documented pre-freeze state rather than a contradiction. A new
`prompt_hash_contract` block states the sentinel semantics, and
`paths.prompt_hash_freeze` names the literal artifact.

`maps.expected_hashes` retains its v5 wording and value and is deliberately
unchanged: no code reads it, and widening this repair to it would exceed the
defect being fixed. It is, however, the one remaining `PENDING_*` string in the
config that `config_contract_sha256` does **not** normalize, so it is now
documented in the config as immutable for the whole lifecycle: amending it
would move the contract hash and simultaneously invalidate the pinned
code/resource authorization manifest, genesis GPU ledger and resource ticket
inside a no-clobber namespace. Materialized map hashes are recorded in the
preflight manifest (`map_hashes`) and bound by the payload-hash review
manifest, which `verify_payload_review` already checks against
`preflight["map_hashes"]`; they are never written back into this field.

The new `prompt_hash_freeze` block in the preflight manifest is verified rather
than self-attested: `verify_frozen_prompt_hashes` checks all eight of its
fields — `path`, `sha256` against the file, `payload_sha256`,
`config_binding_at_freeze` against the freeze artifact, `keys`, and all three
booleans.

The review-pin sentinels (`payload_review_sha256`,
`gpu_execution_authorization_sha256`,
`resource_reconciliation_authorization_sha256`) are **not** affected by this
defect: `_verified_review_file` already rejects any non-64-hex pin with
`HALT_REVIEW_LINEAGE`, which is the correct fail-closed treatment for a stage
that is not yet authorized. Only `prompt_hashes` lacked the stage-aware
relaxation, because only `prompt_hashes` is produced by the stage that reads it.

## Prospective file hashes

| File | SHA-256 |
|---|---|
| `scripts/analysis/c04_a0t_small_v1_v6_common.py` | `81b10f586cfa5d619db459505ba2c8c43a89fc50e0c1e978e500fb4932633f68` |
| `scripts/analysis/c04_a0t_small_v1_v6_preflight.py` | `c86d439c2a8da82122e094aaa00bc6f3bb9db34591648f501f29ac5be8bb8ec0` |
| `scripts/analysis/c04_a0t_small_v1_v6_gpu_ledger.py` | `960cad30528c4bc0633cd103b33943c327ada13b12786c28d23552baadf2e695` |
| `scripts/analysis/c04_a0t_small_v1_v6_producer.py` | `ae02336047d60b8e8ae1f122859598b8b76d78093d07af4a6ba4222f34cfc478` |
| `scripts/wrappers/c04_a0t_small_v1_v6_preflight.sh` | `22bb4a47c6c21b06bb61b994ea873da682c3332bc0a9dec29c96cc0d1429f770` |
| `scripts/wrappers/c04_a0t_small_v1_v6.sh` | `7c8915733ad39288bcbd5aea9358376b869f95a8fc7845528778c9f716b8413c` |
| `scripts/wrappers/c04_a0t_small_v1_v6_reconcile.sh` | `d41f93e4e4c59f451cc1421b52acfb6eefcb5104c6bd74a1fa277f9322fd8f80` |
| `scripts/slurm/c04_a0t_small_v1_v6_preflight.sbatch` | `4051c73eeacf14ca302174c0448b36051685020aa589d8445b048469f46d665f` |
| `scripts/slurm/c04_a0t_small_v1_v6.sbatch` | `d42f9191352509fe2e4bff3e41b8a3b65f1b0e6f42879202a605750fb024b0d7` |
| `scripts/slurm/c04_a0t_small_v1_v6_reconcile.sbatch` | `c5e47308b323b60e07658e2e40e696677cbb937a6f85c68a54f201ded9c435e5` |
| `schemas/c04/c04_a0t_small_v1_v6_prompt_record.schema.json` | `007327448d3173936d5c0368a9ac83980753f5100fc2dbd3cef6ba0902768ff9` |
| `schemas/c04/c04_a0t_small_v1_v6_canonical_record.schema.json` | `7c05774a2e794137c5794bf7869be930766dbc7dfa7116b9cd6a213c1684bf9c` |
| `schemas/c04/c04_a0t_small_v1_v6_stage_authorization.schema.json` | `00f341b140332df3f0b1be7c02d455f894ff7098c0e11c7802f3e76fb3b77e00` |
| `schemas/c04/c04_a0t_small_v1_v6_payload_review.schema.json` | `0557760d97527b23b3fa26ababaaba36fd0eb35961fbaa0c11c31c505a027990` |
| `schemas/c04/c04_a0t_small_v1_v6_resource_final_state.schema.json` | `1274d0ad85125f72add92230c3bb8f71b25de41daf3c7c3ebf044e3951525231` |
| `configs/c04/c04_a0t_small_v1_v6.json` (pre-authority) | `98f2ca603538a22635904c299fa8623352dc516d003da430dcaf642336bdbd94` |

## Static validation and execution state

- `python -m py_compile` passed for all four v6 programs.
- `bash -n` passed for all three wrappers and all three sbatch files.
- `jq` parsing passed for the config and all five schemas.
- The implementation and frozen-design hash closures match.
- Static search over the v6 wrappers and sbatch files found no `--time`, array,
  dependency, `sbatch`, `scontrol`, `squeue`, `scancel` or `srun` operation.
  The only `--gres` in the v6 set is in `c04_a0t_small_v1_v6.sbatch`, the
  still-unauthorized GPU producer entrypoint, exactly as in v5. The CPU
  preflight sbatch requests 8 CPU / 64 GB and no GPU.
- Frozen v5 config/record hashes remain
  `5228c08fefcd41c354202dfd7a750b46e1a5d4d66a8b3562b736c9d943f526a6`
  and `aa49585cc27853793b349868bb6996ab5f051a44cfa1ff39c2fee0e7e1a704a1`.

## Deviation from the v1-v5 preparation discipline, stated explicitly

v1-v5 were prepared with **no Python execution at all**, and that is precisely
how a program that could never pass its own first gate reached the SLURM queue
and burned an ~8-hour hold. v6's preparation therefore included a read-only
login-node validation, run from a scratchpad directory outside the repository:

- imported the four frozen v6 modules and executed only pure contract logic —
  `self_test_fixtures()`, `prompt_hash_contract_fixtures()`,
  `resolve_prompt_hashes`, `assert_literal_prompt_hashes`,
  `build_prompt_hash_freeze_payload`, `verify_prompt_hash_freeze_payload`,
  `config_contract_sha256` and `assert_literal_prompt_hash_binding` — on
  synthetic dicts and on the real v6 config. All 20 self-test fixtures and all
  individually asserted contract cases passed;
- confirmed that the computed hashes are identical to the four values printed
  in `slurm/logs/c04_a0t_small_v1_v5_preflight_13805.err`, which proves the
  version rename did not perturb any prompt byte;
- exercised **both** config states explicitly, because they behave differently
  and the distinction matters:
  - the **pre-authority snapshot** frozen above — the exact bytes of
    `configs/c04/c04_a0t_small_v1_v6.json`, file SHA-256
    `98f2ca603538a22635904c299fa8623352dc516d003da430dcaf642336bdbd94`, with
    `preflight_materialization_authorized: false` — HALTs on *both* paths with
    `HALT_PROMPT_HASH_SENTINEL`. The sentinel relaxation is gated on the
    authorization flag, so an unauthorized config cannot freeze anything; the
    wrapper's `jq` guard independently refuses it with exit 2;
  - the **post-authority state** (the same config with materialization `true`,
    which is what the authority snapshot will ship) passes the freeze path with
    binding `SENTINEL_PENDING_CPU_PREFLIGHT_FREEZE` and still HALTs on the
    downstream path;
  - a simulated **post-freeze** config carrying the four literal hashes passes
    the downstream path with binding `LITERAL_BOUND`;
- confirmed the config contract hash is identical across the sentinel→literal
  transition and across the authority flip, and that it still moves when
  `resources.small_cap_gpu_seconds` or `teacher_contract.num_frames` is
  tampered with;
- confirmed the GPU-ledger binding gate rejects an unfrozen config, accepts a
  frozen one, is defined before `claim()`, and is present in both ledger
  validators.

No dataset label, video byte, model weight, teacher, GPU, SLURM command or
project artifact was involved; `preflight()` and the producer main path were
never called; `artifacts/c04/` remained absent throughout. Deterministic
`sha256` hashing of the frozen files and of the normalized config contract was
also performed on the login node, since the freeze cannot be produced any other
way.

One disclosed side effect: importing the four modules wrote `.pyc` files into
the pre-existing `scripts/analysis/__pycache__/`. They are in no hash map, are
invalidated by source mtime and size, are never read by the SLURM job, and
cannot influence the freeze.

## Fields that are documentation-only, stated so no reader over-trusts them

`maps.role_input_dim`, `maps.role_output_dim`, `maps.le3_shape`,
`maps.additive_shape` and `maps.scale` sit inside the config contract but are
read by no module. `preflight()` generates the role and dense-JL payloads purely
from `..._v6_common.py` constants (`TEACHER_DIM`, `ROLE_DIM`, `LE3_INPUT_DIM =
14 x 257 = 3598`, `ADDITIVE_INPUT_DIM = 4 x 256 = 1024`, literal `1.0/16.0`) and
never compares declared geometry against generated geometry. The declared values
currently agree with the constants — 3584, 256, `[256, 3598]`, `[256, 1024]`,
`0.0625` — and both are inside hash closures that every stage re-verifies, so
they cannot silently diverge without a hash mismatch. Binding them by assertion
is a real improvement, but it is not this repair, and it is deliberately left to
the payload-hash review stage rather than widened into an ordering-bug fix.
`maps.expected_hashes` is documentation-only in the same sense, with the extra
lifecycle-immutability constraint noted above.

## Reconstructing the reviewed revisions

`scripts/analysis/c04_a0t_small_v1_v6_preflight.py` was edited twice after the
five code/resource review rounds, so the reviewed bytes are not the frozen
bytes. Both deltas are exactly reconstructible, which is stronger than an
archived copy because any future reviewer can reproduce them:

Starting from the frozen file
(`c86d439c2a8da82122e094aaa00bc6f3bb9db34591648f501f29ac5be8bb8ec0`), applied on
a copy, in this order:

1. **Re-insert** the four import lines that were removed — `SCHEMA_VERSION,`
   immediately after `    RUN_ID,`; `exclusive_publish_bytes,` then
   `exclusive_publish_json,` immediately after `    dense_rademacher_payload,`;
   and `require_exact_keys,` immediately after `    model_hash_closure,`. The
   result is
   `7c64ddf624df8151863f24c9d2e947aea8fd0c232a4e8684aea08833e42056a9`, the
   revision that existed between the two post-review edits.
2. **Delete** the five-line `implementation_authorized` gate block from
   `verify_static_config`. The result is
   `8f7dcd44785126a82ba52fe2be4e3c61e4b6f771eb14a839bd016d13faf70111`, the
   revision reviewed in code/resource rounds 1-5.

Both steps were verified by reconstruction against the frozen bytes. The four
re-inserted names were never referenced in the file body, so their removal
changed no behavior; the gate block is the only behavioral post-review change.

**Erratum.** An earlier revision of this record stated the reconstruction as
"delete the gate block" alone. That was true of the file at the moment it was
written, but false once the imports were removed: applied to the frozen bytes it
yields `4d4dd033929560e923394cb704421cb50133bf1463e9027c0132cf79f75b5ebf`, a
revision that never existed. The independent unlock reviewer caught it. The
correction is recorded rather than silently overwritten, because a
reconstruction claim that a reviewer cannot reproduce is the same class of
unverified static assertion that put job 13805 in the queue.

The v6 tree is untracked in git, so the reviewed revision was not archived as a
file. That is a process defect recorded here rather than hidden: future
post-review in-place edits must retain the reviewed bytes.

`verify_static_config` also gains one assertion beyond the prompt-hash repair:
`implementation_authorized` must be exactly `True`. Every downstream stage
already required it, but no preflight gate read it, so a `false` value would
have let the CPU preflight create the no-clobber namespace and only then wedge
every later stage — the same "irreversible resource consumed before the
rejecting check" pattern this repair exists to remove. The flag is `true` in
the snapshot either way; the gate makes that a checked precondition rather than
an assumption.

Beyond that read-only validation, runtime behavior remains untested. Every
teacher/GPU/Slurm-execution/reconciliation authorization is false, every stage
verdict and pin except the design GO remains pending, and no v6 review or
runtime artifact exists.
