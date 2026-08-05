# C04-A0T-SMALL-v1 v7 — Fresh Independent Code/Resource Review, ROUND 5

Reviewer: fresh independent static reviewer (no exposure to the authoring reasoning)
Date: 2026-07-31
Stage reviewed: `CPU_PREFLIGHT` code/resource review of implementation-v7, fifth revision
Predecessors, all four left byte-intact:
- `refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW.md` (round 1, `REVISE 2C/2H/3I`)
- `refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_ROUND2.md` (round 2, `REVISE 0C/2H/3I`)
- `refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_ROUND3.md` (round 3, `REVISE 0C/1H/3I`)
- `refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_ROUND4.md` (round 4, `REVISE 0C/1H/0I`)

Execution authority conferred by this review: **none**.

Note on the deliverable name: the request names
`C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW.md`, which is the round-1 file. This
review is written to the round-5 path so all four earlier reviews survive
unaltered.

---

## Verdict

**GO — 0 Critical / 0 High / 0 Important (0C / 0H / 0I)**

The round-4 High is closed, and I closed it by recomputation rather than by
reading the repair claim: I drove the **real** reader
(`verify_resource_reconciliation_authorization`, with the manifest on disk, the
pin computed from its bytes, and `_verified_review_file`'s schema validation
intact) in both seal regimes against a sandbox root, and both now pass. I also
re-narrowed the schema back to its round-4 form in a mutation sandbox and
confirmed the new fixture turns red, so the repair carries a regression guard
rather than being a point fix.

Every finding from rounds 1, 2, 3 and 4 — eighteen in total — holds closed at
the **current** hashes. Three files moved since round 4
(`c04_a0t_small_v1_v7_common.py`, `…_stage_authorization.schema.json`,
`configs/c04/c04_a0t_small_v1_v7.json`); the other fourteen are byte-identical
to what round 4 reviewed. I re-derived the closures anyway rather than inherit
them, and I accounted for every line of the three that moved.

I found no new Critical, High or Important. What remains are fourteen
non-blocking observations, most of them inherited and explicitly carried by
earlier rounds; two are new and both are fixture-coverage gaps over code I
verified behaves correctly as frozen. None of them can cost a GPU allocation,
corrupt or wedge an artifact, or invalidate a scientific claim in the payload as
frozen, so none is rated above an observation.

**The payload is ready.** `authorization.preflight_materialization_authorized`
and `review.code_resource_verdict` are still `false` / `PENDING`, which is the
correct pre-authorization state; this review does not flip them.

---

## Method and reviewer-boundary compliance

- **No SLURM job was submitted, held, released, requeued or cancelled.** The
  only Slurm interaction was read-only `sacct` (three invocations) and one
  read-only `squeue`.
- **No GPU, teacher, model-weight or frame-decode work.** No `.safetensors` or
  any model file was opened, not even metadata; no video was decoded. The only
  video-adjacent operation was reading `id` / `window_text` / `language` out of
  the two train ASR JSONLs.
- **No file under `/data/jehc223/RGCL` was created, modified or deleted** other
  than this review file. Verified by re-hashing all seventeen frozen files
  before and after (17/17 identical), by re-fingerprinting the whole v6 artifact
  tree before and after, and by `find … -newermt` over `scripts/`, `schemas/`,
  `configs/` and `artifacts/c04/`, which returns nothing.
- **No dataset label value was materialized.** Both ASR files were read only
  through the frozen `project_train_asr_line` projector. Measured counters over
  the full files: `label_field_syntactically_skipped` = 744 (HateMM) / 579
  (MHC-ZH), `label_value_materialized` = **0 / 0**. HateMM identifiers were
  hashed, counted and compared, never printed and never reasoned from as labels.
- **All work in a scratchpad outside the repository**
  (`…/scratchpad/review-r5`), with `PYTHONDONTWRITEBYTECODE=1` on every
  invocation. Modules were imported only from scratchpad copies — byte-identical
  ones for read-only work, and for sandbox work copies whose sole edit is `ROOT`
  (verified by reverse-substituting the patch and diffing to zero against the
  frozen file). `python -m py_compile` was never used. `find … -name '*.pyc'
  -newermt` over the repository returns nothing.
- **`artifacts/c04/a0t_small_v1_impl_v7/` does not exist** and was not created.
- **`artifacts/c04/campaign/gpu_ledger.json` is byte-identical**
  (`fc6ca12c…`, start and end). Its append path *was* exercised — six spend
  magnitudes plus duplicate-append attempts — but only against copies inside a
  scratchpad sandbox root.

### Hash verification (all 17, before and after)

Every pinned SHA-256 in the request matched disk on first read and again after
all work. Truncated to 16 hex; full values compared.

| File | pinned / measured (start) | measured (end) |
|---|---|---|
| `…v7_common.py` | `5fc5259ec4a98b47` | match |
| `…v7_preflight.py` | `ecdc8568dfab0a50` | match |
| `…v7_gpu_ledger.py` | `944023b3aafc04df` | match |
| `…v7_producer.py` | `7a3c3a794454c585` | match |
| `…v7_preflight.sh` | `914dd5df80ab45d5` | match |
| `…v7.sh` | `645e501140690cec` | match |
| `…v7_reconcile.sh` | `7af043225285f129` | match |
| `…v7_preflight.sbatch` | `919316c70ae79d9f` | match |
| `…v7.sbatch` | `00ddeeed57d1f585` | match |
| `…v7_reconcile.sbatch` | `d8f634ec88d762be` | match |
| `…prompt_record.schema.json` | `541d02455aee3af9` | match |
| `…canonical_record.schema.json` | `bacbddaeba138068` | match |
| `…stage_authorization.schema.json` | `2edac849da8a3bf4` | match |
| `…payload_review.schema.json` | `7edebdfe81bb5180` | match |
| `…resource_final_state.schema.json` | `e2f9dca545874a4b` | match |
| `configs/c04/c04_a0t_small_v1_v7.json` | `0af5b6bdc12eb641` | match |
| `artifacts/c04/campaign/gpu_ledger.json` | `fc6ca12c32427625` | match |

All 15 `implementation_hashes` and all 15 `frozen_design_hashes` verify against
disk (15/15 and 15/15), and the `implementation_hashes` entry for
`stage_authorization.schema.json` is `2edac849…`, i.e. the round-5 schema edit
was propagated into the config as it had to be. The config is **not** listed
inside its own `implementation_hashes`.

### What moved since round 4

Round 4 recorded `…common.py = 2e4272c4…`, `…stage_authorization.schema.json =
b367eb03…`, `config = 3f436ea2…`. Those three are the only files whose hash
differs today. Every other file under review is byte-identical to the round-4
payload, so round 4's byte-level findings on them transfer — but I re-derived
them independently regardless (below).

### v6 predecessor unmodified

- `artifacts/c04/a0t_small_v1_impl_v6/freeze/preflight_manifest.json` is
  self-consistent (`payload_sha256` reproduces) and **all 14** of its
  `staged_output_hashes` verify byte-for-byte.
- All 15 entries of `configs/c04/c04_a0t_small_v1_v6.json →
  implementation_hashes` verify.
- Concatenated fingerprint over every file under
  `artifacts/c04/a0t_small_v1_impl_v6/`: `bf3f7a38701b1b84e310c2c7950d17e0226f…`,
  identical at start and end of this session and identical to the value round 4
  recorded.

---

## Round-4 H-1 — CLOSED, verified end to end and guarded against recurrence

**The finding.** `stage_authorization.schema.json` pinned the reconciliation
`payload_binding.provisional_gpu_usage_sha256` to `^[0-9a-f]{64}$` while the
code and the final-state schema had moved to a `NO_SEAL_PUBLISHED` sentinel, so
the seal-free reconciliation path was unsatisfiable: no terminal resource state
could be published on a breach, watchdog kill, OOM or post-claim HALT.

**1. The two schemas that describe this field now agree — byte-checked.**
Structural JSON comparison of the v7 stage-authorization schema against its v6
predecessor, version-token-normalized, shows **exactly one** difference in the
entire file:

```
REMOVED /properties/payload_binding/oneOf[2]/properties/provisional_gpu_usage_sha256/$ref
        = "#/definitions/sha256"
ADDED   /properties/payload_binding/oneOf[2]/properties/provisional_gpu_usage_sha256/anyOf
        = [{"$ref": "#/definitions/sha256"}, {"const": "NO_SEAL_PUBLISHED"}]
```

which is character-for-character the same construct already present at
`resource_final_state.schema.json:88-97`. Nothing else in the schema moved.

**2. Both regimes are satisfiable end to end against the real reader.** I built
a sandbox root containing the 15 implementation files, the 15 frozen design
files and the config at their true relative paths, wrote a real reconciliation
authorization manifest to
`refine-logs/C04_A0T_SMALL_V1_V7_RESOURCE_RECONCILIATION_AUTHORIZATION.json`,
set the config pin to that file's SHA-256, and called the **unstubbed**
`verify_resource_reconciliation_authorization` — so the manifest went through
`_verified_review_file` (64-hex pin check → file-hash equality →
`validate_schema` against the frozen stage-authorization schema), then
`verify_closure_hash`, then the full `expected`-dict comparison including
`payload_binding`, then both `verify_bound_file_map` sweeps:

```
(a) sealed regime,   provisional = <64 hex>          -> PASS (pin 460760b0a75a…)
(b) seal-free regime, provisional = NO_SEAL_PUBLISHED -> PASS (pin 37c0cf66d718…)
(c) foreign sentinel "MAYBE"                          -> RuntimeError: schema failure ['payload_binding']
(d) seal-free, reconciliation verdict PENDING         -> RuntimeError: reconciliation verdict is not GO
```

Round 4's measured `(a) schema failure / (b) binding mismatch` — the empty
intersection — is gone. The `NO_SEAL_SENTINEL` the seal-free tail passes
(`gpu_ledger.py:456`) is now exactly what the schema admits.

**3. The widening did not loosen anything else.** `payload_binding` is a
`oneOf` of three variants; a value matching two of them would now fail. Variant
2 (`additionalProperties: false` over four other keys) and variant 3 (eight
keys) are disjoint, and variant 1 is a const string. I validated every manifest
shape the code can demand:

| manifest the code requires | result |
|---|---|
| `CPU_PREFLIGHT` (`payload_binding: "NO_PREFLIGHT_PAYLOAD_YET"`) | VALID |
| `GPU_TEACHER_PRELABEL_SEAL` (4-key payload binding) | VALID |
| `CPU_POST_JOB_RECONCILIATION`, **sealed** (64-hex) | VALID |
| `CPU_POST_JOB_RECONCILIATION`, **seal-free** (sentinel) | **VALID** (was SCHEMA FAILURE in round 4) |
| reconciliation binding with a foreign `"MAYBE"` | REJECTED |
| reconciliation binding missing a required key | REJECTED |
| reconciliation binding with an extra key | REJECTED |

**4. No third description of the field remains.** A repository-wide grep for
`provisional_gpu_usage_sha256` returns, inside the v7 set, exactly two schema
descriptions (stage-authorization and resource-final-state, now identical
`anyOf`s) and the code sites that carry the value. No other schema constrains
it; `cfg["schemas"]` has only five entries and the seal manifest has none. The
one residual duplication is the string constant itself, defined in both
`common.py:2097` and `gpu_ledger.py:58` rather than imported — recorded as
observation 1, not a finding, because the two agree today and the fixture binds
the common-side spelling to the schema.

**5. The repair is guarded, not point-fixed.** Two new fixtures (#35, #36 of
52) round-trip a full reconciliation manifest through `validate_schema` in both
regimes and require a foreign pin to fail. I re-narrowed the schema back to its
round-4 form in a mutation sandbox:

```
FROZEN                       -> 52 fixtures, failing: []
schema re-narrowed to 64-hex -> 52 fixtures, failing:
                                ['reconciliation_manifest_round_trips_in_both_seal_regimes']
```

The fixture is therefore non-vacuous and would have caught the round-4 defect.
Renaming `common.NO_SEAL_SENTINEL` one-sidedly also turns it red, so the
common-side constant is pinned to the schema's `const`.

**6. The seal-free path now publishes a real terminal state.** I built the
`resource_final_state` record exactly as `publish_or_verify_resource_final_state`
builds it and validated it against the frozen schema across the full range:

| terminal sacct seconds | sealed | seal-free |
|---|---|---|
| 0 / 6000 / 7200 / 7250 / 7800 | VALID | VALID |
| 7801 | REJECTED | REJECTED |

`TERMINAL_SECONDS_HARD_MAX = 7800` and all three schema maxima are 7800, so
there is no gap in which `strict_validate_terminal_ledger` passes and the schema
rejects. `seal_published` and `terminal_elapsed_exceeded_cap` are both in the
schema's 29 required keys.

---

## Rounds 1-3 findings — re-derived at the current hashes

| # | Finding | Status | How I re-derived it this round |
|---|---|---|---|
| R1 C-A | reconciler's exact-key set had outgrown the writer | **CLOSED by construction** | `gpu_ledger.py:27-56` imports `PROVISIONAL_USAGE_KEYS` and `BUDGET_GUARD_KEYS` from `common.py` and passes them to `require_exact_keys` at `:465,:471`; the writer validates against the same objects. Deleting one member of `PROVISIONAL_USAGE_KEYS` makes the whole preflight suite **raise** `provisional usage writer exact-key failure` — i.e. the CPU preflight fails first, before any GPU. |
| R1 C-B | `proposition_cosine` could exceed schema `maximum: 1` | **CLOSED, fixture real** | Deleting `max(-1.0, min(1.0, …))` from `common.cosine` turns `cosine_of_identical_vectors_is_within_the_schema_bound` red (measured). |
| R1 H-A | campaign write side reachable only after a seal | **CLOSED** | `campaign_record` keys on `resource/allocation_claim.json` with an `allocation_entry_marker.json` fallback and is `--mode campaign-record`, run **first** in the reconcile wrapper under `set -e`, independent of `seal/`. |
| R1 H-B | in-job guard had no margin over the wrapper `timeout` | **CLOSED** | Executed the frozen `BudgetGuard` (extracted by `ast`, `/proc/uptime` faked) over an 18-point grid: guard lead over the wrapper SIGTERM is `300 + c` and never below 300, independent of producer start time. Table below. `SLURM_JOB_START_TIME` appears in the v7 set only in one producer docstring and one config prose string. |
| R1 I-C1 | accumulator enforced 28800 s, not the binding 7200 s | **CLOSED** | `campaign_effective_cap` = `min(7200, 28800)` = 7200 on the frozen ledger; reserve 7200 accepted, **7201 refused**; a 100 s recorded spend refuses the next 7200 s reservation. |
| R1 I-C2 | ~90 s margin to the hard ceiling; breach unrecoverable | **CLOSED** | `watchdog_reserve_seconds = 300` (asserted `== 300` in the preflight and the GPU ledger; `120` appears nowhere). Worst-case sacct elapsed ≈ `P0 + 6900 + 30 + mark_exit`, so ≈265 s of headroom; and an over-cap terminal elapsed is now recorded and flagged, publishable to 7800 s. |
| R1 I-C3 | preflight never round-trips a record against a downstream contract | **CLOSED for the record types the GPU stage writes** | Full `prompt_record` (normal + zero-frame) and full `canonical_record` (four reliability regimes) are built as the producer builds them and validated against the frozen schemas; `NUM_FRAMES 8→7` turns both red. |
| R2 H-1 | accumulator could brick itself on an over-cap write | **CLOSED** | Six append magnitudes against sandbox copies: no append raised, every post-append load succeeded, over-cap rows carry `aggregate_exceeds_effective_cap: true`, every one refuses the next 7200 s reservation, duplicate append halts. Table below. |
| R2 H-2 | reader restated the writer's key set | **CLOSED** | Same import proof as R1 C-A; a `require_exact_keys` census across the four modules shows both sides naming the same `frozenset` objects for the provisional-usage, budget-guard, GPU-ledger, resource-ticket and allocation-claim contracts. |
| R2 I-1 | `cosine` unreachable from the preflight | **CLOSED** | `cosine` is now in `common.py` (AST diff confirms it moved out of `producer.py`) and clamp deletion turns a fixture red. |
| R2 I-2 | post-loop canonicalization + seal phase unguarded | **CLOSED** | `guard.require_remaining(600, …)` at `producer.py:1822`, inside the `try` whose `except BudgetDeadlineReached` publishes the accounting-only breach record and returns 40. Measured effective budget for that phase: `900 + c` seconds. |
| R2 I-3 | GPU-seconds burned before `claim()` recorded nowhere | **CLOSED** | The wrapper writes the entry marker before its first Python call; `campaign_record` falls back to it; with neither artifact present it prints `no allocation entry; nothing to record` and returns 0. |
| R3 H-1 | GPU wrapper `mkdir`-ed the no-clobber namespace before any authorization gate | **CLOSED — reproduced end to end** | Byte-faithful wrapper replay, table below. |
| R3 I-1 | `watchdog_reserve` unmeasured; over-cap terminal elapsed unpublishable | **CLOSED** | See R1 I-C2 and the final-state matrix above. |
| R3 I-2 | `reconcile-terminal` seal-dependent | **CLOSED** | Relocated by round 4 into its H-1, which is now closed (section above). |
| R3 I-3 | duplicated writer/reader key sets; no full-record round-trip | **CLOSED** | Both halves re-derived above. |
| R4 I-1 | `watchdog_reserve` 120→300, over-cap terminal recorded | **CLOSED** | Arithmetic re-derived; 7800 code constant and 7800 schema maxima agree exactly. |
| R4 I-2 | terminal resource state on every terminal path | **CLOSED** | The final blocker was R4 H-1, closed above; both seal regimes now publish. |
| R4 I-3 | shared key-set constants + full-record fixtures | **CLOSED** | `ast` census plus the mutation table below. |

### R3 H-1 — the GPU wrapper gate, replayed

I replayed the frozen GPU wrapper against a sandbox root (only `cd` and
`PYTHON_BIN` repointed — verified by reverse-substitution diffing to zero), with
a stub `python`:

| run | config state | preflight manifest | exit | filesystem effect |
|---|---|---|---|---|
| 1 | the **frozen** config (`gpu_authorized: false`) | absent | 1 (jq gate) | **nothing — `artifacts/` never created** |
| 2 | fully GPU-authorized | absent | 2 (`HALT_REVIEW_LINEAGE: no frozen preflight manifest`) | **nothing** |
| 3 | fully GPU-authorized | present | proceeds past the gate | `…/freeze` + `…/resource` created, marker written |

Nothing irreversible precedes the gate: the statements before it are
`set -euo pipefail`, `cd`, `readonly` assignments, one `/proc/uptime` read, the
EXIT trap arming and two environment tests. The EXIT trap is armed earlier, so I
checked it specifically — `mark_exit` computes `root_path(...)` (which never
creates a directory) and **returns immediately when the marker is absent**;
run 1 confirms empirically that the trap fired and nothing appeared. Every
`mkdir` in the four modules is at `common.py:760`, `gpu_ledger.py:134/632/1218`,
`producer.py:921/1116/1942/1963`, `preflight.py:519/533`; none is reachable
before its stage's gate. `preflight.preflight()` tests `namespace.exists()` as
its first statement, and `claim()`'s first statement is `validate_gpu_environment`
(authorization flags + campaign headroom) **before** `create_entry_marker`.

### R1 H-B / R3 I-1 — guard arithmetic, executed

`cap=7200, reserve=300, ticket watchdog=6900, item margin=300, seal reserve=600`.

| claim `c` | producer start `e` | guard fires @entry+ | wrapper SIGTERM @entry+ | lead | latest seal start | seal budget |
|---|---|---|---|---|---|---|
| 0 | 0 / 30 / 120 | 6600 | 6900 | **300** | entry+6000 | **900** |
| 5 | 5 / 35 / 125 | 6595 | 6900 | **305** | entry+5995 | **905** |
| 30 | 30 / 60 / 150 | 6570 | 6900 | **330** | entry+5970 | **930** |
| 60 | 60 / 90 / 180 | 6540 | 6900 | **360** | entry+5940 | **960** |
| 120 | 120 / 150 / 240 | 6480 | 6900 | **420** | entry+5880 | **1020** |
| 300 | 300 / 330 / 420 | 6300 | 6900 | **600** | entry+5700 | **1200** |

The lead is `300 + c`, independent of producer start time, because the deadline
is anchored to the allocation-entry `/proc/uptime` reading. Every degenerate
case halts: entry in the future, margin 0 or ≥ watchdog, seal reserve 0 or
≥ watchdog, no budget remaining. `claim()` independently rejects a ticket whose
`watchdog != remaining − reserve`, and `verify_claimed_resource` rejects a
`C04_WATCHDOG_SECONDS` larger than the ticket's.

### R2 H-1 — campaign write side, executed on sandbox copies

| appended `gpu_seconds` | append raised | on-disk aggregate | over-cap flag | later `load` | next 7200 s reservation | duplicate append |
|---|---|---|---|---|---|---|
| 0 | none | 0 | `false` | OK | accepted | HALT |
| 100 | none | 100 | `false` | OK | REFUSED | HALT |
| 7199 | none | 7199 | `false` | OK | REFUSED | HALT |
| 7200 | none | 7200 | `false` | OK | REFUSED | HALT |
| **7250** | **none** | **7250** | **`true`** | **OK** | **REFUSED** | HALT |
| 30000 | none | 30000 | `true` | OK | REFUSED | HALT |

Every rejecting check (head race, duplicate job id, non-integer seconds) runs
before the write; the over-cap flag and both `campaign_effective_cap`
evaluations are computed before `os.replace`; nothing after the write can raise;
`load_campaign_gpu_ledger` has no cap check at all. `record_campaign_gpu_spend`
verifies an already-present row instead of appending, so `campaign-record` and
`reconcile-terminal` in one reconcile run cannot double-count.

---

## Claim 0 — no scientific semantic changed: **CONFIRMED**

**Selection re-derived from the v7 frozen rule reproduces the v6 frozen
allowlists exactly**, label-blind:

| dataset | train N | ids == v6 allowlist | digests == v6 allowlist | ranks 0..199 | transcript sha256 + scalar count == v6 source manifest (200/200) | sha256 of ordered id list |
|---|---|---|---|---|---|---|
| HateMM | 744 | **True** | **True** | True | **True** | `091fb1826cbc7f80…` |
| MHC_zh | 579 | **True** | **True** | True | **True** | `6c98c0d75891ce43…` |

Identical to the values rounds 3 and 4 measured. The transcript check
independently pins normalization, cap, head/tail split and separator as
unchanged.

**Prompt hashes recomputed from the v7 sources equal the v6 frozen artifact:**

```
system   1ffc06755b18f9315dc930aaf02a0c79cf1f9e2d91d0ec5bae2fa0c1436ed048
A        cecb3555daa7a4cc0840db154086138dae9260795bde2b64e89f6724d80abb9b
B        9521bee7fe7a6964d4ff61605bbe5a02eb4cfd149fd5aa9cbf975accbfe40314
combined a42268e4c0d1afb2b9576dd968d7f6250fc989fe3b97077a55363e63e1e2bb8a
```
equal to `artifacts/c04/a0t_small_v1_impl_v6/freeze/prompt_hashes.json`: **True**.

**Version-token-normalized tree diff, every residual line accounted for:**

| file | changed lines | accounted for by |
|---|---|---|
| `prompt_record`, `canonical_record`, `payload_review` schemas | **0** | — (so the canonical schema's `maximum: 1` is byte-identical to v6) |
| `stage_authorization.schema.json` | **7** | the round-5 repair **only** — structural JSON comparison shows exactly one `$ref` → `anyOf` substitution and nothing else |
| `resource_final_state.schema.json` | 128 | structural comparison: 3 maxima 7200→7800, 1 `anyOf`, 2 new booleans, `required` 27→29; the rest re-indentation |
| all three `.sbatch`, `*_preflight.sh` | 0 | — |
| `*_reconcile.sh` | 6 | R1 H-A only (`campaign-record` before `reconcile-terminal`) |
| `*_v7.sh` | 60 | R2 I-3 + R3 H-1 (the authorization gate and manifest test) |
| `preflight.py` | 42 | R2/R3/R4 I-3 |
| `gpu_ledger.py` | 320 | C-A / H-A / H-2 / I-1 / I-2 / I-3 |
| `common.py` | 952 | C-1 / I-1 / I-2 / I-3 + the round-5 fixtures, pure additions |
| `producer.py` | 473 | C-1 / I-1 / I-3 |
| `config.json` | 141 | structural comparison below |

**AST function-level diff makes this exact.** Nothing scientific moved:

- `common.py`: **20 added definitions, 0 removed, exactly one changed
  (`self_test_fixtures`)**; 20 added module constants, **0 removed, 0 changed**.
  So `SELECT_TAG`, `SELECT_SUFFIX`, `SELECT_N`, `NUM_FRAMES`, `TRANSCRIPT_CAP`
  / `HEAD` / `TAIL` / `SEPARATOR`, `MAX_NEW_TOKENS`, `RELIABLE_CONFIDENCE_MIN`,
  `PROPOSITION_COSINE_MIN`, `ROLE_DIM`, `TEACHER_DIM`, `Q_DIM`, all three map
  tags, `SYSTEM_PROMPT`, `_SCHEMA_TEXT`, `PROMPT_A`, `PROMPT_B` are provably
  untouched, and `render_slot`, `build_slot_reliability`, `materialize_role_map`,
  `dense_rademacher_payload`, `parse_teacher_response`, `q_product`,
  `safe_vector`, `merkle_root`, `normalize_transcript`, `normalize_proposition`,
  `selection_digest` appear in none of the added/removed/changed sets.
- `preflight.py`: 2 changed (`verify_static_config`, `preflight`), 0 added, 0 removed.
- `gpu_ledger.py`: 3 added (`campaign_record`, `record_campaign_gpu_spend`,
  `_reconciliation_lineage_tail`), 9 changed, 0 removed.
- `producer.py`: 10 added (the `BudgetGuard` family, the containment
  precondition, the breach publisher), **1 removed (`cosine`, moved into
  `common.py`)**, 6 changed.

**Config, structural comparison v6 → v7**, every difference accounted for:
`preflight_materialization_authorized true → false` (correct pre-review state),
`code_resource_verdict GO → PENDING` and its pin → sentinel, 15 refreshed
`implementation_hashes`, 2 added `paths` (`budget_breach`,
`campaign_gpu_ledger`), 4 added `resources` keys (campaign aggregate/phase caps,
guard item margin, guard seal reserve), `watchdog_reserve_seconds 120 → 300`,
and the `v7_scope` prose block. Nothing else.

**`config_contract_sha256` normalization, measured independently:** filling all
four prompt hashes, flipping every authorization flag, setting all four review
pins and setting all four verdicts to `GO` all leave it unmoved (the v5
impossibility stays closed); `watchdog_reserve_seconds`,
`guard_seal_reserve_seconds`, `maps.expected_hashes`, `selection.suffix`,
`reliability.proposition_agreement_cosine_min`, `teacher_contract.num_frames`,
an `implementation_hashes` entry, `schemas.stage_authorization` and
`review.downstream_review_requires_terminal_resource_state` all move it.

---

## Claim C-1 — the prompt renderer: **CONFIRMED**

1. **The v6 form could never have succeeded.** Replacing
   `template.replace(TRANSCRIPT_PLACEHOLDER, transcript)` with
   `template.format(transcript=transcript)` makes the whole fixture suite
   **raise** `KeyError: '"source_relation"'` — both templates embed
   `_SCHEMA_TEXT`, whose literal JSON braces `str.format` reads as replacement
   fields.
2. **The substitution is exactly the frozen one.** Fixtures
   `prompt_render_places_transcript_at_the_tail` and
   `prompt_render_keeps_the_literal_json_schema_braces` pass on the frozen bytes,
   and both prompt hashes reproduce.
3. **No prompt byte changed** — the four hashes above.
4. **No `.format(transcript=` call site survives.** Two textual hits: the
   `render_prompt` docstring and the deliberate regression fixture
   `prompt_render_regression_str_format_would_raise`. `producer.py` has zero.
5. **Guard rails non-vacuous for the two cases a caller controls** (unknown
   form, non-string transcript — fixture-covered). The three template-shape
   guards cannot fire on the frozen templates by construction; they are
   edit-detectors, which is their stated purpose.

---

## Claim I-1 — teacher-visible containment: **CONFIRMED**

1. **Runs before the model is loaded.** `assert_teacher_visible_precondition` at
   `producer.py:1677`, before `from_pretrained` (`:1686`, `:1694`); repeated per
   item at `:1709` inside `one_forward`, before `apply_chat_template`.
2. **Strict in both directions.** I fed `teacher_visible_texts` ten malformed
   message structures — wrong message count, unknown role, unknown content type,
   string frame payload, frame count ≠ 8, text part with an extra key, message
   with an extra key, two video parts, non-string text body, empty content list —
   and **all ten raise**. Mutation-wise, accepting an unknown content part or a
   string frame payload turns fixtures red.
3. **The ban is wide.** Measured on the real tranche: **402** tokens = all 200
   HateMM + all 200 MHC-ZH selected identifiers + `hate_video_` and
   `non_hate_video_`, with both datasets' identifiers banned in both datasets'
   prompts, so cross-item leakage is refused as firmly as self-leakage. Each
   token is expanded to `{raw, NFKC, NFKC.casefold}` and matched against
   `{NFKC(text), NFKC(text).casefold()}`. Shortest banned token is 11 characters.
   Dropping `non_hate_video_` turns `teacher_visible_ban_list_covers_both_datasets`
   red.
4. **No false positive on the real 400 transcripts.** All **800** renderings
   (400 items × 2 forms) through the real `build_messages` +
   `teacher_visible_texts` + `assert_teacher_visible_containment`, label-blind:
   **800 accepted, 0 rejected**, 0.70 s. Positive controls, all caught: a
   self-identifier appended, a cross-dataset identifier appended, `HATE_VIDEO_99`
   (case), a full-width `ｈａｔｅ_ｖｉｄｅｏ_` (NFKC), and a post-render template tamper.
5. **It cannot pass vacuously.** An empty ban list and an identifier absent from
   the ban list are both rejected (`identifier missing from ban list`), and
   disabling that precondition turns `teacher_visible_unbanned_identifier_rejected`
   red.

**The HateMM ID-label asymmetry is handled correctly, and the code says so.**
Every HateMM training identifier is `hate_video_*` or `non_hate_video_*`, so the
identifier *is* the binary label; MHC-ZH identifiers are opaque BiliBili codes
carrying no label information. **The sealed ID-only allowlist therefore provides
label containment for MHC-ZH only, and none at all for HateMM.**
`LABEL_BEARING_ID_SUBSTRINGS` encodes exactly this — measured live as
`{'HateMM': ('hate_video_', 'non_hate_video_'), 'MHC_zh': ()}`. HateMM label
containment is supplied instead by (a) this runtime check, which bans both the
identifiers and the two prefixes from every teacher-visible field, and (b) the
label-blindness of the selection rule, which I established independently by
reproducing both allowlists from `id` alone with
`label_value_materialized == 0`.

---

## Claim I-2 — the selection self-test is a known-answer vector: **CONFIRMED**

Both pinned digests recomputed outside the module by hand concatenation:

```
sha256(utf8("C04-A0T-SMALL-v1" + "HateMM" + "c04-known-answer-vector" + "20260729"))
  = 871e0363e1b01a82…   matches SELECTION_KNOWN_ANSWER_DIGESTS["HateMM"]
sha256(utf8("C04-A0T-SMALL-v1" + "MHC_zh" + "c04-known-answer-vector" + "20260729"))
  = 41bfb637f5cbb26b…   matches SELECTION_KNOWN_ANSWER_DIGESTS["MHC_zh"]
```

Mutating `SELECT_TAG` or `SELECT_SUFFIX` each turns `selection_known_answer_vector`
red (measured). The identifier is synthetic and belongs to neither dataset, so
the fixture pins the rule without naming a real, label-bearing video id.

---

## Claim I-3 — both ceilings machine-checked and fail-closed: **CONFIRMED**

### Tranche ceiling (7200 s)

One absolute deadline, computed once in `BudgetGuard.at_job_start` and never
recomputed (`remaining_seconds()` only reads it). Grid table above.

**Where the guard is and is not called.** `deadline_check` at `producer.py:1758`
(item boundary, before frame decode) and `:1704` (first statement of
`one_forward`, before `build_messages`); `guard.require_remaining(600, …)` at
`:1822` before the canonicalization and seal phase. Nowhere inside a decode, a
forward, a write or the seal's atomic staging. It may only ever STOP work before
a unit begins; there is no path that truncates, shortens or alters an output.

**What a breach leaves on disk.** `publish_budget_breach_record` writes
`resource/budget_breach.json` with the lineage, the guard snapshot, per-dataset
completed counts, teacher-call and frame-pack counters,
`outputs_truncated_or_altered: 0`, `seal_published: false`,
`no_performance_claim: true` and
`no_scientific_verdict_is_published_by_a_budget_breach: true` — no metric, no
teacher output, no reliability rate, no CONTINUE/KILL verdict. The producer
returns **40**; the wrapper has a dedicated exit-40 branch that `jq -e`-asserts
those fields and exits 40, distinctly from the 124/137/143 branch and from a
generic non-zero; a breach record on a zero-exit run is itself refused (exit 3).

### Campaign ceiling (28800 s aggregate, 7200 s effective)

**Checked before the ticket is consumed.** `assert_campaign_aggregate_headroom`
is called inside `validate_gpu_environment`, which is the **first** statement of
`claim()` — before `create_entry_marker` (`:627`), before `verify_gpu_lineage`
(`:628`), and ~130 lines before the ticket is read and consumed. It is also
called at `preflight.py:167`, before the namespace is materialized, and in the
producer before any model or data work.

**Read side, executed against sandbox copies:**

| mutation | result |
|---|---|
| pristine genesis, reserve 7200 | accepted |
| pristine genesis, reserve **7201** | HALT `would take the C04 campaign to 7201s against the 7200s effective ceiling` |
| ledger absent | HALT `campaign ledger is absent` |
| `payload_sha256` tampered | HALT `payload mismatch` |
| foreign `schema_version` | HALT `foreign campaign ledger schema` |
| foreign `run_id` | HALT `foreign campaign ledger run id` |
| aggregate cap raised to 999999 | HALT `cap is not the amendment cap` |
| aggregate cap lowered to 7200 | HALT `cap is not the amendment cap` |
| phase advanced, phase cap left 7200 | HALT `phase cap does not match the phase` |
| phase cap raised alone | HALT `phase cap does not match the phase` |
| unknown phase | HALT `unknown campaign phase` |
| first tranche carries an advance token | HALT `first-tranche phase carries an advance token` |
| head link wrong | HALT `campaign ledger head link` |
| aggregate ≠ Σ rows | HALT `aggregate does not equal its rows` |
| row chain break | HALT `chain break` |
| non-positive reservation | HALT `requested a non-positive reservation` |
| phase advanced **and** cap raised consistently | accepted — observation 5 |

**No stage can create or reset it.** `preflight.py:162-169` verifies and
deliberately never creates it; the only writer is `append_campaign_gpu_job`,
which appends and never truncates; the campaign path lies outside `ARTIFACT_ROOT`
and would be rejected by the preflight's staging namespace check.

**Its opening zero is evidence-backed.** I ran `sacct` myself, read-only:

```
13805|c04_a0t_small_v1_v5_preflight|0 |billing=8,cpu=8,mem=64G,node=1|FAILED
13840|c04_a0t_small_v1_v6_preflight|19|billing=8,cpu=8,mem=64G,node=1|COMPLETED
```

matching `genesis_evidence.rows` verbatim including `alloc_tres`, elapsed and
state. A full accounting sweep (`sacct -X -S 2020-01-01`) returns exactly **two**
C04 rows — these two — and **zero** C04 rows carrying any `gres/gpu` allocation,
so `gpu_seconds: 0` for both and
`these_are_the_only_c04_jobs_in_the_accounting_record: true` are both true.
`squeue` shows no queued or running job for this user.

**Write side reaches every path that burns GPU-seconds:**

| path | claim | marker | campaign row |
|---|---|---|---|
| exit 40 budget breach | yes | yes | **yes** |
| watchdog TERM/KILL (124/137/143) | yes | yes | **yes** |
| OOM / decode failure / any producer HALT after claim | yes | yes | **yes** |
| fully successful sealed run | yes | yes | **yes** (idempotently twice) |
| HALT before `claim()` publishes | no | **yes** (wrapper writes it) | **yes, via the fallback** |
| allocation never entered | no | no | **no** — prints and returns 0 |

---

## Additional checks

- **`--time`:** zero occurrences anywhere in the v7 set; each sbatch carries an
  explicit comment that the omission is deliberate.
- **Arrays / dependencies / chained submission / release / resubmission:**
  zero occurrences of `scontrol`, `scancel`, `srun`, `salloc`, `--array`,
  `--dependency`, `afterok`, `requeue`. The three textual `sbatch` hits are
  `implementation_hashes` keys naming the sbatch files. All three wrappers and
  all three Python entrypoints reject `SLURM_ARRAY_JOB_ID` /
  `SLURM_JOB_DEPENDENCY`.
- **`--gres`:** exactly one occurrence, `scripts/slurm/c04_a0t_small_v1_v7.sbatch:3`,
  `--gres=gpu:a100:1`. The preflight sbatch requests **no GPU**. The reconcile
  sbatch requests no GPU and its wrapper additionally rejects a non-empty
  `CUDA_VISIBLE_DEVICES` or `SLURM_GPUS_ON_NODE`, as does
  `validate_cpu_reconciliation_environment`.
- **Resources:** GPU sbatch = **1 GPU / 8 CPU / 64 GB** exactly; preflight = 8
  CPU / 64 GB, no GPU; reconcile = 1 CPU / 4 GB, no GPU.
  `resources.gpu_count/cpus/ram_gb = 1/8/64`, asserted in the preflight, the GPU
  ledger and the producer.
- **No OCR entrypoint, no network or external API client, no dev/test path, no
  cross-dataset path, no label reader:** zero hits for `import requests`,
  `urllib`, `httpx`, `aiohttp`, `import socket`, `boto3`, `openai`,
  `huggingface_hub`, `tesseract`, `easyocr`, `paddleocr`, `pytesseract`. The
  only `subprocess` in the whole tree is `gpu_ledger.py:277`,
  `sacct -X -n -P -j <id> -o JobIDRaw,ElapsedRaw,AllocTRES,State` — read-only.
  The only `label` reference on the data path is `_skip_json_value`, which
  advances the parser past the token and increments a skip counter; the projector
  then requires the decoded key set to be exactly `{id, window_text, language}`
  (measured: 1323 rows, 0 label values materialized). `HF_HUB_OFFLINE=1` /
  `TRANSFORMERS_OFFLINE=1` are exported by the wrapper and asserted by the
  producer; both `from_pretrained` calls pass `local_files_only=True`.
- **Authorization flags in the correct pre-review state:** exactly **one of
  seventeen** is `true` (`implementation_authorized`); all sixteen others are
  `false`, including `preflight_materialization_authorized`. All four review
  pins are `PENDING_*` sentinels; all four verdicts are `PENDING`; all four
  `prompt_hashes` are the freeze sentinel; `maps.expected_hashes` is the
  documented sentinel.
- **Unearned pins are rejected.** Against the frozen config:

  | entrypoint | result |
  |---|---|
  | `verify_historical_code_resource_authorization` | HALT `code/resource authorization SHA-256 is unpinned` |
  | `verify_payload_review` | HALT `payload hash verdict is not GO` |
  | `verify_gpu_execution_authorization` | HALT `GPU execution verdict is not GO` |
  | `verify_historical_gpu_execution_authorization` | HALT `authorization SHA-256 is unpinned` |
  | `verify_resource_reconciliation_authorization` | HALT `reconciliation verdict is not GO` |
  | `resolve_prompt_hashes(freeze=False)` | HALT `prompt hash A is unfrozen…` |
  | `resolve_prompt_hashes(freeze=True)`, materialization `false` | HALT (same) |
  | `resolve_prompt_hashes(freeze=True)`, materialization `true` | accepted — the single intended relaxation |
  | `verify_static_config` on the frozen config | HALT `preflight authorization is false` |
  | …with `watchdog_reserve_seconds` reverted to 120 | HALT `reserve: 120 != 300` |
  | …with `guard_seal_reserve_seconds` = 0 | HALT `guard seal reserve: 0 != 600` |
  | …with `guard_item_margin_seconds` = 120 | HALT `guard item margin: 120 != 300` |
  | …with the phase cap raised to 28800 | HALT `campaign first-tranche phase cap: 28800 != 7200` |
  | …with `gpu_authorized: true` | HALT `preflight authorization.gpu_authorized` |

- **Fixture suite:** 52 checks (round 4: 50; round 3: 47; v6: 25), all pass on
  the frozen bytes, no duplicate name (a duplicate would be silently dropped by
  `dict(self_test_fixtures())`).

### Mutation battery — the suite is non-vacuous

Fifteen independent mutations of production code, each run against the whole
frozen suite from a scratchpad sandbox:

| mutation | fixtures that turn red |
|---|---|
| re-narrow `stage_authorization` `provisional_gpu_usage_sha256` to 64-hex | `reconciliation_manifest_round_trips_in_both_seal_regimes` |
| rename `common.NO_SEAL_SENTINEL` | `reconciliation_manifest_round_trips_in_both_seal_regimes` |
| delete the `cosine` clamp | `cosine_of_identical_vectors_is_within_the_schema_bound` |
| `render_prompt` → `str.format` | suite **raises** `KeyError: '"source_relation"'` |
| `SELECT_TAG` mutated | `selection_known_answer_vector` |
| `SELECT_SUFFIX` mutated | `selection_known_answer_vector` |
| drop the `non_hate_video_` prefix | `teacher_visible_ban_list_covers_both_datasets` |
| `NUM_FRAMES` 8 → 7 | `full_canonical_record_round_trips_in_every_reliability_regime`, `full_prompt_record_round_trips_against_its_schema` |
| `TRANSCRIPT_CAP` 2048 → 4096 | `transcript_cap` |
| delete a `PROVISIONAL_USAGE_KEYS` member | suite **raises** `provisional usage writer exact-key failure` |
| disable the ban-list-membership precondition | `teacher_visible_unbanned_identifier_rejected` |
| accept an unknown content part | `teacher_visible_unknown_part_rejected` |
| accept string frame payloads | `teacher_visible_frame_path_rejected` |
| **`RELIABLE_CONFIDENCE_MIN` 3 → 2** | **none** — observation 3 |
| **`PROPOSITION_COSINE_MIN` 0.80 → 0.70** | **none** — observation 3 |

Two further mutations produced no red fixture and are recorded as observations 2
and 4: removing the case-folded haystack, and disabling the template-equality
check inside `assert_teacher_visible_containment`.

### The "irreversible resource before the rejecting check" family

I enumerated every entrypoint again and found **no remaining instance**:

- **GPU wrapper** — the `jq -e` authorization gate and the frozen-preflight-manifest
  existence test both precede the first `mkdir`; the EXIT trap armed earlier
  cannot create anything (replayed, run 1 above).
- **`claim()`** — `validate_gpu_environment` (config identity, prompt-hash
  binding, all seventeen authorization flags, resource caps, and
  `assert_campaign_aggregate_headroom`) is its first statement, before
  `create_entry_marker`, before the lock, before the ticket.
- **CPU preflight** — `verify_static_config` (which itself calls the campaign
  headroom check) and `verify_code_resource_authorization` run in `main()` before
  `preflight()`; `preflight()` tests `namespace.exists()` first, stages
  everything into a temp directory, and materializes with a single `os.rename`
  as its last statement.
- **Reconcile wrapper** — a nine-clause `jq -e` gate precedes the first Python
  call; `campaign_record` and `reconcile_terminal` both begin with
  `validate_cpu_reconciliation_environment`.
- **`campaign-record` before `reconcile-terminal`** is the one place where an
  irreversible write (the campaign append) precedes a stage that can halt. That
  ordering is deliberate and correct: accounting must never be refused, the
  append is idempotent, over-cap totals are recorded and flagged rather than
  rejected, and no later check can un-record. Round 2's H-1 concern (the append
  bricking the reader) is measurably gone.

---

## Non-blocking observations

Ordered roughly by how much I would want them closed before a v8, not by risk.
None is a finding.

1. **`NO_SEAL_SENTINEL` is defined twice rather than imported** —
   `common.py:2097` and `gpu_ledger.py:58`, both `"NO_SEAL_PUBLISHED"`.
   `gpu_ledger.py`'s 28-name import list from `common` does not include it. The
   two agree today (verified), and the new fixture binds the *common*-side
   spelling to the schema's `const`; but a one-sided edit to the
   *gpu_ledger*-side constant — the one the production seal-free tail actually
   uses — would turn no fixture red and would resurface exactly the round-4
   High, post-GPU. This is the last instance of the duplicated-literal shape the
   round-2 H-2 and round-4 I-3 repairs eliminated everywhere else; a one-line
   import into the existing `from … common import (…)` list closes it. It is
   *not* a finding because the frozen bytes agree and both files are inside
   `config_contract_sha256`, so drift requires a fresh review.
2. **NEW: disabling the template-equality check in
   `assert_teacher_visible_containment` turns no fixture red.** The check
   (`texts == [SYSTEM_PROMPT, render_prompt(form, transcript)]`) is the
   structural half of the containment argument — it is what makes the
   amendment's broader ban (prediction, neighbour, rank, margin, dataset
   statistic, fold role) satisfied by construction rather than by enumeration.
   `teacher_visible_template_tamper_rejected` does not discriminate it, because
   its tamper string appends `hate_video_3`, which the substring scan catches
   independently. A tamper fixture whose injected text contains **no** banned
   token would close the gap. The frozen code is correct — I verified the
   post-render tamper is rejected with `rendered text is not the frozen
   template`.
3. **`RELIABLE_CONFIDENCE_MIN` and `PROPOSITION_COSINE_MIN` are never
   cross-checked against their `config.reliability` copies** (unlike
   `resources`, which is asserted key by key). Mutating either turns no fixture
   red. Both the config and `common.py` are inside `config_contract_sha256`.
   Carried from round 4 (observation 10).
4. **Removing the case-folded haystack turns no fixture red** — the only
   leaking fixture uses an exact-case token. The frozen code is correct: I
   verified `HATE_VIDEO_99` is caught. Carried from round 4 (observation 2).
5. **The campaign ledger's `phase` is not cryptographically bound to any
   authorization artifact.** No code advances it (verified: `CAMPAIGN_PHASE_CAPS`
   is read-only everywhere), and every inconsistent advance halts — but a
   hand-edited ledger with `phase: CONDITIONAL_FULL_BANK` **and**
   `phase_cap_gpu_seconds: 28800` loads and raises the effective ceiling to
   28800. That is the human gate the amendment intends, but it is
   indistinguishable from an authorized one. Carried from rounds 2-4.
6. **`append_campaign_gpu_job` takes no lock on the campaign file.** It is
   protected only by the per-namespace `resource/gpu_ledger.lock`, which would
   not exclude a future namespace's reconciler; the head-hash race check turns a
   collision into a halt rather than corruption. Carried from rounds 2-4.
7. **`campaign_record`'s marker-fallback branch validates nothing about the
   marker** — it reads `["slurm_job_id"]` with no `schema_version`, `run_id` or
   self-hash check, unlike the claim branch. `sacct` must still show a terminal
   one-GPU row. Carried from rounds 3-4.
8. **No fixture round-trips a `resource_final_state` record.** I closed the live
   question by executing the writer's exact shape across sealed/seal-free ×
   {0, 6000, 7200, 7250, 7800, 7801}: writer and schema agree everywhere, with
   no gap at the 7800 boundary. Structural exposure only. Carried from round 4.
9. **NEW (and very small): the window between `claim()` publishing the
   allocation claim (`gpu_ledger.py:788`) and appending the ledger job row
   (`:814`).** A hard kill inside those ~26 in-lock, pure-CPU statements would
   leave a claim and a consumption record with an empty `ledger["jobs"]`, and
   `reconcile_terminal` would then halt on `reconciliation requires one GPU job`
   with no publishable final state. The campaign accounting is unaffected (the
   claim exists, so `campaign-record` works). `mark_exit`'s comment shows the
   window is known and deliberately handled on the exit side. I record it only
   because it is the sole remaining route to an unpublishable final state I
   could find; it needs a SIGKILL landing between two adjacent statements.
10. **`BudgetGuard.at_job_start` checks `item_margin` and `seal_reserve`
    individually but not their sum.** Not live (300 + 600 = 900 ≪ 6900). Carried
    from rounds 3-4.
11. **The exit-40 wrapper branch runs `jq -e` under `set -e`**, so an absent or
    malformed breach record surfaces as exit 1 rather than 40. The EXIT trap
    still records 40 in the ledger and marker, so the loss is cosmetic. Carried
    from rounds 2-4.
12. **`TERMINAL_SECONDS_HARD_MAX = 7800` lives only in `gpu_ledger.py` and,
    duplicated, in three schema maxima** — every other cap is also in
    `config.resources`. Both files are hash-pinned. Carried from round 4.
13. **`maps.expected_hashes` is protected only by inclusion in the contract
    hash.** Mutating it moves `config_contract_sha256` (measured), but no code
    asserts its literal value. Carried from rounds 1-4.
14. **Any GPU allocation entry, even one that HALTs before `claim()`, forecloses
    the namespace's single GPU opportunity**, because the wrapper writes the
    entry marker before its first Python call and then refuses any later job id.
    This is a deliberate trade — accountability for "every GPU-second" against
    reversibility — and the round-3 H-1 gate removes the config/authorization
    state as a possible cause. Recorded so the trade stays visible. Carried from
    round 4.

Round 4's observation 11 also holds: the CPU preflight transitively imports
`gpu_ledger.py` (hence `subprocess`) to share `GPU_LEDGER_KEYS` and
`RESOURCE_TICKET_KEYS`; it calls no `gpu_ledger` function, so `sacct` stays
unreachable from it, but
`slurm_submit_release_resubmit_entrypoint_present: false` is a statement about
reachability rather than about the import graph.

---

## Summary

**Verdict: GO (0C / 0H / 0I). No execution authority is conferred by this
review.**

What this GO means: the code and resource contract of implementation-v7 is
sound, at the seventeen hashes pinned in the round-5 request. The round-4 High
is closed by recomputation and carries a regression fixture; every finding from
rounds 1-4 holds closed at the current hashes; the scientific semantics are
byte-provably unchanged from v6 (both allowlists, all four prompt hashes, all
400 transcript digests and scalar counts reproduce, and the AST diff shows no
scientific definition or constant moved); both ceilings are machine-checked and
fail-closed; the accounting reaches every path that burns a GPU-second; the
"irreversible resource before the rejecting check" family — this campaign's
signature failure — is closed at every entrypoint I could reach; and the
52-fixture preflight suite is non-vacuous under fifteen independent mutations.

What this GO does **not** do: it does not flip
`authorization.preflight_materialization_authorized`, it does not set
`review.code_resource_verdict`, and it authorizes no GPU, no SLURM submission
and no teacher work. Those remain the owning stage's acts, and the payload-hash
review and GPU execution review are still `PENDING` by design.
