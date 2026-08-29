# C04-A0T-SMALL-v1 v7 — Fresh Independent Code/Resource Review, ROUND 4

Reviewer: fresh independent static reviewer (no exposure to the authoring reasoning)
Date: 2026-07-31
Stage reviewed: `CPU_PREFLIGHT` code/resource review of implementation-v7
Execution authority conferred by this review: **none**

---

## Verdict

**REVISE (0 Critical / 1 High / 0 Important)**

`authorization.preflight_materialization_authorized` must remain `false` and
`review.code_resource_verdict` must remain `PENDING`.

Every finding from rounds 1, 2 and 3 is now closed — I re-derived all seventeen
of them rather than reading the repair claims, and eight of the nine round-4
repair sub-claims survive independent re-derivation. The ninth does not.

The single High is a **relocated defect, not a new one**: round-3 I-2 said the
seal-free terminal paths (exit 40, 124/137/143, OOM, post-claim HALT) could not
complete the mandated `CPU_POST_JOB_RECONCILIATION` stage. The round-4 repair
gave `verify_reconciliation_lineage` a seal-free tail and widened
`c04_a0t_small_v1_v7_resource_final_state.schema.json` to accept the
`NO_SEAL_PUBLISHED` sentinel — but did **not** widen
`c04_a0t_small_v1_v7_stage_authorization.schema.json`, whose reconciliation
`payload_binding` still pins `provisional_gpu_usage_sha256` to
`^[0-9a-f]{64}$`. The code therefore demands a reconciliation authorization
manifest that its own frozen schema forbids, and the seal-free path halts at
exactly the same place, for a new reason. I proved both branches of the
alternative unsatisfiable by execution.

This is worth one more cycle for one reason above all: the fix is a two-line
JSON widening that is **free today and impossible after the CPU preflight
runs**, because it moves the schema's SHA-256 → `implementation_hashes` →
`config_contract_sha256`, which is pinned inside the no-clobber preflight
manifest, genesis GPU ledger and resource ticket.

**Apart from that one schema line, the payload is sound enough to proceed.** The
scientific semantics are byte-provably unchanged from v6, both ceilings are
machine-checked and fail-closed, the recurring "irreversible resource before the
rejecting check" family is now closed at every entrypoint I could find, and the
preflight fixture suite is genuinely non-vacuous under twenty independent
mutations.

| # | Severity | Finding |
|---|---|---|
| H-1 | High | `stage_authorization.schema.json` cannot express the `NO_SEAL_PUBLISHED` binding that `verify_resource_reconciliation_authorization` requires on the seal-free path, so round-3 I-2 is relocated rather than closed; unfixable after the preflight materializes. |

---

## Method and reviewer-boundary compliance

- **No SLURM job was submitted, held, released, requeued or cancelled.** The
  only Slurm interaction was read-only `sacct` (two invocations, below).
- **No GPU, teacher, model-weight or frame-decode work.** No `.safetensors` was
  opened; no video was decoded; the only video-adjacent operation was reading
  file names/lengths of ASR JSONL rows.
- **No file under `/data/jehc223/RGCL` was created, modified or deleted** other
  than this review file. Verified by `git status --porcelain` (no tracked C04
  modification) and by re-hashing all seventeen frozen files plus the entire v6
  tree before and after.
- **No dataset label value was materialized.** Every ASR read went through the
  frozen `project_train_asr_line`, which decodes only `id`, `window_text` and
  `language` and syntactically skips `label`. HateMM identifiers were handled as
  contained identifiers: they are never printed, and where an identifier had to
  appear in output I printed a truncated SHA-256 instead.
- **All work in a scratchpad outside the repository**, with
  `PYTHONDONTWRITEBYTECODE=1` on every invocation. Modules were imported from
  byte-identical scratchpad copies (or ROOT-patched copies), never from the
  repository path. `python -m py_compile` was never used. A post-hoc
  `find … -name '*.pyc' -newermt '-3 hours'` over both the repository and the
  scratchpad returned nothing.
- **`artifacts/c04/a0t_small_v1_impl_v7/` does not exist** and was not created;
  confirmed at the end of the session.
- **`artifacts/c04/campaign/gpu_ledger.json` is byte-identical.** Its append path
  *was* exercised — eleven times, at six spend magnitudes — but only against
  copies inside the scratchpad sandbox root.

### Hash verification (all 17, before and after)

Every pinned SHA-256 in the request matched disk on first read and again after
all work. Truncated to 16 hex for legibility; full values were compared:

| File | pinned | measured (start) | measured (end) |
|---|---|---|---|
| `…v7_common.py` | `2e4272c4…` | match | match |
| `…v7_preflight.py` | `ecdc8568…` | match | match |
| `…v7_gpu_ledger.py` | `944023b3…` | match | match |
| `…v7_producer.py` | `7a3c3a79…` | match | match |
| `…v7_preflight.sh` | `914dd5df…` | match | match |
| `…v7.sh` | `645e5011…` | match | match |
| `…v7_reconcile.sh` | `7af04322…` | match | match |
| `…v7_preflight.sbatch` | `919316c7…` | match | match |
| `…v7.sbatch` | `00ddeeed…` | match | match |
| `…v7_reconcile.sbatch` | `d8f634ec…` | match | match |
| `…prompt_record.schema.json` | `541d0245…` | match | match |
| `…canonical_record.schema.json` | `bacbddae…` | match | match |
| `…stage_authorization.schema.json` | `b367eb03…` | match | match |
| `…payload_review.schema.json` | `7edebdfe…` | match | match |
| `…resource_final_state.schema.json` | `e2f9dca5…` | match | match |
| `configs/c04/c04_a0t_small_v1_v7.json` | `3f436ea2…` | match | match |
| `artifacts/c04/campaign/gpu_ledger.json` | `fc6ca12c…` | match | match |

### v6 predecessor unmodified

Concatenated SHA-256 over the 16 v6 source/config/schema files:
`180fc756bc11f8fa…`, identical at start and end. Concatenated SHA-256 over
every file under `artifacts/c04/a0t_small_v1_impl_v6/`:
`bf3f7a38701b1b84…`, identical at start and end (and identical to the value
round 3 recorded). Directory mtimes remain `Jul 31 05:11`.

---

## Round-1 findings — closure status

| # | Round-1 finding | Status | How I re-derived it |
|---|---|---|---|
| C-A | reconciler's exact-key set had not grown with the writer | **CLOSED by construction** | `ast` census of every `require_exact_keys` call in all four modules: the second argument is `set(PROVISIONAL_USAGE_KEYS)` / `set(BUDGET_GUARD_KEYS)` on **both** the writer (`common.py:2073,2090`) and the reader (`gpu_ledger.py:465,471`). Mutating the constant by deleting one member makes the *writer* raise inside the preflight fixture — i.e. the CPU preflight fails first, before any GPU. |
| C-B | `proposition_cosine` could exceed schema `maximum: 1` | **CLOSED, and the fixture is real** | Deleting `max(-1.0, min(1.0, …))` from a scratchpad copy of `common.cosine` turns `cosine_of_identical_vectors_is_within_the_schema_bound` red (measured). The clamped value round-trips through the frozen canonical schema's `reliability` definition and through a full canonical record. |
| H-A | campaign write side reachable only after a seal | **CLOSED** | `campaign-record` is a distinct ledger mode keyed on `resource/allocation_claim.json` with an `allocation_entry_marker.json` fallback, run **first** in the reconcile wrapper under `set -e`, and independent of `seal/`. Re-derived by executing `campaign_record`'s branch selection against a sandbox namespace. |
| H-B | in-job guard had no margin over the wrapper `timeout` | **CLOSED, with more margin than round 3 measured** | Executed `BudgetGuard.at_job_start` over an 18-point grid of claim/producer-start offsets. Guard lead over the wrapper SIGTERM is `300 + c` seconds and never below 300, independent of producer start time. `SLURM_JOB_START_TIME` appears nowhere in the v7 tree except one docstring sentence. |
| I-C1 | accumulator enforced 28800 s, not the binding 7200 s | **CLOSED** | `campaign_effective_cap` measured at `min(7200, 28800) = 7200` on the frozen ledger; a recorded spend of 100 s refuses the next 7200 s reservation; thirteen distinct read-side mutations all halt (table below). |
| I-C2 | ~90 s of margin to the hard 7200 s ceiling; breach unrecoverable | **CLOSED** (see round-3 I-1) | `watchdog_reserve_seconds` 120 → 300 lifts the worst-case margin from ≈85 s to ≈265 s, and an over-cap terminal elapsed is now publishable up to 7800 s with a flag. Both measured. |
| I-C3 | preflight never round-trips a record against a downstream contract | **CLOSED for the record types the GPU stage writes** | Full `prompt_record` (normal + zero-frame) and full `canonical_record` (four reliability regimes) are now built exactly as the producer builds them and validated against the frozen schemas inside the preflight fixture set. Mutating `parse_teacher_response`'s slot shape, `build_slot_reliability`'s key set, or `NUM_FRAMES` turns them red (measured). |

Round-1 non-blocking observations re-checked: the pre-model-load containment
pass now goes through `producer.build_messages` + `teacher_visible_texts`, the
same path as the per-forward call site (re-derived: I ran the real 800
renderings through it); `maps.expected_hashes` is still protected only by
inclusion in the contract hash (mutating it moves `config_contract_sha256` —
measured — but no code asserts its literal value).

---

## Round-2 findings — closure status

| # | Round-2 finding | Status | How I re-derived it |
|---|---|---|---|
| H-1 | accumulator could brick itself on an over-cap write | **CLOSED** | Six append magnitudes (0 / 100 / 7199 / 7200 / 7250 / 30000 s) executed against sandbox copies of the campaign ledger. No append raised; every post-append `load_campaign_gpu_ledger()` succeeded; the over-cap rows carry `aggregate_exceeds_effective_cap: true`; every one of them refuses the next 7200 s reservation; a duplicate append halts with `campaign row already recorded`. |
| H-2 | reader restated the writer's key set; the fixture self-compared | **CLOSED** | `ast` proof above. |
| I-1 | `cosine` unreachable from the preflight; no full-record round-trip | **CLOSED** | Clamp deletion turns a fixture red; both full-record round-trips exist and are non-vacuous. |
| I-2 | post-loop canonicalization + seal phase unguarded | **CLOSED** | `guard.require_remaining(600, "the canonicalization and seal phase")` at `producer.py:1822`, inside the `try` whose `except BudgetDeadlineReached` publishes the accounting-only breach record and returns 40. Measured effective budget for that phase: `900 + c` seconds. |
| I-3 | GPU-seconds burned before `claim()` were recorded nowhere | **CLOSED** | The wrapper writes the entry marker before its first Python call; `campaign_record` falls back to it; with neither artifact present it prints `no allocation entry; nothing to record` and returns 0 rather than fabricating a row. |

---

## Round-3 findings — closure status

| # | Round-3 finding | Status | How I re-derived it |
|---|---|---|---|
| H-1 | GPU wrapper `mkdir`-ed the no-clobber namespace before any authorization gate | **CLOSED — verified end to end** | See below. |
| I-1 | `watchdog_reserve` unmeasured; over-cap terminal elapsed unpublishable | **CLOSED** | See below. |
| I-2 | `reconcile-terminal` seal-dependent | **NOT CLOSED — relocated.** See **Finding H-1**. |
| I-3 | three duplicated writer/reader key sets; no full-record round-trip | **CLOSED** | See below. |

### Round-3 H-1 — closed, and I reproduced the closure

I replayed the frozen GPU wrapper byte-faithfully against a sandbox root with a
stub `python`, in three configurations:

| run | config state | preflight manifest | exit | filesystem effect |
|---|---|---|---|---|
| 1 | the **frozen** config (`gpu_authorized: false`) | absent | 1 (jq gate) | **nothing — `artifacts/` was never created** |
| 2 | fully GPU-authorized | absent | 2 (`HALT_REVIEW_LINEAGE: no frozen preflight manifest`) | **nothing** |
| 3 | fully GPU-authorized | present | proceeds | `resource/` + entry marker + serial lock, then `claim` |

Nothing irreversible precedes the gate. The statements before it are `set -euo
pipefail`, `cd`, `readonly` assignments, one read of `/proc/uptime`, the EXIT
trap arming, and two environment tests. **The EXIT trap is armed earlier than
the gate**, so I checked it specifically: `mark_exit` does
`marker_path = root_path(...)` (which never creates a directory) and returns
immediately when the marker is absent; the only preceding work is
`json.load` of the config. Run 1 confirms this empirically — the trap fired and
no directory appeared. The trap is additionally invoked with `|| true`.

I separately enumerated every other way the v7 namespace could be created before
the preflight: `campaign_record` never `mkdir`s; `reconcile_terminal`'s
`lock_path.parent.mkdir` is preceded by `validate_cpu_reconciliation_environment`
(requires `post_job_reconciliation_authorized: true`) and
`verify_reconciliation_lineage`; the reconcile wrapper's own nine-clause `jq`
gate precedes both; the preflight's `preflight()` tests `namespace.exists()` as
its first statement. All three wrapper gates were executed against the frozen
config and all three refuse it.

### Round-3 I-1 — closed; the arithmetic re-derived

`watchdog_reserve_seconds` is 300 in the config and asserted `== 300` in both the
preflight and the GPU ledger; `120` appears nowhere in the v7 tree. Chain:

- ticket `watchdog_seconds = 7200 − 300 = 6900`; `claim()` re-derives and rejects
  any other value, and rejects `watchdog ≤ minimum_submit_remaining_seconds (300)`.
- `claim()` returns `6900 − c` (`c` = claim duration); the wrapper's `timeout`
  therefore SIGTERMs at `entry + 6900`, SIGKILLs at `entry + 6930`.
- worst-case `sacct ElapsedRaw ≈ P0 + 6930 + mark_exit`, so the margin to the
  7200 s ceiling is ≈ **265 s** (round 3 measured ≈ 85 s). `P0` was measured
  sub-second on job 13840.
- an over-cap terminal elapsed is now **recorded and flagged** rather than
  refused, bounded by `TERMINAL_SECONDS_HARD_MAX = 7800`.

Publication behaviour, measured by driving
`publish_or_verify_resource_final_state` against a sandbox namespace:

| terminal sacct seconds | seal | result |
|---|---|---|
| 6000 | yes | PUBLISHED, `terminal_elapsed_exceeded_cap: false` |
| 6000 | **no** | PUBLISHED, `seal_published: false`, `provisional…: NO_SEAL_PUBLISHED` |
| 7250 | yes | **PUBLISHED**, `terminal_elapsed_exceeded_cap: true` |
| 7250 | no | **PUBLISHED**, both flags set |
| 7800 | yes | PUBLISHED (exact boundary) |
| 7801 | yes | HALT `terminal GPU seconds outside [0,7800]` |

The code constant and the schema maxima agree exactly at 7800 — there is no gap
in which `strict_validate_terminal_ledger` passes and the schema rejects. The
schema diff versus v6 is exactly and only: three maxima 7200 → 7800,
`provisional_gpu_usage_sha256` gains the `NO_SEAL_PUBLISHED` alternative, and two
new required booleans. No scientific field moved. (The remaining 120 diff lines
are pure re-indentation; verified by structural JSON comparison.)

### Round-3 I-3 — closed

`ast` census of all fifteen `require_exact_keys` call sites across the four
modules:

| contract | writer | reader | argument |
|---|---|---|---|
| GPU ledger (15 keys) | `preflight.py:436` | `gpu_ledger.py:306` | `set(GPU_LEDGER_KEYS)` — same object |
| resource ticket (16) | `preflight.py:463` | `gpu_ledger.py:741` | `set(RESOURCE_TICKET_KEYS)` — same object |
| allocation claim (12) | `gpu_ledger.py:785` | `gpu_ledger.py:388` | `set(ALLOCATION_CLAIM_KEYS)` — same object |
| provisional GPU usage | `common.py:2090` | `gpu_ledger.py:465` | `set(PROVISIONAL_USAGE_KEYS)` — same object |
| budget guard | `common.py:2073` | `gpu_ledger.py:471` | `set(BUDGET_GUARD_KEYS)` — same object |

`preflight.py` imports `GPU_LEDGER_KEYS` and `RESOURCE_TICKET_KEYS` from
`gpu_ledger.py`, so both writers are checked at CPU-preflight time, before any
GPU. One key-set literal survives — `{"index","filename","size","sha256"}` at
`producer.py:848` — but its writer is 80 lines away in the same file and the
first frame pack is re-validated immediately after creation, i.e. before the
first forward, so a drift there costs one model load and zero forwards.

The full-record fixtures are real. Twenty independent mutations of production
code, each run against the whole 50-fixture suite:

| mutation | fixtures that turn red |
|---|---|
| delete the `cosine` clamp | `cosine_of_identical_vectors_is_within_the_schema_bound` |
| `render_prompt` → `str.format` | suite **raises** `KeyError: '"source_relation"'` |
| `SELECT_TAG` / `SELECT_SUFFIX` mutated | `selection_known_answer_vector` |
| drop the dataset term from the digest | `selection_known_answer_vector`, `selection_dataset_and_id_sensitivity` |
| reorder the digest concatenation | `selection_known_answer_vector` |
| one whitespace byte changed in prompt A | `prompt_bytes_unchanged_by_the_render_repair` |
| `build_slot_reliability` gains/loses a key | 7 fixtures incl. `full_canonical_record_round_trips_in_every_reliability_regime` |
| `parse_teacher_response` slot dict gains a key | `full_prompt_record_round_trips_against_its_schema` |
| `NUM_FRAMES` 8 → 7 | both full-record fixtures |
| provisional writer gains a field | suite **raises** `provisional usage writer exact-key failure` |
| delete a member of `PROVISIONAL_USAGE_KEYS` | suite **raises** (writer side fires first) |
| drop the HateMM label-bearing prefixes | `teacher_visible_ban_list_covers_both_datasets` |
| accept an unknown content part | `teacher_visible_unknown_part_rejected` |
| accept string frame payloads | `teacher_visible_frame_path_rejected` |
| `TRANSCRIPT_CAP` 2048 → 4096 | `transcript_cap` |

Three mutations produced **no** red fixture and are recorded as observations
below: removing the case-folded haystack from the containment check, and moving
`RELIABLE_CONFIDENCE_MIN` or `PROPOSITION_COSINE_MIN`.

---

## Claim 0 — no scientific semantic changed: **CONFIRMED**

**Selection re-derived from the v7 frozen rule reproduces the v6 frozen
allowlists exactly**, label-blind (only `id`/`window_text`/`language` decoded):

| dataset | train N | ids == v6 allowlist | digests == v6 allowlist | sha256 of the ordered id list |
|---|---|---|---|---|
| HateMM | 744 | **True** | **True** | `091fb1826cbc7f80…` |
| MHC_zh | 579 | **True** | **True** | `6c98c0d75891ce43…` |

**Prompt hashes recomputed from the v7 sources equal the v6 frozen artifact and
the pinned fixture literals:**

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
| `prompt_record`, `canonical_record`, `stage_authorization`, `payload_review` schemas | **0** | — (so the canonical schema's `maximum: 1` is byte-identical to v6) |
| `resource_final_state.schema.json` | 128 | round-4 I-1/I-2 only; structural JSON comparison shows exactly 3 maxima + 1 `anyOf` + 2 new booleans, the rest re-indentation |
| all three `.sbatch`, `*_preflight.sh` | 0 | — |
| `*_reconcile.sh` | 8 | round-1 H-A only |
| `*_v7.sh` | 62 | round-2 I-3 (18) + round-4 H-1 authorization gate & manifest test (44) |
| `preflight.py` | 42 | round-2/3/4 I-3 |
| `gpu_ledger.py` | 332 | C-A / H-A / H-2 / I-1 / I-2 / I-3 |
| `common.py` | 958 | C-1 / I-1 / I-2 / I-3, pure additions |
| `producer.py` | 497 | C-1 / I-1 / I-3 |
| `config.json` | 140 | version tokens, `v7_scope` prose, `resources` keys, `paths`, refreshed `implementation_hashes` |

`config_contract_sha256` normalization measured independently: filling all four
prompt hashes, flipping any authorization flag, setting all four review pins and
setting all four verdicts do **not** move it (so the v5 impossibility stays
closed); eight other mutations — `watchdog_reserve_seconds`,
`guard_seal_reserve_seconds`, `maps.expected_hashes`, `selection.suffix`,
`reliability.proposition_agreement_cosine_min`, `teacher_contract.num_frames`,
an `implementation_hashes` entry, and
`review.downstream_review_requires_terminal_resource_state` — all **do** move it.

---

## Claim C-1 — the prompt renderer: **CONFIRMED**

1. **The v6 form could never have succeeded.** Both templates embed
   `_SCHEMA_TEXT`, whose literal `{"source_relation":…}` and `{"S":0,…}` braces
   `str.format` reads as replacement fields. Executing
   `PROMPTS[form].format(transcript="x")` raises `KeyError: '"source_relation"'`
   for both forms — pinned by the `prompt_render_regression_str_format_would_raise`
   fixture, which I verified fires (removing the repair makes the whole suite
   raise rather than merely fail).
2. **The substitution is exactly the frozen one.** `render_prompt(form, T)` equals
   `PROMPTS[form][:-len("{transcript}")] + T` for both forms, and the rendered
   text still contains both literal JSON brace groups.
3. **No prompt byte changed** — the four hashes above.
4. **No `.format(transcript=` call site survives**: two textual hits, one in the
   `render_prompt` docstring and one inside the regression fixture. `producer.py`
   has zero.
5. **The guard rails are non-vacuous** for the two cases a caller controls
   (unknown form, non-string transcript — both fixture-covered). The three
   template-shape guards (`count != 1`, non-terminal placeholder, prefix
   preserved) cannot fire on the frozen templates by construction; they are
   edit-detectors, which is their stated purpose.

---

## Claim I-1 — teacher-visible containment: **CONFIRMED**

1. **Runs before the model is loaded.** `assert_teacher_visible_precondition` is
   called at `producer.py:1677`, before `idempotent_complete` (1678) and before
   the `transformers` import and `from_pretrained` (1681-1694). It is repeated
   per item at `producer.py:1709`, inside `one_forward`, before
   `apply_chat_template`.
2. **Strict in both directions.** `teacher_visible_texts` raises on: a message
   list of the wrong length, wrong keys, wrong role, non-list/empty content, a
   part without `type`, a `text` part with extra keys or a non-string body, a
   `video` part with extra keys, a frame count ≠ 8, a frame that is a
   `str`/`bytes`/`bytearray`/`Path`, an unknown content type, and any part census
   other than exactly one video part and exactly two text parts. Four of these
   are fixture-pinned and I confirmed by mutation that removing the
   unknown-type and string-frame guards turns fixtures red.
3. **The ban is wide.** `forbidden_teacher_visible_tokens` bans **both**
   datasets' 400 identifiers in **both** datasets' prompts (so cross-item leakage
   is refused as firmly as self-leakage) plus the two HateMM label-bearing
   prefixes — 402 tokens, exactly `2·200 + 2`. Each token is expanded to
   `{raw, NFKC, NFKC.casefold}` and matched against `{NFKC(text),
   NFKC(text).casefold()}`. The check also refuses to run at all if the item's
   own identifier is not on the ban list.
4. **No false positive on the real 400 transcripts.** I ran all **800**
   renderings (400 items × 2 forms) through `build_messages` +
   `teacher_visible_texts` + `assert_teacher_visible_containment` with the real
   normalized transcripts, label-blind: **0 false positives**, 0.69 s. Positive
   controls, all caught: a self-identifier appended, a cross-dataset identifier
   appended, `HATE_VIDEO_99` (case), and a full-width `ｈａｔｅ_ｖｉｄｅｏ_` (NFKC).
   Shortest banned token is `hate_video_` (11 chars); shortest identifier is 12.
5. **It cannot pass vacuously.** The assertion first requires
   `texts == [SYSTEM_PROMPT, render_prompt(form, transcript)]`; appending
   `"\nVideo id: hate_video_3"` to the rendered text is rejected by that equality
   before the substring scan runs (fixture `teacher_visible_template_tamper_rejected`).

**The HateMM ID-label asymmetry is handled correctly and stated explicitly in the
code.** `common.py:909-926` records that MHC-ZH identifiers are opaque BiliBili
codes while every HateMM identifier is `hate_video_*` or `non_hate_video_*` and
therefore *is* the label, so **the sealed ID-only allowlist provides label
containment for MHC-ZH only, and none at all for HateMM**. HateMM label
containment is supplied instead by this runtime check (which bans both the
identifiers and the two prefixes from every teacher-visible field) plus the
selection rule's label-blindness, which I established independently by
reproducing the allowlists from `id` alone. `producer.py:1658-1662` records the
same asymmetry in the access ledger.

---

## Claim I-2 — the selection self-test is a known-answer vector: **CONFIRMED**

`SELECTION_KNOWN_ANSWER_DIGESTS` are two hard-coded literals over a synthetic
identifier (`c04-known-answer-vector`) belonging to neither dataset, so the
fixture pins the rule without naming a real, label-bearing video id. It is
independent of the module's own code path: mutating the tag, the suffix, the
dataset term, or the concatenation order each turns
`selection_known_answer_vector` red (measured, four separate mutations), and
dropping the dataset term additionally turns `selection_dataset_and_id_sensitivity`
red. The v6 tautology (`selection_digest(x) == selection_digest(x)`) is gone.

---

## Claim I-3 — both ceilings machine-checked and fail-closed

### Tranche ceiling (7200 s) — **CONFIRMED**

One absolute deadline, computed once in `BudgetGuard.at_job_start` and never
recomputed; `remaining_seconds()` only reads it. Measured over an 18-point grid:

| claim `c` | producer start `e` | guard fires at | wrapper SIGTERM | lead | latest seal-phase start | seal budget |
|---|---|---|---|---|---|---|
| 0 | 0 / 30 / 120 | entry+6600 | entry+6900 | **300 s** | entry+6000 | **900 s** |
| 5 | 5 / 35 / 125 | entry+6595 | entry+6900 | **305 s** | entry+5995 | **905 s** |
| 30 | 30 / 60 / 150 | entry+6570 | entry+6900 | **330 s** | entry+5970 | **930 s** |
| 60 | 60 / 90 / 180 | entry+6540 | entry+6900 | **360 s** | entry+5940 | **960 s** |
| 120 | 120 / 150 / 240 | entry+6480 | entry+6900 | **420 s** | entry+5880 | **1020 s** |
| 300 | 300 / 330 / 420 | entry+6300 | entry+6900 | **600 s** | entry+5700 | **1200 s** |

The lead is `300 + c` and is **independent of producer start time**, because the
deadline is anchored to the allocation-entry `/proc/uptime` reading rather than
to `time.monotonic()` at guard construction.

**Where the guard is and is not called.** `deadline_check(guard, "item …")` at
`producer.py:1758` — at the item boundary, before frame decode; `deadline_check`
at `producer.py:1704` — the first statement of `one_forward`, before
`build_messages`; `guard.require_remaining(600, …)` at `producer.py:1822` —
before the canonicalization and seal phase. It is called nowhere inside a decode,
a forward, a write, or the seal's atomic staging. `BudgetDeadlineReached`
subclasses `RuntimeError` and is caught only by the `except` at
`producer.py:1826`; I checked every `except` on the path and none swallows it.

**What a breach leaves on disk:** `publish_budget_breach_record` writes
`resource/budget_breach.json` carrying the lineage, the guard snapshot, the
per-dataset completed count, the teacher-call and frame-pack counters,
`outputs_truncated_or_altered: 0`, `seal_published: false`,
`no_performance_claim: true` and
`no_scientific_verdict_is_published_by_a_budget_breach: true`. It contains no
metric, no teacher output, no reliability rate and no CONTINUE/KILL verdict.
The producer returns **40**; the wrapper has a dedicated exit-40 branch that
re-asserts those three fields with `jq -e` and propagates 40 distinctly from the
124/137/143 branch and from a generic non-zero. A breach record on a zero-exit
run is itself an error (`HALT_INVALID_FREEZE`). All confirmed by reading the
frozen bytes and by the sandbox wrapper replay.

### Campaign ceiling (28800 s, effective 7200 s) — **CONFIRMED**

**Checked before the ticket is consumed.** `assert_campaign_aggregate_headroom`
is called at `gpu_ledger.py:190` inside `validate_gpu_environment`, which is the
**first** statement of `claim()` — before `create_entry_marker`, before
`verify_gpu_lineage`, and ~570 lines before the ticket is read. It is also called
at `preflight.py:167` (before the namespace is materialized) and at
`producer.py:213` (before any model or data work).

**Read side, executed against sandbox copies:**

| mutation | result |
|---|---|
| pristine genesis (0 s spent) | accepted (7200 ≤ 7200) |
| ledger absent | HALT `campaign ledger is absent` |
| `payload_sha256` tampered | HALT `payload mismatch` |
| `aggregate_gpu_seconds` ≠ Σ rows | HALT `payload mismatch` |
| foreign `schema_version` | HALT `foreign campaign ledger schema` |
| foreign `run_id` | HALT `foreign campaign ledger run id` |
| aggregate cap raised to 999999 | HALT `cap is not the amendment cap` |
| aggregate cap lowered to 7200 | HALT `cap is not the amendment cap` |
| phase advanced, phase cap left 7200 | HALT `phase cap does not match the phase` |
| phase cap raised alone | HALT `phase cap does not match the phase` |
| unknown phase | HALT `unknown campaign phase` |
| first tranche carries an advance token | HALT `first-tranche phase carries an advance token` |
| head link wrong | HALT `campaign ledger head link` |
| non-positive reservation | HALT `requested a non-positive reservation` |
| phase advanced **and** cap raised consistently | accepted — see observation 3 |

**No stage can create or reset it.** `preflight.py:162-169` verifies it and
comments that it is deliberately never created there; the only writer is
`append_campaign_gpu_job`, which appends and never truncates. A missing ledger
halts every stage rather than defaulting to zero.

**Write side is idempotent, sacct-derived and cannot brick:**

| appended `gpu_seconds` | append raised | on-disk aggregate | over-cap flag | later `load` | next 7200 s reservation | duplicate append |
|---|---|---|---|---|---|---|
| 0 | none | 0 | `false` | OK | accepted | HALT `already recorded` |
| 100 | none | 100 | `false` | OK | REFUSED | HALT `already recorded` |
| 7199 | none | 7199 | `false` | OK | REFUSED | HALT `already recorded` |
| 7200 | none | 7200 | `false` | OK | REFUSED | HALT `already recorded` |
| **7250** | **none** | **7250** | **`true`** | **OK** | **REFUSED** | HALT `already recorded` |
| 30000 | none | 30000 | `true` | OK | REFUSED | HALT `already recorded` |

Every rejecting check (`head race`, duplicate job id, non-integer seconds) runs
before the write; the two `campaign_effective_cap` evaluations and the flag are
computed before `os.replace`; nothing after the write can raise. Re-recording the
same job verifies `gpu_seconds` and `sacct_state` instead of appending, so the
two calls in one reconcile run (`campaign-record`, then `reconcile_terminal`)
cannot double-count.

**Its opening zero is evidence-backed.** I ran `sacct` myself:

```
13805|c04_a0t_small_v1_v5_preflight|0 |billing=8,cpu=8,mem=64G,node=1|FAILED
13840|c04_a0t_small_v1_v6_preflight|19|billing=8,cpu=8,mem=64G,node=1|COMPLETED
```

exactly matching the ledger's `genesis_evidence.rows`, including
`gres_gpu_present: false` and `gpu_seconds: 0`. A sweep of the entire accounting
record (`sacct -X -S 2020-01-01`) returns **two** C04 rows in total — these two —
and **zero** C04 rows with any `gres/gpu` allocation. The claim
`these_are_the_only_c04_jobs_in_the_accounting_record: true` is true.

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

- **`--time`:** absent from all three sbatch files and all three wrappers; each
  sbatch carries an explicit comment that the omission is deliberate.
- **Arrays / dependencies / chained submission / release / resubmission:**
  absent. Zero occurrences of `sbatch`, `scontrol`, `scancel`, `srun`, `salloc`,
  `--array`, `--dependency`, `afterok` or `requeue` anywhere in the v7 set; the
  only textual hits for "release"/"resubmit" are authorization-flag names and
  prose. All three wrappers and all three Python entrypoints reject
  `SLURM_ARRAY_JOB_ID` / `SLURM_JOB_DEPENDENCY`.
- **`--gres`:** exactly one occurrence, `scripts/slurm/c04_a0t_small_v1_v7.sbatch:3`,
  `--gres=gpu:a100:1`. The preflight sbatch requests no GPU. The reconcile sbatch
  requests no GPU and its wrapper additionally rejects a non-empty
  `CUDA_VISIBLE_DEVICES` or `SLURM_GPUS_ON_NODE`, as does
  `validate_cpu_reconciliation_environment`.
- **Resources:** GPU sbatch = **1 GPU / 8 CPU / 64 GB** exactly; preflight = 8
  CPU / 64 GB, no GPU; reconcile = 1 CPU / 4 GB, no GPU.
  `resources.gpu_count/cpus/ram_gb = 1/8/64` and are asserted in the preflight,
  the GPU ledger and the producer.
- **No OCR entrypoint, no network/API client, no dev/test path, no cross-dataset
  path, no label reader:** zero hits for `requests`(module), `urllib`, `httpx`,
  `aiohttp`, `socket`, `boto3`, `tesseract`, `easyocr`, `paddleocr`,
  `pytesseract`; the single textual "requests" is the word inside a frame-decode
  docstring. The only `subprocess` in the whole tree is `gpu_ledger.py:277`,
  `sacct -X -n -P -j <id> -o JobIDRaw,ElapsedRaw,AllocTRES,State` — read-only.
  The only `label` reference on the data path is `_skip_json_value`, which
  advances the parser past the token and increments a skip counter; the projector
  then requires the decoded key set to be exactly `{id, window_text, language}`.
  `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` are exported by the wrapper and
  asserted by the producer; both `from_pretrained` calls pass
  `local_files_only=True`. `root_path` rejects any `dev`/`test`/`validation`-like
  path component, `train_asr_path` rejects symlinks and out-of-root ASR, and
  `video_path` pins each dataset's physical root separately.
- **Authorization flags in the correct pre-review state:** exactly one of
  seventeen is `true` (`implementation_authorized`); all sixteen others are
  `false`, including `preflight_materialization_authorized`. All four review pins
  are `PENDING_*` sentinels and all four verdicts are `PENDING`.
  `maps.expected_hashes` is the documented sentinel string.
- **Unearned pins are rejected.** Against the frozen config: the preflight's
  static gate halts at `HALT_INVALID_FREEZE: preflight authorization is false`;
  `resolve_prompt_hashes(freeze=False)` halts on the sentinel;
  `resolve_prompt_hashes(freeze=True)` with materialization `false` halts;
  only `freeze=True` + materialization `true` is accepted, the single intended
  relaxation. Reverting `watchdog_reserve_seconds` to 120 halts
  (`reserve: 120 != 300`); zeroing `guard_seal_reserve_seconds` halts; setting
  `gpu_authorized: true` at preflight halts; a wrong prompt hash halts.
- **Every stage manifest shape the code can demand, checked against its frozen
  schema:**

  | manifest the code requires | validates? |
  |---|---|
  | `CPU_PREFLIGHT` (`payload_binding: "NO_PREFLIGHT_PAYLOAD_YET"`) | VALID |
  | `GPU_TEACHER_PRELABEL_SEAL` (4-key payload binding) | VALID |
  | `CPU_POST_JOB_RECONCILIATION`, **sealed** run | VALID |
  | `CPU_POST_JOB_RECONCILIATION`, **seal-free** run | **SCHEMA FAILURE** — see H-1 |
  | payload-hash review (incl. the attestation identity) | VALID |

---

# Findings

## HIGH H-1 — the seal-free reconciliation is unsatisfiable: the code demands an authorization manifest that the frozen `stage_authorization` schema forbids, so round-3 I-2 is relocated, not closed

**Where.**
`scripts/analysis/c04_a0t_small_v1_v7_gpu_ledger.py:58` (`NO_SEAL_SENTINEL = "NO_SEAL_PUBLISHED"`),
`:447-460` (the seal-free tail sets `provisional_sha = NO_SEAL_SENTINEL`),
`:540-551` (`_reconciliation_lineage_tail` passes it into
`verify_resource_reconciliation_authorization`),
`scripts/analysis/c04_a0t_small_v1_v7_common.py:735-746` (the manifest must equal
that value exactly) and `:479` (`_verified_review_file` schema-validates the
manifest first), versus
`schemas/c04/c04_a0t_small_v1_v7_stage_authorization.schema.json:159`
(`"provisional_gpu_usage_sha256": {"$ref": "#/definitions/sha256"}`, i.e.
`^[0-9a-f]{64}$`).

**Mechanism.** Round 4 widened the *final-state* schema to accept the sentinel:

```
resource_final_state.schema.json : "anyOf": [ {"$ref": "#/definitions/sha256"},
                                              {"const": "NO_SEAL_PUBLISHED"} ]
stage_authorization.schema.json  : {"$ref": "#/definitions/sha256"}       <-- not widened
```

`payload_binding` in the stage-authorization schema is a `oneOf` of three
variants. The reconciliation binding cannot match variant 1 (a const string) or
variant 2 (`additionalProperties: false` over four other keys), and variant 3
rejects a non-hex `provisional_gpu_usage_sha256`. So a manifest carrying the
sentinel matches **zero** variants.

**Measured.** I built both manifests a reviewer could possibly sign for a
seal-free run and drove the real `verify_reconciliation_lineage` (only the four
upstream verifiers stubbed; the reconciliation-authorization path left intact)
against a sandbox namespace with the claim, the entry marker and no seal:

```
(a) provisional_gpu_usage_sha256 = "NO_SEAL_PUBLISHED"   (what the CODE demands)
    -> RuntimeError: resource reconciliation authorization schema failure:
       ['payload_binding']: … is not valid under any of the given schemas
(b) provisional_gpu_usage_sha256 = <64 hex>              (what the SCHEMA demands)
    -> RuntimeError: HALT_REVIEW_LINEAGE: reconciliation authorization mismatch payload_binding

control: sealed run, 64-hex provisional  ->  passes the manifest schema and the
         binding check, and proceeds to the seal-content checks
```

Both options fail; the intersection is empty. The failure is raised twice over —
first inside `reviewed_pre_reconciliation_ledger_sha256`, again inside
`_verified_review_file` — so it is not routed around.

**Failure scenario.** The GPU tranche runs and terminates on any non-sealing
path — the exit-40 budget breach (the very path the tranche ceiling exists to
create), a watchdog TERM/KILL at 124/137/143, an OOM, a decode failure, or any
producer HALT after `claim()`. `campaign-record` correctly writes the true
sacct spend into the cross-version accumulator, so the amendment's 8 GPU-hour
ceiling stays honest — that half is genuinely fixed. Then `reconcile-terminal`
dies in `verify_reconciliation_lineage` no matter which manifest the reviewer
signed. The reconcile wrapper's recovery branch is guarded by
`jq -e '.state == "SACCT_TERMINAL_RECONCILED"'`, which is false (the ledger is
still `EXIT_RECORDED_PENDING_SACCT` or `CLAIMED_ACTIVE`), so it takes
`exit "$C04_RECONCILE_STATUS"`. The per-namespace ledger keeps its 7200 s
reservation forever, `resource/resource_final_state.json` is never published, and
`review.downstream_review_requires_terminal_resource_state: true` then blocks
every downstream review of that namespace permanently. This is, line for line,
the outcome round-3 I-2 described.

**Why this warrants another cycle rather than an Important.** The repair is two
JSON lines. But it is a change to a file whose SHA-256 is in
`config.implementation_hashes`, hence in `config_contract_sha256` (I measured
that an `implementation_hashes` edit moves the contract hash), hence in the
values the CPU preflight bakes into the preflight manifest, the genesis GPU
ledger and the single-use resource ticket — inside a no-clobber namespace. So
the window in which this is a two-line edit closes the moment the preflight runs;
after that the only remedy is a full v8 namespace rebuild, which is the cost this
campaign has already paid three times. It is also a claim the review request
asserts as delivered ("a terminal resource state is published on every terminal
path"), and it is not: the assertion holds for the code and fails for the frozen
contract the code must satisfy.

**What would close it.** Widen the third `payload_binding` variant of
`schemas/c04/c04_a0t_small_v1_v7_stage_authorization.schema.json` exactly as
`resource_final_state.schema.json` was already widened:

```json
"provisional_gpu_usage_sha256": {
  "anyOf": [ {"$ref": "#/definitions/sha256"}, {"const": "NO_SEAL_PUBLISHED"} ]
}
```

then refresh `config.implementation_hashes` for that file. To make the repair
non-recurring rather than point-fixed, add a preflight fixture that builds a
`CPU_POST_JOB_RECONCILIATION` authorization manifest in **both** regimes — with a
64-hex provisional digest and with `NO_SEAL_PUBLISHED` — and runs each through
`validate_schema` against `schemas.stage_authorization`, with a non-vacuity case
(a third value such as `"MAYBE"`) that must fail. That is the same round-trip
discipline the round-4 I-3 repair applied to `prompt_record` and
`canonical_record`; extending it to the manifests is what would have caught this
before I did. Consider extending it to `resource_final_state` at the same time
(see observation 1).

---

## Non-blocking observations

1. **No fixture round-trips a `resource_final_state` record.** Round 3 asked for
   three record types; two were added. The final state is schema-validated at
   `gpu_ledger.py:1094`, immediately before publication — i.e. after the whole
   allocation. I closed the live question by executing the writer across its full
   range (sealed/seal-free × 0/6000/7250/7800/7801 s): writer and schema agree
   everywhere, with no gap at the 7800 boundary. Structural exposure only.
2. **Removing the case-folded haystack from `assert_teacher_visible_containment`
   turns no fixture red.** The frozen code is correct — I verified a mixed-case
   leak (`HATE_VIDEO_99`) is caught — but the only leaking fixture uses an
   exact-case token, which the raw variant already catches. A mixed-case fixture
   would close the coverage gap.
3. **The campaign ledger's `phase` is not cryptographically bound to any
   authorization artifact.** No code advances it (verified: `CAMPAIGN_PHASE_CAPS`
   is read-only everywhere), and every inconsistent advance halts — but a
   hand-edited ledger with `phase: CONDITIONAL_FULL_BANK` **and**
   `phase_cap_gpu_seconds: 28800` loads and raises the effective ceiling to 28800.
   Unchanged from rounds 2-3.
4. **`append_campaign_gpu_job` takes no lock on the campaign file.** It is
   protected only by the per-namespace `resource/gpu_ledger.lock`, which would not
   exclude a future namespace's reconciler. The head-hash race check turns a
   collision into a halt rather than corruption. Unchanged from rounds 2-3.
5. **`campaign_record`'s marker-fallback branch validates nothing about the
   marker** — it reads `["slurm_job_id"]` with no `schema_version`, `run_id` or
   self-hash check, unlike the claim branch. `sacct` must still show a terminal
   one-GPU row, so the risk is low, but the asymmetry is gratuitous. Unchanged
   from round 3.
6. **`BudgetGuard.at_job_start` checks `item_margin` and `seal_reserve`
   individually but not their sum.** Not live (300 + 600 = 900 ≪ 6900).
   Unchanged from round 3.
7. **The exit-40 wrapper branch runs `jq -e` under `set -e`**, so an absent or
   malformed breach record surfaces as exit 1 rather than 40. The EXIT trap still
   records 40 in the ledger and marker (`C04_FINAL_STATUS` is already set), so the
   loss is cosmetic. Unchanged from rounds 2-3.
8. **`TERMINAL_SECONDS_HARD_MAX = 7800` lives only in `gpu_ledger.py` and,
   duplicated, in three schema maxima.** Every other cap is also in
   `config.resources`. Both files are hash-pinned so drift requires a review, but
   the config no longer fully describes the resource contract.
9. **The reconcile wrapper's closing `jq -e` does not surface
   `terminal_elapsed_exceeded_cap` or `seal_published`.** Both are printed to
   stderr by the Python stages and recorded in the final state, so a downstream
   reader can see them; the wrapper exits 0 either way.
10. **`RELIABLE_CONFIDENCE_MIN`, `PROPOSITION_COSINE_MIN` and the other
    reliability thresholds are never cross-checked against their
    `config.reliability` copies** (unlike `resources`, which is asserted key by
    key). Mutating either constant turns no fixture red. Both the config and
    `common.py` are inside `config_contract_sha256`, so drift requires a review,
    but the duplication is unguarded. Same shape as the `maps.expected_hashes`
    observation carried since round 1.
11. **The CPU preflight now transitively imports `gpu_ledger.py`, and therefore
    `subprocess`,** in order to share `GPU_LEDGER_KEYS` / `RESOURCE_TICKET_KEYS`.
    An `ast` reachability check confirms the preflight uses only those two
    constants and calls no `gpu_ledger` function, so `sacct` remains unreachable
    from it — but the preflight's `slurm_submit_release_resubmit_entrypoint_present:
    false` assertion is now a statement about reachability rather than about the
    import graph.
12. **Any GPU allocation entry, even one that HALTs before `claim()`, forecloses
    the namespace's single GPU opportunity**, because the wrapper writes the entry
    marker before its first Python call and then refuses any later job id. This is
    a deliberate trade — accountability for "every GPU-second" (round-2 I-3)
    against reversibility — and the round-4 gate removes the config/authorization
    state as a possible cause. Recorded so the trade stays visible, not as a
    defect.

---

## Summary

**Verdict: REVISE (0C / 1H / 0I). No execution authority is conferred.**
`authorization.preflight_materialization_authorized` must remain `false`;
`review.code_resource_verdict` must remain `PENDING`.

Everything else in this payload is, as far as four rounds of adversarial
recomputation can establish, sound: the science is byte-identical to v6, the two
ceilings are machine-checked and fail-closed, the accounting reaches every path
that burns a GPU-second, the fixture suite is non-vacuous under twenty
mutations, and the "irreversible resource before the rejecting check" family —
this campaign's signature failure — is closed at every entrypoint I could reach.
Close H-1 and I would expect the next round to be a GO.
