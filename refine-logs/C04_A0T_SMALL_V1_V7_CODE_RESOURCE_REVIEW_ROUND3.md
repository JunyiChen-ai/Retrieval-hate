# C04-A0T-SMALL-v1 v7 — Fresh Independent Code/Resource Review, ROUND 3

Reviewer: fresh independent static reviewer (no exposure to the authoring reasoning)
Date: 2026-07-31
Stage reviewed: `CPU_PREFLIGHT` code/resource review of implementation-v7, third revision
Predecessors, both left byte-intact:
- `refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW.md` (round 1, `REVISE 2C/2H/3I`)
- `refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW_ROUND2.md` (round 2, `REVISE 0C/2H/3I`)

Execution authority conferred by this review: **none**.

Note on the deliverable name: the request names
`C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW.md`, which is the round-1 file. This
review is written to the round-3 path so both earlier reviews survive unaltered.

## Verdict

**REVISE — 0 Critical / 1 High / 3 Important (0C / 1H / 3I)**

All ten prior findings (round-1's eight, round-2's five, with three shared) were
re-derived from the frozen bytes. **Both round-1 Criticals, both round-1 Highs,
round-1 I-C1, and both round-2 Highs are genuinely closed** — in every case I
reproduced the original trigger and confirmed the frozen bytes no longer fail,
rather than accepting the summary. Round-2 I-1 and I-3 are closed for the halves
the round-3 request names. Round-1 I-C2 is **still not closed** and is now three
rounds old.

The one High is **new**, and it is the failure family the request asked me to
enumerate, found in the one place none of the previous rounds looked: the GPU
wrapper's shell preamble. `scripts/wrappers/c04_a0t_small_v1_v7.sh` creates the
single-shot no-clobber namespace with `mkdir -p` **before any authorization
gate** — a gate its two sibling wrappers both carry and it does not. An
out-of-order or unauthorized GPU submission therefore permanently forecloses the
CPU preflight for v7 and forces a full namespace rebuild.

Claims 0, C-1, I-1 and I-2 all reproduced exactly, as in both prior rounds.

## Method and reviewer-boundary compliance

- No SLURM job submitted, held, released, requeued or cancelled. `sacct` and
  `squeue` used read-only only.
- No GPU, teacher, model-weight or frame-decode work. No model file was opened
  at all this session (not even metadata).
- **No file under `/data/jehc223/RGCL` was created, modified or deleted other
  than this review file.** Verified two ways at session end: all 17 pinned
  SHA-256 values re-verify (including
  `artifacts/c04/campaign/gpu_ledger.json` →
  `fc6ca12c32427625d0b80c16b7802ef9a574ced0dbf0288edc3938d217267414`), and
  `find … -newermt '2026-07-31 22:00'` over `scripts/`, `schemas/`, `configs/`
  and `artifacts/c04/` returns nothing. No `.pyc` was written for any v7 module
  (`__pycache__` contains only pre-existing entries, newest a v5 file dated
  2026-07-30).
- `artifacts/c04/a0t_small_v1_impl_v7/` does not exist and was not created.
- No dataset label value was materialized. Both ASR files were read only through
  the frozen `project_train_asr_line` projector (`id`, `window_text`,
  `language`). HateMM identifiers were hashed, counted and compared, never
  printed and never reasoned from as labels.
- All work in `…/scratchpad/review-r3`, outside the repository, with
  `PYTHONDONTWRITEBYTECODE=1` on every invocation. `common.py` was imported only
  from byte-identical scratchpad copies; mutation experiments ran against a copy
  whose sole edit was `ROOT`, repointed to a sandbox tree. `BudgetGuard` was
  executed by extracting its frozen `ClassDef` with `ast`, never by importing
  `producer.py`. `py_compile` was not used.

### Hash verification (all 17, before and after)

Every pinned SHA-256 in the request table matched disk at session start and
re-matched at session end (17/17, byte-for-byte diff against the request table).
In addition:

- all 15 `configs/c04/c04_a0t_small_v1_v7.json → implementation_hashes` verify
  against disk (15/15) and equal the request table;
- all 15 `frozen_design_hashes` verify against disk (15/15);
- the config is **not** listed inside its own `implementation_hashes`, so the
  authorization flips the pipeline requires between stages cannot break
  `verify_bound_file_map`.

### v6 predecessor unmodified

- `artifacts/c04/a0t_small_v1_impl_v6/freeze/preflight_manifest.json` is
  self-consistent (`payload_sha256` reproduces) and **all 14** of its
  `staged_output_hashes` verify byte-for-byte on disk.
- all 15 entries of `configs/c04/c04_a0t_small_v1_v6.json → implementation_hashes`
  verify against disk.
- every v6 artifact carries a single mtime, `2026-07-31 05:11` (job 13840),
  predating every v7 source file (`2026-07-31 21:45/21:46`).

---

## Round-1 findings — closure status

| # | Round-1 finding | Status | How I re-derived it |
|---|---|---|---|
| C-A | reconciler's exact-key set had not grown with the writer | **CLOSED, by construction** | `ast` census: `gpu_ledger.py` imports `PROVISIONAL_USAGE_KEYS` and `BUDGET_GUARD_KEYS` (28-name import list) and passes them to `require_exact_keys` at lines 453/459; the writer `build_provisional_gpu_usage` validates against the same objects. One-sided drift can no longer exist for this contract. |
| C-B | `proposition_cosine` could exceed schema `maximum: 1` | **CLOSED, and the fixture is real** | Deleting the clamp from a scratchpad copy turns `cosine_of_identical_vectors_is_within_the_schema_bound` **red** (measured, see Claim I-3 below). A full canonical record with the clamped value validates. |
| H-A | campaign write side reachable only after a seal | **CLOSED** | `campaign-record` is a distinct mode keyed on `allocation_claim.json` with a marker fallback, run first in the reconcile wrapper under `set -e`. |
| H-B | in-job guard had no margin over the wrapper `timeout` | **CLOSED** | Frozen `BudgetGuard` executed over a 5×4 grid of claim/verification durations: lead is `300 + claim_duration`, **never below 300 s**, and independent of verification time. `SLURM_JOB_START_TIME` appears nowhere except one docstring. |
| I-C1 | accumulator enforced 28800 s, not the binding 7200 s | **CLOSED** | Effective cap measured as `min(phase, aggregate) = 7200`; a recorded spend of 1 s refuses the next 7200 s reservation; ten distinct phase/cap mutations all halt. |
| I-C2 | ~90 s margin to the hard 7200 s ceiling; breach unrecoverable | **NOT CLOSED — three rounds old** | see **Important I-1**. The blast radius shrank (the campaign ledger no longer bricks, and the in-job guard makes the `timeout` path a tail event), so this is now Important rather than the round-2 High that subsumed it. |
| I-C3 | preflight never round-trips a record against a downstream contract | **PARTIALLY CLOSED** | see **Important I-3**. Narrowed a second time; two channels remain. |

Both round-1 non-blocking observations re-checked and unchanged: the
pre-model-load containment pass still self-compares in one half (the message
assembly half is non-vacuous only at `producer.py:1709`, which does run before
every forward), and `maps.expected_hashes` is protected only by inclusion in the
contract hash (mutating it moves `config_contract_sha256`; no code asserts its
literal value).

## Round-2 findings — closure status

| # | Round-2 finding | Status | How I re-derived it |
|---|---|---|---|
| H-1 | accumulator could brick itself on an over-cap write | **CLOSED** | Executed five append magnitudes (0/100/7200/7250/30000 s) against a sandbox ledger. See the table under Claim I-3. |
| H-2 | reader restated the writer's key set; the fixture self-compared | **CLOSED as scoped** | `ast` proof above. Residual: three *other* writer/reader contracts remain duplicated literals — all three agree today (measured) — see **Important I-3**. |
| I-1 | `cosine` unreachable from the preflight; no full-record round-trip | **CLOSED for the clamp; the wider half remains** | Clamp deletion now turns a fixture red (measured). No fixture yet builds a full `canonical_record`/`prompt_record`/`resource_final_state`. See **Important I-3**. |
| I-2 | post-loop canonicalization + seal phase unguarded | **CLOSED** | `guard.require_remaining(600, "the canonicalization and seal phase")` sits at `producer.py:1822`, inside the same `try` whose `except BudgetDeadlineReached` publishes the accounting-only breach record and returns 40. Measured effective budget for that phase is **≈900 s**, not 600 (see below). |
| I-3 | GPU-seconds burned before `claim()` were recorded nowhere | **CLOSED** | The wrapper writes the entry marker at `c04_a0t_small_v1_v7.sh:39-64`, before its first Python call; `campaign_record` falls back to it, and with neither artifact present it prints and returns 0 rather than fabricating a row. |

---

## Claim 0 — no scientific semantic changed: **CONFIRMED**

### Selection re-derived from the v7 frozen rule reproduces the v6 frozen allowlists

Both train ASR files read through the v7 projector, ranked by the v7
`selection_digest` with the v7 tie-break, first 200 taken:

| dataset | train N | recomputed order == v6 frozen allowlist | sha256(newline-joined ordered ids) |
|---|---|---|---|
| HateMM | 744 | **True** | `091fb1826cbc7f80…` |
| MHC_zh | 579 | **True** | `6c98c0d75891ce43…` |

Also reproduced per dataset, against the v6 frozen artifacts: every stored
`selection_sha256` (200/200), contiguous ranks 0..199, all 200
`transcript_sha256` and all 200 `transcript_scalar_count`. That last pair
independently pins the transcript normalization, cap, head/tail split and
separator as unchanged.

### Prompt hashes

Recomputed from the v7 sources:

```
system   1ffc06755b18f9315dc930aaf02a0c79cf1f9e2d91d0ec5bae2fa0c1436ed048
A        cecb3555daa7a4cc0840db154086138dae9260795bde2b64e89f6724d80abb9b
B        9521bee7fe7a6964d4ff61605bbe5a02eb4cfd149fd5aa9cbf975accbfe40314
combined a42268e4c0d1afb2b9576dd968d7f6250fc989fe3b97077a55363e63e1e2bb8a
```

All four equal `artifacts/c04/a0t_small_v1_impl_v6/freeze/prompt_hashes.json`
and the in-module fixture literals. **No prompt byte changed.**

### Version-token-normalized tree diff — every residual line accounted for

Both trees copied to the scratchpad, `s/v6|V6/vTOK/` and `s/v7|V7/vTOK/`
applied, diffed file by file:

| file | changed lines | accounted for by |
|---|---|---|
| **all 5 schemas** | **0** | — (so the canonical schema's `maximum: 1` is byte-identical to v6) |
| `*_preflight.sh`, `*_preflight.sbatch`, `*.sbatch`, `*_reconcile.sbatch` | 0 | — |
| `*_reconcile.sh` | 8 | H-A only (`campaign-record` before `reconcile-terminal`) |
| `*_v7.sh` | 18 | I-3 only (exit-40 branch, breach-record `jq`, zero-exit breach guard) |
| `preflight.py` | 33 | I-3 only |
| `gpu_ledger.py` | 172 | I-3 / H-A / C-A / H-2 only |
| `common.py` | 813 | C-1 + I-1 + I-2 + I-3, **pure additions** |
| `producer.py` | 497 | C-1 + I-1 + I-3 |
| `config.json` | 137 | version tokens, `v7_scope` prose, three `resources` keys, two `paths`, refreshed `implementation_hashes` |

A function-level AST diff makes this exact:

- `common.py`: **33 additions, 0 removals, exactly one changed definition**
  (`self_test_fixtures`).
- `gpu_ledger.py`: 2 additions (`campaign_record`, `record_campaign_gpu_spend`),
  5 changed (`main`, `reconcile_terminal`,
  `validate_cpu_reconciliation_environment`, `validate_gpu_environment`,
  `verify_reconciliation_lineage`).
- `producer.py`: 5 additions (`BudgetGuard`, `BudgetDeadlineReached`,
  `BUDGET_BREACH_EXIT_CODE`, `assert_teacher_visible_precondition`,
  `publish_budget_breach_record`), **1 removal (`cosine`, moved to `common.py`)**,
  6 changed (`build_messages`, `deadline_check`, `main`, `verify_authorization`,
  `verify_claimed_resource`, `verify_execution_lineage`).
- `preflight.py`: 1 changed (`verify_static_config`).

Nothing else moved. `SYSTEM_PROMPT`, `_SCHEMA_TEXT`, `PROMPT_A`, `PROMPT_B`,
`SELECT_TAG`, `SELECT_SUFFIX`, `SELECT_N`, `NUM_FRAMES`, the transcript
constants, the confidence/cosine thresholds, the five-rate KILL taxonomy,
`build_slot_reliability`, `render_slot`, `materialize_role_map`,
`dense_rademacher_payload`, `parse_teacher_response`, `q_product`,
`safe_vector`, `merkle_root` are untouched — none appears in the added, removed
or changed sets.

`self_test_fixtures()` now returns **47** checks (round 2: 45; round 1: 37;
v6: 25). All 47 pass on the frozen bytes, and no fixture name is duplicated (a
duplicate would be silently dropped by `dict(self_test_fixtures())` in
`preflight.run_self_tests`).

---

## Claim C-1 — the prompt renderer: **CONFIRMED**

1. **Could the v6 form ever have succeeded?** No. Both `PROMPT_A.format(transcript="x")`
   and `PROMPT_B.format(...)` raise `KeyError: '"source_relation"'` against the
   frozen templates; the defect is unconditional on both forms.
2. **Exact substitution?** Yes: `render_prompt(form, t) == PROMPTS[form][:-len("{transcript}")] + t`
   for both forms, including transcripts containing CJK, newlines and literal
   `{braces}`; the module templates are not mutated.
3. **Any prompt byte changed?** No — the four hashes match the v6 frozen artifact.
4. **Any surviving `.format(transcript=` call site?** None. The two textual
   occurrences are a docstring (`common.py:221`) and the deliberate regression
   fixture `prompt_render_regression_str_format_would_raise` (`common.py:2262`),
   which asserts the v6 form still raises `KeyError`.
5. **Guards non-vacuous?** The four pre-substitution guards (unknown form,
   non-`str` transcript, `count != 1`, non-terminal placeholder) are genuine and
   two are exercised by `prompt_render_rejects_unknown_form_and_non_string`. The
   two post-hoc guards remain weak (`startswith(prefix)` is a tautology given a
   unique terminal placeholder; `endswith(transcript)` is vacuous for an empty
   transcript). Unchanged from rounds 1-2; still an observation.

---

## Claim I-1 — teacher-visible containment: **CONFIRMED**

1. **Before model load?** Yes. `assert_teacher_visible_precondition(inputs)` at
   `producer.py:1677`; `from_pretrained` at `producer.py:1686`. Repeated per item
   inside `one_forward` at `producer.py:1709`, after `build_messages` and before
   `apply_chat_template`/`generate`.
2. **Strict in both directions?** Yes. `teacher_visible_texts` raises on a wrong
   message-list length, unexpected message keys, an unexpected role, non-list or
   empty content, a content part that is not a dict or lacks `type`, a text part
   with extra keys or a non-string body, a video part with extra keys, a frame
   list whose length ≠ 8, any frame that is `str`/`bytes`/`bytearray`/`Path`, any
   unknown content type, and a final census requiring exactly one video part and
   exactly two text parts. Six are pinned by fixtures.
3. **What is banned, is it wide enough?** Measured on the real tranche: **402**
   tokens — all 200 HateMM + all 200 MHC-ZH selected identifiers plus
   `hate_video_` and `non_hate_video_` — with both datasets' identifiers banned
   in both datasets' prompts, so a cross-item leak is refused as firmly as a
   self-leak. Matching is over `{token, NFKC(token), casefold(NFKC(token))}`
   against both `NFKC(text)` and `casefold(NFKC(text))`. The wider protection is
   the equality `texts == [SYSTEM_PROMPT, render_prompt(form, transcript)]`: the
   only variable content reaching the teacher is the transcript, so the
   amendment's broader ban (prediction, neighbor, rank, margin, error status,
   dataset statistic, fold role, intended use) is satisfied structurally rather
   than by enumeration.
4. **False positives on the real 400 transcripts?** **None — 800 accepted, 0
   rejected.** As a margin, the 402-token ban list was scanned against **all
   1323** train transcripts (744 + 579): **0** rows contain any banned token.
   Shortest banned token is 11 characters, so accidental substring collision is
   not a live risk.
5. **Can it pass vacuously?** No, at the per-forward site. Measured: an empty ban
   list is rejected, and an identifier absent from the list is rejected.

### The HateMM ID-label asymmetry — stated explicitly, as requested

**The asymmetry is handled correctly.** Every HateMM training identifier is
`hate_video_*` or `non_hate_video_*`, so the identifier *is* the binary label;
MHC-ZH identifiers are opaque BiliBili `BV` codes carrying no label information.
Therefore **the sealed ID-only allowlist delivers label containment for MHC-ZH
only, and none at all for HateMM.** `LABEL_BEARING_ID_SUBSTRINGS` encodes exactly
this (`("hate_video_", "non_hate_video_")` for HateMM, `()` for `MHC_zh`), the
`common.py` comment block states it, and `config.json →
v7_scope.I1_teacher_visible_containment` records it. The consequence is drawn
correctly: for HateMM, selection label-blindness rests on *hash reproduction of
the selection rule* (independently reproduced above, 200/200 on both datasets)
and teacher label-blindness on this *runtime* check — not on the allowlist.

---

## Claim I-2 — the selection self-test is a known-answer vector: **CONFIRMED**

Both pinned digests recomputed from first principles outside the module by hand
concatenation, and both match:

```
sha256(utf8("C04-A0T-SMALL-v1" + "HateMM" + "c04-known-answer-vector" + "20260729"))
  = 871e0363e1b01a823f09a5a0bb9187749da74f1dfe8e454a733e21c218f6a384
sha256(utf8("C04-A0T-SMALL-v1" + "MHC_zh" + "c04-known-answer-vector" + "20260729"))
  = 41bfb637f5cbb26bb0b9edfd44d19a3775d8df1712f9b49bde667863fbd37134
```

Literals independent of the module's own code path. `selection_dataset_and_id_sensitivity`
additionally requires three distinct digests. The identifier is synthetic and
belongs to neither dataset, so the fixture pins the rule without naming a
label-bearing real id.

---

## Claim I-3 — both ceilings machine-checked and fail-closed: **SUBSTANTIALLY DELIVERED**

### Tranche ceiling (7200 s) — guard leads the wrapper, and the seal phase is now guarded

`BudgetGuard.at_job_start` is constructed exactly once, in
`verify_claimed_resource`; the deadline is stored and never recomputed. There are
exactly two `deadline_check` sites — `producer.py:1758` at the item boundary and
`producer.py:1704` inside `one_forward` — both strictly *before* a unit of work,
plus one `require_remaining` at `producer.py:1822` before the canonicalization
phase. The guard never truncates, shortens or rewrites an output;
`BudgetDeadlineReached` is caught only by the outer handler, which publishes an
accounting-only record and returns 40.

I executed the frozen `BudgetGuard` class (extracted with `ast`, never imported
from the repository path) over a grid of claim durations `c` and
lineage/model-hash durations `v`, with the ticket watchdog at
`cap − reserve = 7080 s`:

| claim `c` | verify `v` | guard fires @entry+ | wrapper timeout @entry+ | guard lead | seal phase may start until entry+ | seal budget before SIGTERM |
|---|---|---|---|---|---|---|
| 0 | 0…120 | 6780 | 7080 | **300 s** | 6180 | **900 s** |
| 5 | 0…120 | 6775 | 7080 | **305 s** | 6175 | **905 s** |
| 30 | 0…120 | 6750 | 7080 | **330 s** | 6150 | **930 s** |
| 60 | 0…120 | 6720 | 7080 | **360 s** | 6120 | **960 s** |
| 120 | 0…120 | 6660 | 7080 | **420 s** | 6060 | **1020 s** |

The lead is `guard_item_margin_seconds + c` and is **independent of `v`**,
because `at_job_start` subtracts elapsed-since-entry from a watchdog that already
had `c` subtracted. Round-1 H-B is closed. The degenerate cases all halt:
allocation entry in the future, no budget remaining, margin `0` or `≥ watchdog`,
seal reserve `0` or `≥ watchdog`.

**Round-2 I-2 is closed.** The post-loop phase is no longer unguarded, and its
effective budget is ~900 s rather than the ~300 s round 2 measured. The reserve's
size is still a human estimate rather than a measurement, but the CPU half is
cheap enough to make 900 s plausible: I built a **full** canonical record exactly
as `canonicalize_dataset` builds it (25.8 KB, real `q_product`/`f32le_b64`/
`apply_role`/`fixed_projection` shapes) and Draft7 validation costs **2.4 ms**,
so 400 records cost ≈1 s and the post-publication re-verification's ~2400
validations ≈6 s. The dominant term is the ~7200 small embedding lookups, not the
CPU work. Recorded as an observation, not a finding.

The breach record was inspected field by field: lineage, job id, terminal state,
exit code, both caps, the guard snapshot, per-dataset completed counts, teacher
and frame-pack counters, `outputs_truncated_or_altered: 0`,
`seal_published: false`, `no_scientific_verdict_is_published_by_a_budget_breach: true`.
No metric, no teacher output, no reliability rate, no CONTINUE/KILL verdict. The
wrapper's exit-40 branch `jq -e`s exactly the three fields the record carries,
and a zero-exit run that left a breach record is refused with exit 3.

### Campaign ceiling — read side fail-closed and phase-scoped

`assert_campaign_aggregate_headroom` is called from `validate_gpu_environment`,
the first statement of `claim()`, i.e. before `create_entry_marker`, before
`verify_gpu_lineage`, and well before the allocation claim and the ticket
consumption record are published — genuinely before the single-use ticket is
consumed. It is also called by the CPU preflight before the namespace is
materialized, and by the producer before any model or data work.

Fail-closed matrix, executed against a sandbox copy (repository untouched):

| mutation | result |
|---|---|
| pristine genesis (0 s spent) | **accepted** (7200 ≤ 7200) |
| ledger absent | HALT `campaign ledger is absent` |
| `payload_sha256` tampered | HALT `payload mismatch` |
| aggregate ≠ Σ rows | HALT `aggregate does not equal its rows` |
| foreign `schema_version` | HALT `foreign campaign ledger schema` |
| foreign `run_id` | HALT `foreign campaign ledger run id` |
| aggregate cap raised to 999999 | HALT `cap is not the amendment cap` |
| aggregate cap lowered to 7200 | HALT `cap is not the amendment cap` |
| phase advanced, phase cap left at 7200 | HALT `phase cap does not match the phase` |
| phase cap raised alone | HALT `phase cap does not match the phase` |
| unknown phase | HALT `unknown campaign phase` |
| first-tranche phase carries an advance token | HALT `first-tranche phase carries an advance token` |
| row chain break | HALT `chain break` |
| well-formed, 1 s already spent | loads; next 7200 s reservation **REFUSED** |
| well-formed, 7000 s already spent | loads; next 7200 s reservation **REFUSED** |
| well-formed, 7250 s (over-cap) already spent | loads; next 7200 s reservation **REFUSED** |
| non-positive reservation | HALT `requested a non-positive reservation` |

**Effective ceiling is 7200 s today.** The only route to 28800 s is a coordinated
hand edit that changes `phase`, `phase_cap_gpu_seconds` and
`phase_advance_authorization` together and reseals the payload — i.e. the human
gate the amendment intends. No code path writes any of the three. Round-1 I-C1
stays closed.

**Opening zero is evidence-backed.** I read `sacct` myself, read-only:

```
13805|c04_a0t_small_v1_v5_preflight|0 |billing=8,cpu=8,mem=64G,node=1|FAILED
13840|c04_a0t_small_v1_v6_preflight|19|billing=8,cpu=8,mem=64G,node=1|COMPLETED
```

Both rows match `genesis_evidence` verbatim including `alloc_tres`, elapsed
seconds and states; neither carries `gres/gpu`, so `gpu_seconds: 0` is correct
for both. A full accounting sweep for `c04` job names returns exactly these two
rows, corroborating `these_are_the_only_c04_jobs_in_the_accounting_record: true`.

### Write side — round-2 H-1 closed (the accumulator can no longer brick itself)

Executed against a sandbox ledger at five magnitudes:

| appended `gpu_seconds` | append raised | on-disk aggregate | over-cap flag | later `load` | next 7200 s reservation | idempotent re-verify | chained 2nd append |
|---|---|---|---|---|---|---|---|
| 0 | none | 0 | `false` | OK | accepted | OK | OK |
| 100 | none | 100 | `false` | OK | REFUSED | OK | OK |
| 7200 | none | 7200 | `false` | OK | REFUSED | OK | OK |
| **7250** | **none** | **7250** | **`true`** | **OK** | **REFUSED** | **OK** | **OK** |
| 30000 | none | 30000 | `true` | OK | REFUSED | OK | OK |

I also traced every check in `load_campaign_gpu_ledger` against what
`append_campaign_gpu_job` writes and confirmed **no check can fail on the
just-written file**: payload digest, schema, run id, aggregate cap, phase,
phase cap, advance token, chain links, row digests, aggregate-equals-rows and
head link are all satisfied by construction, and the cap check has been removed
from the load path entirely. Every rejecting check (head race, duplicate job,
non-integer seconds) runs before the write, and the new
`aggregate_exceeds_effective_cap` flag and both `campaign_effective_cap` calls
are evaluated before `os.replace`. The `load_campaign_gpu_ledger()` on the return
line is therefore genuinely non-raising, and the designed over-cap branch is no
longer dead code.

### Write side — round-1 H-A and round-2 I-3 closed

| path | `allocation_claim.json` | `allocation_entry_marker.json` | campaign row written? |
|---|---|---|---|
| exit 40 budget breach | yes | yes | **yes** |
| watchdog TERM/KILL (124/137/143) | yes | yes | **yes** |
| OOM / decode failure / any producer HALT after claim | yes | yes | **yes** |
| fully successful sealed run | yes | yes | **yes** (and again idempotently from `reconcile_terminal`) |
| HALT *before* `claim()` publishes the claim | no | **yes** (wrapper writes it at `c04_a0t_small_v1_v7.sh:39-64`, before the first Python call) | **yes, via the fallback** |
| allocation never entered | no | no | **no** — prints "no allocation entry; nothing to record", returns 0 |

The fallback cannot fabricate a row: with neither artifact present it records
nothing, and with the marker present it still requires `sacct` to show a terminal
row whose `gres/gpu` count is exactly 1. `record_campaign_gpu_spend` verifies an
already-present row instead of appending, so the two calls in one reconcile
wrapper run (once from `campaign-record`, once from `reconcile-terminal`) cannot
double-count.

---

## Additional checks

- **`--time` directive:** absent from all three sbatch files and all three
  wrappers; each sbatch carries an explicit comment that the omission is
  deliberate.
- **Arrays / dependencies / chained submission / release / resubmission:**
  absent. No `sbatch`, `scontrol`, `scancel`, `srun` or `salloc` token anywhere
  in the v7 set. The only `subprocess` call in the entire tree is
  `gpu_ledger.py:241`, `sacct -X -n -P -j <id> -o JobIDRaw,ElapsedRaw,AllocTRES,State`
  — read-only. All three wrappers and both Python entrypoints reject
  `SLURM_ARRAY_JOB_ID` / `SLURM_JOB_DEPENDENCY`.
- **`--gres`:** exactly one occurrence, `scripts/slurm/c04_a0t_small_v1_v7.sbatch:3`,
  `--gres=gpu:a100:1`. The preflight sbatch requests no GPU; the reconcile sbatch
  requests no GPU and its wrapper additionally rejects a non-empty
  `CUDA_VISIBLE_DEVICES` or `SLURM_GPUS_ON_NODE`.
- **Resources:** GPU sbatch = **1 GPU / 8 CPU / 64 GB** exactly; preflight = 8
  CPU / 64 GB, no GPU; reconcile = 1 CPU / 4 GB, no GPU. `resources.gpu_count/cpus/ram_gb`
  = 1/8/64 and are asserted in the preflight, the GPU ledger and the producer.
- **No OCR entrypoint, no network/API client, no dev/test path, no cross-dataset
  path, no label reader:** confirmed by targeted grep for `requests`, `urllib`,
  `httpx`, `aiohttp`, `socket`, `boto3`, `tesseract`, `easyocr`, `paddleocr`,
  `pytesseract` — the only textual hit is the word "requests" inside a
  frame-decode docstring. The only `label` reference on the data path is
  `_skip_json_value`, which advances the parser past the token syntactically and
  increments a skip counter; the projector then requires the decoded key set to
  be exactly `{id, window_text, language}`. `HF_HUB_OFFLINE=1` /
  `TRANSFORMERS_OFFLINE=1` are exported by the wrapper and asserted by the
  producer, and both `from_pretrained` calls pass `local_files_only=True`.
  `root_path` rejects any `dev`/`test`/`validation`-like path component.
- **Authorization flags in the correct pre-review state:** exactly one flag of
  seventeen is `true` (`implementation_authorized`); all sixteen others —
  including `preflight_materialization_authorized`, `teacher_authorized`,
  `gpu_authorized`, `slurm_authorized`, `small_tranche_execution_authorized` and
  `post_job_reconciliation_authorized` — are `false`. All four review pins are
  sentinels and all four verdicts are `PENDING`. `maps.expected_hashes` is the
  documented sentinel string.
- **Unearned review pins are sentinels the code rejects.** Executed against the
  frozen config:

  | entrypoint | result |
  |---|---|
  | `verify_historical_code_resource_authorization` | HALT `code/resource authorization SHA-256 is unpinned` |
  | `verify_payload_review` | HALT `payload hash verdict is not GO` |
  | `verify_gpu_execution_authorization` | HALT `GPU execution verdict is not GO` |
  | `verify_historical_gpu_execution_authorization` | HALT `authorization SHA-256 is unpinned` |
  | `verify_resource_reconciliation_authorization` | HALT `reconciliation verdict is not GO` |
  | `resolve_prompt_hashes(freeze=False)` | HALT `sentinel … outside the authorized freeze run` |
  | `resolve_prompt_hashes(freeze=True)`, materialization `false` | HALT (same) |
  | `resolve_prompt_hashes(freeze=True)`, materialization `true` | accepted — the single intended relaxation |

- **Config contract normalization is exactly as narrow as documented.** Measured:
  `authorization.*`, the four review pins, the four review verdicts and
  `prompt_hashes.*` do **not** move `config_contract_sha256` (including filling
  all four prompt hashes at once — the v5 impossibility stays closed). Fourteen
  other mutations — `selection.suffix`, all four resource caps/margins,
  `maps.expected_hashes`, `teacher_contract.num_frames`,
  `model.snapshot_revision`, `paths.campaign_gpu_ledger`,
  `run.implementation_version`, `schemas.canonical_record`,
  `reliability.proposition_agreement_cosine_min`,
  `datasets.HateMM.train_asr_sha256` — **do** move it.
- **No live writer/reader mismatch anywhere.** All five `require_exact_keys`
  contracts extracted by `ast` and compared against the dict literal each writer
  builds: GPU ledger 15/15, resource ticket 16/16, allocation claim 12/12,
  provisional GPU usage (shared constant), budget guard (shared constant).
- **No live schema mismatch anywhere.** A **full** canonical record built exactly
  as `canonicalize_dataset` builds it validates against the frozen schema in all
  four reliability regimes — all-`stable` with the clamped agreeing cosine,
  all-`missing`, `single_valid`, and `conflict` — and a full prompt record
  validates for both a normal decode and the zero-frame degenerate case
  (`total_frame_indices: 0`, `requested_indices: []`) at `sequence_index` 399.

---

# Findings

## HIGH H-1 — the GPU wrapper creates the single-shot no-clobber namespace before any authorization gate, so one out-of-order submission forecloses v7 permanently

**Where.** `scripts/wrappers/c04_a0t_small_v1_v7.sh:39-40`
(`readonly ENTRY_MARKER=…/resource/allocation_entry_marker.json` then
`mkdir -p "$(dirname "$ENTRY_MARKER")"`), versus
`scripts/analysis/c04_a0t_small_v1_v7_preflight.py:324-326`
(`namespace = root_path(ARTIFACT_ROOT); if namespace.exists(): raise FileExistsError`).

**Mechanism.** The GPU wrapper's only pre-Python checks are `SLURM_JOB_ID`
non-empty and no array/dependency (lines 31-38). It then unconditionally
`mkdir -p`s `artifacts/c04/a0t_small_v1_impl_v7/resource`, writes the allocation
entry marker, and only *afterwards* invokes `gpu_ledger.py --mode claim`, whose
`validate_gpu_environment` is the first code that ever looks at
`cfg["authorization"]`. **The two sibling wrappers both carry a config gate and
this one carries none** — measured by grep: `…_preflight.sh` has one
`authorization` reference (its `preflight_materialization_authorized` block),
`…_reconcile.sh` has four (a nine-clause `jq -e` requiring
`post_job_reconciliation_authorized`, every other flag false, the reconciliation
verdict `GO` and a 64-hex pin), and `…_v7.sh` has **zero**. Its three `jq -e`
calls are the entry-marker match, the breach-record check and the seal check —
none of them authorization.

**Measured.** I replayed the wrapper's lines 31-64 byte-faithfully against a
sandbox root, with `claim` stubbed:

```
namespace dir now exists : YES
entry marker written     : YES
...and only NOW would 'gpu_ledger.py --mode claim' run its first check.
frozen config says gpu_authorized = false, teacher_authorized = false

preflight.py:325  if namespace.exists(): raise FileExistsError -> namespace.exists() == True
=> CPU preflight would raise: FileExistsError("no-clobber namespace refusal: artifacts/c04/a0t_small_v1_impl_v7")
```

**Failure scenario.** Someone submits `scripts/slurm/c04_a0t_small_v1_v7.sbatch`
before the CPU preflight has run — today's state, with `gpu_authorized: false`
and `preflight_materialization_authorized: false`. SLURM grants an A100. The
wrapper creates the namespace and the marker. `claim` then halts within seconds
(`assert_literal_prompt_hash_binding` fires on the prompt-hash sentinel, and
`authorization.teacher_authorized` would fire immediately after). The EXIT trap
runs `mark-exit`, which finds the marker, stamps it, finds no ledger and
returns. The A100 is released having done nothing. But:

- `artifacts/c04/a0t_small_v1_impl_v7/` now exists, so
  `preflight.preflight()` raises `FileExistsError` on **every** future run. v7
  can never be materialized;
- deleting the directory is not a remedy the design admits — the whole no-clobber
  scheme rests on the namespace being created exactly once, atomically, by the
  preflight's `os.rename`;
- the surviving path is a full namespace rebuild to v8, which is precisely the
  cost this campaign has now paid three times (v5 → v6 → v7).

This is the failure family the request asked me to enumerate, in its purest form:
an irreversible, single-shot resource is consumed by an unconditional `mkdir -p`
**before** the check that would reject the run. It is also the cheapest one to
close and the only one where the correct guard already exists, verbatim, in two
sibling files.

**What would close it.** Put an authorization gate ahead of line 40 in
`scripts/wrappers/c04_a0t_small_v1_v7.sh`, in the same style the reconcile
wrapper already uses — `jq -e` on `.authorization.gpu_authorized`,
`.teacher_authorized`, `.slurm_authorized`, `.small_tranche_execution_authorized`
all `true`, `.review.gpu_execution_verdict == "GO"` and a 64-hex
`gpu_execution_authorization_sha256` — plus a plain `[[ -f
artifacts/c04/a0t_small_v1_impl_v7/freeze/preflight_manifest.json ]]` test, so a
GPU submission that precedes the preflight refuses before it touches the
filesystem. All of it is `exit 2` before any `mkdir`.

## IMPORTANT I-1 — round-1 I-C2 is still open: the wrapper `timeout` is anchored to a different clock than `sacct`, defended by an unmeasured 120 s reserve, and an over-run makes the final state permanently unpublishable

**Where.** `scripts/wrappers/c04_a0t_small_v1_v7.sh:14-15,101-105`
(`/proc/uptime` anchor, `timeout … --kill-after=30s "${C04_ACTIVE_WATCHDOG_SECONDS}s"`)
versus `gpu_ledger.py:1253-1256` (`terminal sacct GPU seconds exceed 7200 cap`),
`gpu_ledger.py:916-923` (`terminal GPU seconds outside [0,7200]`) and
`schemas/c04/c04_a0t_small_v1_v7_resource_final_state.schema.json:63-80`
(`maximum: 7200` on three fields, `cap_gpu_seconds: {"const": 7200}`).

**Mechanism, re-derived.** `claim()` returns `active = 7080 − c`, where `c` is
measured from the wrapper's own `/proc/uptime` read. `timeout` starts after the
claim and therefore fires at `P0 + 7080` where `P0` is the SLURM-prolog-to-wrapper
offset, plus 30 s of `--kill-after`, plus the EXIT-trap `mark-exit` write —
so worst-case wall time ≈ `P0 + 7113`. `sacct` `ElapsedRaw`, however, is measured
from SLURM job start, i.e. from `P0` earlier. Nothing in the code measures or
bounds `P0`; the entire defence is the fixed `watchdog_reserve_seconds = 120`, of
which 30 s is already spoken for by `--kill-after`. Round 2 measured `P0` at
sub-second on job 13840, so the realistic margin is ≈85 s.

**Failure scenario.** If terminal `sacct` elapsed exceeds 7200 s:
`campaign-record` correctly records the true spend (accounting is never refused —
that part is right), but `reconcile_terminal` then raises
`HALT_RESOURCE_CAP: terminal sacct GPU seconds exceed 7200 cap`;
`strict_validate_terminal_ledger` would reject it independently; and the
`resource_final_state` schema pins three fields at `maximum: 7200`, so no final
state can ever be published for that run. The reconcile wrapper's recovery branch
requires the ledger to already be `SACCT_TERMINAL_RECONCILED`, which it never
becomes, so it propagates the failure. `review.downstream_review_requires_terminal_resource_state: true`
then blocks every downstream review of that namespace permanently, and the run
cannot be repaired without editing `gpu_ledger.py`, which moves its SHA-256,
hence `config.implementation_hashes`, hence `config_contract_sha256`, hence the
values already pinned inside the no-clobber preflight manifest, genesis ledger
and resource ticket.

**Why this is Important and no longer High.** The round-2 High that subsumed it
was rated on the brick — an over-cap write making the cross-version accumulator
permanently unloadable. That is genuinely fixed (measured above). What remains is
confined to the namespace. And the in-job guard now stops the producer at
`entry + 6780 − c`, so the wrapper `timeout` fires only if a single item outruns
the 300 s margin or the seal phase outruns ~900 s — a tail, not a certainty. But
it is a hard ceiling whose breach is unrecoverable, defended by a constant nobody
has measured, and it is now three rounds old.

**What would close it.** Either derive the wrapper's `timeout` from
`cap − (measured job-start-to-wrapper-start offset) − reserve` — the offset is a
two-line `sacct -X -n -P -j "$SLURM_JOB_ID" -o Start` or, more simply, a larger
`watchdog_reserve_seconds` chosen so `reserve > kill_after + mark_exit + P0_max`
provably holds — or make an over-cap terminal sacct row a *recorded* over-run:
written to the per-namespace ledger and a distinct final state with the schema
bound relaxed to a flagged maximum, rather than an unrecoverable halt. The
campaign accumulator already demonstrates the second pattern working.

## IMPORTANT I-2 — `reconcile-terminal` is still seal-dependent, so the exit-40 breach path that I-3 introduced cannot complete its own mandated post-job stage

**Where.** `gpu_ledger.py:1188-1191` (`reconcile_terminal` → `verify_reconciliation_lineage`)
and `gpu_ledger.py:448-449`
(`provisional = load_json(cfg["paths"]["provisional_gpu_usage"])`, a file inside
`seal/` that only the producer's final atomic seal publication creates), ordered
by `scripts/wrappers/c04_a0t_small_v1_v7_reconcile.sh:66-83`.

**Mechanism.** Round-1 H-A asked for the terminal accounting stage to be made
independent of the seal. The repair delivered that for the *campaign* accumulator
by adding a separate `campaign-record` mode — which is correct and is why H-A is
closed — but left `reconcile-terminal` itself keyed on the seal. On any terminal
path that produces no seal, `verify_reconciliation_lineage` dies with an
uncaught `FileNotFoundError` on `seal/provisional_gpu_usage.json`. The wrapper's
recovery branch is guarded by `jq -e '.state == "SACCT_TERMINAL_RECONCILED"'`,
which is false (the ledger is still `EXIT_RECORDED_PENDING_SACCT`), so the
wrapper takes `exit "$C04_RECONCILE_STATUS"`.

**Failure scenario.** Every non-sealing terminal outcome:

- **exit 40 budget breach** — the path I-3 exists to create, and which the
  wrapper documents as "a terminal state" with "an accounting-only breach
  record";
- **watchdog TERM/KILL (124/137/143)**;
- **OOM, decode failure, or any producer HALT after `claim()`**.

In each, the campaign row is written correctly, and then the per-namespace GPU
ledger is left at `EXIT_RECORDED_PENDING_SACCT` holding a 7200 s reservation that
is never replaced by real sacct seconds, `resource_final_state.json` is never
published, and the mandated `CPU_POST_JOB_RECONCILIATION` job exits non-zero.
Because the campaign accumulator is now independent and correct, the cross-version
damage is nil — which is why this is Important and not High — but the design
intent of exit 40 ("a clean, accounting-complete terminal state, distinct from an
engineering failure") is not delivered end to end, and the namespace's resource
story is left permanently unfinalized on precisely the path the ceiling exists to
produce.

**What would close it.** Split `verify_reconciliation_lineage` into the parts
that need only the claim/marker/ledger (job identity, lineage chain,
reconciliation authorization) and the parts that need the seal, and make the seal
half conditional on `seal/seal_manifest.json` existing — exactly the
"conditional refinements" phrasing round 1 used. `reconcile_terminal` should then
reconcile the ledger from sacct and publish a `resource_final_state` on every
terminal path, carrying an explicit `seal_published: false` (and, where present,
`budget_breach_sha256`) so the absence of science is recorded rather than
inferred from a missing file.

## IMPORTANT I-3 — the preflight round-trip is narrowed a second time but two channels remain open, one of them with post-GPU blast radius

This subsumes round-1 I-C3, round-2 H-2's residual and round-2 I-1's wider half.
There is **no live defect today** — I measured that — so this is a recurrence
channel, which is why it is Important.

**(a) No full producer record is ever round-tripped before the GPU is spent.**
`downstream_contract_fixtures()` is real work and closes the specific hole that
produced round-1 C-B: it builds a `provisional_gpu_usage` record through the
production builder, and it validates `build_slot_reliability` output against the
*actual* `reliability` definition of the frozen canonical schema in six states.
And the C-B clamp is now genuinely protected — deleting
`max(-1.0, min(1.0, …))` from `common.cosine` turns
`cosine_of_identical_vectors_is_within_the_schema_bound` **red**, because two of
that fixture's three pinned vectors (`[1/3]*3584` and `[0.7,-0.3,0.2]*8`)
overshoot to exactly `1.0000000000000002` unclamped. Measured:

```
FROZEN bytes : 47 fixtures, failing: []
CLAMP DELETED: 47 fixtures, failing: ['cosine_of_identical_vectors_is_within_the_schema_bound']
```

But no fixture yet builds a **full** `canonical_record`, `prompt_record`,
`frame_pack_manifest` or `resource_final_state` and validates it against the
contract its consumer applies. `validate_schema` at preflight is exercised only
against the stage-authorization manifest. The full canonical record is first
schema-validated at `producer.py:1402`, i.e. **only after all 800 forwards are
paid for**; the full `resource_final_state` only at `gpu_ledger.py:1075`, after
the whole allocation. I closed the question of whether that exposure is live by
building both records exactly as the producer builds them: all four canonical
regimes and both prompt-record cases validate. The structural exposure is
unchanged; only its known triggers are closed.

**(b) Three writer/reader key-set contracts are still duplicated literals.**
Round-2 H-2 is closed for the two constants it named — `gpu_ledger.py` now
imports `PROVISIONAL_USAGE_KEYS` and `BUDGET_GUARD_KEYS` and passes them to
`require_exact_keys`, so one-sided drift is impossible by construction there.
The other three, from the same round-2 census, were not converted:

| contract | writer | reader | reader's set | agree today | when drift would surface |
|---|---|---|---|---|---|
| GPU ledger (15) | `preflight.py:414` | `gpu_ledger.py:270` | **literal** | yes | at `claim()` — GPU allocation entered, namespace poisoned |
| resource ticket (16) | `preflight.py:435` | `gpu_ledger.py:710` | **literal** | yes | at `claim()`, before ticket consumption |
| **allocation claim (12)** | `gpu_ledger.py:761` | `gpu_ledger.py:372` | **literal** | yes | **at `CPU_POST_JOB_RECONCILIATION` — after the full 2 GPU-hour spend** |

The third is the round-1 C-A shape verbatim: the writer is in `claim()` and the
reader is in `verify_reconciliation_lineage`, 390 lines apart in the same file,
and a one-sided edit would not be observable until the A100 was gone.

**What would close it.** (a) Add preflight fixtures that build one full
`canonical_record` — in the degenerate regimes: all `missing`, all
`single_valid`, and the agreeing `stable` case with `proposition_cosine` at and
just above 1.0 — one full `prompt_record` including the zero-frame case, and one
`resource_final_state`, and run each through `validate_schema` against the frozen
schema its consumer uses. My scratchpad harness shows this is ~40 lines and
2.4 ms per record. (b) Promote the remaining three key sets to named constants in
`common.py` and import them on both sides, exactly as was just done for the other
two; a fixture that only compares a constant with itself cannot substitute for
this, as round 2 demonstrated.

---

## Non-blocking observations

1. **The pre-model-load containment pass is half self-comparing.** `texts` is
   derived from the same `render_prompt` call the assertion compares against, so
   before model load the pass establishes "no banned token in any of the 400
   transcripts, and the assembled message shape is legal". The message-assembly
   half becomes non-vacuous only at `producer.py:1709`, which does run before
   every forward, so the stated requirement is met. Unchanged from rounds 1-2.
2. **`maps.expected_hashes` is protected only by inclusion in the contract hash.**
   Mutating it moves `config_contract_sha256` (verified), but no code asserts its
   literal value, so the prose invariant in `prompt_hash_contract` is not
   machine-checked. Unchanged from rounds 1-2.
3. **The campaign ledger's `phase` is not cryptographically bound to any
   authorization artifact.** No code advances it (verified), but an advanced
   ledger is indistinguishable from an authorized one. Unchanged from round 2.
4. **`append_campaign_gpu_job` takes no lock on the campaign file.** It is
   protected only by the *per-namespace* `resource/gpu_ledger.lock`, which would
   not exclude a future namespace's reconciler; the head-hash race check turns a
   collision into a halt rather than corruption, but a lock on the campaign path
   itself would close it. Unchanged from round 2.
5. **`campaign_record`'s marker-fallback branch validates nothing about the
   marker.** The claim branch verifies `claim_sha256`; the marker branch reads
   `["slurm_job_id"]` with no `schema_version`, `run_id` or self-hash check,
   unlike `verify_reconciliation_lineage:427-429`. The marker is written only by
   the wrapper and by `create_entry_marker`, and `sacct` must still show a
   terminal one-GPU row, so the risk is low — but the asymmetry is gratuitous.
6. **`BudgetGuard.at_job_start` checks `item_margin` and `seal_reserve`
   individually but not their sum.** `margin=300, seal_reserve=7000` builds a
   guard that can never let the seal phase start. Not live (300 + 600 = 900 ≪
   7080), but the sanity check is one-sided.
7. **The exit-40 wrapper branch runs `jq -e` under `set -e`.** A malformed or
   absent breach record surfaces as exit 1 rather than exit 40, losing the
   distinct code the branch exists to propagate. Fail-closed, but the
   distinctness is lost. Unchanged from round 2.
8. **The seal phase's 600 s reserve is still an estimate, not a measurement.**
   Its effective budget is ~900 s (reserve + item margin), and the CPU half is
   cheap (2.4 ms per canonical-record validation, ≈1 s for 400; ≈6 s for the
   ~2400 post-publication re-validations), so 900 s is plausible — but if it is
   exceeded the outcome is still exit 124 with no seal and no breach record.

---

## Summary table

| # | Severity | Finding |
|---|---|---|
| H-1 | High | `scripts/wrappers/c04_a0t_small_v1_v7.sh:40` `mkdir -p`s the single-shot no-clobber namespace and writes the entry marker before **any** authorization gate — a gate both sibling wrappers carry and this one does not. One out-of-order or unauthorized GPU submission makes the CPU preflight's `no-clobber namespace refusal` permanent and forces a full v8 rebuild. Demonstrated end to end. |
| I-1 | Important | Round-1 I-C2 untouched for a third round: the wrapper `timeout` is anchored to `/proc/uptime` while `sacct` measures from job start, defended by an unmeasured 120 s reserve of which 30 s is `--kill-after`; an over-7200 s terminal elapsed makes `reconcile_terminal` and the `resource_final_state` schema (`maximum: 7200`) permanently unsatisfiable. Blast radius reduced by the H-1 fix and the in-job guard, so no longer High. |
| I-2 | Important | `reconcile-terminal` is still seal-dependent, so exit 40, exit 124/137/143, OOM and any post-claim HALT cannot complete the mandated post-job stage: `resource_final_state.json` is never published and the namespace ledger keeps a 7200 s reservation forever. The campaign accumulator is independent and correct, so cross-version damage is nil. |
| I-3 | Important | The preflight blind spot is narrowed twice but two channels remain: no full `canonical_record`/`prompt_record`/`resource_final_state` is ever round-tripped before the GPU is spent (measured: no live mismatch), and three of five writer/reader key-set contracts are still duplicated literals (measured: all agree today), one of whose readers runs only after the full 2 GPU-hour spend. |

**Verdict: REVISE (0C / 1H / 3I). No execution authority is conferred.
`authorization.preflight_materialization_authorized` must remain `false` and
`review.code_resource_verdict` must remain `PENDING`. H-1 should be closed before
any GPU submission is even possible, because its trigger is a submission made in
exactly the config state the repository is in today, and its cost is the entire
v7 namespace.**
