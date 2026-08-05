# C04-A0T-SMALL-v1 v7 — Fresh Independent Code/Resource Review, ROUND 2

Reviewer: fresh independent static reviewer (no exposure to the authoring reasoning)
Date: 2026-07-31
Stage reviewed: `CPU_PREFLIGHT` code/resource review of implementation-v7, second revision
Predecessor: `refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW.md` (`REVISE 2C/2H/3I`), left intact
Execution authority conferred by this review: **none**

## Verdict

**REVISE — 0 Critical / 2 High / 3 Important (0C / 2H / 3I)**

Both round-1 Criticals and both round-1 Highs are genuinely repaired at the level
of what this run would do: I reproduced each defect's trigger and confirmed the
frozen bytes no longer fail. `I-C1` is fully closed. Claims 0, C-1, I-1 and I-2
all reproduced exactly.

The two Highs below are not soft. **H-1** is the same failure family the request
named — an irreversible resource written before the check that rejects it — and
the I-3 phase-cap repair *created* it: `campaign-record` now writes an unfiltered
sacct row into the one artifact designed to outlive every implementation
namespace, before the stage that would refuse that value runs, and an over-cap
write makes the campaign accumulator permanently unloadable for all future C04
work. **H-2** is a repair claim that is not delivered: the round-2 summary states
that the two halves of the `provisional_gpu_usage` contract "build and validate
through single shared constants, and a fixture asserts they agree." Neither is
true. The reader still carries a hand-maintained literal, and I reproduced the
exact v7-first-draft defect with all 45 preflight fixtures still green.

## Method and reviewer-boundary compliance

- No SLURM job submitted, held, released, requeued or cancelled. `sacct` and
  `squeue` used read-only only.
- No GPU, teacher, model-weight or frame-decode work. No model file was opened.
- No file under `/data/jehc223/RGCL` was created, modified or deleted other than
  this review file. `artifacts/c04/campaign/gpu_ledger.json` re-hashes to
  `fc6ca12c32427625d0b80c16b7802ef9a574ced0dbf0288edc3938d217267414` after every
  write-capable probe (all mutation experiments ran against a scratchpad copy
  with `common.ROOT` repointed outside the repository).
- `artifacts/c04/a0t_small_v1_impl_v7/` does not exist and was not created.
- No dataset label value was materialized. Both ASR files were read only through
  the frozen `project_train_asr_line` projector (`id`, `window_text`,
  `language`). HateMM identifiers were hashed and compared, never printed and
  never reasoned from as labels.
- All work in `…/scratchpad/review-r2`, outside the repository, with
  `PYTHONDONTWRITEBYTECODE=1` on every invocation. `common.py` was imported from
  a byte-identical scratchpad copy; `BudgetGuard` was executed by extracting its
  frozen `ClassDef` with `ast`, never by importing `producer.py` from the
  repository path. `py_compile` was not used.

### Hash verification (all 17, before and after)

Every pinned SHA-256 in the request table matches disk exactly, re-verified at
the end of the session. In addition:

- all 15 `configs/c04/c04_a0t_small_v1_v7.json → implementation_hashes` verify
  against disk (15/15) and equal the request table;
- all 15 `frozen_design_hashes` verify against disk (15/15).

### v6 predecessor unmodified

- `artifacts/c04/a0t_small_v1_impl_v6/freeze/preflight_manifest.json` is
  self-consistent (`payload_sha256` reproduces) and **all 14** of its
  `staged_output_hashes` verify byte-for-byte on disk.
- all 15 entries of `configs/c04/c04_a0t_small_v1_v6.json → implementation_hashes`
  verify against disk.
- every v6 artifact carries mtime `2026-07-31 05:11:50` (job 13840), predating
  every v7 source file.

---

## Round-1 findings — closure status

| # | Round-1 finding | Status | Evidence |
|---|---|---|---|
| C-A | reconciler's exact-key set had not grown with the writer | **CLOSED (live defect)** — see **H-2** for the unrepaired recurrence channel | reader set is now 14 keys (v6: 11) and matches the writer exactly; all **five** writer/reader pairs agree |
| C-B | `proposition_cosine` could exceed the schema `maximum: 1` | **CLOSED** | clamp present in `cosine()`; overshoot re-measured at 572/2000; no comparison outcome changes |
| H-A | campaign write side reachable only after a seal | **CLOSED for every named path** — residual gap in **I-3** | new `campaign-record` mode keyed on `allocation_claim.json`, run first in the reconcile wrapper |
| H-B | in-job guard had no margin over the wrapper `timeout` | **CLOSED** | measured lead = `300 + claim_duration` s, always ≥ 300 s; `SLURM_JOB_START_TIME` gone |
| I-C1 | accumulator enforced 28800 s, not the binding 7200 s | **CLOSED** | effective cap = `min(phase, aggregate)` = 7200 s today; no code path advances the phase |
| I-C2 | ~90 s margin to the hard 7200 s ceiling; breach unrecoverable | **NOT CLOSED, blast radius now larger** | see **H-1** |
| I-C3 | preflight never round-trips a record against a downstream contract | **PARTIALLY CLOSED** | see **I-1** |

Both round-1 non-blocking observations were re-checked:

- The pre-model-load containment pass now assembles through `build_messages` and
  extracts through `teacher_visible_texts`, i.e. the same path as the per-forward
  call site (`producer.py:1636-1646`). **The half that compares `texts` against
  `[SYSTEM_PROMPT, render_prompt(form, transcript)]` still self-compares there**,
  because `texts` is derived from the same `render_prompt` call; what the pass
  genuinely establishes before model load is "no banned token in any of the 400
  transcripts, and the assembled message shape is legal". The message-assembly
  half becomes non-vacuous only at `producer.py:1702`, which does run before
  every forward, so the stated requirement is met.
- `maps.expected_hashes` remains protected only by inclusion in the contract
  hash (verified: mutating it moves `config_contract_sha256`). No code asserts
  its literal value. Unchanged from round 1, still non-blocking.

---

## Claim 0 — no scientific semantic changed: **CONFIRMED**

### Selection re-derived from the v7 frozen rule reproduces the v6 frozen allowlists

Both train ASR files were read through the v7 projector, ranked by the v7
`selection_digest` with the v7 tie-break, first 200 taken:

| dataset | train N | recomputed order == v6 frozen allowlist | sha256(newline-joined ordered id list) |
|---|---|---|---|
| HateMM | 744 | **True** | `091fb1826cbc7f80…` |
| MHC_zh | 579 | **True** | `6c98c0d75891ce43…` |

Also reproduced, per dataset: every `selection_sha256` stored in the v6
allowlist (200/200), and — against the v6 `source_manifest.json` — all 200
`transcript_sha256` and all 200 `transcript_scalar_count` values. That last
check independently pins the transcript normalization, cap, head/tail split and
separator as unchanged.

### Prompt hashes

Recomputed from the v7 sources:

```
system   1ffc06755b18f9315dc930aaf02a0c79cf1f9e2d91d0ec5bae2fa0c1436ed048
A        cecb3555daa7a4cc0840db154086138dae9260795bde2b64e89f6724d80abb9b
B        9521bee7fe7a6964d4ff61605bbe5a02eb4cfd149fd5aa9cbf975accbfe40314
combined a42268e4c0d1afb2b9576dd968d7f6250fc989fe3b97077a55363e63e1e2bb8a
```

Equal to the in-module fixture literals **and** to
`artifacts/c04/a0t_small_v1_impl_v6/freeze/prompt_hashes.json`. No prompt byte
changed.

### Version-token-normalized tree diff — every residual line accounted for

Both trees copied to the scratchpad, `s/v6|V6/vTOK/` and `s/v7|V7/vTOK/`
applied, diffed file by file:

| file | changed lines | accounted for by |
|---|---|---|
| **all 5 schemas** | **0** | — (so the canonical schema's `maximum: 1` is byte-identical to v6) |
| `*_preflight.sh`, `*_preflight.sbatch`, `*.sbatch`, `*_reconcile.sbatch` | 0 | — |
| `*_reconcile.sh` | 8 | H-A (`campaign-record` before `reconcile-terminal`) |
| `*_v7.sh` | 18 | I-3 (exit-40 branch, breach-record `jq`, zero-exit breach guard) |
| `preflight.py` | 30 | I-3 (campaign cap/path/margin/phase assertions + headroom call) |
| `gpu_ledger.py` | 145 | I-3 / H-A / C-A only |
| `common.py` | 769 | C-1 + I-1 + I-2 + I-3, **pure additions** |
| `producer.py` | 472 | C-1 + I-1 + I-3 |
| `config.json` | 135 | version tokens, `v7_scope` prose, three new `resources` keys, two new `paths`, refreshed `implementation_hashes` |

A function-level AST diff makes this exact. `common.py`: **32 additions, 0
removals, exactly one changed definition** (`self_test_fixtures`, the I-2 fixture
swap plus the new fixture groups). `producer.py`: 5 additions
(`BudgetGuard`, `BudgetDeadlineReached`, `BUDGET_BREACH_EXIT_CODE`,
`assert_teacher_visible_precondition`, `publish_budget_breach_record`) and 7
changed definitions (`build_messages` = C-1, `cosine` = C-B, `deadline_check`,
`verify_authorization`, `verify_claimed_resource`, `verify_execution_lineage`,
`main` = I-3). Nothing else moved: `SYSTEM_PROMPT`, `_SCHEMA_TEXT`, `PROMPT_A`,
`PROMPT_B`, `SELECT_TAG`, `SELECT_SUFFIX`, `SELECT_N`, `NUM_FRAMES`, the
transcript constants, the confidence/cosine thresholds, the five-rate KILL
taxonomy, `build_slot_reliability`, `render_slot`, `materialize_role_map`,
`dense_rademacher_payload`, `parse_teacher_response` are untouched.

`self_test_fixtures()` now returns **45** checks (round-1 v7: 37; v6: 25). All 45
pass on the frozen bytes, and no fixture name is duplicated (a duplicate name
would be silently dropped by `dict(self_test_fixtures())` in
`preflight.run_self_tests`).

---

## Claim C-1 — the prompt renderer: **CONFIRMED**

1. **Could the v6 form ever have succeeded?** No, executed against the frozen
   templates: `PROMPT_A.format(transcript="x")` and `PROMPT_B.format(...)` both
   raise `KeyError: '"source_relation"'`. Unconditional on both forms.
2. **Exact substitution?** Yes. For a transcript containing non-ASCII, CJK, a
   newline and a literal `{braces}` sequence, `render_prompt(form, t) ==
   PROMPTS[form][:-len("{transcript}")] + t` for both forms, with the module-level
   templates unmutated.
3. **Any prompt byte changed?** No — the four hashes match the v6 frozen artifact.
4. **Any surviving `.format(transcript=` call site?** None. The two textual
   occurrences are a docstring (`common.py:221`) and the deliberate regression
   fixture `prompt_render_regression_str_format_would_raise` (`common.py:2194`),
   which asserts the v6 form still raises `KeyError`.
5. **Guards non-vacuous?** The four pre-substitution guards
   (`prompt_form not in PROMPT_FORMS`, non-`str` transcript, `count != 1`,
   non-terminal placeholder) are genuine, and two are exercised by
   `prompt_render_rejects_unknown_form_and_non_string`. The two post-hoc guards
   remain weak (`startswith(prefix)` is a tautology given a unique terminal
   placeholder; `endswith(transcript)` is vacuous for an empty transcript).
   Unchanged from round 1, still an observation rather than a finding.

---

## Claim I-1 — teacher-visible containment: **CONFIRMED**

1. **Before model load?** Yes — `assert_teacher_visible_precondition(inputs)` at
   `producer.py:1670`, `from_pretrained` at `producer.py:1679`. Repeated per item
   inside `one_forward` at `producer.py:1702`, after `build_messages` and before
   `apply_chat_template`/`generate`.
2. **Strict in both directions?** Yes. `teacher_visible_texts` raises on a wrong
   message-list length, unexpected message keys, an unexpected role, non-list or
   empty content, a content part that is not a dict or lacks `type`, a text part
   with extra keys or a non-string body, a video part with extra keys, a frame
   list whose length ≠ 8, any frame that is `str`/`bytes`/`bytearray`/`Path`, any
   unknown content type, and a final census requiring exactly one video part and
   exactly two text parts. Six of these are pinned by fixtures.
3. **What is banned, and is it wide enough?** Measured on the real tranche: the
   ban list is **402** tokens — all 200 HateMM + all 200 MHC-ZH selected
   identifiers plus `hate_video_` and `non_hate_video_` — and both datasets'
   identifiers are banned in both datasets' prompts, so cross-item leakage is
   refused as firmly as self-leakage. Matching is over
   `{token, NFKC(token), casefold(NFKC(token))}` against both `NFKC(text)` and
   `casefold(NFKC(text))`. The wider protection is the equality
   `texts == [SYSTEM_PROMPT, render_prompt(form, transcript)]`: the only variable
   content reaching the teacher is the transcript, so the amendment's broader ban
   (prediction, neighbor, rank, margin, error status, dataset statistic, fold
   role, intended use) is satisfied structurally rather than by enumeration.
4. **False positives on the real 400 transcripts?** **None — 0 of 800.** I ran the
   frozen guard label-blind over all 400 selected transcripts × both forms: zero
   rejections. As a margin, the 402-token ban list was scanned against **all
   1323** train transcripts (744 + 579): **0** rows contain any banned token.
   Shortest banned token is 11 characters, so accidental substring collision is
   not a live risk.
5. **Can it pass vacuously?** Not at the per-forward call site:
   `if not forbidden or video_id not in forbidden: raise` makes an empty or
   incomplete ban list a halt, and
   `teacher_visible_unbanned_identifier_rejected` proves an identifier absent
   from the list is rejected. The pre-load pass is weaker in one half, as noted
   above.

### The HateMM ID-label asymmetry — stated explicitly, as requested

**The asymmetry is handled correctly.** Every HateMM training identifier is
`hate_video_*` or `non_hate_video_*`, so the identifier *is* the binary label;
MHC-ZH identifiers are opaque BiliBili `BV` codes carrying no label information.
Therefore **the sealed ID-only allowlist delivers label containment for MHC-ZH
only, and none at all for HateMM.** `LABEL_BEARING_ID_SUBSTRINGS` encodes exactly
this (`("hate_video_", "non_hate_video_")` for HateMM, `()` for MHC_zh), the
`common.py` comment block states it, and `config.json →
v7_scope.I1_teacher_visible_containment` records it. The consequence is drawn
correctly: for HateMM, selection label-blindness rests on *hash reproduction of
the selection rule* (independently reproduced above, 200/200 on both datasets)
and teacher label-blindness on this *runtime* check — not on the allowlist.

---

## Claim I-2 — the selection self-test is a known-answer vector: **CONFIRMED**

Both pinned digests recomputed from first principles outside the module, by hand
concatenation:

```
sha256(utf8("C04-A0T-SMALL-v1" + "HateMM" + "c04-known-answer-vector" + "20260729"))
  = 871e0363e1b01a823f09a5a0bb9187749da74f1dfe8e454a733e21c218f6a384   [matches]
sha256(utf8("C04-A0T-SMALL-v1" + "MHC_zh" + "c04-known-answer-vector" + "20260729"))
  = 41bfb637f5cbb26bb0b9edfd44d19a3775d8df1712f9b49bde667863fbd37134   [matches]
```

The digests are literals independent of the module's own code path. Mutation
sensitivity measured: tag → breaks; suffix → breaks; dataset term → breaks;
identifier → breaks; concatenation order → breaks. The companion fixture
`selection_dataset_and_id_sensitivity` additionally requires three distinct
digests. The chosen identifier is synthetic and belongs to neither dataset, so
the fixture pins the rule without naming a label-bearing real id.

---

## Claim I-3 — both ceilings machine-checked and fail-closed: **PARTIALLY DELIVERED**

### Tranche ceiling (7200 s) — guard now leads the wrapper (round-1 H-B closed)

`BudgetGuard.at_job_start` is constructed exactly once, in
`verify_claimed_resource`; the deadline is stored and never recomputed. There are
exactly two `deadline_check` call sites — `producer.py:1751` at the item boundary
and `producer.py:1697` inside `one_forward` — both strictly *before* a unit of
work. `BudgetDeadlineReached` is caught only by the outer handler, which
publishes an accounting-only record and returns 40.

`SLURM_JOB_START_TIME` appears nowhere in the v7 tree except one docstring line.
The anchor is now the allocation-entry `/proc/uptime` reading carried in
`allocation_claim.json`. I executed the frozen `BudgetGuard` class (extracted by
`ast`, not imported) against the frozen config:

| claim duration `c` | lineage+model-hash time `v` | guard fires | wrapper `timeout` | lead |
|---|---|---|---|---|
| 0 s | 0 s | entry + 6780 s | entry + 7080 s | **300 s** |
| 5 s | 30 s | entry + 6775 s | entry + 7080 s | **305 s** |
| 30 s | 60 s | entry + 6750 s | entry + 7080 s | **330 s** |
| 60 s | 120 s | entry + 6720 s | entry + 7080 s | **360 s** |

The guard's lead is `guard_item_margin_seconds + c`, i.e. never less than 300 s,
because the wrapper's `C04_ACTIVE_WATCHDOG_SECONDS` already has `c` subtracted
and `at_job_start` subtracts the elapsed-since-entry again. The margin comfortably
covers a worst-case item (two 7B forwards at 8 frames / 256 new tokens), so the
exit-40 breach record is reachable. **Round-1 H-B is closed.** What the margin
does *not* cover is the unguarded post-loop phase — see **I-2**.

The breach record was inspected field by field: lineage, job id, terminal state,
exit code, both caps, the guard snapshot, per-dataset completed counts, teacher
and frame-pack counters, `outputs_truncated_or_altered: 0`, `seal_published:
false`, `no_scientific_verdict_is_published_by_a_budget_breach: true`. No metric,
no teacher output, no reliability rate, no CONTINUE/KILL verdict. Exit 40 is
distinct and the wrapper propagates it distinctly, and refuses a zero-exit run
that left a breach record behind.

### Campaign ceiling — read side sound and phase-scoped (round-1 I-C1 closed)

`assert_campaign_aggregate_headroom` is called from `validate_gpu_environment`,
the first statement of `claim()`, i.e. before `create_entry_marker`, before
`verify_gpu_lineage`, and well before the allocation claim and the ticket
consumption record are published — genuinely before the single-use ticket is
consumed. It is also called by the CPU preflight *before* the no-clobber
namespace is materialized, and by the producer before any model or data work.

Fail-closed matrix, executed against a scratchpad copy (repository untouched):

| mutation | result |
|---|---|
| ledger absent | HALT `campaign ledger is absent` |
| aggregate ≠ Σ rows | HALT `payload mismatch` |
| `payload_sha256` tampered | HALT `payload mismatch` |
| foreign `schema_version` | HALT `foreign campaign ledger schema` |
| foreign `run_id` | HALT `foreign campaign ledger run id` |
| aggregate cap raised to 999999 | HALT `cap is not the amendment cap` |
| aggregate cap lowered to 7200 | HALT `cap is not the amendment cap` |
| `phase` advanced, phase cap left at 7200 | HALT `phase cap does not match the phase` |
| `phase_cap` raised alone | HALT `phase cap does not match the phase` |
| first-tranche phase carries an advance token | HALT `first-tranche phase carries an advance token` |
| row chain break | HALT `chain break` |
| well-formed, 1 s already spent | loads, next 7200 s reservation **REFUSED** |
| well-formed, 7000 s already spent | loads, next 7200 s reservation **REFUSED** |
| genesis (0 s) | loads, 7200 s reservation accepted (effective cap 7200) |

**The effective ceiling is 7200 s today**, `min(phase_cap, aggregate_cap)`, and it
is the amendment clause that actually binds ("an aggregate maximum of 2
GPU-hours across both datasets and all C04 jobs",
`C04_USER_AMENDMENT_V2.md:32`). Any recorded spend of even one second refuses the
next 7200 s reservation, so a v8 rebuild after a v7 failure is refused — which is
what round-1 I-C1 asked for. **No code path writes `phase` or
`phase_advance_authorization`**: `append_campaign_gpu_job` copies the ledger dict
and mutates only `jobs`, `aggregate_gpu_seconds`, `head_payload_sha256`,
`ledger_revision`, `payload_sha256`. The only route to 28800 s is a hand edit
that changes `phase` and `phase_cap_gpu_seconds` together and reseals the payload
— i.e. the human gate the amendment intends. **Round-1 I-C1 is closed.**

**Opening zero is evidence-backed.** `sacct` read read-only by me:

```
13805|c04_a0t_small_v1_v5_preflight|…|0 |billing=8,cpu=8,mem=64G,node=1|FAILED
13840|c04_a0t_small_v1_v6_preflight|…|19|billing=8,cpu=8,mem=64G,node=1|COMPLETED
```

Both rows match the ledger's `genesis_evidence` verbatim including the
`alloc_tres` strings, elapsed seconds and states; neither carries `gres/gpu`, so
`gpu_seconds: 0` is correct for both. A full accounting sweep for `c04` job names
returns exactly these two rows, corroborating
`these_are_the_only_c04_jobs_in_the_accounting_record: true`.

### Campaign ceiling — write side now seal-independent (round-1 H-A closed)

`campaign-record` is a distinct ledger mode keyed on
`resource/allocation_claim.json`, which `claim()` publishes before it appends the
ledger job row, and the reconcile wrapper runs it **first**, under `set -e`,
before `reconcile-terminal`. Every path the request named therefore reaches it:

| path | `allocation_claim.json` present? | campaign row written? |
|---|---|---|
| exit 40 budget breach | yes | **yes** |
| watchdog TERM/KILL (124/137/143) | yes | **yes** |
| OOM / decode failure / any producer HALT after claim | yes | **yes** |
| fully successful sealed run | yes | **yes** (and again idempotently from `reconcile_terminal`) |
| HALT *before* `claim()` publishes the claim | no | **no** — see **I-3** |

`record_campaign_gpu_spend` verifies an already-present row instead of appending,
so a recovery reconciliation cannot double-count, and every numeric field
originates from `sacct_row` with `accounting_source: "sacct"`.

---

## Additional checks

- **`--time` directive:** absent from all three sbatch files and all three
  wrappers; each sbatch carries an explicit comment that the omission is
  deliberate.
- **Arrays / dependencies / chained submission / release / resubmission:**
  absent. No `sbatch`, `scontrol`, `scancel`, `srun` or `salloc` anywhere. The
  only `subprocess` call in the entire tree is `gpu_ledger.py:236`,
  `sacct -X -n -P -j <id> -o JobIDRaw,ElapsedRaw,AllocTRES,State` — read-only.
  All three wrappers and both Python entrypoints reject `SLURM_ARRAY_JOB_ID` /
  `SLURM_JOB_DEPENDENCY`.
- **`--gres`:** exactly one occurrence, `scripts/slurm/c04_a0t_small_v1_v7.sbatch:3`,
  `--gres=gpu:a100:1`. The preflight sbatch requests no GPU; the reconcile sbatch
  requests no GPU and its wrapper additionally rejects a non-empty
  `CUDA_VISIBLE_DEVICES` or `SLURM_GPUS_ON_NODE`.
- **Resources:** GPU sbatch = 1 GPU / 8 CPU / 64 GB (exactly as required);
  preflight = 8 CPU / 64 GB, no GPU; reconcile = 1 CPU / 4 GB, no GPU.
  `resources.gpu_count/cpus/ram_gb` = 1/8/64 and are asserted in the preflight,
  the GPU ledger and the producer.
- **No OCR entrypoint, no network/API client, no dev/test path, no cross-dataset
  path, no label reader:** confirmed by targeted grep. No `requests`, `urllib`,
  `httpx`, `aiohttp`, `socket`, `boto3` or any OCR library is imported anywhere;
  the only textual hit is the word "requests" inside a frame-decode docstring.
  The only `label` reference on the data path is `_skip_json_value`, which
  advances the parser past the token syntactically and increments a skip counter;
  the projector then requires the decoded key set to be exactly
  `{id, window_text, language}`. `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`
  are exported by the wrapper and asserted by the producer, and both
  `from_pretrained` calls pass `local_files_only=True`. `root_path` rejects any
  `dev`/`test`/`validation`-like path component.
- **Authorization flags in the correct pre-review state:** exactly one flag is
  `true` (`implementation_authorized`); all sixteen others — including
  `preflight_materialization_authorized`, `teacher_authorized`, `gpu_authorized`,
  `slurm_authorized`, `small_tranche_execution_authorized` and
  `post_job_reconciliation_authorized` — are `false`. The preflight wrapper
  blocks on `preflight_materialization_authorized != true`; the reconcile wrapper
  blocks on `post_job_reconciliation_authorized != true`.
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
  | `resolve_prompt_hashes(freeze=True)` with materialization `false` | HALT (same) |

- **Config contract normalization is exactly as narrow as documented.** Filling
  the four prompt-hash keys does **not** move `config_contract_sha256` (measured),
  so the v5 impossibility is closed rather than displaced. Mutating
  `selection.suffix`, `resources.small_cap_gpu_seconds`,
  `resources.campaign_aggregate_cap_gpu_seconds`,
  `resources.campaign_first_tranche_phase_cap_gpu_seconds`,
  `resources.guard_item_margin_seconds`, `maps.expected_hashes`,
  `teacher_contract.num_frames`, `model.snapshot_revision` or
  `paths.campaign_gpu_ledger` **does** move it. Only the `authorization` block,
  the four review pins, the four review verdicts and `prompt_hashes` are
  normalized out, each separately bound by a strict stage-authorization manifest.
- **Writer/reader key-set census.** All five `require_exact_keys` contracts in
  `gpu_ledger.py` were extracted by `ast` and compared against the dict literal
  each writer builds:

  | contract | writer | reader | agree today |
  |---|---|---|---|
  | GPU ledger (15 keys) | `preflight.py:411` | `gpu_ledger.py:265` | yes |
  | resource ticket (16) | `preflight.py:432` | `gpu_ledger.py:731` | yes |
  | allocation claim (12) | `gpu_ledger.py:782` | `gpu_ledger.py:367` | yes |
  | provisional GPU usage (14) | `common.build_provisional_gpu_usage` | `gpu_ledger.py:445` | yes |
  | budget guard (7) | `BudgetGuard.accounting_snapshot` | `gpu_ledger.py:470` | yes |

  No live mismatch anywhere. All five are nevertheless maintained as **duplicated
  literals** with no mechanical cross-check — see **H-2**.

---

# Findings

## HIGH H-1 — `campaign-record` writes an unfiltered sacct row into the cross-version accumulator before the stage that would refuse it, and an over-cap write makes the campaign ledger permanently unloadable

**Where.** `gpu_ledger.py:1104-1141` (`campaign_record`, no cap check) and
`gpu_ledger.py:1256-1259` (`reconcile_terminal`, which *does* hard-refuse
`elapsed > 7200`), ordered by `scripts/wrappers/c04_a0t_small_v1_v7_reconcile.sh:63,66`;
`common.py:1264-1303` (`append_campaign_gpu_job`) and `common.py:1188`
(`load_campaign_gpu_ledger`).

**Mechanism, measured against a scratchpad copy of the ledger.** With the frozen
genesis ledger and a terminal sacct row of 7250 s:

```
append_campaign_gpu_job(row) RAISED: HALT_CAMPAIGN_AGGREGATE_CAP: campaign aggregate already exceeds the cap
  file on disk after the raise: aggregate=7250  jobs=1  ledger_revision=1
  subsequent load_campaign_gpu_ledger()          RAISES
  subsequent assert_campaign_aggregate_headroom() RAISES
  retrying the same idempotent append            RAISES
```

Three separate defects compose here:

1. **No cap check on the write side of `campaign-record`.** `reconcile_terminal`
   raises `terminal sacct GPU seconds exceed 7200 cap` for exactly this value,
   but the wrapper runs `campaign-record` *first*, and `campaign_record` calls
   `sacct_row` → `record_campaign_gpu_spend` → `append_campaign_gpu_job` with no
   bound at all. **The permissive stage is the irreversible writer, and it runs
   before the strict one.** This is the failure family the request asked me to
   hunt, applied to the one artifact that deliberately lives outside every
   no-clobber namespace so it can survive an implementation-version bump.
2. **`append_campaign_gpu_job`'s deliberate over-phase branch is dead code.**
   Its comment states "Accounting must never be refused — the spend is already
   real", prints a warning and continues; but the function ends with
   `return load_campaign_gpu_ledger()`, and that call re-applies
   `total > campaign_effective_cap(ledger)` and raises. The designed behaviour is
   unreachable by construction: the row is written and the process then dies.
3. **The write is unrecoverable.** After it, `load_campaign_gpu_ledger()` raises
   on every future call, so `assert_campaign_aggregate_headroom` halts for **every
   future C04 stage in every future namespace** (v8, extraction, adaptation), and
   `record_campaign_gpu_spend`'s idempotent verification path halts too, so the
   row cannot even be re-verified. Repair requires hand-editing a chained,
   payload-hashed ledger — precisely the tamper the chain exists to make evident.
   Meanwhile `campaign-record` exits non-zero, `set -e` aborts the reconcile
   wrapper before `reconcile-terminal`, the per-namespace ledger stays in
   `EXIT_RECORDED_PENDING_SACCT`, `resource_final_state.json` is never published,
   and `review.downstream_review_requires_terminal_resource_state: true` blocks
   every downstream review permanently.

**Failure scenario / reachability.** `sacct` `ElapsedRaw` is measured from SLURM
job start; the wrapper's `timeout` is measured from the wrapper's own
`/proc/uptime` read and is `cap − reserve = 7080 s`, plus `--kill-after=30s`,
plus the EXIT-trap `mark-exit` write. The whole defence is the fixed 120 s
`watchdog_reserve_seconds`, of which 30 s is already spoken for by `kill-after`.
Nothing in the code measures or bounds the job-start-to-wrapper-start offset.
This is round-1 **I-C2**, which the round-2 summary records as "subsumed by the
H-B margin"; it is not — the H-B margin moved the *in-job guard*, not the wrapper
`timeout`, and the wrapper-timeout path stays live whenever any single unit
between guard checks outruns the 300 s margin. I did measure the offset on this
cluster's most recent job (13840: `sacct` Start `05:11:31`, step stderr created
`05:11:31.87`), so today's prolog is sub-second and the realistic worst case is
≈ 7090 s — inside the cap. The trigger is therefore a tail event, not a
certainty, which is why this is High and not Critical. But the reserve is an
unmeasured constant, the run is unrepeatable, and the damage is unbounded and
cross-version.

**What would close it.** (a) Apply the same `> cap_gpu_seconds` refusal in
`campaign_record` that `reconcile_terminal` already applies, *before*
`append_campaign_gpu_job`, so the two stages cannot disagree; and/or (b) make
`append_campaign_gpu_job` genuinely honour its own comment — record the over-cap
spend, set an explicit `phase_over_cap: true` marker that `load_campaign_gpu_ledger`
accepts for reading while `assert_campaign_aggregate_headroom` refuses every
further reservation, and drop the trailing full re-load (or re-load with the
cap assertion relaxed for the just-written revision). (c) Independently, close
the I-C2 root: derive the wrapper's `timeout` from
`cap − (measured job-start-to-wrapper-start offset) − reserve`, or enlarge
`watchdog_reserve_seconds` so the worst-case sacct elapsed is provably < 7200 s.
A CPU-preflight fixture that appends a 7201 s row to an in-memory ledger and
asserts the result is still loadable would have caught the whole thing.

## HIGH H-2 — the writer/reader drift channel that produced round-1 C-A is still open, and the fixture named as the guarantee is a tautology

**Where.** `common.py:108-132` (`PROVISIONAL_USAGE_KEYS`, `BUDGET_GUARD_KEYS`),
`common.py:2035-2062` (`build_provisional_gpu_usage`), `common.py:2132-2137`
(the fixture `provisional_usage_writer_matches_reader_key_set`), versus
`gpu_ledger.py:445-482` (the reader).

**What the round-2 summary claims.** "Both halves now build and validate through
single shared constants, and a fixture asserts they agree."

**What the frozen bytes do.** `gpu_ledger.py`'s `from … common import (…)` list
contains 26 names; `PROVISIONAL_USAGE_KEYS` and `BUDGET_GUARD_KEYS` are **not
among them**. The reader validates against two hand-written set literals, exactly
as in v6. Only the writer half goes through a shared constant. And the fixture
compares `set(build_provisional_gpu_usage(...))` against `set(PROVISIONAL_USAGE_KEYS)`
— but `build_provisional_gpu_usage` already ends with
`require_exact_keys(record, set(PROVISIONAL_USAGE_KEYS), …)`, so the fixture
compares the constant with itself. It cannot observe `gpu_ledger.py` at all.

**Measured.** I reproduced the exact v7-first-draft defect on a scratchpad copy:
added one field to `PROVISIONAL_USAGE_KEYS` and to the builder, left
`gpu_ledger.py` byte-identical.

```
DRIFTED writer: fixture count 45   failing: []
writer now has 15 keys, reader still has 14   ->   drift present: True
=> CPU preflight self-test still PASSES on the drifted writer
```

**Failure scenario.** The current bytes have no live mismatch — I verified all
five writer/reader pairs agree today, so this run would not fail. The finding is
that the *recurrence* channel is intact and invisible: the next edit that adds an
accounting field to the seal record reproduces round-1 C-A verbatim, all 45
preflight fixtures stay green, and the mismatch surfaces only at
`CPU_POST_JOB_RECONCILIATION` — after the full 2 GPU-hour A100 allocation has
been spent, inside a no-clobber namespace that cannot be repaired without moving
`gpu_ledger.py`'s SHA-256, hence `config.implementation_hashes`, hence
`config_contract_sha256`, hence invalidating the values already pinned in the
preflight manifest, genesis ledger and resource ticket. A reviewer granting `GO`
on the strength of the round-2 summary would be granting it on a description that
does not match the code.

**What would close it.** Import `PROVISIONAL_USAGE_KEYS` and `BUDGET_GUARD_KEYS`
into `gpu_ledger.py` and pass them to `require_exact_keys` there, so the two
halves are literally the same object; and, for the four other duplicated
contracts (GPU ledger, resource ticket, allocation claim, and their writers),
either do the same or add a real CPU-preflight fixture that parses the reader's
key set — e.g. via `ast` over `gpu_ledger.py`, or by exposing each reader set as
a named constant the fixture can import — and asserts equality with the writer's.
The fixture must fail when only one side is edited; today it cannot.

## IMPORTANT I-1 — the preflight round-trip closes the `reliability` fragment only; the C-B clamp itself, and every other producer record, are still unexercised before the GPU is spent

`downstream_contract_fixtures()` is real work and I do not want to understate it:
it builds a `provisional_gpu_usage` record through the production builder, and it
validates `build_slot_reliability` output against the *actual* `reliability`
definition of the frozen canonical schema in six states (cosine 1.0, cosine
`1 + 2^-52`, `null` cosine, `missing`, `single_valid`, and all four slots). The
`reliability_rejects_an_unclamped_cosine` fixture does prove the schema bound is
real. But the coverage stops there:

- **`cosine()` lives in `producer.py`, which the CPU preflight never imports.**
  No fixture calls it. The fixture's `over_one = 1.0 + 2.0**-52` is a hand-written
  constant, so deleting `max(-1.0, min(1.0, …))` from `producer.cosine` leaves all
  45 fixtures green and re-opens round-1 C-B in full. The clamp is correct — I
  verified it, and re-measured the underlying overshoot at 572/2000 trials with
  bfloat16-rounded 3584-dim vectors — but it is protected by nothing.
- No fixture ever builds a **full** `canonical_record`, a `prompt_record`, a
  `frame_pack_manifest`, a `resource_final_state` or a `gpu_ledger`/`resource_ticket`/
  `allocation_claim` record and validates it against the contract its consumer
  applies. `validate_schema` at preflight is exercised only against the
  stage-authorization manifest.
- The full canonical record is first schema-validated at `producer.py:1395`,
  inside `canonicalize_dataset`, i.e. **only after all 800 forwards are paid
  for**. That structural exposure is unchanged; only its one known trigger is
  closed.

The blind spot is therefore narrowed, not closed, and it is the same shape as
v6's "no self-test ever rendered a prompt".

**What would close it.** Build one full `canonical_record` and one full
`prompt_record` in the preflight — including the degenerate cases (all slots
`missing`, all `single_valid`, the agreeing `stable` case) — and run them through
`validate_schema` against the frozen schemas; and give the producer's own pure
functions (`cosine` at minimum) a fixture reachable from the CPU preflight, e.g.
by moving `cosine` into `common.py` or by having the preflight import the
producer's pure-CPU helpers. Both are free.

## IMPORTANT I-2 — the mandatory post-loop canonicalization and seal phase is unguarded and must fit inside a margin sized for one item; overrun is a total, unresumable loss

`deadline_check` is called at exactly two sites, both inside the item loop's
`try` block. Everything after the loop — `canonicalize_dataset` for both datasets
(≈ 18 tokenizer + GPU-embedding round-trips per item × 400 items, plus two
`256×3598` projections per item, plus full canonical-record schema validation per
item), the Merkle roots over 400 canonical and 800 prompt records, the ~8 MB seal
write, and the post-publication `idempotent_complete` re-verification — runs with
**no deadline check at all**.

The only budget that phase has is whatever the item loop left, and the item loop
is permitted to run right up to the guard deadline. `guard_item_margin_seconds`
is documented and asserted as buying back "at least one worst-case item"
(`margin <= 0 or margin >= watchdog_seconds` is the only check on its size);
nothing states or enforces that it must also cover the seal phase.

**Failure scenario.** The last item completes just before the guard deadline —
which is the *expected* case if the tranche is sized near the budget (400 items ×
2 forwards against 6780 s implies ~8 s per forward). Canonicalization then has
`margin + claim_duration` ≈ 300-360 s. If it overruns, the wrapper's `timeout`
SIGTERMs mid-canonicalization: exit 124, **no seal, no breach record**, and the
run is unresumable — a fresh SLURM job gets a new job id, so `claim()` takes the
`ledger["jobs"]` non-empty branch and raises `single-use tranche already claimed`.
Up to 2 GPU-hours are lost with nothing but per-item checkpoints to show for it,
and (per H-1) the campaign accumulator then records that spend, which under the
7200 s phase cap forecloses any C04 retry. I could not measure the seal phase's
real duration without running GPU work, so I state the exposure rather than a
probability.

**What would close it.** Add a `deadline_check(guard, "canonicalization and seal")`
immediately before the canonicalization loop, with its own reserve, so an
insufficient remainder becomes a clean exit-40 breach with an accounting record
instead of a mid-phase SIGTERM; and size `guard_item_margin_seconds` (or a new
`guard_seal_phase_margin_seconds`) from a measured seal-phase cost rather than
from one item.

## IMPORTANT I-3 — the campaign accumulator is blind to GPU-seconds burned by an allocation that halts before `claim()` publishes the allocation claim

`campaign_record` keys entirely on `resource/allocation_claim.json`; if it is
absent it prints "no allocation claim; nothing to record" and returns 0. But the
allocation claim is published only at `gpu_ledger.py:796`, after
`validate_gpu_environment`, `create_entry_marker`, the full `verify_gpu_lineage`
chain (preflight manifest + 15 implementation hashes + 15 design hashes + payload
review + GPU authorization), the ledger load/validate/reconcile, and the ticket
validation. Every one of those can HALT — and an A100 allocation that halts there
has still consumed real GPU-seconds that `sacct` will bill and that the amendment
counts ("including **every** GPU-second consumed by the first tranche and any
later C04 extraction/adaptation job", `C04_USER_AMENDMENT_V2.md:45`). The same
applies to a wrapper-level exit before the first Python call. There is no other
code path that records them.

The amounts are small (tens of seconds per failed attempt, since the 16.6 GB
model tree is hashed later, in the producer), so this is Important rather than
High. But this project has now produced three implementation namespaces, and a
pre-claim HALT is exactly the outcome its recent history keeps producing; the
accumulator that exists to make the ceiling machine-checked should not be the one
artifact that forgets them.

**What would close it.** Have `campaign-record` fall back to
`resource/allocation_entry_marker.json` (written by the wrapper before any check)
when the allocation claim is absent, and record the sacct row for that job id
with an explicit `no_claim_published: true` marker; or require the reconcile
stage to run for every C04 GPU job id, claim or no claim.

---

## Non-blocking observations

1. **The pre-model-load containment pass is half self-comparing** (detailed under
   Claim I-1). Requirement met by the per-forward site; worth keeping in the
   record.
2. **`maps.expected_hashes` is protected only by inclusion in the contract hash.**
   No code asserts its literal value, so the prose invariant in
   `prompt_hash_contract` is not machine-checked. Unchanged from round 1.
3. **The campaign ledger's `phase` is not cryptographically bound to any
   authorization artifact.** No code advances it (verified), but an advanced
   ledger is indistinguishable from an authorized one; the amendment's gate is
   human-only. Consider binding `phase_advance_authorization` to the SHA-256 of
   the conditional-tranche code/resource authorization manifest.
4. **`append_campaign_gpu_job` takes no lock on the campaign file.** It is
   protected only by the *per-namespace* `resource/gpu_ledger.lock`, which would
   not exclude a future namespace's reconciler. A lock on the campaign path
   itself would close it.
5. **The reader does not validate `budget_guard` field *values*.** It checks the
   seven keys; `guard_may_only_stop_work_before_an_item: false` would pass.
   Likewise `campaign_gpu_ledger_sha256_at_seal` is recorded but never compared
   to anything.
6. **The GPU wrapper writes the allocation entry marker before any check**
   (`c04_a0t_small_v1_v7.sh:41-64`), so a HALT at the very first Python gate
   leaves a marker whose `jq -e` job-id assertion forbids any later GPU attempt
   in that namespace. Resubmission is forbidden by policy anyway, so this is
   consistent rather than wrong — but it means a config typo caught at
   `validate_gpu_environment` costs a full namespace rebuild.
7. **The exit-40 wrapper branch runs `jq -e` under `set -e`.** A malformed breach
   record would surface as exit 1 rather than exit 40, losing the distinct code
   the branch exists to propagate. Fail-closed, but the distinctness is lost.

---

## Summary table

| # | Severity | Finding |
|---|---|---|
| H-1 | High | `campaign-record` writes an unfiltered sacct row into the cross-version campaign accumulator before `reconcile-terminal`'s 7200 s refusal runs; an over-cap write is performed and *then* rejected, leaving the ledger permanently unloadable for all future C04 work. Subsumes round-1 I-C2, whose root (wrapper `timeout` anchored to a different clock than `sacct`, defended by an unmeasured 120 s reserve) is untouched. |
| H-2 | High | The round-1 C-A recurrence channel is still open: `gpu_ledger.py` validates against hand-maintained key-set literals and does not import `PROVISIONAL_USAGE_KEYS`/`BUDGET_GUARD_KEYS`; the fixture cited as the guarantee compares the writer's constant with itself. Reproduced the original defect with all 45 fixtures green. No live mismatch today. |
| I-1 | Important | Round-1 I-C3 narrowed, not closed: `cosine()`'s clamp — the entire C-B repair — is exercised by no preflight fixture, and no full `canonical_record`/`prompt_record`/`resource_final_state` is ever round-tripped. The full canonical record is still first validated only after all 800 forwards. |
| I-2 | Important | The unguarded post-loop canonicalization + seal phase must fit inside a margin sized for one item; overrun yields exit 124 with no seal, no breach record, and no resumption. |
| I-3 | Important | GPU-seconds burned by an allocation that HALTs before `claim()` publishes `allocation_claim.json` are never recorded in the campaign accumulator, against the amendment's "every GPU-second". |

**Verdict: REVISE (0C / 2H / 3I). No execution authority is conferred.
`authorization.preflight_materialization_authorized` must remain `false` and
`review.code_resource_verdict` must remain `PENDING`. H-1 in particular should be
closed before any GPU allocation, because its damage is not confined to this
namespace: it can brick the accumulator every later C04 stage depends on.**
