# C04-A0T-SMALL-v1 Implementation-v7 Record

Date: 2026-07-31
Status: **PROSPECTIVE / FRESH INDEPENDENT REVIEW REQUESTED / EXECUTION BLOCKED**
Scientific tag: `C04-A0T-SMALL-v1`
Implementation version: `v7_prospective`
Supersedes: `artifacts/c04/a0t_small_v1_impl_v6` (CPU preflight job `13840`,
payload review `GO (0C/0H/3I)`)

## Why a new namespace at all

The v6 payload review returned `GO` with three Important findings, all of which
name properties that are *asserted* rather than *checked*. Closing them requires
editing the producer, the common module, the GPU ledger and the config — and
`config_contract_sha256` does **not** normalize `implementation_hashes`, so any
such edit moves the config contract hash, which is baked into the v6
code/resource authorization manifest, the v6 genesis GPU ledger and the v6
resource ticket, all inside a no-clobber namespace. The v6 files are therefore
untouched by construction, and v7 is a full namespace rebuild in the same
discipline the v5→v6 transition used.

**No v6 byte was edited.** `artifacts/c04/a0t_small_v1_impl_v6/` and all 15 v6
implementation files retain their recorded hashes.

## What v7 changes

All three wrappers, all three sbatch files and all five JSON schemas are
byte-identical to their v6 predecessors modulo the `v6`→`v7` version-token
rename, except for the wrapper change in §3 below. Four repairs follow: the
three payload-review Importants, and one Critical defect found while closing the
first of them.

### C-1 (Critical, found here). The v6 prompt renderer could never render a prompt

`scripts/analysis/c04_a0t_small_v1_v6_producer.py:768` — the only prompt-render
call site in the whole v6 tree — reads:

```python
{"type": "text", "text": PROMPTS[prompt_form].format(transcript=transcript)},
```

Both templates embed `_SCHEMA_TEXT`, which contains the literal JSON
`{"source_relation":"current_presenter|…"}` and `{"S":0,"P":0,"T":0,"H":0}`.
`str.format` reads those braces as replacement fields. Verified against the
frozen v6 bytes (`c04_a0t_small_v1_v6_common.py`,
`81b10f586cfa5d619db459505ba2c8c43a89fc50e0c1e978e500fb4932633f68`) by static
`ast` parse, with no import of the repository module:

```
PROMPT_A: .format(transcript=...) RAISES KeyError: '"source_relation"'
PROMPT_B: .format(transcript=...) RAISES KeyError: '"source_relation"'
```

Each template holds exactly one `{transcript}`, and it is the final token, so
the intended substitution is unambiguous.

**Why nothing caught it.** No v6 fixture renders a prompt: `grep -n 'PROMPTS\['`
over `..._v6_common.py` and `..._v6_preflight.py` returns nothing. The 25 v6
self-tests could all pass with this defect live, and the v6 payload reviewer read
line 768 correctly ("`producer.py:768` interpolates only `{transcript}` into the
prompt templates") without executing it.

**What it would have cost.** `one_forward` runs after
`Qwen2_5_VLForConditionalGeneration.from_pretrained(...).to("cuda")`. The job
would have consumed the single-use resource ticket, written the
`allocation_entry_marker`, entered the no-clobber namespace, loaded 7B weights
onto the A100, and then died on the **first** forward with a `KeyError`. That is
the third instance of one family — v5's static gate (job 13805), v6's `mark_exit`
genesis-bump, and now this — in which an irreversible resource is consumed before
the check that rejects the run.

**The repair.** `render_prompt(prompt_form, transcript)` in
`..._v7_common.py` performs a literal single-placeholder substitution and
asserts, fail-closed, that the placeholder is unique and terminal, that the
template prefix survives unaltered, and that the transcript lands at the tail.
All four render call sites in v7 route through it; no `.format(transcript=` call
site remains outside a docstring.

**No prompt byte changes.** The four frozen prompt hashes are re-asserted as a
fixture and still equal the v6 freeze exactly:

| key | value |
|---|---|
| `system` | `1ffc06755b18f9315dc930aaf02a0c79cf1f9e2d91d0ec5bae2fa0c1436ed048` |
| `A` | `cecb3555daa7a4cc0840db154086138dae9260795bde2b64e89f6724d80abb9b` |
| `B` | `9521bee7fe7a6964d4ff61605bbe5a02eb4cfd149fd5aa9cbf975accbfe40314` |
| `combined` | `a42268e4c0d1afb2b9576dd968d7f6250fc989fe3b97077a55363e63e1e2bb8a` |

This repair is **outside** the three authorized closures. It is recorded
prominently rather than folded in silently, because without it the authorized
GPU submission cannot produce a single teacher response.

### I-1. Teacher-visible containment is now a checked precondition

The user amendment requires that the teacher see no label. In v6 that held
because one line of prompt assembly happened to interpolate only `{transcript}`.
The review also named the asymmetry that makes this worth enforcing: MHC-ZH
identifiers are opaque BiliBili codes, but **every HateMM identifier
(`hate_video_*`, `non_hate_video_*`) is the label**, so the sealed ID-only
allowlist provides label containment for MHC-ZH and none at all for HateMM.
Selection label-blindness is established by hash reproduction; teacher
label-blindness is now established by this runtime check.

- `teacher_visible_texts(messages)` walks the assembled message structure and
  returns every string a forward can read. It is strict in both directions: an
  unrecognised role, content part, payload type, frame count, or a frame that is
  a `str`/`bytes`/`Path` all raise. A later edit that adds a teacher-visible
  field cannot silently escape the check.
- `forbidden_teacher_visible_tokens` bans **both** datasets' 200 identifiers in
  **both** datasets' prompts (402 tokens: 400 identifiers plus `hate_video_` and
  `non_hate_video_`), so a cross-item leak is refused as firmly as a self-leak.
- `assert_teacher_visible_containment` additionally requires the rendered text to
  be exactly `[SYSTEM_PROMPT, render_prompt(form, transcript)]` — i.e. the frozen
  template on the frozen, hash-pinned transcript — and matches every banned token
  against the raw, NFKC and NFKC-casefolded forms of both texts.
- It fires **before any teacher forward** in two places: as a batch precondition
  over all 400 identifiers × 2 forms in `assert_teacher_visible_precondition`,
  run before the model is loaded, and again per item inside `one_forward` before
  `apply_chat_template`, where the frame payload also becomes checkable.
- The resulting census is recorded in the sealed access ledger under
  `teacher_visible_containment`.

Measured against the real data, label-blind (only `id`/`window_text`/`language`
decoded from either ASR file): **800 renderings checked, 0 halts**. The v7
selection recomputed from the frozen rule reproduces the v6 frozen allowlists
**exactly** on both datasets, which is the direct evidence that no scientific
selection semantic moved.

### I-2. The tautological self-test is replaced by a known-answer vector

v6's `selection_deterministic` compared `selection_digest("HateMM","x")` to
itself and could not fail under any mutation of the tag, suffix, dataset term,
concatenation order or digest function. It is replaced by two checks:

- `selection_known_answer_vector` — pinned literal digests for a fixed synthetic
  identifier (`c04-known-answer-vector`, belonging to neither dataset, so the
  fixture pins the rule without naming a label-bearing id):
  `HateMM` → `871e0363e1b01a823f09a5a0bb9187749da74f1dfe8e454a733e21c218f6a384`,
  `MHC_zh` → `41bfb637f5cbb26bb0b9edfd44d19a3775d8df1712f9b49bde667863fbd37134`;
- `selection_dataset_and_id_sensitivity` — the dataset term and the identifier
  each move the digest.

Non-vacuity was probed directly: tampering `SELECT_TAG` or `SELECT_SUFFIX` breaks
the known-answer vector in both cases.

### I-3. Both ceilings are machine-checked and fail-closed

**The 2 GPU-hour tranche ceiling, in-job.** `BudgetGuard.at_job_start` computes
ONE absolute deadline at job start and never recomputes it, modelled on the C02
bounded extraction (`scripts/slurm/c02_density_extract.sbatch`). It takes the
tighter of the ticket watchdog remainder and, when `SLURM_JOB_START_TIME` is
present, an allocation-start epoch deadline of `start + cap - reserve`; the basis
actually used is recorded. A future `SLURM_JOB_START_TIME`, or no remaining
budget at job start, halts.

The guard may only ever **STOP work before an item begins** — it is called at the
per-video item boundary before frame decode, and at the top of `one_forward`
before any tensor work. It is never called after the teacher loop, so it cannot
abort a run whose teacher work is already paid for. A breach therefore leaves the
in-progress item entirely unwritten, every completed per-item checkpoint record
intact, and no seal at all, since the seal is published only when all 800 records
exist.

On breach the producer catches `BudgetDeadlineReached`, publishes an
**accounting-only** record at `resource/budget_breach.json` (job id, deadline
basis, elapsed, completed counts, caps; no metric, no teacher output, no
reliability rate, no CONTINUE/KILL verdict) and returns exit code **40**, which
is distinct from every other exit path. The wrapper reports 40 separately from an
engineering failure, `jq`-verifies the record's terminal state, and refuses a
zero-exit run that nonetheless left a breach record. Resubmission remains
forbidden.

**The 8 GPU-hour campaign ceiling.** `artifacts/c04/campaign/gpu_ledger.json` is
a new campaign-scoped accumulator. It deliberately lives **outside** every
implementation namespace, because the namespace ledger is hash-pinned by the
resource ticket and preflight manifest and cannot be extended in place, and
because the accumulator must survive an implementation version bump. Its
integrity comes from a payload digest plus a per-row hash chain
(`previous_payload_sha256` → `row_payload_sha256`, from `GENESIS`) rather than a
fixed external pin, since it is appended to after each job.

- Read side: `assert_campaign_aggregate_headroom` halts unless
  `aggregate + reservation ≤ 28800`. It is called at three points, each before an
  irreversible step — in the CPU preflight before the no-clobber namespace is
  created, in `validate_gpu_environment` **before `claim()`** consumes the
  single-use ticket, and in the producer's `verify_authorization` at job start.
  A missing, malformed, foreign, mis-capped or chain-broken ledger halts rather
  than defaulting to zero spend. No stage may create it — only verify it.
- Write side: `record_campaign_gpu_spend` appends the real sacct row at terminal
  reconciliation, idempotently (an already-recorded job is verified, not
  re-added, so a recovery reconciliation cannot double-count).

The genesis opens at zero as an **audited fact**: `sacct` shows the only two C04
jobs ever are `13805` (`c04_a0t_small_v1_v5_preflight`, FAILED, elapsed 0) and
`13840` (`c04_a0t_small_v1_v6_preflight`, COMPLETED, elapsed 19), and neither
carries `gres/gpu` in `AllocTRES` — both are `billing=8,cpu=8,mem=64G,node=1`.
Those rows are recorded in the file under `genesis_evidence`.

## Prospective file hashes

| File | SHA-256 |
|---|---|
| `scripts/analysis/c04_a0t_small_v1_v7_common.py` | `5fc5259ec4a98b47fa95851272b43fd6f8bdd7767d2c65a50d6c09889ebe2690` |
| `scripts/analysis/c04_a0t_small_v1_v7_preflight.py` | `ecdc8568dfab0a50e5f6701fba7c09fe939fcdba3af12e35243f6d11e9af873a` |
| `scripts/analysis/c04_a0t_small_v1_v7_gpu_ledger.py` | `944023b3aafc04dfdeee59fe920ed77b4ef882c4b56c39968bc0f12ee96758e3` |
| `scripts/analysis/c04_a0t_small_v1_v7_producer.py` | `7a3c3a794454c5856238234cb305a410b831ca624cb0e0e8391694992ebeae26` |
| `scripts/wrappers/c04_a0t_small_v1_v7_preflight.sh` | `914dd5df80ab45d5aa4102e6f4718c31282c9e48b9b7bee96e11dc6d54bf59b0` |
| `scripts/wrappers/c04_a0t_small_v1_v7.sh` | `645e501140690cece68e438422bfdc45005af1a5f392c70a86dc7c2e3713df5c` |
| `scripts/wrappers/c04_a0t_small_v1_v7_reconcile.sh` | `7af043225285f129e59ffd385782bed214e96cd39a296ecdeeb2a7f5fd16c8e0` |
| `scripts/slurm/c04_a0t_small_v1_v7_preflight.sbatch` | `919316c70ae79d9f019de6952acc761eaec065f60afec17da4dee573613ded39` |
| `scripts/slurm/c04_a0t_small_v1_v7.sbatch` | `00ddeeed57d1f585a7305738b259fcb6604eb8c545082650406ea6fe403aeacc` |
| `scripts/slurm/c04_a0t_small_v1_v7_reconcile.sbatch` | `d8f634ec88d762be8e797edf3bcc0433f033cbc1feb3c19faced71bde5ef44cd` |
| `schemas/c04/c04_a0t_small_v1_v7_prompt_record.schema.json` | `541d02455aee3af9293e978a8628f438bd78feca08d8b54f442e1ac8c77084f3` |
| `schemas/c04/c04_a0t_small_v1_v7_canonical_record.schema.json` | `bacbddaeba13806829e5dffa09fcab55a76a89420f03310a495ac6cccde578b3` |
| `schemas/c04/c04_a0t_small_v1_v7_stage_authorization.schema.json` | `2edac849da8a3bf4ebd3ad82e39de6f6d22e76bcfed3d8688379f349e08f10a8` |
| `schemas/c04/c04_a0t_small_v1_v7_payload_review.schema.json` | `7edebdfe81bb5180d9968d91ee797c6cf0ba14bbf71ec585d0b75a60d9cad81a` |
| `schemas/c04/c04_a0t_small_v1_v7_resource_final_state.schema.json` | `e2f9dca545874a4b2c8e65932b987b068563da560937965e11f684be255cd53d` |
| `configs/c04/c04_a0t_small_v1_v7.json` (authority snapshot) | `60a945c1c9a96238335250e283fbdf1d90f82dc055ea021c66df8a99089ccd63` |
| `artifacts/c04/campaign/gpu_ledger.json` (genesis) | `e84517d69ce7aa9a87c600b920882b1e19f118385fb571cc38acd544560dd14e` |

## Static and read-only validation performed

Following the v6 discipline — the v1–v5 practice of shipping unexecuted code is
exactly what put job 13805 in the queue, and it is what left C-1 undetected
through five code/resource rounds and a payload review.

- `python -m py_compile` passed for all four v7 programs; `bash -n` passed for
  all three wrappers and all three sbatch files; `jq` parsed the config, all five
  schemas and the campaign ledger.
- All 15 `implementation_hashes` re-verified against disk.
- **37 self-test fixtures, 0 failed** (v6 had 20 in `self_test_fixtures`; v7
  adds 9 containment, 5 render and 2 campaign-cap checks, and replaces 1
  tautology with 2 real checks).
- Prompt-hash contract exercised on all four config states: the pre-authority
  snapshot (`preflight_materialization_authorized: false`) HALTs on **both**
  paths with `HALT_PROMPT_HASH_SENTINEL`; the post-authority state passes the
  freeze path with binding `SENTINEL_PENDING_CPU_PREFLIGHT_FREEZE` and still
  HALTs downstream; a simulated post-freeze config passes downstream with
  `LITERAL_BOUND`.
- Config-contract neutrality re-tested and still as narrow as claimed: the
  authority flip and the sentinel→literal transition do **not** move
  `config_contract_sha256`, while `resources.small_cap_gpu_seconds`,
  `resources.campaign_aggregate_cap_gpu_seconds` and
  `teacher_contract.num_frames` each do.
- `BudgetGuard` exercised on five states: absent `SLURM_JOB_START_TIME` (falls
  back to the ticket watchdog remainder), an allocation 7000 s old (tightens to
  ~79 s on the `SLURM_ALLOCATION_START_EPOCH` basis), an exhausted allocation
  (HALTs at job start), a future start time (HALTs), and an expired guard (stops
  before an item with `BudgetDeadlineReached`).
- Campaign ledger verified end to end: genesis loads and chain-verifies, a
  28801 s reservation is refused, a non-positive reservation is refused.
- The containment precondition was run against **all 400 real transcripts and
  both prompt forms — 800 renderings, 0 halts** — so the guard is not a
  false-positive risk on the live tranche.

No dataset label value, video byte, model weight, teacher, GPU or SLURM command
was involved. `artifacts/c04/a0t_small_v1_impl_v7/` does not exist.
`PYTHONDONTWRITEBYTECODE=1` was set throughout and validation ran from a
scratchpad outside the repository, so no `.pyc` was written.

## Execution state

Every teacher/GPU/SLURM/reconciliation authorization is `false`.
`preflight_materialization_authorized` is `false` in this pre-authority
snapshot. `code_resource_verdict` is `PENDING` and its pin is the sentinel
`PENDING_V7_CODE_RESOURCE_REVIEW`, which `_verified_review_file` rejects with
`HALT_REVIEW_LINEAGE`. No v7 review or runtime artifact exists. The unified
pilot gate, the `+0.030/+0.030` two-dataset target and the amendment's full-bank
`+0.050/+0.050` DIRECT-OOF and STUDENT-OOF gates are untouched and unwaived.

## Independent review history — five rounds, REVISE ×4 then GO

Every round was a fresh reviewer with no exposure to the implementation
reasoning, given only the frozen bytes and a review request. Every finding was
accepted and repaired; none was argued away. The rounds are recorded because
what they found is the point.

| round | verdict | what it caught |
|---|---|---|
| 1 | `REVISE 2C/2H/3I` | the reconciler's strict provisional key set was not updated alongside the writer, which would have halted the terminal reconciliation of a fully paid-for tranche inside a no-clobber namespace; `proposition_cosine` could exceed the schema's `maximum: 1` by one ulp exactly when the two prompt forms agreed, failing validation after all 800 forwards |
| 2 | `REVISE 0C/2H/3I` | the campaign accumulator could brick itself — an over-cap row was written and *then* raised, after which every later load raised forever; the reader's key set was still a hand-written twin, so one-sided drift stayed invisible |
| 3 | `REVISE 0C/1H/3I` | the GPU wrapper `mkdir`-ed the no-clobber namespace and wrote the entry marker **before any code read an authorization flag**, so a submission in the repository's normal `gpu_authorized: false` state would have made the CPU preflight refuse that namespace forever; `reconcile-terminal` was seal-dependent, so exactly the clean-breach path the budget guard exists to create left the ledger holding a 7200 s reservation forever |
| 4 | `REVISE 0C/1H/0I` | the stage-authorization schema still pinned the reconciliation `provisional_gpu_usage_sha256` to 64-hex while the code and the final-state schema had moved to a `NO_SEAL_PUBLISHED` sentinel, so the seal-free path was unsatisfiable — the intersection of code and schema was empty |
| 5 | **`GO 0C/0H/0I`** | round-4 High closed by recomputation; all eighteen prior findings re-derived at the current hashes rather than inherited |

Six of those findings are the same shape as C-1 and as the v5 and v6 defects
before it: **an irreversible step taken before the check that would reject the
run.** That is now the failure mode this implementation is explicitly built and
reviewed against.

The round-5 reviewer recorded two non-blocking observations that are accepted
rather than repaired, because repairing them would move the reviewed bytes and
void the `GO`:

1. `NO_SEAL_SENTINEL` is *defined* in both `..._v7_common.py` and
   `..._v7_gpu_ledger.py` rather than imported. The frozen bytes agree, both
   files are inside `config_contract_sha256`, and the fixture binds the
   common-side spelling to the schema — but a future one-sided edit would
   resurface the round-4 High after the GPU is spent. **Any v8 must import it.**
2. Disabling the template-equality check inside
   `assert_teacher_visible_containment` turns no fixture red, because the
   existing tamper fixture's injected text itself contains a banned token. The
   check is verified correct as frozen and is independently exercised against
   all 800 real renderings; a tamper fixture using no banned token would close
   the coverage gap.

## Execution state at the authority snapshot

`preflight_materialization_authorized` and `implementation_authorized` are
`true`; every teacher/GPU/SLURM/reconciliation authorization remains `false`.
`code_resource_verdict` is `GO` pinned to
`09669bed4816e92b0b9df417ed6cd9bc288fc0f523d10358edf34a48821e6377`; the payload,
GPU-execution and reconciliation pins remain sentinels that
`_verified_review_file` rejects with `HALT_REVIEW_LINEAGE`.

The authority flip is contract-neutral, verified directly: the config contract
is `ed9cc74d16dbcd6ab2cdda9e1a8243cce5c44328807be07ee100341700599707` both
before and after setting `preflight_materialization_authorized`, the verdict and
the pin.

The v6 tree is untouched — all 15 v6 implementation hashes verify unchanged.
The unified pilot gate, the `+0.030/+0.030` two-dataset target and the
amendment's full-bank `+0.050/+0.050` DIRECT-OOF and STUDENT-OOF gates are
untouched and unwaived. No metric, result or CONTINUE/KILL verdict is published
by anything in this record.

## The two GPU ledgers, and which one is authoritative

There are two, they are not redundant, and the pointer direction matters. A
future agent that "reconciles" them backwards would silently destroy the
campaign's accounting.

| file | scope | mutability | role |
|---|---|---|---|
| `artifacts/c04/a0t_small_v1_impl_v7/resource/gpu_ledger.json` | this implementation namespace only | frozen at preflight, then mutated only by `claim`/`mark-exit`/`reconcile-terminal` for **this one allocation** | the **frozen historical snapshot** of a single tranche. Its genesis bytes are hash-pinned by `resource_ticket.json::genesis_gpu_ledger_sha256` and by the preflight manifest's `staged_output_hashes`. It dies with the namespace. |
| `artifacts/c04/campaign/gpu_ledger.json` | **every C04 GPU job, across every implementation version** | append-only, one sacct-derived row per terminal GPU job, chained by `previous_payload_sha256` → `row_payload_sha256` from `GENESIS` | the **authoritative accumulator** for the amendment's aggregate ceiling. It deliberately outlives every namespace. |

**Pointer direction, stated so it cannot be inverted:** the campaign file reads
*from* sacct and never *from* the namespace ledger; the namespace ledger never
reads the campaign file. A namespace ledger is evidence about one allocation; the
campaign file is the only place the aggregate lives. **Never overwrite,
regenerate or "sync" the campaign file from a namespace ledger** — a namespace
ledger knows only its own allocation, so doing so would erase every prior
version's spend and reopen the ceiling.

The campaign file could not have been the namespace ledger extended in place: that
file's genesis bytes are pinned inside a no-clobber tree, so extending it would
invalidate the resource ticket and the preflight manifest simultaneously. This
placement is approved by the team lead (2026-07-31).

The two ceilings are also not the same number. `phase_cap_gpu_seconds` is
**7200 s** and `aggregate_cap_gpu_seconds` is 28800 s; the effective ceiling is
`min(...)`, i.e. **7200 s today**. The amendment binds the approved first tranche
at "an aggregate maximum of 2 GPU-hours across both datasets and all C04 jobs";
the 8 GPU-hour figure binds only after `PASS_C04_SMALL_V2`, a fresh independent
result-to-claim `GO` and a new code/resource review advance the phase. The
independent payload reviewer checked this reading against the amendment text and
confirmed it; the team lead ruled the same way (2026-07-31): enforcing 7200 s
today is not a deviation from the contract, it *is* the contract.

## The failure family, and the check that closes it

This lineage has now produced the same defect three times. All three are the same
shape: **an irreversible resource is consumed before the check that would reject
the run.**

| # | where | what was consumed first | what would have rejected it |
|---|---|---|---|
| 1 | v5 static gate | an ~8-hour SLURM hold and a queue slot (job `13805`) | `verify_static_config` asserted config↔computed prompt-hash equality *before* the freeze that produces those hashes — the run could never pass its own first gate |
| 2 | v6 `mark_exit` | the genesis ledger's revision and state, permanently breaking the resource ticket's `genesis_gpu_ledger_sha256` pin inside a no-clobber namespace | a claim-time HALT — which converted a clean pre-claim refusal into a wedged run |
| 3 | v6 render defect | the single-use resource ticket, the allocation entry marker, the no-clobber namespace, and 7B weights resident on an A100 | `str.format` on a template containing literal JSON braces — raised `KeyError` on the **first** forward |

Instance 3 is the instructive one, because **two independent safeguards passed it
through**: 25 self-test checks all green, and a competent human line-read by an
independent payload reviewer who quoted `producer.py:768` accurately. Neither
executed it. The five v7 code/resource review rounds then found six more of the
same shape, including a GPU wrapper that `mkdir`-ed the no-clobber namespace
before any code read an authorization flag, and a campaign ledger that bricked
itself by writing an over-cap row and then raising.

**The check, stated so the next lineage can apply it mechanically:**

> Before any single-use ticket is consumed or any no-clobber namespace is
> entered, a **zero-cost dry execution must exercise the first real operation of
> the payload path** — not a check that the operation exists, not a reading of
> its call site, but the operation itself, on real inputs, at zero resource cost.

Concretely, the v7 template for that check:

- render **every** prompt the tranche will send — all 400 identifiers × 2 forms —
  in the CPU preflight and in a login-node dry run, before the model is loaded.
  This is what caught defect 3, and it measured **800 renderings, 0 halts**;
- round-trip **every record type a later stage re-reads** against that stage's
  actual schema and key set (`downstream_contract_fixtures`). This is what would
  have caught two of the five round-1/round-2 Criticals at zero cost instead of
  after the A100 was spent;
- **execute** each negative fixture rather than asserting it — a check that
  cannot fail under any mutation of the thing it guards is not a check. Both v6
  tautologies (`selection_deterministic`, and the `role_*_shape` family the v7
  payload reviewer filed as I-1) are of this kind.

A corollary worth stating plainly, because it is the part that is easy to
rationalise away: **"the code has been read by a careful independent reviewer" is
not evidence that the code runs.** Defect 3 survived exactly that.
