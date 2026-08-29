# C04-A0T-SMALL-v1 v7 — Fresh Independent Static Review Request

Date: 2026-07-31
Stage: `CPU_PREFLIGHT` code/resource review of implementation-v7
Requested verdict granularity: `GO / REVISE` with `nC / nH / nI` severity counts
Execution authority conferred by this review: **none**

You are a fresh independent reviewer. You have not seen the reasoning that
produced these files and you should not seek it out. Review the frozen bytes
against the contract below.

## Absolute reviewer boundary

- **Do not submit, hold, release, requeue or cancel any SLURM job.** `squeue`
  and `sacct` read-only are permitted.
- **Do not run any GPU, teacher, model-weight or frame-decode work.**
- **Do not create, modify or delete any file under `/data/jehc223/RGCL`** other
  than the single review file you are asked to write.
- **Do not materialize any dataset label value.** If you read either ASR file,
  decode only `id`, `window_text` and `language`. HateMM identifiers are
  themselves label-bearing; treat them as identifiers, not as labels to reason
  from.
- Work from a scratchpad outside the repository with `PYTHONDONTWRITEBYTECODE=1`
  set, so no `.pyc` is written anywhere. Prefer static `ast` parse or a
  byte-identical scratchpad copy over importing a module from its repository
  path.
- `artifacts/c04/a0t_small_v1_impl_v7/` must not exist when you finish, and you
  must not create it.

## Files under review (frozen; re-verify every hash against disk)

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
| `configs/c04/c04_a0t_small_v1_v7.json` | `0af5b6bdc12eb641571de199b02530d31277343b666458aebb0f36265086dfcd` |
| `artifacts/c04/campaign/gpu_ledger.json` | `fc6ca12c32427625d0b80c16b7802ef9a574ced0dbf0288edc3938d217267414` |

## Round 2 — what changed since the round-1 review

Round 1 returned `REVISE (2C / 2H / 3I)`. All eight findings were accepted and
repaired; nothing was argued away. Re-test each repair independently, and treat
the round-1 findings as **unproven again** until you have re-derived them.

- **C-A** — the reconciler read `seal/provisional_gpu_usage.json` with strict key
  equality against a key set the writer had outgrown. Both halves now build and
  validate through single shared constants, and a fixture asserts they agree.
  Check that the two really cannot drift, and that the fixture is not vacuous.
- **C-B** — `proposition_cosine` could exceed the canonical schema's `maximum: 1`
  by one ulp exactly when the two prompt forms agreed. Verify the clamp is
  correct, that it changes no comparison outcome against the 0.80 agreement
  floor, and that the fixture proving the bound is real actually fails without it.
- **H-A** — the campaign accumulator's write side was reachable only after a
  successful seal. There is now a distinct `campaign-record` ledger mode keyed
  off `resource/allocation_claim.json`, run before `reconcile-terminal` in the
  same CPU allocation. Determine whether **every** path that burns GPU-seconds
  now reaches it, including exit 40, a watchdog TERM, an OOM and any HALT.
- **H-B** — the in-job guard is now anchored to the allocation entry marker's
  `/proc/uptime` reading and held `resources.guard_item_margin_seconds` ahead of
  the wrapper `timeout`, and no longer consults `SLURM_JOB_START_TIME` at all.
  Verify the guard now always leads the wrapper, that the margin covers a
  worst-case item, and that a breach record is therefore reachable.
- **I-C1** — the accumulator is phase-scoped. Confirm the effective ceiling is
  the tighter of the phase and aggregate caps, that it is 7200 s today, and that
  the phase cannot advance without an authorization the code refuses to invent.
- **I-C2** — assess whether the H-B margin actually buys enough headroom that a
  terminal sacct elapsed can no longer approach the hard 7200 s ceiling.
- **I-C3** — the CPU preflight now round-trips records against the downstream
  contracts that read them. Judge whether the coverage is real and whether the
  blind spot is genuinely closed, or merely narrowed.

Also re-check the two round-1 non-blocking observations: the pre-model-load
containment pass now assembles messages and extracts through the same path as
the per-forward call site, and `maps.expected_hashes` remains protected only by
inclusion in the contract hash.

## Round 3 — what changed since the round-2 review

Round 2 returned `REVISE (0C / 2H / 3I)` and confirmed both round-1 Criticals
closed. All five new findings were accepted and repaired. Re-derive each.

- **H-1** — the accumulator could brick itself: an over-cap row was written and
  *then* raised, after which every later load raised forever. Loading is now
  accounting and never refuses; every rejecting check runs before the write;
  an over-ceiling total is recorded with a flag, and refusal happens on the next
  allocation. Verify no code path can raise after the file is replaced, and that
  a recorded over-run really does refuse the next reservation.
- **H-2** — the reader now imports and uses `PROVISIONAL_USAGE_KEYS` and
  `BUDGET_GUARD_KEYS` rather than restating them. Verify one-sided drift is now
  impossible by construction, not merely detected.
- **I-1** — `cosine` moved into the module the CPU preflight actually exercises,
  with fixtures over the clamp. Verify that deleting the clamp now turns a
  fixture red.
- **I-2** — a new `resources.guard_seal_reserve_seconds` is required before the
  post-loop canonicalization and seal phase begins. Verify the reserve is
  actually large enough for that phase and that an insufficient remainder stops
  cleanly with a breach record instead of a SIGTERM.
- **I-3** — `campaign-record` now falls back to `allocation_entry_marker.json`
  when no claim was published. Verify that marker exists on every path where a
  GPU allocation was entered, and that the fallback cannot fabricate a row.

Treat every claim above as unproven until you re-derive it.

## Round 4 — what changed since the round-3 review

Round 3 returned `REVISE (0C / 1H / 3I)` and confirmed every round-1 and
round-2 finding closed. All four new findings were accepted and repaired.

- **H-1 (round 3)** — the GPU wrapper `mkdir`-ed the no-clobber namespace and
  wrote the entry marker before any code read an authorization flag. It now
  carries a `jq -e` authorization gate and a frozen-preflight-manifest existence
  test **ahead of** the `mkdir`. Verify no irreversible step precedes the gate,
  including via the EXIT trap armed earlier in the script.
- **I-1** — `watchdog_reserve_seconds` is 120 → 300, and an over-cap terminal
  elapsed is recorded and flagged rather than refused (bounded by a new
  `TERMINAL_SECONDS_HARD_MAX`; the final-state schema was widened and gained
  `terminal_elapsed_exceeded_cap`). Verify the headroom arithmetic and that a
  marginal over-run is now publishable.
- **I-2** — `verify_reconciliation_lineage` has a seal-free tail, and a terminal
  resource state is published on every terminal path with `seal_published`
  recorded. Verify the exit-40, 124/137/143, OOM and post-claim-HALT paths all
  reach a published final state, and that the seal-free path cannot forge one.
- **I-3** — `GPU_LEDGER_KEYS`, `ALLOCATION_CLAIM_KEYS` and
  `RESOURCE_TICKET_KEYS` are shared constants asserted on both the writer and
  reader side, and full `prompt_record` / `canonical_record` round-trip fixtures
  were added (four reliability regimes plus the zero-frame case, with a
  non-vacuity check). Verify no writer/reader key-set contract remains
  duplicated, and that the new fixtures fail when a record is malformed.

Treat every claim above as unproven until you re-derive it.

## Round 5 — what changed since the round-4 review

Round 4 returned `REVISE (0C / 1H / 0I)` and confirmed every earlier finding
closed except one it relocated. That single High is repaired.

- **H-1 (round 4)** — `stage_authorization.schema.json` still pinned the
  reconciliation `payload_binding.provisional_gpu_usage_sha256` to 64-hex while
  the code and the final-state schema had moved to admitting a
  `NO_SEAL_PUBLISHED` sentinel, so the seal-free reconciliation path was
  unsatisfiable and no terminal resource state could ever be published on a
  breach, watchdog-kill, OOM or post-claim-HALT path. The `anyOf` is now
  mirrored into the stage-authorization schema, and two fixtures round-trip a
  full reconciliation manifest through `validate_schema` in **both** seal
  regimes while confirming a foreign pin is still refused.

Verify the two schemas that describe this one field now agree, that both
regimes are genuinely satisfiable end to end against the real reader, and that
no third description of the same field remains anywhere.

## Authority chain (frozen, read-only, not under review)

- `refine-logs/C04_USER_AMENDMENT_V2.md` — the user-approved bounded-teacher
  amendment. This is the binding resource and scope contract.
- `refine-logs/C04_REFINED_PROPOSAL_V4.md`, `refine-logs/C04_EXPERIMENT_PLAN_V4.md`,
  `refine-logs/C04_V4_DESIGN_REVIEW.md` — the frozen design (`GO 0C/0H/0I`).
- `refine-logs/C04_A0T_SMALL_V1_V6_PAYLOAD_REVIEW.md` — the predecessor payload
  review (`GO 0C/0H/3I`) whose three Important findings v7 exists to close.
- The v6 implementation set (`*_v6_*`) and `artifacts/c04/a0t_small_v1_impl_v6/`
  are the predecessor. **They must not be modified.** Confirm they are not.

## What v7 claims, and what you must independently test

v7 is a full namespace rebuild. It claims to be byte-identical to v6 modulo the
`v6`→`v7` version-token rename, plus exactly four repairs. Test each claim; do
not accept any of them on assertion.

### Claim 0 — no scientific semantic changed

The selection rule, tag, suffix, per-dataset count, prompt **text** and its four
SHA-256 values, prompt-hash rule, frame rule, transcript normalization and cap,
reliability thresholds and the five-rate KILL taxonomy, fallback semantics,
role/JL map construction, resource caps and every authorization flag must be
unchanged in meaning from v6.

Strongest available test: re-derive the 200+200 selection from the v7 frozen
rule and confirm it reproduces the **v6 frozen allowlists** exactly. Also
recompute the four prompt hashes from the v7 sources and confirm they equal the
v6 frozen values. Diff the v7 tree against v6 modulo the version token and
account for **every** residual changed line.

### Claim C-1 — the prompt renderer

v6 rendered prompts with `PROMPTS[form].format(transcript=...)`. v7 replaces
that with a dedicated renderer. Determine for yourself:

1. whether the v6 form could ever have succeeded, given the frozen prompt text;
2. whether the v7 renderer produces exactly the substitution the frozen design
   intends, with every other character of the template literal;
3. whether any prompt byte changed (the four frozen hashes are the test);
4. whether any `.format(transcript=` call site survives anywhere in v7;
5. whether the guard rails inside the renderer are non-vacuous.

### Claim I-1 — teacher-visible containment is a fail-closed precondition

The amendment requires the teacher to see no label. v7 must make "no identifier
and no label-bearing string reaches any teacher-visible field" a **checked**
precondition that fires **before any teacher forward**, not a property of prompt
assembly. Test:

1. that the check runs before the model is loaded, and again per item before
   each forward;
2. that it is strict in both directions — an unrecognised message role, content
   part, payload type, frame count, or a frame carrying a string/path must
   raise, so a future edit adding a teacher-visible field cannot escape;
3. what exactly is banned, and whether the ban is wide enough (consider
   cross-item leakage, NFKC and case folding);
4. that it does **not** false-positive on the real 400 transcripts — verify this
   yourself, label-blind;
5. that the check cannot pass vacuously.

State explicitly in your review whether the HateMM ID-label asymmetry is handled
correctly: HateMM identifiers are `hate_video_*`/`non_hate_video_*` and therefore
*are* the label, so the sealed ID-only allowlist gives label containment for
MHC-ZH only.

### Claim I-2 — the selection self-test is a known-answer vector

v6's `selection_deterministic` compared a pure function to itself. Confirm the
replacement is a genuine known-answer test: that the pinned digests are literals
independent of the module's own code path, and that mutating the tag, the
suffix, the dataset term or the identifier breaks it.

### Claim I-3 — both ceilings are machine-checked and fail-closed

**Tranche ceiling (2 GPU-hours, 7200 s).** One absolute deadline computed once
at job start and never recomputed. The guard may only ever STOP work **before an
item begins** — it must never truncate, shorten or alter an output. A breach must
produce an accounting-only record (no metric, no teacher output, no reliability
rate, no CONTINUE/KILL verdict) and a **distinct** exit code, propagated
distinctly by the wrapper. Verify by tracing where the guard is and is not
called, and what state a breach leaves on disk.

**Campaign ceiling (8 GPU-hours, 28800 s).** The amendment's aggregate ceiling
covers "every GPU-second consumed by the first tranche and any later C04
extraction/adaptation job". Verify that the accumulator is checked at job start
**before** the single-use resource ticket is consumed, that a missing, malformed,
foreign, mis-capped or chain-broken ledger halts rather than defaulting to zero,
that no stage can create or reset it, that the write side is idempotent and
sacct-derived, and that its opening zero is evidence-backed rather than assumed
(check `sacct` yourself for jobs 13805 and 13840).

## Additional checks

- The `--time` directive, job arrays, dependencies, chained submission, release
  and resubmission must be absent from every v7 wrapper and sbatch file. The only
  `--gres` may be in the GPU producer sbatch. The preflight sbatch must request
  no GPU.
- Resources must be exactly 1 GPU / 8 CPU / 64 GB.
- No OCR entrypoint, no network or external API client, no dev/test path, no
  cross-dataset path, no label reader.
- Every authorization flag in the config must be in the correct pre-review
  state, and every review pin that is not yet earned must be a sentinel that the
  code rejects.
- Look specifically for the failure family this project has now hit three times:
  **an irreversible resource consumed before the check that would reject the
  run.** Enumerate any remaining instance you find.

## Deliverable

Write exactly one file, `refine-logs/C04_A0T_SMALL_V1_V7_CODE_RESOURCE_REVIEW.md`,
containing your verdict (`GO` or `REVISE`), severity counts, and one section per
claim recording what you recomputed and what you found. Report findings as
Critical / High / Important with the reasoning that makes each one actionable.
Do not soften a finding to reach `GO`, and do not accept a claim you did not
test.
