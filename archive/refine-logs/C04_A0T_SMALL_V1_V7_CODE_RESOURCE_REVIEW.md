# C04-A0T-SMALL-v1 v7 — Fresh Independent Code/Resource Review

Reviewer: fresh independent static reviewer (no prior exposure to the authoring reasoning)
Date: 2026-07-31
Stage reviewed: `CPU_PREFLIGHT` code/resource review of implementation-v7
Execution authority conferred by this review: **none**

## Verdict

**REVISE — 2 Critical / 2 High / 3 Important (2C / 2H / 3I)**

Both Criticals are instances of the exact failure family the request asked me to
hunt: **an irreversible resource consumed before the check that would reject the
run.** In v7 the defect has moved one stage past the v6 render defect — the run
now gets *further*, spends the whole A100 allocation, and only then hits a gate
it cannot pass. Neither Critical is reachable by the CPU preflight self-test, so
neither would surface before the GPU is spent.

Everything the request asked me to verify about Claim 0, Claim C-1 and Claim I-2
reproduced exactly. Claim I-1 is substantially delivered. Claim I-3 is where the
findings concentrate.

## Method and reviewer-boundary compliance

- No SLURM job submitted, held, released, requeued or cancelled. `sacct` and
  `squeue` used read-only only.
- No GPU, teacher, model-weight or frame-decode work. Model *weights* were never
  read or loaded; only the 350-byte `preprocessor_config.json`,
  `generation_config.json` and `chat_template.json` metadata files were opened,
  plus a static read of the installed `transformers` source.
- No file under `/data/jehc223/RGCL` was created, modified or deleted other than
  this review file. Verified after every write-capable probe:
  `artifacts/c04/campaign/gpu_ledger.json` still hashes to
  `e84517d69ce7aa9a87c600b920882b1e19f118385fb571cc38acd544560dd14e`.
- `artifacts/c04/a0t_small_v1_impl_v7/` does not exist and was not created.
- No dataset label value was materialized. The ASR files were decoded only
  through the frozen `project_train_asr_line` projector (`id`, `window_text`,
  `language`). HateMM identifiers were treated as contained identifiers: they
  were compared and hashed, never printed and never reasoned from as labels.
- All work in `…/scratchpad/review-r1`, outside the repository, with
  `PYTHONDONTWRITEBYTECODE=1` on every invocation. `common.py` was imported from
  a byte-identical scratchpad copy, never from its repository path.

### Hash verification (all 17, re-verified against disk)

Every pinned SHA-256 in the request table matches disk exactly. All 15 entries of
`configs/c04/c04_a0t_small_v1_v7.json → implementation_hashes` match disk and
match the request table. All 15 entries of `frozen_design_hashes` verified with
`sha256sum -c`: 15/15 OK.

### v6 predecessor unmodified

The v6 sources and `artifacts/c04/a0t_small_v1_impl_v6/` are untouched. Strongest
available positive evidence: the v6 `freeze/preflight_manifest.json` is
self-consistent (`payload_sha256` reproduces) and **all 14** of its
`staged_output_hashes` still verify byte-for-byte against the v6 artifacts on
disk. All v6 artifacts carry a single mtime of `2026-07-31 05:11` (job 13840),
predating every v7 source file (`20:4x`).

---

## Claim 0 — no scientific semantic changed: **CONFIRMED**

### Selection re-derived from the v7 frozen rule reproduces the v6 frozen allowlists exactly

I re-read both train ASR files through the v7 projector, ranked by the v7
`selection_digest` with the v7 tie-break, and took the first 200:

| dataset | train N | recomputed order == v6 frozen allowlist | sha256(ordered id list) |
|---|---|---|---|
| HateMM | 744 | **True** | `6ff2917b4eaba00c7b92828be6f614cf5eb1b3c1fe8f728aabbd88eaadc76a5a` (identical for both) |
| MHC_zh | 579 | **True** | `2688edcb5a4f0beb228ee7d02c4bb47a20bf462a362f963ca3438e532cef86d3` (identical for both) |

Also reproduced, per dataset: every stored `selection_sha256` in the v6
allowlist (200/200), contiguous ranks 0..199, and — against the v6
`source_manifest.json` — all 200 `transcript_sha256` values and all 200
`transcript_scalar_count` values. That last check independently pins the
transcript normalization and the head/tail cap as unchanged.

### Prompt hashes recomputed from the v7 sources

```
system   1ffc06755b18f9315dc930aaf02a0c79cf1f9e2d91d0ec5bae2fa0c1436ed048
A        cecb3555daa7a4cc0840db154086138dae9260795bde2b64e89f6724d80abb9b
B        9521bee7fe7a6964d4ff61605bbe5a02eb4cfd149fd5aa9cbf975accbfe40314
combined a42268e4c0d1afb2b9576dd968d7f6250fc989fe3b97077a55363e63e1e2bb8a
```
Equal to the in-module fixture literals **and** to the values in the v6 frozen
artifact `artifacts/c04/a0t_small_v1_impl_v6/freeze/prompt_hashes.json`. No
prompt byte changed.

### Version-token-normalized tree diff — every residual line accounted for

I copied both trees to the scratchpad, applied `s/v6/vX/g; s/V6/VX/g` and
`s/v7/vX/g; s/V7/VX/g`, and diffed file-by-file:

| file | changed lines | accounted for by |
|---|---|---|
| all 5 schemas | 0 | — |
| `*_preflight.sh`, `*_preflight.sbatch`, `*.sbatch`, `*_reconcile.sh`, `*_reconcile.sbatch` | 0 | — |
| `preflight.py` | 21 | I-3 only (campaign cap/path assertions + headroom call) |
| `gpu_ledger.py` | 68 | I-3 only (campaign imports, headroom at claim, `record_campaign_gpu_spend`) |
| `*_v7.sh` | 18 | I-3 only (exit-40 branch + breach-record jq + zero-exit breach guard) |
| `common.py` | 468 | C-1 `render_prompt` + I-1 containment block + I-3 campaign ledger + I-2 fixture replacement + new fixtures. **Pure additions** apart from the single I-2 fixture swap. |
| `producer.py` | 386 | C-1 call-site swap at `build_messages` + I-1 precondition/per-item check + I-3 `BudgetGuard`/breach + campaign assertions |
| `config.json` | 132 | version tokens, `v7_scope` prose, `campaign_aggregate_cap_gpu_seconds`, `paths.campaign_gpu_ledger`, `paths.budget_breach`, refreshed `implementation_hashes` |

Nothing outside the four declared repairs changed. `SYSTEM_PROMPT`,
`_SCHEMA_TEXT`, `PROMPT_A`, `PROMPT_B`, `SELECT_TAG`, `SELECT_SUFFIX`,
`SELECT_N`, `NUM_FRAMES`, the transcript cap/head/tail/separator, the confidence
and cosine thresholds, the five-rate KILL taxonomy, `render_slot`,
`build_slot_reliability`, `materialize_role_map`, `dense_rademacher_payload`,
the resource caps and every authorization flag are untouched.

### Self-test surface

`self_test_fixtures()` now returns **37** checks (v6: 25). All 37 pass on the
frozen bytes.

---

## Claim C-1 — the prompt renderer: **CONFIRMED, and the v6 form provably could never have succeeded**

1. **Could the v6 form ever have succeeded?** No. Executed against the frozen
   template text:
   ```
   PROMPT_A.format(transcript="X")  ->  KeyError: '"source_relation"'
   PROMPT_B.format(transcript="X")  ->  KeyError: '"source_relation"'
   ```
   and even supplying that field yields `KeyError: '"S"'` from the confidence
   literal. There is no argument set under which `str.format` returns; the
   defect is unconditional on both forms.
2. **Does v7 produce exactly the intended substitution?** Yes. For both forms and
   for a transcript containing non-ASCII, CJK, a newline and a literal
   `{braces}` sequence: `render_prompt(form, t) == PROMPTS[form][:-len("{transcript}")] + t`,
   with the prefix byte-identical and the tail byte-identical. A transcript that
   itself contains the literal `{transcript}` is substituted once only (the
   `count == 1` guard is on the *template*, and `str.replace` on the single
   trailing occurrence). The module-level templates are not mutated.
3. **Did any prompt byte change?** No — see Claim 0; all four hashes match the v6
   frozen artifact.
4. **Any surviving `.format(transcript=` call site?** None in the v7 tree. The one
   surviving textual occurrence is inside the deliberate regression fixture
   `str_format_render`, which asserts that the v6 form still raises `KeyError`.
5. **Are the renderer's guards non-vacuous?** Mostly yes, with one observation.
   `prompt_form not in PROMPT_FORMS`, `not isinstance(transcript, str)`,
   `count != 1` and `not endswith(placeholder)` are all genuine and the fixture
   `prompt_render_rejects_unknown_form_and_non_string` exercises two of them.
   The two post-hoc guards are weaker: `rendered.startswith(prefix)` is a tautology
   given `str.replace` on a unique terminal placeholder, and
   `rendered.endswith(transcript)` is vacuously true for an empty transcript
   (measured: an empty transcript renders and passes). Neither is load-bearing —
   the real protection is the `count == 1` + `endswith` pair on the template — so
   this is an observation, not a finding.

---

## Claim I-1 — teacher-visible containment: **substantially CONFIRMED**

1. **Runs before the model is loaded?** Yes.
   `assert_teacher_visible_precondition(inputs)` is `producer.py:1624`;
   `Qwen2_5_VLForConditionalGeneration.from_pretrained` is `producer.py:1633`.
   Measured on the real tranche: **800** renderings checked before model load
   (400 identifiers × 2 forms), ban list **402** tokens.
   The check is repeated per item at `producer.py:1656`, inside `one_forward`,
   after `build_messages` and before `apply_chat_template`/`generate`.
2. **Strict in both directions?** Yes. `teacher_visible_texts` raises on: a
   message list of the wrong length, unexpected message keys, an unexpected
   role, non-list/empty content, a content part that is not a dict or lacks
   `type`, a text part with extra keys or a non-string body, a video part with
   extra keys, a frame list whose length ≠ 8, **any frame that is a `str`,
   `bytes`, `bytearray` or `Path`**, any unknown content `type`, and a final
   census requiring exactly one video part and exactly two text parts. A future
   edit that adds a teacher-visible field cannot pass. Six of these are pinned by
   fixtures.
3. **What is banned, and is it wide enough?** All 200 HateMM + 200 MHC-ZH
   selected identifiers **plus** the two HateMM label-bearing prefixes, and both
   datasets' identifiers are banned in both datasets' prompts, so cross-item
   leakage is refused as firmly as self-leakage. Matching is done over
   `{token, NFKC(token), casefold(NFKC(token))}` against both `NFKC(text)` and
   `casefold(NFKC(text))`, so a fullwidth or case-altered identifier is caught.
   The wider protection is the equality `texts == [SYSTEM_PROMPT, render_prompt(form, transcript)]`:
   the only variable content reaching the teacher is the transcript, so the
   amendment's broader ban (prediction, neighbor, rank, margin, error status,
   dataset statistic, fold role, intended use) is structurally satisfied rather
   than enumerated. I also confirmed the pinned `chat_template.json` renders a
   video part as `<|vision_start|><|video_pad|><|vision_end|>` and discards the
   `video` value, so no path or identifier can leak through templating.
4. **Does it false-positive on the real 400 transcripts?** **No — 0 of 800.**
   I ran the frozen guard label-blind over all 400 selected transcripts × both
   forms: zero rejections. As a robustness margin I also scanned the same
   402-token ban list against **all 1323** train transcripts (744 + 579): **0**
   rows contain any banned token. Shortest banned token is 11 characters
   (`hate_video_`), so accidental substring collision is not a live risk.
5. **Can it pass vacuously?** Not at the per-forward call site. `if not forbidden
   or video_id not in forbidden: raise` makes an empty or incomplete ban list a
   halt, and the fixture `teacher_visible_unbanned_identifier_rejected` proves an
   identifier absent from the ban list is rejected. **Observation:** in the
   pre-model-load pass the `texts` argument is constructed inside
   `assert_teacher_visible_precondition` from the same `render_prompt` call the
   assertion compares against, so *that* half of the predicate is self-comparing
   there; the pre-load pass effectively verifies only "no banned token in the
   transcript". The message-assembly half becomes non-vacuous only at
   `producer.py:1656`, which does run before every forward, so the stated
   requirement is met — but the two passes are not equally strong and the review
   record should say so.

### The HateMM ID-label asymmetry — stated explicitly, as requested

**The asymmetry is handled correctly, and the code says so in the right place.**
Every HateMM training identifier is `hate_video_*` or `non_hate_video_*`; the
identifier *is* the binary label. MHC-ZH identifiers are opaque BiliBili `BV`
codes carrying no label information. Therefore the sealed ID-only allowlist
delivers label containment **for MHC-ZH only, and none at all for HateMM**.
`LABEL_BEARING_ID_SUBSTRINGS` encodes exactly this (`("hate_video_",
"non_hate_video_")` for HateMM, empty tuple for MHC-ZH), the `common.py` comment
block states the asymmetry, and `config.json → v7_scope.I1_teacher_visible_containment`
records it. The consequence is correctly drawn: for HateMM, selection
label-blindness is established by *hash reproduction of the selection rule*
(which I independently reproduced above), and teacher label-blindness by this
*runtime* check — not by the allowlist.

---

## Claim I-2 — the selection self-test is a known-answer vector: **CONFIRMED**

The v6 fixture `selection_digest("HateMM","x") == selection_digest("HateMM","x")`
was a tautology. The v7 replacement pins two literals. I recomputed both from
first principles, outside the module, with a hand-written concatenation:

```
sha256(utf8("C04-A0T-SMALL-v1" + "HateMM" + "c04-known-answer-vector" + "20260729"))
  = 871e0363e1b01a823f09a5a0bb9187749da74f1dfe8e454a733e21c218f6a384   [matches]
sha256(utf8("C04-A0T-SMALL-v1" + "MHC_zh" + "c04-known-answer-vector" + "20260729"))
  = 41bfb637f5cbb26bb0b9edfd44d19a3775d8df1712f9b49bde667863fbd37134   [matches]
```

The digests are literals independent of the module's own code path. Mutation
sensitivity, measured: tag → breaks; suffix → breaks; dataset term → breaks;
concatenation order → breaks; identifier → breaks. The companion fixture
`selection_dataset_and_id_sensitivity` additionally requires three distinct
digests. The chosen identifier is synthetic and belongs to neither dataset, so
the fixture pins the rule without naming a label-bearing real id — a good choice.

---

## Claim I-3 — both ceilings machine-checked and fail-closed: **PARTIALLY DELIVERED**

### Tranche ceiling (7200 s) — guard logic correct, guard *ordering* not enforced

Traced: `BudgetGuard.at_job_start` is constructed exactly once, in
`verify_claimed_resource`; the deadline is stored and never recomputed
(`check()` only compares, `accounting_snapshot()` only reports). There are
exactly two `deadline_check` call sites — `producer.py:1705` at the item
boundary, before frame-pack creation, and `producer.py:1651` inside
`one_forward`, before `build_messages`. The guard is never invoked during or
after a unit of work and has no path that truncates, shortens or rewrites an
output. `BudgetDeadlineReached` is caught only by the outer handler, which
publishes an accounting-only record and returns 40.

The breach record was inspected field by field: it carries lineage, job id,
terminal state, exit code, the two caps, the guard accounting snapshot,
per-dataset completed counts, teacher-call and frame-pack counters,
`outputs_truncated_or_altered: 0`, `seal_published: false`,
`no_scientific_verdict_is_published_by_a_budget_breach: true`. **No metric, no
teacher output, no reliability rate, no CONTINUE/KILL verdict.** Exit code 40 is
distinct from the watchdog codes and the wrapper propagates it distinctly
(dedicated branch, jq-verifies the record, exits 40) and also refuses a zero-exit
run that left a breach record behind. On-disk state after a breach: per-item
checkpoint JSONs and frame packs intact, breach record present, GPU ledger in
`EXIT_RECORDED_PENDING_SACCT`, **no seal**.

The defect is in the *ordering* between this guard and the wrapper's `timeout` —
see **High H-B**.

### Campaign ceiling (28800 s) — read side correct, write side unreachable

**Ordering.** `assert_campaign_aggregate_headroom` is called from
`validate_gpu_environment`, which is the first statement of `claim()`, i.e.
before `create_entry_marker`, before `verify_gpu_lineage`, and well before the
allocation claim and the single-use ticket-consumption record are published. The
check is genuinely before the ticket is consumed. It is also called by the CPU
preflight before the no-clobber namespace is materialized, and by the producer
before any model or data work.

**Fail-closed matrix**, executed against scratchpad copies (repo untouched):

| mutation | result |
|---|---|
| ledger absent | HALT `campaign ledger is absent` |
| aggregate ≠ Σ rows | HALT `payload mismatch` |
| `payload_sha256` tampered | HALT `payload mismatch` |
| foreign `schema_version` | HALT `foreign campaign ledger schema` |
| foreign `run_id` | HALT `foreign campaign ledger run id` |
| cap raised to 999999 | HALT `cap is not the amendment cap` |
| cap lowered to 7200 | HALT `cap is not the amendment cap` |
| row chain break (`previous ≠ GENESIS`) | HALT `chain break` |
| well-formed 25000 s already spent | HALT `would take the C04 campaign to 32200s` |
| well-formed 21601 s already spent | HALT `would take the C04 campaign to 28801s` |
| well-formed 20000 s already spent | accepted (27200 ≤ 28800) — correct |

It never defaults to zero. **No stage can create or reset it**: the only writer
is `append_campaign_gpu_job`, which calls `load_campaign_gpu_ledger()` first (so
it cannot bootstrap from nothing), and the preflight explicitly verifies the path
rather than staging it — the campaign path lies outside `ARTIFACT_ROOT` and would
be rejected by the staging loop's namespace check.

**Write side is idempotent and sacct-derived**: `record_campaign_gpu_spend`
verifies an already-present row instead of appending (so a recovery
reconciliation cannot double-count) and every numeric field originates from
`sacct_row` with `accounting_source: "sacct"`.

**Opening zero is evidence-backed.** I checked `sacct` myself, read-only:

```
13805|c04_a0t_small_v1_v5_preflight| 0|billing=8,cpu=8,mem=64G,node=1|FAILED
13840|c04_a0t_small_v1_v6_preflight|19|billing=8,cpu=8,mem=64G,node=1|COMPLETED
```

Both rows match the ledger's `genesis_evidence` verbatim, including the
`alloc_tres` strings, elapsed seconds and states. Neither carries `gres/gpu`, so
`gpu_seconds: 0` is correct for both. A full `sacct` sweep for `c04` job names
returns exactly these two rows, corroborating
`these_are_the_only_c04_jobs_in_the_accounting_record: true`. The live ledger
loads and fully verifies: aggregate 0 s, cap 28800 s, 0 jobs, head `GENESIS`,
revision 0.

The read side is therefore sound. The write side is not reachable — see
**Critical C-A** and **High H-A**.

---

## Additional checks

- **`--time` directive:** absent from all three sbatch files and all three
  wrappers. Each sbatch carries an explicit comment that its omission is
  deliberate.
- **Arrays / dependencies / chained submission / release / resubmission:**
  absent. There is no `sbatch`, `scontrol`, `scancel`, `srun` or `salloc`
  anywhere in the v7 set. The *only* `subprocess` call in the entire tree is
  `gpu_ledger.py:227`, `sacct -X -n -P -j <id> -o JobIDRaw,ElapsedRaw,AllocTRES,State`
  — read-only. All three wrappers and both Python entrypoints reject
  `SLURM_ARRAY_JOB_ID` / `SLURM_JOB_DEPENDENCY`.
- **`--gres`:** exactly one occurrence, `scripts/slurm/c04_a0t_small_v1_v7.sbatch:3`,
  `--gres=gpu:a100:1`. The preflight sbatch requests no GPU. The reconcile sbatch
  requests no GPU, and its wrapper additionally rejects a non-empty
  `CUDA_VISIBLE_DEVICES` or `SLURM_GPUS_ON_NODE`.
- **Resources:** GPU sbatch = 1 GPU / 8 CPU / 64 GB; preflight sbatch = 8 CPU /
  64 GB, no GPU; reconcile sbatch = 1 CPU / 4 GB, no GPU. `resources.gpu_count/cpus/ram_gb`
  = 1/8/64 and are asserted in the preflight, the GPU ledger and the producer.
- **No OCR entrypoint, no network/API client, no dev/test path, no cross-dataset
  path, no label reader:** confirmed by targeted grep. No `requests`, `urllib`,
  `http`, `socket`, or any OCR library is imported anywhere. The only `label`
  reference on the data path is `_skip_json_value`, which advances the parser
  past the `label` token syntactically and increments a skip counter without
  converting it to a Python value; the projector then requires the decoded key
  set to be exactly `{id, window_text, language}`. `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` are exported by the wrapper and *asserted* by the
  producer, and both `from_pretrained` calls pass `local_files_only=True`.
  `root_path` rejects any `dev`/`test`/`validation`-like path component.
- **Authorization flags in the correct pre-review state:** exactly one flag is
  `true` (`implementation_authorized`); all sixteen others, including
  `preflight_materialization_authorized`, `teacher_authorized`, `gpu_authorized`,
  `slurm_authorized`, `small_tranche_execution_authorized` and
  `post_job_reconciliation_authorized`, are `false`. The preflight wrapper
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
  | `resolve_prompt_hashes(freeze=True)` with materialization false | HALT (same) |

- **Config contract normalization is exactly as narrow as documented.** Filling
  the four prompt-hash keys in does not move `config_contract_sha256` (verified),
  so the v5 impossibility is genuinely closed rather than displaced. Mutating
  `selection.suffix`, `resources.small_cap_gpu_seconds`,
  `resources.campaign_aggregate_cap_gpu_seconds`, `maps.expected_hashes`,
  `teacher_contract.num_frames` or `model.snapshot_revision` *does* move it.
  Only the four review pins, the four review verdicts, the `authorization` block
  and `prompt_hashes` are normalized out — each of which is separately bound by a
  strict stage-authorization manifest. Minor observation: `maps.expected_hashes`
  is protected only by being inside the contract hash; no code asserts its literal
  value, so the prose invariant in `prompt_hash_contract` is not machine-checked.
- **Environment:** the pinned interpreter
  `/data/jehc223/miniconda3/envs/HateVideo/bin/python` (3.11.8) has jsonschema
  4.26.0 (Draft7Validator present), PIL 11.1.0, numpy 1.26.4, torch 2.6.0+cu124,
  transformers 4.49.0, decord 0.6.0, av 17.0.0; `jq` 1.6 is on PATH.
  `Qwen2_5_VLForConditionalGeneration` imports. `GenerationConfig.validate()`
  with `do_sample=False, temperature=0.0` warns but does not raise. Statically
  traced `make_batched_videos`: `videos=[frames]` with 8 PIL images is correctly
  interpreted as one video of eight frames. None of these produce a finding.

---

# Findings

## CRITICAL C-A — the terminal CPU reconciliation can never pass its own gate: the producer writes three fields the reconciler's exact-key check forbids

**Where.** `scripts/analysis/c04_a0t_small_v1_v7_producer.py:1840-1859` (the
`provisional` record) versus
`scripts/analysis/c04_a0t_small_v1_v7_gpu_ledger.py:436-452`
(`require_exact_keys(provisional, {...}, "provisional GPU usage")`).

**Measured (AST extraction of both key sets, v6 and v7):**

```
v6 producer provisional dict (11)  == v6 reconciler require_exact_keys (11)   MATCH
v7 producer provisional dict (14)  vs v7 reconciler require_exact_keys (11)   *** MISMATCH ***
    producer-only: budget_guard, campaign_aggregate_cap_gpu_seconds,
                   campaign_gpu_ledger_sha256_at_seal
```

v7's I-3 work added three fields to `seal/provisional_gpu_usage.json` and did not
update the reconciler's exact-key gate. `require_exact_keys` is strict equality
(`set(value) != expected → raise`), so the check fails unconditionally.

**Failure scenario.** The GPU job runs to completion, spends up to the full
2 GPU-hours on an A100, publishes the seal, and exits 0. The mandatory
`CPU_POST_JOB_RECONCILIATION` job then runs `reconcile_terminal` →
`verify_reconciliation_lineage`, which reaches the `provisional` check *before*
it touches the ledger, and raises
`HALT_…: provisional GPU usage exact-key failure`. Consequences, all of them
terminal:

- the per-namespace GPU ledger stays in `EXIT_RECORDED_PENDING_SACCT` with a
  7200 s reservation and is never reconciled to real sacct seconds;
- `resource_final_state.json` is never published, so the seal's
  `resource_final_state_required_before_any_downstream_review: true` and the
  config's `review.downstream_review_requires_terminal_resource_state: true`
  block every downstream review permanently;
- `record_campaign_gpu_spend` — the *write half of I-3* — is never reached, so
  the 8 GPU-hour accumulator never learns that the tranche happened.

The reconcile wrapper's recovery branch does not help: it retries only when the
ledger is already `SACCT_TERMINAL_RECONCILED`, which it never becomes, so the
wrapper propagates the failure. And the run cannot be repaired by editing
`gpu_ledger.py`, because that changes its SHA-256, which changes
`config.implementation_hashes`, which changes `config_contract_sha256`, which
invalidates the `config_contract_sha256` values already pinned inside the
no-clobber `artifacts/c04/a0t_small_v1_impl_v7/` preflight manifest, genesis GPU
ledger and resource ticket. This is the v5 impossibility displaced one stage
further — onto a fully consumed A100 allocation.

**What would close it.** Add the three new keys to the reconciler's
`require_exact_keys` set (and assert their types/values there, as the other
fields are asserted), *or* move the new accounting into a separate sibling
artifact and leave `provisional_gpu_usage.json` on its v6 key set. Then add a
CPU-preflight fixture that builds the producer's `provisional` dict and feeds it
to the reconciler's key set, so writer/reader drift cannot recur silently.

## CRITICAL C-B — `proposition_cosine` can exceed the canonical schema's `maximum: 1` whenever prompts A and B agree, halting the producer after all 800 forwards

**Where.** `producer.py:1096-1101` (`cosine`), `common.py:1468`
(`"proposition_cosine": proposition_cosine`), and
`schemas/c04/c04_a0t_small_v1_v7_canonical_record.schema.json`
`definitions.reliability.properties.proposition_cosine` →
`{"type":"number","minimum":-1,"maximum":1}`, enforced by
`validate_schema(record, cfg["schemas"]["canonical_record"], …)` at
`producer.py:1353`.

**Mechanism.** When prompt A and prompt B yield the same proposition after
`normalize_proposition`, the two mean embeddings are bit-identical. `cosine`
then computes `S / (sqrt(S) * sqrt(S))`, and `sqrt(S)**2` differs from `S` by up
to one ulp, so the result can exceed 1.0.

**Measured.** Over 2000 trials using bfloat16-rounded mean embeddings at the
pinned `TEACHER_DIM = 3584` and realistic magnitudes (mimicking
`embedding(ids).mean(dim=1)[0].float().cpu().tolist()`):

```
cosine(v, v) > 1.0  in  504 / 2000 trials  (25.2%)   max observed 1.0000000000000002
```

and feeding that value into a record built exactly as `canonicalize_dataset`
builds it:

```
FAIL -> ['slots','P','proposition_cosine']: 1.0000000000000002 is not valid under any of the given schemas
```

Probability that at least one of the 400 items trips it, as a function of the
number *k* of items where A and B agree exactly: k=1 → 0.25, k=5 → 0.77,
k=20 → 0.997, k=100 → 1.000. Exact agreement is not an edge case — it is the
outcome the design *rewards* (the `stable` state and the 0.80 cosine floor exist
to detect it), and decoding is greedy (`do_sample=False`, `num_beams=1`), so
short propositions will match verbatim across the two prompts routinely.

**Failure scenario.** All 800 teacher forwards complete and are checkpointed.
`canonicalize_dataset` then raises
`HALT: canonical HateMM/<id> schema failure: ['slots','P','proposition_cosine'] …`
at the first affected item — i.e. after essentially the entire 2 GPU-hour
allocation has been spent, and **before any seal is written**. No seal means no
reconciliation (see H-A), so this also silently loses the GPU-second accounting.
Nothing in the CPU preflight can catch it: `self_test_fixtures()` never builds a
canonical record and never invokes `validate_schema` on one. This is inherited
from v6 (the schema is byte-identical), but v6 could never reach it because of
the render defect; v7 is the first version in which it is reachable, so it is a
v7 finding.

**What would close it.** Clamp the returned cosine into `[-1.0, 1.0]` in
`cosine()` (a one-line `max(-1.0, min(1.0, …))`, which changes no science — the
0.80 threshold comparison is unaffected), or relax the schema bound. Independently,
add the CPU-preflight fixture described in Important I-C so that a canonical
record built from the `stable`/agreeing case is schema-validated before any GPU
is requested.

## HIGH H-A — the campaign accumulator's write side is reachable only on the fully-sealed path, so the 8 GPU-hour ceiling keeps reading a stale zero

**Where.** `gpu_ledger.py:1272` (`record_campaign_gpu_spend`) is called only from
`reconcile_terminal`, which first runs `verify_reconciliation_lineage`, which
opens `cfg["paths"]["provisional_gpu_usage"]` — a file that lives inside
`…/seal/` and is created only by the producer's final atomic seal publication.

**Failure scenario.** Every non-sealing outcome leaves the campaign accumulator
unwritten while GPU-seconds were genuinely burned:

- **budget breach (exit 40)** — the very path I-3 introduces: the guard stops
  before an item, no seal is published, so `provisional_gpu_usage.json` does not
  exist, `reconcile_terminal` dies with `FileNotFoundError`, and up to 2 GPU-hours
  are never recorded;
- **watchdog TERM/KILL (exit 124/137/143)** — same;
- **any producer HALT, OOM, decode failure, or C-B above** — same;
- **a fully successful run** — blocked instead by Critical C-A.

Taken together with C-A, `append_campaign_gpu_job` is unreachable on *every*
path in v7. A subsequent C04 namespace (v8, or a later extraction/adaptation job)
would call `assert_campaign_aggregate_headroom` and read `aggregate_gpu_seconds:
0`, authorizing a fresh 7200 s reservation as though nothing had been spent. The
accumulator that exists specifically to survive an implementation-version bump
therefore does not, in practice, remember anything.

**What would close it.** Make the terminal accounting stage independent of the
seal: derive the original job id from `resource/allocation_claim.json` (which
exists from claim time, before any teacher work) rather than requiring
`seal/provisional_gpu_usage.json`, and treat the seal/provisional checks as
conditional refinements. The campaign row should be appended from sacct for any
terminal GPU job that consumed the ticket, sealed or not.

## HIGH H-B — the in-job tranche guard is given no margin over the wrapper `timeout` it was introduced to replace, so the exit-40 breach path is usually pre-empted

**Where.** `producer.py:407-433` (`BudgetGuard.at_job_start`) versus
`scripts/wrappers/c04_a0t_small_v1_v7.sh:101-105` (`timeout … "${C04_ACTIVE_WATCHDOG_SECONDS}s"`).

Both deadlines are set to the same budget (`cap − reserve = 7080 s`), with no
enforced ordering between them.

**Failure scenario, `SLURM_JOB_START_TIME` present** (documented as exported by
this cluster's SLURM — `man sbatch` lists `SLURM_JOB_START_TIME`; version
25.11.4): the guard takes the `SLURM_ALLOCATION_START_EPOCH` branch and its
deadline lands at `allocation_start + 7080`. The wrapper's `timeout` was started
after the SLURM prolog, the bash preamble, the jq marker write and the `claim`
call, so it fires at `allocation_start + P + δ + 7080` — only `P + δ` (a few
seconds) later. But the guard is checked **once per item**, and an item is two
7B-VLM forwards (order 10-20 s at the 8-frame/256-new-token settings). The item
that straddles the deadline therefore runs past `timeout` and is SIGTERM'd
mid-forward, exit 124, **no breach record**, before the next item-boundary check
can fire. With a few seconds of margin against ~20 s items, the clean exit-40
path is taken only a small fraction of the time.

**Failure scenario, `SLURM_JOB_START_TIME` absent** (a config change, or any
invocation of the wrapper outside `sbatch`): the guard falls back to
`TICKET_WATCHDOG_REMAINDER`, `deadline = now_monotonic + watchdog_seconds`. But
`now_monotonic` is taken when `verify_claimed_resource` runs — after
`verify_model_snapshot` has SHA-256'd the **16.60 GB** pinned model/processor
tree (≈23 s of hashing at this node's measured ~710 MB/s, plus disk read), and
after three `verify_bound_file_map` sweeps and the full lineage chain. The
guard's deadline therefore lands *V ≈ 25-60 s later* than the wrapper's timeout,
and the guard can never fire at all. The tranche ceiling reverts to exactly the
v6 behaviour the I-3 repair exists to replace: `timeout` killing mid-forward.

**What would close it.** Anchor the guard to allocation entry rather than to
guard-construction time (the wrapper already computes and passes
`--allocation-start-uptime-seconds`; pass the same value to the producer and
derive the deadline from `/proc/uptime` rather than `time.monotonic()`), and give
the guard a hard margin over the wrapper's `timeout` of at least one worst-case
item — e.g. guard deadline `= allocation_start + cap − reserve − item_margin`
with the wrapper timeout left where it is, plus a startup assertion that the
guard deadline is strictly earlier than the remaining `timeout` budget.

## IMPORTANT I-C1 — the campaign accumulator enforces the conditional-tranche ceiling, not the currently binding first-tranche ceiling

`refine-logs/C04_USER_AMENDMENT_V2.md` places the 8 GPU-hour ceiling on the
**conditional full-bank tranche**, which is unlocked only by `PASS_C04_SMALL_V2`
plus a fresh result-to-claim `GO` plus a new code/resource review. The clause
binding *now* is stricter: the first tranche must observe "an aggregate maximum
of **2 GPU-hours** across both datasets **and all C04 jobs**". `CAMPAIGN_AGGREGATE_CAP_GPU_SECONDS`
is 28800 and `load_campaign_gpu_ledger` halts unless the on-disk cap equals
28800, so the accumulator cannot express the phase-1 limit at all.

**Failure scenario, measured.** With a well-formed ledger recording that the
first tranche already spent its full 7200 s, `assert_campaign_aggregate_headroom(7200)`
**accepts** a second C04 GPU allocation (7200 + 7200 = 14400 ≤ 28800). Given this
project has already produced three implementation namespaces (v5, v6, v7), a v8
rebuild after a v7 failure is the expected case, and it would take the campaign
to 4 GPU-hours under an amendment clause that caps it at 2 until the conditional
tranche is separately authorized. The v7 run *itself* is compliant (0 + 7200 ≤
7200), so this is not a blocker for this job — it is a gap in the ceiling that
outlives it.

**What would close it.** Carry a phase-scoped cap: keep 28800 as the campaign
ceiling but add `phase_cap_gpu_seconds: 7200` with a phase token in the ledger
that only the conditional-tranche authorization may advance, and check the
reservation against `min(phase_cap, campaign_cap)`. (Separately, `append_campaign_gpu_job`
takes no lock on the shared campaign file — it is protected only by the
*per-namespace* `gpu_ledger.lock`, which would not exclude a v8 reconciler; a
lock on the campaign path itself would close that.)

## IMPORTANT I-C2 — the hard 7200 s equality leaves ~90 s of margin, and exceeding it makes reconciliation permanently impossible

`reconcile_terminal` raises `HALT_RESOURCE_CAP: terminal sacct GPU seconds exceed
7200 cap` when `sacct` `ElapsedRaw` exceeds 7200;
`strict_validate_terminal_ledger` independently rejects seconds outside
`[0, 7200]`; and the `resource_final_state` schema pins
`terminal_sacct_gpu_seconds` / `aggregate_accounted_gpu_seconds` /
`aggregate_reconciled_terminal_gpu_seconds` to `maximum: 7200`.

**Failure scenario.** The wrapper's own budget is `7080 s` of `timeout`, plus
`--kill-after=30s`, plus the EXIT-trap `mark-exit` call, all measured from wrapper
start — while `sacct` measures from *allocation* start. The margin between the
worst-case wrapper wall time (`P + 7080 + 30 + ε`) and the hard 7200 s ceiling is
therefore only about 90 s, consumed by the SLURM prolog, node setup, `conda`/bash
startup and the `mark-exit` write. If it is exceeded — most plausibly on a
watchdog-terminated run, exactly the case where the reserve matters — the ledger
can never be reconciled, `resource_final_state.json` can never be published, and
the namespace is wedged for the same no-clobber reason described in C-A. A hard
ceiling whose breach is unrecoverable should not be defended by an unmeasured
90 s.

**What would close it.** Either lower the wrapper's effective budget so the
worst-case sacct elapsed is provably under 7200 s (e.g. reserve 300 s rather than
120 s, or subtract the measured allocation-start-to-wrapper-start offset from the
timeout), or make an over-cap terminal sacct row a *recorded* over-run — written
to the ledger, the campaign accumulator and a distinct final state — rather than
an unrecoverable halt.

## IMPORTANT I-C3 — the CPU preflight self-test never validates a producer record against its own JSON Schema, nor cross-checks writer/reader key sets

The 37 fixtures cover the prompt-hash contract, the containment guard, the
renderer, the selection rule, the parser, the transcript cap and the reliability
states. They do **not** construct a `prompt_record`, a `canonical_record`, a
`frame_pack_manifest`, a `provisional_gpu_usage` record or a
`resource_final_state` record and validate it against the schema/key-set the
downstream stage will apply. `validate_schema` is exercised at preflight only
against the stage-authorization manifest.

This is precisely the blind spot that let v6's `str.format` defect through — "no
v6 self-test ever rendered a prompt" — and it is the same blind spot that lets
C-A and C-B through in v7. I verified by construction that such fixtures would
have caught both: a synthetic canonical record with
`proposition_cosine = 1.0000000000000002` fails validation immediately, and a
one-line comparison of the producer's `provisional` key set against the
reconciler's `require_exact_keys` set flags the mismatch statically.

**What would close it.** Add CPU-preflight fixtures that (a) build one
`prompt_record` and one `canonical_record` — including the degenerate cases: all
slots `missing`, all `single_valid`, and the *agreeing* `stable` case with
`proposition_cosine` at and just above 1.0 — and run them through
`validate_schema` against the frozen schemas; and (b) assert that every
producer-written artifact's key set equals the key set the consuming stage's
`require_exact_keys` demands. Both are pure-CPU, cost nothing, and would have
turned both Criticals into preflight failures.

---

## Summary table

| # | Severity | Finding |
|---|---|---|
| C-A | Critical | Terminal CPU reconciliation can never pass: producer writes 3 keys the reconciler's `require_exact_keys` forbids; blocks final state, wedges the namespace after the A100 is spent, and makes the I-3 campaign write side unreachable |
| C-B | Critical | `proposition_cosine` exceeds the canonical schema's `maximum: 1` in 25.2% of exact A/B agreements; producer HALTs after all 800 forwards, before any seal |
| H-A | High | Campaign accumulator write side reachable only on the fully-sealed path; every failure mode (incl. the new exit-40 breach) leaves the 8 GPU-hour ceiling reading a stale zero |
| H-B | High | In-job tranche guard has no enforced margin over the wrapper `timeout`; the exit-40 accounting path is usually pre-empted by a mid-forward SIGTERM, and is unreachable entirely if `SLURM_JOB_START_TIME` is absent |
| I-C1 | Important | Accumulator enforces the 8 GPU-hour conditional-tranche ceiling, not the currently binding 2 GPU-hour "all C04 jobs" first-tranche ceiling; a v8 namespace would be authorized a second 7200 s |
| I-C2 | Important | ~90 s of margin between worst-case sacct elapsed and the hard 7200 s ceiling; exceeding it makes reconciliation permanently impossible |
| I-C3 | Important | CPU preflight never schema-validates a producer record nor cross-checks writer/reader key sets — the blind spot that produced C-A and C-B |

**Verdict: REVISE (2C / 2H / 3I). No execution authority is conferred. The
`preflight_materialization_authorized` flag must remain `false` and
`review.code_resource_verdict` must remain `PENDING` until at least the two
Criticals are closed and the closing fixtures are added to the CPU preflight.**
