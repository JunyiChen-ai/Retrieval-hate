# C04-A0T-SMALL-v1 v6 CPU-Preflight Payload Review

Date: 2026-07-31
Reviewer: independent payload reviewer (Opus, fresh context, no exposure to the
v6 implementation reasoning)
Subject: the payload frozen by SLURM job `13840` under
`artifacts/c04/a0t_small_v1_impl_v6/`
Final verdict: **GO (0 Critical / 0 High / 3 Important)**

This is the fresh payload-review verdict the frozen contract requires
(`C04_A0T_SMALL_V1_V6_CPU_PREFLIGHT_UNLOCK_RECORD.md` §"Boundary after unlock":
"After the preflight terminates, an independent collector/reviewer must inspect
the frozen artifacts and issue a fresh payload-review verdict. No GPU or
downstream stage becomes authorized merely because the CPU preflight succeeds.")

## Reviewer boundary actually observed

No SLURM job was submitted, held, released or cancelled (`sacct`/`squeue`
read-only only). No GPU, teacher, model weight or frame decode was run. No file
under `/data/jehc223/RGCL` was created, modified or deleted except the four
documentation files this review is required to write
(`refine-logs/C04_A0T_SMALL_V1_V6_PAYLOAD_REVIEW.md`, `TARGET_REVIEW_RAW.md`,
`TARGET_STATE.json`, `TARGET_FINDINGS.md`, `TARGET_LOOP.md`). No frozen artifact
was touched. No dataset **label value** was materialized: every ASR read in this
review decoded only `id`, `window_text` and `language`, so the reviewer held the
same label-blindness the pipeline claims.

Verification was performed by recomputation, from a scratchpad outside the
repository, with `PYTHONDONTWRITEBYTECODE=1` set throughout, so no `.pyc` was
written anywhere (the defect the round-3 unlock reviewer filed). The frozen
`..._v6_common.py` was **not** imported from its repository path: the prompt
constants were extracted by static `ast` parse of the frozen bytes, and the
fixture spot-check ran against a byte-identical copy placed in the scratchpad
(`sha256 81b10f58…`, confirmed equal to the pinned source).

## Verdict

`GO (0 Critical / 0 High / 3 Important)`.

The frozen payload is well-formed, internally consistent, and reproduces exactly
from the pinned config and module constants. Every check below was recomputed,
not read off the job's own `all_passed`. The three Important findings are
documentation/assumption gaps that must be closed before the teacher stage; none
of them impugns a single frozen byte.

## 1. Prompt-hash freeze correctness — CLOSED, the 13805 failure mode is gone

`artifacts/c04/a0t_small_v1_impl_v6/freeze/prompt_hashes.json` carries exactly
four keys, each a literal 64-hex value, none of them the sentinel:

| key | frozen value | recomputed independently |
|---|---|---|
| `system` | `1ffc06755b18f9315dc930aaf02a0c79cf1f9e2d91d0ec5bae2fa0c1436ed048` | match |
| `A` | `cecb3555daa7a4cc0840db154086138dae9260795bde2b64e89f6724d80abb9b` | match |
| `B` | `9521bee7fe7a6964d4ff61605bbe5a02eb4cfd149fd5aa9cbf975accbfe40314` | match |
| `combined` | `a42268e4c0d1afb2b9576dd968d7f6250fc989fe3b97077a55363e63e1e2bb8a` | match |

Recomputation method: `SYSTEM_PROMPT`, `_SCHEMA_TEXT`, `PROMPT_A`, `PROMPT_B`
and the four enum tuples were extracted from the frozen source by `ast` parse
(no module import), then hashed by the rule at
`scripts/analysis/c04_a0t_small_v1_v6_common.py:1179-1195` — three raw UTF-8
digests plus one canonical-JSON digest over `{system, A, B, enum_order}`. All
four reproduce exactly.

Further checks, all passing:

- `payload_sha256` = `eb485b9af1704fef4158baa0b00efde03970fd9aec81f670d581c6b5b6620a36`
  recomputes over the body with the field removed;
- `downstream_binding` is `LITERAL_BOUND`; `config_binding_at_freeze` is
  `SENTINEL_PENDING_CPU_PREFLIGHT_FREEZE`;
- key set is exactly `{A, B, combined, system}`;
- the string `PENDING_CPU_PREFLIGHT_HASH_FREEZE` occurs in the whole 15-file
  namespace **only** in this file, and there only as the value of
  `pending_sentinel_token` (a token *name*) — never as the value of any of the
  four keys;
- the file bytes equal `canonical_json(payload) + b"\n"`, file sha256
  `c340cfc721131fff1b21f6b66522c9b1d818812418601007adf4c4d3401a0f77`;
- the four values are byte-identical to those printed in job `13805`'s stderr,
  so the v5→v6 rename perturbed no prompt byte.

The 13805 failure mode (config↔computed equality asserted *before* the freeze
that produces the values) is genuinely closed rather than bypassed: the config
still holds the four sentinels, `resolve_prompt_hashes(cfg, freeze_stage=True)`
accepted them only because `preflight_materialization_authorized` is exactly
`True`, and the artifact it produced holds no sentinel at all. The job's own log
records `config_prompt_hash_binding = SENTINEL_PENDING_CPU_PREFLIGHT_FREEZE`,
which is the pre-freeze state, while the frozen artifact and the preflight
manifest both carry the literals. That is the contract working in both
directions.

## 2. Allowlist integrity — EXACT, zero dev/test contamination

The selection rule was re-derived from the frozen config
(`tag = C04-A0T-SMALL-v1`, `suffix = 20260729`, `n = 200`, payload
`utf8(tag || dataset || video_id || suffix)`, ascending sha256 with ascending
`video_id` tie-break) and applied to the train ASR IDs.

| | HateMM | MHC_zh |
|---|---|---|
| train ASR sha256 recomputed = pinned | yes | yes |
| train ASR size recomputed = pinned | 3900160 | 610605 |
| train rows = `expected_train_n` | 744 | 579 |
| train IDs unique | yes | yes |
| **selected ID sequence reproduces exactly** | **yes** | **yes** |
| ranks contiguous 0..199 | yes | yes |
| every `selection_sha256` recomputed | yes | yes |
| strictly ascending `(digest, id)` | yes | yes |
| merkle root recomputed | `5897b44ce04d4c75eaca34c2b86b68a39eea8b3d678dc211f11c3c9dd2dcf055` | `24d40b0ecc4ea6610eb570ca7ecb3543a032c3698672ce23ae2d5030495ae336` |
| `selection_contract` == config `selection` block | yes | yes |
| selected ∩ dev | **0** (dev n=107) | **0** (dev n=78) |
| selected ∩ test | **0** (test n=215) | **0** (test n=149) |
| selected ⊆ train | yes | yes |
| train ∩ dev, train ∩ test | 0, 0 | 0, 0 |

Exactly 200 + 200, matching the user amendment. Label-blindness of the
*selection* is proven rather than asserted: the digest payload is a pure
function of tag, dataset, video id and suffix, and reproducing the exact 200-ID
sequence from IDs alone shows no label could have entered the ranking. See
Important I-1 for what this does and does not imply about the sealed artifact.

## 3. Access ledger — train-only, zero test-like attempts

`freeze/access_ledger.json`, sha256
`c45cc12888c35b7da71c444a0d4b77a2325609f69bc85e41fd4cd683f2414ead`:

- 402 events, exactly two kinds: `OPEN_TRAIN_ASR_PROJECTED_FIELDS_ONLY` ×2 and
  `HASH_TRAIN_VIDEO` ×400 (201 per dataset);
- both ASR opens resolve to the pinned `train_asrK4_whisper-large-v3.jsonl`
  files; each `path_sha256` recomputes from its own `resolved_path`;
- `projected_field_counters` = `{label_field_syntactically_skipped: 1323,
  label_value_materialized: 0}`; 1323 = 744 + 579, i.e. every train row's label
  field was syntactically skipped and none was decoded;
- `events_merkle_root` `60f22f387b381a20087c7c6beacee120c4ec5c239f10fe7a4ce7b3187bda7569`
  recomputes; `event_count` equals `len(events)`;
- no recorded path contains a `dev`, `test` or `val` path component (the only
  such substrings anywhere in the namespace are the field names
  `regular_file_device`, `dev_or_test_locator_present`, `no_test_paths`,
  `label_value_materialized` and the self-test schema name);
- the 200 `HASH_TRAIN_VIDEO` events per dataset match the source-manifest rows
  1:1 **in order**, on `video_id_sha256`, `resolved_train_relative`,
  `regular_file_device` and `regular_file_inode`.

Byte counts are not carried by the ledger; they live in the source manifests and
were verified against disk in §4.

## 4. Source manifests vs disk — 400/400 re-hashed, zero drift

Every one of the 400 selected videos was re-resolved through its lexical symlink
and re-hashed: **3,430,759,978 bytes (3.20 GiB) read, 0 mismatches** on
`video_sha256`, `video_size`, `regular_file_device`, `regular_file_inode`,
`resolved_train_relative` and `video_path`, and every resolved path lies inside
the pinned physical train root. Transcript fields were re-derived from the ASR
projection (`id`/`window_text`/`language` only) through the frozen
`normalize_transcript` rule: all 400 `transcript_sha256`,
`transcript_scalar_count` and `language` values match.

Merkle roots recomputed:
`a8eab8ad30a208c7e12385b80133b70682d1f474b961d167f6946d190ca541c0` (HateMM),
`af2f8d7a061c4acc3ee409b46cf3af10a1051b6e27c8d60920c5dd86e5593087` (MHC_zh).
Source-manifest ID order equals allowlist ID order in both datasets. Both files
equal `canonical_json(obj) + b"\n"`. No row carries a label field.

All 15 payload files were also checked against the per-file `bytes`/`sha256`
table already recorded in `TARGET_STATE.json`: **15/15 exact**, no drift since
the job ended.

## 5. Maps — byte-exact reconstruction, declared geometry confirmed

Both dense payloads were regenerated from scratch by the frozen
`dense_rademacher_payload` rule and compared byte for byte:

| payload | byte length | expected | rebuild | value set |
|---|---|---|---|---|
| `le3_256x3598.f32le` | 3684352 | 256 × 3598 × 4 = 3684352 | **byte-exact** | exactly {−0.0625, +0.0625}, 460416 / 460672 |
| `additive_256x1024.f32le` | 1048576 | 256 × 1024 × 4 = 1048576 | **byte-exact** | exactly {−0.0625, +0.0625}, 130867 / 131277 |

All four role maps also rebuild byte-exactly from `materialize_role_map`; each
has 256 unique indices in `[0, 3584)`, signs ⊆ {−1, +1}, a self-consistent
`payload_sha256`, and the four index lists are pairwise distinct.

This closes, empirically, the concern the implementation record raised about its
own `maps.*` fields being "documentation-only, read by no module": the declared
`role_input_dim = 3584`, `role_output_dim = 256`, `le3_shape = [256, 3598]`,
`additive_shape = [256, 1024]` and `scale = 0.0625` were each checked against
the module constants (`TEACHER_DIM`, `ROLE_DIM`, `14 × 257`, `4 × 256`,
`1.0/16.0`) and against the materialized bytes. They agree on every count. They
are still unbound by assertion in code; the frozen artifacts now pin them in
fact.

## 6. Resource state — zero GPU consumed, ceiling correctly encoded, nothing unlocked

`resource/gpu_ledger.json` (file sha256
`0fb1b347c2f014340304ed83344cbb9c18e6f42c7467a0efe20266c18194541f`, internal
`payload_sha256` `fbdffa00…` self-consistent):

`state = GENESIS_UNCLAIMED`, `ledger_revision = 0`, `jobs = []`,
`aggregate_accounted_gpu_seconds = 0`,
`aggregate_reconciled_terminal_gpu_seconds = 0`, `cap_gpu_seconds = 7200`,
`single_allocation_only = true`, `resubmit_authorized = false`,
`requires_terminal_reconciliation = false`.

`resource/resource_ticket.json` (internal `payload_sha256` `dc4e71d8…`
self-consistent): `consumed = false`, `single_use = true`,
`no_submit_performed = true`, `authorized_slurm_allocation_count = 1`,
`completed_gpu_seconds = 0`, `remaining_seconds = 7200`,
`watchdog_seconds = 7080` (= 7200 − 120 reserve),
`issued_by_slurm_job_id = "13840"`, and `genesis_gpu_ledger_sha256` equals the
ledger file's own sha256 on disk.

7200 s is exactly the user amendment's **2 GPU-hour** first-tranche ceiling
("an aggregate maximum of 2 GPU-hours across both datasets and all C04 jobs").
The amendment's **8 GPU-hour** ceiling belongs to the conditional full-bank
tranche and is not encoded anywhere — see Important I-3.

Independent confirmation that nothing has been spent: `sacct` shows exactly two
C04 jobs in the accounting record, `13805` (v5, FAILED 1:0) and `13840` (v6,
COMPLETED 0:0, elapsed 00:00:19), and **neither has `gres/gpu` in its
AllocTRES** — both are `billing=8,cpu=8,mem=64G,node=1`. `resource_ticket_consumed.json`,
`allocation_claim.json`, `allocation_entry_marker.json`, `gpu_ledger.lock`,
`seal/` and `checkpoints/` are all absent. The user queue is empty.

`preflight_manifest.json` records `execution_authorized = false` and
`terminal_state = PREFLIGHT_HASH_FREEZE_PENDING_PAYLOAD_REVIEW`; its
`payload_sha256` is
`f1d2be7958ea06fddb09f3f3ad53a70ad9111147e3cda2ac49632138dfb0d308` and its file
sha256 is
`06bf6b38f424dd53d142367abd029dfa1f485380fb1482d72beabb7f5943ad1a`.

`staged_output_hashes` holds 14 entries and all 14 match disk exactly; none
escapes the namespace. The namespace holds 15 files. The single difference is
`preflight_manifest.json` itself, which cannot list its own hash — `preflight.py`
builds `staged_output_hashes` at line 471, hashes the manifest at 476 and stages
it at 477. Verified by construction, not a gap; the manifest's integrity is
carried by its own `payload_sha256` and is pinned at the next stage as
`preflight_manifest_sha256`.

## 7. Config contract and authority chain — recomputed, all pins exact

- All 15 `implementation_hashes` recomputed from disk: **15/15 exact**.
- Authorized config `configs/c04/c04_a0t_small_v1_v6.json` =
  `40ec6d97062498989ff9da21ebd6385aaee7fa3d2071d55b5664a1c5a135fc19`.
- Normalized config contract independently recomputed by the rule at
  `common.py:289-317` =
  `2b66775c44b727e35d52680d39eb838226d4f0a64fffd007d3e50ffcea79cdc5`, equal to
  the authority manifest pin **and** to the copy carried in all four of the
  frozen prompt-hash artifact, preflight manifest, GPU ledger and resource
  ticket.
- Authority manifest `5e56041adc5ef13527803f2c7950834cf59e38238a72dfb7a5c6a61b7e75b52f`;
  its `closure_sha256` `5375c39341933155286640563f5a3d588372acd2668a0cbbe3ba84639592639e`
  recomputes over the body.
- Implementation record `1b2d0bef47f32975c172278ac4d69b06b460d75b9431ab241ba5a5579dba5294`,
  unlock record `7a5623204e68b2e305286865af4bdb3fa38c3c6d84fd09f9d42ef0275700706e`,
  code/resource review `0e14bf43…`, unlock review `e6553ad1…`, job stdout
  `c7e3f85255234c4fc5fb6e949e371d2572a5a6a3466907af6a029c003639cdeb` (1399 B),
  job stderr `e3b0c442…` (0 B, the empty-string digest) — all match their
  recorded pins.

The contract-neutrality claim was tested directly rather than accepted: filling
the four sentinels in with the literal hashes leaves the contract hash at
`2b66775c…`, while tampering with `resources.small_cap_gpu_seconds` or
`teacher_contract.num_frames` moves it. The normalization is therefore as narrow
as the implementation record claims.

## 8. Self-test honesty — 25 checks confirmed real, 1 tautology found

The frozen fixture functions were executed from a byte-identical copy outside
the repository. `prompt_hash_contract_fixtures()` returns 13 entries;
`self_test_fixtures()` returns 20 (the 13 plus 7); `run_self_tests` in
`preflight.py` adds five more — `role_S_shape`, `role_P_shape`, `role_T_shape`,
`role_H_shape` and `no_test_paths` — giving exactly the 25 the job reported.
The fixture name set is a strict subset of the frozen check set, so no check
name in the manifest is unaccounted for.

Non-vacuity probes on four checks (more than the three requested):

1. `valid_form` / `malformed_form` genuinely discriminate: a well-formed
   response parses to `form_valid = True`, the string `"{"` to `False`. Two
   checks that cannot both hold by accident.
2. `transcript_cap` is a real arithmetic assertion over a real branch:
   `len(normalize_transcript("x"*3000)) = 2061 = 1024 + 13 + 1024`, while the
   short input takes the other branch and returns length 10. It would fail if
   head, tail or separator moved.
3. `prompt_hash_sentinel_bearing_payload_rejected` is real and load-bearing: a
   payload whose `payload_sha256` was recomputed to be internally *valid* but
   whose four keys hold the sentinel is still rejected, with
   `HALT_PROMPT_HASH_SENTINEL: frozen payload key A is still the sentinel`.
4. `_raises_runtime_error` returns `False` for a no-op and `False` for a
   `ValueError`, `True` only for a `RuntimeError`, so no negative fixture can
   pass by raising the wrong exception type or by not raising at all.

Additionally, the four `role_*_shape` checks constrain a payload I independently
rebuilt byte-for-byte, and `no_test_paths` scans 8 real config strings and would
fire on a `test_seen_asrK4…` path. One check is vacuous — see Important I-2.

## Findings

### Critical — none
### High — none

### Important

**I-1. The HateMM identifier is itself label-bearing, so the sealed "ID-only"
allowlist does not provide label containment.**
The user amendment says "The ID-only allowlist and its hash must be sealed
before any label value is readable." For MHC-ZH the identifiers are opaque
BiliBili codes (`BV1vs4y127aA`, …) and the guarantee holds literally. For
HateMM the identifiers are `hate_video_*` and `non_hate_video_*`: the sealed
allowlist and source manifest store plain IDs, so **any reader of the frozen
payload has the HateMM label of all 200 selected videos**. Note the asymmetry
inside the payload itself — the access ledger deliberately stores only
`video_id_sha256`, which buys nothing once the allowlist is in hand.
*What this does not break.* The label-blindness of the **selection** is proved,
not assumed: the exact 200-ID sequence reproduces from tag, dataset, id and
suffix alone. And the draw is demonstrably unengineered — the selected 200 split
78 hate / 122 non-hate (0.390) against a train prior of 298 / 446 (0.401), which
is what an unbiased hash draw looks like. The label also does not reach the
teacher: `producer.py:768` interpolates only `{transcript}` into the prompt
templates, and the templates contain no id field.
*Why it is filed anyway.* The seal must not be relied on downstream as a
label-containment property for HateMM, and "the teacher sees no label" is
currently guaranteed by one line of prompt assembly rather than by the artifact.
Closure: state the asymmetry explicitly in the GPU-execution authorization, and
make "no video id in any teacher-visible field" a checked precondition of the
producer rather than a property of its current code.

**I-2. One of the 25 self-test checks is a tautology, so the quoted coverage
overstates by one.**
`selection_deterministic` asserts
`selection_digest("HateMM","x") == selection_digest("HateMM","x")` — a pure
function compared to itself. It cannot fail under any mutation of the selection
rule, tag, suffix or digest payload. The number "25 checks, all_passed" is
quoted as evidence in the implementation record, the unlock record and
`TARGET_STATE.json`; 24 of those 25 carry information. This is non-blocking
because the property it nominally guards is independently proven here by the
exact reproduction in §2, but the check should either become a known-answer
vector (a pinned digest for a fixed `(dataset, id)`) or be documented as a smoke
call rather than a contract test.

**I-3. The amendment's 8 GPU-hour aggregate C04 ceiling is not encoded in any
machine-checked artifact.**
The config, the genesis ledger and the resource ticket all encode
`7200 s` — correctly, that is the first tranche's 2 GPU-hour cap. The
amendment's conditional full-bank tranche carries "an aggregate C04 ceiling of
**8 GPU-hours**, including every GPU-second consumed by the first tranche and
any later C04 extraction/adaptation job". Nothing in this payload accumulates
toward that figure or would refuse at it; today it is enforced by prose only.
That is acceptable for this stage, since the conditional tranche needs its own
code/resource review — but that review must add a cross-tranche accumulator, and
the first tranche's actual spend must be carried into it. Recorded here so it is
not discovered after 2 GPU-hours have already been burned.

### Observations considered and deliberately not filed

- `frozen_payload.total_bytes = 5178606` in `TARGET_STATE.json` is not the sum
  of the 15 file sizes (5174184); it is `du -sb artifacts/c04`, i.e. the parent
  tree's apparent size including directory inodes. Every per-file `bytes` and
  `sha256` in that table is exact against disk, so this is a field-labelling
  ambiguity with zero integrity impact. Recorded for the record rather than
  filed.
- The 14-vs-15 `staged_output_hashes` gap is by construction and already
  disclosed; verified at `preflight.py:471/476/477`.
- The two carried-forward pre-GPU blockers named in `TARGET_STATE.json`
  (`allocation_entry_marker` non-re-runnability; the post-freeze config
  amendment to literal prompt hashes) are pre-existing recorded boundaries, not
  new payload findings. They are restated below because they still gate the GPU
  stage.

## What this GO authorizes, and what remains blocked

**A GO here means exactly one thing: the payload frozen by job 13840 is
well-formed and faithful to the frozen contract, and the C04 candidate is
therefore *eligible* to be considered for a separately-authorized teacher small
tranche.** It authorizes no work.

Specifically **still blocked**:

- **Any GPU work whatsoever**, until explicit main-dialogue execution
  authorization. `teacher_authorized`, `gpu_authorized`, `slurm_authorized`,
  `small_tranche_execution_authorized` and `post_job_reconciliation_authorized`
  are all `false` in the frozen config and are not changed by this review.
- **Dev/test teacher — forever, under the amendment.** "No dev/test content or
  teacher call, API, external pool, cross-dataset input, cross-dataset fit or
  cross-dataset calibration is allowed." `dev_authorized` and `test_authorized`
  remain `false`; the frozen payload contains zero dev/test IDs.
- OCR, external API, network, cross-dataset data, label access before seal,
  chained submission, release, resubmission, and reuse of any artifact
  namespace.

**This markdown file is not the machine-checked authorization.** `verify_payload_review`
(`common.py:459-502`) requires all of: `review.payload_hash_verdict == "GO"` in
the config; a file at `refine-logs/C04_A0T_SMALL_V1_V6_PAYLOAD_HASH_REVIEW.json`
validating against `schemas/c04/c04_a0t_small_v1_v6_payload_review.schema.json`;
a 64-hex `review.payload_review_sha256` pin (today the sentinel
`PENDING_CPU_PREFLIGHT_AND_PAYLOAD_REVIEW`, which `_verified_review_file`
rejects with `HALT_REVIEW_LINEAGE`); a self-consistent `closure_sha256`; a
`reviewed_payload_sha256` over the body minus the two attestation fields; and
`attested_closure_sha256 = sha256("C04-PAYLOAD-REVIEW-GO-v6\n" +
reviewed_payload_sha256)`. None of that exists. Whoever builds it must pin the
values this review verified:

| field | value |
|---|---|
| `preflight_manifest_sha256` | `06bf6b38f424dd53d142367abd029dfa1f485380fb1482d72beabb7f5943ad1a` |
| `config_contract_sha256` | `2b66775c44b727e35d52680d39eb838226d4f0a64fffd007d3e50ffcea79cdc5` |
| `code_resource_authorization_sha256` | `5e56041adc5ef13527803f2c7950834cf59e38238a72dfb7a5c6a61b7e75b52f` |
| `prompt_hashes`, `map_hashes`, `staged_output_hashes` | must equal the preflight manifest's, all verified above |

Three items must additionally be closed or explicitly accepted before any GPU
stage:

1. the post-freeze config amendment replacing the four prompt-hash sentinels
   with the literal values (contract-neutral by design, confirmed in §7) must be
   performed and independently reviewed — the producer requires `LITERAL_BOUND`;
2. the GPU wrapper writes a job-id-and-uptime-pinned `allocation_entry_marker`
   before its first Python call, so a claim-time HALT leaves the no-clobber
   namespace non-re-runnable without manual marker removal;
3. Important I-3 above: the 8 GPU-hour aggregate ceiling needs a machine-checked
   accumulator before the conditional full-bank tranche.

No scientific content is published by this review. The CPU preflight produced no
metric, no result, no CONTINUE/KILL verdict, and consumed no scientific gate.
The unified pilot gate and the `+0.030 / +0.030` two-dataset target are
untouched, and the amendment's full-bank `+0.050 / +0.050` DIRECT-OOF and
STUDENT-OOF gates are unchanged and unwaived.
