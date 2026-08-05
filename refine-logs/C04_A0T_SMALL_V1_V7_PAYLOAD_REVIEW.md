# C04-A0T-SMALL-v1 impl-v7 — Independent PAYLOAD REVIEW

**Verdict:** `GO`
**Severity:** **0C / 0H / 8I**
**Scope:** the payload frozen by CPU-preflight SLURM job `13850`
(`c04_a0t_small_v1_v7_preflight`, COMPLETED 0:0, elapsed 00:00:19,
`billing=8,cpu=8,mem=64G,node=1` — **no GPU**) under
`artifacts/c04/a0t_small_v1_impl_v7/`.
**Reviewer:** fresh independent payload reviewer, no exposure to the reasoning
that produced the artifacts. Every number below is RECOMPUTED; nothing is
accepted on assertion.
**Method:** independent re-implementation of each frozen rule from the config
text and the written contract, plus a mutation/vacuity harness run against a
byte-identical scratchpad COPY of the frozen modules. No repository file was
created, modified or deleted except this one; a 99-file SHA-256 baseline taken
before the review re-verified clean at the end (0 mismatches). No SLURM job was
submitted, held, released, requeued or cancelled; `sacct`/`squeue` were read-only.
No GPU, teacher, model-weight or frame-decode work was run. No dataset label
value was materialized: only `id`, `window_text` and `language` were decoded from
any ASR JSONL.

**Bytes re-hashed by this review:** 3,430,759,978 (400 videos) +
4,510,765 (2 train ASR files) + 16,595,961,188 declared model/processor tree
(sizes verified on all 14 files; SHA-256 re-verified on 6 of 14 including one
1.09 GB shard, plus the tree-hash derivation over all 14 rows) +
the 14 staged payload files. **Mismatches: 0.**

---

## 0. The HateMM ID-label asymmetry — stated for the record

**HateMM identifiers ARE the label.** Every HateMM train identifier is
`hate_video_*` or `non_hate_video_*`. The sealed 200-ID allowlist for HateMM
therefore discloses the label of all 200 selected items by construction, and so
does `HateMM.source_manifest.json` (`video_path`, `resolved_train_relative`) and
the access ledger's 200 `HASH_TRAIN_VIDEO` events (`resolved_train_relative`).
I confirmed this directly: 200 of the 402 access-ledger events contain a raw
`hate_video`/`non_hate_video` string.

**Consequently the sealed ID-only allowlist provides label containment for
MHC-ZH only.** For HateMM the amendment's "ID-only allowlist sealed before any
label value is readable" clause buys nothing at the artifact level.

What *does* hold for HateMM, and what I verified rather than assumed:

1. **Selection label-blindness** — the rule is a pure function of
   `(tag, dataset, video_id, suffix)`. I re-implemented it from the config and
   reproduced both allowlists byte-for-byte. No label value enters the rule.
2. **Teacher label-blindness** — I replayed the producer's pre-model
   fail-closed precondition over the **real 400 transcripts × 2 prompt forms =
   800 renderings**: 0 failures, and 0 occurrences of any of the 402 banned
   tokens inside any real transcript under the NFKC/casefold variants the guard
   applies. The teacher will not see an identifier.

The implementation states this asymmetry itself
(`v7_scope.I1_teacher_visible_containment`, and
`containment_summary.hatemm_identifiers_are_label_bearing: true`). It is a
limitation of the amendment's containment design, not a defect in this payload.

---

## 1. Prompt-hash freeze

Recomputed the four hashes from the frozen prompt **source**, lifting
`SYSTEM_PROMPT`, `PROMPT_A`, `PROMPT_B` and the four enum tuples by static
`ast` evaluation (no module import, no execution of the frozen code):

| key | recomputed | frozen artifact | preflight manifest |
|---|---|---|---|
| `system` | `1ffc06755b18f9315dc930aaf02a0c79cf1f9e2d91d0ec5bae2fa0c1436ed048` | MATCH | MATCH |
| `A` | `cecb3555daa7a4cc0840db154086138dae9260795bde2b64e89f6724d80abb9b` | MATCH | MATCH |
| `B` | `9521bee7fe7a6964d4ff61605bbe5a02eb4cfd149fd5aa9cbf975accbfe40314` | MATCH | MATCH |
| `combined` | `a42268e4c0d1afb2b9576dd968d7f6250fc989fe3b97077a55363e63e1e2bb8a` | MATCH | MATCH |

- **Sentinel nowhere as a value.** None of the four keys in
  `freeze/prompt_hashes.json` holds `PENDING_CPU_PREFLIGHT_HASH_FREEZE`.
  The token appears exactly twice in the payload, both times as metadata:
  `pending_sentinel_token` (the token's *name*) and
  `config_binding_at_freeze: "SENTINEL_PENDING_CPU_PREFLIGHT_FREEZE"` (the name
  of the *pre-freeze config state*). `preflight_manifest.prompt_hash_freeze`
  records `literal_hashes_written: true`, `sentinel_written: false`.
- **Self-consistency.** `payload_sha256` recomputes to
  `c062f83ec4ced97ccb2df29699081123c1e7c30f00877e4e8fd7a6c06563fdb8` (MATCH);
  the file SHA-256 `9be8c55a509df2591227a3a953a3d5371142859d44152e3011800fd8517bcc2e`
  equals `preflight_manifest.prompt_hash_freeze.sha256` and the
  `staged_output_hashes` entry. `downstream_binding: "LITERAL_BOUND"` with the
  contract "a config still holding the pending sentinel must HALT", which
  `resolve_prompt_hashes(cfg, freeze_stage=False)` enforces on every later
  entrypoint and the GPU wrapper's `jq` gate enforces before `mkdir`.
- **The v6 Critical is genuinely dead.** `PROMPT_A.format(transcript="x")` still
  raises `KeyError: '"source_relation"'` (I ran it); `render_prompt` substitutes
  the single trailing placeholder, preserves the literal
  `{"source_relation":…}` and `{"S":0,"P":0,"T":0,"H":0}` braces, and lands the
  transcript at the tail. Rendered length across the real 400 items:
  min 872 / max 2932 / mean 1442 characters for form A. An adversarial
  transcript containing the literal `{transcript}` token substitutes exactly
  once and still satisfies the tail guard.
- The config's own `prompt_hashes` block still carries the sentinel; that is
  correct at this stage and contract-neutral, because `config_contract_sha256`
  normalizes the field (confirmed by independent recomputation, §7).

---

## 2. Allowlist integrity and split containment

Re-derived the 200+200 selection from the frozen rule
(`sha256(tag ‖ dataset ‖ video_id ‖ suffix)`, ascending, tie-break ascending
`video_id`), rebuilt both allowlist objects and their canonical bytes:

| | HateMM | MHC_zh |
|---|---|---|
| train rows read | 744 (= `expected_train_n`) | 579 (= `expected_train_n`) |
| duplicate ids | 0 | 0 |
| allowlist file **byte-identical** | **yes** (26,277 B) | **yes** (25,424 B) |
| allowlist SHA-256 | `02028929091d3a9e139a8f5587a3a128e550fc2d3832f260e2d742e5059d4925` | `bcfaaa657a2e7cc9d991f958707702ce717f129fec107ff1080f029c48ccff5d` |
| merkle root | `5897b44ce04d4c75eaca34c2b86b68a39eea8b3d678dc211f11c3c9dd2dcf055` | `24d40b0ecc4ea6610eb570ca7ecb3543a032c3698672ce23ae2d5030495ae336` |
| ranks 0..199 contiguous | yes | yes |
| every per-row `selection_sha256` reproduces | 200/200 | 200/200 |
| strictly ascending `(digest, video_id)` | yes | yes |
| exact ID sequence matches my derivation | yes | yes |
| unique IDs | 200 | 200 |
| rank-199 vs excluded rank-200 digest | `44f82322b499ad37…` < `4578136720576ecf…` | `587975847a808eeb…` < `5910c30f32a0a15c…` |

**Contamination: ZERO on both datasets.**

| | HateMM | MHC_zh |
|---|---|---|
| selected ∩ dev | **0** (dev n=107) | **0** (dev n=78) |
| selected ∩ test | **0** (test n=215) | **0** (test n=149) |
| selected ⊆ train | **True** | **True** |
| train ∩ dev / train ∩ test / dev ∩ test | 0 / 0 / 0 | 0 / 0 / 0 |

(dev/test ID sets read from `dev_seen_asrK4_*.jsonl` / `test_seen_asrK4_*.jsonl`
decoding `id` only.)

I also verified the `v7_scope` claim that v7 reproduces the v6 selection: both
allowlist files are **byte-identical** between the v6 and v7 namespaces on both
datasets, both source-manifest merkle roots are identical, and all six map
payloads are byte-identical. Only `prompt_hashes.json` differs (lineage fields).

---

## 3. Source manifests vs disk

Re-resolved all 400 selected videos through the lexical root
(`data/video/<ds>/All/<id>.mp4`, required to be a symlink) to the physical train
root, enforced regular-file `(st_dev, st_ino)` identity between lexical and
resolved, re-hashed every video, and re-derived every transcript hash, scalar
count and language through the frozen NFKC + CRLF/CR→LF + head/tail
normalization.

| | HateMM | MHC_zh |
|---|---|---|
| manifest **byte-identical** | **yes** (87,779 B) | **yes** (85,065 B) |
| SHA-256 | `b26eea0c8bc258a4058e6e7579af12cdad3a17b11b332c667ae63ff44443d669` | `0b7694503141f9226d6ba43624d654a0d9de3c4b07b0215bc9b4e25ed22c1ec4` |
| merkle root | `a8eab8ad30a208c7e12385b80133b70682d1f474b961d167f6946d190ca541c0` | `af2f8d7a061c4acc3ee409b46cf3af10a1051b6e27c8d60920c5dd86e5593087` |
| videos re-hashed / bytes | 200 / 1,367,215,645 | 200 / 2,063,544,333 |
| unique (device, inode) | 200/200 | 200/200 |
| language census | `{en: 200}` | `{zh: 200}` |
| transcript scalars min/max/mean | 3 / 2061 / 1026.3 | 1 / 916 / 115.9 |
| empty/whitespace-only transcripts | 0 | 0 |
| **row mismatches** | **0** | **0** |

Train ASR sources verified against the config: HateMM
`d47d4062…f24124`, 3,900,160 B; MHC_zh `1c3ce3d2…4b21a5d`, 610,605 B — both
sizes and hashes MATCH, neither is a symlink.

Note (not a defect): 45 HateMM transcripts land at exactly 2061 scalars
(= 1024 head + 13 separator + 1024 tail), i.e. 13 above the declared
`transcript_cap_unicode_scalars: 2048`. The cap is a truncation *trigger*, not
an output bound; the value is exactly what the declared head/tail/separator
rule produces. See I-2.

---

## 4. Access ledger

`freeze/access_ledger.json` — `event_count` **402**, and the events merkle root
`60f22f387b381a20087c7c6beacee120c4ec5c239f10fe7a4ce7b3187bda7569`
**recomputes exactly** from the 402 event objects.

- **Operation census:** `OPEN_TRAIN_ASR_PROJECTED_FIELDS_ONLY` ×2,
  `HASH_TRAIN_VIDEO` ×400. Nothing else. No teacher, frame, OCR, network or
  dev/test operation appears.
- **Label field:** `label_field_syntactically_skipped: 1323`. My **independent
  count** over both files is **1323** (= 744 + 579; every row carries exactly one
  `label` key). `label_value_materialized: 0`.
  The ASR top-level key census on both datasets is
  `{audio_ok, chunks, duration, id, label, language, timestamps, window_bounds,
  window_text}`; the projector decodes only `id`, `window_text`, `language` and
  advances an index over everything else via `_skip_json_value`, which I read
  line by line: it is a pure scanner (string/composite/scalar span advance) and
  never calls a decoder on the skipped span. See I-8 on the counter's form.
- **No dev/test path component anywhere:** I scanned every `resolved_path` and
  `resolved_train_relative` in all 402 events against the same forbidden-component
  set the code uses (`dev`, `development`, `test`, `tests`, `testing`,
  `validation`, `test_`/`dev_` prefixes, `_test`/`_dev` suffixes) — **0 hits**.
- **Cross-binding:** the 200 `video_id_sha256` values per dataset equal
  `sha256(video_id)` of the allowlist IDs **in allowlist rank order**, on both
  datasets. The ledger and the allowlist describe the same 400 items in the same
  order.
- `static_surface_assertions` are all `false` and are explicitly flagged
  `static_assertions_are_not_runtime_counters: true` — correctly not presented
  as measurements.

---

## 5. Maps — rebuilt and compared byte for byte

Re-implemented `HashStream` (SHA-256 CTR with big-endian u64 counter, rejection
sampling), the Fisher–Yates shuffle, the sign-bit extraction and the dense
Rademacher rule from the module source, driving them from the **config's**
declared tags, dims and scale.

| payload | rebuilt | on disk | **byte-identical** | SHA-256 |
|---|---|---|---|---|
| `role_S.json` | 2,062 B | 2,062 B | **yes** | `d60c9b7d46d03019e785c13e8c154485918c8c033106f6b398365d2802b259b3` |
| `role_P.json` | 2,056 B | 2,056 B | **yes** | `d1f99aee531cd3bccb4ccdad079948776da5e69050033bc263131ab5acc539e8` |
| `role_T.json` | 2,039 B | 2,039 B | **yes** | `1150f7e7fa5413e1b8bb742346dd05383ed4f02ce7cb37580054679747b98bbb` |
| `role_H.json` | 2,057 B | 2,057 B | **yes** | `b7bf0cca7a493156dc8753c9c3bfd2d5c091f7c967a6e04956da368411849219` |
| `le3_256x3598.f32le` | 3,684,352 B | 3,684,352 B | **yes** | `381b0b92f469034beccbce0b3b1ec8f318ba9efdb2dceeb7af88d92d8f01612c` |
| `additive_256x1024.f32le` | 1,048,576 B | 1,048,576 B | **yes** | `0c0ee6a78b5ceda885751bd4f0947614aa1e13005449008902da3837cae49100` |

**Materialized geometry vs declared vs module constants:**

- Each role map: `|indices| = 256`, `unique = 256`, all indices `< 3584`
  (observed maxima 3576/3571/3579/3580), `signs ⊆ {−1, +1}`, and each payload's
  own `payload_sha256` recomputes. Declared `role_input_dim: 3584` =
  `TEACHER_DIM`; `role_output_dim: 256` = `ROLE_DIM`.
- `le3_shape [256, 3598]` = `[ROLE_DIM, 14 × Q_DIM]` with `Q_DIM = 257`
  → `[256, 3598]`. Confirmed against the producer: `compose_features` builds
  exactly 14 subsets, each contributing a 257-long `q_product`, and asserts
  `len(le3_input) == LE3_INPUT_DIM`. File size 256×3598×4 = 3,684,352 B ✓.
- `additive_shape [256, 1024]` = `[ROLE_DIM, 4 × ROLE_DIM]` → `[256, 1024]`;
  file size 256×1024×4 = 1,048,576 B ✓.
- Both dense payloads decode to exactly two distinct float32 values,
  `{−0.0625, +0.0625}` = ±`scale`, with near-balanced signs
  (le3 460,416 / 460,672; additive 130,867 / 131,277).
- `fixed_projection` returns `256 + 1 = 257` values, matching the canonical
  schema's `vector257` for `LOWER_ORDER_LE3`/`ADDITIVE`; `apply_role` returns
  256, matching `vector256`. The pinned model's `config.json` declares
  `hidden_size: 3584`, equal to `TEACHER_DIM`, so `apply_role`'s dimension guard
  cannot fire after the forwards are paid for.

---

## 6. Resource state

**Zero GPU consumed by any C04 job — verified by my own `sacct` read.** The
accounting record (1,270 jobs since 2026-01-01) contains exactly **three** C04
rows, none with a `gres/gpu` allocation:

| JobIDRaw | JobName | ElapsedRaw | AllocTRES | State |
|---|---|---|---|---|
| 13805 | `c04_a0t_small_v1_v5_preflight` | 0 | `billing=8,cpu=8,mem=64G,node=1` | FAILED |
| 13840 | `c04_a0t_small_v1_v6_preflight` | 19 | `billing=8,cpu=8,mem=64G,node=1` | COMPLETED |
| 13850 | `c04_a0t_small_v1_v7_preflight` | 19 | `billing=8,cpu=8,mem=64G,node=1` | COMPLETED |

**Namespace GPU ledger** (`resource/gpu_ledger.json`, file SHA-256
`4d0e52ec3c3431696725c0ed86801cdc4c922959d71343fc04136835a042e659`):
`state: GENESIS_UNCLAIMED`, `jobs: []`, `ledger_revision: 0`,
`aggregate_accounted_gpu_seconds: 0`,
`aggregate_reconciled_terminal_gpu_seconds: 0`, `cap_gpu_seconds: 7200`,
`requires_terminal_reconciliation: false`, `resubmit_authorized: false`,
`single_allocation_only: true`. `payload_sha256` recomputes ✓.

**Resource ticket** (`resource/resource_ticket.json`, file SHA-256
`8a066ab904185d72c24a479a11098ca0dcf4ce50ed36f9e7a7cc2737ea77e348`):
`consumed: false`, `single_use: true`,
`authorized_slurm_allocation_count: 1`, `completed_gpu_seconds: 0`,
`remaining_seconds: 7200`, `no_submit_performed: true`,
`issued_by_slurm_job_id: "13850"`.
**Watchdog = cap − reserve: 6900 = 7200 − 300 ✓.**
`genesis_gpu_ledger_sha256` equals the ledger's actual file hash ✓.
`payload_sha256` recomputes ✓.

**Namespace inventory:** exactly 15 files — the 14 `staged_output_hashes`
entries plus `preflight_manifest.json` itself (which cannot contain its own
hash). **Absent, as required:** `checkpoints/`, `seal/`, `allocation_claim.json`,
`resource_ticket_consumed.json`, `allocation_entry_marker.json`,
`resource_final_state.json`, `budget_breach.json`, and both lock files. The
ticket is unconsumed and no GPU-stage state exists.

**Campaign accumulator** (`artifacts/c04/campaign/gpu_ledger.json`, file SHA-256
`fc6ca12c32427625d0b80c16b7802ef9a574ced0dbf0288edc3938d217267414`):
`payload_sha256` recomputes ✓, `jobs: []`, `aggregate_gpu_seconds: 0`,
`head_payload_sha256: "GENESIS"`, `ledger_revision: 0`,
`phase: FIRST_TRANCHE`, `phase_cap_gpu_seconds: 7200`,
`aggregate_cap_gpu_seconds: 28800`, `phase_advance_authorization: NOT_AUTHORIZED`.

**Is the encoded effective ceiling what the amendment binds today? — YES.**
I read `C04_USER_AMENDMENT_V2.md` myself. Its "Approved first tranche" section
binds the tranche to "an aggregate maximum of **2 GPU-hours** across both
datasets and all C04 jobs". The **8 GPU-hour** figure appears only under
"Conditional full-bank tranche", which the amendment gates on
`PASS_C04_SMALL_V2` *followed by* a fresh independent result-to-claim `GO`
*before* it may even request a code/resource review. That phase has not been
entered. `campaign_effective_cap` = `min(phase_cap, aggregate_cap)` =
`min(7200, 28800)` = **7200 s = 2 GPU-hours**, and
`assert_campaign_aggregate_headroom` — called both by the CPU preflight and by
`validate_gpu_environment` *before* `claim()` consumes the ticket — refuses on
`spent + requested > effective_cap`. **The encoded ceiling is right.** Caveats
in I-3, I-4, I-6.

**Accepted-design note (not a finding).** The GPU wrapper writes the job-id-bound
`allocation_entry_marker.json` before `claim()` verifies the lineage, so any halt
after that point forecloses the single authorized allocation and forces a v8
rebuild. This is the deliberate round-3 I-3 repair: the marker is the only
evidence a pre-claim HALT ever entered the allocation, and the amendment
authorizes exactly one allocation with resubmission forbidden. The round-4 H-1
repair already moved the full `jq` authorization gate and the frozen-manifest
existence test *ahead* of the `mkdir`, so what remains behind the marker is only
environment-shaped rejection (e.g. `CUDA_VISIBLE_DEVICES` shape). I record it
rather than file it.

---

## 7. Authority chain — recomputed independently

Every rule below was re-implemented from the config text, not imported.

- **`config_contract_sha256` = `ed9cc74d16dbcd6ab2cdda9e1a8243cce5c44328807be07ee100341700599707`**
  — reproduced by normalizing `authorization`, `prompt_hashes`, the four
  `REVIEW_PIN_FIELDS` and the four `REVIEW_STATUS_FIELDS` and canonical-hashing
  the rest. Matches the value carried by the config's pinned authorization
  manifest, the preflight manifest, the genesis GPU ledger and the resource
  ticket — all four agree.
- **Code/resource authorization manifest** — file SHA-256
  `09669bed4816e92b0b9df417ed6cd9bc288fc0f523d10358edf34a48821e6377`,
  equal to `config.review.code_resource_authorization_sha256` and to
  `preflight_manifest.code_resource_authorization_sha256`.
  **`closure_sha256` recomputes to `0be41f279b5728332d259980b02fe9e60431057f5c7330d2968d10b951e61176`
  (MATCH).** `stage: CPU_PREFLIGHT`, `verdict: GO`,
  `payload_binding: NO_PREFLIGHT_PAYLOAD_YET`. Its
  `implementation_hashes`, `frozen_design_hashes`, `source_hash_closure` and
  `model_hash_closure` are each identical to the values derived from the config.
- **All 15 implementation hashes verified against disk: 15/15 MATCH, 0 mismatches.**
  (4 analysis modules, 3 wrappers, 3 sbatch, 5 schemas.)
- **All 15 frozen design hashes verified against disk: 15/15 MATCH**, including
  `refine-logs/C04_USER_AMENDMENT_V2.md` =
  `c3180ecb8708d60bc41717961322083ed03ea4a6bf0bc5617d290981ccc7278f` and the
  design GO review `C04_V4_DESIGN_REVIEW.md` =
  `340ae2c156e7acab8a19dcda9625f883058377ca618bdc4fd59177900738a854`.
- **Preflight manifest** — `payload_sha256` recomputes to
  `d999a8ba6a18313edff6d3380f902f3122378df587ed2eaa87a9749c324ff98e` (MATCH);
  **file SHA-256 `b348c2fea15df61a3110f5305927898f7fccb5444e9fc8c69f0a61fc51ab801b`**.
  All **14** `staged_output_hashes` re-verified against disk: **0 mismatches**.
- **Model snapshot pins** — snapshot dir present at the pinned revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`; all 14 declared file sizes match
  on disk; both tree hashes reproduce from the declared `path\tsize\tsha256\n`
  rows (`model` `55705d03…e28c22`, `processor` `f77f6022…20710e`); SHA-256
  spot-checked and MATCHED on `config.json`, `generation_config.json`,
  `model-00005-of-00005.safetensors` (1.09 GB), `preprocessor_config.json`,
  `chat_template.json`, `tokenizer_config.json`.

---

## 8. Self-test honesty and vacuity probing

**The "57 checks" number is real.** I re-ran the suite against a byte-identical
scratchpad copy of the frozen modules: **57 checks, all True**, and the check
**name set is identical** to the frozen manifest's — 0 only-in-mine, 0
only-in-frozen. Accounting: `prompt_hash_contract_fixtures` 13 +
`teacher_visible_containment_fixtures` 9 + `downstream_contract_fixtures` 15 +
15 inline fixtures = 52 from `self_test_fixtures()`, plus 4 `role_*_shape` and
`no_test_paths` added by the preflight = **57**. No name collides (the suite is
collapsed into a `dict`, so a duplicate name would have silently reduced the
count; 52 + 5 = 57 proves none did).

I then ran a **25-mutation vacuity matrix**. Highlights of what the suite
*does* catch (it is far from a rubber stamp):

| mutation | caught by |
|---|---|
| selection suffix / tag / concat order altered | `selection_known_answer_vector` |
| `render_prompt` reverted to `str.format` (the v6 Critical) | suite raises `KeyError` |
| cosine clamp removed (round-1 C-B) | `cosine_of_identical_vectors_is_within_the_schema_bound` |
| containment ban list emptied for HateMM | `teacher_visible_ban_list_covers_both_datasets` |
| containment loop made a no-op | `teacher_visible_identifier_in_transcript_rejected` |
| provisional-usage writer/reader key drift (round-1 C-A) | writer `require_exact_keys` raises |
| `NUM_FRAMES` 8→4 | both full-record round-trip fixtures |
| `SYSTEM_PROMPT` byte changed | `prompt_bytes_unchanged_by_the_render_repair` |
| campaign aggregate cap 28800→36000 | `campaign_aggregate_cap_is_the_amendment_eight_gpu_hours` |
| `TRANSCRIPT_CAP` 2048→4096 | `transcript_cap` |

**Ten mutations were caught by nothing.** They are the basis of I-1, I-2, I-3
and I-7:

| unguarded mutation | consequence |
|---|---|
| role-map tag changed | all four role-map payloads change; 0 checks fail |
| role-map sign bit inverted | all four change; 0 checks fail |
| `HashStream` counter endianness `>Q`→`<Q` | all four change; 0 checks fail |
| Fisher–Yates loop direction reversed | all four change; 0 checks fail |
| dense-JL sign byte `digest[-1]`→`digest[0]` | both `.f32le` payloads change; 0 checks fail |
| `TRANSCRIPT_HEAD` 1024→512 | truncation geometry changes; 0 checks fail |
| `CAMPAIGN_PHASE_CAPS["FIRST_TRANCHE"]` 7200→28800 | today's binding ceiling quadruples; 0 checks fail |
| `campaign_effective_cap` → aggregate only | phase floor dropped; 0 checks fail |
| `RELIABLE_CONFIDENCE_MIN` 3→0 | reliability semantics change; 0 checks fail |
| `PROPOSITION_COSINE_MIN` 0.80→0.0 | agreement floor removed; 0 checks fail |
| `bounded_proposition` 32→4096 words | HateMM proposition bound removed; 0 checks fail |
| merkle odd-level duplicate-last-leaf rule changed | all merkle roots change; 0 checks fail |

`ROLE_DIM` 256→128 changes all four role-map payloads and is caught only
*incidentally*, by the canonical-record schema's `count: 256` const — not by any
`role_*_shape` check.

**Late-rejection ("resource consumed before the rejecting check") sweep.** I
looked specifically for the shape the campaign keeps tripping on, and found the
GPU path clean on every axis I could test from CPU:

- `claim()` runs `validate_gpu_environment` (auth flags, literal prompt-hash
  binding, campaign headroom, one visible GPU) and the **full lineage
  verification** (`verify_preflight_manifest` re-hashing all 14 staged files,
  code authorization, payload review, GPU authorization) **before** the ticket is
  consumed and before the ledger row is appended.
- The producer's teacher-visible precondition runs over all 400 identifiers ×
  2 forms **before** `from_pretrained` — and I proved on the **real** data that
  it passes 800/800, so it cannot abort a paid-for allocation.
- `sequence_index` max is `199×2+1 = 399`, exactly the schema's `maximum: 399` —
  it fits, with zero headroom.
- `raw_output` has no `maxLength`, so no teacher output can trip the prompt-record
  schema at item *N*; `transport_error` is `const ""` and the producer always
  writes `""`.
- `content` fields are never empty in any reachable state (valid ⇒ non-empty
  parsed content; invalid ⇒ `NO_CONTENT_<slot>`), so `minLength: 1` on
  `render_map` and `reliability.content` cannot fire late.
- `proposition_cosine` is clamped into `[−1, 1]`, matching the schema bound.
- The pinned model's `hidden_size` is 3584 = `TEACHER_DIM`, so `apply_role`'s
  guard cannot fire after 800 forwards.
- The role maps and both JL matrices are re-hashed at job start by
  `verify_preflight_manifest` (they are in `staged_output_hashes`), so the
  unhashed `np.memmap` read inside `canonicalize_dataset` is already covered.
- `guard.require_remaining(600, "the canonicalization and seal phase")` gates the
  post-loop phase, and the item-boundary `deadline_check` gates each item.

---

## Findings

### I-1 — The four `role_*_shape` checks are vacuous, and the two JL payloads have no self-test at all

`role_X_shape` asserts `len(indices) == ROLE_DIM`, `len(signs) == ROLE_DIM`,
`len(set(indices)) == ROLE_DIM`, `set(signs) ⊆ {−1, 1}`. Every one of those is
**structurally guaranteed** by `materialize_role_map`: it shuffles
`range(TEACHER_DIM)` and truncates to `ROLE_DIM`, so the indices are always
`ROLE_DIM` distinct values, and the sign derivation always yields ±1. The check
also compares against `ROLE_DIM` itself, so it is self-referential. Four
independent mutations (role tag, sign-bit polarity, `HashStream` counter
endianness, Fisher–Yates direction) each change **all four** role-map payload
hashes and fail **zero** checks. The dense JL payloads are worse: no check
references them at all, and flipping the Rademacher sign rule changes both
`.f32le` files invisibly. This is the same shape as the v6
`selection_deterministic` fixture that the v6 review filed as I-2 — repaired for
selection with a pinned known-answer vector, not repaired for the maps.
**Impact is contained:** the six map payloads are now frozen, their SHA-256s are
in `map_hashes` and `staged_output_hashes`, `verify_preflight_manifest` re-hashes
them at GPU job start, `verify_gpu_execution_authorization` requires the GPU
manifest's `payload_binding.map_hashes` to equal the preflight's, and I have
independently reproduced all six byte-for-byte (§5). **Closure:** add one pinned
known-answer digest per role map and per JL payload, exactly as
`SELECTION_KNOWN_ANSWER_DIGESTS` does for selection.

### I-2 — `transcript_cap` is self-referential on head/tail, and the declared 2048 cap is a trigger, not a bound

The check is
`len(normalize_transcript("x"*3000)) == TRANSCRIPT_HEAD + len(SEP) + TRANSCRIPT_TAIL`
— it compares the function's output against the same constants the function
used, so halving `TRANSCRIPT_HEAD` to 512 fails nothing. (It is not fully
vacuous: raising `TRANSCRIPT_CAP` to 4096 *is* caught, because the 3000-scalar
probe then stops being truncated.) Separately, `transcript_cap_unicode_scalars:
2048` reads as an output bound but the truncation rule emits
1024 + 13 + 1024 = **2061** scalars; **45 of the 400** items sit at exactly 2061.
No downstream consumer is harmed and the value follows exactly from the declared
head/tail/separator, but a reader who takes the field name literally will be
wrong by 13 scalars. **Closure:** pin the truncated length as a literal
(`== 2061`) rather than as an expression over the constants, and rename or
document the field as a truncation trigger.

### I-3 — Neither campaign check can fail if today's binding ceiling is lifted

`campaign_aggregate_cap_is_the_amendment_eight_gpu_hours` asserts
`CAMPAIGN_AGGREGATE_CAP_GPU_SECONDS == 8*3600`;
`campaign_cap_strictly_exceeds_the_small_tranche_cap` asserts `28800 > 7200`.
Both pin the **conditional** 8-hour ceiling. Neither touches the mechanism that
actually holds the tranche at 2 hours today: setting
`CAMPAIGN_PHASE_CAPS["FIRST_TRANCHE"] = 28800`, or reducing
`campaign_effective_cap` to the aggregate alone, each passes all 57 checks. The
two check *names* read as if the campaign ceiling is under test when the binding
half of it is not. The real guards exist outside the suite — `load_campaign_gpu_ledger`
cross-checks the on-disk `phase_cap_gpu_seconds` against the phase constant, and
`common.py`'s SHA-256 is in `implementation_hashes` — and I verified the encoded
effective ceiling is 7200 s (§6). **Closure:** add
`campaign_effective_cap_is_the_two_gpu_hour_first_tranche` asserting
`campaign_effective_cap({"phase_cap_gpu_seconds": 7200,
"aggregate_cap_gpu_seconds": 28800}) == 7200`, plus a fixture that a
`CONDITIONAL_FULL_BANK` ledger is refused while the phase is unadvanced.

### I-4 — The campaign accumulator's own prose describes a looser rule than the one enforced

`read_rule` states that every stage "checks `aggregate_gpu_seconds` + its own
reservation against `aggregate_cap_gpu_seconds`" — i.e. against 28800. The code
checks against `campaign_effective_cap` = 7200. `record_campaign_gpu_spend` also
prints progress as `aggregate/28800`. The code is *tighter* than its own
description, so the direction is fail-safe, but this is the one artifact
deliberately designed to outlive every implementation namespace and to be read by
future stages as the authority on the ceiling; it currently documents a ceiling
four times the binding one. **Closure:** rewrite `read_rule` to name
`min(phase_cap_gpu_seconds, aggregate_cap_gpu_seconds)` and print progress
against the effective cap.

### I-5 — `genesis_evidence.these_are_the_only_c04_jobs_in_the_accounting_record` is now stale

The campaign ledger was written at 21:45 and records jobs 13805 and 13840 with
the absolute flag `these_are_the_only_c04_jobs_in_the_accounting_record: true`.
Job 13850 — the very preflight that froze this payload — is now a third C04 row
in `sacct`. The substantive claim ("no C04 job has consumed a GPU-second") is
still true and I re-verified it independently, but a tamper-evident artifact now
carries a falsifiable absolute that is false. **Closure:** restate the field as
`no_c04_job_has_consumed_a_gpu_second` with the read timestamp, or have the
campaign-record stage refresh the census.

### I-6 — Advancing the campaign phase from 2 h to 8 h is gated by a free-form string, not by a hash

`phase_advance_authorization: "NOT_AUTHORIZED"` is checked only when
`phase == "FIRST_TRANCHE"`; once `phase` is set to `CONDITIONAL_FULL_BANK` the
field is not examined at all, and `load_campaign_gpu_ledger` accepts the ledger
as long as `phase_cap_gpu_seconds` matches the constant for the new phase. So
quadrupling the binding ceiling requires editing two fields and recomputing
`payload_sha256` — the chain of per-row digests is untouched because no row
changes. The amendment's actual conditions (`PASS_C04_SMALL_V2`, a fresh
independent result-to-claim `GO`, and a new code/resource review) are bound to
nothing machine-checkable. **Closure:** require `phase_advance_authorization` to
be a 64-hex SHA-256 of the future full-bank code/resource authorization manifest,
and have `load_campaign_gpu_ledger` verify that file exists, hashes to that pin
and carries `verdict: GO` before it honours any phase above `FIRST_TRANCHE`.

### I-7 — Config-declared constants are never reconciled against the module constants that actually execute

`verify_static_config` checks run identity, dataset order, `count_per_dataset`,
`selection.sort`, the resource block, the design verdict, the authorization
flags, the two hash maps and the ASR size/hash. It never compares
`selection.tag`, `selection.suffix`, `selection.digest_payload`,
`selection.tie_break`, **any** field of `teacher_contract` (`num_frames`,
transcript cap/head/tail/separator, `max_new_tokens`, `do_sample`, `temperature`,
`num_beams`, `retry_count`, `ocr`, `title_input`), **any** field of
`reliability`, or `maps` geometry/tags/scale against the module constants that
are actually used. The reliability thresholds appear a *third* time as bare
literals inside `canonicalize_dataset` (`0.85`, `0.10`, `0.20`, `0.90`, `0.60`).
A config amendment to any of these would silently not take effect while every
manifest re-pinned around it. **I verified by recomputation that all of them
currently agree** — in particular I rebuilt both allowlists using the *config's*
`tag` and `suffix` and got byte-identical files, and I rebuilt both JL payloads
using the *config's* tags, shapes and scale and got byte-identical files — so
the declared rule and the executed rule are the same rule today. **Closure:** add
`assert_equal` lines binding each config field to its module constant, and read
the five reliability thresholds from the config instead of re-typing them.

### I-8 — `label_value_materialized: 0` is a hard-coded literal, not a measurement

`project_train_asr_line` returns `{"label_field_syntactically_skipped": n,
"label_value_materialized": 0}` — the `0` is written into the return dict
unconditionally and can never be anything else. Both the access ledger and the
preflight manifest present it beside a real counter (1323), which invites reading
it as an observation. The underlying property is nonetheless **true**, and I
verified it two ways rather than trusting the literal: (a) I read `_skip_json_value`
in full and confirmed it is a pure index-advancing scanner over string,
composite and scalar spans with no decode call on the skipped region; (b) my
independent projector reproduced all 400 transcript hashes and scalar counts
byte-for-byte while decoding only `id`/`window_text`/`language`, and the
independent `label`-key count (1323) equals the recorded one. **Closure:** rename
the field to something declarative (`label_value_decode_call_sites: 0`), or make
it an actual measurement by incrementing a counter at the single decode call site
and asserting the `label` key never reaches it.

---

## 9. Values a downstream machine-checked manifest must pin

`refine-logs/C04_A0T_SMALL_V1_V7_PAYLOAD_HASH_REVIEW.json`, validated against
`schemas/c04/c04_a0t_small_v1_v7_payload_review.schema.json`, must carry:

```
schema_version                      c04_payload_review_v7
run_id                              C04-A0T-SMALL-v1
implementation_version              v7_prospective
verdict                             GO
preflight_manifest_sha256           b348c2fea15df61a3110f5305927898f7fccb5444e9fc8c69f0a61fc51ab801b
config_contract_sha256              ed9cc74d16dbcd6ab2cdda9e1a8243cce5c44328807be07ee100341700599707
code_resource_authorization_sha256  09669bed4816e92b0b9df417ed6cd9bc288fc0f523d10358edf34a48821e6377
```

`prompt_hashes`, `map_hashes` and `staged_output_hashes` **must be copied
verbatim from the preflight manifest** — `verify_payload_review` compares all
three for exact equality. I confirm each of the three matches the preflight
manifest and reproduces from source/disk:

```
prompt_hashes.system    1ffc06755b18f9315dc930aaf02a0c79cf1f9e2d91d0ec5bae2fa0c1436ed048
prompt_hashes.A         cecb3555daa7a4cc0840db154086138dae9260795bde2b64e89f6724d80abb9b
prompt_hashes.B         9521bee7fe7a6964d4ff61605bbe5a02eb4cfd149fd5aa9cbf975accbfe40314
prompt_hashes.combined  a42268e4c0d1afb2b9576dd968d7f6250fc989fe3b97077a55363e63e1e2bb8a

map_hashes.roles.S      d60c9b7d46d03019e785c13e8c154485918c8c033106f6b398365d2802b259b3
map_hashes.roles.P      d1f99aee531cd3bccb4ccdad079948776da5e69050033bc263131ab5acc539e8
map_hashes.roles.T      1150f7e7fa5413e1b8bb742346dd05383ed4f02ce7cb37580054679747b98bbb
map_hashes.roles.H      b7bf0cca7a493156dc8753c9c3bfd2d5c091f7c967a6e04956da368411849219
map_hashes.le3_f32le_sha256       381b0b92f469034beccbce0b3b1ec8f318ba9efdb2dceeb7af88d92d8f01612c
map_hashes.additive_f32le_sha256  0c0ee6a78b5ceda885751bd4f0947614aa1e13005449008902da3837cae49100

staged_output_hashes (14 entries, all re-verified against disk)
  .../freeze/HateMM.allowlist.json          02028929091d3a9e139a8f5587a3a128e550fc2d3832f260e2d742e5059d4925
  .../freeze/HateMM.source_manifest.json    b26eea0c8bc258a4058e6e7579af12cdad3a17b11b332c667ae63ff44443d669
  .../freeze/MHC_zh.allowlist.json          bcfaaa657a2e7cc9d991f958707702ce717f129fec107ff1080f029c48ccff5d
  .../freeze/MHC_zh.source_manifest.json    0b7694503141f9226d6ba43624d654a0d9de3c4b07b0215bc9b4e25ed22c1ec4
  .../freeze/access_ledger.json             242549d96a5ce6e30dffc6c8542c0891fcc6b346fe33c479cbe5c64d49c29348
  .../freeze/maps/additive_256x1024.f32le   0c0ee6a78b5ceda885751bd4f0947614aa1e13005449008902da3837cae49100
  .../freeze/maps/le3_256x3598.f32le        381b0b92f469034beccbce0b3b1ec8f318ba9efdb2dceeb7af88d92d8f01612c
  .../freeze/maps/role_H.json               b7bf0cca7a493156dc8753c9c3bfd2d5c091f7c967a6e04956da368411849219
  .../freeze/maps/role_P.json               d1f99aee531cd3bccb4ccdad079948776da5e69050033bc263131ab5acc539e8
  .../freeze/maps/role_S.json               d60c9b7d46d03019e785c13e8c154485918c8c033106f6b398365d2802b259b3
  .../freeze/maps/role_T.json               1150f7e7fa5413e1b8bb742346dd05383ed4f02ce7cb37580054679747b98bbb
  .../freeze/prompt_hashes.json             9be8c55a509df2591227a3a953a3d5371142859d44152e3011800fd8517bcc2e
  .../resource/gpu_ledger.json              4d0e52ec3c3431696725c0ed86801cdc4c922959d71343fc04136835a042e659
  .../resource/resource_ticket.json         8a066ab904185d72c24a479a11098ca0dcf4ce50ed36f9e7a7cc2737ea77e348
```

`reviewed_payload_sha256` = `sha256_obj` of the manifest minus
`attested_closure_sha256`, `reviewed_payload_sha256` and `closure_sha256`;
`attested_closure_sha256` = `sha256("C04-PAYLOAD-REVIEW-GO-v7\n" ‖
reviewed_payload_sha256)`; `closure_sha256` = `sha256_obj` of everything but
itself. The config must then set `review.payload_hash_verdict: "GO"` and
`review.payload_review_sha256` to the manifest file's SHA-256.

---

## 10. Verdict

**`GO` — 0 Critical, 0 High, 8 Informational.**

Every frozen artifact reproduces byte-for-byte from the frozen rule: both
allowlists, both source manifests (400 videos, 3.43 GB re-hashed, 0 row
mismatches), all four role maps, both dense JL payloads, the four prompt hashes,
the access-ledger merkle root, and the whole authority chain including the
`closure_sha256`. Dev/test contamination is zero on both datasets and the
selection is a strict subset of train. No GPU-second has been consumed by any
C04 job, the single-use ticket is unconsumed with `watchdog = cap − reserve =
6900`, the campaign accumulator's effective ceiling is the 7200 s the amendment
binds today, and the namespace contains no claim, no consumption record, no
checkpoint and no seal. The v6 Critical is dead and its repair is fixture-pinned;
the pre-model containment precondition — the one guard that could abort a
paid-for allocation — passes 800/800 on the real data. The eight findings are all
"a guard that cannot fail" or "prose that names the wrong number"; every property
they fail to guard I have verified by direct recomputation, so none of them
leaves an unverified assumption in the payload. The next stage — a single
authorized GPU teacher tranche over exactly these 400 IDs — is safe to
authorize on this payload.
