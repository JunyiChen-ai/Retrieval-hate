# C04-A0T-SMALL-v1 impl-v8 — Independent Payload Review (round 1)

Date: 2026-08-01
Reviewer: fresh independent Opus payload reviewer; zero exposure to the build
reasoning and zero exposure to the code/resource review.
Subject: the materialised namespace `artifacts/c04/a0t_small_v1_impl_v8/`
produced by CPU SLURM job `13855` (COMPLETED, 0:0, 976 s, no GPU in AllocTRES).

## VERDICT: `GO (0C / 0H / 5I)`

No defect was found that the GPU stage would re-read and reject. Every hash,
every derivation, and every measured number in the frozen payload reproduced
independently, including several recomputed from scratch rather than checked
against the project's own code.

## What was RECOMPUTED, and what each gave

### 1. Hashes (all 414 + self-digests)
| quantity | result |
|---|---|
| `freeze/preflight_manifest.json` file SHA-256 | `bd93adf8...d50c1b` — equals the request anchor |
| manifest `payload_sha256` recomputed over body | `07ab0ef3...f187f6` — matches claim |
| `staged_output_hashes` re-hashed against disk | **414/414 match, 0 missing, 0 mismatched** (400 frame-pack manifests + 8 freeze artifacts + 4 map files + 2 resource files) |
| `resource/resource_ticket.json` `payload_sha256` | `023f6f47...0b489c` — matches |
| ticket `genesis_gpu_ledger_sha256` = `7b7b71ce...41e4a` | equals sha256 of the namespace `resource/gpu_ledger.json` bytes |
| ns `gpu_ledger.json` `payload_sha256` | matches; state `GENESIS_UNCLAIMED`, `jobs: []`, `ledger_revision: 0` |
| campaign ledger | payload digest OK, single row (job 13852, 1978 s), `previous_payload_sha256 = GENESIS`, row digest OK, head OK, aggregate == rows |
| campaign headroom | effective cap `min(7200, 28800) = 7200`; `1978 + 5222 = 7200 <= 7200` passes; `+5223` refused |

### 2. The 200+200 selection — reviewer's own implementation, no project imports
| | HateMM | MHC-ZH |
|---|---|---|
| ASR file SHA-256 | `d47d4062...f24124` = config pin | `1c3ce3d2...4b21a5d` = config pin |
| pool | 744 unique | 579 unique |
| **exact match to frozen allowlist incl. rank order** | **True, 0 diffs** | **True, 0 diffs** |
| rank 0 | `hate_video_334` / `003d3d31...` | `BV1vs4y127aA` / `001d1c3c...` |
| rank 199 | `hate_video_56` / `44f82322...` | `BV1Tp4y1k7HG` / `587975847...` |
| merkle_root | `5897b44c...` match | `24d40b0e...` match |
| source-manifest merkle | `a8eab8ad...` match | `af2f8d7a...` match |
| cross-dataset ID overlap | empty | |

`projected_field_counters`: `label_field_syntactically_skipped = 1323` — exactly
`744 + 579`, one per ASR line; `label_value_materialized = 0`. Zero fields named
`label*` anywhere in the 413 namespace JSONs.

### 3. Prompt hashes — derived from the module source via `ast`, not by calling their function
All four (`system` `1ffc0675...`, `A` `cecb3555...`, `B` `9521bee7...`,
`combined` `a42268e4...`) match the v6 freeze, the manifest and
`freeze/prompt_hashes.json`. Freeze artifact `payload_sha256` `9e47a9dc...e719d5`
recomputed and matches; file SHA-256 `671fddab...5ff651` matches both the
manifest attestation and its staged entry. `guarded_access_audit.events_merkle_root`
`1df335bc...b0dffe` recomputed over all 802 events — match.

### 4. Frame packs — all 400 manifests and all 3200 PNGs
- **3200/3200 PNG SHA-256 and byte sizes match** their manifest rows; no symlinks.
- 400/400 manifest `payload_sha256` recomputed; exact key set; all 13 binding
  fields re-derived independently — 0 mismatches.
- `requested_indices` re-derived as `min(N-1, floor((i+0.5)*N/8))` with the
  `N==0 -> []` branch: 400/400 match.
- Directory contents exactly `{manifest.json, 00..07.png}` for all 400; pack-root
  entry sets equal the allowlists. Every manifest's file hash equals its
  `staged_output_hashes` entry.
- Backends: decord 388, pyav 11, none 1. The 11 pyav items are all MHC_zh
  (`BV18N4y1B7qA`, `BV16M4y1c7eo`, `BV1oS4y1c7v3`, `BV1iD4y1B7Sw`, `BV1Yh411B7XD`,
  `BV1tq4y1G74f`, `BV1UK4y1j7N9`, `BV1CN4y1h79K`, `BV13j411R79M`, `BV1fU4y1X7Gu`,
  `BV1AT411y7ER`); all eight frames verified for each.
- `HateMM/hate_video_95`: `total_frame_indices: 0`, backend `none`,
  `frame_decode_failed: [true]x8`, `requested_indices: []`; all eight PNGs are
  336x336 with per-channel extrema `(0,0)` — pure black, byte-verified.
- **v7 vs v8 HateMM frame bytes: 0 differences over 1600 PNGs.**

### 5. Visual geometry — independently re-measured for ALL 400 items
Ran the frozen PNGs through
`AutoProcessor.from_pretrained(snapshot, local_files_only=True, max_pixels=151200).image_processor`
on CPU (transformers 4.49.0, torch 2.6.0).

- **400/400 exact match** on `video_grid_thw`, `patch_tokens`, `frame_size`,
  `merged_tokens`, `vision_sdpa_fp32_bytes`.
- HateMM min 1456 / median 2880 / max 3072 / mean 2848.16 (manifest 2848.2)
- MHC_zh min 480 / median 2880 / max 3072 / mean 2865.44 (manifest 2865.4)
- Global max 3072 -> `3072^2*16*4 = 603,979,776 B = 0.562 GiB`.

**The check that mattered most.** The preflight measures through
`processor.image_processor(...)`; the producer measures through the full
`processor(text=[chat], ...)` and hard-halts on
`tokens != expected_visual_tokens`. On transformers >= 4.5x these can diverge (a
separate `video_processor`). Both paths were run on **38 items** (incl.
`hate_video_95`, all 11 pyav items, both datasets' extremes, 20 random):
`Qwen2_5_VLProcessor` in 4.49.0 has **no `video_processor` attribute** and routes
videos to `self.image_processor` with the same `max_pixels`. **A-vs-B
disagreements: 0. Path-B vs frozen mismatches: 0.** The per-forward equality
assert is satisfiable.

### 6. Projection arithmetic — recomputed from inputs
```
window     = 5222 - 300 - 300 - 600           = 4022    (claim 4022)
forwards   = 800 x 4.4248                     = 3539.8  (claim 3539.8)
total      = 208.1 + 69.7 + 3539.8            = 3817.6  (claim 3817.6)
margin     = 4022 - 3817.6                    = 204.4   (claim 204.4)
fraction   = 204.4 / 4022                     = 0.0508  (claim 0.0508)
affordable = (4022 - 208.1 - 69.7) / 800      = 4.6803  (claim 4.6803)
capped     = 208.1 + 69.7 + 800 x 2.9         = 2597.8  (claim 2597.8)
had the 834.7 s of CPU work stayed in-job: 4652.3 > 4022 -> the gate would have FAILED
```

**The basis was also refit from raw v7 data.** Native geometry re-measured for
v7's 200 HateMM packs (no `max_pixels`) and its 400 recorded `elapsed_seconds`
regressed:
- v7 native tokens: min 1456 / median 8160 / max 9248 / mean 6823.1 — matches the
  `basis_note` exactly.
- Least squares: `elapsed = 1.9913 + 3.567e-04 x native_tokens` — identical to the
  claimed fit.
- v7's 46 forwards on the 23 items where native geometry already equalled capped
  geometry: mean **2.9002 s** — exactly the claimed 2.9.
- Regression evaluated at v8's capped geometry: **3.0072 s/forward -> 800 forwards
  = 2405.7 s**, total ~2683.5 s, margin ~**1338.5 s (33.3 %)**.
- Residual sd 0.834 s, p95 +1.06 s, max +5.25 s.
- Sum of v7 forwards = 1769.9 s; job 13852 elapsed 1978 s => residual 208.1 s. v7
  has exactly 200 HateMM + 1 MHC_zh frozen packs, so that residual already
  contains both datasets' full `load_selected_inputs` and v7's 201 in-allocation
  pack builds (~110 s) that v8 does not repeat.

### 7. Label containment and boundary hygiene
- **800/800 prompt renderings** re-assembled independently and scanned against
  the full ban set — 402 tokens expanded to 602 NFKC/casefold variants, against
  NFKC and casefolded haystacks. **0 violations.**
- All 400 transcripts re-derived from the raw ASR: 400/400 `transcript_sha256`
  match, 0 language mismatches, 0 scalar-count mismatches.
- All 400 videos: `resolve()` lands in the pinned physical root, symlink policy
  holds, live `st_dev`/`st_ino`/`st_size` match the frozen manifest for all 400.
  **All 400 `video_sha256` recomputed over 3.43 GB — 0 mismatches.**
- `_forbidden_path_component` re-implemented and applied to every path the GPU
  stage will `root_path()`: 414 staged + 30 bound-file-map + 400 lexical/resolved
  video paths — 0 forbidden components.
- Namespace: 3615 files = 3200 `.png` + 413 `.json` + 2 `.f32le`. No dev/test
  path, no cross-dataset path, no OCR artifact, no network/API artifact.

### 8. Maps, contract, bound files, model snapshot
- The four role maps re-implemented from scratch (HashStream + Fisher-Yates +
  sign-bit extraction) -> objects byte-identical to the frozen files.
- `le3_256x3598.f32le` (3,684,352 B) and `additive_256x1024.f32le` (1,048,576 B)
  re-derived byte-for-byte identical.
- `config_contract_sha256` re-implemented -> `e4d041df...6fe294`, equal to the
  manifest, the ticket and the ns ledger. Verified invariant under the exact
  post-review config amendment the pipeline requires. Control: amending
  `maps.expected_hashes` does move it, so the normalization is not vacuous.
- `implementation_hashes` (15) and `frozen_design_hashes` (15): 30/30 live files
  match their pins.
- Code/resource authorization `c0a12d1a...150344`: file hash = config pin =
  manifest pin; `closure_sha256` recomputed OK.
- Model snapshot: all 14 files re-hashed; `model_tree_sha256 = 55705d03...e28c22`
  and `processor_tree_sha256 = f77f6022...20710e` recomputed — both match.
- `self_test`: 74 checks, all true.

## GPU-stage satisfiability trace

Order of irreversibility in the GPU wrapper: jq authorization gate ->
preflight-manifest existence -> entry marker -> `claim` -> **ticket consumption**
-> `flock` -> producer.

`claim()` runs `validate_gpu_environment` (env + `assert_campaign_aggregate_headroom(5222)`)
and then **`verify_gpu_lineage` — which re-hashes all 414 staged outputs,
`verify_historical_code_resource_authorization` (including both
`verify_bound_file_map` sweeps), `verify_payload_review` and
`verify_gpu_execution_authorization` — BEFORE the ticket is consumed.** The
ordering is correct for this failure family.

Checks that fire only after consumption, and their status:

| post-consumption check | status |
|---|---|
| `verify_authorization` (env, constants, headroom, 30 bound files, literal prompt-hash path) | satisfied |
| `verify_model_snapshot` (14 files, two tree hashes) | satisfied |
| `verify_claimed_resource` (ticket digest, genesis pin, `consumed==False`, `4922 == 5222-300`, `5222 <= 5222`) | satisfied |
| `load_selected_inputs` x2 (file hashes, rank order, per-row digest, sorted replay, ASR hash, 400 transcript hashes, 400 st_dev/st_ino/video_sha256, lexical path) | all recomputed, satisfied |
| `assert_teacher_visible_precondition` (800 renderings vs 402-token ban list) | 0 violations |
| `strict_validate_frame_pack` x400 (manifest digest, key set, 13 binding fields, backend enum, index rule, decode vector, 8 rows, dir entries, 3200 hashes+sizes, inode identity, staged pin) | 400/400 satisfied |
| `visual_patch_tokens(...) != expected_visual_tokens` x800 | 400/400 satisfied via both processor paths; `visual_geometry.items` covers exactly the 400 selected IDs (no KeyError) |
| `assert_visual_token_ceiling` (<= 4096) | max 3072 |

Pre-state is clean for `claim()`: `resource/` holds only `gpu_ledger.json` and
`resource_ticket.json`; `seal/` absent; neither checkpoint directory exists, so
`load_checkpoint` returns `{}` and `idempotent_complete` short-circuits False.
`verify_reconciliation_lineage` also passes with `allow_claimed_gpu_ledger=True`,
so the terminal CPU stage is satisfiable.

**Residual, unverifiable-now dependency:** `payload_review` and
`gpu_execution_authorization` do not yet exist. `verify_payload_review` requires
`body["staged_output_hashes"] == preflight["staged_output_hashes"]` (all 414
verbatim), `preflight_manifest_sha256 == bd93adf8...d50c1b`, plus the two-layer
digest `sha256_obj(core) == reviewed_payload_sha256` and
`attested = sha256("C04-PAYLOAD-REVIEW-GO-v8\n" + reviewed_payload_sha256)`. The
payload-review schema is consistent with that reader and constructible.

## Findings — five Informational, none a repair blocker

**I-1 — a breach after 400 completed HateMM forwards yields no seal and no verdict.**
`run.dataset_execution_order = "strictly_serial"` with `run.datasets =
["HateMM","MHC_zh"]`; the seal requires all 800 records. The exit-40 path
preserves per-item checkpoints, but no authorized stage can consume a 400-record
partial, and `resubmit_authorized: false` plus `single_allocation_only: true`
close the namespace. A breach converts a fully completed HateMM dataset into zero
scientific output. No repair available inside a no-clobber namespace; the
GPU-execution authorization should record this explicitly as the accepted
downside.

**I-2 — `guard_seal_reserve_seconds = 600` is the only wholly unmeasured constant
in the window derivation.** v7 never reached the seal. If it is too small the
overrun is killed by the wrapper `timeout` (exit 124, no seal AND no breach
record — strictly worse than exit 40). Mitigating measurements: the guard
deadline sits at entry+4622 while `timeout` fires at entry+4922, so the seal has
>= 900 s even pessimistically (~2200 s in the corroborated case); the seal's
heaviest measurable component is ~1600 `strict_validate_frame_pack` calls
(~6.8 GB of PNG re-hashing), clocked at 3200 hashes in 1.6 s. No repair needed.

**I-3 — MHC-ZH transcript poverty in the frozen selection.** Measured from
`freeze/MHC_zh.source_manifest.json`: MHC_zh p10 = 8 unicode scalars, median
63.5, and **24/200 items carry fewer than 10 scalars**, against HateMM p10 = 38 /
median 999. Zero items are empty, so every prompt renders and nothing halts. If
the tranche returns `KILL_C04_TEACHER_SEMANTIC_RELIABILITY` on MHC_zh, the
failure is not separable from input poverty at this sample size. No repair
permitted (no redraw, no replacement under the amendment); the result-to-claim
stage must carry this caveat.

**I-4 — `HateMM/hate_video_95` is a frozen all-black pack.** Contract-conformant
(`teacher_contract.zero_frame_rule`) and byte-identical to v7's pack. Its
transcript is full (2061 scalars), so the item is not a total void. 2 of 800
forwards see no visual evidence. Named so the usable-rate statistics can be read
with it in view.

**I-5 — the audit ledger pseudonymises inconsistently.** Each `HASH_TRAIN_VIDEO`
event stores `video_id_sha256` (hashed) but `resolved_train_relative` in the
clear (e.g. `"hate_video_334.mp4"`), i.e. the label. **Not** a contract breach —
the amendment binds what a *teacher forward* can read, and the teacher-visible
surface was verified exhaustively clean — but the audit artifact hashes one copy
of the identifier and prints the other. Zero runtime cost; the risk is a later
stage mistaking the audit ledger for a label-free artifact. No repair inside the
no-clobber namespace; recorded here.

## Judgement on the 5.1 % margin

**Acceptable — but the 5.1 % is not the number the decision rests on, and the
record should say so.**

204.4 s is ~46 forwards at the basis rate. As a safety margin it is thin enough
to be consumed by any unmodelled 5 % effect, and if 4.4248 s were the honest
expectation it would be inadequate. It is not the honest expectation; it is a
deliberately wrong-in-the-safe-direction bound, and the wrongness was verified
rather than accepted:

- The basis was measured at **native** resolution (mean 6823.1 pre-merge tokens,
  independently re-measured). The run executes at **capped** geometry (mean
  2848.2 / 2865.4, max 3072) — 2.4x fewer tokens, on the quantity vision
  attention is quadratic in.
- An independent least-squares fit over v7's 400 forwards reproduces
  `1.9913 + 3.567e-04*tokens` exactly. At v8's frozen geometry: **3.0072 s/forward
  -> 2683.5 s total -> 1338.5 s margin (33.3 %)**.
- The 46 v7 forwards where native geometry already equalled capped geometry are a
  direct, unextrapolated measurement in the target regime: **2.9002 s** ->
  2597.8 s -> **35.4 % margin**. Two independent estimators agreeing within 3 %.
- `projected_fixed_overhead_seconds = 208.1` still contains v7's 201
  in-allocation frame-pack builds (~110 s) that v8 does not repeat.
- The one genuinely unmeasured input, MHC-ZH forward cost, is bounded on the
  favourable side by two measurements: its capped geometry is essentially
  identical to HateMM's, and its prompts are **shorter** in tokens (mean 287.5 vs
  482.8; median 253 vs 476), so prefill is smaller, not larger.

To breach, the true mean per forward would have to exceed **4.6803 s** — exceed
v7's *native* mean while running on 2.4x fewer visual tokens. Even pricing every
forward at the regression mean plus one residual sd (3.84 s) lands at ~3350 s, a
17 % margin. No mechanism produces the breach regime.

Two things further reduce the cost of being wrong, verified rather than assumed:
the failure is a clean exit-40 that never truncates or alters an output; and the
campaign accumulator records **actual sacct seconds, not the reservation** — a
clean ~2700 s run leaves ~2500 s of FIRST_TRANCHE headroom for a future C04
namespace rather than burning 7200/7200. The 5222 s is a reservation, not a spend.

The countervailing fact the record must state plainly is I-1: because the seal
requires all 800 records and the namespace cannot be resubmitted, a breach at any
point destroys the scientific output of a fully completed HateMM dataset. That
raises the stakes of the tail; it does not change the arithmetic. **GO.**
