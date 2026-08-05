# C04-A0T-SMALL-v1 v6 CPU-Preflight Unlock Record

Date: 2026-07-30  
Status: **AUTHORITY PREPARED / PENDING FRESH INDEPENDENT UNLOCK REVIEW / NOT SUBMITTED**

## What this replaces

The v5 CPU-preflight authority was spent by SLURM job `13805`, which terminated
`FAILED` `1:0` in `00:00:00` on a self-contradictory static contract. That was an
engineering fail-closed HALT: **no metric, result, decision or CONTINUE/KILL
verdict was published**, and `artifacts/c04/` does not exist. The v5
config/source pair is unrunnable by construction and is retired; it must not be
resubmitted. The delayed primary record of that event is
`iteration_8_c04_impl_v5_cpu_preflight_engineering_halt` in `TARGET_STATE.json`.

## Frozen CPU-preflight authority

| Artifact | SHA-256 |
|---|---|
| Authorized config `configs/c04/c04_a0t_small_v1_v6.json` | `40ec6d97062498989ff9da21ebd6385aaee7fa3d2071d55b5664a1c5a135fc19` |
| Normalized config contract | `2b66775c44b727e35d52680d39eb838226d4f0a64fffd007d3e50ffcea79cdc5` |
| Authority manifest `refine-logs/C04_A0T_SMALL_V1_V6_CODE_RESOURCE_AUTHORIZATION.json` | `5e56041adc5ef13527803f2c7950834cf59e38238a72dfb7a5c6a61b7e75b52f` |
| Authority closure | `5375c39341933155286640563f5a3d588372acd2668a0cbbe3ba84639592639e` |
| Implementation record | `1b2d0bef47f32975c172278ac4d69b06b460d75b9431ab241ba5a5579dba5294` |
| Code/resource review transcription | `0e14bf43dbb5627d539a6a5fdcc4625e0089d4ab912d9dd6ce1ee82731b2a271` |
| Code/resource review request | `5e423ed64114eeb6c943cc87eb2e84715e0c1d26b832e07c2dc9a486e52ad248` |
| Pre-authority reviewed config | `98f2ca603538a22635904c299fa8623352dc516d003da430dcaf642336bdbd94` |
| Unlock review transcription | `refine-logs/C04_A0T_SMALL_V1_V6_CPU_PREFLIGHT_UNLOCK_REVIEW.md` |

Two provenance notes, so nothing here reads as more than it is:

- The manifest's `reviewer.implementation_record_sha256` pins `1b2d0bef…`, which
  is the implementation record **as corrected in response to unlock rounds 1 and
  2**. The revision the five code/resource rounds actually bound is
  `208141759d691cf5768eb3195dc4a5d9e0d7c399b9a89d7a4ae1195b576b1862`, pinned
  inside the code/resource review request `5e423ed6…`. The manifest was
  deliberately not rebuilt to add that token, because rebuilding churns
  closure → config pin for no safety gain; it is to be added at the next
  legitimate rebuild.
- The read-only login-node validation wrote `.pyc` caches into the pre-existing
  `scripts/analysis/__pycache__/`. Those caches were mtime+size-valid against the
  pinned sources, so CPython would have loaded them — a bytecode layer outside
  every hash closure, even though it was provably derived from the pinned bytes.
  They were **deleted before submission**, so the job compiles from the
  hash-verified sources. They are in no hash map, so their removal moved no hash.

The authorized config differs from the reviewed pre-authority config
(`98f2ca60…`) in exactly **four** fields — three normalized authority fields
plus one implementation-hash consequence:

- `authorization.preflight_materialization_authorized`: `false` → `true`
- `review.code_resource_verdict`: `PENDING` → `GO`
- `review.code_resource_authorization_sha256`: `PENDING_CODE_RESOURCE_REVIEW` →
  `5e56041a…`
- `implementation_hashes["scripts/analysis/c04_a0t_small_v1_v6_preflight.py"]`:
  `8f7dcd44…` → `c86d439c…`

The first three are normalized by `config_contract_sha256`. The fourth is not,
and it is stated plainly rather than buried: it is the consequence of two
post-review edits to `scripts/analysis/c04_a0t_small_v1_v6_preflight.py` —
(a) the `implementation_authorized` gate that closed the code/resource review's
final Important, and (b) removal of four unused imports (`SCHEMA_VERSION`,
`exclusive_publish_bytes`, `exclusive_publish_json`, `require_exact_keys`) that
the first unlock review flagged. Reverting all four config fields reproduces
`98f2ca60…` byte for byte, which is the check the unlock reviewer should run.

Because the fourth field is inside the contract hash, the reviewed pre-authority
contract and the authorized contract `2b66775c…` are **not** equal; the
authority manifest binds the authorized contract, and no prior artifact pins any
earlier one, so nothing is inconsistent.

Both post-review edits are exactly reconstructible, so the delta is auditable
without an archived copy. On a copy of the frozen file
(`c86d439c…`): **re-insert** the four import lines — `SCHEMA_VERSION,` after
`    RUN_ID,`; `exclusive_publish_bytes,` then `exclusive_publish_json,` after
`    dense_rademacher_payload,`; `require_exact_keys,` after
`    model_hash_closure,` — giving
`7c64ddf624df8151863f24c9d2e947aea8fd0c232a4e8684aea08833e42056a9`; then
**delete** the five-line `implementation_authorized` gate block, giving
`8f7dcd44785126a82ba52fe2be4e3c61e4b6f771eb14a839bd016d13faf70111`, the revision
reviewed in code/resource rounds 1-5.

An earlier revision of this record and of the implementation record stated the
recipe as "delete the gate block" alone, which was true before the imports were
removed and false afterwards (it yields `4d4dd033…`, a revision that never
existed). The first unlock review caught it; the erratum is recorded in the
implementation record rather than silently overwritten. These edits are the
first thing this unlock review must audit.

## Fixed CPU-only entrypoint

- Absolute path: `/data/jehc223/RGCL/scripts/slurm/c04_a0t_small_v1_v6_preflight.sbatch`
- Repository-relative: `scripts/slurm/c04_a0t_small_v1_v6_preflight.sbatch`
- SHA-256: `4051c73eeacf14ca302174c0448b36051685020aa589d8445b048469f46d665f`
- Fixed wrapper: `scripts/wrappers/c04_a0t_small_v1_v6_preflight.sh`
- Wrapper SHA-256: `22bb4a47c6c21b06bb61b994ea873da682c3332bc0a9dec29c96cc0d1429f770`

Submission, once unlocked, is exactly:

```bash
sbatch scripts/slurm/c04_a0t_small_v1_v6_preflight.sbatch
```

The job is CPU-only: 8 CPU / 64 GB, no GPU request, no `--time`, no array, no
dependency, no submission chain. If it lands in `PENDING (JobHeldUser)`, wait
for automatic release — v5's job waited ~7h52m, which is normal. Never force
release.

## Authorization state

Only `implementation_authorized` and `preflight_materialization_authorized` are
true. Teacher, GPU, Slurm-GPU, small-tranche execution, post-job reconciliation,
dev, test, OCR, external API, network, cross-dataset, label-value-before-seal,
chain, release and resubmit are all false. Payload-hash, GPU-execution and
resource-reconciliation reviews remain `PENDING` with unpinned sentinels, which
`_verified_review_file` rejects with `HALT_REVIEW_LINEAGE` for any stage that
tries to run early. Prompt and map payload hashes remain pending; the v6
artifact namespace `artifacts/c04/a0t_small_v1_impl_v6` is absent, and so is its
parent `artifacts/c04`.

## Pre-submission verification actually performed

Unlike v1-v5, this authority was not frozen on static reading alone — that is
exactly how a program that could never pass its own first gate reached the
queue. Every CPU-preflight gate was executed read-only on the login node against
these exact authorized bytes, stopping before `preflight()` so nothing was
staged, published or renamed. `main()` itself was never called: the first three
gates are the ones `main()` invokes, in that order; `verify_model_snapshot` and
`load_dataset_evidence` are reached only from inside `preflight()` and were
therefore invoked directly, with the same config object:

| Gate | Result |
|---|---|
| `verify_static_config` | PASS; config prompt-hash binding `SENTINEL_PENDING_CPU_PREFLIGHT_FREEZE` |
| `verify_code_resource_authorization` | PASS; manifest pin `5e56041a…` |
| `run_self_tests` | PASS; 25 checks, `all_passed=True` |
| `verify_model_snapshot` | PASS in 13.3 s; model tree `55705d03…`, processor tree `f77f6022…`, 8 + 6 files |
| `load_dataset_evidence` HateMM | PASS; 200 selected of 744; `label_field_syntactically_skipped=744`, `label_value_materialized=0` |
| `load_dataset_evidence` MHC_zh | PASS; 200 selected of 579; `label_field_syntactically_skipped=579`, `label_value_materialized=0` |
| all 400 selected videos resolve | PASS (stat only, no hashing) |
| freeze payload build + verify | PASS; literal, sentinel-free |
| staged path containment | PASS; all 15 staged paths inside the namespace |

No label value was materialized, no video or model byte was written, no teacher
or GPU ran, no SLURM command was issued, and `artifacts/c04/` was absent before
and after.

Because `preflight()` was never entered, the untested surface is larger than the
write phase alone. It is, in full:

- `sha256_file` over the 400 selected videos (~3.3 GB) — the table above is
  stat-only and does **not** cover video hashing;
- `dense_rademacher_payload` for the 256x3598 LE3 and 256x1024 additive payloads
  (~1.18M SHA-256 evaluations);
- `merkle_root` over the 200-row allowlist and source-manifest rows per dataset;
- assembly and `sha256_obj` of the GPU ledger, resource ticket, access ledger
  and preflight manifest;
- the write phase itself: staging into a temp directory, `fsync`, and the atomic
  `os.rename`.

None of these has a known failure mode — the payload generators are pure
functions over module constants, `struct.pack(">H", col)` is safe for the
maximum column index 3597, and the rename target is confirmed absent on the same
filesystem with 1.8 TB free — but they are untested, and this record exists
because v5 was frozen on an over-confident static claim.

## Boundary after unlock

A GO authorizes exactly one submission of the fixed CPU-only entrypoint above.
It does not authorize teacher inference, any GPU allocation, the small tranche,
reconciliation, dev/test evaluation, OCR, external API or network access, label
access, chained submission, release, resubmission, or reuse of any existing
artifact namespace. After the preflight terminates, an independent
collector/reviewer must inspect the frozen artifacts and issue a fresh
payload-review verdict; no GPU or downstream stage becomes authorized merely
because the CPU preflight succeeds.
