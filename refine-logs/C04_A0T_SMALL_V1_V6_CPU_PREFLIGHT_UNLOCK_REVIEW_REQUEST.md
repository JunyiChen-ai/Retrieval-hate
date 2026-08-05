# C04-A0T-SMALL-v1 v6 CPU-Preflight Unlock Review Request

Date: 2026-07-30  
Requested verdict form: `GO | REVISE | KILL` with exact
`N Critical / N High / N Important` counts.  
Review mode: **static, read-only**.

## What is being unlocked

Exactly one submission of the fixed CPU-only entrypoint
`scripts/slurm/c04_a0t_small_v1_v6_preflight.sbatch`. Nothing else.

The v5 authority was spent by SLURM job `13805`, which died `FAILED` `1:0` in
`00:00:00` on a self-contradictory static contract (its preflight demanded
`prompt_hashes() == cfg["prompt_hashes"]` while the config still held the
`PENDING_CPU_PREFLIGHT_HASH_FREEZE` sentinel that this very run exists to fill
in). No metric, result, decision or CONTINUE/KILL verdict was published.
`artifacts/c04/` does not exist.

This is unlock round 4. Round 3 returned `GO (0C / 0H / 3I)`; all three were
documentation/hygiene items and none touched an executable byte:
(a) the `__pycache__` disclosure was wrong — the caches were mtime+size-valid so
CPython would have loaded them; they were deleted before submission and the
unlock record now states the mechanism; (b) the manifest pins the
post-correction implementation record rather than the code/resource-reviewed
revision `20814175…` — stated in the unlock record, manifest deliberately not
rebuilt per that reviewer's own instruction; (c) no unlock-review transcription
existed — `refine-logs/C04_A0T_SMALL_V1_V6_CPU_PREFLIGHT_UNLOCK_REVIEW.md`
(`4bf745f74a7b0423d6be7d5c94572f98469ed7fdde30fde5dbcffef14123707e`) now transcribes rounds 1-3. No hash of any executable
artifact moved: config `40ec6d97…`, manifest `5e56041a…`, contract `2b66775c…`
and implementation record `1b2d0bef…` are unchanged from round 3.

Unlock review 1 returned `GO (0C / 0H / 4I)`; unlock review 2 returned
`REVISE (0C / 1H / 2I)` and is closed as follows. **H1**: the documented
reconstruction recipe for the reviewed `preflight.py` revision was false against
the frozen bytes (it yielded `4d4dd033…`, a revision that never existed, because
the recipe was written before the unused imports were removed). The
implementation record now carries the correct two-step recipe plus an explicit
erratum, and the unlock record propagates it; both were verified by
reconstruction against the frozen bytes. **I1**: the unlock record's
"untested surface is exactly the write phase" is replaced by a full enumeration
(video hashing, dense-JL payload generation, merkle roots, ledger/ticket/manifest
assembly, and the write phase). **I2**: the record no longer claims the gates
were run "in the order `main()` calls them" — it states that `main()` was never
called and that `verify_model_snapshot`/`load_dataset_evidence` were invoked
directly. Correcting the implementation record forced a manifest rebuild and a
new config pin; the normalized config contract is unchanged at `2b66775c…`.

Unlock review 1's four Importants were closed as: the manifest's stale `implementation_record_sha256` (the manifest was
rebuilt against the now-final record), the four dead imports (removed), the
unbound `maps.*` geometry declarations (documented as documentation-only in the
implementation record, per that review's stated alternative), and the
unretained reviewed revision (shown to be exactly reconstructible; see finding
4). Verify each closure yourself.

The v6 implementation closure passed independent code/resource review over five
rounds (`REVISE 0C/2H/2I` → `REVISE 0C/1H/2I` → `REVISE 0C/1H/2I` →
`GO 0C/0H/1I` → `GO 0C/0H/1I`), transcribed in
`refine-logs/C04_A0T_SMALL_V1_V6_CODE_RESOURCE_REVIEW.md`. The last Important
(`implementation_authorized` never checked by the preflight gate) was closed in
place afterwards by one added gate in
`scripts/analysis/c04_a0t_small_v1_v6_preflight.py`. **That post-review change
has not itself been independently reviewed — verify it.**

## Snapshot to review

| Artifact | SHA-256 |
|---|---|
| Authorized config `configs/c04/c04_a0t_small_v1_v6.json` | `40ec6d97062498989ff9da21ebd6385aaee7fa3d2071d55b5664a1c5a135fc19` |
| Normalized config contract | `2b66775c44b727e35d52680d39eb838226d4f0a64fffd007d3e50ffcea79cdc5` |
| Authority manifest | `5e56041adc5ef13527803f2c7950834cf59e38238a72dfb7a5c6a61b7e75b52f` |
| Authority closure | `5375c39341933155286640563f5a3d588372acd2668a0cbbe3ba84639592639e` |
| Unlock record | `7a5623204e68b2e305286865af4bdb3fa38c3c6d84fd09f9d42ef0275700706e` |
| Implementation record | `1b2d0bef47f32975c172278ac4d69b06b460d75b9431ab241ba5a5579dba5294` |
| Code/resource review transcription | `0e14bf43dbb5627d539a6a5fdcc4625e0089d4ab912d9dd6ce1ee82731b2a271` |
| Pre-authority reviewed config | `98f2ca603538a22635904c299fa8623352dc516d003da430dcaf642336bdbd94` |
| Entrypoint sbatch | `4051c73eeacf14ca302174c0448b36051685020aa589d8445b048469f46d665f` |
| Fixed wrapper | `22bb4a47c6c21b06bb61b994ea873da682c3332bc0a9dec29c96cc0d1429f770` |

The config's own `implementation_hashes` block pins all 15 implementation files;
`frozen_design_hashes` pins all 15 design files.

## Required findings

1. **Hash and closure integrity.** Recompute every hash above. Recompute the
   authority manifest's `closure_sha256` as the SHA-256 of the canonical JSON
   (`ensure_ascii=False, sort_keys=True, separators=(",",":")`) of its other 15
   keys. Any mismatch is Critical.
2. **Authority/config agreement.** The manifest's `authorization_snapshot`,
   `implementation_hashes`, `frozen_design_hashes`, `design_go_review_sha256`,
   `source_hash_closure`, `model_hash_closure`, `stage`, `verdict` and
   `payload_binding` must match the authorized config and the frozen design GO
   exactly, and must satisfy
   `schemas/c04/c04_a0t_small_v1_v6_stage_authorization.schema.json`.
3. **The flip is minimal and honestly described.** The authorized config must
   differ from the reviewed pre-authority config (`98f2ca60…`) in exactly four
   fields: `preflight_materialization_authorized`, `code_resource_verdict`,
   `code_resource_authorization_sha256` (all three normalized by
   `config_contract_sha256`), and the `implementation_hashes` entry for
   `scripts/analysis/c04_a0t_small_v1_v6_preflight.py`, which is *not*
   normalized and which follows from the post-review gate addition in finding 4.
   Reverting exactly those four must reproduce `98f2ca60…` byte for byte. Any
   fifth difference is at least High. Judge whether the unlock record describes
   this accurately, and whether any already-frozen artifact pins the earlier
   contract hash (the record claims none does).
4. **The two post-review edits.** `scripts/analysis/c04_a0t_small_v1_v6_preflight.py`
   was edited twice after the five code/resource rounds. On a COPY outside the
   repository, starting from the frozen file: (a) **re-insert** the four import
   lines — `SCHEMA_VERSION,` after `    RUN_ID,`; `exclusive_publish_bytes,`
   then `exclusive_publish_json,` after `    dense_rademacher_payload,`;
   `require_exact_keys,` after `    model_hash_closure,` — which must hash to
   `7c64ddf624df8151863f24c9d2e947aea8fd0c232a4e8684aea08833e42056a9`; then
   (b) **delete** the five-line `implementation_authorized` gate block, which
   must hash to
   `8f7dcd44785126a82ba52fe2be4e3c61e4b6f771eb14a839bd016d13faf70111`, the
   revision reviewed in rounds 1-5. Confirm the gate strictly tightens, is
   placed before anything is materialized, and that the four removed names were
   genuinely unreferenced in the file body.
5. **Authorization surface.** Only `implementation_authorized` and
   `preflight_materialization_authorized` may be true. Teacher, GPU, Slurm-GPU,
   small tranche, reconciliation, dev, test, OCR, external API, network,
   cross-dataset, label-value-before-seal, chain, release and resubmit must all
   be false. Payload/GPU/reconciliation review pins must remain unpinned
   sentinels that fail closed.
6. **The entrypoint can do only self-test plus freeze.** The sbatch must be
   CPU-only with no `--time`, no array, no dependency, no GPU request, and the
   wrapper must reach no teacher, GPU, dev/test, label, OCR, network or
   submission path.
7. **Namespace.** `artifacts/c04/a0t_small_v1_impl_v6` and its parent
   `artifacts/c04` must be absent, and no v6 runtime artifact may exist.
8. **Would it fail again?** State explicitly whether any gate on the CPU
   preflight path would fail on these exact bytes. The unlock record documents a
   read-only execution of every gate; treat that as a claim to audit, not as
   evidence.

## Review boundary

Do not run, import or execute any Python. Do not submit, hold, release, cancel
or otherwise touch any SLURM job; read-only `sacct`/`squeue` is allowed. Do not
open dataset label values, videos or model weights. Do not modify, create or
delete any file. Hashing files read-only with `sha256sum` is expected.

A GO unlocks one CPU preflight only. It does not authorize any teacher/GPU
stage, payload acceptance, reconciliation, result or scientific claim.
