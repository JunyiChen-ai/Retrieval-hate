# M1 CACHE Plan Amendment v2 (input-symlink guard fix; run_id v1→v2 REPLACE)

Date: 2026-07-13

Author: **Claude Opus 4.8**, **m1-prep role only** — separate from the independent amendment reviewer,
the fresh 0C/0H M1 v2 code reviewer, the execution authorizer, and the executor. This authors the v2
plan amendment (REPLACE the spent v1 cache run_ids), the symlink-tolerant guard fix, the lineage
clone-rename, and the re-freeze (`M1_CACHE_V2_FREEZE.md`). It reviews, authorizes, and executes nothing.

Discipline: static amendment + code fix + freeze, plus one approved non-lineage real-path smoke
(`M1_SMOKE2_RECORD.md`). No lineage cache submitted, no git commit, no MHC data or label read.

---

## 0. Why v2

`M1_CACHE_V1_RESULT_TO_CLAIM_REVIEW.md`: the v1 cache runs (`runs[4]` job **13003**, `runs[5]` job
**13004**) **double-burned at Stage-1 evidence-pack build** (~2 s each), before any model load or write.
Root cause (personally re-verified): the train mp4 files `data/video/<ds>/All/*.mp4` are **in-repo
symlinks whose targets live outside the repo** (`/data/jehc223/Multihateclip/…`, `/data/jehc223/HateMM/…`);
the v1 `canonical_root_path` does `resolved = candidate.resolve()` then `resolved.relative_to(ROOT)`,
which **follows the symlink to the external corpus and raises** `path escapes repository root`. The mp4
is never even opened — the guard fires on `.resolve()`.

Readlink topology (review §1.3), re-verified this session: mp4 symlinks escape for **all three**
datasets (MHC 790/790, MHC_zh 806/806, HateMM 1066/1066); `gt`, `ASR`, `lora_frames` are real in-repo
files. The mp4 symlink is the **sole** escape class, but it is dataset-universal.

Fail-closed hygiene is **clean**: `artifacts/lb_scgp_global/v1/m1/` does not exist; no cache/manifest/
ledger/lock; no seal contact; zero scientific information. The v1 lineage of `runs[4]/[5]` is **CLOSED**
(single-submit spent); `runs[6]` seal budget is **untouched**.

## 1. Fix ruling (review §3, adopted)

**Rejected** the coordination proposal "remove the video byte-hash so the mp4 is never opened; blast
radius = builder only" — three factual errors: (i) the mp4 **must** be decoded for 16 frames
(`lora_frames` has only 8 jpg/video; producer never reads it), so it is opened regardless; (ii) the
guard is routed at **both** the builder and producer sites, so a builder-only change still burns the
producer; (iii) the crash is the guard's `.resolve()`, not the hash `open()`, so dropping the hash does
not remove the escape.

**Adopted (B):** a symlink-tolerant containment guard on the video path at **both** sites; retain
`video_sha256` (bytes read via the OS-followed target). Byte-hash removal is orthogonal/optional and
would itself change `evidence_pack_sha256`; not taken.

## 2. Decision — run_id v1→v2 REPLACE-in-place (machine)

- `runs[4].run_id` `…-MHC-v1 → …-MHC-v2`; `runs[5].run_id` `…-MHC_zh-v1 → …-MHC_zh-v2`; `run_order[4]/[5]`
  updated in lock-step.
- `runs[6]` seal `run_id` **unchanged** (`…-SEAL-v1`; its budget was untouched); its `dependencies`
  re-pointed to `[…-MHC-v2, …-MHC_zh-v2]`.
- **Unchanged:** artifact paths, `artifact_schema_ids` (`scgp_global_cache_replica_v2` /
  `scgp_global_cache_seal_v1`), slurm blocks, budgets. Added `v1_burn_and_v2_replace` /
  `v2_dependency_sync` provenance notes on the run rows and a `dependency_dag.m1_v2_replace_record`.

## 3. Decision — symlink-tolerant `canonical_video_path` (code)

New `canonical_video_path(rel, dataset)` in `…_m1_cache_v2_common.py` enforces containment on the
symlink **LOCATION**, not the resolved target:

- `rel` has no `..` component and is repo-relative;
- `rel` is under the preregistered per-dataset root `data/video/<dataset>/All/`;
- the LOCATION (`ROOT/rel`, normalized lexically, **without** following symlinks) stays under `ROOT`, and
  (if it exists) its **parent** directory resolves in-repo (only the mp4 leaf may be a link);
- (if it exists) the leaf is a **regular file or a symlink** (`lstat`, no follow).

It returns the in-repo LOCATION for OS-level decode/hash (the decoder/hasher follows the link at the OS
layer). The external mp4 target is **permitted by design** and recorded in the access ledger
(`followed_target`, `is_symlink`, `followed_target_in_repo`) for audit. The allowlist / `FORBIDDEN_TOKENS`
classifier still runs on `rel`, so `forbidden_path_read_count` / `non_allowlisted_train_content_read_count`
stay 0 and no val/test/label path becomes reachable. **The only relaxation is that the mp4's resolved
target need not be under ROOT** — exactly the designed data layout.

Applied at **both** sites: `evidence_pack_v2.build_dataset_packs` and `producer_v2.main`'s video loop
(and `note_video_read`, which now records the followed target). Verified this session: on a real MHC
symlink mp4 the guard returns the in-repo location without raising (v1 raised here), records the external
target, keeps forbidden counters 0; and it still rejects `..` and non-video-root paths.

## 4. Decision — lineage clone-rename (12 entities v1→v2; 2 schemas carried forward)

Per the realbank v1→v2 precedent, the **12** code/config/wrapper/sbatch entities carrying the code-lineage
`v1` token are cloned to `v2` (`_m1_cache_v2_common.py`, `_m1_evidence_pack_v2.py`,
`_m1_cache_producer_v2.py`, `_m1_cache_seal_v2.py`, the three `m1_cache_*_v2.json` configs, the two
`…_m1_cache*_v2.sh` wrappers, the three `…_m1_cache_*_v2.sbatch`). The behavioral change is **guard-only**;
every other line is a byte-clone with the internal `v1→v2` reference updates (imports, run_id constants,
config/wrapper/sbatch/script paths, `input_builder_hash` source paths).

The **2 artifact schemas** (`scgp_global_cache_replica_v2.schema.json`,
`scgp_global_cache_seal_v1.schema.json`) are **carried forward unchanged**: their `v2`/`v1` is the
**schema-contract** version (bound in `machine.artifact_schemas` and `runs[].artifact_schema_ids`), which
the symlink plumbing fix does not touch. Renaming them would desync the frozen machine
`artifact_schema_ids` and fabricate a contract-version bump the review did not request. The v2 configs
reference these two schema files unchanged; the producer still cross-validates each replica's observables
against the Run1-frozen `scgp_global_cert_v2.schema.json`. **The v1 entities are retained as the burned
lineage record** (not deleted), mirroring the realbank v1 retention.

## 5. Hash cascade (before → after)

| file | before (post-fix v1) | after (v2) |
|---|---|---|
| `EXPERIMENT_PLAN.machine.json` | `7638ac78…` | `ab0a06fb…` |
| `EXPERIMENT_PLAN.md` | `e5ec9bc4…` | `e5ec9bc4…` (unchanged; no literal M1 run_id) |
| `EXPERIMENT_TRACKER.md` | `f36e3dec…` | `86db7a5f…` |
| `EXPERIMENT_PLAN_HASHES.sha256` | `9de299fd…` | `3d603edc…` |

Pre-v2 plan backed up at `EXPERIMENT_PLAN.machine.json.pre_m1_v2_amendment.bak` (`7638ac78…`). The v2
entity SHAs and the configs' post-cascade `authoritative_inputs` are recorded in `M1_CACHE_V2_FREEZE.md`.

## 6. Delta-review mandate (review §4.4) — carried into the freeze

The v2 delta review MUST reproduce and extend the §1.3 per-dataset per-input-root **readlink topology**
check and assert the guard tolerates exactly the video-symlink escape and nothing else. The re-freeze
adds this as a **mandatory simulation-table row**, and the real-path smoke (§7 / `M1_SMOKE2_RECORD.md`)
exercises the **frozen** `canonical_video_path` on real symlinked mp4 (not a re-implementation — the v1
smoke's fatal gap).

## Status flags

- `ready_for_review = true` — ready for the independent v2 amendment review + fresh 0C/0H v2 code review
  (which must re-derive the readlink-topology row and the handoff/simulation tables).
- `ready_for_execution = false` — the six-step v2 gate's steps 5 (exact-hashes/no-clobber) and 6 (one
  re-submit each for MHC-v2 / MHC_zh-v2, then seal) remain, and are not m1-prep's role.

## Required statements

- No performance evidence exists or is claimed; the v1 burn produced zero scientific information and the
  v2 cache is unproduced.
- The only project gold is `parent_video_binary_label`; no segment/frame/span/localization/stance/target/
  mechanism/rationale/fragment gold is introduced. Train labels are not opened; the mp4 read is the
  authorized train-video evidence read; the followed external target is recorded for audit only.
- M2, validation/test, and training remain locked. This amendment unlocks neither v2 execution nor
  anything downstream.
- The m1-prep role is separate from the independent amendment-review, fresh code-review,
  execution-authorization, and executor roles. This document authorizes no execution.
