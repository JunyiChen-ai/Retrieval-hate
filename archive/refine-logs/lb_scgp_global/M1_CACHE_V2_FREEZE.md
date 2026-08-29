# M1 CACHE v2 Implementation Freeze (input-symlink guard fix)

Date: 2026-07-13

Author: **Claude Opus 4.8**, **m1-prep role only** — separate from the independent amendment reviewer,
the fresh 0C/0H v2 code reviewer, the execution authorizer, and the executor. Freezes the v2 M1 cache
implementation (symlink-tolerant containment guard) and records the SHAs, the fix diff, the three
handoff tables, and the runtime cross-check simulation table **with the new mandatory per-dataset
per-input-root readlink-topology row** (M1_CACHE_V1_RESULT_TO_CLAIM_REVIEW.md §4.4). It authorizes no
execution.

Discipline: static amendment + code fix + freeze, plus one approved non-lineage real-path smoke
(`M1_SMOKE2_RECORD.md`). `py_compile`, `bash -n`, `jq`, `sha256sum`, `grep`, `find`, `readlink`, and a
light in-process unit test of the frozen `canonical_video_path` on a real symlinked mp4 (pure path/lstat,
no GPU/decode/label) were the only executions. No lineage cache submitted; no git commit; no MHC label
read. `artifacts/lb_scgp_global/v1/m1/` remains absent.

Root cause and fix ruling: `M1_CACHE_V1_RESULT_TO_CLAIM_REVIEW.md` + `M1_CACHE_PLAN_AMENDMENT_V2.md`. In
one line: the train mp4s are in-repo **symlinks escaping the repo**; the v1 `canonical_root_path`
`resolve()` fired at the video site (double-burn, jobs 13003/13004). v2 = REPLACE run_ids v1→v2 + a
**symlink-tolerant `canonical_video_path`** at the builder+producer video sites; `video_sha256` retained;
zero other isolation weakening.

---

## 1. Entities frozen (14) — SHA256

**12 clone-renamed v1→v2** (behavioral change = guard-only; all else byte-clone + internal `v1→v2` ref
updates):

| # | entity | SHA256 |
|---|---|---|
| 1 | `scripts/analysis/lb_scgp_global_r2_m1_cache_v2_common.py` | `56fbb403675db0f5ce34415bf36c145877d3bf0291a5cca1720f7508c90c6ebe` (canonical_video_path + note_video_read fix) |
| 2 | `scripts/analysis/lb_scgp_global_r2_m1_evidence_pack_v2.py` | `54a0a97e6f0843bca5efb3040af1f0e858bdb9e801a414f2a8cc623025cd77cc` (build_dataset_packs video site) |
| 3 | `scripts/analysis/lb_scgp_global_r2_m1_cache_producer_v2.py` | `ee34eb9a7bf2e74faf41073bef93a424b1af077e55dcea87581c160c8bda9a3a` (main video loop) |
| 4 | `scripts/analysis/lb_scgp_global_r2_m1_cache_seal_v2.py` | `62d1d100668eb7ebc5bf3298c9ca6d2404251943403b601dfa0c9492bc0d15f6` (byte-clone; import→v2_common) |
| 5 | `configs/lb_scgp_global_r2/m1_cache_mhc_v2.json` | `4e42013dc133a75688877de80f802fe89062fb937596df137a311cfbc64f8ed9` |
| 6 | `configs/lb_scgp_global_r2/m1_cache_mhc_zh_v2.json` | `ce9bcab2d324c1efb2fa49e6a3d787c3e5ab9a742b96fad8cc6f4e9a8647c764` |
| 7 | `configs/lb_scgp_global_r2/m1_cache_seal_v2.json` | `6d6b4faf364b0967befda1ef75f0cd9645294168289092e58d6e69912c6a8270` |
| 8 | `scripts/wrappers/lb_scgp_global_r2_m1_cache_v2.sh` | `4c165fe2038f773498ab405ca1cbd5e4b58d9c7863a9a55a1f447f6d99b050a7` |
| 9 | `scripts/wrappers/lb_scgp_global_r2_m1_cache_seal_v2.sh` | `17f47f609c1fcd8eb413359aa0e90b5663d9f9cb878c0284e980ea36fb90793a` |
| 10 | `scripts/slurm/lb_scgp_global_r2_m1_cache_mhc_v2.sbatch` | `ac986d8b0e7b8db90d5e5c333148c5a8722098a330358e129836f913e6197b49` |
| 11 | `scripts/slurm/lb_scgp_global_r2_m1_cache_mhc_zh_v2.sbatch` | `8be1f45411548171b9c34b9951554990a9fc469fc1474e37246908b292f829e7` |
| 12 | `scripts/slurm/lb_scgp_global_r2_m1_cache_seal_v2.sbatch` | `e8f8e70651ad7fce73a52d4c5bca425279f66b65e24f198b208b38b2dd2e1997` |

**2 artifact schemas carried forward UNCHANGED** (contract-versioned identifiers bound in
`machine.artifact_schemas` / `runs[].artifact_schema_ids`; the symlink fix does not touch the cert/seal
contract, so renaming them would desync the frozen machine bindings):

| # | entity | SHA256 (unchanged since v1 freeze) |
|---|---|---|
| 13 | `schemas/lb_scgp_global_r2/scgp_global_cache_replica_v2.schema.json` | `4bfcfea2d4dd38fd8a8125fb803fbde0c5ec05fa12a96d13b908246a6f03f68d` |
| 14 | `schemas/lb_scgp_global_r2/scgp_global_cache_seal_v1.schema.json` | `f4605bb7bd26f730c75e20636841f243f0bf10080b6ea12a70363d9a42790ce1` |

**The v1 entities are retained** as the burned-lineage record (not deleted), mirroring realbank v1
retention. Run1-frozen `scgp_global_cert_v2.schema.json` (`4d3f1663…`) still cross-validated per replica.

Amendment lineage (bound in the v2 configs' `authoritative_inputs`): `M1_CACHE_PLAN_AMENDMENT_V2.md`
`5f0036e8…`; `.machine.json` `df1dea76…`; `_HASHES.sha256` `be189f4d…`; plus the burn diagnosis
`M1_CACHE_V1_RESULT_TO_CLAIM_REVIEW.md` `2fe32335…`. Plan cascade (post-fix-v1 → v2):
`EXPERIMENT_PLAN.machine.json` `7638ac78…`→`ab0a06fb…`; `EXPERIMENT_PLAN.md` `e5ec9bc4…` (unchanged);
`EXPERIMENT_TRACKER.md` `f36e3dec…`→`86db7a5f…`; `EXPERIMENT_PLAN_HASHES.sha256` `9de299fd…`→`3d603edc…`.
Pre-v2 plan backed up at `EXPERIMENT_PLAN.machine.json.pre_m1_v2_amendment.bak` (`7638ac78…`). The v2
configs' `authoritative_inputs` bind the post-cascade quartet (verified).

## 2. Fix diff

**`canonical_video_path(rel, dataset)`** (new, `…_v2_common.py`): containment on the symlink **LOCATION**
(no `..`; under `data/video/<dataset>/All/`; `ROOT/rel` lexically under `ROOT`; parent dir resolves
in-repo; leaf is regular-file-or-symlink via `lstat`). Returns the in-repo location for OS-follow
decode/hash; the external target is **not** required under `ROOT`. **`note_video_read`** rewritten to call
`canonical_video_path`, still run `forbidden_reason` on `rel`, and record `{is_symlink, followed_target,
followed_target_in_repo}` for audit. Sites changed (was `canonical_root_path`):
`evidence_pack_v2.build_dataset_packs`, `producer_v2.main` video loop. `video_sha256` retained (bytes via
followed symlink). `input_builder_hash` source paths updated to the v2 builder+common.

Verified this session on a real MHC symlink mp4 (`…/Multihateclip/English/video_mp4/…`): the guard
**returns the in-repo location without raising** (v1 raised here), `location.exists()` follows the link,
the ledger records the external `followed_target` with `followed_target_in_repo=false` and
`is_symlink=true`, and all forbidden zero-counters stay 0. Negative cases (`..` traversal, non-video-root)
still raise fail-closed.

## 3. Dependencies (unchanged from v1)

`torch` 2.6.0, `transformers` 4.49.0, `numpy` 1.26.4, `jsonschema` 4.26.0, `decord` 0.6.0 or `av` 17.0.0;
in-repo `utils.generate_subclip_embedding_HF.load_video_frames`. No `qwen_vl_utils`. Function-level imports
enumerated + `dependency_check()` fails closed before the model loads.

## 4. Per-run handoff tables (video row updated to the symlink-tolerant guard)

Identical structure to the v1 freeze §4; the only change is the **train-video row**:

| run | video-read row: writer/path/guard → reader/guard | verdict |
|---|---|---|
| cache-MHC-v2 / cache-MHC_zh-v2 | frozen mp4 **symlink** `data/video/<ds>/All/<id>.mp4` (target out-of-repo) → builder `canonical_video_path` (location-containment) + `note_video_read` (records followed target) + `sha256_file`; producer `canonical_video_path` + `load_video_frames` | **PASS** (v1 was FAIL here; decode DEFERRED to smoke/runtime) |
| seal-v1 | reads only the two producers' in-repo `cache.jsonl`/`cache_manifest.json` (no video) | PASS (needs run4/run5) |

All other rows (config, machine plan, replica/cert/seal schemas, gt title, ASR, model weights, cache.jsonl,
manifest, ledger, seal decision) are unchanged from the v1 freeze §4 and remain 0-FAIL, in-repo, no
`$TMPDIR`. The mp4 external target is the **only** out-of-repo path, and it is authorized-by-design train
evidence recorded for audit — not a containment breach.

## 5. Runtime cross-check simulation table — with the MANDATORY new row

| Row | Assert (site) | Reads | Static verdict | PASS? |
|---|---|---|---|---|
| 1 | wrapper `RUN_ID==EXPECTED` + `.run.*` via `jq` (v2 configs) | config | MHC-v2 / MHC_zh-v2 / SEAL-v1 all match | **PASS** |
| 2 | `require_slurm_cache` `CUDA_VISIBLE_DEVICES` (FIX-2 carried) | env | sbatch `--gres=gpu:a100:1` | **PASS (runtime; smoke2-confirmed)** |
| 3 | `dependency_check` | env | present (HateVideo) | **PASS (runtime)** |
| 4 | `bash -n` (5 v2 scripts) + `py_compile` (4 v2 modules) this session | scripts | all clean | **PASS** |
| 5 | `verify_config` + `verify_machine_cache`/`_seal` (v2) | config+machine | runs[4]/[5]→v2, runs[6] deps→v2, index+run_id+slurm+model+frames+replicas+ocr all match (verified) | **PASS** |
| 6 | replica/cert_v2/seal schema validation (unchanged) | schemas | valid Draft-07, strict; synthetic records validate | **PASS** |
| 7 | **`canonical_video_path` tolerates the mp4 symlink; rejects `..`/non-video-root** | fs (lstat) | real MHC symlink → in-repo location, no raise; `..`/non-root → raise (verified) | **PASS** |
| **8 (NEW MANDATORY)** | **per-dataset per-input-root readlink topology; guard tolerates exactly the video-symlink escape and nothing else** | fs (readlink) | see §5.1 below — video mp4 escapes for all 3 datasets; gt/ASR/lora_frames real in-repo | **PASS** |
| 9 | forbidden zero_counters all 0 at seal | manifest | ledger opens only allowlisted evidence; video read is authorized; every forbidden path raises | **DEFERRED-TO-RUNTIME (fail-closed)** |
| 10 | model load + R=4 decode/generate on **symlinked** mp4 | weights+mp4 | offline weights present; guard returns location; decode follows link | **PASS (empirically settled by M1_SMOKE2_RECORD.md)** |
| 11 | no-clobber (producer/seal + wrapper trap) | artifact dir | `artifacts/lb_scgp_global/v1/m1/` absent | **PASS** |

### 5.1 MANDATORY row 8 — per-dataset per-input-root readlink topology (review §1.3, re-verified)

| dataset | video `*.mp4` | escapes repo? | target root | `lora_frames` | `gt/train.jsonl` | `ASR/…train…` |
|---|---|---|---|---|---|---|
| MHC | symlink 790/790 | **YES 790/790** | `/data/jehc223/Multihateclip/English/video_mp4/` | 0 symlinks (real) | real in-repo | real in-repo |
| MHC_zh | symlink 806/806 | **YES 806/806** | `/data/jehc223/Multihateclip/Chinese/video/` | 0 symlinks (real) | real in-repo | real in-repo |
| HateMM (smoke) | symlink 1066/1066 | **YES 1066/1066** | `/data/jehc223/HateMM/video/` | 0 symlinks (real) | real in-repo | real in-repo |

**The mp4 symlink is the SOLE escape class across all input roots, dataset-universal.** `canonical_video_path`
tolerates exactly this escape (only under `data/video/<ds>/All/`, only the leaf may be a link, parent must
resolve in-repo) and nothing else; `gt`/`ASR`/`lora_frames` reads still route through `canonical_root_path`
(in-repo containment) unchanged. This is the row the v1 handoff/simulation tables lacked — the third
blind-spot class (input symlink topology) is now modeled per dataset.

## 6. Zero-gold self-attestation (v2 code)

`grep` over the four v2 `.py`: **0** gold-field ACCESS patterns (no `["label"]`/`.get("label")`/split/
neighbor/query/prediction/margin). Same disposition as v1 (comments, `label_read_allowed` flag, output
booleans, `label_text` param). The evidence pack fields are unchanged; the new ledger record adds only
`{is_symlink, followed_target, followed_target_in_repo}` audit fields (no label/split/seed/neighbor). Train
labels are not opened; the mp4 read is authorized train evidence.

## 7. Residuals for the fresh v2 code review

- **R-1 (guard scope).** `canonical_video_path` relaxes containment **only** for the mp4 leaf under
  `data/video/<ds>/All/` and only on the resolved target; parent-dir escape, `..`, and non-video-root all
  still raise. The reviewer should confirm no other input site routes an escaping symlink through
  `canonical_root_path` (readlink row 8 shows none do).
- **R-2 (video_sha256 retained).** Per the review, byte-hash removal is orthogonal/optional and would
  change `evidence_pack_sha256`; not taken. The builder reads each mp4 once (hash) and the producer decodes
  once — both via the followed symlink.
- **R-3 (schemas carried forward).** The 2 artifact schemas are unchanged (contract-versioned); the v2
  configs bind them. Confirm the machine `artifact_schema_ids` still read `scgp_global_cache_replica_v2` /
  `scgp_global_cache_seal_v1` (they do).
- **R-4 (smoke fidelity).** `M1_SMOKE2_RECORD.md` calls the **frozen** `canonical_video_path` on real
  symlinked HateMM mp4 (the burn surface), not a re-implementation — closing the v1 smoke gap.

## Status flags

- `ready_for_review = true` — ready for the independent v2 amendment review + fresh 0C/0H v2 code review
  (must re-derive §4, §5 incl. the mandatory row 8, and §6).
- `ready_for_execution = false` — the six-step v2 gate's step 5 (exact-hashes/no-clobber) and step 6 (one
  re-submit each for MHC-v2 / MHC_zh-v2, then seal) remain, and are not m1-prep's role.

## Required statements

- No performance evidence exists or is claimed; the v1 burn produced zero scientific information.
- The only project gold is `parent_video_binary_label`; none introduced. Train labels not opened; the mp4
  read is authorized train evidence; the followed external target is recorded for audit only.
- M2, validation/test, and training remain locked; this freeze unlocks neither v2 execution nor anything
  downstream. The m1-prep role authorizes no execution.
