# M1 CACHE Execution Record — runs[4]/[5] double-FAILED (repo-escape guard on symlinked videos)

Date: 2026-07-13

Author: **Claude Opus 4.8**, **independent M1 execution-authorization + executor role** — separate
from m1-prep, the amendment reviewer, and the fresh 0C/0H code reviewer. This record closes the
executor's obligation for the M1 cache block. The v2 lineage / remediation is handled by a separate
role and is **out of scope here**.

Model declaration: ran as Claude Opus 4.8 (`claude-opus-4-8`, 1M context), per CLAUDE.md. No
deviation.

Discipline: authorized submissions only; no frozen entity/config/schema/plan edited; no resubmission;
evidence gathered read-only (`sacct`, `ls`, `find`, `readlink`, log/source reads). Every fact below
was re-verified from the primary logs and on-disk state this session (numeric-provenance discipline),
not transcribed from the diagnosis handed to me.

---

## 0. Outcome

**Both cache runs FAILED at preflight-adjacent stage in ~2 s each; NO artifact produced; the failure
is a fail-closed repo-escape guard, not corruption.** The seal (runs[6]) was **never submitted** (its
gate — both caches COMPLETED — was not met). Per the authorization scope (§7 of
`M1_CACHE_EXECUTION_AUTHORIZATION.md`), a FAILED cache is **not** resubmitted: evidence collected,
reported to `main`, awaiting result-to-claim. The single submit of runs[4] and runs[5] is **spent**.

| run | job | state | exit | elapsed | artifact | single-submit |
|---|---|---|---|---|---|---|
| runs[4] CACHE-MHC-v1 | 13003 | **FAILED** | 1:0 | 00:00:02 | none | **SPENT — do not resubmit** |
| runs[5] CACHE-MHC_zh-v1 | 13004 | **FAILED** | 1:0 | 00:00:02 | none | **SPENT — do not resubmit** |
| runs[6] CACHE-SEAL-v1 | — | **NOT SUBMITTED** | — | — | none | **UNTOUCHED (budget intact)** |

---

## 1. Timeline (sacct, independently re-queried)

`sacct -j 13003,13004` this session:

| JobID | JobName | Submit | Start | End | Elapsed | State | ExitCode |
|---|---|---|---|---|---|---|---|
| 13003 | `lbscgp_global_r2_m1_cache_mhc_v1` | 15:49:14 | 15:49:19 | 15:49:21 | 00:00:02 | FAILED | 1:0 |
| 13003.batch | batch | 15:49:19 | 15:49:19 | 15:49:21 | 00:00:02 | FAILED | 1:0 |
| 13004 | `lbscgp_global_r2_m1_cache_mhc_zh_v1` | 15:49:14 | 15:49:19 | 15:49:21 | 00:00:02 | FAILED | 1:0 |
| 13004.batch | batch | 15:49:19 | 15:49:19 | 15:49:21 | 00:00:02 | FAILED | 1:0 |

Both were submitted at 15:49:14 (this session, exactly one each), auto-released from `JobHeldUser`
within ~5 s (not forced), started 15:49:19, and died 15:49:21. `.out` files are 0 bytes; `.err` files
carry the traceback (`slurm/logs/lbscgp_global_r2_m1_cache_mhc_v1_13003.err` 1586 B;
`…_mhc_zh_v1_13004.err` 1587 B). The failure preceded any GPU/model activity — it is in the very
first pipeline stage (evidence-pack build), before model load, decode, or `generate`.

## 2. Root cause (independently confirmed from the primary logs + source)

**Failure site — identical for both datasets** (`…_13003.err` / `…_13004.err`, lines 11–23):

```
producer_v1.py:329  raise SystemExit(main())
producer_v1.py:136  built = evpack.build_dataset_packs(args.dataset, ledger, hash_videos=True)
evidence_pack_v1.py:183  video_fs, _ = canonical_root_path(video_rel)
common.py:151       raise RuntimeError(f"path escapes repository root: {path}") from exc
        (from common.py:149  rel = resolved.relative_to(root)  -> ValueError)
```

- MHC/13003: `RuntimeError: path escapes repository root: data/video/MHC/All/-0IpEC2xXT0.mp4`; the
  ValueError names the resolved target `/data/jehc223/Multihateclip/English/video_mp4/-0IpEC2xXT0.mp4`.
- MHC_zh/13004: `path escapes repository root: data/video/MHC_zh/All/BV117421N7HM.mp4`; resolved
  target `/data/jehc223/Multihateclip/Chinese/video/BV117421N7HM.mp4`.

**Mechanism (source-verified):** `canonical_root_path` (`common.py:143–152`) does
`resolved = candidate.resolve()` (line 147) — which **follows the symlink to its real target** — then
`resolved.relative_to(root)` (line 149). Because the real target lives at `/data/jehc223/Multihateclip/`
(outside `/data/jehc223/RGCL`), `relative_to` raises `ValueError`, re-raised as the repo-escape
`RuntimeError` at line 151. In `build_dataset_packs`, `canonical_root_path(video_rel)` is called
**unconditionally at line 183**, for the first train video in sorted order — **before** the
`if video_fs.exists() and hash_videos:` gate at line 184. So the guard fires on path canonicalization
itself; the `hash_videos=True` flag governs only the downstream read/hash at 184–186, which is never
reached. (The producer was going to hash every train mp4 for `evidence_pack_sha256` — freeze R-2 /
review M-3 — but the block happens one step earlier, at `resolve()`.)

**Data layout is the trigger, not missing/corrupt data.** Independently confirmed:
`data/video/MHC/All/-0IpEC2xXT0.mp4` and `data/video/MHC_zh/All/BV117421N7HM.mp4` are **symlinks**
(created Jul 1) pointing into `/data/jehc223/Multihateclip/{English,Chinese}/…`; both targets **exist
and are readable** (`test -e` true, `readlink -f` resolves). The videos are present and fine — the
guard simply refuses a resolved path outside the repo root. This is exactly the "no out-of-repo path"
invariant the code review praised, meeting the real dataset layout for the first time (the smoke used
HateMM videos, whose symlink resolution evidently did not escape, so this path was not exercised).

**Class:** fail-closed guard, deterministic, dataset-independent (both datasets hit it at the first
video). No science/scope/label issue; no corruption possible.

## 3. Fail-closed hygiene verification (independently checked)

- `artifacts/lb_scgp_global/v1/m1/` — **ABSENT** (`ls` errors "No such file or directory"). No
  `cache/MHC/`, no `cache/MHC_zh/`, no `cache.jsonl`, no `cache_manifest.json`, no `access_ledger.json`,
  no `cache_seal_decision.json` were created. The producer died in `build_dataset_packs`, upstream of
  any `exclusive_publish_json[l]` call, so nothing was ever written or half-written.
- `find artifacts/lb_scgp_global/v1/ -name '*.publish.lock' -o -name 'cache*.json*' …` returns **only
  pre-existing m0 artifacts** (`m0/contract_freeze.json.publish.lock`, `m0/realbank_resource/*`) —
  none from M1. No stray M1 lock, no partial M1 output. Clean.
- Seal job name `lbscgp_global_r2_m1_cache_seal_v1` — **ZERO** rows in `sacct` full history
  (`--starttime 2020-01-01`). runs[6] was never submitted; its single-submit budget is intact. Per the
  authorization, the seal is gated on both caches COMPLETED — unmet — so it correctly was not, and must
  not be, submitted for this (v1) lineage.
- The persistent progress monitor self-terminated on detecting both jobs FAILED (both-terminal exit
  condition); it wrote no repo state.

## 4. Single-submit accounting (binding)

- runs[4] (13003) and runs[5] (13004): single submit **SPENT** (each job ran and died). **Do not
  resubmit** either job name for the v1 lineage (authorization §7).
- runs[6] seal: **never submitted**, budget **untouched**; not eligible under v1 (gate unmet).
- Any remediation requires a **new (v2) lineage** with fresh job names, out of this executor's scope.

## 5. Note for the v2 remediation role (informational only — not a fix by me)

The block is a one-line data-access-layer mismatch: `canonical_root_path` hard-resolves symlinks and
rejects targets outside the repo, but the contract train videos are symlinks into
`/data/jehc223/Multihateclip/`. Any v2 fix (e.g. allowlisting the external video root, comparing the
un-resolved in-repo path, or de-referencing the symlink dir under an explicit data-root allowlist) is
a **frozen-entity change** requiring re-freeze + re-review + a fresh execution authorization — none of
which this executor performs. Recorded here only so the v2 role has the exact failure coordinates:
`common.py:147 resolve()` + `:149 relative_to(root)`, called at `evidence_pack_v1.py:183`.

## 6. Required statements

- No performance evidence exists or is claimed; both runs died before any inference. No accuracy /
  macro-F1, no training, no kNN, no MLLM/OCR output produced.
- No train label and no validation/test content or label was read; the failure occurred while opening
  train title/ASR + resolving a train-video path (allowlisted train evidence only). The only project
  gold `parent_video_binary_label` was not opened.
- M2, validation/test, and training remain locked. No seal decision was produced; nothing downstream
  is unlocked.
- Executor = Claude Opus 4.8. Actions this phase: two authorized cache submissions (exactly one each),
  read-only evidence collection, this record, and one status message to `main`. No frozen entity was
  edited; no job was resubmitted; no Python compute was run on the login node.
