# M1 CACHE v1 lineage — result-to-claim final judgment (runs[4] job 13003 / runs[5] job 13004)

Reviewer: fresh, zero-context, independent result-to-claim role.
Date: 2026-07-13. Scope: **read-only** verification of the failed v1 cache producers + this
single authored file. No experiments, no code edits, no re-submits.

Verdict in one line: **infrastructure preflight burn — INPUT symlink escapes the repo-root
containment guard. Zero scientific information. Fail-closed hygiene clean (no artifacts).
v1 lineage of runs[4]/[5] is CLOSED. runs[6] seal budget untouched.** The coordination
meeting's proposed fix ("remove video byte-hash so the mp4 is never opened; blast radius =
builder only") is **factually incorrect on three counts** and must not be adopted as stated;
the correct fix is a **symlink-tolerant containment guard applied to BOTH the builder and the
producer video-path sites** (details in §3).

---

## 1. Facts — personally verified

### 1.1 The two deaths (primary logs, read directly)

Both jobs died in ~2 s during **Stage 1** (evidence-pack build), before any model load,
any inference, any write:

- **runs[4] job 13003** (`lbscgp_global_r2_m1_cache_mhc_v1_13003.err`):
  `RuntimeError: path escapes repository root: data/video/MHC/All/-0IpEC2xXT0.mp4`
  underlying `ValueError: '/data/jehc223/Multihateclip/English/video_mp4/-0IpEC2xXT0.mp4' is
  not in the subpath of '/data/jehc223/RGCL'`.
- **runs[5] job 13004** (`..._mhc_zh_v1_13004.err`):
  `RuntimeError: path escapes repository root: data/video/MHC_zh/All/BV117421N7HM.mp4`
  underlying target `/data/jehc223/Multihateclip/Chinese/video/BV117421N7HM.mp4`.

Identical stack for both:
`producer_v1.py:329 → main():136  build_dataset_packs(hash_videos=True)
→ evidence_pack_v1.py:183  canonical_root_path(video_rel)
→ common.py:149  resolved.relative_to(root)  → ValueError
→ common.py:151  raise RuntimeError("path escapes repository root")`.

### 1.2 Root cause — the guard fires on `.resolve()`, NOT on the hash `open()`

`canonical_root_path` (common.py:143-152) does `resolved = candidate.resolve()` then
`resolved.relative_to(root)`. `data/video/{ds}/All/{vid}.mp4` is a **symlink**; `.resolve()`
follows it to the external corpus, and `relative_to(/data/jehc223/RGCL)` raises. **The crash
is the symlink-follow inside the containment guard — the mp4 is never even opened/read.** This
distinction is load-bearing for the fix verdict (§3): anything that removes only the *hashing*
`open()` does not remove the *resolve* that actually crashes.

### 1.3 Input symlink topology — full readlink sweep of every input root (the mandated check)

| dataset | `data/video/{ds}/All/*.mp4` | symlink target | escapes repo? | `lora_frames/{ds}/…` | `gt/{ds}/train.jsonl` | `ASR/{ds}/…train…jsonl` |
|---|---|---|---|---|---|---|
| MHC | symlink, **790/790** | `/data/jehc223/Multihateclip/English/video_mp4/` | **YES 790/790** | in-repo real (8 jpg/vid) | in-repo real | in-repo real |
| MHC_zh | symlink, **806/806** | `/data/jehc223/Multihateclip/Chinese/video/` | **YES 806/806** | in-repo real (8 jpg/vid) | in-repo real | in-repo real |
| HateMM (smoke) | symlink, **1066/1066** | `/data/jehc223/HateMM/video/` | **YES 1066/1066** | in-repo real (8 jpg/vid) | in-repo real | in-repo real |

**The mp4 symlinks are the SOLE escape point across all input roots, and they escape for ALL
THREE datasets — HateMM included.** `lora_frames` (0 symlinks among frame files; verified with
`find -type l` = 0), `gt`, and `ASR` are real in-repo files. Their parent dirs are real dirs,
not symlinks. So there is exactly one link-escape class to fix, but it is dataset-universal.

### 1.4 Why HateMM smoke (job 13002) did not catch it — the real reason is worse than "wrong dataset"

The premise "HateMM didn't trigger it, so check whether HateMM is a real file" is **false**:
HateMM mp4 are symlinks escaping the repo exactly like MHC/MHC_zh (1066/1066 above). The smoke
did not trip the guard because **the smoke script is a throwaway RE-IMPLEMENTATION of the
producer inner loop that bypasses `canonical_root_path` entirely**:

- `lb_scgp_global_r2_m1_smoke_nonlineage.py:131` builds the video path with a plain join,
  `vpath = ROOT / HATEMM_VIDEO / f"{vid}.mp4"` — **no `canonical_root_path`, no `.resolve()`**.
- Line 133-134: `if vpath.exists(): load_video_frames(str(vpath), 16)` — `exists()` and the
  decoder follow the symlink at the OS level and succeed (10/10 decoded).
- It imports only `build_user_prompt`, `parse_certificate`, `require_slurm_cache`,
  `build_messages`. It **does NOT** import/run `build_dataset_packs` or the producer `main()` —
  the exact functions that call `canonical_root_path(video_rel)`.

Consequence: **the frozen `build_dataset_packs` + guard-on-mp4 path was never exercised by any
preflight, on any dataset.** A HateMM smoke of the *real* code path would have caught this,
because HateMM mp4 escape too. The smoke's own claim "the exact sealed code path is exercised"
is true for GPU/decode/parse/determinism but **false for the video-path guard** — precisely
where it broke.

### 1.5 Fail-closed hygiene — clean

`artifacts/lb_scgp_global/v1/m1/` **does not exist**. No `cache.jsonl`, no `cache_manifest.json`,
no `access_ledger.json`, no `*.publish.lock`. Death at Stage 1 (line 183) is upstream of every
`exclusive_publish_*` write and upstream of `require_slurm_cache` model load. No partial artifact,
no isolation-counter perturbation, no seal contact. **No product exists; nothing to reconcile.**

### 1.6 Milestone-state consequences

- **v1 lineage of runs[4]/[5] is CLOSED** (single-submit discipline burned; the run_ids
  `LBSCGP-GLOBAL-M1-CACHE-{MHC,MHC_zh}-v1` are spent and cannot be re-submitted).
- **runs[6] `…-SEAL-v1` budget is UNTOUCHED** — it depends on the (nonexistent) cache outputs of
  runs[4]/[5] and was never eligible to launch. verified: machine plan runs[6] deps =
  `[…-MHC-v1, …-MHC_zh-v1]`.
- **Zero scientific information.** No accuracy/F1/consensus/parse-rate datum was produced; the
  M1 cache remains entirely unproduced. This burn tells us nothing about the method — only about
  the plumbing.

---

## 2. Qualitative classification

**Infrastructure preflight burn, same path-hardening family as the realbank-v1 burn (job 12994).**
Both die with the identical `RuntimeError: path escapes repository root` from the identical guard
(`canonical_root_path`), but from opposite ends of the pipeline:

| | realbank-v1 (job 12994) | **M1 cache v1 (jobs 13003/13004)** |
|---|---|---|
| escaping path | **OUTPUT / handoff** temp file | **INPUT** media file |
| escape mechanism | `mktemp "${TMPDIR:-/tmp}/…"` → ambient `$TMPDIR` out-of-repo; producer's guarded `read_json` of it raises | `data/video/{ds}/All/*.mp4` is an in-repo symlink whose **resolved target** is out-of-repo; the guard's `.resolve()` raises |
| blind-spot class | ambient-env-derived output path | **input symlink topology** |
| detection surface missed | handoff table modeled writer→reader for artifacts, not `$TMPDIR` | handoff table + code review modeled input reads as "guarded ⇒ safe," never modeled that a symlinked input makes the guard *fire* |

This is best labeled **the same defect genus (repo-root containment guard applied to a path that
legitimately lives partly outside the repo), second species (input symlink) after the first
species (ambient output temp).** I record it as such; the campaign-wide "#5/#6" tally is the team
lead's ledger and outside what I can verify from the two logs — the *mechanism* and *genus* are
what I certify.

Corroboration that this was a latent, reviewed-past defect, not a fresh regression:
- `M1_CACHE_CODE_REVIEW.md` M-3 (=freeze R-2) **kept** the byte-hash, calling content-addressing
  "the safer default for a correct U_D," and flagged only the *extra IO* as non-blocking — it never
  considered that resolving a symlinked-external mp4 would crash the guard.
- `M1_CACHE_CODE_REVIEW.md` §5 handoff table certified "all reads through `canonical_root_path`;
  **0 out-of-repo paths**" — reading the guard as a *containment guarantee* on inputs, when for a
  symlinked input the guard *raises* instead of guaranteeing.
- `M1_CACHE_CODE_REVIEW.md` Row 17 reasoned the video-read failure mode is "unreadable video →
  `(None,False)` → text-only → canonical unresolved (fail-open)" — modeling a failure surface
  (`load_video_frames`) that execution never reaches, because it dies earlier and **fail-closed**
  at the guard.

---

## 3. Fix verdict — independent evaluation of the two proposals

I evaluate the meeting's preferred **(A) remove-video-byte-hash** against **(B) symlink-whitelist
(guard-relaxation)**, and reject the framing of (A) as stated.

### 3.1 Three factual corrections to proposal (A) as written

1. **"mp4 全程不打开 (mp4 never opened)" is infeasible.** The producer must decode **16** uniform
   frames from each mp4 (`load_video_frames(str(video_fs), NUM_FRAMES=16)`, producer:182). The
   only pre-extracted frames on disk are `data/lora_frames/{ds}/{vid}/frame_*.jpg` = **8 frames
   per video, not 16** (verified), and the producer never references `lora_frames` (grep = 0
   hits). So the mp4 **must** be opened by the producer regardless of any hashing decision. There
   is no code path in which the mp4 goes un-opened.
2. **"爆炸半径仅 builder 实体 (blast radius = builder only)" is wrong.** The mp4 path is routed
   through the guard at **two entities**: builder (`evidence_pack_v1.py:183` and `:185`
   `note_video_read`) *and* producer (`producer_v1.py:178` and `:181` `note_video_read`, then
   `:182` decode). Removing the builder's hash leaves the producer's `canonical_root_path(
   pack["video_relpath"])` at line 178 to crash on the identical symlink. The producer entity must
   change too.
3. **Byte-hash removal does not remove the escape.** The crash is the guard's `.resolve()`
   (§1.2), not the hash `open()`. Dropping the hash while still routing the mp4 through
   `canonical_root_path` still escapes.

Additionally, "帧字节 (frame bytes) content-address" is internally inconsistent with "mp4 never
opened": frame bytes require decoding the mp4. Moving the content address onto decoded frames also
migrates `evidence_pack_sha256` from a Stage-1 (label-blind, no-GPU) *spec* hash to a Stage-2
*post-decode* hash — a schema/semantics change that touches the frozen cert/replica field list and
`input_builder_hash`, which **enlarges** the blast radius rather than shrinking it. (Note: dedup is
already a no-op — `evidence_pack_sha256` includes `video_id`, so `unique_pack_count == video_count`
always; `video_sha256` currently buys change-detection provenance, not dedup.)

### 3.2 Verdict: adopt (B) — symlink-tolerant containment on the video path, BOTH sites

The mp4-symlink-to-external-corpus is the **designed** data layout for all three datasets, not an
anomaly. The bug is that a repo-**containment** guard (correct for outputs and in-repo evidence
files) is mis-applied to an input that legitimately lives partly outside the repo. The fix is to
stop resolving the mp4 symlink *through the containment guard*, while preserving fail-closed
isolation:

- Introduce a `canonical_video_path(rel, dataset)` (or fix `note_video_read` + the two call
  sites) that enforces containment on the **symlink LOCATION** — the relative path must be under
  the allowlisted `data/video/{ds}/All/`, contain no `..` traversal, and lie inside the repo tree
  as a path string — and returns the OS path for decoding **without** requiring the resolved
  target to be under repo root. The decoder then follows the symlink at OS level (exactly as the
  smoke empirically proved works, 10/10).
- Apply it at **both** `evidence_pack_v1.py:183/185` and `producer_v1.py:178/181`.
- This preserves the isolation contract: the allowlist + `FORBIDDEN_TOKENS` still run on the
  relative path, so `forbidden_path_read_count` / `non_allowlisted_train_content_read_count` stay
  0; no val/test/label path becomes reachable. There is **no isolation regression** — reading the
  train mp4's bytes for a 16-frame decode is exactly the authorized train-video evidence read.

**Byte-hash removal is orthogonal and optional.** It is a legitimate R-2 IO win (drops 790+806
full-file reads in the builder) and would let the builder stop touching the mp4 at all. If the
team wants it, do it as a *separate* decision, and be explicit that dropping `video_sha256` from
the spec **changes `evidence_pack_sha256`** and therefore requires re-freezing the content-address
semantics. It is **neither necessary nor sufficient** for the escape fix. Recommended default:
**keep the fix minimal (guard only, both sites); leave the byte-hash decision to the meeting**,
noting that even if the hash is removed, the producer's symlink-tolerant guard is still required.

Ranked recommendation: **(B) guard fix, both sites — REQUIRED.** Byte-hash removal — **OPTIONAL,
separate, with its own address re-freeze if the field is dropped.**

---

## 4. v2 opening conditions (ordered, each mandatory)

1. **Small plan amendment** (`M1_CACHE_PLAN_AMENDMENT` v2): rename `runs[4]/[5]` `run_id`
   `…-MHC-v1 → …-MHC-v2`, `…-MHC_zh-v1 → …-MHC_zh-v2` (REPLACE the spent v1 ids in both `runs[i]`
   and `run_order[i]`); update `runs[6]` `…-SEAL-v1` `dependencies` to the two v2 ids in lock-step;
   annotate the `video_sha256` / frame-hash semantics if the byte-hash decision (§3.2) is taken.
   Re-hash the machine plan; the producer's `verify_machine_cache` pins index+run_id and will
   assert the rename.
2. **Builder + producer fix** per §3.2: symlink-tolerant `canonical_video_path` at all four sites
   (`evidence_pack_v1.py:183/185`, `producer_v1.py:178/181`), preserving allowlist/forbidden-token
   isolation. (Plus the optional byte-hash change if elected.)
3. **Re-freeze** the touched frozen entities (new sha256 for `_m1_evidence_pack_v1.py`,
   `_m1_cache_v1_common.py`, and — if changed — the producer), refresh `M1_CACHE_FREEZE.md` and
   `input_builder_hash`/`common_sha256` provenance.
4. **Delta review** with a **NEW MANDATORY item, added to the simulation-table institution:**
   *for every target dataset, `readlink`-verify the topology of every input root* (video, frames,
   gt, ASR) and record which resolve outside the repo, then assert the guard tolerates exactly the
   video-symlink escape and nothing else. The §1.3 table is the reference the v2 delta review must
   reproduce and extend. **The re-freeze must also add a preflight that exercises the actual frozen
   `build_dataset_packs` on real symlinked mp4** (e.g. a tiny in-process call), not a
   re-implementation — see §5.
5. **Execution authorization** (exact-hashes / no-clobber gate) as before.
6. **One re-submit each** for `runs[4]` (MHC-v2) and `runs[5]` (MHC_zh-v2) under single-submit
   discipline; `runs[6]` seal follows on their outputs.

---

## 5. Institutional lessons

1. **A smoke must exercise the ACTUAL frozen code path, not a re-implementation of it.** The v1
   smoke re-rolled the video-loading loop with a raw `ROOT / … / f"{vid}.mp4"` join and imported
   only the leaf helpers, so the one function that broke (`build_dataset_packs` →
   `canonical_root_path` on the mp4) was never run by any preflight on any dataset. "Smoke must
   cover the target dataset's input topology" (HateMM ≠ MHC reality) is *also* true, but secondary:
   HateMM mp4 escape too, so a smoke of the *real* path would have caught it even on the
   non-contract dataset. The primary rule is **exercise the frozen entities themselves.**
2. **Input symlink topology is the third blind-spot class of the handoff-table institution** —
   after (i) environment/package availability and (ii) ambient-env output/handoff paths
   (`$TMPDIR`, realbank-v1). The handoff/simulation tables modeled the pipeline's *writes* and
   in-repo *artifact* reads, and treated every input read as "guarded by `canonical_root_path` ⇒
   contained." For a symlinked input the guard does not contain — it **raises**. The institution
   must now model, per dataset, the readlink topology of every *input* root and assert the guard's
   behavior on each (contain vs. tolerate-escape vs. fire).
3. **Distinguish the guard's fire-surface from the operation's fail-surface.** The code review's
   Row-17 reasoning ("bad video → fail-open to unresolved") attached the video-failure model to
   `load_video_frames`, but the real first-contact failure is fail-closed at the containment guard,
   strictly upstream. Failure-mode analysis must locate the *earliest* path-touching call, not the
   most semantically obvious one.

---

### Bottom line

Judgment: **v1 lineage of runs[4]/[5] CLOSED as an infrastructure preflight burn (input-symlink
escape of the repo-root containment guard); zero scientific information; fail-closed hygiene clean
(no artifacts, no seal contact); runs[6] seal budget untouched.** Fix: **reject "remove byte-hash /
mp4 never opened / builder-only" as stated (three factual errors); adopt symlink-tolerant
containment on the video path at BOTH the builder and producer sites; treat byte-hash removal as a
separate optional IO decision with its own address re-freeze.** Reopen via the §4 six-step v2 gate,
whose delta review MUST add the per-dataset per-input-root readlink topology check and a preflight
that runs the real frozen `build_dataset_packs` on symlinked mp4.
