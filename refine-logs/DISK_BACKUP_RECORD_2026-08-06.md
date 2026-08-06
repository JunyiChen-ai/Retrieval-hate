# Disk Backup Record — 2026-08-06 (B2 gap backup)

**STATUS: COMPLETE — 23.1 GB / ~96k files pushed, all four legs SHA1-verified 0 differences
(SLURM job 14146, COMPLETED 01:31:10, ExitCode 0:0). Nothing deleted locally. One item left for
the user: 40.8 GB of raw video (Multihateclip + HateMM + HateClipSeg) is NOT on B2 — see §5.**

Follows the convention of `refine-logs/DISK_BACKUP_RECORD_2026-07-14.md` (same remote, same
tool, same push→verify contract). **This operation is backup-only — nothing was deleted
locally.** Quota at start: **287G / 290G soft** (no grace running).

User order: 检查 B2 上是否有相关数据,如果没有就推上去 — check whether B2 holds the relevant
data; push up whatever is missing.

## Credentials / tool

- `rclone v1.70.3` at `/data/jehc223/home/.local/bin/rclone`.
- Config `/data/jehc223/home/.config/rclone/rclone.conf`, remote `[b2]`, account
  `005780f1d135e750000000001`, bucket **`junyi-data`**.
- **Credentials verified working** — `rclone lsd b2:` returned the bucket, all list/copy/check
  operations below succeeded. No re-auth needed.

## 1. What was ALREADY on B2 (survey, 2026-08-06)

Bucket `junyi-data` top level: `AlphaSteer/`, `ImpliHateVid/`, `NIPS2026/`, `RGCL_video/`,
`harmful meme/`, `hate-followup/`, plus four loose zips (`FVC.zip`, `FakeSV.zip`, `HVideo.zip`,
`HVideo_datasets.zip`).

| B2 prefix | Objects | Size | Notes |
|---|---|---|---|
| `RGCL_video/logs` | 33,955 | **1.020 TiB** | disk_guard checkpoint mirror (nested layout, see DISK_POLICY §6) |
| `RGCL_video/manual_backup_2026-07-14` | 3,255 | 143.878 GiB | the July backup: `lora_p9` (978 f) + `Retrieval` (2277 f) |
| `RGCL_video/adapters` | 288 | 18.325 GiB | `lora_p9` adapters |
| `RGCL_video/embeddings` | 4,008 | 1.977 GiB | CLIP/Qwen feature caches (canonical mirror) |
| `RGCL_video/CLIP_Embedding` | 12 | 14.131 MiB | legacy prefix |
| `RGCL_video/archives` | 12 | 3.737 MiB | MHC / MHC_zh archives |
| `RGCL_video/slurm_logs` | 26 | 2.896 MiB | |
| `RGCL_video/research-wiki` | 6 | 66.987 KiB | |
| `RGCL_video/scripts` | 2 | 11.014 KiB | |
| `ImpliHateVid/` | 2,012 | **50.128 GiB** | **raw video** (Explicit/Implicit/Non Hate Videos) |
| `hate-followup/` | 4 | 8.181 GiB | `{hatemm,mhclip_en,mhclip_zh}_processed.tar`, `ihv_killtest_payload.tar` |

The July manifest reconciles exactly: `manual_backup_2026-07-14/{lora_p9,Retrieval}` still holds
978 + 2277 = 3255 objects, matching the 2026-07-14 record. **The July restore path is intact.**

## 2. The gap (measured, not assumed)

Verified with `rclone check <local> <b2 path> --one-way`:

| Local asset | Size | Files | On B2 before? |
|---|---|---|---|
| `data/CLIP_Embedding` | 2.2G | 4,042 | **4,008 matched, 34 MISSING** |
| `logging/lora` | 7,036,173,444 B (6.6G) | 549 | **0 matched, 549 MISSING** — B2 `logs/lora` held only the 54 `optimizer.pt`/`scheduler.pt` objects (9.035 GiB) that disk_guard had pushed; every `adapter_model.safetensors` + tokenizer/config file was local-only |
| `artifacts/` | 5.8G | 10,678 | **no B2 copy at all** |
| `data/` (non-embedding) | 3,024,028,028 B | ~25.5k | **no B2 copy at all** |
| `refine-logs/` | 713M | 1,007 | **no B2 copy** (incl. `router_ckpt_snapshot` 355M + `b5_ckpt_snapshot` 336M — the head checkpoints the July record cites as B5's safekeep) |
| `slurm/`, `scripts/`, `external/`, `src/`, `.git` | ~530M | ~14k | **no B2 copy** |
| `/data/jehc223/baselines/MoRE` | 5,435,867,686 B (5.2G) | 43,251 | **no B2 copy** — baseline derived features (2.8G) + 6 trained `*_best.pth` (2.5G) |

The 34 missing feature caches are the campaign's recent expensive derivations:

```
HateClipSeg/test_seen_subclip{K30,K4}_openai_clip-vit-large-patch14-336_HF.pt
HateMM/  {train,dev_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF-c02den-{NAT,RFULL,RW1..RW4}.pt
HateMM/  {train,dev_seen}_Qwen2.5-VL-7B-Instruct-LoRA-curric-nullop2merge_HF.pt
HateMM/  SPANSTATS_HateMM_{train,dev_seen}_...-curric-bidir-textpool_HF.json
MHC_zh/  {train,dev_seen}_Qwen2.5-VL-7B-Instruct-LoRA_HF-c02den-{NAT,RFULL,RW1..RW4}.pt
MHC_zh/  {train,dev_seen}_Qwen2.5-VL-7B-Instruct-LoRA-nullop2merge_HF.pt
MHC_zh/  SPANSTATS_MHC_zh_{train,dev_seen}_...-bidir-textpool_HF.json
```
(192,189,506 B total.)

**OCR banked outputs:** searched — none exist under `RGCL/`. The only OCR artefacts on disk are
in other projects (`ExMRD*`, `baselines/MoRE/preprocess`) and `~/.EasyOCR` / `~/.paddleocr`
model caches. `refine-logs/OCR_FORENSIC_RECON.md` is a document, not data. Nothing to back up
(consistent with the standing user veto on the OCR channel).

## 3. What was uploaded — SLURM job 14146

`scripts/slurm/b2_backup_manual_2026-08-06.sbatch` (CPU-only, no `--gres`, **no `--time`**),
submitted 2026-08-06, `PENDING (JobHeldUser)` → auto-released → ran on `foscsmlprd01`,
**`COMPLETED` in 01:31:10, ExitCode 0:0**.
Per leg: `rclone copy --transfers 8 --checkers 16 --b2-chunk-size 96M`, then
`rclone check <src> <dest> --one-way` (**SHA1 checksum comparison**, rclone's B2 default —
not size/mtime).

| Leg | Source | B2 destination | Bytes | Files |
|---|---|---|---|---|
| `embeddings` | `data/CLIP_Embedding` | `RGCL_video/embeddings` | 192,189,506 (34 new) | 4,042 checked |
| `lora` | `logging/lora` | `RGCL_video/logs/lora` | 7,036,173,444 | 549 |
| `repo` | `/data/jehc223/RGCL` | `RGCL_video/manual_backup_2026-08-06/RGCL` | 10,447,776,708 | 52,289 |
| `baselines_MoRE` | `/data/jehc223/baselines/MoRE` | `RGCL_video/manual_backup_2026-08-06/baselines_MoRE` | 5,435,867,686 | 43,251 |

**Total pushed ≈ 23.1 GB / ~96k files** (measured wall clock 1h31m ≈ 4.2 MB/s effective — the
cost was per-object round-trips, not bandwidth; ~18 files/s at `--transfers 8`).

**Layout rationale.** Legs 1–2 go to the **canonical disk_guard mirror paths** so
`scripts/b2_pull.sh` and `disk_guard.sh` keep resolving them (both are purely additive — no
local file collided with an existing B2 object, so nothing was overwritten). Legs 3–4 go to a
dated `manual_backup_2026-08-06/` prefix, matching the July convention. The `repo` leg
therefore **excludes** `logging/lora/**` and `data/CLIP_Embedding/**` (already covered by legs
1–2, no point paying to store them twice) and `data/_src_Multihateclip/**` (a symlink to the
27G raw `Multihateclip` tree — see §5).

### Verification results — ALL FOUR LEGS CLEAN

Job 14146 `COMPLETED`, elapsed **01:31:10**, ExitCode `0:0`. Final `rclone check --one-way`
(SHA1) per leg:

| Leg | Matching files | Differences | Check elapsed |
|---|---|---|---|
| `embeddings` | **4,042 / 4,042** | **0** | 4.5s |
| `lora` | **549 / 549** | **0** | 4.2s |
| `repo` | **52,329 / 52,329** | **0** (after top-up, see below) | 2m34s + 2m |
| `baselines_MoRE` | **43,251 / 43,251** | **0** | 1m36s |

Resulting B2 state:

```
RGCL_video/manual_backup_2026-08-06/RGCL           52,329 obj    9.729 GiB (10,445,998,189 B)
RGCL_video/manual_backup_2026-08-06/baselines_MoRE 43,251 obj    5.052 GiB ( 5,424,714,772 B)
RGCL_video/logs/lora                                  603 obj   15.588 GiB (was 54 obj / 9.035 GiB)
RGCL_video/embeddings                               4,042 obj    2.156 GiB (was 4,008 / 1.977 GiB)
```

**Self-referential log gap (found, closed).** The `repo` leg's first pass returned
`copy_exit=1` / `check_exit=1` with exactly **3** differences — all of them this job's *own*
live log files, which sit inside the source tree and were being appended to while being copied:

```
slurm/logs/b2_backup_0806_14146_repo.rclone.log   corrupted on transfer: sha1 hashes differ
slurm/logs/b2_backup_0806_14146_repo.check.log    file not in B2
slurm/logs/b2backup0806_14146.out                 sizes differ
```

No project data was involved; 52,324 of 52,325 files matched on the first pass. After the job
ended (logs now static) a top-up `rclone copy` of `slurm/logs` and `refine-logs` was run, then
the **full repo tree was re-checked end to end: 52,329 matching files, 0 differences.**

*Lesson for the next run of this script:* add
`--exclude "slurm/logs/b2_backup_*"` to the `repo` leg, or write the transfer logs outside the
tree being copied. A tree that contains its own transfer log can never self-verify in one pass.

### Symlinks — skipped by rclone, manifest captured

rclone does not follow symlinks without `-L/--copy-links`, so the **5,111 symlinks** in the repo
were listed-and-skipped rather than copied. This loses no data, but it does lose the alias
structure, so the full link→target map is recorded at
`refine-logs/SYMLINK_MANIFEST_2026-08-06.tsv` (also inside the `repo` leg on B2).

- **5,098** of them are the `data/video/{ImpliHateVid,HateMM,MHC,MHC_zh,HateClipSeg,MHCsmoke}`
  video-path alias farm. **2,009 of these are ALREADY DANGLING** — the ImpliHateVid ones point
  into `/data/jehc223/home/tmp/tmp.28In8lDvPs/`, a long-deleted temporary rclone mount of the
  B2 `ImpliHateVid/` bucket prefix. They were broken before this backup and are recorded as
  such in the manifest.
- The **11 non-video symlinks all point at files that ARE in this backup**, so restoring them is
  a pure `ln -s` replay:
  - `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_c1settle_hatemm_s0.pt`
    → `{train,dev_seen}_p9c3_hatemm_s0.pt` (note: `test_seen_c1settle` and `dev_seen_c1settle`
    are both aliases of the *same* `dev_seen_p9c3` file)
  - `artifacts/c06_falsifier/fidelity_view/mint_{hatemm,zh}_s{0,1,2}_ffull.npz`
    → `artifacts/c06_falsifier/mints/mint_{hatemm,zh}_N_s{0,1,2}_ffull.npz`
  - `data/_src_Multihateclip` → `/data/jehc223/Multihateclip` (raw video, §5)
  - `.aris/tools` → `/data/jehc223/Auto-claude-code-research-in-sleep/tools`

Replay after restore with:

```bash
awk -F'\t' 'NR>1 && $3=="ok" {print $1"\t"$2}' \
    /data/jehc223/RGCL/refine-logs/SYMLINK_MANIFEST_2026-08-06.tsv \
  | while IFS=$'\t' read -r l t; do mkdir -p "$(dirname "$l")"; ln -sfn "$t" "$l"; done
```

## 4. RESTORE — exact copy-paste commands

Run from a compute node / sbatch (the login node reaps sustained transfers).

```bash
RCLONE=/data/jehc223/home/.local/bin/rclone
BASE=b2:junyi-data/RGCL_video

# (1) feature caches  -> data/CLIP_Embedding   (canonical mirror, 4042 files / 2.2G)
$RCLONE copy $BASE/embeddings /data/jehc223/RGCL/data/CLIP_Embedding \
    --transfers 8 --checkers 16 --b2-chunk-size 96M

# (2) LoRA adapters   -> logging/lora          (549 files / 6.6G; B2 also holds the 54
#     optimizer.pt/scheduler.pt objects that were pruned locally — restoring them is optional
#     and only needed to RESUME training, not to run inference)
$RCLONE copy $BASE/logs/lora /data/jehc223/RGCL/logging/lora \
    --transfers 8 --checkers 16 --b2-chunk-size 96M \
    --exclude "**/optimizer.pt" --exclude "**/scheduler.pt"

# (3) repo snapshot (artifacts/, data/ non-embedding, refine-logs ckpt snapshots, slurm/logs,
#     scripts, src, external, .git) -> 10.4G / 52289 files
$RCLONE copy $BASE/manual_backup_2026-08-06/RGCL /data/jehc223/RGCL \
    --transfers 8 --checkers 16 --b2-chunk-size 96M

# (4) MoRE baseline (derived features + 6 trained *_best.pth) -> 5.2G / 43251 files
$RCLONE copy $BASE/manual_backup_2026-08-06/baselines_MoRE /data/jehc223/baselines/MoRE \
    --transfers 8 --checkers 16 --b2-chunk-size 96M

# verify any of the above after restore (SHA1, one-way):
$RCLONE check /data/jehc223/RGCL/logging/lora $BASE/logs/lora --one-way \
    --exclude "**/optimizer.pt" --exclude "**/scheduler.pt"
```

Still valid from the July record (unchanged, re-verified present this session):

```bash
$RCLONE copy $BASE/manual_backup_2026-07-14/lora_p9   /data/jehc223/RGCL/logging/lora_p9   ...
$RCLONE copy $BASE/manual_backup_2026-07-14/Retrieval /data/jehc223/RGCL/logging/Retrieval ...
```

## 5. NOT uploaded — raw video, needs a user decision

Raw video was surveyed by listing only (no re-upload attempted). **Result: only ImpliHateVid is
on B2; Multihateclip and HateMM raw are not, and never were** — DISK_POLICY §5 items **P2/P3**
are still marked "None have been done", which the bucket listing confirms.

| Dataset | Local | Files | On B2? |
|---|---|---|---|
| `/data/jehc223/Multihateclip` (En + Zh) | **27G** | 17,243 | **NO** |
| `/data/jehc223/HateMM` (video + frames + quad) | **9.6G** | 44,299 | **NO** |
| `/data/jehc223/HateClipSeg/videos` | **4.2G** | 439 | **NO** |
| `/data/jehc223/ImpliHateVid` | 3.7M (annotations only) | 12 | **YES — 50.128 GiB of raw video, B2-ONLY** |

**Total un-backed-up raw video: 40.8 GB / ~62k files.** Time estimate using **this session's
own measured rates** (not extrapolated): ~18 objects/s at `--transfers 8` for small files, and
~89 MB/s for large files (July job, 143.878 GiB in 27m29s).

| Item | Size | Files | Estimated wall clock | Binding cost |
|---|---|---|---|---|
| `HateMM` | 9.6G | 44,299 | **~40 min** | object count (mostly extracted frames) |
| `Multihateclip` | 27G | 17,243 | **~20 min** | mixed; 27G of video ≈ 5 min of bytes |
| `HateClipSeg/videos` | 4.2G | 439 | **~2 min** | bytes only |
| **all three** | **40.8G** | **61,981** | **~1 h** | |

Storage cost at B2's $6/TB-month is ~$0.25/month for all three.

Not launched, per the standing instruction to report anything in this size class first. Two
things make it a genuine decision rather than a formality:

1. `hate-followup/` on B2 already holds `hatemm_processed.tar` (1.24 GB),
   `mhclip_en_processed.tar` (2.47 GB) and `mhclip_zh_processed.tar` (3.48 GB) — *processed*
   derivatives, **not** substitutes for raw video, but they may already cover the actual
   re-derivation need.
2. HateMM's 44,299 files are mostly extracted **frames** (2.4G), which are re-derivable from the
   6.2G of video. Uploading video-only would cut the object count by roughly an order of
   magnitude at little loss.

**Recommended if approved:** `HateMM` video-only + `HateClipSeg/videos` first (small, and
HateClipSeg holds a frozen-unconsumed test split), then `Multihateclip`.

**Asymmetry worth flagging:** ImpliHateVid's raw video exists **only on B2** — there is no local
copy (3.7M of annotations is all that is on disk). For that dataset B2 is not a backup, it is
the sole copy.

## 6. Also not uploaded (deliberate)

- `/data/jehc223/models` (17G) and `~/.cache/huggingface` (6.3G) — public HF weights,
  re-downloadable; DISK_POLICY §3 already rules them out of scope.
- `/data/jehc223/{AlphaSteer,SafetyContradiction,ExMRD_ours,NIPS2026,...}` — other projects,
  DO-NOT-TOUCH per DISK_POLICY §2. (`AlphaSteer` and `NIPS2026` already have their own B2
  prefixes.)
- `external/baselines/*` third-party git clones are inside the `repo` leg (102M) — cheap enough
  that excluding them was not worth the restore ambiguity.

## 7. Git

The `.git` directory (130M, 5,455 objects) **is** included in the `repo` leg, so the full commit
history has an off-machine copy independent of any git remote.

> **Corrected 2026-08-07.** This section originally warned that `origin` points at the upstream
> `https://github.com/JingbiaoMei/RGCL` and that a push of this campaign's history therefore had
> nowhere to land. That is no longer true: a second remote `project` →
> `https://github.com/JunyiChen-ai/Retrieval-hate.git` now exists (see commit f07cc87), so the
> history does have a real destination. The B2 `.git` copy stays useful as an independent
> off-machine copy, but it is belt-and-braces rather than the only route.
A separate `git bundle` was therefore not created (it would duplicate `.git` for no added
recoverability). **This record was not committed by this agent** — git is handled elsewhere.

---
_SLURM job 14146 COMPLETED 2026-08-06 (01:31:10, ExitCode 0:0). Four legs, 23.1 GB / ~96k files
pushed to `b2:junyi-data`; every leg re-verified by SHA1 at 0 differences (4,042 + 549 + 52,329
+ 43,251 files). **Backup only — nothing was deleted locally; quota unchanged at 287G/290G.**
The one remaining gap is raw video (§5): Multihateclip 27G, HateMM 9.6G, HateClipSeg 4.2G —
never mirrored, ~1 h to push, awaiting a user decision. Note that ImpliHateVid's raw video is
B2-ONLY (no local copy), so for that dataset the bucket is the sole copy, not a backup._

---

# 2026-08-07 — RAW VIDEO uploaded (closes §5, and DISK_POLICY §5 P2/P3)

**STATUS: COMPLETE — 29.920 GiB / 3,938 files pushed, all three datasets SHA1-verified at
0 differences (SLURM job 14238, COMPLETED 00:09:36, ExitCode 0:0). Nothing deleted locally.**

User ruling: 把这几个数据集上面的原始视频都上传到B2 — upload the raw videos of these datasets
to B2. This supersedes the "awaiting a user decision" state that §5 above was left in.

## Destination — the prefix DISK_POLICY §5 already promised

```
b2:junyi-data/RGCL_video/raw/{Multihateclip,HateMM,HateClipSeg}
```

Chosen so DISK_POLICY §5's own restore line (`scripts/b2_pull.sh raw/Multihateclip
/data/jehc223/Multihateclip`) becomes literally true rather than aspirational.
`RGCL_video/raw/` was **verified empty before the run** — purely additive, nothing overwritten.

## What was uploaded

Job `scripts/slurm/b2_backup_raw_video_2026-08-07.sbatch` (CPU-only, no `--gres`, **no
`--time`**), `PENDING (JobHeldUser)` → auto-released → `COMPLETED` on `foscsmlprd01`.
`rclone copy --transfers 8 --checkers 16 --b2-chunk-size 96M`, then
`rclone check --one-way` (**SHA1**) with the identical filter set.

| Dataset | B2 prefix | Size | Objects | Copy time | Verify |
|---|---|---|---|---|---|
| Multihateclip | `RGCL_video/raw/Multihateclip` | 19.638 GiB (21,086,603,512 B) | 2,410 | 6m14s | **2,410 matching / 0 differences** |
| HateMM | `RGCL_video/raw/HateMM` | 6.121 GiB (6,572,419,912 B) | 1,089 | 1m51s | **1,089 matching / 0 differences** |
| HateClipSeg | `RGCL_video/raw/HateClipSeg` | 4.160 GiB (4,467,115,855 B) | 439 | 1m08s | **439 matching / 0 differences** |
| **total** | `RGCL_video/raw` | **29.920 GiB (32,126,139,279 B)** | **3,938** | **9m36s** | **0 errors in all three copy logs** |

Video payload by directory:

| Directory | Bytes | Files | Kind |
|---|---|---|---|
| `Multihateclip/English/video_mp4` | 7,377,606,298 | 792 | mp4 (transcoded) |
| `Multihateclip/English/video` | 5,162,273,843 | 792 | 772 webm + 20 mp4 (as-downloaded originals) |
| `Multihateclip/Chinese/video` | 8,545,430,223 | 814 | mp4 |
| `HateMM/video` | 6,570,593,842 | 1,083 | mp4 |
| `HateClipSeg/videos` | 4,452,757,045 | 395 | 364 mp4 + 26 webm + 5 mkv |

English ships **both** the as-downloaded `video/` (webm) and the transcoded `video_mp4/`;
both were uploaded, since `video/` is the true original and `video_mp4/` is what the pipeline
reads. Alongside the video, the small metadata the loaders actually open was included so each
mirror restores standalone: `annotation(new).json` + `splits/` for Multihateclip (per language)
and HateMM, and HateClipSeg's `Dataset/`, `Images/`, `pilot/`, `lexicons.json`, `README.md`
(8.5M total — `Dataset/` carries the frozen-unconsumed split).

**Estimate vs actual — my §5 estimate was wrong, conservatively.** §5 projected ~1 h by
assuming per-object round-trips would bind, as they did for the 96k small files on 2026-08-06
(~18 objects/s). For raw video the binding constraint is bandwidth, not object count: 3,938
large files moved at **~56–66 MB/s**, so the real cost was **9m36s**. Rule of thumb going
forward: object-count-bound for small files, bandwidth-bound above roughly 1 MB/file.

## What was EXCLUDED, and why it is safe

Per the ruling ("skip HateMM's re-derivable frame files"), the derived image/audio trees were
left off. **All of them remain untouched on local disk** — this was an upload, not a move.

| Excluded | Size | Files | Re-derivation |
|---|---|---|---|
| `HateMM/frames/` | 2.4G | 34,567 jpg | decode from `HateMM/video/` |
| `HateMM/quad/` | 1.1G | 8,643 jpg | 4-frame grids from `HateMM/video/` |
| `Multihateclip/{English,Chinese}/quad/` | 5.6G | 13,229 jpg | 4-frame grids from `*/video/` |
| `Multihateclip/{English,Chinese}/audios/` | 1.6G | 1,604 wav | `ffmpeg` audio track from `*/video/` |

**Every excluded byte is re-derivable from a byte that IS in this backup**, and none of these
directories has a consumer in this repo — verified by grep over `src/`, `scripts/`, `configs/`
(the only `frames` hits are the unrelated `data/CLIP_Embedding/HateMM/frameset_qwen7b_8f/`
caches and dict keys, not these paths). The live pipeline never reads them: it decodes frames
straight from the video at run time via decord —
`src/utils/generate_video_archive_HF.py:319 _decode_with_decord(video_path, num_frames)`,
default `--num_frames 8` — and the audio route re-extracts from video in
`scripts/analysis/apx_extract_egemaps.py` and `scripts/analysis/laud_extract_whisper.py`.
The paths `src/` actually opens are exactly the ones backed up: `<ds>/video`, `splits/`,
`annotation(new).json`, `HateClipSeg/lexicons.json`.

## RESTORE — exact copy-paste commands

```bash
RCLONE=/data/jehc223/home/.local/bin/rclone
BASE=b2:junyi-data/RGCL_video/raw

$RCLONE copy $BASE/Multihateclip /data/jehc223/Multihateclip \
    --transfers 8 --checkers 16 --b2-chunk-size 96M
$RCLONE copy $BASE/HateMM       /data/jehc223/HateMM \
    --transfers 8 --checkers 16 --b2-chunk-size 96M
$RCLONE copy $BASE/HateClipSeg  /data/jehc223/HateClipSeg \
    --transfers 8 --checkers 16 --b2-chunk-size 96M

# equivalent via the repo helper (this is the DISK_POLICY §5 line, now real):
scripts/b2_pull.sh raw/Multihateclip /data/jehc223/Multihateclip
scripts/b2_pull.sh raw/HateMM        /data/jehc223/HateMM
scripts/b2_pull.sh raw/HateClipSeg   /data/jehc223/HateClipSeg

# verify after restore (one-way, SHA1) -- filters MUST match the upload:
$RCLONE check /data/jehc223/Multihateclip $BASE/Multihateclip --one-way \
    --exclude "*/quad/**" --exclude "*/audios/**"
$RCLONE check /data/jehc223/HateMM        $BASE/HateMM --one-way \
    --exclude "frames/**" --exclude "quad/**"
$RCLONE check /data/jehc223/HateClipSeg   $BASE/HateClipSeg --one-way
```

A restore yields video + annotations but **not** `frames/`, `quad/`, `audios/` — regenerate
those from the restored video if some future route needs them. Nothing in the current codebase
does.

## Process notes

- **Transfer logs were written outside the copied trees** (`RGCL/slurm/logs/b2_rawvid_14238_*`,
  while the sources are `/data/jehc223/{Multihateclip,HateMM,HateClipSeg}`), applying the
  forward-fix from §3 above. Result: all three legs verified clean on the **first** pass, with
  none of the self-referential log churn that cost the 2026-08-06 `repo` leg a second pass.
- **`scripts/disk_guard.sh` was deliberately NOT run.** Usage sat at 287G against the guard's
  250G default threshold, so invoking it per DISK_POLICY §4 would have triggered its
  push-verify-**prune** path and deleted local checkpoints. This job is backup-only and the
  ruling says nothing is deleted locally, so the guard was omitted on purpose — noting it here
  because §4 otherwise reads as "every sbatch runs the guard".
- Quota is **unchanged at 288G/290G**; all three datasets still show their full local file
  counts (17,243 / 44,299 / 439) and every excluded directory is still present on disk.
- Zero symlinks in all three trees, so the rclone symlink caveat from §3 does not apply here.

## Remaining state

`RGCL_video/raw/` now holds Multihateclip, HateMM and HateClipSeg. Combined with the
pre-existing `ImpliHateVid/` prefix (50.128 GiB, B2-only), **all four raw video corpora used by
this project now have an off-machine copy.** The §5 asymmetry still stands and still matters:
ImpliHateVid has no local copy, so for that one dataset B2 is the sole copy, not a backup.

_SLURM job 14238 COMPLETED 2026-08-07 (00:09:36, ExitCode 0:0). 29.920 GiB / 3,938 objects to
`b2:junyi-data/RGCL_video/raw/`; every leg SHA1-verified at 0 differences, 0 errors across all
copy logs. Backup only — nothing deleted locally, nothing overwritten. DISK_POLICY §5 P2/P3
updated from "None have been done" to DONE in the same commit as this record._
