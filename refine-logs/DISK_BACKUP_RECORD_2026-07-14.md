# Disk Backup-then-Delete Record — 2026-07-14

Agent: disk-backup executor. Scope: back up two closed-route dirs to B2 cloud, verify, then
delete locally to relieve quota. Companion audit: `refine-logs/DISK_AUDIT_2026-07-14.md`.

## User ruling (verbatim intent)

> Back up BOTH `logging/lora_p9` (83G) and `logging/Retrieval` (61G) to the B2 cloud remote
> first, verify, THEN delete locally, and document thoroughly so anything needed later can be
> found and restored.

Quota context at ruling time: ~396G vs 290G soft limit, grace counting; this operation
targets ~252G (396 − 144) after both deletions.

## Quota — BEFORE (2026-07-14, via `quota -s`)

```
Disk quotas for user jehc223 (uid 135258174):
 /dev/mapper/data-data   396G*   quota 290G   limit 3000G   grace 13:00 (running)   files 1212k
```

(~106G over the 290G soft quota. `lfs` binary is no longer on PATH this session; `quota -s`
reports the identical project/user quota the audit captured as "lfs quota".)

## Pre-manifests (source, login-node metadata)

| Dir | du -sh | Bytes (`du -sb`) | Files (`find -type f`) |
|-----|--------|------------------|------------------------|
| logging/lora_p9   | 83G | 89,108,371,110 | 978  |
| logging/Retrieval | 61G | 65,379,640,211 | 2277 |

**logging/lora_p9** — P9 (LMM-RGCL video) per-seed LoRA adapter + classifier caches. Depth-1
run dirs (each = adapter_model.safetensors + classifier.bin + configs + checkpoint-N + trainer
logs/plots): `qwen25vl_{mhc,mhc_zh,hatemm}_c3_s{0,1,2}`, plus `predict/`, and the
`_d3*/_c3prime*/_c3repro/_d3f4/_c3primef4` variant seeds. 978 files total.

**logging/Retrieval** — RAC / RGCL-video retrieval memory: per-dataset (`MHC`, `HateMM`,
`MHC_zh`, `ImpliHateVid`, `MHCsmoke`, `MHC_temporal`, `MHC_zh_temporal`) head checkpoints,
kNN memory banks, archives, and per-route subdirs (`RAC_video`, `RAC_video_CLIP`,
`RAC_video_consensus{,_v2,_mm,_seeds}`, `seg*_{l0,l05,drift,milmax}`, `RAC_video_archive{,_seeds}`,
`RAC_video_transcript`, `RAC_video_p{3pool,4aux,5cf,8sum}`, `RAC_tarc_g*`, `c1settle`,
`RAC_vc_sens_*`). Includes 32B-named seed dirs. 2277 files total.

## Destination (B2, distinct subpath — no collision with disk_guard mirror)

```
b2:junyi-data/RGCL_video/manual_backup_2026-07-14/lora_p9
b2:junyi-data/RGCL_video/manual_backup_2026-07-14/Retrieval
```

Remote `b2` = bucket `junyi-data` (rclone v1.70.3, `/data/jehc223/home/.local/bin/rclone`).
disk_guard's own mirror lives under `RGCL_video/{logs,embeddings,adapters,archives,...}`; this
manual backup uses the disjoint `manual_backup_2026-07-14/` prefix (verified empty pre-copy).

Transfer job: `scripts/slurm/b2_backup_manual.sbatch` (CPU-only, no --gres, no --time),
SLURM job **13157**. Per dir: `rclone copy … --transfers 8 --checkers 16 --b2-chunk-size 96M`
then `rclone check <src> <dest> --one-way`. Logs:
`slurm/logs/b2_backup_manual_13157_<dir>.{rclone,check}.log` and job stdout
`slurm/logs/b2backup_13157.out`.

## Retrieval deletion-safety gate — CLEARED

Binding rule: delete Retrieval only if no runtime dependency from `b5_conv_probe.py` or the
b5probe sbatch (job 13156, PENDING). Verdict: **no dependency — safe to delete.** Evidence:
- `scripts/analysis/b5_conv_probe.py` loads CLIP features from `data/CLIP_Embedding` (line 185),
  the 11 heads from `refine-logs/b5_ckpt_snapshot/` (line 203, SNAP), and passes
  `archive_bank=None` (lines 209/212). G-repro anchors are hardcoded dicts (lines 72–91,
  transcribed from the `enc3s_MHC_zh_*_13115.trainlog` primaries), not read from disk at runtime.
- `scripts/analysis/b5_conv_probe.sbatch` — grep `Retrieval|logging` → NONE.
- Only job 13156 was queued besides this backup; no other job references Retrieval.
- The 12 B5 head checkpoints are independently safekept at `refine-logs/b5_ckpt_snapshot/`
  (manifest `refine-logs/B5_HEADS_SAFEKEEP_MANIFEST.md`), so B5 does not need `logging/Retrieval`.

## rclone check evidence — BACKUP VERIFIED CLEAN

Job 13157 `COMPLETED` (elapsed 00:27:29, ExitCode 0:0). Both `rclone copy` exited 0 and both
`rclone check --one-way` exited 0 with zero differences and file counts matching the pre-manifests
exactly (978 and 2277). Quoted from the primary check logs:

```
# slurm/logs/b2_backup_manual_13157_lora_p9.check.log
2026/07/14 21:53:33 NOTICE: B2 bucket junyi-data path RGCL_video/manual_backup_2026-07-14/lora_p9: 0 differences found
2026/07/14 21:53:33 NOTICE: B2 bucket junyi-data path RGCL_video/manual_backup_2026-07-14/lora_p9: 978 matching files
Checks:               978 / 978, 100%, Listed 2102

# slurm/logs/b2_backup_manual_13157_Retrieval.check.log
2026/07/14 22:11:39 NOTICE: B2 bucket junyi-data path RGCL_video/manual_backup_2026-07-14/Retrieval: 0 differences found
2026/07/14 22:11:39 NOTICE: B2 bucket junyi-data path RGCL_video/manual_backup_2026-07-14/Retrieval: 2277 matching files
Checks:              2277 / 2277, 100%, Listed 5310
```

Both dirs are fully and verifiably present on B2. **Restore is guaranteed (see RESTORE section).**

## Deletion — BLOCKED (pending user authorization)

**Local deletion has NOT been performed.** The `rm -rf logging/lora_p9` / `rm -rf logging/Retrieval`
was denied by the Claude Code permission classifier: irreversible local destruction of these
pre-existing paths requires the actual user's authorization (a teammate/orchestrator message does
not meet the consent bar, by design). Both dirs remain on disk (83G + 61G).

To complete deletion, the user must either approve the `rm -rf` when prompted, or add a Bash
permission rule, then run (safe — backup verified clean above):

```bash
rm -rf /data/jehc223/RGCL/logging/lora_p9
rm -rf /data/jehc223/RGCL/logging/Retrieval
```

Expected result: frees ~144G, quota 396G → ~252G. The Retrieval safety gate is already cleared
(no runtime dependency; B5 heads safekept), so no further checks are needed before deletion.

## Quota — AFTER

Unchanged so far (nothing deleted): `396G* / 290G soft / 3000G hard`, grace running, 1212k files.
Will drop to ~252G once the deletion above is authorized and run.

## RESTORE — exact copy-paste commands

Restore either dir from B2 back to the local repo (run from a compute node / sbatch; the login
node reaps sustained transfers):

```bash
RCLONE=/data/jehc223/home/.local/bin/rclone
# lora_p9 (P9 LMM-RGCL video per-seed caches, 83G / 978 files)
$RCLONE copy b2:junyi-data/RGCL_video/manual_backup_2026-07-14/lora_p9 \
    /data/jehc223/RGCL/logging/lora_p9 --transfers 8 --checkers 16 --b2-chunk-size 96M
# Retrieval (RAC/RGCL-video retrieval memory + archives, 61G / 2277 files)
$RCLONE copy b2:junyi-data/RGCL_video/manual_backup_2026-07-14/Retrieval \
    /data/jehc223/RGCL/logging/Retrieval --transfers 8 --checkers 16 --b2-chunk-size 96M
# verify after restore:
$RCLONE check /data/jehc223/RGCL/logging/lora_p9   b2:junyi-data/RGCL_video/manual_backup_2026-07-14/lora_p9   --one-way
$RCLONE check /data/jehc223/RGCL/logging/Retrieval b2:junyi-data/RGCL_video/manual_backup_2026-07-14/Retrieval --one-way
```

## What each dir contained & which records cite it

**logging/lora_p9** (route CLOSED, P9 negative). Cited by:
`research-wiki/EXP_p9_lmm_rgcl_video.md` (P9 run outputs), `research-wiki/DECISION_MEMO_pending.md`,
`research-wiki/TERMINUS_round2_mllm_plus3.md`, `refine-logs/DISK_AUDIT_2026-07-14.md`,
and B4 forensic recon. Recovery if lost = re-run the P9 wave (GPU, hours–days).

**logging/Retrieval** (RAC archives + closed verdicts; the 13115/13150-era run outputs).
Cited by the B-line records (`B1/B2/B3/B4_*` execution/recon/impl/prereg/verdict notes),
`refine-logs/C1_SETTLE_DEV_RECORD.md`, `refine-logs/B5_PROBE_DESIGN.md`,
`research-wiki/EVAL_localization_hatemm.md`, `research-wiki/EVAL_temporal_memory_W4.md`,
`refine-logs/DISK_AUDIT_2026-07-14.md`. Recovery if lost = re-generate embeddings/archives (GPU).

---
_Backup ran as SLURM job 13157 (COMPLETED 2026-07-14, 00:27:29) and BOTH dirs verified 100% clean
on B2 (0 differences, 978 + 2277 matching files). Local deletion is BLOCKED on user permission
(irreversible-destruction gate) and remains the only outstanding step; restore is guaranteed._
