# Disk Audit & Conservative Cleanup — 2026-07-14

Agent: disk-audit. Scope: RGCL (hateful-video) project only. NO GPU, NO SLURM.
Rule: when in doubt -> class B/C (report), never class A (delete).

## Quota (before)

```
lfs quota -h /data/jehc223 (user jehc223, uid 135258174):
  /dev/mapper/data-data   403G*   quota 290G   limit 3000G   grace 14:05 (running)
  filesystem: 14T total, 479G avail, 97% used (shared)
```

User is **~113G OVER the 290G soft quota**; hard limit 3000G (not at risk). Grace timer active.

## Depth-1 inventory: /data/jehc223 (403G total)

| Size | Path | Owner / verdict |
|------|------|-----------------|
| 152G | RGCL | THIS project (144G of it = 2 protected closed-route dirs; see below) |
| 80G  | miniconda3 | shared: 76G envs (10 projects) + 4.0G pkgs cache |
| 47G  | home | 36G .cache (30G HF models + 5.6G pip) + tooling |
| 36G  | SafetyContradiction | OTHER project — hands off |
| 28G  | ExMRD_ours | OTHER project — hands off |
| 27G  | Multihateclip | OTHER project / MHC dataset copy — hands off |
| 14G  | NIPS2026 | OTHER project — hands off |
| 9.6G | HateMM | raw dataset (RGCL uses; irreplaceable) — keep |
| 5.2G | baselines | mixed — hands off |
| 4.2G | HateClipSeg | raw dataset (RGCL P6/P10) — keep |
| <2G  | ExMRD, HVGuard, EMNLP2, difference_awareness, ipiguard, ... | small / other |

## RGCL depth breakdown (152G)

| Size | Path | Class | Note |
|------|------|-------|------|
| 83G  | logging/lora_p9 | **C** | P9 per-seed caches; route CLOSED but **UNBACKED**, cited by B4/EXP_p9 records. gitignored. |
| 61G  | logging/Retrieval | **C** | RAC video archive/embeddings (incl. 32B-named seed dirs); **UNBACKED**. gitignored. (brief said ~41G; actual 61G) |
| 1.4G | logging/lora | B(minor) | MHC + MHC_zh LoRA head checkpoints (3 checkpoint-N each). gitignored. |
| 3.8G | data | C | 2.6G lora_frames + 1.2G CLIP_Embedding (feature caches, back closed verdicts) + small gt/ASR/scores |
| 2.0G | artifacts | C | 1.7G sav_f0 — **mtime TODAY 2026-07-14, likely in active-agent use**; rest tiny. gitignored (204 metadata files force-tracked) |
| 98M  | .git | keep | ~126 commits unpushed; nothing tracked may be deleted |
| <90M | slurm, scripts, RA-HMD, refine-logs, research-wiki, src | keep | code/logs, mostly tracked |

## HF model cache — /data/jehc223/home/.cache/huggingface/hub (29G, SHARED across projects)

All sizable entries are live models or other-project assets; big-name models are empty stubs.

| Size | Model | Verdict |
|------|-------|---------|
| 16G  | Qwen2.5-VL-7B-Instruct | **KEEP** — active encoder |
| 3.2G | openai/clip-vit-large-patch14-336 | KEEP — frozen-CLIP baseline |
| 2.9G | whisper-large-v3 (+282M base) | KEEP — ASR pipeline |
| 1.6G | bert-base-chinese | keep (MHC-ZH) |
| ~4G  | vit-base, sentence-transformers x4, RMBG-1.4, deberta-v3, bert-uncased x2 | small, mixed projects |
| 8-96K | InternVL2/3/3.5-8B..78B, Qwen3-VL-8B/235B, Qwen2.5-VL-32B/72B(.locks only), mistral, safety datasets | **empty stubs / orphan .locks** — aborted downloads, ~0 bytes |

- Qwen2.5-VL-32B / 72B: download logs show 100% fetch, but **model dirs are gone from hub** (only KB `.locks` stubs remain) -> confirms memory "32B/72B absent". No standalone 32B/72B checkpoint exists on disk.
- Second cache /data/jehc223/.cache is empty (LLaVA-NeXT-Video-34B symlink there is dangling).
- Safety datasets (vlsbench/mssbench/SIUO/MM-SafetyBench/VLGuard) belong to SafetyContradiction — hands off.

## conda envs — /data/jehc223/miniconda3/envs (76G, 10 envs)

HateVideo 7.4G + HateVideoVLM 7.3G = RGCL's. Others (SafetyContradiction 12G, implihate 11G, ExMRD 9.3G, AlphaSteerRepro 9.2G, ipiguard 8.3G, MoRE_env/paddle) = other projects. **All keep — active shared infra.**

---

## CLASS A — DELETE NOW (regenerable byproducts)

| Path | Size | Justification | Status |
|------|------|---------------|--------|
| ~/.cache/pip (http-v2) | 5.6G | pip download cache; pure regenerable | pending |
| miniconda3 conda clean --all | ~0.93G | 197 tarballs (329M) + 93 unused extracted pkgs (605M) + index cache; conda never touches installed envs | pending |

(RGCL __pycache__ = 6.4MB total — negligible, left alone to avoid interfering with active agents.)

## CLASS B — RECOMMEND, USER CONFIRMATION REQUIRED

| Path | Size | What / why keep-or-cut | Recovery if deleted |
|------|------|------------------------|---------------------|
| logging/lora | 1.4G | MHC/MHC_zh LoRA head checkpoints, 3 intermediate checkpoint-N per run. Could keep only final. Small win. | Re-train head (features cached) |
| logging/lora_p9 | 83G | P9 route CLOSED. **UNBACKED** — deletion permanent. Biggest single lever. Class-C by brief but this is THE quota fix if user accepts loss. | Re-run P9 wave (GPU, hours-days) |
| logging/Retrieval | 61G | RAC embeddings/archives, closed verdicts. **UNBACKED** — deletion permanent. Second-biggest lever. | Re-generate embeddings (GPU) |

## CLASS C — DO NOT DELETE (report only)

- logging/lora_p9 (83G) & logging/Retrieval (61G) — unbacked, cited by records (also listed in B as the only large levers; default = keep).
- data/CLIP_Embedding (1.2G), data/lora_frames (2.6G) — feature caches backing closed verdicts.
- artifacts/sav_f0 (1.7G) — mtime today, likely active-agent in-use.
- Qwen2.5-VL-7B (16G), CLIP (3.2G), whisper (3.2G) — live models.
- Raw datasets HateMM (9.6G), HateClipSeg (4.2G), Multihateclip (27G) — irreplaceable.
- Anything git-tracked; all OTHER-project dirs (SafetyContradiction, ExMRD_ours, NIPS2026, ...).

---

## Execution log (class A) — DONE

- `pip cache purge`: removed 1169 files, **5.77G** freed (5.6G -> 88K). Regenerable.
- `conda clean --all -y`: 197 tarballs + 93 unused pkgs + index cache, **~1.9G** freed (pkgs 4.0G -> 2.1G). Installed envs untouched.
- Total class-A freed: **~7.5G**.

## Quota (after)

```
/dev/mapper/data-data   396G*   quota 290G   limit 3000G   grace 14:00 (still running)
du /data/jehc223 = 395G
```

Still **~106G over the 290G soft quota**. Class-A cleanup alone cannot clear it.

## Bottom line for user

Within RGCL, the only large recoverable space is the two **UNBACKED, closed-route** dirs:
- `logging/lora_p9` (83G) and `logging/Retrieval` (61G) = **144G**.
Deleting either is permanent (no backup) but neither route is active. **Requires user ruling** — I did NOT touch them.
Outside RGCL (hands off, reported): SafetyContradiction 36G, ExMRD_ours 28G, Multihateclip 27G, NIPS2026 14G.

---

**Update 2026-07-14:** User ruled backup-then-delete for both large dirs. Both were backed up to
`b2:junyi-data/RGCL_video/manual_backup_2026-07-14/{lora_p9,Retrieval}` and verified clean
(rclone check: 0 differences, 978 + 2277 matching files; SLURM job 13157 COMPLETED). Local
`rm -rf` deletion is BLOCKED on the user-permission (irreversible-destruction) gate — the ~144G
is not yet freed. Full evidence + restore/delete commands: see `DISK_BACKUP_RECORD_2026-07-14.md`.

