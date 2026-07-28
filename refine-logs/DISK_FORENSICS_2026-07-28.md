# DISK FORENSICS — 2026-07-28 (READ-ONLY MAP + RECLAIM MENU)

**Executor:** disk-forensics subagent. **Mode:** read-only mapping. **Date:** 2026-07-28 NZST.

**Nothing was deleted, moved, compressed, pushed to B2, or reclaimed. `scripts/disk_guard.sh` was
NOT run and NOT modified.** Every command below is a read (`quota`, `df`, `du`, `find`, `ls`,
`grep`, `squeue`). This document is a map and a menu for the user to rule on — it executes nothing.

---

## 1. THE NUMBERS (verbatim)

### 1.1 `quota -s` — exit status 1, output fully parseable

```
Disk quotas for user jehc223 (uid 135258174):
     Filesystem   space   quota   limit   grace   files   quota   limit   grace
/dev/mapper/data-data
                   304G*   290G   3000G   5days   1234k       0       0
/dev/mapper/rhel-home
                    36K  92160K    100M              13       0       0
/dev/mapper/rhel-tmp
                   160K    200M    250M             156       0       0
```

Raw (1K-block) form, same call without `-s`:

```
/dev/mapper/data-data
                317766476* 304087040 3145728000   5days 1233904       0       0
```

| field | value |
|---|---|
| **Quota type** | **per-USER** (`Disk quotas for user jehc223 (uid 135258174)`). Not a group/project quota. |
| **Filesystem** | `/dev/mapper/data-data`, mounted at **`/data`** |
| **Current usage** | **304G** (`317,766,476` 1K-blocks = 303.0 GiB). The `*` marks over-soft-limit. |
| **Soft quota** | **290G** (`304,087,040` blocks) |
| **Hard limit** | **3000G** (`3,145,728,000` blocks) — not remotely close |
| **Grace remaining** | **`5days`** — this is the exact field as `quota` prints it, verbatim, not an estimate. |
| **Over by** | **~14G** above soft |
| **Inodes** | 1234k used, **no inode quota** (0/0) — inodes are not a risk |

Note on grace granularity: `quota` renders the grace field in the largest whole unit; `5days`
means the timer has more than 5 and fewer than 6 days left. It switches to `HH:MM` display under
one day. `research-wiki/DISK_POLICY.md` documents the grace period as 6 days, consistent with a
soft-limit crossing that happened within roughly the last day. There is no finer-resolution field
exposed by `quota`; `repquota` requires root.

The exit code is the whole bug: **`quota` returned 1 while printing a complete, correct,
parseable table.** Over-quota is a *signal* encoded in the exit status, not a failure.

### 1.2 `df -h` on the relevant mount

```
Filesystem             Size  Used Avail Use% Mounted on
/dev/mapper/data-data   14T   13T  1.7T  89% /data
```

The underlying volume is 14T with 1.7T free — **the filesystem is not full.** This is purely a
per-user quota event. `$HOME` is `/data/jehc223/home`, i.e. **home is on the same quota'd
filesystem** and counts against the same 290G. (`/dev/mapper/rhel-home` at 36K/100M is a
vestigial mount, irrelevant.)

### 1.3 In-flight work

`squeue -u jehc223` — **empty. No SLURM jobs running or queued.** Nothing is mid-write right now,
so there is no job to be killed by an EDQUOT this instant. That is the good news; it also means
the next submitted job is the one that would hit it.

---

## 2. WHERE THE SPACE LIVES

Total under `/data/jehc223` = **302.6 GiB** (matches the 304G quota reading; all quota-covered
space is under this one root — a `find /data -maxdepth 1 -user jehc223` returns only
`/data/jehc223`).

### 2.1 Top level

| Path | Size | Owner project |
|---|---|---|
| `/data/jehc223/miniconda3` | **77.3 GiB** | shared (9 envs) |
| `/data/jehc223/home` | **60.0 GiB** | shared ($HOME, on the same quota) |
| `/data/jehc223/SafetyContradiction` | **35.6 GiB** | OTHER project |
| `/data/jehc223/ExMRD_ours` | **27.7 GiB** | OTHER project |
| `/data/jehc223/Multihateclip` | **26.8 GiB** | RGCL — RAW VIDEO |
| **`/data/jehc223/RGCL`** | **23.6 GiB** | **THIS project — confirmed ~24G** |
| `/data/jehc223/models` | **16.2 GiB** | RGCL (Molmo2-8B) |
| `/data/jehc223/NIPS2026` | **13.7 GiB** | OTHER project |
| `/data/jehc223/HateMM` | **9.6 GiB** | RGCL — RAW VIDEO |
| `/data/jehc223/baselines` | 5.2 GiB | RGCL (MoRE baseline) |
| `/data/jehc223/HateClipSeg` | 4.2 GiB | RGCL — RAW VIDEO |
| `/data/jehc223/ExMRD` | 1.2 GiB | OTHER project |
| everything else | ~1.2 GiB | mixed |

**RGCL is confirmed at 23.6 GiB — 7.8% of the total.** The other ~279 GiB decomposes as:

| bucket | size | note |
|---|---|---|
| conda environments (9) | 74.5 GiB | only 2 are RGCL's (`HateVideo`, `HateVideoVLM` = 14.6 GiB) |
| other people's projects on this account | 78.2 GiB | SafetyContradiction + ExMRD_ours + NIPS2026 + ExMRD |
| raw video datasets | 40.6 GiB | Multihateclip + HateMM + HateClipSeg — **never leaves the machine** |
| HuggingFace model cache | 30.2 GiB | `~/.cache/huggingface` |
| **Claude Code scratchpad** | **21.5 GiB** | `~/tmp/claude-135258174` — see §2.3, this is the anomaly |
| Molmo2-8B local weights | 16.2 GiB | downloaded 2026-07-27 |
| editor/tooling caches | ~6.4 GiB | `.vscode-server`, `.npm`, `.codex`, `.claude`, `.nvm`, `.local` |
| conda pkgs cache | 2.0 GiB | |

### 2.2 Top 30 directories by size

```
  302.61 GiB  /data/jehc223
   77.29 GiB  /data/jehc223/miniconda3
   74.51 GiB  /data/jehc223/miniconda3/envs
   60.01 GiB  /data/jehc223/home
   35.57 GiB  /data/jehc223/SafetyContradiction
   34.50 GiB  /data/jehc223/SafetyContradiction/ckpt/pku_insulting_behavior_train
   30.68 GiB  /data/jehc223/home/.cache
   30.19 GiB  /data/jehc223/home/.cache/huggingface
   27.72 GiB  /data/jehc223/ExMRD_ours
   26.80 GiB  /data/jehc223/Multihateclip
   23.57 GiB  /data/jehc223/RGCL
   22.02 GiB  /data/jehc223/home/tmp
   21.53 GiB  /data/jehc223/home/tmp/claude-135258174
   16.15 GiB  /data/jehc223/models/Molmo2-8B-bf16
   15.95 GiB  /data/jehc223/ExMRD_ours/data/FakeSV
   15.47 GiB  /data/jehc223/RGCL/logging
   15.09 GiB  /data/jehc223/Multihateclip/English
   13.70 GiB  /data/jehc223/NIPS2026
   11.98 GiB  /data/jehc223/miniconda3/envs/SafetyContradiction
   11.71 GiB  /data/jehc223/Multihateclip/Chinese
   10.82 GiB  /data/jehc223/ExMRD_ours/.git/objects
   10.04 GiB  /data/jehc223/miniconda3/envs/implihate
    9.58 GiB  /data/jehc223/HateMM
    9.13 GiB  /data/jehc223/miniconda3/envs/AlphaSteerRepro
    9.05 GiB  /data/jehc223/NIPS2026/reproduce/processed_features
    9.01 GiB  /data/jehc223/miniconda3/envs/ExMRD
    8.91 GiB  /data/jehc223/RGCL/logging/Retrieval
    8.24 GiB  /data/jehc223/miniconda3/envs/ipiguard
    7.96 GiB  /data/jehc223/Multihateclip/Chinese/video
    7.29 GiB  /data/jehc223/miniconda3/envs/HateVideoVLM
    7.28 GiB  /data/jehc223/miniconda3/envs/HateVideo
    6.87 GiB  /data/jehc223/Multihateclip/English/video_mp4
    6.55 GiB  /data/jehc223/RGCL/logging/lora
    6.12 GiB  /data/jehc223/HateMM/video
```

### 2.3 The anomaly: 21.5 GiB of training checkpoints inside a temp scratchpad

Effectively all of `~/tmp/claude-135258174` sits in **one** Claude Code session directory:

```
21.53 GiB  ~/tmp/claude-135258174/-data-jehc223-RGCL/e8f03e41-3e21-4cea-b12c-29207373bfca/scratchpad
├── 14.82 GiB  molmo2_probe/Retrieval/HateMM/RAC_molmo2_probe_{A,B,C}/…/ckpt/
├──  4.53 GiB  errpat/Retrieval/HateMM/RAC_errpat_proxy/…/ckpt/
├──  1.68 GiB  molmo2_probe_dryrun/Retrieval/
├──  0.22 GiB  r3/adapter_model.safetensors
└──  0.22 GiB  ks2_cwd/adapter/adapter_model.safetensors
```

Composition: **390 files named `epoch_model_<N>_<acc>.pt`, 15.6 GiB total**, at ~46 MB each.
These are *per-epoch* head checkpoints — every one of 30 epochs × 3 arms × 3 seeds retained —
written into the scratchpad because those probe runs were launched with `cwd` set there, so the
trainer's relative `Retrieval/<dataset>/…` output path landed under `/tmp`, not under `RGCL/`.

This is the single largest, cheapest, lowest-risk pool of reclaimable space on the account, and
it is **completely invisible to `disk_guard.sh`**, whose scope is `RGCL_ROOT` only.

Same pattern, in-repo: `RGCL/logging` holds another **180 `epoch_model_*.pt`, 7.0 GiB**.

---

## 3. CLASSIFICATION — every contributor ≥2 GiB

Legend: **(A)** safely reclaimable · **(B)** reclaimable but load-bearing · **(C)** do not touch ·
**(D)** unknown / needs a human.

### 3.1 Class (A) — safely reclaimable

| Path | Size | Regeneration / consequence of deleting |
|---|---|---|
| `~/tmp/…/e8f03e41…/scratchpad/molmo2_probe` | **14.82 GiB** | Per-epoch heads of the **Molmo2 probe, verdict KILL** (`refine-logs/MOLMO2_PROBE_RECORD.md`: "*VERDICT: KILL* … FAIL on both metrics on both protocols, not marginal, not a split"). Numbers are already transcribed into the record. Regeneration = re-run the $0 CPU-head probe from `data/CLIP_Embedding/HateMM/{train,dev_seen,test_seen}_Molmo2-8B_HF.pt` (which are class C and stay). Cost: **0 GPU-h**, ~10 min CPU (the 9 arms ran 10:08→10:17 on 2026-07-27). Consequence: nothing — a killed route loses its intermediate weights. |
| `~/.cache/huggingface/hub` redundant weight **formats** | **4.45 GiB** | Five models cache the *same weights twice or four times*: `clip-vit-large-patch14-336` (`pytorch_model.bin` 1.59 GiB alongside `model.safetensors`), `bert-base-chinese` (`tf_model.h5` + `flax_model.msgpack` + `pytorch_model.bin` = 1.21 GiB), `vit-base-patch16-224` (same three = 0.97 GiB), `deberta-v3-base` (0.35 GiB), `RMBG-1.4` (0.33 GiB). `transformers` loads `model.safetensors` preferentially. Regeneration: re-download on demand. Consequence: none unless a script pins `use_safetensors=False`. |
| `~/.vscode-server` | **4.03 GiB** | VS Code remote server + extensions + CLI. Re-downloads automatically on next Remote-SSH connect. Consequence: one slow reconnect. |
| `__pycache__` (all of `/data/jehc223`) | **3.29 GiB** | Byte-compiled Python, mostly in conda `site-packages`. Regenerated on next import. Consequence: marginally slower first import. |
| `miniconda3/pkgs` | **1.91 GiB** | conda package tarball cache (`conda clean --all`). Regeneration: re-download when an env is next built. Consequence: none to existing envs. |
| `~/tmp/…/scratchpad/molmo2_probe_dryrun` | **1.68 GiB** | Dry-run byproduct of the same killed probe. Consequence: none. |
| `~/tmp/…/scratchpad/{r3,ks2_cwd}` adapters | 0.44 GiB | Two copies of one 234 MB `adapter_model.safetensors` (2026-07-26). Consequence: none — scratch copies. |
| `~/.cache/huggingface/hub/models--bert-base-uncased` | 0.41 GiB | **Exact duplicate** of `models--google-bert--bert-base-uncased` — verified *same blob hash* `68d45e234eb4…` stored as two separate 440,449,768-byte files (old vs new repo naming, not hardlinked). Consequence: none, the canonical `google-bert/` copy remains. |
| `~/.npm/_cacache` | 0.31 GiB | npm cache. Regenerates. |
| `~/tmp` misc (`torchinductor_jehc223` 156M, `naive_frames_m_pgwijw` 290M, ~20 `.tmp*` dirs) | ~0.50 GiB | Compile caches and abandoned temp extractions. Consequence: none. |
| `slurm/logs/disk_guard.log` | 0.06 GiB | 797,051 lines, 322 of them the "could not determine quota" warning. Symptomatic, not large. Truncating loses guard history only. |

**Class (A) total ≈ 31.9 GiB.**

### 3.2 Class (B) — reclaimable but load-bearing

| Path | Size | Restore path and cost |
|---|---|---|
| `RGCL/logging/Retrieval` | **8.91 GiB** | Entirely `*/mntp_s1_cpuhead/*` — the **MNTP S1 route, which STOPped at its kill-switch** (`refine-logs/MNTP_S1_RECORD.md`: `KS-MNTP-1` → "**STOP (do not continue)**"; HateMM recovery fraction −0.1999, "below the crater — KILL-side"). Note the *earlier* `logging/Retrieval` was already backed up to B2 and deleted on 2026-07-14 (`DISK_BACKUP_RECORD_2026-07-14.md`, 2277 files, `rclone check` 0 differences, restorable from `b2:junyi-data/RGCL_video/manual_backup_2026-07-14/Retrieval`); **this 8.91 GiB is regrowth from runs after that date and is NOT on B2.** Restore cost if deleted without a push: re-run MNTP S1 extraction (job 13654, `COMPLETED 00:49:16`, GPU) + head training. **≈1 GPU-h.** If pushed to B2 first (the guard's own push-verify-prune, or `scripts/slurm/b2_backup_manual.sbatch`), restore is a free download. |
| `RGCL/logging/lora` intermediate `checkpoint-N/` dirs | **4.56 GiB** | 27 HF-trainer mid-training checkpoints across 9 run dirs. The final adapter + classifier sit alongside them at run-dir root and are **not** in this figure. Restore cost: full LoRA retrain (multi-GPU-h) — but these are resume-points, not results; nothing reads them. |
| `RGCL/data/lora_frames` | **2.57 GiB** | Decoded frames for LoRA SFT. Regenerable from the raw video already on disk (class C) by re-running frame extraction: CPU-only, ~tens of minutes. |
| `~/tmp/…/scratchpad/errpat` | **4.53 GiB** | Proxy retrains consumed by `scripts/analysis/errpat_hatemm_{forensics,clusters,ceilings}.py`, which glob `epoch_model_%d_*.pt` under a `--proxy_root`. Memory records head retrains at ~52 s CPU, so regeneration is cheap — **but see §3.3: this is currently live.** |

### 3.3 Class (C) — do not touch

| Path | Size | Why |
|---|---|---|
| `RGCL/data/CLIP_Embedding` | **2.05 GiB** | The 7168-d fused train caches every $0 pregate depends on, incl. `HateMM/{train,dev_seen,test_seen}_Molmo2-8B_HF.pt`. Deleting = every pregate needs GPU re-extraction. **Also: this path is in `disk_guard.sh`'s `_DG_ALLOWED_ROOTS` — see §5.3.** |
| `RGCL/scripts/analysis` (incl. `p2_out/cache_*.json`, 52 × `*_OUT.json`) | 0.16 GiB | Deployed-space neighbour lists and pregate outputs. Small; irreplaceable without GPU. |
| `RGCL/data/{ASR,Archive,gt}` | 38 MiB | Transcripts, archive memory, ground-truth splits. Tiny, load-bearing. |
| `RGCL/refine-logs/{b5_ckpt_snapshot,router_ckpt_snapshot}` | 0.68 GiB | Safekept head checkpoints (`B5_HEADS_SAFEKEEP_MANIFEST.md`); explicitly the reason `logging/Retrieval` was safe to delete in July. |
| `RGCL/artifacts` | 1.98 GiB | Banked route artifacts (`sav_f0` 1.70 GiB, `ssr`, `lb_scgp*`, `c3_nontarget`). SAV is a closed route, but these are the *evidence*, not intermediates. |
| `/data/jehc223/Multihateclip` | **26.80 GiB** | **RAW VIDEO — hard rule, never leaves the machine, never pushed to B2.** Not re-derivable locally. |
| `/data/jehc223/HateMM` | **9.58 GiB** | **RAW VIDEO** (+ extracted frames). Same rule. |
| `/data/jehc223/HateClipSeg` | **4.16 GiB** | **RAW VIDEO.** Same rule. Holds the frozen 60/10/30 split. |
| `~/tmp/…/scratchpad/errpat` | 4.53 GiB | **Live.** `ERRPAT_MHC-ZH_2026-07-26.md` was modified **today, 2026-07-28 09:09**, and an `errpat-audit` agent is active in this session. Class (B) on the merits, class (C) *today*. Do not touch until that agent lands. |
| `miniconda3/envs/{HateVideo,HateVideoVLM}` | 14.57 GiB | `HateVideo` is the project env named in `CLAUDE.md`. Rebuildable but a multi-hour, dependency-fragile operation (the `python-socks`/`aiohttp-socks` Modal deps are easy to lose). |

### 3.4 Class (D) — unknown / needs a human

| Path | Size | The question |
|---|---|---|
| `SafetyContradiction/ckpt/pku_insulting_behavior_train` | **34.50 GiB** | **Largest single directory on the account.** A full ~8B model in bf16 (4 shards, 18.5 GB) *plus* `checkpoint-2/` (**17.25 GiB**) — a checkpoint at **training step 2** (`trainer_state.json`, `trainer_log.jsonl` = 130 bytes). Dated 2026-03-11, untouched since. A step-2 checkpoint is near-certainly worthless, and the parent may be a re-derivable SFT output. **Different project — the user must rule.** Potential: 17–34 GiB. |
| `ExMRD_ours/.git/objects` | **10.82 GiB** | **100,709 loose objects; the packfiles are only 380 MiB.** A plain `git gc`/`git repack` would reclaim roughly 10 GiB **without losing a single commit or byte of history** — it is a repack, not a deletion. But it *writes* to another project's repo, so it is out of scope here. Different project — user ruling. |
| `ExMRD_ours/data/FakeSV` | 15.95 GiB | Frames + quads for another project's dataset. Derived, probably regenerable, not ours. |
| `NIPS2026/reproduce/processed_features` | 9.05 GiB | Another project's derived feature cache. |
| `miniconda3/envs/{SafetyContradiction,implihate,AlphaSteerRepro,ExMRD,ipiguard,MoRE_env,MoRE_paddle}` | **59.9 GiB** | Seven envs for other projects. Any that are dead are pure reclaim (`conda env remove`), and this is the **single largest soft target on the account**. Needs the user to say which projects are still alive. |
| `/data/jehc223/models/Molmo2-8B-bf16` | **16.15 GiB** | Downloaded 2026-07-27; the probe it was downloaded for returned **KILL** the same day. Re-downloadable from HF (bandwidth + `CONVERSION_NOTE.md` conversion step). Keep only if Molmo2 is coming back for a different cell — that is a research call, not a disk call. |
| `baselines/MoRE` | 5.15 GiB | Baseline reimplementation data + reruns. Status unclear. |

---

## 4. QUANTIFYING THE MENU

Current: **304G**. Soft limit **290G**. Sane working headroom target **260G**.

| goal | must free | |
|---|---|---|
| under 290G (stop the grace clock) | **14 GiB** | |
| under 260G (working headroom) | **44 GiB** | |

### 4.1 Is (A) enough?

**Class (A) ≈ 31.9 GiB.**

- **Under 290G: YES, comfortably — and with a wide margin.** In fact `molmo2_probe` alone
  (14.82 GiB) clears the soft limit by itself, landing at ~289G. Class (A) in full lands at
  **~272G**, i.e. 18G of headroom under the soft limit, with **zero GPU cost, zero research
  consequence, and nothing needing restore.**
- **Under 260G: NO.** (A) falls ~12 GiB short.

### 4.2 Smallest (B) addition to reach 260G

| item | size | restore cost |
|---|---|---|
| `RGCL/logging/Retrieval` (MNTP S1 — route STOPped at kill-switch) | 8.91 GiB | ~1 GPU-h re-extraction **if deleted cold**; **free** if pushed to B2 first |
| `RGCL/logging/lora` intermediate `checkpoint-N/` dirs only | 4.56 GiB | nothing reads them; final adapters are untouched |
| **(A) + these two** | **≈45.4 GiB** | → **~258G. Under 260G.** |

Deliberately excluded from that minimum: the 4.53 GiB `errpat` scratchpad (live today, §3.3) and
all raw video (class C, hard rule).

If the user wants real slack rather than a bare 260G, the honest observation is that the three
biggest levers are all **outside RGCL**: the dead conda envs (up to 59.9 GiB), the
SafetyContradiction step-2 checkpoint (17.25 GiB), and an `ExMRD_ours` repack (~10 GiB, lossless).
Any one of those dwarfs everything RGCL can offer. **RGCL is not the problem — it is 7.8% of the
disk and its own reclaimable share is ~13 GiB.**

### 4.3 What a naive guard run would destroy — AT-RISK FLAGS

If `scripts/disk_guard.sh` were "fixed" and run right now (`THRESHOLD=250G`, `TARGET=250G`,
usage 304G ⇒ needs to free **54 GiB**):

1. **Step (a) rclone VFS cache** — `~/.cache/rclone/vfs*` does not exist. No-op.
2. **Step (b) HF cache** — skipped by default (`DISK_GUARD_HF_PURGE=0`), and `models--*` is
   protected by an explicit guard even when enabled. Safe.
3. **Step (c) `reclaim_logging_checkpoints`** — this is the one that bites. It globs
   **every `*.pt` under `RGCL/logging`, oldest-first**, and for each does push→verify→prune to
   `b2:junyi-data/RGCL_video/logs/<relpath>`. Because it needs 54 GiB and `logging/` only holds
   **15.47 GiB in total**, the stop condition (`_freed >= _need_bytes`) is **never reached** —
   so it would walk the entire list and prune **all 180 `epoch_model_*.pt` (7.0 GiB) plus every
   other `.pt` under `logging/`, i.e. the whole of `logging/Retrieval` and `logging/lora`,**
   then still log `MANUAL INTERVENTION required` and stop at ~288G.
   *Mitigation already in the code:* every prune is gated on a verified B2 copy, so this is
   recoverable, not destructive. But it would happen **unattended, mid-job**, and would burn
   ~15 GiB of B2 upload to end up still over target.
4. **`RGCL/data/CLIP_Embedding` is in `_DG_ALLOWED_ROOTS` (line 85).** No current step targets it,
   so it is safe *today* — but the campaign's most load-bearing artifact is sitting on the
   guard's permission list with nothing but the absence of a code path protecting it. **This is a
   latent hazard and should be removed from the allowlist regardless of what else is decided.**
5. **Out of the guard's reach entirely:** the 21.5 GiB scratchpad, the 30.2 GiB HF cache
   duplicates, `models/Molmo2-8B-bf16`, and every other project. The guard cannot see the biggest
   wins and can only eat this project's own checkpoints.

---

## 5. OBVIOUS-WINS CHECKLIST (§5 of the tasking)

| check | finding |
|---|---|
| **Duplicate model weights** | **YES, 4.86 GiB.** (a) Five HF models cache the same weights in 2–4 formats simultaneously (`pytorch_model.bin` / `tf_model.h5` / `flax_model.msgpack` beside `model.safetensors`) = **4.45 GiB**. (b) `models--bert-base-uncased` and `models--google-bert--bert-base-uncased` are the *same blob hash* `68d45e234eb4…` stored twice as separate 440,449,768-byte files = **0.41 GiB**. Qwen2.5-VL-7B (16 GiB, 5 shards) is correctly sharded, **not** duplicated; whisper-large-v3 is single-copy. `models/Molmo2-8B-bf16` is a local conversion, not an HF-cache dupe. |
| **`.trash` / `tmp` / `scratch` / staging dirs** | **YES — this is the headline.** `~/tmp/claude-135258174` = **21.53 GiB**, 99% of it one session's `scratchpad/` holding 390 per-epoch `.pt` checkpoints (15.6 GiB) from a **killed** probe. Plus ~20 abandoned `~/tmp/.tmp*` dirs and `torchinductor`/`naive_frames` leftovers (~0.5 GiB). No `.trash` dir exists. |
| **Half-written / zero-byte `.pt`** | **NO CORRUPTION FOUND — clean bill of health.** Zero-byte match count across all `*.pt`/`*.pkl`/`*.safetensors`/`*.bin` under `/data/jehc223` = **3**, and all three are HuggingFace `.no_exist/` sentinel markers (`models--{deberta-v3-base,clip-vit-large-patch14-336,Qwen2.5-VL-7B-Instruct}/.no_exist/*/model.safetensors`) — these are *intentional* zero-byte "this file is absent upstream" markers, not truncation. No `*.pt` under 1 KB. No `*.incomplete` files in the HF cache. **Conclusion: no prior EDQUOT event has damaged any artifact.** The banked caches are intact. |
| **`slurm/logs/disk_guard.log` (67 MB)** | Confirmed 67,376,640 bytes / **797,051 lines**, last written 2026-07-27 13:20. Contains **322** `could not determine quota usage` warnings. Its **highest successful reading ever is 289G** — 1G under the limit — exactly as reported: the guard has never once observed an over-quota state, because observing one is what breaks it. Not a space problem; a smoking gun. |
| **`RA-HMD/LLAMA-FACTORY*` weights** | **NO large weights.** `RA-HMD/LLAMA-FACTORY` = **0 bytes** (empty, uninitialised submodule). `RA-HMD/LLAMA-FACTORY-Ver202512` = **15 MiB** (11 MiB `data/`, 3 MiB `src/`) — source only. `RA-HMD/Stage2` = 236 KiB. The whole `RA-HMD` tree is 16 MiB. **Not a contributor; rule it out.** |

---

## 6. THE GUARD BUG — REPORTED, NOT FIXED

### 6.1 The one-line change (written out, **NOT applied**)

`scripts/disk_guard.sh:134`, currently:

```bash
    _q="$(quota -s 2>/dev/null)" || { return 1; }
```

The `||` branch fires on `quota`'s exit status. But `quota` exits **1 precisely when a filesystem
is over quota** — the signal, not an error — so the guard throws away a perfectly good reading at
the only moment it matters. The very next line (`[[ -z "$_q" ]] && return 1`) *already* handles
the genuine failure modes, because a missing command or a real error yields empty stdout.

**Proposed replacement (one line changed, one line added):**

```bash
    command -v quota >/dev/null 2>&1 || return 1     # real failure: command missing
    _q="$(quota -s 2>/dev/null)"                     # exit status is a SIGNAL (1 == over quota), not an error
```

This distinguishes correctly:

| situation | `quota` exit | stdout | outcome |
|---|---|---|---|
| command missing | n/a | — | `command -v` fails → `return 1` ✔ real failure |
| real error / no quotas | 0 or ≠0 | empty | existing `[[ -z "$_q" ]]` → `return 1` ✔ |
| **under quota** | **0** | table | parsed ✔ |
| **over quota (today)** | **1** | table | **parsed ✔ — the bug, fixed** |

### 6.2 Would a fixed guard have a safe operating point? **No — not without re-scoping first.**

The arithmetic is decisive. With `TARGET=250G` and usage 304G, the guard must free **54 GiB**.
Its only prunable step operates on `RGCL_ROOT/logging`, which holds **15.47 GiB**. Even after
deleting **100% of this project's checkpoints** it lands at ~288G — still 38 GiB over its own
target. Its scope (`RGCL_ROOT` = 23.57 GiB) is **7.8% of the 302.6 GiB total** and contains only
~13 GiB of legitimately reclaimable material.

So a merely-parse-fixed guard has exactly one behaviour available: prune everything it can reach,
upload ~15 GiB to B2 doing so, and still fail. That is strictly worse than staying blind.

**Both must change together, and both are user decisions:**

1. **`DISK_GUARD_TARGET_GB` must become reachable within scope.** Something like
   `THRESHOLD=295 / TARGET=288` is achievable by the existing step (c); 250 is not, from this
   scope, ever. A guard whose target is unreachable is a guard that always runs to completion.
2. **`RGCL_ROOT` / the allowlist need re-scoping to where the space actually is** — the
   `~/tmp/claude-*/**/scratchpad/**/epoch_model_*.pt` pool (21.5 GiB, pure byproduct) and the HF
   redundant-format blobs (4.45 GiB) are the correct targets, and both are currently invisible to
   it. Conversely `RGCL/data/CLIP_Embedding` should come **off** `_DG_ALLOWED_ROOTS` (§4.3 item 4).
3. Better still, fix the generator rather than the collector: the probe sbatch files already
   demonstrate the right pattern —
   `find "$RUNDIR" -name 'epoch_model_*.pt' -delete` appears in
   `scripts/slurm/train_{p4aux,p5cf,p8sum}.sbatch` but **not** in the paths that produced these
   390 files. Retaining only the selected epoch would have prevented this entire event.

---

## 7. RECOMMENDATION (for user approval — NOT executed)

**Minimal action, and it is genuinely minimal:** delete the two Molmo2 probe scratch trees.

```
~/tmp/claude-135258174/-data-jehc223-RGCL/e8f03e41-3e21-4cea-b12c-29207373bfca/scratchpad/molmo2_probe          14.82 GiB
~/tmp/claude-135258174/-data-jehc223-RGCL/e8f03e41-3e21-4cea-b12c-29207373bfca/scratchpad/molmo2_probe_dryrun    1.68 GiB
```

**16.5 GiB → ~287G. Under the soft limit, grace clock stops.** Cost: zero GPU, zero research
consequence — the route is KILLed with its verdict already written to
`refine-logs/MOLMO2_PROBE_RECORD.md`, and the class-(C) Molmo2 feature caches that would let
anyone re-run it in ~10 min of CPU are untouched in `data/CLIP_Embedding/HateMM/`.

**Recommended action (full class A):** add the HF format duplicates (4.45 GiB), the
`bert-base-uncased` twin (0.41 GiB), `.vscode-server` (4.03 GiB), `__pycache__` (3.29 GiB),
`miniconda3/pkgs` (1.91 GiB) and the scratch adapters/tmp leftovers (~1.0 GiB) →
**~31.9 GiB → ~272G**, ~18G of headroom, still with **nothing to restore and no GPU spent**.

**Do not** run `disk_guard.sh` even after fixing the parse, until `TARGET` and `RGCL_ROOT` are
re-scoped (§6.2) — as configured it would prune every checkpoint this project owns and still
miss its target.

**Two questions only the user can answer**, each worth more than everything above:
- Are any of the seven non-RGCL conda envs dead? (**up to 59.9 GiB**)
- Can `SafetyContradiction/ckpt/…/checkpoint-2` — a **step-2** checkpoint from March —
  go? (**17.25 GiB**) And may `ExMRD_ours` be repacked with `git gc` (**~10 GiB, lossless,
  no history lost**)?

---

## 8. PROVENANCE

All figures from read-only commands run 2026-07-28 on `foscsmlprd01`: `quota -s`, `quota`,
`df -h`, `du -sb`/`du -h --max-depth=N`, `find`, `ls -la`, `grep`, `squeue -u jehc223`.
Sizes in GiB from `du -sb` (binary) unless quoting `quota`/`df` output verbatim, which use their
own units. `du` totals `/data/jehc223` at 302.61 GiB against `quota`'s 304G reading — the ~1G
delta is sparse-file/block-accounting, not a missing directory.

Cross-referenced: `refine-logs/DISK_BACKUP_RECORD_2026-07-14.md`,
`refine-logs/DISK_AUDIT_2026-07-14.md`, `research-wiki/DISK_POLICY.md`,
`refine-logs/MOLMO2_PROBE_RECORD.md`, `refine-logs/MNTP_S1_RECORD.md`,
`refine-logs/ERRPAT_*.md`, `scripts/disk_guard.sh`.

**No file outside this one was created, modified, moved or deleted by this audit.**
