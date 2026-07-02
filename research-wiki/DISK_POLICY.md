# DISK_POLICY — sustainable quota management for RGCL video

**Status:** active · **Owner:** jehc223 · **Last updated:** 2026-07-01

The `/data` quota is **soft 290G / hard 3000G**, with a **6-day grace** period once
usage crosses the soft limit. The login node *is* the compute node
(`foscsmlprd01`) and reaps non-SLURM background processes, so **CRON is
unreliable** — enforcement must ride on the SLURM jobs themselves. This document
is the standing policy so we stop leaning on the grace period.

---

## 1. Policy in one line

**B2 is the cold store. Local stays lean. Every sbatch job runs `disk_guard.sh`
first. Nothing is deleted until a byte-identical copy is verified on B2
(push → verify → prune).**

- **B2 = cold store.** All derived artifacts (CLIP embeddings, training
  checkpoints/logs) are mirrored to `b2:junyi-data/RGCL_video/`. B2 is the source
  of truth for anything reproducible/derived.
- **Local = lean.** The working tree keeps only what the *current* run needs.
  Old checkpoints and embeddings live on B2 and are pulled on demand
  (`scripts/b2_pull.sh`).
- **Per-sbatch guard.** `scripts/disk_guard.sh` runs at the top of every job.
  Below threshold it is a no-op; above threshold it reclaims space in a fixed,
  safe order.
- **Push → verify → prune.** A local file is deleted **only** after its SHA1 is
  confirmed byte-identical on B2. No verify, no delete. Ever.

---

## 2. Audit table (2026-07-01, after cache purge + safe prune)

Quota at time of audit: **282G / 290G** (≈8G under soft limit).

Ranked largest-first. "Ours?" = belongs to THIS project.

| Path | Size | Ours? | On B2? | Prune now? | Notes |
|---|---|---|---|---|---|
| `/data/jehc223/AlphaSteer` | **93G** | **NO — OTHER PROJECT** | n/a | **DO-NOT-TOUCH** | Separate project. Biggest single item. If it is also yours, offloading it is the highest-leverage move (see Proposal). |
| `/data/jehc223/miniconda3` | 59G | shared | n/a | no | All conda envs; `HateVideo` env is 7.2G of this. Shared across projects. |
| `/data/jehc223/SafetyContradiction` | 36G | other project | n/a | DO-NOT-TOUCH | Separate project. |
| `/data/jehc223/ExMRD_ours` | 28G | other project | n/a | DO-NOT-TOUCH | Separate project. |
| `/data/jehc223/Multihateclip` | **27G** | **YES** | no | **PROPOSE** | RAW dataset (En 16G + Zh 12G). Source video for MHC/MHC_zh. Not re-derivable locally. Offload only after feature extraction + user approval. |
| `/data/jehc223/home` | 14G | shared | n/a | no | `$HOME`. `.cache/huggingface` = 6.3G (CLIP weights). |
| `/data/jehc223/NIPS2026` | 14G | other project | n/a | DO-NOT-TOUCH | Separate project. |
| `/data/jehc223/HateMM` | **9.6G** | **YES** | no | **PROPOSE** | RAW dataset (video 6.2G + frames 2.4G + quad 1.1G). Not re-derivable locally. |
| `…/miniconda3/envs/HateVideo` | 7.2G | **YES** (in miniconda3) | no | no | The project conda env. Keep. |
| `…/home/.cache/huggingface` | 6.3G | shared | partial | no | HF model cache — CLIP ViT-L/14-336 is needed by the pipeline. Conservative: do NOT auto-purge models. |
| `/data/jehc223/RGCL` | ~1.5G→ small | **YES** | — | — | The repo. Now ~code only after prune (was 459M; logging+embeddings pruned). |
| `/data/jehc223/RGCL/logging` | ~1.0G → 4K | **YES** | **yes** | **DONE (pruned)** | Derived training checkpoints (`*.pt`). All 51 SHA1-verified on B2 under `logs/…`. Pruned 2026-07-01. |
| `/data/jehc223/RGCL/data/CLIP_Embedding` | 33M → 0 | **YES** | **yes** | **DONE (pruned)** | Derived CLIP frame embeddings. 12 files, 1:1 mirror on B2 `embeddings/…`, `rclone check` clean. Pruned 2026-07-01. |
| `/data/jehc223/RGCL/data/gt` | 5.5M | **YES** | — | no | Ground-truth labels/splits. Code-adjacent, tiny — keep. |
| `/data/jehc223/ImpliHateVid` | 3.6M | **YES** | (video on B2 `b2:junyi-data/ImpliHateVid`) | no | Annotation JSON/splits; raw video already lives on B2, mounted at run time. Tiny locally — keep. |
| `/data/jehc223/RGCL/slurm/logs` | ~216K | **YES** | no | no | SLURM `.out` logs + `disk_guard.log`. Trivial. Keep. |

**Reclaimable-now (ours, verified on B2):** only `logging/` (~1.0G) and
`CLIP_Embedding` (33M) — both already pruned. **Total ≈ 1.0G.** This is why the
safe offload alone cannot reach 250G: the repo's own derived footprint is under
1.5G. Real headroom needs an approved offload (Section 5).

---

## 3. What `disk_guard.sh` does

`scripts/disk_guard.sh` — a quota watchdog meant to run at the **top of every
sbatch job**. Safe to `source` or execute; it is idempotent and **never uses
`set -e`**, so it can never abort the parent job (it always finishes `return 0` /
`exit 0`).

**Config (env vars, all optional):**

| Var | Default | Meaning |
|---|---|---|
| `DISK_GUARD_THRESHOLD_GB` | `250` | Reclaim only when `/data` usage exceeds this. |
| `DISK_GUARD_TARGET_GB` | = threshold | Reclaim until under this. |
| `RGCL_ROOT` | `/data/jehc223/RGCL` | Project root. |
| `B2_PREFIX` | `b2:junyi-data/RGCL_video` | rclone remote + prefix. |
| `DISK_GUARD_LOG` | `$RGCL_ROOT/slurm/logs/disk_guard.log` | Append-only action log (in a *stable* dir, not the prunable `logging/`). |
| `DISK_GUARD_DRY_RUN` / `--dry-run` | off | Simulate: log what it *would* do, write/delete nothing. |
| `DISK_GUARD_HF_PURGE` | `0` | If `1`, allow removing only HF `datasets--*` / `.locks` (never `models--*`). |

**Reclaim order** (re-checks usage after each step, stops once under target):

1. **rclone VFS cache** — `rm -rf ~/.cache/rclone/vfs*`. Pure cache, always safe.
   (The `gen_embed_impli` job spins up a 25G VFS cache; the *next* guard run
   reclaims it if space is tight.)
2. **HF cache** — conservative: **skipped by default** (the run needs CLIP).
   With `DISK_GUARD_HF_PURGE=1` it removes only `datasets--*`/`.locks`, never
   `models--*`.
3. **Push → verify → prune oldest `logging/` checkpoints** — for each `*.pt`
   (oldest mtime first): push to a deterministic `logs/<relpath-under-logging>`
   on B2, verify the **exact** B2 path has a matching SHA1, then delete locally.
   Stops once projected freed space reaches target.
4. **Still over target?** Log a loud `MANUAL INTERVENTION` warning and stop.
   It will **never** touch raw datasets (`Multihateclip`/`HateMM`/`ImpliHateVid`),
   `AlphaSteer`, other projects, or any path outside the allowlist.

**Safety guardrails (all hard-coded):**

- **Deletes are gated on BOTH** (a) SHA1 verified byte-identical on B2 **and**
  (b) canonical path under an allowlist (`logging/`, `data/CLIP_Embedding`,
  `~/.cache/rclone`, `~/.cache/huggingface`).
- **Blocklist:** anything resolving under `/data/jehc223/AlphaSteer` is refused.
- **Raw-dataset refusal:** paths containing `Multihateclip`/`MultiHateClip`/
  `HateMM_raw`/`ImpliHateVid_raw` are refused.
- **Unknown usage → no-op:** if `quota -s` can't be parsed, nothing destructive runs.

**Dry-run:** `bash scripts/disk_guard.sh --dry-run` (or `DISK_GUARD_DRY_RUN=1`).
Verified 2026-07-01: at 282G it correctly identifies candidates and prints the
exact push/verify/prune plan without touching anything; with a high threshold it
is a clean no-op.

---

## 4. Push-then-prune convention (per-sbatch)

**One-line change already applied** to all three jobs, right after
`conda activate HateVideo`:

```bash
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true
```

Run as a subprocess (not sourced) so it cannot mutate the job's shell options or
abort it. Applied to `scripts/slurm/{gen_embed,gen_embed_impli,train}.sbatch`.

**Artifact push already in place (confirmed):**

- `gen_embed.sbatch` → `b2_push.sh data/CLIP_Embedding/$DATASET embeddings/$DATASET`
- `gen_embed_impli.sbatch` → `b2_push.sh data/CLIP_Embedding/ImpliHateVid embeddings/ImpliHateVid`
- `train.sbatch` → `b2_push.sh logging logs/$DATASET`

**Pruning is handled by the guard, not inline.** Jobs push (copy, keep local);
the *next* job's `disk_guard.sh` prunes verified artifacts when over threshold.
This is deliberately safer than an inline `--move`: prune only happens after an
independent SHA1 verify, and only when we actually need the space. If you want an
individual job to prune its own output immediately after a verified push, use
`b2_push.sh <local> <subpath> --move` **only** for embeddings (1:1 layout);
avoid `--move` for `logging/` because the on-B2 layout is nested and differs from
local (see Section 6 caveat).

---

## 5. PROPOSAL — bigger offloads that need your approval

Safe automated steps top out around ~1G because the repo's derived footprint is
tiny. To build a real buffer, one of these needs your OK. **None have been done.**

Restore for all: `scripts/b2_pull.sh <subpath> <local_path>` (or `rclone copy`).

| # | Item | Size | B2 destination | How to restore | Risk / precondition |
|---|---|---|---|---|---|
| **P1** | **`AlphaSteer` (if it is also yours)** | **93G** | e.g. `b2:junyi-data/AlphaSteer/` | `rclone copy b2:junyi-data/AlphaSteer /data/jehc223/AlphaSteer` | **Highest-leverage single move.** Only if you confirm it is your project and inactive. NOT touched by any automation. |
| **P2** | **`Multihateclip` raw video** (En 16G + Zh 12G) | **27G** | `b2:junyi-data/RGCL_video/raw/Multihateclip/` | `scripts/b2_pull.sh raw/Multihateclip /data/jehc223/Multihateclip` | Offload **after** CLIP embeddings are extracted + pushed (they already are). Re-extraction needs the raw video, so keep on B2. Push-verify-then-delete. |
| **P3** | **`HateMM` raw** (video 6.2G + frames 2.4G + quad 1.1G) | **9.6G** | `b2:junyi-data/RGCL_video/raw/HateMM/` | `scripts/b2_pull.sh raw/HateMM /data/jehc223/HateMM` | Same as P2. Frames (2.4G) are re-derivable from video; could offload frames first as a smaller step. |
| **P4** | HF cache trim | up to ~6G | (re-download from HF hub on demand) | auto on next run if online, or `huggingface-cli download` | Only unused models; keep CLIP ViT-L/14-336. Needs online access to restore, so lower priority. |

**Recommended order:** P1 (if yours) → P2 → P3. P1 alone takes us from 282G to
~189G — a comfortable buffer with room for large model weights. P2+P3 together
free ~37G (282G → ~245G) without touching the other project.

Before any P-item: `rclone copy <local> <B2 dest>`, then
`rclone check <local> <B2 dest>` (must be 0 differences), then delete. This is
the same push-verify-prune contract the guard enforces.

---

## 6. Caveats worth remembering

- **B2 `logs/` layout ≠ local.** On B2, checkpoints sit under
  `logs/<top-dataset>/Retrieval/<dataset>/…` (nested, sometimes duplicated across
  top-level dataset dirs). Local is `logging/Retrieval/<dataset>/…`. A naive
  `rclone check logging/ …/logs/` reports "differences" that are **path
  mismatches, not missing data**. Verify by **SHA1**, not by path, and restore
  from the nested B2 path. `disk_guard.sh` handles this by pushing to a
  deterministic `logs/<relpath>` and verifying that exact path.
- **`gen_embed_impli` VFS cache (25G).** Bounded by `--vfs-cache-max-size 25G`
  under `~/.cache/rclone`. Guard step (a) reclaims it on the next run.
- **HF cache is shared and conservative.** Never auto-delete `models--*`.

---

## 7. Qwen2.5-VL headroom check (MLLM upgrade)

The MLLM feasibility work wants a vision-language model. Weight footprint vs the
buffer this policy currently creates:

| Model | Weights | Headroom needed (~weights + activations/scratch) | Fits current plan? |
|---|---|---|---|
| **Qwen2.5-VL-3B** | ~7G | ~10G | **Yes** — fits after the safe prune (≈8G free at 282G is marginal; one small offload, e.g. P3 frames or P4, makes it comfortable). |
| **Qwen2.5-VL-7B** | ~15G | **~20G** | **No, not yet.** The current safe plan leaves ~8G free. Needs an **approved offload**: P1 (93G, best) or P2/P3 (≈37G) creates the ~20G headroom. |

**Bottom line:** the automated per-sbatch plan keeps us lean but does **not** by
itself create the ~20G needed for Qwen2.5-VL-7B (~15G weights). A **3B model
(~7G) fits** the current buffer. To run **7B**, approve **P1** (highest leverage,
if AlphaSteer is yours) or **P2+P3** (raw datasets, ~37G, no other-project
involvement). After P1, ~189G used → ~100G+ free — 7B fits easily.

---

## 8. Executed offloads

- **P1 — AlphaSteer (~93G)** offloaded to `b2:junyi-data/AlphaSteer/` on 2026-07-01; verified byte-identical (`rclone check` = 0 differences, 10447 matching files, source/dest sizes 10447 objects / 99535572675 bytes match), local `/data/jehc223/AlphaSteer` deleted, reclaimed ~93G (quota 282G → 189G). Restore: `rclone copy b2:junyi-data/AlphaSteer/ /data/jehc223/AlphaSteer`.
