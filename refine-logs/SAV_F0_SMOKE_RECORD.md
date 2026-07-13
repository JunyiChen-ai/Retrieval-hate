# SAV (C2) F-G0 SMOKE — execution record

**Executor:** SAV smoke executor (separate authorized step; extraction smoke ONLY — the full
chain `sbatch scripts/slurm/sav_f0.sbatch` requires a further authorization after SMOKE_PASS).
**Authority:** `refine-logs/SAV_F0_CODE_REVIEW.md` §"SMOKE PRESCRIPTION" (APPROVED-to-smoke).
**Date:** 2026-07-13 (NZST).

## STATUS: **WAITING** (job held; no verdict yet)

- **Job id:** `13058` (`sav_f0_smoke`), submitted ONCE at `2026-07-13T23:00:51` NZST.
- **State at record time (2026-07-13 23:39:04 NZST):** `PENDING (JobHeldUser)`, Elapsed 00:00:00.
  - `sacct -j 13058 -X`: `13058  sav_f0_smoke  PENDING  0:0  Submit=2026-07-13T23:00:51  Start=Unknown  Elapsed=00:00:00`
  - `squeue`: `PENDING (JobHeldUser)`
- Held ~38 min past submission; per project policy `JobHeldUser` auto-releases and must
  **never** be force-released (can stall hours). Check-in window (~40 min) exhausted →
  recording WAITING per the executor protocol. **No resubmit** (single-submission ceremony);
  the job remains queued and will run when released.
- `artifacts/sav_f0/` does not exist yet — extraction has not started; nothing to verify yet.

## Submission

- Command: `sbatch scripts/slurm/sav_f0_smoke.sbatch` → `Submitted batch job 13058`.
- Pre-flight (verified live before submission):
  - all 6 banked pooled caches present: `data/CLIP_Embedding/<ds>/{train,dev_seen}_Qwen2.5-VL-7B-Instruct_HF.pt`
    (HateMM 21M/3.0M, MHC 16M/2.2M, MHC_zh 16M/2.2M);
  - `artifacts/sav_f0/extract/` absent (clean start);
  - first-2 ids per split (the smoke set) exclude the undecodable `hate_video_95`:
    HateMM train `[hate_video_98, non_hate_video_80]`, val `[non_hate_video_58, non_hate_video_190]`;
    MHC train `[4V0KGql_fUI, 5snzFreG79c]`, val `[fuUtL4mbTDU, ZZOSLjm0LqE]`;
    MHC_zh train `[BV1f8411b7Xz, BV1em4y1B7bQ]`, val `[BV1qV411w7t1, BV1cp421o7gr]`
    → all 12 smoke videos expected decodable (`ok=true`).
- **Output-location note (CLI checked before writing the sbatch):** `sav_f0_extract.py`
  accepts only `--datasets/--splits/--device/--limit` — no RUN_ID / output-redirect flag —
  so the smoke writes into the SAME cache dir the full run uses
  (`artifacts/sav_f0/extract/<ds>/<split>/<id>.pt`). Confirmed resume-compatible: the
  extractor is per-video resumable (skip-if-exists, `sav_f0_extract.py:306-309`) with atomic
  same-dir writes (`sav_f0_common.atomic_torch_save`), per `SAV_F0_IMPL_NOTES.md`
  ("per-video cache (skip-if-exists resume)"; "Fully resumable"). The full run will skip the
  12 smoke caches (warm start) and REWRITE each split `_manifest.json` at full count, so the
  smoke's `n=2` manifests are transient and non-blocking.

## sbatch content (`scripts/slurm/sav_f0_smoke.sbatch`, new file)

```bash
#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=sav_f0_smoke
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
#SBATCH --error=/data/jehc223/RGCL/slurm/logs/%x_%j.err
# NOTE: intentionally NO --time (project policy). 1 GPU (frozen forward only) + 8 CPU.
#
# SAV (C2) F-G0 SMOKE — exercises the REAL GPU entry point on a tiny real subset per the
# smoke prescription in refine-logs/SAV_F0_CODE_REVIEW.md. This is EXTRACTION ONLY:
#   python scripts/analysis/sav_f0_extract.py --datasets HateMM,MHC,MHC_zh --splits train,val --limit 2
# The guard/probe are NOT smoked (they assert the full EXPECTED_COUNTS by design).
#
# Output location: the extractor CLI supports only --datasets/--splits/--device/--limit; it has
# NO RUN_ID or output-redirect flag, so the 2-per-split caches are written to the SAME dir the
# full run uses (artifacts/sav_f0/extract/<ds>/<split>/<id>.pt). This is warm-start compatible:
# the extractor is per-video resumable (skip-if-exists) with atomic same-dir writes
# (SAV_F0_IMPL_NOTES.md "per-video cache (skip-if-exists resume)"; "Fully resumable"). The full
# run will skip these pre-existing per-video caches and REWRITE the split _manifest.json with the
# complete full-count manifest, so the smoke's n=2 manifest is transient and non-blocking.
# HF offline env matches scripts/slurm/sav_f0.sbatch.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "########## [SAV F-G0 SMOKE] extraction (datasets=HateMM,MHC,MHC_zh limit=2) ##########"
python scripts/analysis/sav_f0_extract.py --datasets HateMM,MHC,MHC_zh --splits train,val --limit 2

echo "======== sav_f0 SMOKE extraction DONE (${SLURM_JOB_ID:-nojob}) ========"
```

## Pass criteria to verify on COMPLETED (from the prescription; not yet evaluable)

1. stdout: geometry cross-check OK + `registered 28 o_proj hooks`.
2. 12 per-video `.pt` (2×2×3) with keys `id,label,ok,img_pooled[3584],text_pooled[3584],
   img_hidden_final[3584],img_head_final[28,28,128],img_head_spanmean[28,28,128],meta`;
   `img_pooled` unit-norm; 6 `_manifest.json` with `complete==true`, `n==2`.
3. PRIMARY-guard preview: ≥1 decodable id per dataset with
   `cos(fresh img_pooled, cached img_feats) ≥ 0.999` AND
   `cos(fresh text_pooled, cached text_feats) ≥ 0.999` vs the banked enc3s caches
   (loaded via `sav_f0_common.load_cached_pooled`, the same path `sav_f0_guard.py` uses).
4. `img_head_final` finite, non-trivial per-head L2-norm spread.

A verifier implementing 2-4 is staged at
`<scratchpad>/verify_smoke.py` (CPU-only, imports `sav_f0_common`).

## Verdict

**NONE YET — WAITING on `JobHeldUser` auto-release of job 13058.** This record will be
updated with the sacct terminal line, all four measured criteria (with provenance), and
SMOKE_PASS / SMOKE_FAIL once the job runs to a terminal state.
