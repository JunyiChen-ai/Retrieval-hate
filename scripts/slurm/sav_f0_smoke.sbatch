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
