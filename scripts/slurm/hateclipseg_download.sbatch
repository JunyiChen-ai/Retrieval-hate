#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=hcs_download
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit and NO --gres (CPU + network only).
# HateClipSeg full-corpus download (<=480p, 28G hard budget).
# Checkpoint-resumable: safe to resubmit; OK / FAIL_PERM ids are skipped.
set -euo pipefail
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1
python3 /data/jehc223/HateClipSeg/pilot/download_all.py
