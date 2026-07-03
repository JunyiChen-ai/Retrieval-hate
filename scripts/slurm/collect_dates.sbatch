#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=collect_dates
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit and NO --gres (CPU + network only).
#
# DESIGN_iter3 E0a: collect MultiHateClip upload dates.
#   usage: sbatch scripts/slurm/collect_dates.sbatch [en|zh|en,zh]
# Checkpoint-resumable: safe to resubmit; already-queried ids are skipped.
set -euo pipefail

DATASETS="${1:-en,zh}"

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export PYTHONUNBUFFERED=1

python scripts/collect_upload_dates.py --datasets "$DATASETS"

echo "======== COLLECT_DATES ($DATASETS) DONE ========"
