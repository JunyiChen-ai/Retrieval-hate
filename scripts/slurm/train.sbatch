#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=mhc_train
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# Usage: sbatch scripts/slurm/train.sbatch
#   Env overrides: DATASET (default MHC), EPOCHS (default 30).
set -euo pipefail

cd /data/jehc223/RGCL

# Activate the HateVideo conda environment.
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

# Disk quota watchdog: reclaim space (B2-verified prune) before the run.
# Run as a subprocess so it can never abort this job; it is a no-op below the
# threshold. See scripts/disk_guard.sh and research-wiki/DISK_POLICY.md.
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

# Disable wandb; CLIP model is pre-cached, so stay offline.
export WANDB_MODE=disabled HF_HUB_OFFLINE=1

# Pass-through training config (experiments_video.sh reads these from the env).
export DATASET=${DATASET:-MHC}
export EPOCHS=${EPOCHS:-30}

echo "[train] DATASET=${DATASET} EPOCHS=${EPOCHS}"

# experiments_video.sh may be created/updated in parallel; just invoke it.
bash scripts/experiments_video.sh

# On success, push checkpoints + logs to B2 (copy, keep local).
bash scripts/b2_push.sh logging "logs/$DATASET"

echo "[train] done for DATASET=${DATASET}"
