#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=sq_s1_gpu
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time.
set -euo pipefail
cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
RUN_ID=${RUN_ID:?set RUN_ID}; DATASET=${DATASET:?set DATASET}; OUTER_FOLD=${OUTER_FOLD:?set OUTER_FOLD}
python scripts/analysis/sq_s1.py --config "${CONFIG:-configs/sq/sq_v1.json}" \
  --task outer --dataset "$DATASET" --outer-fold "$OUTER_FOLD" --run-id "$RUN_ID"
