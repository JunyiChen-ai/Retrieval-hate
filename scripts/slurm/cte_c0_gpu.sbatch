#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=cte_c0_gpu
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time: project policy.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled

PHASE=${1:?usage: cte_c0_gpu.sbatch micro DATASET RUN_ID}
DATASET=${2:?usage: cte_c0_gpu.sbatch micro DATASET RUN_ID}
RUN_ID=${3:?usage: cte_c0_gpu.sbatch micro DATASET RUN_ID}
test "$#" -eq 3
test "$PHASE" = micro
test "$DATASET" = MHC -o "$DATASET" = MHC_zh
python scripts/analysis/cte_c0.py --config configs/cte/cte_v1.json \
  --phase "$PHASE" --dataset "$DATASET" --run-id "$RUN_ID"
