#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=ssr_b0_rel
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time: project policy.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
bash scripts/disk_guard.sh || true

RUN_ID=${RUN_ID:?set RUN_ID}
DATASET=${DATASET:?set DATASET=MHC|MHC_zh}
MODE=${MODE:?set MODE=smoke|full}
CONFIG=${CONFIG:-configs/ssr/ssr_v1.yaml}
python scripts/analysis/ssr_extract_relations.py --config "$CONFIG" \
  --dataset "$DATASET" --mode "$MODE" --run-id "$RUN_ID"
