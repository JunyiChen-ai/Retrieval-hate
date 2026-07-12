#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=ssr_b1_gate
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time: project policy.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
bash scripts/disk_guard.sh || true

TASK=${TASK:?set TASK=conditional|shuffle}
RUN_ID=${RUN_ID:?set RUN_ID}
DATASET=${DATASET:?set DATASET}
FAMILY=${FAMILY:-}
CONFIG=${CONFIG:-configs/ssr/ssr_v1.yaml}
case "$TASK" in
  conditional)
    python scripts/analysis/ssr_b1_gate.py --config "$CONFIG" --gate conditional \
      --dataset "$DATASET" --family "$FAMILY" --run-id "$RUN_ID"
    ;;
  shuffle)
    python scripts/analysis/ssr_exact_shuffle.py --config "$CONFIG" \
      --universe oof --dataset "$DATASET" --run-id "$RUN_ID"
    ;;
  *) echo "unknown TASK=$TASK" >&2; exit 2 ;;
esac
