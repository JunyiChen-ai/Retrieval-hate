#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=cte_c1_gpu
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time: project policy.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled

PHASE=${1:?usage: cte_c1_gpu.sbatch inner|outer DATASET OUTER_FOLD RUN_ID; or full-freeze DATASET RUN_ID}
DATASET=${2:?dataset required}
test "$DATASET" = MHC -o "$DATASET" = MHC_zh
case "$PHASE" in
  inner|outer)
    test "$#" -eq 4
    OUTER_FOLD=${3:?outer fold required}
    RUN_ID=${4:?run ID required}
    python scripts/analysis/cte_c1.py --config configs/cte/cte_v1.json \
      --phase "$PHASE" --dataset "$DATASET" --outer-fold "$OUTER_FOLD" \
      --run-id "$RUN_ID"
    ;;
  full-freeze)
    RUN_ID=${3:?run ID required}
    test "$#" -eq 3
    python scripts/analysis/cte_c1.py --config configs/cte/cte_v1.json \
      --phase full-freeze --dataset "$DATASET" --run-id "$RUN_ID"
    ;;
  *)
    echo "unknown C1 GPU phase: $PHASE" >&2
    exit 2
    ;;
esac
