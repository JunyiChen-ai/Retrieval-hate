#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=edcm_a0_cpu
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time: project policy.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
bash scripts/disk_guard.sh || true

TASK=${TASK:?set TASK=sanity|reuse-audit|reachability|decision}
RUN_ID=${RUN_ID:?set RUN_ID}
CONFIG=${CONFIG:-configs/edcm/edcm_v1.json}
DATASET=${DATASET:-}

case "$TASK" in
  sanity)
    python -m py_compile scripts/analysis/edcm_a0.py
    bash -n scripts/slurm/edcm_a0_cpu.sbatch
    python -c 'import sys; sys.path.insert(0, "scripts/analysis"); import edcm_a0; edcm_a0.sanity(sys.argv[1])' "$CONFIG"
    ;;
  reuse-audit)
    test -z "$DATASET"
    python scripts/analysis/edcm_a0.py \
      --config "$CONFIG" --task reuse-audit --run-id "$RUN_ID"
    ;;
  reachability)
    test "$DATASET" = "MHC" -o "$DATASET" = "MHC_zh"
    python scripts/analysis/edcm_a0.py \
      --config "$CONFIG" --task reachability --dataset "$DATASET" \
      --run-id "$RUN_ID"
    ;;
  decision)
    test -z "$DATASET"
    python scripts/analysis/edcm_a0.py \
      --config "$CONFIG" --task decision --run-id "$RUN_ID"
    ;;
  *)
    echo "unknown TASK=$TASK" >&2
    exit 2
    ;;
esac
