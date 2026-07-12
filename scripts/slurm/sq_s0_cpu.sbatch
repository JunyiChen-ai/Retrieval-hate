#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=sq_s0_cpu
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time.
set -euo pipefail
cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
TASK=${TASK:?set TASK}; RUN_ID=${RUN_ID:?set RUN_ID}
ARGS=(--config "${CONFIG:-configs/sq/sq_v1.json}" --task "$TASK" --run-id "$RUN_ID")
[[ -n "${DATASET:-}" ]] && ARGS+=(--dataset "$DATASET")
[[ -n "${AUDIT_CSV:-}" ]] && ARGS+=(--audit-csv "$AUDIT_CSV")
python scripts/analysis/sq_s0.py "${ARGS[@]}"
