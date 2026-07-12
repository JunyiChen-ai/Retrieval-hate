#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=cte_c1_cpu
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time: project policy.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled

PHASE=${1:?usage: cte_c1_cpu.sbatch select|decide|verify-freeze RUN_ID}
RUN_ID=${2:?run ID required}
test "$#" -eq 2
case "$PHASE" in
  select|decide|verify-freeze)
    python scripts/analysis/cte_c1.py --config configs/cte/cte_v1.json \
      --phase "$PHASE" --run-id "$RUN_ID"
    ;;
  *)
    echo "unknown C1 CPU phase: $PHASE" >&2
    exit 2
    ;;
esac
