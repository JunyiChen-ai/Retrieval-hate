#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=cte_c0_cpu
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time: project policy.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled

PHASE=${1:?usage: cte_c0_cpu.sbatch sanity|freeze|decide [RUN_ID]}
RUN_ID=${2:-}
case "$PHASE" in
  sanity)
    test "$#" -eq 1
    test -z "$RUN_ID"
    python -m py_compile scripts/analysis/cte_common.py \
      scripts/analysis/cte_c0.py scripts/analysis/cte_c1.py
    bash -n scripts/slurm/cte_c0_gpu.sbatch scripts/slurm/cte_c0_cpu.sbatch \
      scripts/slurm/cte_c1_gpu.sbatch scripts/slurm/cte_c1_cpu.sbatch
    python scripts/analysis/cte_c0.py --config configs/cte/cte_v1.json \
      --phase sanity
    ;;
  freeze|decide)
    test "$#" -eq 2
    test -n "$RUN_ID"
    python scripts/analysis/cte_c0.py --config configs/cte/cte_v1.json \
      --phase "$PHASE" --run-id "$RUN_ID"
    ;;
  *)
    echo "unknown C0 CPU phase: $PHASE" >&2
    exit 2
    ;;
esac
