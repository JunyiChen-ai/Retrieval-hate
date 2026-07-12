#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=lbscgp_g0_cpu
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time: project policy.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
bash scripts/disk_guard.sh || true

TASK=${TASK:?set TASK=freeze|synthetic|decide}
RUN_ID=${RUN_ID:?set frozen RUN_ID}
CONFIG=${CONFIG:-configs/lb_scgp/lb_scgp_v1.json}

# Round3 contract: freeze/decision reject subclip artifacts, segment objectives,
# and quarantine/mixed/protected formal input surfaces.
case "$TASK" in
  freeze|synthetic)
    python scripts/analysis/lb_scgp_g0.py --config "$CONFIG" \
      --task "$TASK" --run-id "$RUN_ID"
    ;;
  decide)
    python scripts/analysis/lb_scgp_independent_verify.py --config "$CONFIG" \
      --task decide --run-id "$RUN_ID"
    ;;
  *)
    echo "Refusing unauthorized task: $TASK" >&2
    exit 2
    ;;
esac
