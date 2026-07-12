#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=lbscgp_sanitize
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time: project policy.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
bash scripts/disk_guard.sh || true

TASK=${TASK:?set TASK=build|verify}
CONFIG=${CONFIG:-configs/lb_scgp/lb_scgp_v1.json}
SOURCE_CONFIG=${SOURCE_CONFIG:-configs/lb_scgp/lb_scgp_sanitizer_sources.json}

# Round3 contract: build emits only the train-only whole-video feature cache
# plus sanitized provenance; verify writes a formal decision with no subclip
# artifact, no segment objective, and no quarantine/source locator surface.
case "$TASK" in
  build)
    python scripts/analysis/lb_scgp_sanitize_inputs.py --config "$CONFIG" \
      --source-config "$SOURCE_CONFIG" \
      --task build --run-id LBSCGP-G0-SANITIZE-MHC_zh-F4-v1
    ;;
  verify)
    python scripts/analysis/lb_scgp_verify_sanitizer.py --config "$CONFIG" \
      --task verify --run-id LBSCGP-G0-SANITIZER-VERIFY-MHC_zh-F4-v1
    ;;
  *)
    echo "Refusing unauthorized sanitizer task: $TASK" >&2
    exit 2
    ;;
esac
