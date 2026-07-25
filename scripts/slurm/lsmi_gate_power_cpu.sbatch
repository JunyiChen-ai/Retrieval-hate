#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=lsmi_power_cpu
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time: project policy.
#
# AMD-4 follow-up to the LSMI $0 gate -- CPU-ONLY, NO --gres LINE => ZERO GPU.
# Disentangles sample size from optimisation budget in the G1 XOR power gate, then (stage
# "data") re-reads the three real datasets on the primary arm under the matched budget.
# Read-only on banked caches; train+dev only; no test split; no state/ mutation.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

EPOCHS=${1:-400}
STAGES=${2:-"power data"}
echo "=== LSMI power job $SLURM_JOB_ID on $(hostname) epochs=$EPOCHS started $(date '+%F %T') ==="
for S in $STAGES; do
  echo "===== POWERSTAGE $S $(date '+%F %T') ====="
  python -u scripts/analysis/lsmi_gate_power.py --stage "$S" --epochs "$EPOCHS" 2>&1 \
    | tee -a "refine-logs/LSMI_GATE_run_power_${S}.log"
done
echo "=== LSMI power job $SLURM_JOB_ID finished $(date '+%F %T') ==="
