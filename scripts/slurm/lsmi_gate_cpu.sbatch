#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=lsmi_gate_cpu
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time: project policy.
#
# LSMI $0 GATE -- CPU-ONLY. NO --gres LINE => ZERO GPU REQUESTED, ZERO GPU CONSUMED.
# Moved off the login node only because the login-node reaper SIGTERMs any non-SLURM
# process past ~2 min of CPU (observed: 12 x exit=143), which no per-cell checkpoint can
# survive for the raw d=3584 KNIFE fits. Reads banked feature caches + labels read-only,
# train+dev only; no test split, no state/ mutation, no network.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

echo "=== LSMI gate CPU job $SLURM_JOB_ID on $(hostname) started $(date '+%F %T') ==="
nvidia-smi -L 2>/dev/null | head -1 || echo "(no GPU visible -- as intended)"

for STAGE in gates main raw; do
  echo "===== STAGE $STAGE $(date '+%F %T') ====="
  python -u scripts/analysis/lsmi_gate.py --stage "$STAGE" 2>&1 \
    | tee -a "refine-logs/LSMI_GATE_run_${STAGE}.log"
  echo "===== STAGE $STAGE done rc=$? $(date '+%F %T') ====="
done

python -u scripts/analysis/lsmi_gate.py --stage merge 2>&1 | tee -a refine-logs/LSMI_GATE_run_merge.log
echo "=== LSMI gate CPU job $SLURM_JOB_ID finished $(date '+%F %T') ==="
