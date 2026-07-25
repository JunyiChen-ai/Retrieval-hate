#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=lsmi_power_cpu
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time: project policy.
#
# AMD-4 / AMD-5 follow-up to the LSMI $0 gate, then the deferred raw arm A4.
# CPU-ONLY, NO --gres LINE => ZERO GPU REQUESTED, ZERO GPU CONSUMED.
# Read-only on banked caches; train+dev only; no test split; no state/ mutation.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

echo "=== LSMI power job $SLURM_JOB_ID on $(hostname) started $(date '+%F %T') ==="
nvidia-smi -L 2>/dev/null | head -1 || echo "(no GPU visible -- as intended)"

# AMD-5 dimension ladder: XOR at OUR n across d' in {8,16,32}, released 30-epoch budget
python -u scripts/analysis/lsmi_gate_power.py --stage dimladder --dims 8,16,32 --epochs 30 2>&1 \
  | tee -a refine-logs/LSMI_GATE_run_power_dimladder.log

# AMD-4 budget ladder: XOR at OUR n, d'=64, matched gradient-step budget
python -u scripts/analysis/lsmi_gate_power.py --stage budget --epochs 400 2>&1 \
  | tee -a refine-logs/LSMI_GATE_run_power_budget.log

# AMD-5 data read at the certified dimensions (superset; the verdict uses d* only)
python -u scripts/analysis/lsmi_gate_power.py --stage data --dims 8,16,32 --epochs 30 --nperm 50 2>&1 \
  | tee -a refine-logs/LSMI_GATE_run_power_data.log

# deferred F41 raw arm A4 (d=3584), then merge everything
python -u scripts/analysis/lsmi_gate.py --stage raw 2>&1 | tee -a refine-logs/LSMI_GATE_run_raw.log
python -u scripts/analysis/lsmi_gate.py --stage merge 2>&1 | tee -a refine-logs/LSMI_GATE_run_merge.log

echo "=== LSMI power job $SLURM_JOB_ID finished $(date '+%F %T') ==="
