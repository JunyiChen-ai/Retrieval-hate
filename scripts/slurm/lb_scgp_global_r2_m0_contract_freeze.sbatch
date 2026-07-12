#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=lbscgp_global_r2_run1
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
#SBATCH --error=/data/jehc223/RGCL/slurm/logs/%x_%j.err
# Intentionally no --time: project policy.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

RUN_ID=LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1
CONFIG=configs/lb_scgp_global_r2/m0_contract_freeze_v1.json

RUN_ID="$RUN_ID" CONFIG="$CONFIG" bash scripts/wrappers/lb_scgp_global_r2_run1.sh
