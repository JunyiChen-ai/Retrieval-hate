#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --job-name=lbscgp_global_r2_realbank_resource_v1
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
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

RUN_ID=LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1
CONFIG=configs/lb_scgp_global_r2/m0_realbank_resource_v1.json

RUN_ID="$RUN_ID" CONFIG="$CONFIG" bash scripts/wrappers/lb_scgp_global_r2_realbank_resource_v1.sh
