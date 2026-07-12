#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=lbscgp_global_r2_run2_v2
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
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

RUN_ID=LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v2
CONFIG=configs/lb_scgp_global_r2/m0_synth_kkt_v2.json

RUN_ID="$RUN_ID" CONFIG="$CONFIG" bash scripts/wrappers/lb_scgp_global_r2_run2_v2.sh
