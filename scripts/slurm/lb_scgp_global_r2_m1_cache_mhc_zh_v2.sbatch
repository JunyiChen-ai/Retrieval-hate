#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=lbscgp_global_r2_m1_cache_mhc_zh_v2
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
#SBATCH --error=/data/jehc223/RGCL/slurm/logs/%x_%j.err
# Intentionally no --time: project policy. m1_cache_parallel_max2: two concurrent
# 1-GPU cache jobs (this + MHC) stay within the 2-GPU cap.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

RUN_ID=LBSCGP-GLOBAL-M1-CACHE-MHC_zh-v2
CONFIG=configs/lb_scgp_global_r2/m1_cache_mhc_zh_v2.json
DATASET=MHC_zh

RUN_ID="$RUN_ID" CONFIG="$CONFIG" DATASET="$DATASET" bash scripts/wrappers/lb_scgp_global_r2_m1_cache_v2.sh
