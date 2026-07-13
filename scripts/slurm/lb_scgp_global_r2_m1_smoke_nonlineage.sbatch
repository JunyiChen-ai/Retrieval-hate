#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=lbscgp_global_r2_m1_smoke
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
#SBATCH --error=/data/jehc223/RGCL/slurm/logs/%x_%j.err
# Intentionally no --time: project policy.
# NON-LINEAGE smoke (M1_CACHE_CODE_REVIEW.md ruling 3): HateMM x10, output only to
# slurm/tmp/, out of the single-submit budget, touches no MHC data and no
# artifacts/lb_scgp_global/v1/m1/. Validates FIX-2 GPU guard + the deferred GPU path.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

python scripts/analysis/lb_scgp_global_r2_m1_smoke_nonlineage.py --n 10
