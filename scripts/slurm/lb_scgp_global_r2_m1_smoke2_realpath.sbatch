#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=m1_smoke2_realpath
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
#SBATCH --error=/data/jehc223/RGCL/slurm/logs/%x_%j.err
# Intentionally no --time: project policy.
# NON-LINEAGE v2 real-path smoke (M1_CACHE_V1_RESULT_TO_CLAIM_REVIEW.md §4.4): runs the FROZEN
# build_dataset_packs on real symlinked HateMM mp4 (the v1 burn surface) + decode + R=4 cert.
# Output only slurm/tmp/, out of the single-submit budget, zero MHC / lineage-artifact contact.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

python scripts/analysis/lb_scgp_global_r2_m1_smoke2_realpath.py --infer_n 5
