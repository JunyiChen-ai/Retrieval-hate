#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=p5gen
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (job auto-releases from JobHeldUser).
#
# P5 counterfactual twins generation (research-wiki/EXP_p5_counterfactual_negs.md):
# Qwen rewrite + self-verdict flip judge + CLIP-encode sanitized text. ~348 anchors
# (EN 168 + ZH 180); ~2 judge + 1-2 rewrite calls each -> ~1000 short generations.
#
# Usage:  smoke : LIMIT=3 sbatch scripts/slurm/p5_generate_twins.sbatch
#         full  : sbatch scripts/slurm/p5_generate_twins.sbatch
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

LIMIT=${LIMIT:-0}
DATASETS=${DATASETS:-MHC,MHC_zh}

echo "[p5gen] DATASETS=$DATASETS LIMIT=$LIMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/analysis/p5_generate_twins.py \
    --datasets "$DATASETS" --limit "$LIMIT" --device cuda

echo "[p5gen] done"
