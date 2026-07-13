#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=c3nt_gen
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (per project policy).
#
# C3 NON-TARGET content pilot: generate dense world-knowledge/implicit-reasoning text
# per video AND embed it via the frozen text pathway (one job does generate -> embed).
# Pre-registration: refine-logs/C3_NONTARGET_PILOT_DESIGN.md.
# 1 GPU only; queue nothing else (the user's 2nd GPU slot). Single submission.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[c3nt_gen] host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/analysis/c3_nontarget_gen.py --datasets HateMM,MHC --device cuda

echo "[c3nt_gen] done=$(date -Is)"
