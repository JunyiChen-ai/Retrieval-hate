#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=b5probeC
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# AUTHORIZED G-REPRO FALLBACK (review sec5 / prereg sec6.3 / design sec6):
# the CPU probe (job 13156) reproduced ALL acc/macroF1 anchors to 4dp but roc mismatched at
# 4dp on 5 Qwen slots (float-eps of the cuda-trained head forward on 3584-dim embeddings).
# This is the single pre-authorized device='cuda', Faiss_GPU=False eval -- the head forward runs
# on the same compute path as 13115, so the continuous votes (and roc) reproduce bit-for-bit.
# ONE submission only. Faiss stays CPU (Faiss_GPU=False). Not the formal single-submit stage.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export B5_PROBE_DEVICE=cuda    # head forward on cuda; script keeps Faiss_GPU=False

echo "host=$(hostname) jobid=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ) device=cuda"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
python ./scripts/analysis/b5_conv_probe.py
echo "end=$(date -u +%FT%TZ) exit=$?"
