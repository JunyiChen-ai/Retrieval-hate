#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=p11_mmK30
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# P11 probe-gate ambiguity remedy (EXP_p11_weaksup_localization.md §4/§5):
# the K=4 MIL-proxy vs K=30 MLLM comparison is granularity-confounded. Extract
# HateMM TRAIN sub-clip CLIP features at K=30 / M=120 (matching the MLLM K30
# windows) so the MIL-proxy can be built at the SAME granularity as the MLLM
# teacher. Reuses src/utils/generate_subclip_embedding_HF.py UNMODIFIED.
# Writes data/CLIP_Embedding/HateMM/train_subclipK30_openai_clip-...-336_HF.pt
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

OUTF="data/CLIP_Embedding/HateMM/train_subclipK30_openai_clip-vit-large-patch14-336_HF.pt"
if [[ -e "$OUTF" ]]; then
    echo "[p11_mmK30] $OUTF exists -- skipping (delete manually to redo)"
    exit 0
fi

python src/utils/generate_subclip_embedding_HF.py \
    --dataset HateMM \
    --splits train \
    --num_subclips 30 \
    --num_frames 120 \
    --model openai/clip-vit-large-patch14-336 \
    --batch_size 32 \
    --device cuda

echo "[p11_mmK30] done"
