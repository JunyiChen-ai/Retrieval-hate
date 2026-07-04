#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=hcs_subclip
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# HateClipSeg zero-training localization eval — sub-clip CLIP cache extraction.
# Reuses src/utils/generate_subclip_embedding_HF.py UNMODIFIED (videos exposed
# via data/video/HateClipSeg/All/<vid>.mp4 symlinks, gt = dummy-label test.jsonl
# written by scripts/analysis/hateclipseg_prep.py).
#
# Two evaluation granularities (both reported):
#   K=4  M=16  : the method's default detection granularity (median window 60s)
#   K=30 M=120 : density-matched to the gold annotation (median video 240s /
#                30 windows = 8.0s per window ~= median gold segment 8.12s)
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

for CFG in "4 16" "30 120"; do
    read -r K M <<< "$CFG"
    OUTF="data/CLIP_Embedding/HateClipSeg/test_seen_subclipK${K}_openai_clip-vit-large-patch14-336_HF.pt"
    if [[ -e "$OUTF" ]]; then
        echo "[hcs_subclip] $OUTF exists -- skipping (FORCE not supported; delete manually to redo)"
        continue
    fi
    echo "[hcs_subclip] === K=$K M=$M ==="
    python src/utils/generate_subclip_embedding_HF.py \
        --dataset HateClipSeg \
        --splits test \
        --num_subclips "$K" \
        --num_frames "$M" \
        --model openai/clip-vit-large-patch14-336 \
        --batch_size 32 \
        --device cuda
done

echo "[hcs_subclip] done"
