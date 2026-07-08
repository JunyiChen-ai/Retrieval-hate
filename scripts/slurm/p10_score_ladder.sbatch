#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=p10_ladder
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: no --time (JobHeldUser auto-release).
# P10-b scale-ladder scorer: score HateMM hateful-train calibration set at K30 (fine)
# then K4 (coarse) with a stronger Qwen2.5-VL scorer, so A-fuse (0.5*K30+0.5*K4) can be
# built on the SAME model (matches the 7B A-fuse recipe). One GPU allocation, two passes.
#   MODEL  Qwen/Qwen2.5-VL-32B-Instruct | Qwen/Qwen2.5-VL-72B-Instruct
#   QUANT  none (32B bf16, fits 80G) | bnb4 (72B nf4, ~40G — P2c-proven)
set -euo pipefail
cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1
# 32B bf16 leaves only ~13G activation headroom on 80G; the K4 pass (30-frame windows)
# needs a ~1.7G single attention alloc -> fragmentation OOM without this (job 12562).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:?set MODEL=Qwen/Qwen2.5-VL-32B-Instruct or -72B-}
QUANT=${QUANT:-none}
SPLITS=${SPLITS:-train}
GTDIR=./data/gt_p10hate
ASRTAG_K30=${ASRTAG_K30:-asrK30_whisper-large-v3}
ASRTAG_K4=${ASRTAG_K4:-asrK4_whisper-large-v3}
echo "[p10_ladder] MODEL=$MODEL QUANT=$QUANT SPLITS=$SPLITS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_k () {  # $1=K  $2=M  $3=asr_tag
  echo "=== scoring K=$1 M=$2 (asr=$3) ==="
  python scripts/analysis/p10_score_segments.py \
    --dataset HateMM --splits "$SPLITS" --gt_dir "$GTDIR" \
    --num_frames "$2" --num_subclips "$1" \
    --asr_tag "$3" --model "$MODEL" --prompt p6 --quant "$QUANT"
}

# K30/M120 and K4/M16 = the EXACT round-1 recipes (P6-anchor fine + P3-default coarse).
# K4 MUST be M=16 (4 frames/window): the 7B train_segscoreK4_qwen.jsonl that round-1
# A-fuse fused was scored at M=16; M=120 (30-frame windows) also OOMs 32B bf16
# (jobs 12562/12568: ~11.5k-token vision seq -> 1.74G SDPA alloc > headroom).
run_k 30 120 "$ASRTAG_K30"
run_k 4  16  "$ASRTAG_K4"
echo "[p10_ladder] done MODEL=$MODEL"
