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

MODEL=${MODEL:?set MODEL=Qwen/Qwen2.5-VL-32B-Instruct or -72B-}
QUANT=${QUANT:-none}
SPLITS=${SPLITS:-train}
GTDIR=./data/gt_p10hate
ASRTAG_K30=${ASRTAG_K30:-asrK30_whisper-large-v3}
ASRTAG_K4=${ASRTAG_K4:-asrK4_whisper-large-v3}
echo "[p10_ladder] MODEL=$MODEL QUANT=$QUANT SPLITS=$SPLITS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_k () {  # $1=K  $2=asr_tag
  echo "=== scoring K=$1 (asr=$2) ==="
  python scripts/analysis/p10_score_segments.py \
    --dataset HateMM --splits "$SPLITS" --gt_dir "$GTDIR" \
    --num_frames 120 --num_subclips "$1" \
    --asr_tag "$2" --model "$MODEL" --prompt p6 --quant "$QUANT"
}

run_k 30 "$ASRTAG_K30"
run_k 4  "$ASRTAG_K4"
echo "[p10_ladder] done MODEL=$MODEL"
