#!/bin/bash
#SBATCH --job-name=v21-mm-val
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/v21-mm-%j.out
set -euo pipefail
source ~/.bashrc
conda activate HateVideo
cd /home/jehc223/Hate-follow-up
python scripts/reproduction_baselines/relation_v21/protocol.py --corpus "$CORPUS" --out-dir "$OUT_DIR" --max-videos "${MAX_VIDEOS:-12}"
python scripts/reproduction_baselines/relation_v21/forward_sequential.py --out-dir "$OUT_DIR"
python scripts/reproduction_baselines/relation_v21/analyze.py --out-dir "$OUT_DIR"
