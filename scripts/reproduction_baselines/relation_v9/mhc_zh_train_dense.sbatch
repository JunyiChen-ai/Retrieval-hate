#!/usr/bin/env bash
#SBATCH --job-name=v9-zh-dense
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/reproduction/relation_v9/slurm-zh-%A_%a.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
methods=(fed_wsvad_1client macilsd cmhkf);seeds=(234 2025 3407);method=${methods[$((SLURM_ARRAY_TASK_ID/3))]};seed=${seeds[$((SLURM_ARRAY_TASK_ID%3))]}
python scripts/reproduction_baselines/relation_v9/produce_train_dense.py --method "$method" --corpus mhclip_zh --seed "$seed" --out-dir "results/reproduction/relation_v9/train_dense/$method/mhclip_zh/seed_$seed" --device cuda
