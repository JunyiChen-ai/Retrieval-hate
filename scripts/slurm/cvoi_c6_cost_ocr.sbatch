#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=cvoi_c6_ocr
#SBATCH --output=logging/slurm/%x_%j.out
set -euo pipefail
cd /home/jehc223/Retrieval-hate
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
python -m scripts.cvoi_acq.cost_driver --type ocr \
 --train-actions artifacts/cvoi_acq/premetric-v2/actions/train_ocr_actions.jsonl \
 --val-actions artifacts/cvoi_acq/premetric-v2/actions/val_ocr_actions.jsonl \
 --out artifacts/cvoi_acq/premetric-v2/c6-cost-v1/ocr_cost_actions.jsonl
