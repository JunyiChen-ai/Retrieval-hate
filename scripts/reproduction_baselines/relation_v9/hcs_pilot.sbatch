#!/usr/bin/env bash
#SBATCH --job-name=relation-v9-hcs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/reproduction/relation_v9/hcs-pilot-%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONPATH="$PWD/scripts/reproduction_baselines"
manifest=results/reproduction/relation_v9/manifests/hateclipseg_vad_fed.json
train_out=results/reproduction/relation_v9/hcs_seed234/train
eval_out=results/reproduction/relation_v9/hcs_seed234/test
python scripts/reproduction_baselines/relation_v9/synthetic_tests.py
python scripts/reproduction_baselines/relation_v9/train.py --manifest "$manifest" --out-dir "$train_out" --device cuda --seed 234
python scripts/reproduction_baselines/relation_v9/infer_eval.py --manifest "$manifest" --checkpoint-dir "$train_out" --out-dir "$eval_out" --device cuda
