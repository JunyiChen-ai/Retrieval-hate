#!/usr/bin/env bash
#SBATCH --job-name=relation-v9-hatemm
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/reproduction/relation_v9/hatemm-pilot-%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONPATH="$PWD/scripts/reproduction_baselines:$PWD/scripts/duplex"
manifest=results/reproduction/relation_v9/manifests/hatemm_macil_vera.json
python scripts/reproduction_baselines/relation_v9/hatemm_preflight.py --manifest "$manifest" --require-complete
python scripts/reproduction_baselines/relation_v9/synthetic_tests.py
python scripts/reproduction_baselines/relation_v9/train.py --manifest "$manifest" \
  --out-dir results/reproduction/relation_v9/hatemm_seed234/train --device cuda --seed 234
python scripts/reproduction_baselines/relation_v9/infer_eval.py --manifest "$manifest" \
  --checkpoint-dir results/reproduction/relation_v9/hatemm_seed234/train \
  --out-dir results/reproduction/relation_v9/hatemm_seed234/test --device cuda
