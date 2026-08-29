#!/usr/bin/env bash
#SBATCH --job-name=v9-hatemm-preflight
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/reproduction/relation_v9/hatemm-preflight-%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONPATH="$PWD/scripts/reproduction_baselines"
python scripts/reproduction_baselines/relation_v9/test_hatemm_label_free_evidence.py
python scripts/reproduction_baselines/relation_v9/hatemm_preflight.py --require-complete
