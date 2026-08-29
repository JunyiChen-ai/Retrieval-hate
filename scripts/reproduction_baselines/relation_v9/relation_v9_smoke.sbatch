#!/usr/bin/env bash
#SBATCH --job-name=relation-v9-smoke
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=results/reproduction/relation_v9/slurm-%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONPATH="$PWD/scripts/reproduction_baselines"
python scripts/reproduction_baselines/relation_v9/synthetic_tests.py
