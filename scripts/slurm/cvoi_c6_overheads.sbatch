#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=cvoi_c6_over
#SBATCH --output=logging/slurm/%x_%j.out
set -euo pipefail
cd /home/jehc223/Retrieval-hate
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
python -m scripts.cvoi_acq.cost_overhead_driver \
 --out artifacts/cvoi_acq/premetric-v2/c6-cost-v1/overhead_costs.jsonl
