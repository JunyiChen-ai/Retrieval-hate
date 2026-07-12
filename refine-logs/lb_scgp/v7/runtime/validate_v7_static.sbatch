#!/usr/bin/env bash
#SBATCH --job-name=lbscgp_v7_static_validate
#SBATCH --output=refine-logs/lb_scgp/v7/logs/%x_%j.out
#SBATCH --error=refine-logs/lb_scgp/v7/logs/%x_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
python refine-logs/lb_scgp/v7/runtime/validate_v7_static.py
