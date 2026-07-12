#!/usr/bin/env bash
#SBATCH --job-name=lbscgp_v6_cert
#SBATCH --output=refine-logs/lb_scgp/v6/logs/%x_%j.out
#SBATCH --error=refine-logs/lb_scgp/v6/logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail
cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
python refine-logs/lb_scgp/v6/runtime/oriented_certificate.py
