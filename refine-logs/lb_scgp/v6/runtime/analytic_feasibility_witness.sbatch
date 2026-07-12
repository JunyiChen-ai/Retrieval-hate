#!/usr/bin/env bash
#SBATCH --job-name=lbscgp_v6_analytic_witness
#SBATCH --output=refine-logs/lb_scgp/v6/logs/%x_%j.out
#SBATCH --error=refine-logs/lb_scgp/v6/logs/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail
cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
python refine-logs/lb_scgp/v6/runtime/analytic_feasibility_witness.py
ANALYTIC_WITNESS_PATH="refine-logs/lb_scgp/v6/results/analytic_feasibility_witness_${SLURM_JOB_ID}.json" \
  python refine-logs/lb_scgp/v6/runtime/analytic_witness_replay.py
