#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=p9b2ad
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: no --time, no GPU. Back up P9 LoRA adapters + classifier heads to B2
# (rule: adapters -> B2, not git). Copy (keep local); disk_guard handles .pt separately.
set -euo pipefail
cd /data/jehc223/RGCL
echo "[p9b2ad] pushing logging/lora_p9 adapters to B2"
bash scripts/b2_push.sh logging/lora_p9 adapters/lora_p9
echo "[p9b2ad] DONE"
