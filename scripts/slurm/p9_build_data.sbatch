#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=p9data
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit. GPU requested only to get a compute allocation
# (login=compute node reaps non-SLURM CPU jobs); frame extraction itself is CPU/decord.
#
# P9 (research-wiki/EXP_p9_lmm_rgcl_video.md) CPU prep: extract 8-frame JPGs + build the
# query-only ShareGPT base json for the requested datasets/splits (idempotent — skips frames
# already on disk). Both P9 forks (in-context / embedding-RGCL) consume these frames + base.
#
# Usage: DATASETS="MHC MHC_zh" SPLITS="test" sbatch scripts/slurm/p9_build_data.sbatch
#        DATASETS="HateMM" sbatch scripts/slurm/p9_build_data.sbatch     # all splits
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1

DATASETS=${DATASETS:-"MHC MHC_zh HateMM"}
SPLITS=${SPLITS:-"train,val,test"}

for DS in $DATASETS; do
  echo "======== build lora_sft data: $DS (splits=$SPLITS) ========"
  python src/utils/build_lora_sft_data.py --dataset "$DS" --splits "$SPLITS"
done
echo "[p9data] done"
