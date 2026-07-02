#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=clip_heads
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# Trains the 4 frozen-CLIP-RGCL heads (one per video dataset) and SAVES them to
# a DISTINCT group dir (RAC_video_CLIP) so they do NOT collide with / get
# overwritten by the Qwen-MLLM RAC_video checkpoints already on disk.
#
# Config is identical to the Phase-0 CLIP baseline (align fusion, triplet +
# hybrid BCE, topk=20, arithmetic vote, 30 ep, seed 0, faiss CPU).  This is the
# exact head whose fused-embedding space we test for cross-dataset transfer.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

MODEL="openai_clip-vit-large-patch14-336_HF"
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-64}
GROUP="RAC_video_CLIP"

for DATASET in HateMM MHC MHC_zh ImpliHateVid; do
    RUNLOG="slurm/logs/clip_head_${DATASET}_$$.trainlog"
    echo "======== TRAIN CLIP head: $DATASET -> $RUNLOG ========"
    python ./src/run_rac.py --batch_size "$BATCH" \
        --lr 0.0001 --epochs "$EPOCHS" --topk 20 --dataset "$DATASET" \
        --model "$MODEL" \
        --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
        --fusion_mode "align" \
        --hard_negatives_loss True --no_hard_negatives 1 \
        --final_eval False --seed 0 --group_name "$GROUP" \
        --metric "cos" --loss "triplet" --batch_norm False \
        --hybrid_loss True \
        --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
        --Faiss_GPU False --force True 2>&1 | tee "$RUNLOG"

    echo ""
    echo "-------- val-selected best ckpt ($DATASET) --------"
    python - "$DATASET" "$GROUP" <<'PY'
import sys, glob, os, re
ds, grp = sys.argv[1], sys.argv[2]
WARM = 5
fs = glob.glob(f"logging/Retrieval/{ds}/{grp}/*/ckpt/best_model_*.pt")
recs = []
for f in fs:
    m = re.search(r"best_model_(\d+)_([\d.]+)\.pt$", os.path.basename(f))
    if m:
        recs.append((float(m.group(2)), int(m.group(1)), f))
elig = [r for r in recs if r[1] >= WARM] or recs
elig.sort(key=lambda r: (r[0], r[1]))
if elig:
    best = elig[-1]
    print(f"BEST_CKPT\t{ds}\tep={best[1]}\tvalacc={best[0]:.4f}\t{best[2]}")
else:
    print(f"NO CKPT for {ds}")
PY
done

echo "======== ALL CLIP HEADS TRAINED ========"
for DATASET in HateMM MHC MHC_zh ImpliHateVid; do
    python - "$DATASET" "$GROUP" <<'PY'
import sys, glob, os, re
ds, grp = sys.argv[1], sys.argv[2]
WARM = 5
fs = glob.glob(f"logging/Retrieval/{ds}/{grp}/*/ckpt/best_model_*.pt")
recs = []
for f in fs:
    m = re.search(r"best_model_(\d+)_([\d.]+)\.pt$", os.path.basename(f))
    if m:
        recs.append((float(m.group(2)), int(m.group(1)), f))
elig = [r for r in recs if r[1] >= WARM] or recs
elig.sort(key=lambda r: (r[0], r[1]))
if elig:
    print(f"CKPT\t{ds}\t{elig[-1][2]}")
PY
done
