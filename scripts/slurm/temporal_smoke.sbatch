#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=tmp_smoke
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# W4 SMOKE: 3-epoch head training on MHC_temporal via the runtime-injection
# wrapper (src/train_temporal_head.py; run_rac.main reused verbatim, no src/
# edits), then a reduced evolving-memory eval (k=5,10; 2 seeds) to validate
# the full protocol path before the 30-epoch wave.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

MODEL="openai_clip-vit-large-patch14-336_HF"
GROUP="RAC_video_temporal_smoke"
DS="MHC_temporal"

echo "======== SMOKE TRAIN (3 ep): $DS ========"
python ./src/train_temporal_head.py --batch_size 64 \
    --lr 0.0001 --epochs 3 --topk 20 --dataset "$DS" \
    --model "$MODEL" \
    --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
    --fusion_mode "align" \
    --hard_negatives_loss True --no_hard_negatives 1 \
    --final_eval False --seed 0 --group_name "$GROUP" \
    --metric "cos" --loss "triplet" --batch_norm False \
    --hybrid_loss True \
    --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
    --Faiss_GPU False --force False

CKPT=$(python - "$DS" "$GROUP" <<'PY'
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
print(elig[-1][2] if elig else "")
PY
)
echo "[smoke] ckpt=$CKPT"
test -n "$CKPT"

mkdir -p logging/temporal_memory
echo "======== SMOKE EVAL: evolving-memory protocol ========"
python ./src/eval_temporal_memory.py --dataset "$DS" --ckpt "$CKPT" \
    --model "$MODEL" --topk 20 --augment True \
    --k_list 5,10 --strategies random,latest,uncertain --n_seeds 2 \
    --out logging/temporal_memory/smoke_${DS}.json

echo "======== TEMPORAL SMOKE DONE ========"
