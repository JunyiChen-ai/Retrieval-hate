#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=tmp_mem
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# W4 FULL WAVE: evolving-memory protocol on the temporal splits
# (DESIGN_iter3 S3).
#   1. Train one frozen-CLIP RGCL head per temporal dataset (30 ep, same
#      config as the RAC_video_CLIP random-split heads; val-selected with
#      warmup>=5) into a NEW group RAC_video_temporal.
#   2. Evolving-memory eval: static memory=temporal-train; augmentation
#      with k in {5,10,20,50} labelled new-period (val) samples under
#      random / latest / uncertain selection; test never enters memory.
#   3. Random-split reference: same protocol (train-only + train+val
#      memory) with the existing RAC_video_CLIP heads on MHC / MHC_zh.
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
GROUP="RAC_video_temporal"
OUTDIR="logging/temporal_memory"
mkdir -p "$OUTDIR"

# Resolve the val-selected best ckpt (warmup>=5) for (dataset, group).
ckpt_for() {
  python - "$1" "$2" <<'PY'
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
}

for DS in MHC_temporal MHC_zh_temporal; do
    echo "======== TRAIN temporal head ($EPOCHS ep): $DS ========"
    python ./src/train_temporal_head.py --batch_size "$BATCH" \
        --lr 0.0001 --epochs "$EPOCHS" --topk 20 --dataset "$DS" \
        --model "$MODEL" \
        --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
        --fusion_mode "align" \
        --hard_negatives_loss True --no_hard_negatives 1 \
        --final_eval False --seed 0 --group_name "$GROUP" \
        --metric "cos" --loss "triplet" --batch_norm False \
        --hybrid_loss True \
        --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
        --Faiss_GPU False --force False

    CKPT=$(ckpt_for "$DS" "$GROUP")
    echo "[ckpt] $DS -> $CKPT"
    test -n "$CKPT"

    echo "======== EVAL evolving memory: $DS ========"
    python ./src/eval_temporal_memory.py --dataset "$DS" --ckpt "$CKPT" \
        --model "$MODEL" --topk 20 --augment True \
        --k_list 5,10,20,50 --strategies random,latest,uncertain \
        --n_seeds 5 \
        --out "$OUTDIR/${DS}_evolving_memory.json"
done

echo "======== RANDOM-SPLIT REFERENCES (same protocol, RAC_video_CLIP heads) ========"
for BASE in MHC MHC_zh; do
    CKPT=$(ckpt_for "$BASE" "RAC_video_CLIP")
    echo "[ckpt] $BASE (random ref) -> $CKPT"
    test -n "$CKPT"
    python ./src/eval_temporal_memory.py --dataset "$BASE" --ckpt "$CKPT" \
        --model "$MODEL" --topk 20 --augment False \
        --out "$OUTDIR/${BASE}_random_ref.json"
done

echo "======== W4 TEMPORAL MEMORY WAVE DONE ========"
