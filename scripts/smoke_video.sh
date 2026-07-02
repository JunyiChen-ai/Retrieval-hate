#!/usr/bin/env bash
set -e

cd /data/jehc223/RGCL

export WANDB_MODE=disabled

DATASET=${DATASET:-MHCsmoke}
MODEL=${MODEL:-openai_clip-vit-large-patch14-336_HF}
EPOCHS=${EPOCHS:-2}
BATCH=${BATCH:-8}

echo "Running RGCL video smoke: DATASET=$DATASET MODEL=$MODEL"

python ./src/run_rac.py --batch_size "$BATCH" \
    --lr 0.0001  --epochs "$EPOCHS"  --topk 20 --dataset "$DATASET" \
    --model "$MODEL" \
    --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
    --fusion_mode "align" \
    --hard_negatives_loss True --no_hard_negatives 1 \
    --final_eval False --seed 0 --group_name "RAC_video_smoke" \
    --metric "cos" --loss "triplet" --batch_norm False \
    --hybrid_loss True \
    --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
    --Faiss_GPU False --device cpu --num_workers 0 --force True
