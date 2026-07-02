#!/usr/bin/env bash
set -e

cd /data/jehc223/RGCL

export WANDB_MODE=disabled

DATASET=${DATASET:-MHC}
MODEL=${MODEL:-openai_clip-vit-large-patch14-336_HF}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-64}
# Installed faiss in the HateVideo env is CPU-only (faiss.get_num_gpus()==0),
# so dense retrieval runs on CPU. The train set is tiny (~550) so this is instant.
FAISS_GPU=${FAISS_GPU:-False}
WARMUP=${WARMUP:-5}

# Multi-granularity (segment-RGCL) knobs:
#   LAMBDA_SEG=0    -> whole-video baseline (exact no-op).
#   LAMBDA_SEG=0.5  -> +multi-granularity sub-clip retrieval.
LAMBDA_SEG=${LAMBDA_SEG:-0}
# Segment-loss variant: full (default) | driftneg | milmax. Only active when LAMBDA_SEG>0.
SEG_MODE=${SEG_MODE:-full}
NUM_SUBCLIPS=${NUM_SUBCLIPS:-4}
# subclip cache path: 'auto' -> data/CLIP_Embedding/<DS>/train_subclipK<K>_<MODEL>.pt
SUBCLIP_CACHE=${SUBCLIP_CACHE:-auto}
GROUP_NAME=${GROUP_NAME:-RAC_video}

echo "Running RGCL video training: DATASET=$DATASET MODEL=$MODEL FAISS_GPU=$FAISS_GPU LAMBDA_SEG=$LAMBDA_SEG SEG_MODE=$SEG_MODE K=$NUM_SUBCLIPS"

python ./src/run_rac.py --batch_size "$BATCH" \
    --lr 0.0001  --epochs "$EPOCHS"  --topk 20 --dataset "$DATASET" \
    --model "$MODEL" \
    --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
    --fusion_mode "align" \
    --hard_negatives_loss True --no_hard_negatives 1 \
    --final_eval False --seed 0 --group_name "$GROUP_NAME" \
    --metric "cos" --loss "triplet" --batch_norm False \
    --hybrid_loss True --warmup "$WARMUP" \
    --lambda_seg "$LAMBDA_SEG" --seg_mode "$SEG_MODE" --num_subclips "$NUM_SUBCLIPS" --subclip_cache "$SUBCLIP_CACHE" \
    --exp_comment "_seg${LAMBDA_SEG}_${SEG_MODE}" \
    --majority_voting "arithmetic" --no_pseudo_gold_positives 1 --Faiss_GPU "$FAISS_GPU" --force True
