#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=tmp_refs
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# W4 repair job: ONLY the random-split reference evals of the evolving-memory
# protocol (job 12214 completed both temporal trainings + evals, then failed
# here because disk_guard had pruned the RAC_video_CLIP best ckpts to B2;
# they have been restored from B2 to the exact original paths).
# NOTE: deliberately does NOT call disk_guard (it would re-prune the ckpts).
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

MODEL="openai_clip-vit-large-patch14-336_HF"
OUTDIR="logging/temporal_memory"
mkdir -p "$OUTDIR"

EXP="RAC_lr0.0001_Bz64_Ep30_cosSim_triplet_drop[0.2, 0.4, 0.1]_topK20__PseudoGold_positive_1_hard_negative_1_seed0_hybrid_loss"
declare -A CK
CK[MHC]="logging/Retrieval/MHC/RAC_video_CLIP/$EXP/ckpt/best_model_26_0.75.pt"
CK[MHC_zh]="logging/Retrieval/MHC_zh/RAC_video_CLIP/$EXP/ckpt/best_model_29_0.8076923076923077.pt"

for BASE in MHC MHC_zh; do
    CKPT="${CK[$BASE]}"
    echo "[ckpt] $BASE (random ref) -> $CKPT"
    if [ ! -f "$CKPT" ]; then
        echo "[refs] ckpt missing locally (pruned again?); restoring from B2"
        rclone copyto "b2:junyi-data/RGCL_video/logs/${CKPT#logging/}" "$CKPT"
        touch "$CKPT"
    fi
    test -f "$CKPT"
    python ./src/eval_temporal_memory.py --dataset "$BASE" --ckpt "$CKPT" \
        --model "$MODEL" --topk 20 --augment False \
        --out "$OUTDIR/${BASE}_random_ref.json"
done

echo "======== W4 RANDOM-SPLIT REFERENCES DONE ========"
