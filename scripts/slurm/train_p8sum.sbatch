#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=p8sum
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (job auto-releases from JobHeldUser).
#
# P8 training (research-wiki/EXP_p8_semantic_compression.md). Drop-in --model text-channel
# swap; conditions distinguished by --exp_comment (--model is NOT in exp_name):
#   A floor  : --model openai_clip-vit-large-patch14-336_HF  (raw chunk-mean; reproduces floor)
#   B ours   : --model p8sum_HF     (MLLM ≤60-word summary, single-chunk)
#   C rent   : --model p8trunc_HF   (first-70-token raw, single-chunk; no MLLM)
#   D concat : --model p8concat_HF  (concat[l2n(raw)|l2n(summary)], dim 1536)
# Seeds 0,1,2, GROUP RAC_video_p8sum, FORCE=False. Set DATASETS to the gated-open ones.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-64}
GROUP="RAC_video_p8sum"
FORCE=${FORCE:-False}
DATASETS=${DATASETS:-"MHC MHC_zh HateMM"}
CONDS=${CONDS:-"A B C D"}
SEEDS=${SEEDS:-"0 1 2"}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for DATASET in $DATASETS; do
  for COND in $CONDS; do
    case "$COND" in
      A) MODEL="openai_clip-vit-large-patch14-336_HF" ;;
      B) MODEL="p8sum_HF" ;;
      C) MODEL="p8trunc_HF" ;;
      D) MODEL="p8concat_HF" ;;
      *) echo "unknown cond $COND"; exit 1 ;;
    esac
    CMT="_p8${COND}"
    for SEED in $SEEDS; do
      TAG="p8sum_${DATASET}_${COND}_s${SEED}"
      RUNLOG="slurm/logs/${TAG}.trainlog"
      echo "======== TRAIN $TAG (model=$MODEL) -> $RUNLOG ========"
      python ./src/run_rac.py --batch_size "$BATCH" \
          --lr 0.0001 --epochs "$EPOCHS" --topk 20 --dataset "$DATASET" \
          --model "$MODEL" --path ./data/ \
          --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
          --fusion_mode "align" \
          --hard_negatives_loss True --no_hard_negatives 1 \
          --final_eval False --seed "$SEED" --group_name "$GROUP" \
          --metric "cos" --loss "triplet" --batch_norm False \
          --hybrid_loss True \
          --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
          --Faiss_GPU False --force "$FORCE" \
          --exp_comment "$CMT" 2>&1 | tee "$RUNLOG"

      RUNDIR=$(python - "$DATASET" "$GROUP" "$CMT" "$SEED" <<'PY'
import sys, glob
ds, grp, cmt, seed = sys.argv[1:5]
pat = "logging/Retrieval/{}/{}/*seed{}{}/ckpt".format(ds, grp, seed, cmt)
hits = sorted(glob.glob(pat))
print(hits[-1] if hits else "")
PY
)
      if [[ -n "$RUNDIR" ]]; then
          find "$RUNDIR" -name 'epoch_model_*.pt' -delete 2>/dev/null || true
      fi
      echo "-------- done $TAG --------"
    done
  done
done

echo "======== ALL P8 RUNS DONE ========"
python scripts/analysis/p8_collect.py --out scripts/analysis/p8_out/p8_results.json || true
