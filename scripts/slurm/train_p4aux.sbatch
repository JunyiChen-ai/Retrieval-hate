#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=p4aux
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (job auto-releases from JobHeldUser).
#
# P4 archive-field auxiliary distillation (research-wiki/EXP_p4_schema_distill.md).
# Trains the CLIP-space RGCL head with the EXACT RAC_video_CLIP floor recipe, plus a
# small aux-distillation term (lambda_aux). Conditions per dataset:
#   floor : --lambda_aux 0     (must reproduce the RAC_video_CLIP floor bit-for-bit)
#   aux   : --lambda_aux 0.1   (pre-registered, no tuning)
# Seeds 0,1,2. GROUP RAC_video_p4aux, FORCE False by default (never overwrite;
# distinct exp_name per seed and the "_p4aux-l0.1" suffix keep floor/aux separate).
# Both protocols (val-selected + final-epoch) are parsed from the trainlogs by
# scripts/analysis/p4_collect.py.
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
GROUP="RAC_video_p4aux"
FORCE=${FORCE:-False}
DATASETS=${DATASETS:-"MHC MHC_zh"}
LAMBDAS=${LAMBDAS:-"0 0.1"}
SEEDS=${SEEDS:-"0 1 2"}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for DATASET in $DATASETS; do
  for LAM in $LAMBDAS; do
    for SEED in $SEEDS; do
      TAG="p4aux_${DATASET}_l${LAM}_s${SEED}"
      RUNLOG="slurm/logs/${TAG}.trainlog"
      echo "======== TRAIN $TAG -> $RUNLOG ========"
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
          --lambda_aux "$LAM" 2>&1 | tee "$RUNLOG"

      # Housekeeping: keep only best_/last_ ckpts (per-epoch ckpts are not needed
      # for P4 -- all metrics are parsed from the trainlog). Saves shared quota.
      RUNDIR=$(python - "$DATASET" "$GROUP" "$LAM" "$SEED" <<'PY'
import sys, glob
ds, grp, lam, seed = sys.argv[1:5]
suf = "" if float(lam) == 0 else "_p4aux-l{}".format(lam)
pat = "logging/Retrieval/{}/{}/*seed{}*{}/ckpt".format(ds, grp, seed, suf)
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

echo "======== ALL P4 RUNS DONE ========"
python scripts/analysis/p4_collect.py --out scripts/analysis/p4_out/p4_results.json || true
