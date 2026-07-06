#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=p5cf
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (job auto-releases from JobHeldUser).
#
# P5 counterfactual hard-negative twins training (research-wiki/EXP_p5_counterfactual_negs.md).
# Exact RAC_video_CLIP floor recipe + one extra per-anchor twin negative. Conditions:
#   floor  : --cf_negs False              (must reproduce the floor bit-for-bit)
#   cf     : --cf_negs True               (ours)
#   cfrand : --cf_negs True --cf_negs_random True  (per-anchor pairing control)
# Seeds 0,1,2, both datasets, GROUP RAC_video_p5cf, FORCE False. Parsed by p5_collect.py.
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
GROUP="RAC_video_p5cf"
FORCE=${FORCE:-False}
DATASETS=${DATASETS:-"MHC MHC_zh"}
CONDS=${CONDS:-"floor cf cfrand"}
SEEDS=${SEEDS:-"0 1 2"}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for DATASET in $DATASETS; do
  for COND in $CONDS; do
    case "$COND" in
      floor)  CF="False"; CFR="False" ;;
      cf)     CF="True";  CFR="False" ;;
      cfrand) CF="True";  CFR="True"  ;;
      *) echo "unknown cond $COND"; exit 1 ;;
    esac
    for SEED in $SEEDS; do
      TAG="p5cf_${DATASET}_${COND}_s${SEED}"
      RUNLOG="slurm/logs/${TAG}.trainlog"
      echo "======== TRAIN $TAG (cf_negs=$CF random=$CFR) -> $RUNLOG ========"
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
          --cf_negs "$CF" --cf_negs_random "$CFR" 2>&1 | tee "$RUNLOG"

      RUNDIR=$(python - "$DATASET" "$GROUP" "$COND" "$SEED" <<'PY'
import sys, glob
ds, grp, cond, seed = sys.argv[1:5]
suf = {"floor": "", "cf": "_p5cf", "cfrand": "_p5cfrand"}[cond]
pat = "logging/Retrieval/{}/{}/*seed{}*{}/ckpt".format(ds, grp, seed, suf)
# exclude the cf/cfrand collision when cond==cf (cf suffix is a prefix of cfrand)
hits = [h for h in sorted(glob.glob(pat))
        if (cond != "cf") or ("_p5cfrand" not in h)]
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

echo "======== ALL P5 RUNS DONE ========"
python scripts/analysis/p5_collect.py --out scripts/analysis/p5_out/p5_results.json || true
