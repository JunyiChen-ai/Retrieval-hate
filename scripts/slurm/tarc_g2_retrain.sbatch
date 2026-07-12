#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=tarc_g2rt
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit. NO b2_push / rclone tail.
#
# TARC B-line G2 retrain (exp-tarc-t0.md §6-G2 step): the two G1 survivors
# (v1prefer, v3lt0.1) trained with MLLM-PREDICTED target (--tarc_target_source
# mllm_pred --tarc_mllm qwen7b), 3 seeds each. NO oracle_probe, NO GT target
# (§5). OFF control is REUSED from G1 job 12975 (same code/seeds) -> not rerun.
# 6 ON arms x ~25 s ~= 3-5 min on one A100, serial.
#
# Verdict (§6): final-epoch protocol VAL delta-acc (arm - G1-OFF), 3-seed mean
#   v1prefer >= +0.0112, v3lt0.1 >= +0.0131 (0.6 x GT-oracle Delta), sign >=2/3.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

CLIP=openai_clip-vit-large-patch14-336_HF
GROUP_NAME=RAC_tarc_g2rt
WARMUP=5
MLLM_TAG=${MLLM_TAG:-qwen7b}

# MLLM-predicted-target flags shared by every arm (no oracle, no GT).
TARC_MLLM="--tarc_target_source mllm_pred --tarc_mllm ${MLLM_TAG}"

arm_flags () {
  case "$1" in
    v1prefer)  echo "$TARC_MLLM --tarc_hn_mode prefer" ;;   # V1 mining
    v3lt0.1)   echo "$TARC_MLLM --lambda_tarc 0.1" ;;       # V3 separation reg
    *) echo "UNKNOWN_ARM_$1" ; return 1 ;;
  esac
}

ARMS=(v1prefer v3lt0.1)
SEEDS=(0 1 2)

run_arm () {
  local ARM="$1" SEED="$2"
  local FLAGS; FLAGS="$(arm_flags "$ARM")"
  local RUNLOG="slurm/logs/tarc_g2rt_${ARM}_seed${SEED}_${SLURM_JOB_ID:-$$}.trainlog"
  echo ""
  echo "########## ARM=$ARM SEED=$SEED FLAGS=[$FLAGS] -> $RUNLOG ##########"
  python ./src/run_rac.py --batch_size 64 \
      --lr 0.0001 --epochs 30 --topk 20 --dataset "HateMM" \
      --model "$CLIP" \
      --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
      --fusion_mode "align" \
      --hard_negatives_loss True --no_hard_negatives 1 \
      --final_eval False --seed "$SEED" --group_name "$GROUP_NAME" \
      --metric "cos" --loss "triplet" --batch_norm False \
      --hybrid_loss True --warmup "$WARMUP" \
      --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
      --lambda_seg 0 --seg_mode full --num_subclips 4 \
      --em_rounds 2 --consensus_topk 10 --consensus_margin 0.2 \
      --exp_comment "_${ARM}_mllm" \
      $FLAGS \
      --Faiss_GPU False --force False 2>&1 | tee "$RUNLOG"

  echo "-------- VAL-SELECTED + FINAL-EPOCH readout: ARM=$ARM seed$SEED --------"
  WARMUP="$WARMUP" ARM="$ARM" SEED="$SEED" python - "$RUNLOG" <<'PY'
import sys, re, os
log = open(sys.argv[1]).read()
val, test = {}, {}
vre = re.compile(r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
tre = re.compile(r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
for m in vre.finditer(log):
    e=int(m.group(1)); val[e]=tuple(float(x) for x in m.groups()[1:])
for m in tre.finditer(log):
    e=int(m.group(1)); test[e]=tuple(float(x) for x in m.groups()[1:])
if not val or not test:
    print("NO_PARSE"); sys.exit(0)
W=int(os.environ.get("WARMUP",5))
arm=os.environ.get("ARM","?"); seed=os.environ.get("SEED","?")
warm=[e for e in val if e>=W] or list(val.keys())
best=max(warm, key=lambda e:(val[e][3], val[e][4]))
a=test.get(best); fe=max(test); b=test[fe]
fev=max(val); vf=val[fev]  # final-epoch VAL row (protocol B judge)
print(f"VALSEL epoch {best}: VAL acc {val[best][3]:.4f} F1 {val[best][0]:.4f} | TEST acc {a[3]:.4f} F1 {a[0]:.4f}")
print(f"FINAL  epoch {fe}:  VAL acc {vf[3]:.4f} F1 {vf[0]:.4f} | TEST acc {b[3]:.4f} F1 {b[0]:.4f}")
# RESULT_ROW: valsel VAL acc/f1, FINAL-epoch VAL acc/f1 (protocol-B judge), TEST both
print(f"RESULT_ROW\t{arm}\tseed{seed}\tVALSEL_VALacc\t{val[best][3]:.4f}\tVALSEL_VALf1\t{val[best][0]:.4f}\tFINAL_ep\t{fev}\tFINAL_VALacc\t{vf[3]:.4f}\tFINAL_VALf1\t{vf[0]:.4f}\tVALSEL_TESTacc\t{a[3]:.4f}\tFINAL_TESTacc\t{b[3]:.4f}")
PY
}

for ARM in "${ARMS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_arm "$ARM" "$SEED"
  done
done

echo ""
echo "======== tarc_g2rt ALL DONE ($SLURM_JOB_ID) ========"
echo "Verdict input: final-epoch VAL acc per arm/seed vs G1-OFF (job 12975)."
