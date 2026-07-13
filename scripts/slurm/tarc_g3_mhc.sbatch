#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=tarc_g3mhc
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit. NO b2_push / rclone tail.
#
# TARC B-line G3 on MHC-EN (exp-tarc-t0.md §6-G3): winning variant v1prefer with
# MLLM-PREDICTED target (--tarc_target_source mllm_pred --tarc_mllm qwen7b), 3 seeds,
# vs the byte-identical OFF baseline. V1 is TRAIN-ONLY mining, so only MHC TRAIN target
# is predicted (data/gt/MHC/target_pred_qwen7b.json); MHC val/test are NEVER read for
# target -> test content stays pristine. This is the single sanctioned TARC test-touch.
#
# Command = the enc3seed MHC config (--dataset MHC --model openai_clip...) + one TARC knob.
# The OFF arm MUST reproduce enc3s_MHC_openai_clip-*_12850 to 4 dp (no-op correctness gate);
# if it does not, the whole batch is void. Expected OFF numbers (read live from the
# job-12850 trainlogs 2026-07-13):
#   seed 0: VALSEL acc 0.7826 F1 0.7113 | FINAL acc 0.7640 F1 0.7145
#   seed 1: VALSEL acc 0.7329 F1 0.6034 | FINAL acc 0.7826 F1 0.7159
#   seed 2: VALSEL acc 0.7702 F1 0.6997 | FINAL acc 0.7888 F1 0.7303
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

CLIP=openai_clip-vit-large-patch14-336_HF
GROUP_NAME=RAC_tarc_g3mhc
WARMUP=5
MLLM_TAG=${MLLM_TAG:-qwen7b}

arm_flags () {
  case "$1" in
    off)       echo "--tarc_target_source off" ;;
    v1prefer)  echo "--tarc_target_source mllm_pred --tarc_mllm ${MLLM_TAG} --tarc_hn_mode prefer" ;;
    *) echo "UNKNOWN_ARM_$1" ; return 1 ;;
  esac
}

ARMS=(off v1prefer)
SEEDS=(0 1 2)

run_arm () {
  local ARM="$1" SEED="$2"
  local FLAGS; FLAGS="$(arm_flags "$ARM")"
  local RUNLOG="slurm/logs/tarc_g3mhc_${ARM}_seed${SEED}_${SLURM_JOB_ID:-$$}.trainlog"
  echo ""
  echo "########## ARM=$ARM SEED=$SEED FLAGS=[$FLAGS] -> $RUNLOG ##########"
  python ./src/run_rac.py --batch_size 64 \
      --lr 0.0001 --epochs 30 --topk 20 --dataset "MHC" \
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
      --exp_comment "_${ARM}" \
      $FLAGS \
      --Faiss_GPU False --force False 2>&1 | tee "$RUNLOG"

  echo "-------- VAL-SELECTED + FINAL-EPOCH readout: ARM=$ARM seed$SEED --------"
  ARM="$ARM" SEED="$SEED" WARMUP="$WARMUP" python - "$RUNLOG" <<'PY'
import sys, re, os
log=open(sys.argv[1]).read()
val,test={},{}
vre=re.compile(r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
tre=re.compile(r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
for m in vre.finditer(log):
    e=int(m.group(1)); val[e]=tuple(float(x) for x in m.groups()[1:])
for m in tre.finditer(log):
    e=int(m.group(1)); test[e]=tuple(float(x) for x in m.groups()[1:])
if not val or not test:
    print("NO_PARSE"); sys.exit(0)
W=int(os.environ.get("WARMUP",5)); arm=os.environ.get("ARM","?"); seed=os.environ.get("SEED","?")
warm=[e for e in val if e>=W] or list(val.keys())
best=max(warm, key=lambda e:(val[e][3],val[e][4]))
a=test.get(best); fe=max(test); b=test[fe]
print(f"VALSEL epoch {best}: TEST acc {a[3]:.4f} F1 {a[0]:.4f} | VAL acc {val[best][3]:.4f}")
print(f"FINAL  epoch {fe}: TEST acc {b[3]:.4f} F1 {b[0]:.4f} | VAL acc {val[fe][3]:.4f}")
print(f"RESULT_ROW\t{arm}\tseed{seed}\tVALSEL_ep\t{best}\tVALSEL_TESTacc\t{a[3]:.4f}\tVALSEL_TESTf1\t{a[0]:.4f}\tFINAL_ep\t{fe}\tFINAL_TESTacc\t{b[3]:.4f}\tFINAL_TESTf1\t{b[0]:.4f}")
PY
}

for ARM in "${ARMS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_arm "$ARM" "$SEED"
  done
done

echo ""
echo "======== tarc_g3mhc ALL DONE ($SLURM_JOB_ID) ========"
echo "Gate MHC OFF no-op vs enc3s_MHC 12850 (4dp): s0 VALSEL 0.7826/0.7113 FINAL 0.7640/0.7145 |"
echo " s1 VALSEL 0.7329/0.6034 FINAL 0.7826/0.7159 | s2 VALSEL 0.7702/0.6997 FINAL 0.7888/0.7303."
