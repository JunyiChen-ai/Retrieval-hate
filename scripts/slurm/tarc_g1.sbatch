#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=tarc_g1
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# TARC B-line G1: GT-target ORACLE CEILING probe (exp-tarc-t0.md §6/§7).
# Cloned from enc3seed.sbatch (original left untouched). Frozen-CLIP RGCL on HateMM,
# 3 seeds, knob ON vs the byte-identical knob-OFF control, paired within seed. Every
# non-OFF arm reads GT target (--tarc_target_source gt_oracle --oracle_probe True),
# so runs are stamped ORACLE_CEILING and their numbers NEVER enter the main table.
# 21 arms x ~25 s ~= 9-12 min on one A100, serial.
#
# OFF arms are the byte-identical baseline: they MUST reproduce the enc3seed HateMM
# CLIP numbers (verified 2026-07-13 from job 12850 trainlogs):
#   seed 0: VALSEL acc 0.8279 F1 0.8172 | FINAL acc 0.8186 F1 0.7997
#   seed 1: VALSEL acc 0.8279 F1 0.8163 | FINAL acc 0.8047 F1 0.7822
#   seed 2: VALSEL acc 0.8047 F1 0.7920 | FINAL acc 0.8140 F1 0.7988
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

CLIP=openai_clip-vit-large-patch14-336_HF
GROUP_NAME=RAC_tarc_g1
WARMUP=5

# GT-target oracle-ceiling flags shared by every non-OFF arm (§5 gate).
TARC_ORACLE="--tarc_target_source gt_oracle --oracle_probe True"

# arm label -> extra TARC flags
arm_flags () {
  case "$1" in
    off)       echo "--tarc_target_source off" ;;                       # byte-identical baseline
    v1prefer)  echo "$TARC_ORACLE --tarc_hn_mode prefer" ;;             # V1 mining
    v1require) echo "$TARC_ORACLE --tarc_hn_mode require" ;;            # V1 mining
    v3lt0.1)   echo "$TARC_ORACLE --lambda_tarc 0.1" ;;                 # V3 separation reg
    v3lt0.5)   echo "$TARC_ORACLE --lambda_tarc 0.5" ;;                 # V3 separation reg
    v2vg0.5)   echo "$TARC_ORACLE --tarc_vote_gamma 0.5" ;;            # V2 vote reweight
    v2vg1.0)   echo "$TARC_ORACLE --tarc_vote_gamma 1.0" ;;            # V2 vote reweight
    *) echo "UNKNOWN_ARM_$1" ; return 1 ;;
  esac
}

# Run order: OFF (3 seeds) first so the no-op check lands early, then V1, V3, V2.
ARMS=(off v1prefer v1require v3lt0.1 v3lt0.5 v2vg0.5 v2vg1.0)
SEEDS=(0 1 2)

run_arm () {
  local ARM="$1" SEED="$2"
  local FLAGS; FLAGS="$(arm_flags "$ARM")"
  local RUNLOG="slurm/logs/tarc_g1_${ARM}_seed${SEED}_${SLURM_JOB_ID:-$$}.trainlog"
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
      --exp_comment "_${ARM}" \
      $FLAGS \
      --Faiss_GPU False --force False 2>&1 | tee "$RUNLOG"

  echo "-------- VAL-SELECTED (warmup>=5) + FINAL-EPOCH readout: ARM=$ARM seed$SEED --------"
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
print(f"VALSEL epoch {best}: TEST F1 {a[0]:.4f} acc {a[3]:.4f} roc {a[4]:.4f}")
print(f"FINAL  epoch {fe}: TEST F1 {b[0]:.4f} acc {b[3]:.4f} roc {b[4]:.4f}")
# val macro-F1/acc too (G1 is judged on VAL delta acc)
print(f"VALSEL_VAL epoch {best}: VAL F1 {val[best][0]:.4f} acc {val[best][3]:.4f} roc {val[best][4]:.4f}")
print(f"RESULT_ROW\t{arm}\tseed{seed}\tvalsel\t{best}\tVALacc\t{val[best][3]:.4f}\tVALf1\t{val[best][0]:.4f}\tTESTacc\t{a[3]:.4f}\tTESTf1\t{a[0]:.4f}\tfinal\t{fe}\tTESTacc\t{b[3]:.4f}\tTESTf1\t{b[0]:.4f}")
PY
}

for ARM in "${ARMS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_arm "$ARM" "$SEED"
  done
done

echo ""
echo "======== tarc_g1 ALL DONE ($SLURM_JOB_ID) ========"
echo "Reminder: every non-OFF arm is an ORACLE_CEILING probe -- its numbers gate the"
echo "line (VAL delta acc >= +0.015, sign >=2/3) and NEVER enter the main table (§5)."
bash scripts/b2_push.sh logging "logs/tarc_g1" || true
