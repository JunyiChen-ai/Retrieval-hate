#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=p3_train
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# P3 (EXP_p3_evidence_pooling §0.5): whole-video training (LAMBDA_SEG=0) on the
# P3 pooling caches. A --model swap between:
#   POOL=mean     floor (16-frame sub-clip mean; equal-weights == mean bit-for-bit)
#   POOL=wsoftT1  PRIMARY (softmax T=1 evidence-weighted pooling)
#   POOL=wmild    SECONDARY (normalize(1+s) milder weighting)
# Recipe identical to experiments_video.sh (align/triplet/cos/hybrid, topk20,
# warmup5). Fresh GROUP RAC_video_p3pool + FORCE=False; exp_comment keeps same-
# seed conditions from colliding.
#
# Usage:
#   DATASET=MHC POOL=mean    SEED=0 sbatch scripts/slurm/train_p3pool.sbatch
#   DATASET=MHC POOL=wsoftT1 SEED=0 sbatch scripts/slurm/train_p3pool.sbatch
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

DATASET=${DATASET:-MHC}
POOL=${POOL:-mean}
MODEL="p3pool_${POOL}_HF"
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-64}
export WARMUP=${WARMUP:-5}
SEED=${SEED:-0}
GROUP_NAME=${GROUP_NAME:-RAC_video_p3pool}
FORCE=${FORCE:-False}
COMMENT="_p3${POOL}"

RUNLOG="slurm/logs/p3_${DATASET}_${POOL}_seed${SEED}_${SLURM_JOB_ID:-$$}.trainlog"
echo "[p3_train] DATASET=$DATASET POOL=$POOL MODEL=$MODEL SEED=$SEED GROUP=$GROUP_NAME -> $RUNLOG"

python ./src/run_rac.py --batch_size "$BATCH" \
    --lr 0.0001 --epochs "$EPOCHS" --topk 20 --dataset "$DATASET" \
    --model "$MODEL" \
    --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
    --fusion_mode "align" \
    --hard_negatives_loss True --no_hard_negatives 1 \
    --final_eval False --seed "$SEED" --group_name "$GROUP_NAME" \
    --metric "cos" --loss "triplet" --batch_norm False \
    --hybrid_loss True --warmup "$WARMUP" \
    --lambda_seg 0 \
    --exp_comment "${COMMENT}" \
    --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
    --Faiss_GPU False --force "$FORCE" 2>&1 | tee "$RUNLOG"

echo ""
echo "======== RESULT ($DATASET / POOL=$POOL seed=$SEED) ========"
WARMUP="$WARMUP" python - "$RUNLOG" <<'PY'
import sys, re, os
log = open(sys.argv[1]).read()
val, test = {}, {}
vre = re.compile(r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
tre = re.compile(r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
for m in vre.finditer(log):
    e=int(m.group(1)); val[e]=tuple(float(x) for x in m.groups()[1:])
for m in tre.finditer(log):
    e=int(m.group(1)); test[e]=tuple(float(x) for x in m.groups()[1:])
if not val:
    print("NO Val_Retrieval lines parsed"); sys.exit(0)
WARMUP=int(os.environ.get("WARMUP",5))
warm=[e for e in val if e>=WARMUP]
cand=warm or list(val.keys())
best=max(cand, key=lambda e:(val[e][3], val[e][4]))
vf1,vp,vr,vacc,vroc=val[best]
print(f"val-selected epoch={best}  VAL maF1 {vf1:.4f} acc {vacc:.4f}")
if best in test:
    tf1,tp,tr,tacc,troc=test[best]
    print(f"VAL-SEL TEST : maF1 {tf1:.4f} acc {tacc:.4f} roc {troc:.4f}")
    print(f"RESULT_ROW\t{best}\t{tf1:.4f}\t{tacc:.4f}\t{troc:.4f}")
fe=max(val.keys())
if fe in test:
    tf1,tp,tr,tacc,troc=test[fe]
    print(f"FINAL-EP TEST (ep {fe}): maF1 {tf1:.4f} acc {tacc:.4f} roc {troc:.4f}")
    print(f"RESULT_ROW_FINAL\t{fe}\t{tf1:.4f}\t{tacc:.4f}\t{troc:.4f}")
PY
echo "==========================================================="
bash scripts/b2_push.sh logging "logs/$DATASET" || true
echo "[p3_train] done DATASET=$DATASET POOL=$POOL SEED=$SEED"
