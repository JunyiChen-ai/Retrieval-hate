#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=c1settle
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (project rule).
#
# C1 settling run (refine-logs/C1_KILL_REVIEW.md §5 "minimal cheap evidence"):
# Faithful SEQUENTIAL Stage-2 on the P9 Stage-1 LoRA-adapted feature cache.
# Train the project's standard RGCL head (align-fusion + triplet-margin cos 0.1
# + BCE hybrid, top-20 kNN vote) on the ADAPTED HateMM seed-0 features, and read
# out DEV (Val_Retrieval / classifier) only.
#
# The python command is BYTE-IDENTICAL to the frozen-Qwen floor
# (scripts/slurm/enc3seed.sbatch / train_archive_baseline.sbatch) EXCEPT --model:
#   frozen floor  --model Qwen2.5-VL-7B-Instruct_HF
#   this run      --model c1settle_hatemm_s0   (-> P9 Stage-1-adapted caches)
# so any DEV delta is attributable ONLY to Stage-1 adaptation of the input feats.
#
# NO TEST TOUCH: --model c1settle_hatemm_s0 resolves to symlinks in
#   data/CLIP_Embedding/HateMM/ where the TEST slot points at the DEV cache
#   (dev_seen_p9c3_hatemm_s0.pt), so the real test_seen_p9c3_hatemm_s0.pt is
#   never loaded or evaluated. Test_Retrieval == Val_Retrieval by construction
#   and is ignored; only Val_Retrieval (DEV) is reported.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

DATASET=HateMM
MODEL=c1settle_hatemm_s0
SEED=0
GROUP_NAME=c1settle
WARMUP=5

RUNLOG="slurm/logs/c1settle_${DATASET}_${MODEL}_seed${SEED}_${SLURM_JOB_ID:-$$}.trainlog"
echo "########## C1 SETTLE: DATASET=$DATASET MODEL=$MODEL SEED=$SEED -> $RUNLOG ##########"

python ./src/run_rac.py --batch_size 64 \
    --lr 0.0001 --epochs 30 --topk 20 --dataset "$DATASET" \
    --model "$MODEL" \
    --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
    --fusion_mode "align" \
    --hard_negatives_loss True --no_hard_negatives 1 \
    --final_eval False --seed "$SEED" --group_name "$GROUP_NAME" \
    --metric "cos" --loss "triplet" --batch_norm False \
    --hybrid_loss True --warmup "$WARMUP" \
    --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
    --lambda_seg 0 --seg_mode full --num_subclips 4 \
    --em_rounds 2 --consensus_topk 10 --consensus_margin 0.2 \
    --exp_comment "_${MODEL}" \
    --Faiss_GPU False --force False 2>&1 | tee "$RUNLOG"

echo ""
echo "======== C1 SETTLE DEV READ-OUT ($DATASET / $MODEL / seed$SEED) ========"
WARMUP="$WARMUP" python - "$RUNLOG" <<'PY'
import sys, re, os
log = open(sys.argv[1]).read()
val = {}
vre = re.compile(r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
for m in vre.finditer(log):
    e = int(m.group(1)); val[e] = tuple(float(x) for x in m.groups()[1:])
# classifier (MLP hybrid) dev acc per epoch
cls = {}
cre = re.compile(r"dev  Epoch (\d+) acc: ([\d.]+)")
for m in cre.finditer(log):
    cls[int(m.group(1))] = float(m.group(2))
if not val:
    print("NO_PARSE_VAL"); sys.exit(0)
W = int(os.environ.get("WARMUP", 5))
warm = [e for e in val if e >= W] or list(val.keys())
best = max(warm, key=lambda e: (val[e][3], val[e][4]))
fe = max(val)
print(f"kNN  read-out  VALSEL epoch {best}: DEV mF1 {val[best][0]:.4f} acc {val[best][3]:.4f} roc {val[best][4]:.4f}")
print(f"kNN  read-out  FINAL  epoch {fe}: DEV mF1 {val[fe][0]:.4f} acc {val[fe][3]:.4f} roc {val[fe][4]:.4f}")
if cls:
    print(f"MLP  read-out  VALSEL epoch {best}: DEV acc {cls.get(best, float('nan')):.4f}")
    print(f"MLP  read-out  FINAL  epoch {fe}: DEV acc {cls.get(fe, float('nan')):.4f}")
print(f"RESULT_ROW\tknn\tvalsel\t{best}\t{val[best][3]:.4f}\t{val[best][0]:.4f}\tfinal\t{fe}\t{val[fe][3]:.4f}\t{val[fe][0]:.4f}")
PY
echo "=========================================================="
echo "[c1settle] done"
