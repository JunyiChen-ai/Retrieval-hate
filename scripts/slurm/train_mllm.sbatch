#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=mllm_train
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# Usage: sbatch scripts/slurm/train_mllm.sbatch <DATASET> <MODEL_TAG>
#   e.g.  sbatch scripts/slurm/train_mllm.sbatch HateMM Qwen2.5-VL-7B-Instruct_HF
#         sbatch scripts/slurm/train_mllm.sbatch HateMM openai_clip-vit-large-patch14-336_HF
#   Trains the RGCL head on the given feature cache, 30 epochs, faiss CPU-only,
#   then parses the val-selected (Val_Retrieval acc, tie-break roc) epoch and prints
#   its Test_Retrieval macro-F1 / macro-P / macro-R / acc.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

DATASET=${1:-HateMM}
MODEL=${2:-Qwen2.5-VL-7B-Instruct_HF}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-64}

RUNLOG="slurm/logs/rgcl_${DATASET}_${MODEL}_$$.trainlog"
echo "[train_mllm] DATASET=$DATASET MODEL=$MODEL EPOCHS=$EPOCHS -> $RUNLOG"

python ./src/run_rac.py --batch_size "$BATCH" \
    --lr 0.0001 --epochs "$EPOCHS" --topk 20 --dataset "$DATASET" \
    --model "$MODEL" \
    --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
    --fusion_mode "align" \
    --hard_negatives_loss True --no_hard_negatives 1 \
    --final_eval False --seed 0 --group_name "RAC_video" \
    --metric "cos" --loss "triplet" --batch_norm False \
    --hybrid_loss True \
    --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
    --Faiss_GPU False --force True 2>&1 | tee "$RUNLOG"

echo ""
echo "======== VAL-SELECTED RESULT ($DATASET / $MODEL) ========"
python - "$RUNLOG" <<'PY'
import sys, re, os
log = open(sys.argv[1]).read()
val = {}
test = {}
vre = re.compile(r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
tre = re.compile(r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
for m in vre.finditer(log):
    e=int(m.group(1)); val[e]=tuple(float(x) for x in m.groups()[1:])
for m in tre.finditer(log):
    e=int(m.group(1)); test[e]=tuple(float(x) for x in m.groups()[1:])
if not val:
    print("NO Val_Retrieval lines parsed"); sys.exit(1)
# Warmup floor: only epochs >= WARMUP are eligible for best-epoch selection.
# This avoids locking onto a spuriously high untrained epoch 0 on tiny dev sets.
# Matches the --warmup arg in src/run_rac.py (default 5).
WARMUP = int(os.environ.get("WARMUP", 5))
warm_epochs = [e for e in val if e >= WARMUP]
if warm_epochs:
    candidate_epochs = warm_epochs
else:
    # Fallback: run was shorter than WARMUP; consider all epochs.
    print(f"WARNING: no epoch >= {WARMUP} found; falling back to all epochs for selection.")
    candidate_epochs = list(val.keys())
# select epoch by val acc (idx 3), tie-break val roc (idx 4)
best = max(candidate_epochs, key=lambda e:(val[e][3], val[e][4]))
vf1,vp,vr,vacc,vroc = val[best]
print(f"selected epoch = {best} (by Val_Retrieval acc>=warmup{WARMUP}, tie-break roc)")
print(f"VAL  : macroF1 {vf1:.4f}  macroP {vp:.4f}  macroR {vr:.4f}  acc {vacc:.4f}  roc {vroc:.4f}")
if best in test:
    tf1,tp,tr,tacc,troc = test[best]
    print(f"TEST : macroF1 {tf1:.4f}  macroP {tp:.4f}  macroR {tr:.4f}  acc {tacc:.4f}  roc {troc:.4f}")
    print(f"RESULT_ROW\t{best}\t{tf1:.4f}\t{tp:.4f}\t{tr:.4f}\t{tacc:.4f}\t{troc:.4f}")
else:
    print(f"no Test line for epoch {best}")
PY
echo "=========================================================="
