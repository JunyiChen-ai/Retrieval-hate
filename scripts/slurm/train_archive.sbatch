#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=arc_train
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# W3: MLLM structured-archive (E0b) integration experiments.
#   ARCHIVE_MODE=knn     (a) kNN memory-key augmentation (--archive_alpha)
#   ARCHIVE_MODE=stream  (b) third feature stream into the classifier head
#   ARCHIVE_MODE=replace (c) control: archive replaces text_feats
# Fresh GROUP dir (default RAC_video_archive) + FORCE=False: existing
# checkpoints can never be overwritten (the run aborts instead).
#
# Usage:
#   DATASET=MHC MODEL=openai_clip-vit-large-patch14-336_HF ARCHIVE_MODE=knn \
#     sbatch scripts/slurm/train_archive.sbatch
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

DATASET=${DATASET:-MHC}
MODEL=${MODEL:-openai_clip-vit-large-patch14-336_HF}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-64}
WARMUP=${WARMUP:-5}
SEED=${SEED:-0}
GROUP_NAME=${GROUP_NAME:-RAC_video_archive}
FORCE=${FORCE:-False}

ARCHIVE_FEATS=${ARCHIVE_FEATS:-auto}
ARCHIVE_MODE=${ARCHIVE_MODE:-knn}
ARCHIVE_ALPHA=${ARCHIVE_ALPHA:-1.0}

# Segment (consensus) stacking knobs -- inert at defaults.
LAMBDA_SEG=${LAMBDA_SEG:-0}
SEG_MODE=${SEG_MODE:-full}
NUM_SUBCLIPS=${NUM_SUBCLIPS:-4}
EM_ROUNDS=${EM_ROUNDS:-2}
CONS_TOPK=${CONS_TOPK:-10}
CONS_MARGIN=${CONS_MARGIN:-0.2}

RUNLOG="slurm/logs/arc_${DATASET}_${MODEL}_${ARCHIVE_MODE}_a${ARCHIVE_ALPHA}_seg${LAMBDA_SEG}${SEG_MODE}_${SLURM_JOB_ID:-$$}.trainlog"
echo "[train_archive] DATASET=$DATASET MODEL=$MODEL ARCHIVE_MODE=$ARCHIVE_MODE ALPHA=$ARCHIVE_ALPHA LAMBDA_SEG=$LAMBDA_SEG SEG_MODE=$SEG_MODE GROUP=$GROUP_NAME -> $RUNLOG"

python ./src/run_rac.py --batch_size "$BATCH" \
    --lr 0.0001 --epochs "$EPOCHS" --topk 20 --dataset "$DATASET" \
    --model "$MODEL" \
    --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
    --fusion_mode "align" \
    --hard_negatives_loss True --no_hard_negatives 1 \
    --final_eval False --seed "$SEED" --group_name "$GROUP_NAME" \
    --metric "cos" --loss "triplet" --batch_norm False \
    --hybrid_loss True --warmup "$WARMUP" \
    --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
    --lambda_seg "$LAMBDA_SEG" --seg_mode "$SEG_MODE" --num_subclips "$NUM_SUBCLIPS" \
    --em_rounds "$EM_ROUNDS" --consensus_topk "$CONS_TOPK" --consensus_margin "$CONS_MARGIN" \
    --archive_feats "$ARCHIVE_FEATS" --archive_mode "$ARCHIVE_MODE" --archive_alpha "$ARCHIVE_ALPHA" \
    --exp_comment "_${MODEL}" \
    --Faiss_GPU False --force "$FORCE" 2>&1 | tee "$RUNLOG"

echo ""
echo "======== VAL-SELECTED RESULT ($DATASET / $MODEL / arc-$ARCHIVE_MODE-a$ARCHIVE_ALPHA seg$LAMBDA_SEG-$SEG_MODE) ========"
python - "$RUNLOG" <<'PY'
import sys, re, os
log = open(sys.argv[1]).read()
val = {}
test = {}
vre = re.compile(r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
tre = re.compile(r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
# NOTE: for EM (consensus) runs the same epoch index appears once per round;
# finditer keeps the LAST occurrence = final EM round, which is the round whose
# best ckpt is reported by run_rac -- consistent.
for m in vre.finditer(log):
    e=int(m.group(1)); val[e]=tuple(float(x) for x in m.groups()[1:])
for m in tre.finditer(log):
    e=int(m.group(1)); test[e]=tuple(float(x) for x in m.groups()[1:])
if not val:
    print("NO Val_Retrieval lines parsed"); sys.exit(1)
WARMUP = int(os.environ.get("WARMUP", 5))
warm_epochs = [e for e in val if e >= WARMUP]
if warm_epochs:
    candidate_epochs = warm_epochs
else:
    print(f"WARNING: no epoch >= {WARMUP} found; falling back to all epochs for selection.")
    candidate_epochs = list(val.keys())
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

# Push run logs to B2 (copy, keep local).
bash scripts/b2_push.sh logging "logs/$DATASET" || true
echo "[train_archive] done DATASET=$DATASET MODEL=$MODEL ARCHIVE_MODE=$ARCHIVE_MODE"
