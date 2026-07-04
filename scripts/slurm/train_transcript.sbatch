#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=trs_train
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# W4 decisive ablation: is the MLLM structured-archive gain structured
# distillation, or just a fix for the CLIP 77-token transcript truncation?
# Identical protocol to scripts/slurm/train_archive.sbatch, but the
# --archive_feats channel is fed a LONG-CONTEXT TRANSCRIPT embedding
# (KEYTAG=transcript, paraphrase-multilingual-mpnet-base-v2 @ 512 tok) or the
# offline archive+transcript double key (KEYTAG=arctrans), both produced by
# scripts/generate_transcript_embedding.py. NO src/ changes.
#
# Usage:
#   DATASET=MHC MODEL=Qwen2.5-VL-7B-Instruct_HF KEYTAG=transcript \
#     ARCHIVE_ALPHA=0.25 sbatch scripts/slurm/train_transcript.sbatch
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

DATASET=${DATASET:-MHC}
MODEL=${MODEL:-Qwen2.5-VL-7B-Instruct_HF}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-64}
WARMUP=${WARMUP:-5}
SEED=${SEED:-0}
GROUP_NAME=${GROUP_NAME:-RAC_video_transcript}
FORCE=${FORCE:-False}

# KEYTAG selects which id-aligned .pt feeds the kNN alpha channel:
#   transcript = long-context transcript embedding (mpnet-512)
#   arctrans   = [l2n(archive) | l2n(transcript)] offline double key
KEYTAG=${KEYTAG:-transcript}
EMBTAG=${EMBTAG:-mpnet512_HF}
ARCHIVE_FEATS="/data/jehc223/RGCL/data/CLIP_Embedding/${DATASET}/{split}_${KEYTAG}_${EMBTAG}.pt"
ARCHIVE_MODE=knn
ARCHIVE_ALPHA=${ARCHIVE_ALPHA:-0.25}

RUNLOG="slurm/logs/trs_${DATASET}_${MODEL}_${KEYTAG}_a${ARCHIVE_ALPHA}_${SLURM_JOB_ID:-$$}.trainlog"
echo "[train_transcript] DATASET=$DATASET MODEL=$MODEL KEYTAG=$KEYTAG ALPHA=$ARCHIVE_ALPHA GROUP=$GROUP_NAME FEATS=$ARCHIVE_FEATS -> $RUNLOG"

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
    --archive_feats "$ARCHIVE_FEATS" --archive_mode "$ARCHIVE_MODE" --archive_alpha "$ARCHIVE_ALPHA" \
    --exp_comment "_${MODEL}_${KEYTAG}" \
    --Faiss_GPU False --force "$FORCE" 2>&1 | tee "$RUNLOG"

echo ""
echo "======== VAL-SELECTED RESULT ($DATASET / $MODEL / $KEYTAG-key a$ARCHIVE_ALPHA) ========"
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
echo "[train_transcript] done DATASET=$DATASET MODEL=$MODEL KEYTAG=$KEYTAG ALPHA=$ARCHIVE_ALPHA"
