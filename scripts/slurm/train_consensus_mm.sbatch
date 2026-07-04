#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=mhc_cons_mm
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# EXP_mm_segment_keys phase 4: consensus denoising with MULTIMODAL segment
# keys (frame CLIP + per-window Whisper-ASR CLIP text) in the E-step vote.
#   CONS_SPACE=mm  -> the new voting space (requires *_subclipK4_mm_* cache)
#   CONS_SPACE=clip -> pre-existing behaviour (control; bit-for-bit verified)
# Protocol: identical to the consensus kill-ablation (lambda_seg=0.5, K=4,
# topk=10, tau=0.2, EM=2, warmup>=5 val-selected). Fresh GROUP dir +
# FORCE=False: existing checkpoints can never be overwritten.
#
# Usage:
#   DATASET=MHC SEED=0 sbatch scripts/slurm/train_consensus_mm.sbatch
#   DATASET=MHC_zh SEED=1 MM_TEXT_WEIGHT=0.5 sbatch scripts/slurm/train_consensus_mm.sbatch
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
export WARMUP=${WARMUP:-5}
SEED=${SEED:-0}
GROUP_NAME=${GROUP_NAME:-RAC_video_consensus_mm}
FORCE=${FORCE:-False}

LAMBDA_SEG=${LAMBDA_SEG:-0.5}
SEG_MODE=${SEG_MODE:-consensus}
NUM_SUBCLIPS=${NUM_SUBCLIPS:-4}
SUBCLIP_CACHE=${SUBCLIP_CACHE:-auto}
EM_ROUNDS=${EM_ROUNDS:-2}
CONS_TOPK=${CONS_TOPK:-10}
CONS_MARGIN=${CONS_MARGIN:-0.2}
CONS_USE_DRIFT=${CONS_USE_DRIFT:-True}
CONS_CONFLICT=${CONS_CONFLICT:-ignore}
CONS_SPACE=${CONS_SPACE:-mm}
MM_TEXT_WEIGHT=${MM_TEXT_WEIGHT:-0.5}
MM_EMPTY_TEXT=${MM_EMPTY_TEXT:-parent}
MM_SUBCLIP_CACHE=${MM_SUBCLIP_CACHE:-auto}

COMMENT="_seg${LAMBDA_SEG}_${SEG_MODE}_vs-${CONS_SPACE}"
if [ "$CONS_SPACE" = "mm" ]; then
    COMMENT="${COMMENT}-w${MM_TEXT_WEIGHT}-${MM_EMPTY_TEXT}"
fi

RUNLOG="slurm/logs/consmm_${DATASET}_${MODEL}_${SEG_MODE}_${CONS_SPACE}_w${MM_TEXT_WEIGHT}_seed${SEED}_${SLURM_JOB_ID:-$$}.trainlog"
echo "[cons_mm] DATASET=$DATASET MODEL=$MODEL LAMBDA_SEG=$LAMBDA_SEG SEG_MODE=$SEG_MODE SPACE=$CONS_SPACE W=$MM_TEXT_WEIGHT EMPTY=$MM_EMPTY_TEXT EM=$EM_ROUNDS TOPK=$CONS_TOPK TAU=$CONS_MARGIN SEED=$SEED GROUP=$GROUP_NAME -> $RUNLOG"

python ./src/run_rac.py --batch_size "$BATCH" \
    --lr 0.0001 --epochs "$EPOCHS" --topk 20 --dataset "$DATASET" \
    --model "$MODEL" \
    --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
    --fusion_mode "align" \
    --hard_negatives_loss True --no_hard_negatives 1 \
    --final_eval False --seed "$SEED" --group_name "$GROUP_NAME" \
    --metric "cos" --loss "triplet" --batch_norm False \
    --hybrid_loss True --warmup "$WARMUP" \
    --lambda_seg "$LAMBDA_SEG" --seg_mode "$SEG_MODE" \
    --num_subclips "$NUM_SUBCLIPS" --subclip_cache "$SUBCLIP_CACHE" \
    --em_rounds "$EM_ROUNDS" --consensus_topk "$CONS_TOPK" \
    --consensus_margin "$CONS_MARGIN" \
    --consensus_use_drift "$CONS_USE_DRIFT" \
    --consensus_conflict "$CONS_CONFLICT" \
    --consensus_space "$CONS_SPACE" \
    --mm_text_weight "$MM_TEXT_WEIGHT" \
    --mm_empty_text "$MM_EMPTY_TEXT" \
    --mm_subclip_cache "$MM_SUBCLIP_CACHE" \
    --exp_comment "${COMMENT}" \
    --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
    --Faiss_GPU False --force "$FORCE" 2>&1 | tee "$RUNLOG"

echo ""
echo "======== VAL-SELECTED RESULT ($DATASET / $SEG_MODE vs-$CONS_SPACE w$MM_TEXT_WEIGHT seed$SEED) ========"
python - "$RUNLOG" <<'PY'
import sys, re, os
log = open(sys.argv[1]).read()
val = {}
test = {}
vre = re.compile(r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
tre = re.compile(r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
# EM runs repeat epoch indices; keep the LAST occurrence (= final EM round).
for m in vre.finditer(log):
    e=int(m.group(1)); val[e]=tuple(float(x) for x in m.groups()[1:])
for m in tre.finditer(log):
    e=int(m.group(1)); test[e]=tuple(float(x) for x in m.groups()[1:])
if not val:
    print("NO Val_Retrieval lines parsed"); sys.exit(1)
WARMUP = int(os.environ.get("WARMUP", 5))
warm_epochs = [e for e in val if e >= WARMUP]
candidate_epochs = warm_epochs or list(val.keys())
if not warm_epochs:
    print(f"WARNING: no epoch >= {WARMUP}; using all epochs.")
best = max(candidate_epochs, key=lambda e:(val[e][3], val[e][4]))
vf1,vp,vr,vacc,vroc = val[best]
print(f"selected epoch = {best} (by Val_Retrieval acc>=warmup{WARMUP}, tie-break roc)")
print(f"VAL  : macroF1 {vf1:.4f}  macroP {vp:.4f}  macroR {vr:.4f}  acc {vacc:.4f}  roc {vroc:.4f}")
if best in test:
    tf1,tp,tr,tacc,troc = test[best]
    print(f"TEST : macroF1 {tf1:.4f}  macroP {tp:.4f}  macroR {tr:.4f}  acc {tacc:.4f}  roc {troc:.4f}")
    print(f"RESULT_ROW\t{best}\t{tf1:.4f}\t{tp:.4f}\t{tr:.4f}\t{tacc:.4f}\t{troc:.4f}")
# final-epoch (selection-free) reading, second pre-registered lens
fe = max(val.keys())
if fe in test:
    tf1,tp,tr,tacc,troc = test[fe]
    print(f"FINAL-EPOCH TEST (ep {fe}): macroF1 {tf1:.4f}  acc {tacc:.4f}  roc {troc:.4f}")
    print(f"RESULT_ROW_FINAL\t{fe}\t{tf1:.4f}\t{tp:.4f}\t{tr:.4f}\t{tacc:.4f}\t{troc:.4f}")
PY
echo "=========================================================="

bash scripts/b2_push.sh logging "logs/$DATASET" || true
echo "[cons_mm] done DATASET=$DATASET SPACE=$CONS_SPACE W=$MM_TEXT_WEIGHT SEED=$SEED"
