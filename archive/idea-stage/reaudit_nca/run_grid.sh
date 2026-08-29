#!/usr/bin/env bash
# RE-AUDIT NCA -- powered head-level grid for the NCA / soft-kNN head loss.
#
# The python command is byte-identical to scripts/slurm/ncafam_family.sbatch
# (the 2026-07-25 frozen NCA family runner) except: no `tee`, per-run checkpoint
# dir removed before and after each run, and --exp_comment tagged RNCA.
# ARM=floor  -> no extra flags (deployed triplet head, --contrast_mode retrieval default)
# ARM=nca01  -> --head_loss nca --nca_tau 0.1
#
# Usage: run_grid.sh RUNDIR ARMS_CSV DATASET MODEL SEEDS_CSV GROUP
set -uo pipefail

cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

RUNDIR="$1"
IFS=',' read -r -a ARMS <<< "$2"
DS="$3"
MODEL="$4"
IFS=',' read -r -a SEEDS <<< "$5"
GROUP="$6"
LOGDIR="$RUNDIR/logs"
mkdir -p "$LOGDIR"

TOTAL=$(( ${#ARMS[@]} * ${#SEEDS[@]} ))
DONE=0; NFAIL=0
START_TS=$(date +%s)
echo "GRID_START total=$TOTAL ds=$DS model=$MODEL group=$GROUP arms=${ARMS[*]} nseeds=${#SEEDS[@]} at $(date -Is)"

for SEED in "${SEEDS[@]}"; do
  for ARM in "${ARMS[@]}"; do
    case "$ARM" in
      floor) ARM_FLAGS="" ;;
      nca01) ARM_FLAGS="--head_loss nca --nca_tau 0.1" ;;
      *) echo "UNKNOWN ARM $ARM"; exit 2 ;;
    esac
    TAG="${DS}_${ARM}_s${SEED}"
    RUNLOG="$LOGDIR/${TAG}.trainlog"
    if [ -s "$RUNLOG" ] && [ "$(grep -c '^Test_Retrieval Epoch' "$RUNLOG")" -eq 60 ]; then
      DONE=$((DONE+1)); echo "PROGRESS ds=$DS arm=$ARM seed=$SEED STATUS=SKIP_DONE"; continue
    fi
    EXPROOT="logging/Retrieval/${DS}/${GROUP}"
    rm -rf "$EXPROOT"
    T0=$(date +%s)
    python ./src/run_rac.py --batch_size 64 \
        --lr 0.0001 --epochs 30 --topk 20 --dataset "$DS" \
        --model "$MODEL" \
        --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
        --fusion_mode "align" \
        --hard_negatives_loss True --no_hard_negatives 1 \
        --final_eval False --seed "$SEED" --group_name "$GROUP" \
        --metric "cos" --loss "triplet" --batch_norm False \
        --hybrid_loss True --warmup 5 \
        --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
        --lambda_seg 0 --seg_mode full --num_subclips 4 \
        --em_rounds 2 --consensus_topk 10 --consensus_margin 0.2 \
        --exp_comment "_RNCA_${ARM}" \
        --Faiss_GPU False --force False ${ARM_FLAGS} > "$RUNLOG" 2>&1
    RC=$?
    NEPOCH=$(grep -c "^Test_Retrieval Epoch" "$RUNLOG")
    rm -rf "$EXPROOT"
    T1=$(date +%s)
    DONE=$((DONE+1))
    if [ $RC -ne 0 ] || [ "$NEPOCH" -ne 60 ]; then
      NFAIL=$((NFAIL+1))
      echo "PROGRESS ds=$DS arm=$ARM seed=$SEED STATUS=FAIL rc=$RC nline=$NEPOCH secs=$((T1-T0))"
    fi
    if [ $((DONE % 6)) -eq 0 ] || [ $DONE -eq $TOTAL ]; then
      echo "PROGRESS_TOTAL done=$DONE/$TOTAL fail=$NFAIL last=$TAG elapsed=$((T1-START_TS))s eta=$(( (T1-START_TS)*(TOTAL-DONE)/(DONE>0?DONE:1) ))s"
    fi
  done
done
echo "GRID_DONE $DONE/$TOTAL fail=$NFAIL at $(date -Is) elapsed=$(( $(date +%s) - START_TS ))s"
