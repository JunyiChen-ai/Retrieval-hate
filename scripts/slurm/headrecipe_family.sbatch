#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=headrecipe
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# HEAD-RECIPE family (F68 wave-2) -- prereg refine-logs/HEADRECIPE_PREREG.md.
# ONE sbatch = ONE pre-registered family = ONE multiplicity bite:
#   2 arms x 2 datasets x 3 seeds = 12 head-only runs on CACHED LoRA features.
#     ARM A (SAM):        --sam True --sam_rho 0.05           (tag SAM_rho0.05)
#     ARM B (mod-drop):   --mod_dropout True --mod_dropout_p 0.3 (tag MODDROP_p0.3)
#   ZH floor  = job 13150 (Qwen2.5-VL-7B-Instruct-LoRA_HF, group RAC_video_b3_lora)
#   HateMM fl = job 13241 (Qwen2.5-VL-7B-Instruct-LoRA-curric_HF, group RAC_video_lora_curric)
# Each run reuses the byte-identical enc3seed python command PLUS the arm's additive,
# getattr-gated flags (${ARM_FLAGS}); every other token is identical to enc3seed_lora_hatemm.sbatch.
# The banked floors are NOT re-run here (they are the paired controls, re-derived in the prereg).
# GROUP_NAME is a fresh group (RAC_video_headrecipe); --force stays False so no banked arm is
# ever overwritten. Features cached => each run ~25-50 s (SAM double-steps ~+15-25 s). <0.15 GPU-h.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

ZH=Qwen2.5-VL-7B-Instruct-LoRA_HF            # ZH floor 13150 (paired control; NOT re-run here)
HM=Qwen2.5-VL-7B-Instruct-LoRA-curric_HF     # HateMM floor 13241 (paired control; NOT re-run here)
GROUP_NAME=RAC_video_headrecipe
WARMUP=5
SAM_FLAGS="--sam True --sam_rho 0.05"
MOD_FLAGS="--mod_dropout True --mod_dropout_p 0.3"

# config list: "DATASET MODEL SEED ARM_TAG ARM_FLAGS..." (ARM_FLAGS is the trailing remainder)
CONFIGS=(
  "MHC_zh $ZH 0 SAM_rho0.05 $SAM_FLAGS"
  "MHC_zh $ZH 1 SAM_rho0.05 $SAM_FLAGS"
  "MHC_zh $ZH 2 SAM_rho0.05 $SAM_FLAGS"
  "HateMM $HM 0 SAM_rho0.05 $SAM_FLAGS"
  "HateMM $HM 1 SAM_rho0.05 $SAM_FLAGS"
  "HateMM $HM 2 SAM_rho0.05 $SAM_FLAGS"
  "MHC_zh $ZH 0 MODDROP_p0.3 $MOD_FLAGS"
  "MHC_zh $ZH 1 MODDROP_p0.3 $MOD_FLAGS"
  "MHC_zh $ZH 2 MODDROP_p0.3 $MOD_FLAGS"
  "HateMM $HM 0 MODDROP_p0.3 $MOD_FLAGS"
  "HateMM $HM 1 MODDROP_p0.3 $MOD_FLAGS"
  "HateMM $HM 2 MODDROP_p0.3 $MOD_FLAGS"
)

run_one () {
  local DATASET="$1" MODEL="$2" SEED="$3" ARM="$4"; shift 4; local ARM_FLAGS="$*"
  local RUNLOG="slurm/logs/hr_${ARM}_${DATASET}_${MODEL}_seed${SEED}_${SLURM_JOB_ID:-$$}.trainlog"
  echo ""
  echo "########## RUN: DATASET=$DATASET MODEL=$MODEL SEED=$SEED ARM=$ARM ($ARM_FLAGS) -> $RUNLOG ##########"
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
      --exp_comment "_${MODEL}_${ARM}" \
      --Faiss_GPU False --force False ${ARM_FLAGS} 2>&1 | tee "$RUNLOG"

  echo "-------- VAL-SELECTED (warmup>=5) + FINAL-EPOCH readout: $DATASET / $MODEL / $ARM / seed$SEED --------"
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
if not val or not test:
    print("NO_PARSE"); sys.exit(0)
W=int(os.environ.get("WARMUP",5))
warm=[e for e in val if e>=W] or list(val.keys())
best=max(warm, key=lambda e:(val[e][3], val[e][4]))
a=test.get(best); fe=max(test); b=test[fe]
print(f"VALSEL epoch {best}: TEST F1 {a[0]:.4f} acc {a[3]:.4f} roc {a[4]:.4f}")
print(f"FINAL  epoch {fe}: TEST F1 {b[0]:.4f} acc {b[3]:.4f} roc {b[4]:.4f}")
print(f"RESULT_ROW\t{sys.argv[1]}\tvalsel\t{best}\t{a[0]:.4f}\t{a[3]:.4f}\t{a[4]:.4f}\tfinal\t{fe}\t{b[0]:.4f}\t{b[3]:.4f}\t{b[4]:.4f}")
PY
}

for cfg in "${CONFIGS[@]}"; do
  read -r D M S ARM FLAGS <<< "$cfg"
  run_one "$D" "$M" "$S" "$ARM" $FLAGS
done

echo ""
echo "======== headrecipe ALL DONE ($SLURM_JOB_ID) ========"
bash scripts/b2_push.sh logging "logs/headrecipe" || true
