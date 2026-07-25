#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=fuscat
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# FUSION-CONCAT head family (goal-hook continuation) -- prereg refine-logs/FUSIONCAT_PREREG.md,
# recon refine-logs/FUSIONSWAP_FORENSIC_RECON.md (934bc9a), promoted by orchestrator finding F83.
# ONE sbatch = ONE pre-registered family = ONE multiplicity bite:
#   1 arm (concat) x 2 datasets x 3 seeds = 6 head-only runs on CACHED LoRA features.
#     concat: --fusion_mode "concat"  (in-repo branch classifier.py:85-86,:138-139; input_shape=map_dim*2)
#   ZH floor  = job 13150 (Qwen2.5-VL-7B-Instruct-LoRA_HF,        group RAC_video_b3_lora)
#   HateMM fl = job 13241 (Qwen2.5-VL-7B-Instruct-LoRA-curric_HF, group RAC_video_lora_curric)
# Each run reuses the byte-identical enc3seed_lora_curric / ncafam_family python command; the ONLY
# differences from the deployed align floor are (i) --fusion_mode "align" -> "concat", (ii) the fresh
# GROUP_NAME (RAC_video_fuscat), (iii) the distinct --exp_comment tag (_${MODEL}_fuscat). ZERO source-code
# edit (concat is an already-wired first-class fusion branch; only align was ever run on video). The
# banked floors are NOT re-run here (they are the paired controls, re-derived in the prereg). --force stays
# False so no banked arm is ever overwritten. Features cached => each run ~20-50 s. ~0.1 GPU-h total.
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
GROUP_NAME=RAC_video_fuscat
WARMUP=5

# config list: "DATASET MODEL SEED"
CONFIGS=(
  "MHC_zh $ZH 0"
  "MHC_zh $ZH 1"
  "MHC_zh $ZH 2"
  "HateMM $HM 0"
  "HateMM $HM 1"
  "HateMM $HM 2"
)

run_one () {
  local DATASET="$1" MODEL="$2" SEED="$3"
  local RUNLOG="slurm/logs/fuscat_${DATASET}_${MODEL}_seed${SEED}_${SLURM_JOB_ID:-$$}.trainlog"
  echo ""
  echo "########## RUN: DATASET=$DATASET MODEL=$MODEL SEED=$SEED FUSION=concat -> $RUNLOG ##########"
  python ./src/run_rac.py --batch_size 64 \
      --lr 0.0001 --epochs 30 --topk 20 --dataset "$DATASET" \
      --model "$MODEL" \
      --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
      --fusion_mode "concat" \
      --hard_negatives_loss True --no_hard_negatives 1 \
      --final_eval False --seed "$SEED" --group_name "$GROUP_NAME" \
      --metric "cos" --loss "triplet" --batch_norm False \
      --hybrid_loss True --warmup "$WARMUP" \
      --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
      --lambda_seg 0 --seg_mode full --num_subclips 4 \
      --em_rounds 2 --consensus_topk 10 --consensus_margin 0.2 \
      --exp_comment "_${MODEL}_fuscat" \
      --Faiss_GPU False --force False 2>&1 | tee "$RUNLOG"

  echo "-------- VAL-SELECTED (warmup>=5) + FINAL-EPOCH readout: $DATASET / $MODEL / seed$SEED --------"
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
  read -r D M S <<< "$cfg"
  run_one "$D" "$M" "$S"
done

echo ""
echo "======== fuscat ALL DONE ($SLURM_JOB_ID) ========"
bash scripts/b2_push.sh logging "logs/fuscat" || true
