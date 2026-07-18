#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=enc3seed
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# CAND-2 curriculum-LoRA-Qwen DRAW-2 REPLICATION head runner (prereg
# refine-logs/CAND2_REP2_PREREG.md). HateMM ONLY -- 3 head-seeds reading the draw-2
# curriculum-LoRA encoder cache (Qwen2.5-VL-7B-Instruct-LoRA-curric-rep2_HF).
# This replicates the draw-1 HateMM curriculum head reads (job 13241) on a SECOND
# independent SFT draw (rep2 SFT seed:1; draw-1 = implicit 42), to convert the F0.2
# single-curriculum-draw caveat into a 2-draw measurement of the F56 HateMM add-over-generic.
# ZH is NOT re-run (draw-1 tied K-C2-2 both protocols; prereg no-ZH-re-run rationale).
# The frozen-CLIP control (HateMM job 12850) and generic-LoRA arm (HateMM job 13235) are
# BANKED and NOT re-run; each pairing is same-code (run_one below is BYTE-IDENTICAL to
# enc3seed.sbatch / enc3seed_lora_curric.sbatch -- differs from the banked controls ONLY
# in --model / --group_name). Curriculum features are cached, so each run is ~20-25 s.
# GROUP_NAME is a fresh group (RAC_video_lora_curric_rep2) so the run creates fresh output
# dirs and NEVER overwrites any banked arm (force stays False); see the prereg for collision.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

CLIP=openai_clip-vit-large-patch14-336_HF   # control arm (HateMM 12850; NOT re-run here)
LORA=Qwen2.5-VL-7B-Instruct-LoRA-curric-rep2_HF
GROUP_NAME=RAC_video_lora_curric_rep2
WARMUP=5

# config list: "DATASET MODEL SEED"
CONFIGS=(
  "HateMM $LORA 0"
  "HateMM $LORA 1"
  "HateMM $LORA 2"
)

run_one () {
  local DATASET="$1" MODEL="$2" SEED="$3"
  local RUNLOG="slurm/logs/enc3s_${DATASET}_${MODEL}_seed${SEED}_${SLURM_JOB_ID:-$$}.trainlog"
  echo ""
  echo "########## RUN: DATASET=$DATASET MODEL=$MODEL SEED=$SEED -> $RUNLOG ##########"
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
echo "======== enc3seed rep2 ALL DONE ($SLURM_JOB_ID) ========"
bash scripts/b2_push.sh logging "logs/enc3seed" || true
