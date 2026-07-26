#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=molmo2_probe
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit. NO GPU -- this probe is $0 on the GPU allocation.
#
# MOLMO2 $0 CPU-head probe (refine-logs/MOLMO2_FORENSIC_RECON.md §6-7).
#
# Three arms x 3 seeds, HateMM, all on CPU, all with the byte-identical run_rac.py command
# from scripts/slurm/enc3seed_lora_curric.sbatch. Only --model, --device, --save_embed,
# --group_name, --output_path and --exp_comment differ between this and the deployed floor.
#
#   A. Molmo2-8B_HF                            -- treatment (frozen encoder swap)
#   B. Qwen2.5-VL-7B-Instruct-LoRA-curric_HF   -- strongest same-path floor (promotion bar)
#   C. Qwen2.5-VL-7B-Instruct_HF               -- frozen-vs-frozen identity control
#
# SAME-PATH DISCIPLINE (F87, ERRPAT §8): CPU-trained heads are not bit-exact to the CUDA floor,
# so arms B and C are RE-RUN here rather than quoted from the banked GPU logs. All three arms
# therefore share one machine state. Banked CPU-proxy reference for arm B (ERRPAT §0.2):
# val-sel 0.8775 / 0.8715, final 0.8760 / 0.8699 -- B should reproduce this.
#
# Nothing under logging/ or data/ is written: output_path is scratch.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

OUT_ROOT="${MOLMO2_PROBE_ROOT:-/data/jehc223/home/tmp/claude-135258174/-data-jehc223-RGCL/e8f03e41-3e21-4cea-b12c-29207373bfca/scratchpad/molmo2_probe}"
mkdir -p "$OUT_ROOT"
WARMUP=5

ARMS=(
  "A Molmo2-8B_HF"
  "B Qwen2.5-VL-7B-Instruct-LoRA-curric_HF"
  "C Qwen2.5-VL-7B-Instruct_HF"
)

run_one () {
  local ARM="$1" MODEL="$2" SEED="$3"
  local RUNLOG="$OUT_ROOT/arm${ARM}_seed${SEED}.trainlog"
  echo ""
  echo "########## ARM=$ARM MODEL=$MODEL SEED=$SEED -> $RUNLOG ##########"
  python ./src/run_rac.py --batch_size 64 \
      --lr 0.0001 --epochs 30 --topk 20 --dataset HateMM \
      --model "$MODEL" \
      --proj_dim 1024 --map_dim 1024 --dropout 0.2 0.4 0.1 \
      --fusion_mode "align" \
      --hard_negatives_loss True --no_hard_negatives 1 \
      --final_eval False --seed "$SEED" --group_name "RAC_molmo2_probe_${ARM}" \
      --metric "cos" --loss "triplet" --batch_norm False \
      --hybrid_loss True --warmup "$WARMUP" \
      --majority_voting "arithmetic" --no_pseudo_gold_positives 1 \
      --lambda_seg 0 --seg_mode full --num_subclips 4 \
      --em_rounds 2 --consensus_topk 10 --consensus_margin 0.2 \
      --exp_comment "_molmo2_probe_${ARM}_s${SEED}" \
      --device cpu --save_embed True \
      --output_path "$OUT_ROOT" \
      --Faiss_GPU False --force False 2>&1 | tee "$RUNLOG"

  WARMUP="$WARMUP" ARM="$ARM" SEED="$SEED" python - "$RUNLOG" <<'PY'
import sys, re, os
log = open(sys.argv[1]).read()
val, test = {}, {}
vre = re.compile(r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
tre = re.compile(r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
for m in vre.finditer(log):
    val[int(m.group(1))] = tuple(float(x) for x in m.groups()[1:])
for m in tre.finditer(log):
    test[int(m.group(1))] = tuple(float(x) for x in m.groups()[1:])
if not val or not test:
    print("NO_PARSE"); sys.exit(0)
W = int(os.environ.get("WARMUP", 5))
warm = [e for e in val if e >= W] or list(val.keys())
best = max(warm, key=lambda e: (val[e][3], val[e][4]))
a = test[best]; fe = max(test); b = test[fe]
arm = os.environ["ARM"]; seed = os.environ["SEED"]
print(f"PROBE_ROW\t{arm}\t{seed}\tvalsel\t{best}\t{a[0]:.4f}\t{a[3]:.4f}\t{a[4]:.4f}\tfinal\t{fe}\t{b[0]:.4f}\t{b[3]:.4f}\t{b[4]:.4f}")
PY
}

for arm in "${ARMS[@]}"; do
  read -r A M <<< "$arm"
  for SEED in 0 1 2; do
    run_one "$A" "$M" "$SEED"
  done
done

echo ""
echo "-------- RAW pre-head encoder geometry (all 3 arms) --------"
python ./scripts/analysis/molmo2_geom_diag.py --dataset HateMM \
    --out ./scripts/analysis/molmo2_geom_diag_OUT.json

echo ""
echo "======== molmo2 CPU probe ALL DONE ($SLURM_JOB_ID) ========"
echo "-------- collected PROBE_ROWs --------"
grep -h "^PROBE_ROW" "/data/jehc223/RGCL/slurm/logs/molmo2_probe_${SLURM_JOB_ID}.out" 2>/dev/null || true
