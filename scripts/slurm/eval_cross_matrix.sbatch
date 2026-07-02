#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=cross_matrix
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# Runs the cross-dataset kNN memory-transfer matrix using the CLIP heads trained
# by train_clip_heads.sbatch (group RAC_video_CLIP). For each (trained-on A,
# memory=test=T) cell it swaps the memory bank and reports macro-F1/acc.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

GROUP="RAC_video_CLIP"

# Resolve the val-selected best CLIP ckpt for a dataset (warmup>=5).
ckpt_for() {
  python - "$1" "$GROUP" <<'PY'
import sys, glob, os, re
ds, grp = sys.argv[1], sys.argv[2]
WARM = 5
fs = glob.glob(f"logging/Retrieval/{ds}/{grp}/*/ckpt/best_model_*.pt")
recs = []
for f in fs:
    m = re.search(r"best_model_(\d+)_([\d.]+)\.pt$", os.path.basename(f))
    if m:
        recs.append((float(m.group(2)), int(m.group(1)), f))
elig = [r for r in recs if r[1] >= WARM] or recs
elig.sort(key=lambda r: (r[0], r[1]))
print(elig[-1][2] if elig else "")
PY
}

declare -A CK
for DS in HateMM MHC MHC_zh ImpliHateVid; do
    CK[$DS]=$(ckpt_for "$DS")
    echo "[ckpt] $DS -> ${CK[$DS]}"
done

run_cell() {  # A M T use_sim
    local A=$1 M=$2 T=$3 US=$4
    python ./src/eval_cross_dataset.py \
        --ckpt "${CK[$A]}" --trained_on "$A" \
        --memory_dataset "$M" --eval_dataset "$T" \
        --model "openai_clip-vit-large-patch14-336_HF" \
        --path ./data/ --topk 20 --include_val True --use_sim "$US" \
        2>&1 | grep -E "RESULT_JSON|CROSS-DATASET|trained-on|memory|eval/test|macro-F1|acc |roc |pos-F1"
}

echo ""
echo "############## IN-DOMAIN DIAGONAL (use_sim=True; matches quoted baselines) ##############"
for DS in HateMM MHC MHC_zh ImpliHateVid; do
    run_cell "$DS" "$DS" "$DS" True
done

echo ""
echo "############## FULL MATRIX (plain arithmetic vote, use_sim=False) ##############"
# in-domain reference (arithmetic) + the informative cross cells
run_cell HateMM       HateMM       HateMM       False
run_cell MHC          MHC          MHC          False
run_cell MHC_zh       MHC_zh       MHC_zh       False
run_cell ImpliHateVid ImpliHateVid ImpliHateVid False
# cross: train A, swap memory+test to T
run_cell HateMM       MHC          MHC          False
run_cell MHC          HateMM       HateMM       False
run_cell HateMM       ImpliHateVid ImpliHateVid False
run_cell ImpliHateVid HateMM       HateMM       False
run_cell MHC          MHC_zh       MHC_zh       False
run_cell MHC_zh       MHC          MHC          False
# extra informative: ImpliHateVid<->MHC, MHC_zh<->HateMM
run_cell MHC          ImpliHateVid ImpliHateVid False
run_cell ImpliHateVid MHC          MHC          False

echo "======== CROSS MATRIX DONE ========"
