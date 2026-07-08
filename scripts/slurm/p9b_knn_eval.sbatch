#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=p9b_knn
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit. CPU-only (faiss-cpu kNN vote), no GPU.
#
# P9b D3-knn / C3'-knn read-out: kNN vote over the SFT'd-LMM embeddings extracted by the
# p9bx (gen_embed_lora) jobs. Loops all 12 (dataset, tag) cells of the wave, runs
# scripts/analysis/p9_knn_eval.py --test for each, writes JSON to scripts/analysis/p9b_knn_out/.
set -uo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1

OUTDIR=/data/jehc223/RGCL/scripts/analysis/p9b_knn_out
mkdir -p "$OUTDIR"

# (dataset tag) pairs — the 12 wave cells. Extraction tags set by the p9bx SubmitLines.
PAIRS=(
  "MHC_zh p9d3_zh_s0"  "MHC_zh p9d3_zh_s1"  "MHC_zh p9d3_zh_s2"
  "MHC_zh p9c3p_zh_s0" "MHC_zh p9c3p_zh_s1" "MHC_zh p9c3p_zh_s2"
  "MHC    p9d3_en_s0"  "MHC    p9d3_en_s1"  "MHC    p9d3_en_s2"
  "MHC    p9c3p_en_s0" "MHC    p9c3p_en_s1" "MHC    p9c3p_en_s2"
)

for p in "${PAIRS[@]}"; do
  set -- $p
  DS="$1"; TAG="$2"
  TRAINPT="data/CLIP_Embedding/${DS}/train_${TAG}.pt"
  if [ ! -f "$TRAINPT" ]; then
    echo "[p9b_knn] SKIP ${DS}/${TAG} — missing ${TRAINPT}"
    continue
  fi
  echo "[p9b_knn] === ${DS} / ${TAG} ==="
  python scripts/analysis/p9_knn_eval.py \
      --dataset "$DS" --tag "$TAG" --topk 20 --test \
      --out "${OUTDIR}/${TAG}.json" || echo "[p9b_knn] FAILED ${DS}/${TAG}"
done

echo "[p9b_knn] DONE — results in ${OUTDIR}"
