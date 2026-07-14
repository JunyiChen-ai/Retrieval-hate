#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=b2_ext32b
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# B2 Stage-E: extract frozen Qwen2.5-VL-32B-Instruct hidden-state embeddings (5120-d)
# for HateMM -> MHC -> MHC_zh, all three splits each, via the UNCHANGED extractor
# src/utils/generate_VideoMLLM_embedding_HF.py. Modelled on gen_embed_mllm.sbatch with
# exactly the pre-registered deltas (exp-encoder-32b-b2.md Stage-E; B2_PREREG_REVIEW.md):
#   (Rev-2, HARD) BOTH --model AND --out_model_tag are passed — omitting the tag would
#       silently write to the 7B cache filenames {split}_Qwen2.5-VL-7B-Instruct_HF.pt.
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (32B bf16 on 80G; proven by the P10
#       scorer, p10_score_ladder.sbatch; the extractor's first 32B run is G-repro-gated).
# set -e => fail-closed between datasets (a crash on HateMM stops MHC/MHC_zh).
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_ID="Qwen/Qwen2.5-VL-32B-Instruct"
OUT_TAG="Qwen2.5-VL-32B-Instruct_HF"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

extract_one () {
    local DATASET="$1"
    echo ""
    echo "########## EXTRACT: DATASET=$DATASET MODEL=$MODEL_ID TAG=$OUT_TAG ##########"
    python src/utils/generate_VideoMLLM_embedding_HF.py \
        --dataset "$DATASET" \
        --num_frames "${NUM_FRAMES:-8}" \
        --device cuda \
        --model "$MODEL_ID" \
        --out_model_tag "$OUT_TAG"

    echo "-------- post-extraction dim/rowcount echo: $DATASET --------"
    python - "$DATASET" "$OUT_TAG" <<'PY'
import sys, torch
ds, tag = sys.argv[1], sys.argv[2]
for split in ("train", "dev_seen", "test_seen"):
    p = f"data/CLIP_Embedding/{ds}/{split}_{tag}.pt"
    try:
        o = torch.load(p, map_location="cpu")
        ids = o["ids"][0]
        print(f"  {p}: rows={len(ids)} img={tuple(o['img_feats'].shape)} "
              f"text={tuple(o['text_feats'].shape)} labels={tuple(o['labels'].shape)}")
    except Exception as e:
        print(f"  {p}: LOAD FAILED: {e}")
PY

    # On success, push 32B embeddings to B2 (copy, keep local for training).
    bash scripts/b2_push.sh "data/CLIP_Embedding/$DATASET" "embeddings/$DATASET"
}

extract_one HateMM
extract_one MHC
extract_one MHC_zh

echo ""
echo "[b2_stage_e] ALL DONE (job ${SLURM_JOB_ID:-$$}). Run G-dims verification, then Stage-C."
