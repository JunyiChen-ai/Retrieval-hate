#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=molmo2_embed
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# Frozen Molmo2-8B (Qwen3-8B LLM + SigLIP2-so400m tower) dual-stream hidden-state
# embeddings for HateMM, in the deployed RGCL cache contract.
# Writes {split}_Molmo2-8B_HF.pt into data/CLIP_Embedding/HateMM/.
#
# Recipe + deviation ledger: refine-logs/MOLMO2_FORENSIC_RECON.md.
# Env: HateVideoVLM (transformers 4.57.6) -- HateVideo's 4.49 CANNOT load Molmo2.
# HateVideo is not mutated (same pattern as P10-c).
#
# Usage: sbatch scripts/slurm/gen_embed_molmo2.sbatch [DATASET]
#   DATASET defaults to HateMM (F44 cap: this lane is HateMM-only).
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideoVLM
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1          # model is a local converted copy; never hit the Hub
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

DATASET="${1:-HateMM}"
MODEL_DIR=/data/jehc223/models/Molmo2-8B-bf16
TAG=Molmo2-8B_HF

echo "======== molmo2 extraction START ($SLURM_JOB_ID) dataset=$DATASET ========"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import transformers, torch; print('transformers', transformers.__version__, '| torch', torch.__version__)"

# ---- smoke: 2 videos of the val split into scratch, never into the real cache ----
SMOKE_OUT="${SCRATCH_SMOKE:-/data/jehc223/RGCL/slurm/logs/molmo2_smoke_out_${SLURM_JOB_ID}}"
echo "-------- SMOKE (2 val videos -> $SMOKE_OUT) --------"
python ./src/utils/generate_VideoMLLM_embedding_molmo2_HF.py \
    --dataset "$DATASET" \
    --model "$MODEL_DIR" \
    --out_model_tag "$TAG" \
    --EXP_FOLDER "$SMOKE_OUT" \
    --splits val --limit 2

python - "$SMOKE_OUT/$DATASET/dev_seen_${TAG}.pt" <<'PY'
import sys, torch
d = torch.load(sys.argv[1], weights_only=False)
i, t = d["img_feats"], d["text_feats"]
assert i.shape[1] == 4096 and t.shape[1] == 4096, (i.shape, t.shape)
assert torch.isfinite(i).all() and torch.isfinite(t).all(), "non-finite features"
assert abs(float(i[0].norm()) - 1.0) < 1e-3, "img not L2-normed"
assert abs(float(t[0].norm()) - 1.0) < 1e-3, "text not L2-normed"
assert not torch.allclose(i, t), "streams collapsed to one vector"
print("SMOKE OK: Dv=Dt=4096, finite, L2-normed, streams distinct")
PY

# ---- full extraction: all three splits into the real cache ----
echo "-------- FULL EXTRACTION (train,val,test) --------"
python ./src/utils/generate_VideoMLLM_embedding_molmo2_HF.py \
    --dataset "$DATASET" \
    --model "$MODEL_DIR" \
    --out_model_tag "$TAG" \
    --splits train,val,test

echo "-------- md5 of the written caches --------"
for s in train dev_seen test_seen; do
  f="data/CLIP_Embedding/$DATASET/${s}_${TAG}.pt"
  [ -f "$f" ] && md5sum "$f"
done

rm -rf "$SMOKE_OUT"
echo "======== molmo2 extraction DONE ($SLURM_JOB_ID) ========"
