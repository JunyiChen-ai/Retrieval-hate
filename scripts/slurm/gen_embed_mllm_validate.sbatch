#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=mllm_val
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# Validation-only: run the Qwen2.5-VL extractor on the first 3 videos of HateMM train
# and print embedding shapes / norms / pairwise sanity BEFORE the full batch.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

echo "[mllm_val] GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Inline validation harness: extract 3 samples and assert shape/norm/distinctness.
python - <<'PY'
import sys, torch
sys.path.insert(0, "src")
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from utils.generate_VideoMLLM_embedding_HF import (
    load_video_frames, _encode, read_gt, IMG_INSTRUCTION, TEXT_INSTRUCTION,
)

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
device = "cuda"
print("Loading model (bf16, sdpa)...", flush=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None,
).to(device).eval()
proc = AutoProcessor.from_pretrained(MODEL, max_pixels=360*420)
print("hidden_size =", model.config.hidden_size, flush=True)

items = read_gt("data/gt/HateMM/train.jsonl")[:3]
img_vecs, txt_vecs = [], []
for it in items:
    vid = it["id"]
    frames, ok = load_video_frames(f"data/video/HateMM/All/{vid}.mp4", 8)
    assert ok, f"could not decode {vid}"
    iv = _encode(frames, IMG_INSTRUCTION, proc, model, device, 360*420, span="prefix")
    tp = (TEXT_INSTRUCTION + "\nTitle: (none)\nTranscript: " + (it["text"] or "(none)"))
    tv = _encode(frames, tp, proc, model, device, 360*420, span="response")
    img_vecs.append(iv); txt_vecs.append(tv)
    print(f"[{vid}] img shape={tuple(iv.shape)} norm={iv.norm():.4f} | "
          f"text shape={tuple(tv.shape)} norm={tv.norm():.4f} | "
          f"img_has_nan={torch.isnan(iv).any().item()} text_has_nan={torch.isnan(tv).any().item()}",
          flush=True)

I = torch.stack(img_vecs); T = torch.stack(txt_vecs)
assert I.shape[1] == 3584 and T.shape[1] == 3584, "dim != 3584"
assert torch.allclose(I.norm(dim=1), torch.ones(3), atol=1e-3), "img not L2-normed"
assert torch.allclose(T.norm(dim=1), torch.ones(3), atol=1e-3), "text not L2-normed"
img_sim = I @ I.t(); txt_sim = T @ T.t()
print("img cos-sim matrix:\n", img_sim.numpy(), flush=True)
print("text cos-sim matrix:\n", txt_sim.numpy(), flush=True)
# distinct videos should NOT be identical (off-diag < 0.9999)
off = img_sim - torch.eye(3)
assert off.abs().max() < 0.9999, "img embeddings collapsed (identical across videos)"
print("VALIDATION PASSED: shapes=3584, L2-normed, distinct across videos, no NaN.", flush=True)
PY

echo "[mllm_val] done"
