#!/usr/bin/env python
"""
molmo2_download_bf16.py -- disk-safe download of allenai/Molmo2-8B.

WHY THIS EXISTS (disk constraint, see refine-logs/MOLMO2_FORENSIC_RECON.md §3):
The upstream repo ships float32 weights = 34.66 GB. The /data quota headroom at
recon time was ~20 GB (270 G used / 290 G soft). A plain snapshot_download would
blow the soft quota before the model was usable.

Strategy: stream one shard at a time -- download fp32 shard, cast floating-point
tensors to bfloat16, write the bf16 shard, delete the fp32 shard. Peak on-disk
footprint is ~18.6 GB (largest fp32 shard 4.98 GB on top of the bf16 shards
already written), final footprint ~17.4 GB.

bf16 is not a compromise here: the deployed Qwen2.5-VL extraction recipe
(src/utils/generate_VideoMLLM_embedding_HF.py) already runs the encoder in
torch.bfloat16, so casting the checkpoint matches the compute dtype of the
floor we are comparing against. Non-floating buffers are left untouched.

Idempotent / resumable: a shard whose bf16 output already exists is skipped.
"""
import argparse
import json
import os
import shutil
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

REPO_ID = "allenai/Molmo2-8B"
SMALL_FILES = [
    "config.json",
    "configuration_molmo2.py",
    "modeling_molmo2.py",
    "processing_molmo2.py",
    "image_processing_molmo2.py",
    "video_processing_molmo2.py",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "model.safetensors.index.json",
]


def gb(n):
    return n / 1e9


def free_headroom_gb():
    """Remaining GB under the 290 G soft quota (best effort; -1 if unknown)."""
    try:
        import subprocess

        out = subprocess.run(["quota", "-s"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "data-data" in line or line.strip().startswith("270G"):
                pass
    except Exception:
        pass
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/jehc223/models/Molmo2-8B-bf16")
    ap.add_argument("--repo", default=REPO_ID)
    args = ap.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, "_fp32_tmp")
    os.makedirs(tmp, exist_ok=True)

    print("[molmo2] target: {}".format(out), flush=True)

    # ---- 1. small files (configs, tokenizer, remote code) -------------------
    for fn in SMALL_FILES:
        dst = os.path.join(out, fn)
        if os.path.exists(dst):
            continue
        p = hf_hub_download(args.repo, fn, local_dir=tmp)
        shutil.move(p, dst)
        print("[molmo2] small file: {}".format(fn), flush=True)

    index = json.load(open(os.path.join(out, "model.safetensors.index.json")))
    shards = sorted(set(index["weight_map"].values()))
    print("[molmo2] {} shards to fetch".format(len(shards)), flush=True)

    # ---- 2. shard-at-a-time fp32 -> bf16 -----------------------------------
    total_bytes = 0
    for i, shard in enumerate(shards, 1):
        dst = os.path.join(out, shard)
        if os.path.exists(dst):
            total_bytes += os.path.getsize(dst)
            print("[molmo2] [{}/{}] {} already converted, skip".format(i, len(shards), shard), flush=True)
            continue

        print("[molmo2] [{}/{}] downloading {} ...".format(i, len(shards), shard), flush=True)
        src = hf_hub_download(args.repo, shard, local_dir=tmp)
        raw = os.path.getsize(src)

        sd = load_file(src)
        conv = {}
        n_cast = 0
        for k, v in sd.items():
            if v.is_floating_point():
                conv[k] = v.to(torch.bfloat16)
                n_cast += 1
            else:
                conv[k] = v
        del sd

        save_file(conv, dst, metadata={"format": "pt"})
        del conv
        os.remove(src)

        new = os.path.getsize(dst)
        total_bytes += new
        print(
            "[molmo2] [{}/{}] {}: {:.2f} GB fp32 -> {:.2f} GB bf16 ({} tensors cast)".format(
                i, len(shards), shard, gb(raw), gb(new), n_cast
            ),
            flush=True,
        )

    # ---- 3. fix the index's advisory total_size ----------------------------
    idx_path = os.path.join(out, "model.safetensors.index.json")
    index["metadata"]["total_size"] = total_bytes
    json.dump(index, open(idx_path, "w"), indent=2)

    # ---- 4. record the dtype deviation next to the weights -----------------
    with open(os.path.join(out, "CONVERSION_NOTE.md"), "w") as f:
        f.write(
            "# Molmo2-8B local conversion\n\n"
            "Source: `{}` (float32 on the Hub, 34.66 GB).\n"
            "This copy stores **bfloat16** weights ({:.2f} GB) -- floating-point tensors were\n"
            "cast fp32->bf16 shard-by-shard at download time to stay under the /data soft quota.\n"
            "Non-floating buffers are bit-identical to upstream.\n\n"
            "bf16 matches the compute dtype of the deployed Qwen2.5-VL extraction recipe\n"
            "(`src/utils/generate_VideoMLLM_embedding_HF.py` loads with `torch_dtype=torch.bfloat16`),\n"
            "so the encoder-swap comparison is dtype-matched.\n\n"
            "Produced by `scripts/analysis/molmo2_download_bf16.py`.\n".format(args.repo, gb(total_bytes))
        )

    shutil.rmtree(tmp, ignore_errors=True)
    print("[molmo2] DONE. bf16 model at {} ({:.2f} GB)".format(out, gb(total_bytes)), flush=True)


if __name__ == "__main__":
    sys.exit(main())
