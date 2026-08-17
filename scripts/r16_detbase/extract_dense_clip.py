#!/usr/bin/env python
"""R16-DETBASE: extract dense 4-FPS CLIP ViT-L/14-336 visual features for HateClipSeg.

Replicates the HateClipSeg paper's localization input ("frozen ViT-Large at each timestamp
t_i", moment rate 4 FPS) with the encoder this project already uses everywhere else
(`openai/clip-vit-large-patch14-336`, `CLIPVisionModel.pooler_output`, 1024-d), so that the
detector features and the project's K=30 window features come from the *same* tower.

Decoding: ffmpeg `fps=4` filter + CLIP's own preprocessing geometry (shortest-side resize to
336, center crop 336, /255, CLIP mean/std).  Output frame i is the video content at t = i/4 s.

Output: one float32 `.npy` per video, shape (T, 1024), at
`data/CLIP_Embedding/HateClipSeg/dense4fps_clipL336/<vid>.npy`.

Usage:  python extract_dense_clip.py --shard 0 --nshard 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPVisionModel

ROOT = Path("/home/jehc223/Retrieval-hate")
VID_DIR = ROOT / "data/video/HateClipSeg/All"
OUT_DIR = ROOT / "data/CLIP_Embedding/HateClipSeg/dense4fps_clipL336"
MODEL = "openai/clip-vit-large-patch14-336"
FPS = 4.0
SIZE = 336
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def find_video(vid: str) -> Path | None:
    for ext in (".mp4", ".webm", ".mkv"):
        p = VID_DIR / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def frame_stream(path: Path, chunk: int = 64):
    """Yield uint8 arrays (n, SIZE, SIZE, 3) of 4-FPS RGB frames."""
    vf = (f"fps={FPS:g},scale=w={SIZE}:h={SIZE}:force_original_aspect_ratio=increase:"
          f"flags=bicubic,crop={SIZE}:{SIZE}")
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0",
           "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    nbytes = SIZE * SIZE * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=nbytes * chunk)
    try:
        while True:
            buf = proc.stdout.read(nbytes * chunk)
            if not buf:
                break
            n = len(buf) // nbytes
            if n == 0:
                break
            yield np.frombuffer(buf[: n * nbytes], dtype=np.uint8).reshape(n, SIZE, SIZE, 3)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "ignore")
        proc.wait()
        if proc.returncode not in (0, None) and err.strip():
            print(f"[ffmpeg] rc={proc.returncode} {path.name}: {err.strip()[:300]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gold = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    vids = sorted(gold.keys())
    mine = [v for i, v in enumerate(vids) if i % args.nshard == args.shard]

    dev = "cuda"
    model = CLIPVisionModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    t0 = time.time()
    for n, vid in enumerate(mine):
        out = OUT_DIR / f"{vid}.npy"
        if out.exists():
            continue
        path = find_video(vid)
        if path is None:
            print(f"[MISS] {vid}", flush=True)
            continue
        feats = []
        with torch.no_grad():
            for arr in frame_stream(path, chunk=args.batch):
                x = torch.from_numpy(np.ascontiguousarray(arr)).to(dev)
                x = x.permute(0, 3, 1, 2).half().div_(255.0)
                x = (x - mean) / std
                o = model(pixel_values=x).pooler_output
                feats.append(o.float().cpu().numpy())
        if not feats:
            print(f"[EMPTY] {vid}", flush=True)
            continue
        F = np.concatenate(feats, 0).astype(np.float32)
        tmp = OUT_DIR / f".{vid}.tmp.npy"
        np.save(tmp, F)
        os.replace(tmp, out)
        dur = gold[vid]["duration"]
        print(f"[{args.shard}] {n+1}/{len(mine)} {vid} T={F.shape[0]} "
              f"(dur={dur:.1f}s -> {dur*FPS:.0f}) t={time.time()-t0:.0f}s", flush=True)
    print(f"[shard {args.shard}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
