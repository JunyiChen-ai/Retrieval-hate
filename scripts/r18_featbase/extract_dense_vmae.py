#!/usr/bin/env python
"""R18-FEATBASE: extract dense VideoMAEv2-giant clip features for HateClipSeg.

Backbone: VideoMAEv2 ViT-g/14 (embed_dim 1408, depth 40, tubelet 2, 16 frames, 224x224),
weights `vit_g_hybrid_pt_1200e_k710_ft` -- the UnlabeledHybrid-1M self-supervised pretrain
followed by supervised K710 fine-tuning.  This is the checkpoint the TAL literature
(VideoMAEv2, AdaTAD, OpenTAD) uses to extract THUMOS/ActivityNet features.

Temporal contract -- aligned 1:1 with the existing 4-FPS CLIP grid so the two feature
arrays can be concatenated index-by-index:
  * output row i corresponds to the same instant as CLIP row i, i.e. t = i/4 s;
  * the clip fed to the encoder is 16 frames decoded at 8 FPS, i.e. a 2.0 s window
    centred on t (frames [2i-8, 2i+8) of the 8-FPS decode, edge-clamped);
  * with --hop H the encoder is only evaluated on every H-th output row and the result is
    linearly interpolated back onto the full 4-FPS grid (H=1 -> no interpolation).

Output: float32 `.npy` per video, shape (T, 1408), T == the CLIP array's T.

Usage:  python extract_dense_vmae.py --hop 2 [--shard 0 --nshard 1] [--limit N] [--bench]
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
import torch.nn.functional as F

ROOT = Path("/home/jehc223/Retrieval-hate")
VID_DIR = ROOT / "data/video/HateClipSeg/All"
CLIP_DIR = ROOT / "data/CLIP_Embedding/HateClipSeg/dense4fps_clipL336"
OUT_DIR = ROOT / "data/CLIP_Embedding/HateClipSeg/dense4fps_vmaev2g"
CKPT = None  # resolved from the HF cache at runtime

DEC_FPS = 8.0          # decode rate
CLIP_LEN = 16          # frames per encoder clip -> 2.0 s receptive field
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- model

def sdpa_attention_forward(self, x):
    """Numerically equivalent replacement for the repo's manual softmax attention,
    routed through torch's fused scaled-dot-product kernel."""
    B, N, C = x.shape
    qkv_bias = None
    if self.q_bias is not None:
        qkv_bias = torch.cat(
            (self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
    qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
    x = x.transpose(1, 2).reshape(B, N, -1)
    return self.proj_drop(self.proj(x))


def build_model(device="cuda"):
    from transformers import AutoConfig
    from huggingface_hub import hf_hub_download
    cfg = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-giant", trust_remote_code=True)
    from transformers import AutoModel
    hf = AutoModel.from_pretrained("OpenGVLab/VideoMAEv2-giant", config=cfg,
                                   trust_remote_code=True)
    vit_cls = type(hf.model)
    attn_cls = type(hf.model.blocks[0].attn)
    attn_cls.forward = sdpa_attention_forward

    mc = dict(cfg.model_config)
    mc["use_mean_pooling"] = True      # the K710 head is a mean-pool head (fc_norm present)
    mc["num_classes"] = 710
    model = vit_cls(**mc)

    ckpt_path = hf_hub_download("Sam3000/vit_g_hybrid_pt_1200e_k710_ft.pth",
                                "vit_g_hybrid_pt_1200e_k710_ft.pth")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ("module", "model", "state_dict"):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] {ckpt_path}\n[ckpt] missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}",
          flush=True)
    assert not [m for m in missing if not m.startswith("pos_embed")], missing
    del hf
    model = model.to(device).half().eval()
    return model


# ---------------------------------------------------------------- decode

def find_video(vid: str) -> Path | None:
    for ext in (".mp4", ".webm", ".mkv"):
        p = VID_DIR / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def decode(path: Path) -> np.ndarray:
    """Whole video at DEC_FPS, short-side-resize + centre crop to SIZE, uint8 (N,H,W,3)."""
    vf = (f"fps={DEC_FPS:g},scale=w={SIZE}:h={SIZE}:force_original_aspect_ratio=increase:"
          f"flags=bicubic,crop={SIZE}:{SIZE}")
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0",
           "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    out = subprocess.run(cmd, capture_output=True)
    nbytes = SIZE * SIZE * 3
    n = len(out.stdout) // nbytes
    if n == 0:
        return np.zeros((0, SIZE, SIZE, 3), dtype=np.uint8)
    return np.frombuffer(out.stdout[: n * nbytes], dtype=np.uint8).reshape(n, SIZE, SIZE, 3)


# ---------------------------------------------------------------- main

@torch.no_grad()
def encode_video(model, frames: np.ndarray, rows: np.ndarray, batch: int,
                 mean, std) -> np.ndarray:
    """rows: output indices on the 4-FPS grid to evaluate.  Returns (len(rows), 1408)."""
    n8 = frames.shape[0]
    feats = []
    for b0 in range(0, len(rows), batch):
        idx_rows = rows[b0: b0 + batch]
        # 8-FPS frame indices for each clip: [2i-8, 2i+8), edge clamped
        off = np.arange(-CLIP_LEN // 2, CLIP_LEN // 2)
        idx = (2 * idx_rows[:, None] + off[None, :]).clip(0, n8 - 1)
        clip = torch.from_numpy(np.ascontiguousarray(frames[idx.reshape(-1)]))
        clip = clip.to("cuda", non_blocking=True)
        x = clip.permute(0, 3, 1, 2).half().div_(255.0)
        x = (x - mean) / std
        x = x.view(len(idx_rows), CLIP_LEN, 3, SIZE, SIZE).permute(0, 2, 1, 3, 4)
        f = model.forward_features(x)
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hop", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    vids = sorted(json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text()).keys())
    mine = [v for i, v in enumerate(vids) if i % args.nshard == args.shard]
    if args.limit:
        mine = mine[: args.limit]

    model = build_model()
    mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1).half()
    std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1).half()

    t0 = time.time()
    n_rows_done = 0
    for n, vid in enumerate(mine):
        out = outdir / f"{vid}.npy"
        if out.exists():
            continue
        T = int(np.load(CLIP_DIR / f"{vid}.npy", mmap_mode="r").shape[0])
        path = find_video(vid)
        frames = decode(path) if path is not None else np.zeros((0, SIZE, SIZE, 3), np.uint8)
        if frames.shape[0] == 0:
            # matches R16 deviation D2: undecodable video -> all-zero features
            Fo = np.zeros((T, 1408), dtype=np.float32)
            print(f"[EMPTY] {vid} T={T}", flush=True)
        else:
            rows = np.arange(0, T, args.hop)
            if rows[-1] != T - 1:
                rows = np.append(rows, T - 1)
            td = time.time()
            Fs = encode_video(model, frames, rows, args.batch, mean, std)
            n_rows_done += len(rows)
            if args.hop == 1:
                Fo = Fs
            else:
                grid = np.arange(T, dtype=np.float64)
                Fo = np.empty((T, Fs.shape[1]), dtype=np.float32)
                for d in range(Fs.shape[1]):
                    Fo[:, d] = np.interp(grid, rows.astype(np.float64), Fs[:, d])
            print(f"[{args.shard}] {n+1}/{len(mine)} {vid} T={T} n8={frames.shape[0]} "
                  f"clips={len(rows)} enc={time.time()-td:.1f}s tot={time.time()-t0:.0f}s",
                  flush=True)
        assert Fo.shape[0] == T
        tmp = outdir / f".{vid}.tmp.npy"
        np.save(tmp, Fo.astype(np.float32))
        os.replace(tmp, out)
        if args.bench and n_rows_done > 0:
            el = time.time() - t0
            print(f"[bench] {n_rows_done} clips in {el:.0f}s -> {n_rows_done/el:.1f} clip/s; "
                  f"375395/{args.hop} clips would take "
                  f"{(375395/args.hop)/(n_rows_done/el)/3600:.2f} h", flush=True)
    print(f"[shard {args.shard}] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
