#!/usr/bin/env python
"""Smoke test: FineLAP frame-level audio-phrase scores on one of our audio tracks.

Follows third_party/FineLAP/demo.py, HF `AutoModel` route only (the scripts/infer.sh route
needs fairseq and an unreleased EAT checkpoint). FineLAP is audio-only.
"""
import argparse

import numpy as np
import torch
from transformers import AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("--wav", default="/home/jehc223/Retrieval-hate/data/AV2A_wav/HateClipSeg/bit_0dcMcI6hYjhw.wav")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default=None)
args = ap.parse_args()

PHRASES = ["Speech", "Music", "Shouting", "Laughter", "Crowd"]
CAPTION = ["A man speaks to the camera while music plays"]

device = torch.device(args.device)
model = AutoModel.from_pretrained("AndreasXi/FineLAP", trust_remote_code=True).to(device)
model.eval()
print("[smoke] model loaded")

with torch.no_grad():
    g_txt = model.get_global_text_embeds(CAPTION)
    print(f"[smoke] global_text_embeds {tuple(g_txt.shape)}")
    g_aud = model.get_global_audio_embeds([args.wav])
    print(f"[smoke] global_audio_embeds {tuple(g_aud.shape)}")
    dense = model.get_dense_audio_embeds([args.wav])
    print(f"[smoke] dense_audio_embeds {tuple(dense.shape)}")
    clip_s = model.get_clip_level_score([args.wav], CAPTION)
    print(f"[smoke] clip_level_score {tuple(clip_s.shape)} = {clip_s.flatten().tolist()}")
    frame_s = model.get_frame_level_score([args.wav], PHRASES)
    print(f"[smoke] frame_level_score {tuple(frame_s.shape)} "
          f"range {frame_s.min().item():.4f}..{frame_s.max().item():.4f}")

f = frame_s.cpu().numpy()
for i, p in enumerate(PHRASES):
    print(f"[smoke]   {p}: mean {f[0, i].mean():.4f} max {f[0, i].max():.4f}")

if args.out:
    np.savez(args.out, frame=f, rate=float(f.shape[-1]), phrases=np.array(PHRASES))
    print(f"[smoke] wrote {args.out}")
