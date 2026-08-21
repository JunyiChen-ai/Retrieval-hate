#!/usr/bin/env python
"""Smoke test: FLAM (openflam) framewise audio-text similarity on one of our audio tracks.

Follows third_party/openflam/test/local_example.py; only the audio path and the text list
are changed. FLAM is audio-only, so the input is the pre-demuxed wav, not the mp4.
"""
import argparse

import librosa
import numpy as np
import torch

import openflam

SR = 48000

ap = argparse.ArgumentParser()
ap.add_argument("--wav", default="/home/jehc223/Retrieval-hate/data/AV2A_wav/HateClipSeg/bit_0dcMcI6hYjhw.wav")
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default=None)
args = ap.parse_args()

TEXTS = [
    "hateful speech targeting a group of people",
    "a person shouting angrily",
    "music",
    "female speaker",
    "male speaker",
]

flam = openflam.OpenFLAM(model_name="v1-base", default_ckpt_path="/tmp/openflam")
flam.to(args.device)
print("[smoke] model loaded")

audio, sr = librosa.load(args.wav, sr=SR)
audio = audio[: int(args.seconds * SR)]
audio_tensor = torch.tensor(audio).unsqueeze(0).to(args.device)
print(f"[smoke] audio {tuple(audio_tensor.shape)} @ {SR} Hz")

with torch.no_grad():
    act = flam.get_local_similarity(audio_tensor, TEXTS, method="unbiased",
                                    cross_product=True).cpu().numpy()
print(f"[smoke] local similarity map shape {act.shape} "
      f"range {act.min():.4f}..{act.max():.4f}")
a = np.asarray(act)
n_frames = a.shape[-1]
print(f"[smoke] native frame rate = {n_frames / args.seconds:.2f} Hz "
      f"({n_frames} frames over {args.seconds} s)")
for i, t in enumerate(TEXTS):
    row = a.reshape(-1, len(TEXTS), n_frames)[0, i] if a.ndim == 3 else a[i]
    print(f"[smoke]   {t!r}: mean {row.mean():.4f} max {row.max():.4f}")

if args.out:
    np.savez(args.out, act=a, rate=n_frames / args.seconds, texts=np.array(TEXTS))
    print(f"[smoke] wrote {args.out}")
