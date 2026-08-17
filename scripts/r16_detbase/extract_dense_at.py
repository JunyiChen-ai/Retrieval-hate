#!/usr/bin/env python
"""R16-DETBASE: dense 4-FPS AUDIO and TEXT features for HateClipSeg, aligned to the visual grid.

Paper (§4.1): "Text features are extracted using a frozen BERT-Base, encoding words from
[t_i-n, t_i] with n = 2 seconds.  Audio features are extracted using frozen Wav2Vec-Emotion
over [t_i-n, t_i] with n = 4 seconds."

Deviations, declared in `idea-stage/R16_DETBASE_FREEZE.md` §2:
  * audio — running the emotion model separately on 380k overlapping 4-second windows costs 16x
    the audio, so the encoder is run ONCE per video over the whole waveform (in 60 s chunks with
    10 s of context on each side, the overlap discarded) and its 49.9 Hz hidden states are
    mean-pooled over each [t-4s, t] window.  Equivalent up to the model's per-input feature
    normalisation.
  * text — Whisper gives us chunk-level, not word-level, timestamps, so [t-2s, t] selects the
    ASR chunks that OVERLAP the window rather than the words inside it.  Repeated window texts
    are encoded once and cached, which is why this is cheap.

Outputs one .npy per video per channel:
  data/CLIP_Embedding/HateClipSeg/dense4fps_w2vemo/<vid>.npy   (T, 1024)
  data/CLIP_Embedding/HateClipSeg/dense4fps_bertbase/<vid>.npy (T, 768)
T is taken from the visual feature file so the three channels are index-aligned.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/jehc223/Retrieval-hate")
VID = ROOT / "data/video/HateClipSeg/All"
VIS_DIR = ROOT / "data/CLIP_Embedding/HateClipSeg/dense4fps_clipL336"
A_DIR = ROOT / "data/CLIP_Embedding/HateClipSeg/dense4fps_w2vemo"
T_DIR = ROOT / "data/CLIP_Embedding/HateClipSeg/dense4fps_bertbase"
W2V = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
BERT = "bert-base-uncased"
FPS = 4.0
SR = 16000
CHUNK_S = 60.0
CTX_S = 10.0
AUD_WIN = 4.0
TXT_WIN = 2.0


def find_video(vid):
    for ext in (".mp4", ".webm", ".mkv"):
        p = VID / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def load_wav(path):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0",
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--channels", default="audio,text")
    args = ap.parse_args()
    chans = set(args.channels.split(","))
    A_DIR.mkdir(parents=True, exist_ok=True)
    T_DIR.mkdir(parents=True, exist_ok=True)

    gold = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    vids = sorted(gold)
    mine = [v for i, v in enumerate(vids) if i % args.nshard == args.shard]

    asr = {}
    with open(ROOT / "data/ASR/HateClipSeg/test_seen_asrK30_whisper-large-v3.jsonl") as fh:
        for ln in fh:
            d = json.loads(ln)
            asr[d["id"]] = d.get("chunks") or []

    dev = "cuda"
    if "audio" in chans:
        from transformers import Wav2Vec2Model
        am = Wav2Vec2Model.from_pretrained(W2V, torch_dtype=torch.float16).to(dev).eval()
    if "text" in chans:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BERT)
        tm = AutoModel.from_pretrained(BERT, torch_dtype=torch.float16).to(dev).eval()

    t0 = time.time()
    for n, vid in enumerate(mine):
        vpath = VIS_DIR / f"{vid}.npy"
        if not vpath.exists():
            print(f"[SKIP-novis] {vid}", flush=True)
            continue
        T = int(np.load(vpath, mmap_mode="r").shape[0])
        ts = (np.arange(T) / FPS).astype(np.float64)

        if "audio" in chans and not (A_DIR / f"{vid}.npy").exists():
            src = find_video(vid)
            wav = load_wav(src) if src else None
            if wav is None:
                A = np.zeros((T, 1024), dtype=np.float32)
            else:
                hs, rate = [], None
                step = int(CHUNK_S * SR)
                ctx = int(CTX_S * SR)
                pos = 0
                frames = []
                while pos < len(wav):
                    a = max(0, pos - ctx)
                    b = min(len(wav), pos + step + ctx)
                    seg = wav[a:b]
                    if len(seg) < 400:
                        break
                    with torch.no_grad():
                        h = am(torch.tensor(seg, device=dev, dtype=torch.float16)[None]
                               ).last_hidden_state[0].float().cpu().numpy()
                    r = len(seg) / max(h.shape[0], 1)          # samples per hidden frame
                    i0 = int(round((pos - a) / r))
                    i1 = int(round((min(pos + step, len(wav)) - a) / r))
                    frames.append(h[i0:i1])
                    rate = SR / r
                    pos += step
                H = np.concatenate(frames, 0) if frames else np.zeros((1, 1024), np.float32)
                rate = rate or 49.9
                A = np.zeros((T, H.shape[1]), dtype=np.float32)
                for i, t in enumerate(ts):
                    lo = int(max(0, (t - AUD_WIN) * rate))
                    hi = int(min(H.shape[0], max(lo + 1, t * rate)))
                    A[i] = H[lo:hi].mean(0) if hi > lo else 0.0
            np.save(A_DIR / f".{vid}.tmp.npy", A)
            os.replace(A_DIR / f".{vid}.tmp.npy", A_DIR / f"{vid}.npy")

        if "text" in chans and not (T_DIR / f"{vid}.npy").exists():
            ch = asr.get(vid, [])
            texts, cache, TX = [], {}, np.zeros((T, 768), dtype=np.float32)
            for t in ts:
                w = " ".join(c[2].strip() for c in ch
                             if c[1] is not None and c[0] is not None
                             and min(c[1], t) - max(c[0], t - TXT_WIN) > 0).strip()
                texts.append(w)
            uniq = sorted(set(texts))
            for i in range(0, len(uniq), 64):
                bt = uniq[i:i + 64]
                enc = tok(bt, return_tensors="pt", padding=True, truncation=True,
                          max_length=128).to(dev)
                with torch.no_grad():
                    o = tm(**enc).last_hidden_state[:, 0].float().cpu().numpy()
                for j, s in enumerate(bt):
                    cache[s] = o[j]
            for i, s in enumerate(texts):
                if s:
                    TX[i] = cache[s]
            np.save(T_DIR / f".{vid}.tmp.npy", TX)
            os.replace(T_DIR / f".{vid}.tmp.npy", T_DIR / f"{vid}.npy")

        print(f"[{args.shard}] {n+1}/{len(mine)} {vid} T={T} t={time.time()-t0:.0f}s",
              flush=True)
    print(f"[shard {args.shard}] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
