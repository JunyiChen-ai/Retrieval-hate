#!/usr/bin/env python
"""R11-SEG: per-K=30-window audio features for HateClipSeg (local only, raw video never leaves).

Stage 1  ffmpeg  <vid>.mp4 -> 16 kHz mono wav in a scratch dir (parallel).
Stage 2  openSMILE eGeMAPSv02 functionals, 88-d per window (CPU, parallel).
Stage 3  wav2vec2 emotion encoder, mean-pooled last hidden state, 1024-d per window (GPU).
         Checkpoint: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim -- the same
         "Wav2Vec-Emotion" family the HateClipSeg paper uses for its audio channel.
         Falls back to facebook/wav2vec2-large-robust if the emotion checkpoint is
         unavailable; the checkpoint actually used is recorded in the output.

Windows come from the canonical grid (idea-stage/r11_seg/out/grid_labels.npz `bounds`).

Output: idea-stage/r11_seg/out/audio_feats.npz
  video_ids (395,), egemaps (395,30,88), w2v (395,30,1024),
  audio_ok (395,) bool, w2v_ckpt (str)
"""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
VID = Path("/home/jehc223/data/HateClipSeg/videos")
WAV = Path("/tmp/claude-135258174/-home-jehc223-Retrieval-hate/bcba499b-babe-4ce2-b144-66370e6f6ffd/scratchpad/hcs_wav")
GRID = ROOT / "idea-stage/r11_seg/out/grid_labels.npz"
OUT = ROOT / "idea-stage/r11_seg/out/audio_feats.npz"
SR = 16000
K = 30
NPROC = int(os.environ.get("R11_NPROC", "12"))
EMO_CKPT = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
FALLBACK = "facebook/wav2vec2-large-robust"


def _find(vid: str) -> Path | None:
    for ext in (".mp4", ".webm", ".mkv"):
        p = VID / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def decode(vid: str):
    out = WAV / f"{vid}.wav"
    if out.exists() and out.stat().st_size > 1000:
        return vid, True
    src = _find(vid)
    if src is None:
        return vid, False
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), "-f", "wav", str(out)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    ok = r.returncode == 0 and out.exists() and out.stat().st_size > 1000
    return vid, ok


_smile = None


def _init_smile():
    global _smile
    import opensmile

    _smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                             feature_level=opensmile.FeatureLevel.Functionals)


def egemaps_one(args):
    global _smile
    vid, bnds = args
    import soundfile as sf

    out = np.zeros((K, 88), dtype=np.float32)
    p = WAV / f"{vid}.wav"
    if not p.exists():
        return vid, out
    data, sr = sf.read(p, dtype="float64")
    n = len(data)
    for k, (a, b) in enumerate(bnds):
        i0, i1 = int(a * sr), min(int(b * sr), n)
        seg = data[i0:i1]
        if len(seg) < int(0.2 * sr):
            continue
        try:
            out[k] = _smile.process_signal(seg, sr).values[0].astype(np.float32)
        except Exception:
            pass
    return vid, out


def main() -> None:
    WAV.mkdir(parents=True, exist_ok=True)
    g = np.load(GRID, allow_pickle=True)
    vids = [str(v) for v in g["video_ids"]]
    bounds = g["bounds"]

    print(f"[stage1] decoding {len(vids)} videos -> wav", flush=True)
    ok = {}
    with ProcessPoolExecutor(NPROC) as ex:
        for i, (v, o) in enumerate(ex.map(decode, vids)):
            ok[v] = o
            if (i + 1) % 50 == 0:
                print(f"  progress decode {i+1}/{len(vids)}", flush=True)
    audio_ok = np.array([ok[v] for v in vids])
    print(f"[stage1] done, ok={audio_ok.sum()}/{len(vids)}", flush=True)

    print("[stage2] eGeMAPSv02 per window", flush=True)
    ege = np.zeros((len(vids), K, 88), dtype=np.float32)
    jobs = [(v, bounds[i]) for i, v in enumerate(vids)]
    with ProcessPoolExecutor(NPROC, initializer=_init_smile) as ex:
        for i, (v, arr) in enumerate(ex.map(egemaps_one, jobs, chunksize=4)):
            ege[vids.index(v)] = arr
            if (i + 1) % 50 == 0:
                print(f"  progress egemaps {i+1}/{len(vids)}", flush=True)
    print("[stage2] done", flush=True)

    print("[stage3] wav2vec2 emotion encoder per window", flush=True)
    import soundfile as sf
    import torch
    from transformers import Wav2Vec2Model

    ckpt = EMO_CKPT
    try:
        model = Wav2Vec2Model.from_pretrained(ckpt)
    except Exception as e:  # pragma: no cover
        print(f"  emotion ckpt unavailable ({e}); falling back to {FALLBACK}", flush=True)
        ckpt = FALLBACK
        model = Wav2Vec2Model.from_pretrained(ckpt)
    model = model.eval().half().cuda()
    dim = model.config.hidden_size
    w2v = np.zeros((len(vids), K, dim), dtype=np.float32)
    MAXLEN = SR * 20  # cap a window at 20 s of audio

    with torch.no_grad():
        for i, v in enumerate(vids):
            p = WAV / f"{v}.wav"
            if not p.exists():
                continue
            data, sr = sf.read(p, dtype="float32")
            n = len(data)
            segs = []
            for k, (a, b) in enumerate(bounds[i]):
                i0, i1 = int(a * sr), min(int(b * sr), n)
                s = data[i0:i1][:MAXLEN]
                if len(s) < 400:
                    s = np.zeros(400, dtype=np.float32)
                s = (s - s.mean()) / (s.std() + 1e-7)
                segs.append(s)
            L = max(len(s) for s in segs)
            batch = np.zeros((K, L), dtype=np.float32)
            am = np.zeros((K, L), dtype=np.int64)
            for k, s in enumerate(segs):
                batch[k, : len(s)] = s
                am[k, : len(s)] = 1
            x = torch.from_numpy(batch).half().cuda()
            a_t = torch.from_numpy(am).cuda()
            h = model(x, attention_mask=a_t).last_hidden_state.float()
            out_len = h.shape[1]
            fl = model._get_feat_extract_output_lengths(a_t.sum(-1)).clamp(max=out_len)
            m = (torch.arange(out_len, device=h.device)[None, :] < fl[:, None]).float()
            pooled = (h * m[:, :, None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
            w2v[i] = pooled.cpu().numpy()
            if (i + 1) % 25 == 0:
                print(f"  progress w2v {i+1}/{len(vids)}", flush=True)
    print("[stage3] done", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, video_ids=np.array(vids), egemaps=ege, w2v=w2v,
                        audio_ok=audio_ok, w2v_ckpt=np.array(ckpt))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
