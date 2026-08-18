#!/usr/bin/env python
"""REPRO campaign — dense 4 fps CLIP-L/336 + wav2vec2-emotion features for
HateMM, MHC-EN and MHC-ZH, in exactly the cache format HateClipSeg already has.

Protocol: `idea-stage/REPRO_CAMPAIGN_FREEZE.md` §1 / §11 G2-G4.

Pipeline is a straight generalisation of the two scripts that built the
HateClipSeg cache, with the same constants so the caches are interchangeable:
  visual  <- scripts/r16_detbase/extract_dense_clip.py
              ffmpeg `fps=4` + scale(shortest side 336, bicubic) + centre crop 336,
              /255, CLIP mean/std, CLIPVisionModel fp16, `pooler_output` (1024-d).
              Output frame i is the video content at t = i/4 s.
  audio   <- scripts/r16_detbase/extract_dense_at.py
              16 kHz mono, Wav2Vec2Model fp16 run once per video in 60 s chunks
              with 10 s of discarded context on each side, hidden states
              mean-pooled over each [t-4s, t] window onto the visual grid.

Both channels write float32 `.npy`, shape (T, 1024), T from the visual decode, to
  data/CLIP_Embedding/<DS>/dense4fps_clipL336/<vid>.npy
  data/CLIP_Embedding/<DS>/dense4fps_w2vemo/<vid>.npy

Idempotent: an existing output file is skipped, so the job can be killed and
restarted. Writes through a `.tmp` file + os.replace so a kill never leaves a
truncated array.

Usage
  python scripts/repro_campaign/extract_dense.py --dataset HateMM --channels visual,audio
  python scripts/repro_campaign/extract_dense.py --dataset HateClipSeg --verify-ids a,b,c \
         --out-suffix _verify           # G3 reproduction check into a side directory
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

ROOT = Path("/home/jehc223/Retrieval-hate")
HOME = Path.home()

VIDEO_DIR = {
    "HateMM": HOME / "data/HateMM/video",
    "MHC": HOME / "data/Multihateclip/English/video_mp4",
    "MHC_zh": HOME / "data/Multihateclip/Chinese/video",
    "HateClipSeg": ROOT / "data/video/HateClipSeg/All",
}
EXTS = (".mp4", ".webm", ".mkv", ".avi")

CLIP_MODEL = "openai/clip-vit-large-patch14-336"
W2V_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
FPS = 4.0
SIZE = 336
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
SR = 16000
CHUNK_S = 60.0
CTX_S = 10.0
AUD_WIN = 4.0


def find_video(ds: str, vid: str) -> Path | None:
    d = VIDEO_DIR[ds]
    for ext in EXTS:
        p = d / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def dataset_ids(ds: str) -> list[str]:
    """Canonical id list = the frame-GT npz built by build_frame_gt.py."""
    z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
    return sorted(str(v) for v in z["video_ids"])


def frame_stream(path: Path, chunk: int):
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
            print(f"[ffmpeg-v] rc={proc.returncode} {path.name}: {err.strip()[:200]}", flush=True)


def load_wav(path: Path):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0",
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def atomic_save(d: Path, vid: str, arr: np.ndarray) -> None:
    tmp = d / f".{vid}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, d / f"{vid}.npy")


def encode_visual(model, path, batch, mean, std, dev):
    feats = []
    with torch.no_grad():
        for arr in frame_stream(path, chunk=batch):
            x = torch.from_numpy(np.ascontiguousarray(arr)).to(dev)
            x = x.permute(0, 3, 1, 2).half().div_(255.0)
            x = (x - mean) / std
            feats.append(model(pixel_values=x).pooler_output.float().cpu().numpy())
    if not feats:
        return None
    return np.concatenate(feats, 0).astype(np.float32)


def encode_audio(am, path, T, dev):
    wav = load_wav(path)
    if wav is None:
        return np.zeros((T, 1024), dtype=np.float32), False
    step, ctx = int(CHUNK_S * SR), int(CTX_S * SR)
    pos, frames, rate = 0, [], None
    while pos < len(wav):
        a = max(0, pos - ctx)
        b = min(len(wav), pos + step + ctx)
        seg = wav[a:b]
        if len(seg) < 400:
            break
        with torch.no_grad():
            h = am(torch.tensor(seg, device=dev, dtype=torch.float16)[None]
                   ).last_hidden_state[0].float().cpu().numpy()
        r = len(seg) / max(h.shape[0], 1)
        i0 = int(round((pos - a) / r))
        i1 = int(round((min(pos + step, len(wav)) - a) / r))
        frames.append(h[i0:i1])
        rate = SR / r
        pos += step
    H = np.concatenate(frames, 0) if frames else np.zeros((1, 1024), np.float32)
    rate = rate or 49.9
    ts = np.arange(T) / FPS
    A = np.zeros((T, H.shape[1]), dtype=np.float32)
    for i, t in enumerate(ts):
        lo = int(max(0, (t - AUD_WIN) * rate))
        hi = int(min(H.shape[0], max(lo + 1, t * rate)))
        if hi > lo:
            A[i] = H[lo:hi].mean(0)
    return A, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(VIDEO_DIR))
    ap.add_argument("--channels", default="visual,audio")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--verify-ids", default="", help="comma list; overrides the id list")
    ap.add_argument("--out-suffix", default="", help="appended to both output dir names")
    args = ap.parse_args()
    chans = set(args.channels.split(","))
    ds = args.dataset

    base = ROOT / "data/CLIP_Embedding" / ds
    V_DIR = base / f"dense4fps_clipL336{args.out_suffix}"
    A_DIR = base / f"dense4fps_w2vemo{args.out_suffix}"
    V_DIR.mkdir(parents=True, exist_ok=True)
    A_DIR.mkdir(parents=True, exist_ok=True)

    ids = ([s for s in args.verify_ids.split(",") if s] if args.verify_ids
           else dataset_ids(ds))
    mine = [v for i, v in enumerate(ids) if i % args.nshard == args.shard]

    dev = "cuda"
    vm = am = None
    if "visual" in chans:
        from transformers import CLIPVisionModel
        vm = CLIPVisionModel.from_pretrained(CLIP_MODEL, torch_dtype=torch.float16).to(dev).eval()
    if "audio" in chans:
        from transformers import Wav2Vec2Model
        am = Wav2Vec2Model.from_pretrained(W2V_MODEL, torch_dtype=torch.float16).to(dev).eval()
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    t0 = time.time()
    stats = {"n": 0, "missing": [], "empty": [], "noaudio": [], "oom_retry": []}
    print(f"[init] ds={ds} chans={sorted(chans)} ids={len(mine)} batch={args.batch} "
          f"vdir={V_DIR} adir={A_DIR}", flush=True)

    for n, vid in enumerate(mine, 1):
        need_v = "visual" in chans and not (V_DIR / f"{vid}.npy").exists()
        need_a = "audio" in chans and not (A_DIR / f"{vid}.npy").exists()
        if not (need_v or need_a):
            continue
        path = find_video(ds, vid)
        if path is None:
            stats["missing"].append(vid)
            print(f"[MISS] {vid}", flush=True)
            continue

        T = None
        if need_v:
            bs = args.batch
            F = None
            while True:
                try:
                    F = encode_visual(vm, path, bs, mean, std, dev)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if bs <= 4:
                        raise
                    bs //= 2
                    stats["oom_retry"].append([vid, bs])
                    print(f"[OOM] {vid} -> batch {bs}", flush=True)
            if F is None:
                stats["empty"].append(vid)
                print(f"[EMPTY] {vid}", flush=True)
                continue
            atomic_save(V_DIR, vid, F)
            T = F.shape[0]
        elif "audio" in chans:
            vp = V_DIR / f"{vid}.npy"
            if not vp.exists():
                print(f"[SKIP-novis] {vid}", flush=True)
                continue
            T = int(np.load(vp, mmap_mode="r").shape[0])

        if need_a and T:
            try:
                A, ok = encode_audio(am, path, T, dev)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                A, ok = np.zeros((T, 1024), np.float32), False
                stats["oom_retry"].append([vid, "audio-zeroed"])
                print(f"[OOM-audio] {vid} zero-filled", flush=True)
            if not ok:
                stats["noaudio"].append(vid)
            atomic_save(A_DIR, vid, A)

        stats["n"] += 1
        if n % 10 == 0 or n == len(mine):
            el = time.time() - t0
            print(f"PROGRESS ds={ds} {n}/{len(mine)} done={stats['n']} vid={vid} T={T} "
                  f"elapsed={el:.0f}s rate={stats['n']/max(el,1e-9):.2f}vid/s "
                  f"eta={(len(mine)-n)/max(stats['n']/max(el,1e-9),1e-9):.0f}s", flush=True)

    stats["wall_seconds"] = round(time.time() - t0, 1)
    stats["n_ids"] = len(mine)
    rep = ROOT / f"idea-stage/repro_campaign/extract_{ds}{args.out_suffix}_shard{args.shard}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(stats, indent=1))
    print(f"[done] ds={ds} {stats['n']} videos in {stats['wall_seconds']}s -> {rep}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
