#!/usr/bin/env python
"""REPRO campaign Wave 0 — ZS-ImageBind embeddings for the four hate-video corpora.

Three channels, all from the raw video (the existing dense cache is CLIP, not
ImageBind, so nothing here can be reused):

  image  4 fps    per-frame VISION embedding      -> (T, 1024)
  video  0.5 fps  per 2 s clip VISION embedding   -> (ceil(D/2), 1024)
  audio  0.5 fps  per 2 s clip AUDIO  embedding   -> (ceil(D/2), 1024)

Transforms are the ImageBind ones shipped in `lavad/libs/ImageBind/imagebind/data.py`:
  vision  Resize(short side 224, bicubic) + CenterCrop(224) + /255 + CLIP mean/std
          (`load_and_transform_vision_data`), implemented as an ffmpeg filter so the
          frames can be streamed instead of written to disk as JPEGs.
  video   2 s clips, `UniformTemporalSubsample(num_samples=2)` inside each clip, same
          normalisation (`load_and_transform_video_data`).
  audio   16 kHz mono, `waveform2melspec` (128 mel bins, target_length 204,
          25 ms window / 10 ms shift), normalise mean=-4.268 std=9.138
          (`load_and_transform_audio_data`).

Two declared deviations from the shipped loaders, both recorded in the results table:
  1. the video channel uses the centre crop only, not `SpatialCrop(224, num_crops=3)`;
     one decode then serves both the image and the video channel.
  2. clips tile the whole video at a 2 s stride instead of `ConstantClipsPerVideoSampler`
     spreading a fixed number of clips over it — a per-clip curve is the point here,
     LAVAD's fixed clip count is not.

Embeddings are stored float16 (cosine similarity against two text prompts is all
they are used for). Idempotent: an existing output is skipped; writes go through
a .tmp + os.replace.

Usage
  python scripts/repro_campaign/extract_imagebind.py --dataset HateMM --channels image,video,audio
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
IB_DIR = ROOT / "third_party/lavad/libs/ImageBind"
IB_CKPT = ROOT / "third_party/_ckpt/imagebind_huge.pth"

VIDEO_DIR = {
    "HateMM": HOME / "data/HateMM/video",
    "MHC": HOME / "data/Multihateclip/English/video_mp4",
    "MHC_zh": HOME / "data/Multihateclip/Chinese/video",
    "HateClipSeg": ROOT / "data/video/HateClipSeg/All",
}
EXTS = (".mp4", ".webm", ".mkv", ".avi")
NO_VIDEO_STREAM = {"hate_video_147", "hate_video_292"}  # freeze §12 D2

FPS = 4.0
SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
SR = 16000
CLIP_S = 2.0            # ImageBind's native clip_duration
FRAMES_PER_CLIP = 2     # UniformTemporalSubsample(num_samples=clip_duration)
NUM_MEL_BINS = 128
TARGET_LENGTH = 204
AUD_MEAN, AUD_STD = -4.268, 9.138

OUT_ROOT = ROOT / "data/CLIP_Embedding"


def find_video(ds: str, vid: str):
    d = VIDEO_DIR[ds]
    for ext in EXTS:
        p = d / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def dataset_ids(ds: str) -> list[str]:
    z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
    return sorted(str(v) for v in z["video_ids"])


def decode_frames(path: Path):
    """All frames at 4 fps, short side 224 bicubic + centre crop 224, uint8 RGB."""
    vf = (f"fps={FPS:g},scale=w={SIZE}:h={SIZE}:force_original_aspect_ratio=increase:"
          f"flags=bicubic,crop={SIZE}:{SIZE}")
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0",
           "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    p = subprocess.run(cmd, capture_output=True)
    nb = SIZE * SIZE * 3
    if not p.stdout or len(p.stdout) < nb:
        return None
    n = len(p.stdout) // nb
    return np.frombuffer(p.stdout[: n * nb], dtype=np.uint8).reshape(n, SIZE, SIZE, 3)


def load_wav(path: Path):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0",
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return None
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def atomic_save(d: Path, vid: str, arr: np.ndarray) -> None:
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{vid}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, d / f"{vid}.npy")


def norm_batch(arr_u8: np.ndarray, mean, std, dev, dtype):
    x = torch.from_numpy(np.ascontiguousarray(arr_u8)).to(dev)
    x = x.permute(0, 3, 1, 2).to(dtype).div_(255.0)
    return (x - mean) / std


def melspec(waveform: torch.Tensor):
    """imagebind.data.waveform2melspec, verbatim numerics."""
    import torchaudio
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=SR, use_energy=False,
        window_type="hanning", num_mel_bins=NUM_MEL_BINS, dither=0.0,
        frame_length=25, frame_shift=10)
    fbank = fbank.transpose(0, 1)
    p = TARGET_LENGTH - fbank.size(1)
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, :TARGET_LENGTH]
    return fbank.unsqueeze(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(VIDEO_DIR))
    ap.add_argument("--channels", default="image,video,audio")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    chans = set(args.channels.split(","))
    ds = args.dataset

    sys.path.insert(0, str(IB_DIR))
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType

    dev = "cuda"
    dtype = getattr(torch, args.dtype)
    model = imagebind_model.imagebind_huge(pretrained=False)
    model.load_state_dict(torch.load(IB_CKPT, map_location="cpu"))
    model = model.to(dev, dtype=dtype).eval()

    mean = torch.tensor(MEAN, device=dev, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev, dtype=dtype).view(1, 3, 1, 1)

    dirs = {c: OUT_ROOT / ds / f"imagebind_{c}" for c in ("image", "video", "audio")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    ids = dataset_ids(ds)
    if args.limit:
        ids = ids[: args.limit]

    stats = {"n": 0, "missing": [], "empty": [], "noaudio": [], "oom": []}
    t0 = time.time()
    print(f"[init] ds={ds} chans={sorted(chans)} ids={len(ids)} dtype={args.dtype} "
          f"batch={args.batch}", flush=True)

    for n, vid in enumerate(ids, 1):
        need = {c: c in chans and not (dirs[c] / f"{vid}.npy").exists()
                for c in ("image", "video", "audio")}
        if not any(need.values()):
            continue
        if vid in NO_VIDEO_STREAM:
            need["image"] = need["video"] = False
            if not need["audio"]:
                continue
        path = find_video(ds, vid)
        if path is None:
            stats["missing"].append(vid)
            print(f"[MISS] {vid}", flush=True)
            continue

        frames = decode_frames(path) if (need["image"] or need["video"]) else None
        if (need["image"] or need["video"]) and frames is None:
            stats["empty"].append(vid)
            print(f"[EMPTY-video] {vid}", flush=True)
            need["image"] = need["video"] = False

        try:
            if need["image"]:
                out = []
                with torch.no_grad():
                    for i in range(0, len(frames), args.batch):
                        x = norm_batch(frames[i:i + args.batch], mean, std, dev, dtype)
                        e = model({ModalityType.VISION: x})[ModalityType.VISION]
                        out.append(e.float().cpu().numpy())
                atomic_save(dirs["image"], vid, np.concatenate(out, 0).astype(np.float16))

            if need["video"]:
                step = int(FPS * CLIP_S)                      # 8 frames per 2 s clip
                nclip = max(1, int(np.ceil(len(frames) / step)))
                clips = []
                for k in range(nclip):
                    seg = frames[k * step:(k + 1) * step]
                    if len(seg) == 0:
                        seg = frames[-1:]
                    idx = torch.linspace(0, len(seg) - 1, FRAMES_PER_CLIP).long().numpy()
                    clips.append(seg[idx])                     # (2, 224, 224, 3)
                out = []
                vb = max(1, args.batch // FRAMES_PER_CLIP)
                with torch.no_grad():
                    for i in range(0, len(clips), vb):
                        blk = np.stack(clips[i:i + vb])        # (b, 2, H, W, 3)
                        b, t = blk.shape[:2]
                        x = norm_batch(blk.reshape(b * t, SIZE, SIZE, 3), mean, std, dev, dtype)
                        # (B, S=1, C, T, H, W): forward() reduces over S, so a 5-dim
                        # (B, C, T, H, W) would be misread as S=C. One clip per item.
                        x = x.reshape(b, t, 3, SIZE, SIZE).permute(0, 2, 1, 3, 4)
                        x = x.unsqueeze(1)
                        e = model({ModalityType.VISION: x})[ModalityType.VISION]
                        out.append(e.float().cpu().numpy())
                atomic_save(dirs["video"], vid, np.concatenate(out, 0).astype(np.float16))

            if need["audio"]:
                wav = load_wav(path)
                if wav is None:
                    stats["noaudio"].append(vid)
                    print(f"[NOAUDIO] {vid}", flush=True)
                else:
                    win = int(CLIP_S * SR)
                    nclip = max(1, int(np.ceil(len(wav) / win)))
                    mels = []
                    for k in range(nclip):
                        seg = wav[k * win:(k + 1) * win]
                        if len(seg) < 400:
                            seg = np.pad(seg, (0, 400 - len(seg)))
                        m = melspec(torch.from_numpy(seg.copy())[None])
                        mels.append((m - AUD_MEAN) / AUD_STD)
                    out = []
                    with torch.no_grad():
                        for i in range(0, len(mels), args.batch):
                            x = torch.stack(mels[i:i + args.batch]).to(dev, dtype)
                            e = model({ModalityType.AUDIO: x})[ModalityType.AUDIO]
                            out.append(e.float().cpu().numpy())
                    atomic_save(dirs["audio"], vid, np.concatenate(out, 0).astype(np.float16))
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            stats["oom"].append(vid)
            print(f"[OOM] {vid}: {e}"[:160], flush=True)
            time.sleep(30)
            continue

        stats["n"] += 1
        if n % 10 == 0 or n == len(ids):
            el = time.time() - t0
            r = stats["n"] / max(el, 1e-9)
            print(f"PROGRESS ds={ds} {n}/{len(ids)} done={stats['n']} vid={vid} "
                  f"elapsed={el:.0f}s rate={r:.2f}vid/s "
                  f"eta={(len(ids)-n)/max(r,1e-9):.0f}s "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)

    stats["wall_seconds"] = round(time.time() - t0, 1)
    stats["n_ids"] = len(ids)
    rep = ROOT / f"idea-stage/repro_campaign/imagebind_{ds}.json"
    rep.write_text(json.dumps(stats, indent=1))
    print(f"[done] ds={ds} {stats['n']} videos in {stats['wall_seconds']}s -> {rep}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
