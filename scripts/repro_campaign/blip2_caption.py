#!/usr/bin/env python
"""REPRO campaign Wave 1 — shared BLIP-2 caption corpus at 1 fps.

LAVAD's `scripts/01_caption.sh` captions **every native frame** of every video
(`frame_interval=1`) after `00_extract_frames.sh` has written them all to disk as
JPEG.  `MODEL_ASSETS_STATUS §2` measured 283 MB of JPEG for one 222 s video; the
four corpora are 84 h of video, i.e. ~700-800 GB, which does not fit beside the
other campaign jobs.

**Declared adaptation.** Frames are *streamed* out of ffmpeg at `fps=1` and
captioned in flight; nothing is written to disk but the captions.  The caption
grid is therefore one image per second, `caption[k]` = the video content at
`t = k` s, and LAVAD's `frame_interval=16` center grid (0.533 s at its assumed
30 fps) becomes a 1.0 s center grid.  Recorded in REPRO_CAMPAIGN_RESULTS §K.

Everything about the captioner itself is LAVAD's: `Salesforce/blip2-opt-6.7b-coco`,
float16, unconditional captioning (no text prompt), `model.generate(**inputs)`
with the checkpoint's own generation config, `batch_decode(skip_special_tokens)`
and `.strip()` — copied from `lavad/src/models/image_captioner.py`.

Output: one JSON per video, `<out>/<DS>/<vid>.json`, `{"0": cap, "1": cap, ...}`,
written `.tmp` + `os.replace`.  A video whose JSON already exists is skipped, so
the run is idempotent and a crash costs at most one video.

CLI
  python scripts/repro_campaign/blip2_caption.py --datasets HateMM,MHC,MHC_zh,HateClipSeg \
      --split test --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
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
NO_VIDEO_STREAM = {"hate_video_147", "hate_video_292"}  # freeze §12 D2
MODEL = "Salesforce/blip2-opt-6.7b-coco"


def find_video(ds: str, vid: str):
    d = VIDEO_DIR[ds]
    for ext in EXTS:
        p = d / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def dataset_rows(ds: str, split: str):
    z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
    out = []
    for i, v in enumerate(z["video_ids"]):
        if split != "all" and str(z["split"][i]) != split:
            continue
        out.append((str(v), float(z["duration"][i])))
    return sorted(out)


# ------------------------------------------------------------------ decode ---
def probe_size(path: Path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True, timeout=120)
    line = (r.stdout or "").strip().splitlines()
    if not line:
        return None
    try:
        w, h = line[0].split("x")[:2]
        return int(w), int(h)
    except Exception:
        return None


def stream_frames(path: Path, nmax: int):
    """Yield PIL RGB images at t = 0, 1, 2, ... seconds via ffmpeg `fps=1`.

    Full resolution is piped and the resize is left to BLIP-2's own processor, so
    the pixels the model sees match `00_extract_frames.sh` + `image_captioner.py`
    up to JPEG quantisation (which this path skips entirely).
    """
    from PIL import Image

    wh = probe_size(path)
    if wh is None:
        return
    w, h = wh
    if w <= 0 or h <= 0 or w * h > 4096 * 2304:  # guard absurd frames
        w, h = min(w, 1920), min(h, 1080)
        scale = f",scale={w}:{h}"
    else:
        scale = ""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
           "-vf", f"fps=1{scale}", "-pix_fmt", "rgb24", "-f", "rawvideo",
           "-frames:v", str(nmax), "-"]
    nbytes = w * h * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         bufsize=nbytes * 2)
    try:
        k = 0
        while k < nmax:
            buf = p.stdout.read(nbytes)
            if buf is None or len(buf) < nbytes:
                break
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            yield k, Image.fromarray(arr)
            k += 1
    finally:
        try:
            p.stdout.close()
        except Exception:
            pass
        p.kill()
        p.wait()


def prefetch(items, ds, fn, depth=2):
    """Decode the next video's frames on a worker thread while the GPU captions."""
    q: queue.Queue = queue.Queue(maxsize=depth)

    def work():
        for it in items:
            try:
                q.put((it, fn(ds, it)))
            except Exception as e:  # decode failure is data, not a crash
                q.put((it, ("error", f"{type(e).__name__}: {e}")))
        q.put(None)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    while True:
        got = q.get()
        if got is None:
            return
        yield got


# ------------------------------------------------------------------- main ---
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="HateMM,MHC,MHC_zh,HateClipSeg")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "data/captions/blip2_1fps"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=6000)
    args = ap.parse_args()

    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    out_root = Path(args.out)
    todo = []
    for ds in args.datasets.split(","):
        (out_root / ds).mkdir(parents=True, exist_ok=True)
        for vid, dur in dataset_rows(ds, args.split):
            if vid in NO_VIDEO_STREAM:
                continue
            if (out_root / ds / f"{vid}.json").exists():
                continue
            todo.append((ds, vid, dur))
    if args.limit:
        todo = todo[: args.limit]
    total_sec = sum(d for _, _, d in todo)
    print(f"PROGRESS plan videos={len(todo)} video_hours={total_sec/3600:.2f}", flush=True)
    if not todo:
        print("[done] nothing to caption", flush=True)
        return 0

    dtype = torch.float16
    processor = Blip2Processor.from_pretrained(MODEL)
    model = Blip2ForConditionalGeneration.from_pretrained(MODEL, torch_dtype=dtype)
    model.to("cuda").eval()
    print(f"PROGRESS model_loaded peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB",
          flush=True)

    t0 = time.time()
    n_done = n_frames = n_fail = 0
    sec_done = 0.0
    for ds, vid, dur in todo:
        path = find_video(ds, vid)
        out_p = out_root / ds / f"{vid}.json"
        if path is None:
            n_fail += 1
            continue
        nmax = min(int(dur) + 2, args.max_frames)
        caps: dict[str, str] = {}
        buf_i, buf_im = [], []
        try:
            for k, im in stream_frames(path, nmax):
                buf_i.append(k)
                buf_im.append(im)
                if len(buf_im) == args.batch_size:
                    caps.update(_caption(model, processor, dtype, buf_i, buf_im))
                    buf_i, buf_im = [], []
            if buf_im:
                caps.update(_caption(model, processor, dtype, buf_i, buf_im))
        except Exception as e:
            print(f"[fail] {ds}/{vid} {type(e).__name__}: {e}", flush=True)
            n_fail += 1
            continue
        if not caps:
            print(f"[fail] {ds}/{vid} no_frames_decoded", flush=True)
            n_fail += 1
            continue
        tmp = out_p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(caps, indent=1))
        os.replace(tmp, out_p)
        n_done += 1
        n_frames += len(caps)
        sec_done += dur
        if n_done % 5 == 0 or n_done == len(todo):
            el = time.time() - t0
            rate = n_frames / max(el, 1e-9)
            eta = (total_sec - sec_done) / max(sec_done / max(el, 1e-9), 1e-9)
            print(f"PROGRESS {n_done}/{len(todo)} vids frames={n_frames} "
                  f"{rate:.1f} img/s elapsed={el/60:.1f}min eta={eta/60:.1f}min "
                  f"fail={n_fail}", flush=True)
    el = time.time() - t0
    print(f"[done] videos={n_done} failed={n_fail} frames={n_frames} "
          f"wall={el/60:.1f}min {n_frames/max(el,1e-9):.1f} img/s", flush=True)
    return 0


@torch.no_grad()
def _caption(model, processor, dtype, idxs, images):
    inputs = processor(images=images, return_tensors="pt").to("cuda", dtype=dtype)
    ids = model.generate(**inputs)
    txt = processor.batch_decode(ids, skip_special_tokens=True)
    return {str(i): t.strip() for i, t in zip(idxs, txt)}


if __name__ == "__main__":
    sys.exit(main())
