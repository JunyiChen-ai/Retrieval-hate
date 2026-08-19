#!/usr/bin/env python
"""REPRO campaign Wave 1 — 1 fps JPEG frames for the LAVAD chain (CPU only).

LAVAD's `00_extract_frames.sh` writes every native frame (~700-800 GB for our
84 h of video, `MODEL_ASSETS_STATUS §2`).  This writes only the 1 fps grid the
ported chain uses (~6 GB for the test split), so `frames/<DS>/<vid>/000123.jpg`
is the content at `t = 123 s`.  Same JPEGs feed LAVAD stages 01, 03 and 06.

No GPU: runs beside a GPU job without taking the campaign lock.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from blip2_caption import (FRAME_ROOT, NO_VIDEO_STREAM, dataset_rows,  # noqa: E402
                           extract_1fps, find_video)


def one(ds: str, vid: str, nmax: int):
    p = find_video(ds, vid)
    if p is None:
        return ds, vid, 0, "missing_file"
    try:
        files = extract_1fps(p, FRAME_ROOT / ds / vid, nmax)
    except Exception as e:
        return ds, vid, 0, f"{type(e).__name__}: {e}"
    return ds, vid, len(files), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="HateMM,MHC,MHC_zh,HateClipSeg")
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=6000)
    args = ap.parse_args()

    jobs = []
    for ds in args.datasets.split(","):
        (FRAME_ROOT / ds).mkdir(parents=True, exist_ok=True)
        for vid, dur in dataset_rows(ds, args.split):
            if vid in NO_VIDEO_STREAM:
                continue
            if (FRAME_ROOT / ds / vid).exists():
                continue
            jobs.append((ds, vid, min(int(dur) + 2, args.max_frames)))
    print(f"PROGRESS plan videos={len(jobs)}", flush=True)

    n_ok = n_fail = n_frames = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            ds, vid, n, err = f.result()
            if err or n == 0:
                n_fail += 1
                print(f"[fail] {ds}/{vid} {err or 'no_frames'}", flush=True)
            else:
                n_ok += 1
                n_frames += n
            if i % 20 == 0:
                print(f"PROGRESS {i}/{len(jobs)} ok={n_ok} fail={n_fail} "
                      f"frames={n_frames}", flush=True)
    print(f"[done] videos={n_ok} failed={n_fail} frames={n_frames}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
