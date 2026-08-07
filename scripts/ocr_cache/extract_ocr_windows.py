#!/usr/bin/env python
"""Build the K=30 window-aligned OCR cache for the Pay-for-Evidence route.

Grid: one frame per K=30 window, sampled at the window midpoint
      t_k = (k + 0.5) * D / K,  k = 0..K-1.

Datasets / whitelists (test videos are NEVER touched):
  HateMM      -> ids from data/gt/HateMM/{train,val}.jsonl  (744 + 107)
                 durations from data/gt/HateMM/hate_spans.json['<id>']['duration']
                 (only the duration field is read; labels/spans are not used)
  HateClipSeg -> ids + durations from data/gt/HateClipSeg/video_durations.jsonl (395)

Outputs (data/OCR/<dataset>/):
  ocr_windows_K30.jsonl   one line per (video_id, window_k)
  ocr_video.jsonl         one line per video with the concatenated text
  meta.json               engine, versions, config, timing, per-video failures

Usage:
  python extract_ocr_windows.py --dataset HateMM --engine easyocr
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
K = 30


# --------------------------------------------------------------------- engines
class EasyOCREngine:
    name = "easyocr"

    def __init__(self, langs=("en",), gpu=True):
        import easyocr

        self.version = getattr(easyocr, "__version__", "1.7.2")
        self.langs = list(langs)
        self.reader = easyocr.Reader(self.langs, gpu=gpu, verbose=False)
        self.tag = f"easyocr-{self.version}+craft+{'_'.join(self.langs)}"

    def run(self, frames):
        """frames: list of HxWx3 BGR uint8 (same size). -> list[list[det]]"""
        out = []
        for im in frames:
            dets = self.reader.readtext(im)
            out.append(
                [
                    {
                        "text": str(t),
                        "conf": round(float(c), 4),
                        "bbox": [[int(round(p[0])), int(round(p[1]))] for p in box],
                    }
                    for box, t, c in dets
                ]
            )
        return out


class PaddleOCREngine:
    name = "paddleocr"

    def __init__(self, lang="en", gpu=True):
        import paddle
        import paddleocr

        self.version = getattr(paddleocr, "__version__", "?")
        self.paddle_version = paddle.__version__
        self.lang = lang
        self.ocr = paddleocr.PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="gpu" if gpu else "cpu",
        )
        self.tag = f"paddleocr-{self.version}+paddle-{self.paddle_version}+PP-OCRv5+{lang}"

    def run(self, frames):
        out = []
        for im in frames:
            try:
                res = self.ocr.predict(im)
            except Exception:
                out.append([])
                continue
            dets = []
            for page in res:
                d = page if isinstance(page, dict) else getattr(page, "res", {})
                texts = d.get("rec_texts", []) or []
                scores = d.get("rec_scores", []) or []
                polys = d.get("rec_polys", d.get("dt_polys", [])) or []
                for i, t in enumerate(texts):
                    c = float(scores[i]) if i < len(scores) else 0.0
                    if i < len(polys):
                        poly = np.asarray(polys[i]).reshape(-1, 2)
                        bbox = [[int(round(float(x))), int(round(float(y)))] for x, y in poly]
                    else:
                        bbox = []
                    dets.append({"text": str(t), "conf": round(c, 4), "bbox": bbox})
            out.append(dets)
        return out


# ----------------------------------------------------------------------- data
def whitelist(dataset):
    """-> {video_id: duration_seconds}. Test ids are never enumerated."""
    if dataset == "HateMM":
        dur = {}
        spans = json.load(open(ROOT / "data/gt/HateMM/hate_spans.json"))
        ids = []
        for split in ("train", "val"):
            with open(ROOT / f"data/gt/HateMM/{split}.jsonl") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        ids.append(json.loads(line)["id"])
        for v in ids:
            e = spans.get(v)
            dur[v] = float(e["duration"]) if e and e.get("duration") else None
        return dur
    if dataset == "HateClipSeg":
        dur = {}
        with open(ROOT / "data/gt/HateClipSeg/video_durations.jsonl") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    dur[r["id"]] = float(r["duration"])
        return dur
    raise SystemExit(f"unknown dataset {dataset}")


VIDEO_DIR = {
    "HateMM": ROOT / "data/video/HateMM/All",
    "HateClipSeg": ROOT / "data/video/HateClipSeg/All",
}


def video_path(dataset, vid):
    d = VIDEO_DIR[dataset]
    for ext in (".mp4", ".mkv", ".webm", ".avi"):
        p = d / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def ffprobe_duration(path):
    import subprocess

    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        return float(out)
    except Exception:
        return None


def read_frames(path, duration, k=K):
    """Return (frames_bgr, times, n_ok). Midpoint sampling t_k=(k+0.5)D/K."""
    from decord import VideoReader, cpu

    vr = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    n = len(vr)
    fps = float(vr.get_avg_fps()) or 25.0
    if not duration or duration <= 0:
        duration = n / fps
    times = [(i + 0.5) * duration / k for i in range(k)]
    idx = [int(min(max(0, round(t * fps)), n - 1)) for t in times]
    batch = vr.get_batch(idx).asnumpy()  # RGB
    frames = [batch[i][:, :, ::-1].copy() for i in range(batch.shape[0])]  # -> BGR
    return frames, times


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["HateMM", "HateClipSeg"])
    ap.add_argument("--engine", default="easyocr", choices=["easyocr", "paddleocr"])
    ap.add_argument("--lang", default="en")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="drop detections below this confidence at write time")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / f"data/OCR/{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    win_path = out_dir / "ocr_windows_K30.jsonl"
    vid_path = out_dir / "ocr_video.jsonl"
    meta_path = out_dir / "meta.json"

    dur = whitelist(args.dataset)
    ids = sorted(dur)
    if args.limit:
        ids = ids[: args.limit]

    done = set()
    if vid_path.exists():
        with open(vid_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["video_id"])
    todo = [v for v in ids if v not in done]
    print(f"[init] dataset={args.dataset} engine={args.engine} "
          f"whitelist={len(ids)} done={len(done)} todo={len(todo)}", flush=True)

    if args.engine == "easyocr":
        eng = EasyOCREngine(langs=[args.lang] if args.lang != "ch" else ["ch_sim", "en"])
    else:
        eng = PaddleOCREngine(lang=args.lang)
    print(f"[init] engine_version={eng.tag}", flush=True)

    fw = open(win_path, "a", buffering=1)
    fv = open(vid_path, "a", buffering=1)
    failures = []
    t_start = time.time()
    n_frames = 0
    for i, vid in enumerate(todo):
        p = video_path(args.dataset, vid)
        if p is None:
            failures.append({"video_id": vid, "error": "video_missing"})
            print(f"[warn] missing video {vid}", flush=True)
            continue
        D = dur.get(vid) or ffprobe_duration(p)
        try:
            frames, times = read_frames(p, D)
        except Exception as e:  # decoder failure
            failures.append({"video_id": vid, "error": f"decode:{type(e).__name__}:{e}"})
            print(f"[warn] decode failed {vid}: {e}", flush=True)
            continue
        try:
            dets_per_frame = eng.run(frames)
        except Exception as e:
            failures.append({"video_id": vid, "error": f"ocr:{type(e).__name__}:{e}"})
            print(f"[warn] ocr failed {vid}: {e}", flush=True)
            continue
        n_frames += len(frames)
        all_text = []
        n_dets = 0
        n_win_text = 0
        for k, dets in enumerate(dets_per_frame):
            dets = [d for d in dets if d["conf"] >= args.min_conf and d["text"].strip()]
            rec = {
                "video_id": vid,
                "window_k": k,
                "t_mid": round(float(times[k]), 3),
                "texts": dets,
                "engine": eng.name,
                "engine_version": eng.tag,
            }
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_dets += len(dets)
            if dets:
                n_win_text += 1
                all_text.append(" ".join(d["text"].strip() for d in dets))
        joined = " \n".join(all_text)
        fv.write(json.dumps({
            "video_id": vid,
            "duration": round(float(D), 3) if D else None,
            "text": joined,
            "n_windows_with_text": n_win_text,
            "n_dets": n_dets,
            "n_chars": len(joined),
            "engine": eng.name,
            "engine_version": eng.tag,
        }, ensure_ascii=False) + "\n")

        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            el = time.time() - t_start
            rate = (i + 1) / max(el, 1e-6)
            eta = (len(todo) - i - 1) / max(rate, 1e-9)
            print(f"[progress] {i+1}/{len(todo)} videos  frames={n_frames}  "
                  f"elapsed={el/60:.1f}min  {rate*60:.1f} vid/min  "
                  f"eta={eta/60:.1f}min", flush=True)

    fw.close()
    fv.close()
    meta = {
        "dataset": args.dataset,
        "engine": eng.name,
        "engine_version": eng.tag,
        "lang": args.lang,
        "K": K,
        "sampling": "midpoint t_k=(k+0.5)*D/K",
        "min_conf_write": args.min_conf,
        "n_whitelist": len(ids),
        "n_processed_this_run": len(todo) - len(failures),
        "n_failures": len(failures),
        "failures": failures,
        "frames_this_run": n_frames,
        "wall_seconds": round(time.time() - t_start, 1),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if meta_path.exists():
        old = json.load(open(meta_path))
        meta["previous_runs"] = (old.get("previous_runs") or []) + [
            {k2: v2 for k2, v2 in old.items() if k2 != "previous_runs"}
        ]
    json.dump(meta, open(meta_path, "w"), indent=1, ensure_ascii=False)
    print(f"[done] {json.dumps({k2: v2 for k2, v2 in meta.items() if k2 not in ('failures','previous_runs')})}",
          flush=True)


if __name__ == "__main__":
    main()
