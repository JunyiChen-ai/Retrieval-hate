#!/usr/bin/env python3
"""DeHate as an external-validation corpus (user request 2026-09-26): media, splits and frame gold.

DeHate ships 6689 videos in ~/data/DeHate/{train,val,test}/<id>.mp4 with an official split (the `Split` column of
DeHate_labels.csv), a video label (`Hate`) and hate spans in seconds (`Hate Segment`). Three stages:

    python scripts/dehate/prepare_dehate.py media    # flat symlink dir, 16 kHz mono wav, 1-fps JPEG frames (CPU)
    python scripts/dehate/prepare_dehate.py splits   # results/reproduction/splits/dehate_{train,val,test}.txt
    python scripts/dehate/prepare_dehate.py gt       # results/reproduction/gt/dehate_{val,test}.npz (+ test sidecar)

media writes the same artifacts, with the same commands, as the other corpora:
    ~/data/DeHate/video/<id>.mp4          symlink to ../<split>/<id>.mp4 (one directory, as the extractors expect)
    data/AV2A_wav/DeHate/<id>.wav         ffmpeg -vn -map 0:a:0 -ac 1 -ar 16000 pcm_s16le (scripts/repro_campaign/run_av2a.py)
    data/frames_1fps/DeHate/<id>/%06d.jpg ffmpeg -vf fps=1 -q:v 2 -start_number 0 (scripts/repro_campaign/blip2_caption.py)
A video without an audio stream gets no wav; its duration then comes from the container, as for the other corpora
(scripts/duplex/extract_clip_features.find_duration).

gt follows docs/duplex/FRAME_EVAL_PROTOCOL.md (1 fps grid, t < duration, half-open containment, degenerate spans
dropped) with the DeHate rules of docs/duplex/FRAME_EVAL_PROTOCOL_DEHATE.md: a span is any "(a, b)" pair of the
`Hate Segment` string (also the one "[a, b]" row), in seconds; a hateful video with no usable span is excluded from
localization (rule (b) of the protocol, as for MultiHateClip); non-hateful videos are all-negative. Duration = the
wav duration from the ASR chunk manifest, the wav header second, the container last (find_duration). No labels are
read by the media and splits stages beyond the split column.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = "/home/jehc223/data/DeHate"
LABELS_CSV = os.path.join(DATA, "DeHate_labels.csv")
VIDEO_DIR = os.path.join(DATA, "video")
WAV_DIR = os.path.join(ROOT, "data", "AV2A_wav", "DeHate")
FRAME_DIR = os.path.join(ROOT, "data", "frames_1fps", "DeHate")
SPLIT_DIR = os.path.join(ROOT, "results", "reproduction", "splits")
GT_DIR = os.path.join(ROOT, "results", "reproduction", "gt")
SPLITS = ("train", "val", "test")
_PAIR = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)")
_BARE = re.compile(r"^\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]$")


def read_rows():
    csv.field_size_limit(1 << 30)
    with open(LABELS_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ids = [r["Video ID"].strip() for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate DeHate video ids")
    for r in rows:
        if r["Split"] not in SPLITS or r["Hate"] not in ("0", "1"):
            raise ValueError("unexpected row %r" % {k: r[k] for k in ("Video ID", "Split", "Hate")})
    return rows


def parse_spans(text):
    """All (start, end) second pairs of a `Hate Segment` string, degenerate ones included (dropped later)."""
    text = (text or "").strip()
    pairs = [(float(a), float(b)) for a, b in _PAIR.findall(text)]
    if not pairs:
        m = _BARE.match(text)
        if m:
            pairs = [(float(m.group(1)), float(m.group(2)))]
    return pairs


# ------------------------------------------------------------------ media
def _one_media(vid, split):
    src = os.path.join(DATA, split, vid + ".mp4")
    link = os.path.join(VIDEO_DIR, vid + ".mp4")
    if not os.path.islink(link):
        os.symlink(os.path.join("..", split, vid + ".mp4"), link)
    wav = os.path.join(WAV_DIR, vid + ".wav")
    wav_state = "cached"
    if not (os.path.exists(wav) and os.path.getsize(wav) > 44):
        tmp = wav + ".tmp"
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-vn", "-map", "0:a:0", "-ac", "1",
                            "-ar", "16000", "-acodec", "pcm_s16le", "-f", "wav", tmp], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) <= 44:
            if os.path.exists(tmp):
                os.unlink(tmp)
            wav_state = "no_audio"
        else:
            os.replace(tmp, wav)
            wav_state = "demuxed"
    out = os.path.join(FRAME_DIR, vid)
    n = len([f for f in os.listdir(out) if f.endswith(".jpg")]) if os.path.isdir(out) else 0
    if n == 0:
        tmp = out + ".tmp"
        subprocess.run(["rm", "-rf", tmp])
        os.makedirs(tmp)
        r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", src, "-vf", "fps=1", "-q:v", "2",
                            "-start_number", "0", "-frames:v", "6000", os.path.join(tmp, "%06d.jpg")],
                           capture_output=True, text=True, timeout=3600)
        n = len([f for f in os.listdir(tmp) if f.endswith(".jpg")])
        if n:
            os.replace(tmp, out)
        else:
            subprocess.run(["rm", "-rf", tmp])
    return vid, wav_state, n


def media(workers):
    rows = read_rows()
    for d in (VIDEO_DIR, WAV_DIR, FRAME_DIR):
        os.makedirs(d, exist_ok=True)
    report = {"no_audio": [], "no_frames": [], "demuxed": 0, "cached": 0}
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_one_media, r["Video ID"].strip(), r["Split"]) for r in rows]
        for i, f in enumerate(as_completed(futs), 1):
            vid, wav_state, n = f.result()
            if wav_state == "no_audio":
                report["no_audio"].append(vid)
            else:
                report[wav_state] += 1
            if n == 0:
                report["no_frames"].append(vid)
            if i % 500 == 0:
                print("%d / %d" % (i, len(rows)), flush=True)
    report["no_audio"].sort()
    report["no_frames"].sort()
    with open(os.path.join(WAV_DIR, "media_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("wav demuxed %d cached %d no_audio %d | no_frames %d" % (report["demuxed"], report["cached"],
                                                                 len(report["no_audio"]), len(report["no_frames"])))


# ------------------------------------------------------------------ splits
def splits():
    rows = read_rows()
    os.makedirs(SPLIT_DIR, exist_ok=True)
    for s in SPLITS:
        ids = sorted(r["Video ID"].strip() for r in rows if r["Split"] == s)
        with open(os.path.join(SPLIT_DIR, "dehate_%s.txt" % s), "w") as fh:
            fh.write("\n".join(ids) + "\n")
        print(s, len(ids))


# ------------------------------------------------------------------ gold
def gt():
    sys.path.insert(0, os.path.join(ROOT, "scripts", "duplex"))
    from frame_eval_common import build_gt_array
    from extract_clip_features import CORPORA, find_duration, find_video_path, load_chunk_durations
    spec = CORPORA["dehate"]
    chunk = load_chunk_durations(spec)
    rows = {r["Video ID"].strip(): r for r in read_rows()}
    for s in ("val", "test"):
        ids = [l.strip() for l in open(os.path.join(SPLIT_DIR, "dehate_%s.txt" % s)) if l.strip()]
        arrays, excluded, sources, dropped_degenerate = {}, [], {}, 0
        for vid in ids:
            r = rows[vid]
            dur, src = find_duration(vid, spec, chunk, find_video_path(spec["video_dir"], vid))
            if dur is None:
                raise ValueError("no duration for %s" % vid)
            sources[src.split("/")[0] if "/" in src else src] = sources.get(src.split("/")[0] if "/" in src else src, 0) + 1
            spans = []
            if r["Hate"] == "1":
                raw = parse_spans(r["Hate Segment"])
                spans = [(a, b) for a, b in raw if b > a]
                dropped_degenerate += len(raw) - len(spans)
                if not spans:
                    excluded.append(vid)
                    continue
            arr = build_gt_array(spans, dur)
            if r["Hate"] == "1" and arr.sum() == 0:
                excluded.append(vid)      # every span lies past the end of the media: no localizable gold
                continue
            arrays[vid] = arr
        np.savez_compressed(os.path.join(GT_DIR, "dehate_%s.npz" % s), **arrays)
        side = {"corpus": "dehate", "split": s, "fps": 1.0, "source": LABELS_CSV,
                "rules": "docs/duplex/FRAME_EVAL_PROTOCOL_DEHATE.md",
                "n_videos": len(arrays), "n_hateful": sum(rows[v]["Hate"] == "1" for v in arrays),
                "n_frames": int(sum(len(a) for a in arrays.values())),
                "n_positive_frames": int(sum(int(a.sum()) for a in arrays.values())),
                "duration_sources": sources, "degenerate_spans_dropped": dropped_degenerate,
                "excluded_positive_without_span": sorted(excluded)}
        with open(os.path.join(GT_DIR, "dehate_%s.json" % s), "w") as fh:
            json.dump(side, fh, indent=1)
        print(s, {k: v for k, v in side.items() if k != "excluded_positive_without_span"},
              "excluded", len(excluded))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("media", "splits", "gt"))
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    {"media": lambda: media(a.workers), "splits": splits, "gt": gt}[a.stage]()
