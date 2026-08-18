#!/usr/bin/env python
"""REPRO campaign — unified frame-level ground truth for the four hate-video benchmarks.

Protocol: `idea-stage/REPRO_CAMPAIGN_FREEZE.md` (frozen 2026-08-19, commit 74b9d87).
  §1  4 fps canonical grid, frame i = instant t = i/4 s, T = floor(4*D).
      1 fps mirror grid, frame i = instant t = i s, T = floor(D), used ONLY to
      reproduce the published TEMPORAL_SPAN_LANDSCAPE §1.3 oracle numbers.
  §3  two mandatory controls: gold video-level broadcast, uniform random floor.
  §4  the three GT conventions (HateMM spans json / MHC votes TSV Duration column /
      HateClipSeg original released segments).

Reads annotation files to build an evaluator.  It does not fit anything: no
hyper-parameter anywhere in this file is chosen by looking at a label.

Outputs
-------
data/gt/frame_gt_4fps/<DS>.npz            per-dataset frame labels (see below)
data/gt/frame_gt_4fps/durations_<DS>.json ffprobe duration cache
idea-stage/repro_campaign/gt_controls.json  the §3 control numbers + §11 G1 gate

NPZ contents (DS in HateMM / MHC / MHC_zh / HateClipSeg)
    video_ids   (N,)  str
    split       (N,)  str   train / val / test / none
    duration    (N,)  float64  seconds
    y_video     (N,)  int8     video-level positive per §4 + deviation D1:
                               1 iff the annotation lists >= 1 span (before clipping)
    y_video_ann (N,)  int8     the dataset's own video-level class label
                               (HateMM `label`; MHC Majority_Voting != 'Normal';
                                HateClipSeg any-toxic segment).  Descriptive only.
    n_spans     (N,)  int16
    spans       (N,)  object   float64 (n_i, 2) array of [start, end) seconds, clipped to [0, D)
    y4          (N,)  object   int8 (T4_i,)  4 fps frame labels     <- canonical
    y1          (N,)  object   int8 (T1_i,)  1 fps frame labels     <- landscape cross-check only
    y4_hateonly (N,)  object   HateClipSeg only: class-1 (hateful) restriction
Usage
-----
    python scripts/repro_campaign/build_frame_gt.py            # build + check
    python scripts/repro_campaign/build_frame_gt.py --workers 24
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/home/jehc223/Retrieval-hate")
HOME = Path.home()
OUT_DIR = ROOT / "data/gt/frame_gt_4fps"
CTRL_OUT = ROOT / "idea-stage/repro_campaign/gt_controls.json"

FPS = 4.0
RANDOM_SEEDS = [20250819 + k for k in range(20)]

# TEMPORAL_SPAN_LANDSCAPE §1.3, at 1 fps on the full corpora.  Frozen in
# REPRO_CAMPAIGN_FREEZE §3; the G1 gate is |ours - target| <= 0.005.
LANDSCAPE_TARGET_AP_1FPS = {
    "HateMM": 0.675,
    "MHC": 0.786,
    "MHC_zh": 0.853,
    "HateClipSeg": 0.530,
}
LANDSCAPE_TARGET_BASE_1FPS = {
    "HateMM": 0.2869,
    "MHC": 0.2466,
    "MHC_zh": 0.2539,
    "HateClipSeg": 0.4638,
}
G1_TOL = 0.005

VIDEO_DIR = {
    "HateMM": HOME / "data/HateMM/video",
    # MHC-EN: the transcoded mp4 set is what the dense features are extracted from,
    # so its ffprobe duration is the one the frame grid must agree with.
    "MHC": HOME / "data/Multihateclip/English/video_mp4",
    "MHC_zh": HOME / "data/Multihateclip/Chinese/video",
    "HateClipSeg": ROOT / "data/video/HateClipSeg/All",
}
MHC_LANG = {"MHC": "English", "MHC_zh": "Chinese"}
SPLIT_DIR = {"HateMM": "HateMM", "MHC": "MHC", "MHC_zh": "MHC_zh"}
EXTS = (".mp4", ".webm", ".mkv", ".avi")


# --------------------------------------------------------------------- helpers
def find_video(ds: str, vid: str) -> Path | None:
    d = VIDEO_DIR[ds]
    for ext in EXTS:
        p = d / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def ffprobe_duration(path: Path) -> float | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=120)
        v = float(r.stdout.strip())
        return v if v > 0 else None
    except Exception:
        return None


def durations_for(ds: str, vids: list[str], workers: int) -> dict[str, float]:
    """ffprobe every video once, cached on disk."""
    cache_p = OUT_DIR / f"durations_{ds}.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    todo = [v for v in vids if v not in cache]
    if todo:
        t0 = time.time()
        print(f"[{ds}] ffprobe {len(todo)} videos ...", flush=True)

        def one(v):
            p = find_video(ds, v)
            return v, (ffprobe_duration(p) if p else None)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, (v, d) in enumerate(ex.map(one, todo), 1):
                cache[v] = d
                if i % 200 == 0:
                    print(f"[{ds}] ffprobe {i}/{len(todo)} t={time.time()-t0:.0f}s", flush=True)
        cache_p.write_text(json.dumps(cache, indent=0, sort_keys=True))
        print(f"[{ds}] ffprobe done in {time.time()-t0:.0f}s", flush=True)
    return cache


def read_split_map(ds: str) -> dict[str, str]:
    """video_id -> train/val/test from the project's frozen jsonl splits."""
    out: dict[str, str] = {}
    if ds == "HateClipSeg":
        sp = json.loads((ROOT / "data/gt/HateClipSeg/p11_split.json").read_text())
        for name in ("train", "val", "test"):
            for v in sp.get(name, []):
                out[v] = name
        return out
    base = ROOT / "data/gt" / SPLIT_DIR[ds]
    for name in ("train", "val", "test"):
        p = base / f"{name}.jsonl"
        if not p.exists():
            continue
        with open(p) as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    out[json.loads(ln)["id"]] = name
    return out


def frames_from_spans(spans: np.ndarray, dur: float, fps: float) -> np.ndarray:
    """Half-open [a, b): frame at instant t is positive iff a <= t < b (freeze §4)."""
    n = int(np.floor(dur * fps))
    y = np.zeros(max(n, 0), dtype=np.int8)
    if n <= 0 or len(spans) == 0:
        return y
    t = np.arange(n, dtype=np.float64) / fps
    for a, b in spans:
        y[(t >= a) & (t < b)] = 1
    return y


def clip_spans(raw, dur: float) -> tuple[np.ndarray, int]:
    """Clip to [0, D), drop empties.  Returns (array, n_truncated)."""
    out, trunc = [], 0
    for a, b in raw:
        a, b = float(a), float(b)
        if b > dur:
            b = dur
            trunc += 1
        if a < 0:
            a = 0.0
        if b > a:
            out.append([a, b])
    arr = np.asarray(out, dtype=np.float64) if out else np.zeros((0, 2), np.float64)
    return arr, trunc


# ------------------------------------------------------------------- per-dataset
def load_hatemm() -> tuple[list[str], dict, dict, dict]:
    """-> ids, raw_spans, video_level_label, duration_from_annotation"""
    g = json.loads((ROOT / "data/gt/HateMM/hate_spans.json").read_text())
    ids = sorted(g)
    return (ids,
            {v: g[v].get("spans") or [] for v in ids},
            {v: int(g[v].get("label", 0)) for v in ids},
            {v: float(g[v]["duration"]) for v in ids if g[v].get("duration")})


def load_mhc(ds: str) -> tuple[list[str], dict, dict, dict, dict]:
    """-> ids(on local disk, deduped), raw_spans, video_level_label, {}, extras"""
    lang = MHC_LANG[ds]
    vdir = VIDEO_DIR[ds]
    on_disk = {p.stem for p in vdir.iterdir() if p.suffix in EXTS}
    spans, ylab, order, dups, off_disk = {}, {}, [], [], []
    for sp in ("train", "valid", "test"):
        with open(ROOT / f"data/gt/mhc_votes/mhc_{lang}_{sp}.tsv") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                vid = row["Video_ID"]
                if vid in spans:
                    dups.append(vid)          # freeze §4: keep first occurrence
                    continue
                if vid not in on_disk:
                    off_disk.append(vid)
                    continue
                spans[vid] = list(ast.literal_eval(row["Duration"] or "[]"))
                ylab[vid] = int(row["Majority_Voting"].strip() != "Normal")
                order.append(vid)
    return sorted(order), spans, ylab, {}, {"dup_ids": sorted(set(dups)),
                                           "n_off_disk": len(set(off_disk))}


def load_hateclipseg() -> tuple[list[str], dict, dict, dict, dict]:
    g = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    ids = sorted(g)
    dur = {}
    with open(ROOT / "data/gt/HateClipSeg/video_durations.jsonl") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                r = json.loads(ln)
                dur[r["id"]] = float(r["duration"])
    spans, hate_spans, ylab = {}, {}, {}
    for v in ids:
        segs = g[v]["segments"]
        # freeze §4: primary binary = "any toxic" = any of classes 1..5
        any_tox = [[s[0], s[1]] for s in segs if sum(s[2][1:]) > 0]
        only_h = [[s[0], s[1]] for s in segs if s[2][1]]
        spans[v] = merge_adjacent(any_tox)
        hate_spans[v] = merge_adjacent(only_h)
        ylab[v] = int(len(any_tox) > 0)
        dur.setdefault(v, float(g[v]["duration"]))
    return ids, spans, ylab, dur, {"hate_only_spans": hate_spans}


def merge_adjacent(segs):
    """Released segments tile the timeline; merge touching/overlapping ones."""
    if not segs:
        return []
    segs = sorted(segs)
    out = [list(segs[0])]
    for a, b in segs[1:]:
        if a <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


# ----------------------------------------------------------------------- controls
def controls(y_frames: list[np.ndarray], y_video: np.ndarray) -> dict:
    """§3: gold video-level broadcast ceiling + uniform random floor, pooled frames."""
    y = np.concatenate(y_frames) if y_frames else np.zeros(0, np.int8)
    s_bc = np.concatenate([np.full(len(f), float(v), np.float64)
                           for f, v in zip(y_frames, y_video)])
    res = {
        "n_frames": int(y.size),
        "n_pos": int(y.sum()),
        "base_rate": float(y.mean()) if y.size else 0.0,
    }
    if 0 < y.sum() < y.size:
        res["broadcast_AP"] = float(average_precision_score(y, s_bc))
        res["broadcast_ROC_AUC"] = float(roc_auc_score(y, s_bc))
        aps, rocs = [], []
        for sd in RANDOM_SEEDS:
            r = np.random.default_rng(sd).random(y.size)
            aps.append(average_precision_score(y, r))
            rocs.append(roc_auc_score(y, r))
        res["random_AP_mean"] = float(np.mean(aps))
        res["random_AP_sd"] = float(np.std(aps, ddof=1))
        res["random_ROC_AUC_mean"] = float(np.mean(rocs))
        res["random_ROC_AUC_sd"] = float(np.std(rocs, ddof=1))
    return res


# ---------------------------------------------------------------------------- main
def build(ds: str, workers: int) -> dict:
    extras: dict = {}
    if ds == "HateMM":
        ids, raw, ylab, ann_dur = load_hatemm()
    elif ds in ("MHC", "MHC_zh"):
        ids, raw, ylab, ann_dur, extras = load_mhc(ds)
    else:
        ids, raw, ylab, ann_dur, extras = load_hateclipseg()

    probe = durations_for(ds, ids, workers)
    split_map = read_split_map(ds)

    keep, dur_l, y_video, y_ann, spans_l, y4_l, y1_l = [], [], [], [], [], [], []
    n_trunc, n_nodur = 0, []
    for v in ids:
        d = probe.get(v) or ann_dur.get(v)
        if not d:
            n_nodur.append(v)
            continue
        rs = raw.get(v, [])
        sp, tr = clip_spans(rs, d)
        n_trunc += tr
        keep.append(v)
        dur_l.append(float(d))
        # freeze §4 + deviation D1: the broadcast oracle's positive video set is
        # "the annotation lists at least one span", taken BEFORE duration clipping.
        y_video.append(int(len(rs) > 0))
        y_ann.append(int(ylab.get(v, 0)))
        spans_l.append(sp)
        y4_l.append(frames_from_spans(sp, d, FPS))
        y1_l.append(frames_from_spans(sp, d, 1.0))

    y_video = np.asarray(y_video, np.int8)
    y_ann = np.asarray(y_ann, np.int8)
    obj = lambda L: np.array(L + [None], dtype=object)[:-1]  # noqa: E731
    payload = dict(
        video_ids=np.array(keep),
        split=np.array([split_map.get(v, "none") for v in keep]),
        duration=np.asarray(dur_l, np.float64),
        y_video=y_video,
        y_video_ann=y_ann,
        n_spans=np.asarray([len(s) for s in spans_l], np.int16),
        spans=obj(spans_l),
        y4=obj(y4_l),
        y1=obj(y1_l),
    )
    if ds == "HateClipSeg":
        h4 = []
        for v, d in zip(keep, dur_l):
            hs, _ = clip_spans(extras["hate_only_spans"][v], d)
            h4.append(frames_from_spans(hs, d, FPS))
        payload["y4_hateonly"] = obj(h4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / f"{ds}.npz", allow_pickle=True, **payload)

    rep = {
        "dataset": ds,
        "n_videos": len(keep),
        "n_video_positive_span_derived": int(y_video.sum()),
        "n_video_positive_dataset_label": int(y_ann.sum()),
        "n_videos_with_span_after_clip": int(sum(len(s) > 0 for s in spans_l)),
        "n_span_video_labelled_negative": int(((y_video == 1) & (y_ann == 0)).sum()),
        "n_positive_label_without_span": int(((y_ann == 1) & (y_video == 0)).sum()),
        "n_spans_truncated_to_duration": int(n_trunc),
        "n_dropped_no_duration": len(n_nodur),
        "dropped_no_duration": n_nodur[:20],
        "single_span_frac": float(np.mean([len(s) == 1 for s in spans_l if len(s) > 0]))
        if any(len(s) for s in spans_l) else 0.0,
        "full_corpus_4fps": controls(y4_l, y_video),
        "full_corpus_1fps": controls(y1_l, y_video),
    }
    rep.update({k: v for k, v in extras.items() if k != "hate_only_spans"})

    # per-split controls (freeze §5: the headline table uses these)
    for name in ("train", "val", "test"):
        idx = [i for i, v in enumerate(keep) if split_map.get(v) == name]
        if idx:
            rep[f"split_{name}_4fps"] = controls([y4_l[i] for i in idx], y_video[idx])

    # §11 G1 gate against the published 1 fps landscape numbers
    tgt = LANDSCAPE_TARGET_AP_1FPS[ds]
    got = rep["full_corpus_1fps"].get("broadcast_AP", float("nan"))
    rep["G1_target_AP_1fps"] = tgt
    rep["G1_measured_AP_1fps"] = got
    rep["G1_abs_diff"] = abs(got - tgt)
    rep["G1_pass"] = bool(abs(got - tgt) <= G1_TOL)
    rep["G1_target_base_rate_1fps"] = LANDSCAPE_TARGET_BASE_1FPS[ds]
    rep["G1_measured_base_rate_1fps"] = rep["full_corpus_1fps"]["base_rate"]
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--datasets", default="HateMM,MHC,MHC_zh,HateClipSeg")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CTRL_OUT.parent.mkdir(parents=True, exist_ok=True)
    all_rep = json.loads(CTRL_OUT.read_text()) if CTRL_OUT.exists() else {}
    for ds in args.datasets.split(","):
        rep = build(ds, args.workers)
        all_rep[ds] = rep
        c1, c4 = rep["full_corpus_1fps"], rep["full_corpus_4fps"]
        print(f"\n=== {ds} ===")
        print(f"  videos={rep['n_videos']}  "
              f"span-positive={rep['n_video_positive_span_derived']}  "
              f"label-positive={rep['n_video_positive_dataset_label']}  "
              f"span-but-label-neg={rep['n_span_video_labelled_negative']}  "
              f"label-pos-no-span={rep['n_positive_label_without_span']}")
        print(f"  1fps  frames={c1['n_frames']:>9d}  base={c1['base_rate']:.4f} "
              f"(target {rep['G1_target_base_rate_1fps']:.4f})  "
              f"broadcastAP={c1.get('broadcast_AP', float('nan')):.4f} "
              f"(target {rep['G1_target_AP_1fps']:.3f})  "
              f"randAP={c1.get('random_AP_mean', float('nan')):.4f}")
        print(f"  4fps  frames={c4['n_frames']:>9d}  base={c4['base_rate']:.4f}  "
              f"broadcastAP={c4.get('broadcast_AP', float('nan')):.4f}  "
              f"broadcastROC={c4.get('broadcast_ROC_AUC', float('nan')):.4f}  "
              f"randAP={c4.get('random_AP_mean', float('nan')):.4f}")
        print(f"  G1 {'PASS' if rep['G1_pass'] else 'FAIL'}  |diff|={rep['G1_abs_diff']:.4f}")
    CTRL_OUT.write_text(json.dumps(all_rep, indent=1, sort_keys=True))
    print(f"\nwrote {CTRL_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
