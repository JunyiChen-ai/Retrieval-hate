#!/usr/bin/env python
"""Train-only video-label probe and label-blind test-score producer."""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "reproduction_baselines"))
from hate_common import data as hdata  # noqa: E402

CORPORA = {
    "hatemm": ROOT / "data/CLIP_Embedding/HateMM/dense4fps_w2vemo",
    "hateclipseg": ROOT / "data/CLIP_Embedding/HateClipSeg/dense4fps_w2vemo",
}
HCS_VIDEO_LABELS = Path(
    "/home/jehc223/data/HateClipSeg/Dataset/video_level_annotation.csv")
VIDEO_DIRS = {
    "hatemm": Path("/home/jehc223/data/HateMM/video"),
    "hateclipseg": ROOT / "data/video/HateClipSeg/All",
}
GRID_DIRS = {
    corpus: ROOT / "results/reproduction/features/clip_b16_1fps" / corpus
    for corpus in CORPORA
}
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".avi")
FPS = 4


def load_seconds(path: Path, length: int | None = None) -> tuple[np.ndarray, int]:
    raw = np.asarray(np.load(path), dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 1024 or raw.shape[0] == 0:
        raise ValueError(f"invalid feature shape {raw.shape} at {path}")
    if not np.isfinite(raw).all():
        raise ValueError(f"non-finite feature at {path}")
    if length is None:
        return raw[::FPS].copy(), 0
    indices = np.arange(int(length), dtype=np.int64) * FPS
    n_padded = int((indices >= raw.shape[0]).sum())
    indices = np.minimum(indices, raw.shape[0] - 1)
    return raw[indices].copy(), n_padded


def score_grid_contract(corpus: str, video_id: str) -> tuple[int, float]:
    """Return the frozen label-blind 1 fps grid length and raw duration.

    The evaluator grid is defined by the corpus audio clock, not necessarily
    the container duration.  Its label-blind counterpart is the frozen 1 fps
    feature grid used by every baseline.  Container duration is retained only
    as an audit value; using its ceiling as the score length is incorrect for
    media whose audio and container clocks differ.
    """
    grid_path = GRID_DIRS[corpus] / f"{video_id}.npy"
    if not grid_path.is_file():
        raise FileNotFoundError(f"canonical 1fps grid missing for {corpus}/{video_id}")
    grid = np.load(grid_path, mmap_mode="r")
    if grid.ndim != 2 or grid.shape[0] <= 0:
        raise ValueError(f"invalid canonical grid {grid.shape} at {grid_path}")
    length = int(grid.shape[0])

    video = next((VIDEO_DIRS[corpus] / f"{video_id}{ext}" for ext in VIDEO_EXTENSIONS
                  if (VIDEO_DIRS[corpus] / f"{video_id}{ext}").is_file()), None)
    if video is None:
        raise FileNotFoundError(f"raw video missing for {corpus}/{video_id}")
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        check=True, capture_output=True, text=True,
    )
    duration = float(proc.stdout.strip())
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid duration for {corpus}/{video_id}: {duration}")
    return length, duration


def load_train_labels(corpus: str, train_ids: list[str]) -> dict[str, int]:
    needed = set(train_ids)
    out: dict[str, int] = {}
    if corpus == "hatemm":
        path = next((Path(p) for p in hdata.HATEMM_ANNOTATION_CANDIDATES
                     if Path(p).is_file()), None)
        if path is None:
            raise FileNotFoundError("HateMM annotation CSV missing")
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                vid = row["video_file_name"].rsplit(".", 1)[0]
                if vid in needed:
                    out[vid] = int(row["label"].strip() == "Hate")
    elif corpus == "hateclipseg":
        if not HCS_VIDEO_LABELS.is_file():
            raise FileNotFoundError("HateClipSeg video-level annotation CSV missing")
        with HCS_VIDEO_LABELS.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                vid = row["Video Id"].strip()
                if vid not in needed:
                    continue
                labels = ast.literal_eval(row["Video-Level Label"])
                out[vid] = int(any(str(x).strip().lower() != "normal" for x in labels))
    else:
        raise ValueError(corpus)
    missing = sorted(needed - set(out))
    if missing:
        raise KeyError(f"{corpus}: {len(missing)} train labels missing")
    return out


def fit_probe(corpus: str) -> tuple[StandardScaler, LogisticRegression, dict]:
    feature_dir = CORPORA[corpus]
    train_ids = hdata.load_split(corpus, "train")
    labels = load_train_labels(corpus, train_ids)
    missing = [v for v in train_ids if not (feature_dir / f"{v}.npy").is_file()]
    if missing:
        raise FileNotFoundError(f"{corpus}: {len(missing)} train features missing")
    class_video_count = {c: sum(labels[v] == c for v in train_ids) for c in (0, 1)}
    arrays, targets, weights = [], [], []
    zero_videos = 0
    for vid in train_ids:
        x, _ = load_seconds(feature_dir / f"{vid}.npy")
        y = int(labels[vid])
        arrays.append(x)
        targets.append(np.full(len(x), y, dtype=np.int64))
        mass = len(train_ids) / (2.0 * class_video_count[y] * len(x))
        weights.append(np.full(len(x), mass, dtype=np.float64))
        zero_videos += int(not np.any(x))
    X = np.concatenate(arrays)
    y = np.concatenate(targets)
    w = np.concatenate(weights)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X, sample_weight=w)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=234)
    model.fit(Xs, y, sample_weight=w)
    return scaler, model, {
        "n_train_videos": len(train_ids),
        "n_train_seconds": int(len(y)),
        "class_video_count": class_video_count,
        "class_weight_sum": {str(c): float(w[y == c].sum()) for c in (0, 1)},
        "zero_feature_train_videos": zero_videos,
        "n_iter": int(model.n_iter_[0]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "corpora": list(CORPORA),
        "feature": "dense4fps_w2vemo",
        "feature_fps": FPS,
        "score_grid": "frozen label-blind clip_b16_1fps row count",
        "raw_container_duration": "audit only; never defines score length",
        "readout": "StandardScaler + LogisticRegression(C=1, lbfgs)",
        "train_weighting": (
            "equal video mass within class, equal class mass, total weight "
            "equals number of train videos"),
        "test_gt_values_read_by_producer": False,
        "test_cohort_membership": "key names of frozen shared-evaluator GT archive",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    producer_report = {}
    for corpus, feature_dir in CORPORA.items():
        scaler, model, fit_report = fit_probe(corpus)
        test_ids = hdata.load_split(corpus, "test")
        with np.load(ROOT / f"results/reproduction/gt/{corpus}_test.npz") as gt_index:
            # Only archive key names define the evaluator cohort; no label
            # array is opened here.
            eligible = [v for v in test_ids if v in gt_index.files]
        scores = {}
        padded_seconds = 0
        zero_videos = 0
        raw_duration_grid_mismatches = []
        for vid in eligible:
            length, raw_duration = score_grid_contract(corpus, vid)
            x, n_pad = load_seconds(feature_dir / f"{vid}.npy", length=length)
            s = model.decision_function(scaler.transform(x)).astype(np.float64)
            if not np.isfinite(s).all() or len(s) != length:
                raise RuntimeError(f"bad score contract for {corpus}/{vid}")
            scores[vid] = s
            padded_seconds += n_pad
            zero_videos += int(not np.any(x))
            if int(math.ceil(raw_duration)) != length:
                raw_duration_grid_mismatches.append({
                    "video_id": vid,
                    "container_duration": raw_duration,
                    "container_ceil": int(math.ceil(raw_duration)),
                    "score_grid_length": length,
                })
        np.savez_compressed(run_dir / f"{corpus}_scores.npz", **scores)
        np.savez_compressed(
            run_dir / f"{corpus}_linear_model.npz",
            scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
            coef=model.coef_, intercept=model.intercept_,
        )
        producer_report[corpus] = {
            **fit_report,
            "n_test_manifest": len(test_ids),
            "n_test_scored": len(scores),
            "padded_test_seconds": padded_seconds,
            "zero_feature_test_videos": zero_videos,
            "raw_duration_grid_mismatch_count": len(raw_duration_grid_mismatches),
            "raw_duration_grid_mismatches": raw_duration_grid_mismatches,
        }
        print(corpus, producer_report[corpus], flush=True)
    (run_dir / "producer_report.json").write_text(json.dumps(producer_report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
