"""Shared score-loading and rank-normalization helpers for diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def load_score_branch(path: Path, branch: str) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            video_id = str(row["video_id"])
            if video_id in rows:
                raise ValueError(f"duplicate score ID in {path}: {video_id}")
            values = np.asarray(row[branch], dtype=np.float64)
            if values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError(f"invalid score branch in {path}: {video_id}")
            rows[video_id] = values
    return rows


def global_empirical_cdf(
        scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    video_ids = sorted(scores)
    lengths = [len(scores[video_id]) for video_id in video_ids]
    values = np.concatenate([scores[video_id] for video_id in video_ids])
    ranked = ((rankdata(values, method="average") - 1.0) /
              max(len(values) - 1, 1))
    output: dict[str, np.ndarray] = {}
    start = 0
    for video_id, length in zip(video_ids, lengths):
        output[video_id] = ranked[start:start + length]
        start += length
    return output


def compact_frame_metrics(result: dict) -> dict:
    return {
        "pr_auc": float(result["pr_auc"]),
        "roc_auc": float(result["roc_auc"]),
        "within": float(result["per_video"]["macro_auc"]),
        "n_videos": int(result["n_videos"]),
        "n_frames": int(result["n_frames"]),
    }


def passes_all(metrics: dict, thresholds: dict) -> bool:
    return bool(
        metrics["pr_auc"] >= thresholds["pr_auc"]
        and metrics["roc_auc"] >= thresholds["roc_auc"]
        and metrics["within"] >= thresholds["within"]
    )
