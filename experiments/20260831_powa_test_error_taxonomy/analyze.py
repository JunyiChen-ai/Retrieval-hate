#!/usr/bin/env python3
"""Read-only POWA test-error taxonomy under Rule 10."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASE))

from hate_common import data as hdata  # noqa: E402
from src.hate_local_features import aligned_local_features  # noqa: E402


CORPORA = ("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg")
SOURCE = REPO / "runs/20260831_powa_error_structure/analysis.json"


def load_rows(path: Path):
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["video_id"]] = np.asarray(row["score_powa"], dtype=float)
    return rows


def macro_auc(scores, gt, ids):
    values = []
    for video_id in ids:
        y, s = gt[video_id], scores[video_id]
        if len(np.unique(y)) == 2:
            values.append(float(roc_auc_score(y, s)))
    return float(np.mean(values)), len(values)


def shifted(values, lag):
    # Positive lag delays the prediction. Replicated edges avoid wraparound.
    index = np.clip(np.arange(len(values)) - lag, 0, len(values) - 1)
    return values[index]


def ranges(binary):
    binary = np.asarray(binary, dtype=bool)
    padded = np.r_[False, binary, False].astype(np.int8)
    change = np.diff(padded)
    return [[int(a), int(b - 1)] for a, b in zip(np.flatnonzero(change == 1),
                                                 np.flatnonzero(change == -1))]


def content_change(features):
    changes = []
    for name in ("audio", "visual", "text"):
        x = np.asarray(features[name], dtype=float)
        delta = np.linalg.norm(np.diff(x, axis=0), axis=1)
        scale = np.median(delta[delta > 0]) if np.any(delta > 0) else 1.0
        changes.append(delta / max(scale, 1e-6))
    return np.mean(changes, axis=0)


def analyze_corpus(corpus, prediction_path):
    scores = load_rows(prediction_path)
    gt = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    ids = sorted(v for v in gt if labels[v] == 1 and len(np.unique(gt[v])) == 2)
    if not set(ids) <= set(scores):
        raise RuntimeError(f"prediction coverage mismatch for {corpus}")

    base_auc, n = macro_auc(scores, gt, ids)
    lag_curve = {}
    for lag in range(-30, 31):
        moved = {v: shifted(scores[v], lag) for v in ids}
        lag_curve[str(lag)] = macro_auc(moved, gt, ids)[0]
    best_lag = max(lag_curve, key=lag_curve.get)

    position = {}
    reverse_position = {}
    per_video = []
    boundary_auc = []
    boundary_recall = []
    for video_id in ids:
        y = np.asarray(gt[video_id], dtype=np.uint8)
        s = np.asarray(scores[video_id], dtype=float)
        if len(y) != len(s):
            raise RuntimeError(f"alignment mismatch {corpus}/{video_id}")
        pos = np.linspace(0.0, 1.0, len(y))
        position[video_id] = pos
        reverse_position[video_id] = 1.0 - pos
        auc = float(roc_auc_score(y, s))
        best_seconds = np.argsort(s, kind="stable")[-min(10, len(s)):][::-1]
        row = {
            "video_id": video_id,
            "auc": auc,
            "duration": len(y),
            "positive_fraction": float(y.mean()),
            "positive_ranges": ranges(y),
            "top_score_seconds": [int(x) for x in best_seconds],
            "top_score_positive_fraction": float(y[best_seconds].mean()),
        }
        try:
            features = aligned_local_features(corpus, video_id)
            change = content_change(features)
            transition = (y[1:] != y[:-1]).astype(np.uint8)
            if len(np.unique(transition)) == 2:
                boundary_auc.append(float(roc_auc_score(transition, change)))
                k = int(transition.sum())
                picked = np.argsort(change, kind="stable")[-k:]
                truth = np.flatnonzero(transition)
                matched = sum(np.any(np.abs(truth - p) <= 2) for p in picked)
                boundary_recall.append(float(matched / max(1, k)))
            row["text_present_fraction"] = float(
                np.linalg.norm(features["text"], axis=1).astype(bool).mean()
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            row["feature_diagnostic_error"] = f"{type(exc).__name__}: {exc}"
        per_video.append(row)

    pos_auc, _ = macro_auc(position, gt, ids)
    rev_auc, _ = macro_auc(reverse_position, gt, ids)
    per_video.sort(key=lambda row: row["auc"])
    positive_fractions = np.asarray([row["positive_fraction"] for row in per_video])
    negative_seconds = np.asarray([
        row["duration"] * (1.0 - row["positive_fraction"]) for row in per_video
    ])
    return {
        "powa_within_roc": base_auc,
        "n_eligible_positive_videos": n,
        "fixed_lag_curve": lag_curve,
        "best_fixed_lag_seconds": int(best_lag),
        "best_fixed_lag_within_roc": float(lag_curve[best_lag]),
        "relative_position_within_roc": pos_auc,
        "reverse_position_within_roc": rev_auc,
        "positive_fraction_median": float(np.median(positive_fractions)),
        "positive_fraction_q25_q75": [
            float(np.quantile(positive_fractions, .25)),
            float(np.quantile(positive_fractions, .75)),
        ],
        "fraction_videos_positive_fraction_gt_0.8": float(
            np.mean(positive_fractions > .8)
        ),
        "negative_seconds_median": float(np.median(negative_seconds)),
        "content_change_boundary_auc_macro": (
            float(np.mean(boundary_auc)) if boundary_auc else None
        ),
        "content_change_topk_boundary_recall_at_2s_macro": (
            float(np.mean(boundary_recall)) if boundary_recall else None
        ),
        "worst_10_videos": per_video[:10],
        "best_10_videos": per_video[-10:][::-1],
    }


def main():
    source = json.loads(SOURCE.read_text())
    corpora = {}
    inputs = {}
    for corpus in CORPORA:
        path = Path(source["corpora"][corpus]["prediction_artifact"])
        inputs[corpus] = {
            "prediction_artifact": str(path),
            "gt_artifact": str(Path(hdata.GT_ROOT) / f"{corpus}_test.npz"),
        }
        corpora[corpus] = analyze_corpus(corpus, path)
    payload = {
        "date": "2026-08-31",
        "split": "test",
        "purpose": "error taxonomy informing later method development",
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "future_test_evidence_status": "iterative/developmental",
        "source_rule10_diagnostic": str(SOURCE),
        "inputs": inputs,
        "corpora": corpora,
    }
    out = REPO / "runs/20260831_powa_test_error_taxonomy/analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
