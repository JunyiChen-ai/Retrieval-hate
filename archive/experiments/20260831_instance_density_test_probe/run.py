#!/usr/bin/env python3
"""Test-first diagnostic for same-corpus bag-label instance density."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(REPO))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from macilsd import align  # noqa: E402
from powa_macil.dataset import usable_text_ids  # noqa: E402
from src.instance_density import (CHANNELS, channel_rows, features, fit_models,
                                  tie_neutral_transport)  # noqa: E402


ANCHORS = {
    "hatemm": Path(
        "/home/jehc223/Hate-follow-up/results/reproduction/powa_macil/"
        "final_maskfix_finetune_hatemm_seed234_e5/hatemm/scores.jsonl"
    ),
    "hateclipseg": REPO / (
        "runs/20260831_powa_starting_point/hcs_maskfix_seed234/scores.jsonl"
    ),
}


def load_anchor(path):
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["video_id"]] = np.asarray(row["score_powa"], dtype=float)
    return rows


def summary(report):
    return {
        "pooled_ap": report["pr_auc"],
        "pooled_roc": report["roc_auc"],
        "within_roc": report["per_video"]["macro_auc"],
        "within_n": report["per_video"]["n_videos_both_classes"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True,
                        choices=("hatemm", "hateclipseg"))
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    labels = hdata.load_labels(args.corpus)
    train_ids = usable_text_ids(
        args.corpus, hdata.load_split(args.corpus, "train")
    )
    test_gt = hdata.gt_arrays(args.corpus, "test")
    test_ids = usable_text_ids(
        args.corpus, hdata.load_split(args.corpus, "test")
    )
    test_ids = [video_id for video_id in test_ids if video_id in test_gt]
    if set(test_ids) != set(test_gt):
        raise RuntimeError("test coverage mismatch")

    models = fit_models(args.corpus, train_ids, labels)
    anchors = load_anchor(ANCHORS[args.corpus])
    if set(anchors) != set(test_gt):
        raise RuntimeError("anchor coverage mismatch")
    branches = {"score_anchor": {}}
    for channel in CHANNELS:
        branches[f"score_probe_{channel}"] = {}
        branches[f"transport_{channel}"] = {}

    for video_id in test_ids:
        parts, n_seconds, snippets = features(args.corpus, video_id)
        index = align.snippet_index_for_seconds(snippets, n_seconds)
        anchor = anchors[video_id]
        if len(anchor) != n_seconds or len(anchor) != len(test_gt[video_id]):
            raise RuntimeError(f"timeline mismatch: {video_id}")
        branches["score_anchor"][video_id] = anchor
        for channel in CHANNELS:
            raw = models[channel].decision_function(
                channel_rows(parts, channel)
            )[index]
            branches[f"score_probe_{channel}"][video_id] = raw
            branches[f"transport_{channel}"][video_id] = (
                tie_neutral_transport(anchor, raw)
            )

    positives = {video_id for video_id in test_gt if labels[video_id] == 1}
    reports = {name: evaluate_scores(scores, test_gt, positives)
               for name, scores in branches.items()}
    payload = {
        "date": "2026-08-31", "split": "test",
        "status": "rule10_iterative_developmental_diagnostic",
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "corpus": args.corpus,
        "train_videos": len(train_ids), "test_videos": len(test_ids),
        "metrics": {name: summary(report) for name, report in reports.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "date": "2026-08-31", "corpus": args.corpus, "split": "test",
        "status": "rule10_iterative_developmental_diagnostic",
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "results": reports,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, default=float) + "\n"
    )
    with (args.out_dir / "scores.jsonl").open("w") as handle:
        for video_id in sorted(test_ids):
            handle.write(json.dumps({
                "video_id": video_id,
                **{name: np.asarray(scores[video_id]).tolist()
                   for name, scores in branches.items()},
            }) + "\n")
    (args.out_dir / "config.json").write_text(json.dumps({
        "date": "2026-08-31", "code_version": "current working tree",
        "corpus": args.corpus, "seed": 234, "epochs": 5,
        "max_rows_per_train_video": 200, "alpha": 1e-4,
        "train_split": "train", "evaluation_split": "test",
        "anchor_scores": str(ANCHORS[args.corpus]),
        "channels": list(CHANNELS),
    }, indent=2) + "\n")
    (args.out_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
