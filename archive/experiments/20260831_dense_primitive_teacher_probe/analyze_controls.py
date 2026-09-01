#!/usr/bin/env python3
"""Frozen attribution controls for the dense typed-primitive qualification."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASE))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402


CORPORA = ("hatemm", "hateclipseg")
MARGIN = 0.010


def noisy_or(values):
    return 1.0 - np.prod(1.0 - np.clip(np.asarray(values), 0.0, 1.0))


def policy(corpus, primitive):
    hostile, target, violence, sexual, self_harm, context = primitive
    targeted = hostile * target * (1.0 - context)
    if corpus == "hatemm":
        return targeted
    if corpus == "hateclipseg":
        abuse = hostile * (1.0 - target) * (1.0 - context)
        return noisy_or((targeted, abuse, violence, sexual, self_harm))
    raise ValueError(corpus)


def controls(corpus, primitive):
    return {
        "compiled_policy": policy(corpus, primitive),
        "hostile_only": float(primitive[0]),
        "untyped_harmful_max": float(np.max(primitive[[0, 2, 3, 4]])),
        "cyclic_primitive_policy": policy(corpus, np.roll(primitive, 1)),
    }


def densify(length, rows, key):
    total = np.zeros(length, dtype=np.float64)
    count = np.zeros(length, dtype=np.float64)
    for row in rows:
        start, end = row["span"]
        total[start:end] += row["control_scores"][key]
        count[start:end] += 1.0
    if not np.all(count > 0):
        raise RuntimeError("raw windows do not cover the full video")
    return total / count


def analyze_corpus(run_dir, corpus):
    gt = hdata.gt_arrays(corpus, "val")
    grouped = defaultdict(list)
    raw_path = run_dir / f"{corpus}_raw.jsonl"
    with raw_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            primitive = np.asarray(row["primitive"], dtype=np.float64)
            row["control_scores"] = controls(corpus, primitive)
            grouped[row["video_id"]].append(row)
    expected = set()
    labels = hdata.load_labels(corpus)
    _, val_ids = hdata.load_train_val(corpus, labels)
    for video_id in val_ids:
        if (labels[video_id] == 1 and video_id in gt
                and len(np.unique(gt[video_id])) == 2):
            expected.add(video_id)
    if set(grouped) != expected:
        raise RuntimeError(
            f"{corpus}: raw coverage mismatch missing={sorted(expected-set(grouped))} "
            f"extra={sorted(set(grouped)-expected)}"
        )
    keys = tuple(next(iter(grouped.values()))[0]["control_scores"])
    metrics = {}
    for key in keys:
        scores = {
            video_id: densify(len(gt[video_id]), rows, key)
            for video_id, rows in grouped.items()
        }
        report = evaluate_scores(
            scores, {video_id: gt[video_id] for video_id in expected}, expected
        )
        metrics[key] = {
            "within_roc": report["per_video"]["macro_auc"],
            "within_n": report["per_video"]["n_videos_both_classes"],
            "pooled_ap_positive_videos_only": report["pr_auc"],
            "pooled_roc_positive_videos_only": report["roc_auc"],
        }
    compiled = metrics["compiled_policy"]["within_roc"]
    best_control = max(
        metrics[key]["within_roc"] for key in keys if key != "compiled_policy"
    )
    return {
        "videos": len(expected),
        "metrics": metrics,
        "compiled_minus_best_control": compiled - best_control,
        "required_margin": MARGIN,
        "pass": compiled - best_control >= MARGIN,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--corpora", nargs="+", choices=CORPORA,
                        default=list(CORPORA))
    args = parser.parse_args()
    corpora = {
        corpus: analyze_corpus(args.run_dir, corpus) for corpus in args.corpora
    }
    payload = {
        "date": "2026-08-31",
        "stage": "dense_primitive_teacher_attribution_controls",
        "split": "validation_positive_all_eligible",
        "test_used": False,
        "corpora": corpora,
        "pass": all(result["pass"] for result in corpora.values()),
    }
    suffix = "_".join(args.corpora) if tuple(args.corpora) != CORPORA else ""
    out = args.run_dir / (f"controls_{suffix}.json" if suffix else "controls.json")
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
