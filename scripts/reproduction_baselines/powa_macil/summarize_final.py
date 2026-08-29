#!/usr/bin/env python3
"""Recompute the final POWA table from frozen dense-score artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402


SEEDS = (234, 2025, 3407)
PATTERNS = {
    "hatemm": "final_maskfix_finetune_hatemm_seed{seed}_e5",
    "mhclip_en": "final_maskfix_finetune_mhclip_en_seed{seed}_e5",
    "mhclip_zh": "final_maskfix_frozen_positive_mhclip_zh_seed{seed}_e5",
    "hateclipseg": "final_maskfix_joint_w48_seed{seed}_e5",
}


def load_branch(path, branch="score_powa"):
    scores = {}
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            scores[row["video_id"]] = np.asarray(row[branch], dtype=float)
    return scores


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gt_sha256(gt):
    digest = hashlib.sha256()
    for video_id in sorted(gt):
        digest.update(video_id.encode("utf-8") + b"\0")
        value = np.asarray(gt[video_id], dtype=np.uint8)
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/reproduction/powa_macil")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    summary = {
        "test_labels_used_in_gradient_training": False,
        "test_evaluated_during_development": True,
        "confirmatory_held_out_test": False,
        "seeds": list(SEEDS), "corpora": {}}
    for corpus, pattern in PATTERNS.items():
        gt = hdata.gt_arrays(corpus, "test")
        labels = hdata.load_labels(corpus)
        hate_ids = {v for v in gt if labels.get(v) == 1}
        seed_scores, seed_metrics = [], []
        for seed in SEEDS:
            path = os.path.join(args.root, pattern.format(seed=seed), corpus,
                                "scores.jsonl")
            scores = load_branch(path)
            if set(scores) != set(gt):
                missing = sorted(set(gt) - set(scores))
                extra = sorted(set(scores) - set(gt))
                raise RuntimeError("coverage mismatch %s seed %d: missing=%s extra=%s"
                                   % (corpus, seed, missing[:5], extra[:5]))
            for video_id in gt:
                if len(scores[video_id]) != len(gt[video_id]):
                    raise RuntimeError("length mismatch %s/%s seed %d" %
                                       (corpus, video_id, seed))
            metrics = evaluate_scores(scores, gt, hate_ids)
            seed_scores.append(scores)
            seed_metrics.append({"seed": seed, "frame_ap": metrics["pr_auc"],
                                 "frame_roc": metrics["roc_auc"],
                                 "scores_sha256": file_sha256(path)})
        aps = np.asarray([x["frame_ap"] for x in seed_metrics])
        rocs = np.asarray([x["frame_roc"] for x in seed_metrics])
        ids = sorted(set.intersection(*(set(x) for x in seed_scores)))
        ensemble = {v: np.mean([x[v] for x in seed_scores], axis=0)
                    for v in ids}
        ensemble_metrics = evaluate_scores(ensemble, gt, hate_ids)
        summary["corpora"][corpus] = {
            "variant_pattern": pattern,
            "n_test_videos": len(gt),
            "gt_sha256": gt_sha256(gt),
            "per_seed": seed_metrics,
            "mean": {"frame_ap": float(aps.mean()),
                     "frame_ap_sample_sd": float(aps.std(ddof=1)),
                     "frame_roc": float(rocs.mean()),
                     "frame_roc_sample_sd": float(rocs.std(ddof=1))},
            "seed_ensemble": {"aggregation": "arithmetic mean of three "
                                             "frozen per-frame scores",
                              "frame_ap": ensemble_metrics["pr_auc"],
                              "frame_roc": ensemble_metrics["roc_auc"]},
        }
    encoded = json.dumps(summary, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
