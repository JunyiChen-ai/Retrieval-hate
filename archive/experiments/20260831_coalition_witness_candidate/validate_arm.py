#!/usr/bin/env python3
"""Fail-closed validation for safely resuming one completed pilot arm."""

from __future__ import annotations

import argparse
from pathlib import Path

from summarize import (
    BASELINE_ROOT,
    read_json,
    require_new_arm_integrity,
    require_no_infonce_integrity,
)


def require_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty artifact: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    parser.add_argument("--arm", required=True, choices=(
        "no_infonce", "all_subset_mil", "synib", "mobius_nonminimal",
        "coalition_witness",
    ))
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    baseline_path = BASELINE_ROOT / args.corpus / "seed_234/frame_eval.json"
    baseline = read_json(baseline_path)
    baseline_result = baseline["results"]["score_fused"]
    expected_videos = int(baseline_result["n_videos"])
    expected_frames = int(baseline_result["n_frames"])

    required = [args.out_dir / "metrics.json", args.out_dir / "run.log", args.out_dir / "run.pid"]
    if args.arm == "no_infonce":
        producer = args.out_dir / "producer" / args.corpus
        required.extend([
            producer / "model.pt", producer / "scores.jsonl", producer / "train_log.json",
        ])
    else:
        required.extend([
            args.out_dir / "model.pt", args.out_dir / "scores.jsonl",
            args.out_dir / "config.json", args.out_dir / "code_version.txt",
            args.out_dir / "train_record.json",
        ])
    for path in required:
        require_file(path)

    metrics = read_json(args.out_dir / "metrics.json")
    if metrics.get("corpus") != args.corpus or metrics.get("split") != "test":
        raise RuntimeError("wrong evaluator corpus/split")
    branch = "score_fused" if args.arm == "no_infonce" else "score_full"
    if branch not in metrics.get("results", {}):
        raise RuntimeError("required evaluator branch absent")
    result = metrics["results"][branch]
    if result.get("n_videos_missing_from_scores") or result.get("n_videos_not_in_gold"):
        raise RuntimeError("test coverage incomplete")
    if int(result.get("n_videos", -1)) != expected_videos:
        raise RuntimeError("test video count mismatch")
    if int(result.get("n_frames", -1)) != expected_frames:
        raise RuntimeError("test frame count mismatch")

    if args.arm == "no_infonce":
        require_no_infonce_integrity(args.corpus, args.out_dir, expected_videos)
    else:
        require_new_arm_integrity(args.corpus, args.arm, args.out_dir, expected_videos)
    print(f"valid completed arm: {args.corpus}/{args.arm}")


if __name__ == "__main__":
    main()

