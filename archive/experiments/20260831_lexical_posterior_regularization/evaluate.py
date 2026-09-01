#!/usr/bin/env python
"""Shared-evaluator test scoring for one frozen Stage A arm."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/reproduction_baselines"))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    scores = {}
    with (run_dir / "scores.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            vid = str(row["video_id"])
            if vid in scores:
                raise ValueError(f"duplicate prediction: {vid}")
            scores[vid] = np.asarray(row["score_fused"], dtype=np.float64)
    gt = hdata.gt_arrays(args.corpus, "test")
    labels = hdata.load_labels(args.corpus)
    hate_ids = {vid for vid in gt if labels.get(vid) == 1}
    result = evaluate_scores(scores, gt, hate_ids)
    if result["n_videos_missing_from_scores"] or result["n_videos_not_in_gold"]:
        raise RuntimeError("prediction does not exactly cover evaluator cohort")
    payload = {"developmental_test_evidence": True, "corpus": args.corpus,
               "arm": args.arm, "result": result}
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(args.corpus, args.arm, "AP", result["pr_auc"], "ROC",
          result["roc_auc"], "within", result["per_video"]["macro_auc"],
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
