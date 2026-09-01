#!/usr/bin/env python3
"""Apply the frozen Stage-P gates to evaluator-written metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GATES = {
    "hatemm": {"pooled_ap": 0.5938316, "pooled_roc": 0.8161838,
               "within_roc": 0.6315317},
    "hateclipseg": {"pooled_ap": 0.6193711, "pooled_roc": 0.6050225,
                    "within_roc": 0.5619079},
}


def load(path: Path):
    report = json.loads(path.read_text())["results"]["score_powa"]
    return {
        "pooled_ap": report["pr_auc"],
        "pooled_roc": report["roc_auc"],
        "within_roc": report["per_video"]["macro_auc"],
        "within_n": report["per_video"]["n_videos_both_classes"],
        "metrics_file": str(path.resolve()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    payload = {"frozen_gates": GATES, "corpora": {}}
    passed = True
    for corpus, gates in GATES.items():
        matched = load(root / f"{corpus}_matched_powa_seed234" /
                       "metrics.json")
        full = load(root / f"{corpus}_full_seed234" / "metrics.json")
        metric_pass = {name: full[name] > value
                       for name, value in gates.items()}
        within_vs_matched = full["within_roc"] > matched["within_roc"]
        corpus_pass = all(metric_pass.values()) and within_vs_matched
        passed = passed and corpus_pass
        payload["corpora"][corpus] = {
            "matched_powa": matched,
            "full": full,
            "full_minus_matched": {
                name: full[name] - matched[name]
                for name in ("pooled_ap", "pooled_roc", "within_roc")
            },
            "strict_sota_metric_pass": metric_pass,
            "within_exceeds_matched": within_vs_matched,
            "corpus_pass": corpus_pass,
        }
    payload["stage_p_pass"] = passed
    payload["decision"] = ("run Stage M attribution" if passed else
                           "kill candidate; do not tune or run Stage M")
    target = Path(args.out)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps({"stage_p_pass": passed,
                      "decision": payload["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
