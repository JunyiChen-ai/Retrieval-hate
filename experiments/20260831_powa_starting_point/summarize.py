#!/usr/bin/env python3
"""Aggregate shared-evaluator outputs without reimplementing metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CORPORA = ("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg")
SEEDS = (234, 2025, 3407)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    payload = {"protocol": "independent-corpus POWA, test, 1fps", "corpora": {}}
    for corpus in CORPORA:
        rows = []
        for seed in SEEDS:
            path = root / f"{corpus}_seed{seed}" / "metrics.json"
            report = json.loads(path.read_text())
            metric = report["results"]["score_powa"]
            rows.append({
                "seed": seed,
                "pooled_ap": metric["pr_auc"],
                "pooled_roc": metric["roc_auc"],
                "within_roc": metric["per_video"]["macro_auc"],
                "within_n": metric["per_video"]["n_videos_both_classes"],
                "metrics_file": str(path.resolve()),
            })
        aggregate = {}
        for key in ("pooled_ap", "pooled_roc", "within_roc"):
            values = np.asarray([row[key] for row in rows], dtype=float)
            aggregate[key] = {
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
            }
        payload["corpora"][corpus] = {"seeds": rows, "aggregate": aggregate}
    maskfix_rows = []
    for seed in SEEDS:
        path = root / f"hcs_maskfix_seed{seed}" / "metrics.json"
        if not path.exists():
            maskfix_rows = []
            break
        report = json.loads(path.read_text())
        metric = report["results"]["score_powa"]
        maskfix_rows.append({
            "seed": seed,
            "pooled_ap": metric["pr_auc"],
            "pooled_roc": metric["roc_auc"],
            "within_roc": metric["per_video"]["macro_auc"],
            "within_n": metric["per_video"]["n_videos_both_classes"],
            "metrics_file": str(path.resolve()),
        })
    if maskfix_rows:
        aggregate = {}
        for key in ("pooled_ap", "pooled_roc", "within_roc"):
            values = np.asarray([row[key] for row in maskfix_rows], dtype=float)
            aggregate[key] = {
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
            }
        payload["hcs_current_maskfix"] = {
            "seeds": maskfix_rows,
            "aggregate": aggregate,
        }
    target = Path(args.out)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
