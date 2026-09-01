#!/usr/bin/env python
"""Assemble authoritative HMM/HCS test metrics and gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


THRESHOLDS = {
    "hatemm": {"pooled_ap": 0.5938315566,
               "pooled_roc": 0.8161837922,
               "within_roc": 0.6315317180},
    "hateclipseg": {"pooled_ap": 0.6193710950,
                    "pooled_roc": 0.6050224699,
                    "within_roc": 0.5619078936},
}
OFFICIAL_START = {
    "hatemm": {"pooled_ap": 0.4930003188,
               "pooled_roc": 0.7382584146,
               "within_roc": 0.6284561854},
    "hateclipseg": {"pooled_ap": 0.5530209589,
                    "pooled_roc": 0.5440720382,
                    "within_roc": 0.5237011979},
}


def metrics(path):
    payload = json.loads(path.read_text())
    row = payload["results"]["score_method"]
    return {
        "pooled_ap": float(row["pr_auc"]),
        "pooled_roc": float(row["roc_auc"]),
        "within_roc": float(row["per_video"]["macro_auc"]),
        "within_n": int(row["per_video"]["n_videos_both_classes"]),
    }


def subtract(left, right):
    return {key: left[key] - right[key]
            for key in ("pooled_ap", "pooled_roc", "within_roc")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.run_dir)
    corpora = {}
    for corpus in THRESHOLDS:
        arms = {
            arm: metrics(root / "test" / corpus / arm / "metrics.json")
            for arm in ("anchor", "shifted", "aligned")
        }
        gate = {
            metric: arms["aligned"][metric] > THRESHOLDS[corpus][metric]
            for metric in ("pooled_ap", "pooled_roc", "within_roc")
        }
        corpora[corpus] = {
            "arms": arms,
            "aligned_minus_shifted": subtract(arms["aligned"], arms["shifted"]),
            "aligned_minus_anchor": subtract(arms["aligned"], arms["anchor"]),
            "aligned_minus_official_start": subtract(
                arms["aligned"], OFFICIAL_START[corpus]),
            "sota_thresholds": THRESHOLDS[corpus],
            "all_sota": all(gate.values()),
            "metric_gates": gate,
            "selection": json.loads(
                (root / "val_search" / corpus / "selection.json").read_text()),
        }
    mechanism_gate = all(
        corpora[corpus]["aligned_minus_shifted"]["within_roc"] > 0
        and (corpora[corpus]["aligned_minus_shifted"]["pooled_ap"] > 0
             or corpora[corpus]["aligned_minus_shifted"]["pooled_roc"] > 0)
        for corpus in corpora)
    payload = {
        "method": "lexically_anchored_dcc",
        "split": "test",
        "corpora": corpora,
        "mechanism_gate": mechanism_gate,
        "hmm_hcs_all_sota": all(row["all_sota"] for row in corpora.values()),
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
