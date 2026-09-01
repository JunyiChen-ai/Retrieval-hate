#!/usr/bin/env python
"""Assemble authoritative HMM/HCS test metrics and gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

THRESHOLDS = {
    "hatemm": {"pooled_ap": .5938315566, "pooled_roc": .8161837922,
               "within_roc": .6315317180},
    "hateclipseg": {"pooled_ap": .6193710950, "pooled_roc": .6050224699,
                    "within_roc": .5619078936}}
OFFICIAL_START = {
    "hatemm": {"pooled_ap": .4930003188, "pooled_roc": .7382584146,
               "within_roc": .6284561854},
    "hateclipseg": {"pooled_ap": .5530209589, "pooled_roc": .5440720382,
                    "within_roc": .5237011979}}


def metrics(path):
    row = json.loads(path.read_text())["results"]["score_method"]
    return {"pooled_ap": float(row["pr_auc"]),
            "pooled_roc": float(row["roc_auc"]),
            "within_roc": float(row["per_video"]["macro_auc"]),
            "within_n": int(row["per_video"]["n_videos_both_classes"])}


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
        arms = {arm: metrics(root / "test" / corpus / arm / "metrics.json")
                for arm in ("anchor", "binary", "permuted", "policy")}
        gates = {metric: arms["policy"][metric] > THRESHOLDS[corpus][metric]
                 for metric in ("pooled_ap", "pooled_roc", "within_roc")}
        corpora[corpus] = {
            "arms": arms,
            "policy_minus_binary": subtract(arms["policy"], arms["binary"]),
            "policy_minus_permuted": subtract(arms["policy"], arms["permuted"]),
            "policy_minus_anchor": subtract(arms["policy"], arms["anchor"]),
            "policy_minus_official_start": subtract(
                arms["policy"], OFFICIAL_START[corpus]),
            "sota_thresholds": THRESHOLDS[corpus],
            "all_sota": all(gates.values()), "metric_gates": gates,
            "selection": json.loads(
                (root / "val_search" / corpus / "selection.json").read_text())}
    mechanism_gate = all(
        corpora[c]["policy_minus_binary"]["within_roc"] > 0 and
        (corpora[c]["policy_minus_anchor"]["pooled_ap"] > 0 or
         corpora[c]["policy_minus_anchor"]["pooled_roc"] > 0)
        for c in corpora)
    payload = {"method": "policy_cluster_transport", "split": "test",
               "corpora": corpora, "mechanism_gate": mechanism_gate,
               "hmm_hcs_all_sota": all(x["all_sota"] for x in corpora.values())}
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
