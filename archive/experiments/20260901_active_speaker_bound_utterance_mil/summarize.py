#!/usr/bin/env python
"""Collect formal HMM/HCS test metrics and fixed gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SOTA = {
    "hatemm": {"pr_auc": .5938315566, "roc_auc": .8161837922,
               "within": .6315317180},
    "hateclipseg": {"pr_auc": .6193710950, "roc_auc": .6050224699,
                    "within": .5619078936},
}


def compact(path):
    payload = json.loads(path.read_text())
    try:
        row = payload["results"]["score_method"]
    except KeyError as error:
        raise KeyError(f"missing shared-evaluator score_method branch in {path}") from error
    return {"pooled_ap": float(row["pr_auc"]),
            "pooled_roc": float(row["roc_auc"]),
            "within_roc": float(row["per_video"]["macro_auc"]),
            "source": str(path.resolve())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.run_dir)
    result = {"corpora": {}}
    all_sota = True
    mechanism = True
    within_gains = []
    for corpus in ("hatemm", "hateclipseg"):
        arms = {arm: compact(root / "test" / corpus / arm / "metrics.json")
                for arm in ("anchor", "permuted", "core")}
        core, permuted = arms["core"], arms["permuted"]
        delta = {key: core[key] - permuted[key]
                 for key in ("pooled_ap", "pooled_roc", "within_roc")}
        thresholds = SOTA[corpus]
        sota_pass = bool(
            core["pooled_ap"] > thresholds["pr_auc"] and
            core["pooled_roc"] > thresholds["roc_auc"] and
            core["within_roc"] > thresholds["within"])
        local_mechanism = bool(
            delta["within_roc"] > 0 and
            (delta["pooled_ap"] > 0 or delta["pooled_roc"] > 0))
        result["corpora"][corpus] = {
            "arms": arms, "core_minus_permuted": delta,
            "sota_pass": sota_pass, "mechanism_partial": local_mechanism}
        all_sota = all_sota and sota_pass
        mechanism = mechanism and local_mechanism
        within_gains.append(delta["within_roc"])
    mechanism = mechanism and max(within_gains) >= .020
    result["mechanism_gate"] = mechanism
    result["all_six_sota_gate"] = all_sota
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
