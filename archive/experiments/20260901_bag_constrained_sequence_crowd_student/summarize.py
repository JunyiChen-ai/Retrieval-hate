#!/usr/bin/env python
"""Collect shared-evaluator HMM/HCS test metrics and fixed gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SOTA = {
    "hatemm": {"pooled_ap": .5938315566, "pooled_roc": .8161837922,
               "within_roc": .6315317180},
    "hateclipseg": {"pooled_ap": .6193710950, "pooled_roc": .6050224699,
                    "within_roc": .5619078936},
}
STARTING_POINT = {
    "hatemm": {"pooled_ap": .4930003188145818,
               "pooled_roc": .7382584145500913,
               "within_roc": .6284561854116418},
    "hateclipseg": {"pooled_ap": .5530209588788647,
                    "pooled_roc": .5440720382028135,
                    "within_roc": .5237011979483558},
}
ARMS = ("core", "token_ds", "unconstrained_bsc")


def compact(path):
    payload = json.loads(path.read_text())
    try:
        row = payload["results"]["score_method"]
    except KeyError as error:
        raise RuntimeError(f"missing shared-evaluator score_method: {path}") from error
    if row["n_videos_missing_from_scores"] or row["n_videos_not_in_gold"]:
        raise RuntimeError(f"incomplete test cohort: {path}")
    return {
        "pooled_ap": float(row["pr_auc"]),
        "pooled_roc": float(row["roc_auc"]),
        "within_roc": float(row["per_video"]["macro_auc"]),
        "source": str(path.resolve()),
    }


def delta(left, right):
    return {key: left[key] - right[key]
            for key in ("pooled_ap", "pooled_roc", "within_roc")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.run_dir)
    result = {"corpora": {}, "fixed_sota": SOTA,
              "official_starting_point": STARTING_POINT}
    all_sota = True
    mechanism = True
    gain_budget = True
    for corpus in ("hatemm", "hateclipseg"):
        arms = {arm: compact(root / "test" / corpus / arm / "metrics.json")
                for arm in ARMS}
        versus_token = delta(arms["core"], arms["token_ds"])
        versus_unconstrained = delta(arms["core"], arms["unconstrained_bsc"])
        sota_margin = delta(arms["core"], SOTA[corpus])
        starting_gain = delta(arms["core"], STARTING_POINT[corpus])
        major_keys = [key for key in SOTA[corpus]
                      if SOTA[corpus][key] - STARTING_POINT[corpus][key] >= .02]
        corpus_gain_budget = max(starting_gain[key] for key in major_keys) >= .02
        corpus_sota = all(value > 0 for value in sota_margin.values())
        corpus_mechanism = (
            all(value >= 0 for value in versus_token.values())
            and all(value >= 0 for value in versus_unconstrained.values())
            and versus_token["within_roc"] > 0
            and versus_unconstrained["within_roc"] > 0
        )
        result["corpora"][corpus] = {
            "arms": arms,
            "core_minus_token_ds": versus_token,
            "core_minus_unconstrained_bsc": versus_unconstrained,
            "core_minus_official_starting_point": starting_gain,
            "core_minus_fixed_sota": sota_margin,
            "matched_control_mechanism_pass": bool(corpus_mechanism),
            "major_gap_metrics": major_keys,
            "gain_budget_pass": bool(corpus_gain_budget),
            "all_three_sota_pass": bool(corpus_sota),
        }
        mechanism = mechanism and corpus_mechanism
        gain_budget = gain_budget and corpus_gain_budget
        all_sota = all_sota and corpus_sota
    result["matched_control_mechanism_gate"] = bool(mechanism)
    result["gain_budget_gate"] = bool(gain_budget)
    result["mechanism_gate"] = bool(mechanism and gain_budget)
    result["all_six_sota_gate"] = bool(all_sota)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
