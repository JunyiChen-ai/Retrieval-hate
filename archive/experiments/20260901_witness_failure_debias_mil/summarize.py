#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


SOTA = {
    "hatemm": {"ap": .5938315566, "roc": .8161837922,
               "within": .6315317180},
    "hateclipseg": {"ap": .6193710950, "roc": .6050224699,
                    "within": .5619078936},
}
OFFICIAL_ANCHOR = {
    "hatemm": {"ap": .4930003188145818, "roc": .7382584145500913,
               "within": .6284561854116418},
    "hateclipseg": {"ap": .5530209588788647, "roc": .5440720382028135,
                    "within": .5237011979483558},
}


def metrics(path):
    row = json.loads(path.read_text())["results"]["score_fused"]
    return {"ap": row["pr_auc"], "roc": row["roc_auc"],
            "within": row["per_video"]["macro_auc"]}


def delta(left, right):
    return {name: left[name] - right[name]
            for name in ("ap", "roc", "within")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    output = {"corpora": {}}
    mechanism, performance = [], []
    for corpus in ("hatemm", "hateclipseg"):
        arms = {arm: metrics(root / corpus / arm / "metrics.json")
                for arm in ("anchor", "uniform", "relative")}
        relative, uniform, anchor = (
            arms["relative"], arms["uniform"], arms["anchor"])
        checks = {
            "within_beats_uniform": relative["within"] > uniform["within"],
            "relative_minus_uniform_within_ge_010":
                relative["within"] - uniform["within"] >= .01,
            "pooled_ap_drop_vs_uniform_le_010":
                relative["ap"] >= uniform["ap"] - .01,
            "pooled_roc_drop_vs_uniform_le_010":
                relative["roc"] >= uniform["roc"] - .01,
        }
        all_sota = all(relative[name] > SOTA[corpus][name]
                       for name in ("ap", "roc", "within"))
        output["corpora"][corpus] = {
            "arms": arms,
            "relative_minus_uniform": delta(relative, uniform),
            "relative_minus_anchor": delta(relative, anchor),
            "same_harness_anchor_minus_official_seed234":
                delta(anchor, OFFICIAL_ANCHOR[corpus]),
            "mechanism_checks": checks,
            "relative_all_three_sota": all_sota,
        }
        mechanism.append(checks["within_beats_uniform"])
        performance.append(all_sota)
    at_least_one_ge_010 = any(
        row["relative_minus_uniform"]["within"] >= .01
        for row in output["corpora"].values())
    output["mechanism_gate"] = {
        "within_beats_uniform_both_corpora": all(mechanism),
        "at_least_one_corpus_gain_ge_010": at_least_one_ge_010,
        "passed": all(mechanism) and at_least_one_ge_010,
    }
    output["performance_gate"] = {
        "both_corpora_all_three_sota": all(performance)}
    output["decision"] = (
        "SOTA_AND_NOVEL" if output["mechanism_gate"]["passed"] and
        output["performance_gate"]["both_corpora_all_three_sota"]
        else "STOP_OR_RULE18_POST_TEST_DECISION")
    (root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
