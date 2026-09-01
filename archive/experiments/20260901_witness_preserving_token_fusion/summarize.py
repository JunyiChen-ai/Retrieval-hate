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
    root = Path(parser.parse_args().run_root).resolve()
    output = {"corpora": {}}
    directions, all_sota = [], []
    for corpus in ("hatemm", "hateclipseg"):
        arms = {arm: metrics(root / corpus / arm / "metrics.json")
                for arm in ("anchor", "aligned", "shifted")}
        diagnostics = {arm: json.loads(
            (root / corpus / arm / "mechanism_diagnostics.json").read_text())
            for arm in ("aligned", "shifted")}
        aligned, shifted, anchor = (
            arms["aligned"], arms["shifted"], arms["anchor"])
        corpus_sota = all(aligned[name] > SOTA[corpus][name]
                          for name in ("ap", "roc", "within"))
        output["corpora"][corpus] = {
            "arms": arms,
            "aligned_minus_shifted": delta(aligned, shifted),
            "aligned_minus_anchor": delta(aligned, anchor),
            "same_harness_anchor_minus_official_seed234":
                delta(anchor, OFFICIAL_ANCHOR[corpus]),
            "mechanism_diagnostics": diagnostics,
            "aligned_all_three_sota": corpus_sota,
        }
        directions.append(aligned["within"] > shifted["within"])
        all_sota.append(corpus_sota)
    one_ge_010 = any(
        row["aligned_minus_shifted"]["within"] >= .01
        for row in output["corpora"].values())
    output["mechanism_gate"] = {
        "aligned_within_beats_shifted_both_corpora": all(directions),
        "at_least_one_corpus_gain_ge_010": one_ge_010,
        "passed": all(directions) and one_ge_010,
    }
    output["performance_gate"] = {
        "both_corpora_all_three_sota": all(all_sota)}
    output["decision"] = (
        "SOTA_AND_NOVEL" if output["mechanism_gate"]["passed"] and
        output["performance_gate"]["both_corpora_all_three_sota"]
        else "TRIGGER_PROCESS_REVIEW")
    (root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
