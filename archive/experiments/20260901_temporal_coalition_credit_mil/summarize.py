#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


SOTA = {
    "hatemm": {"ap": .5938315566, "roc": .8161837922, "within": .6315317180},
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


def diagnostics(path):
    return json.loads(path.read_text())


def delta(left, right):
    return {name: left[name] - right[name] for name in ("ap", "roc", "within")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    output = {"corpora": {}}
    mechanism_rows, performance_rows = [], []
    for corpus in ("hatemm", "hateclipseg"):
        arms = {arm: metrics(root / corpus / arm / "metrics.json")
                for arm in ("anchor", "aligned", "shifted")}
        diag = {arm: diagnostics(
            root / corpus / arm / "mechanism_diagnostics.json")
                for arm in ("aligned", "shifted")}
        aligned = arms["aligned"]
        anchor = arms["anchor"]
        shifted = arms["shifted"]
        mechanism = {
            "within_beats_anchor": aligned["within"] > anchor["within"],
            "pooled_ap_drop_le_010": aligned["ap"] >= anchor["ap"] - .01,
            "pooled_roc_drop_le_010": aligned["roc"] >= anchor["roc"] - .01,
            "within_beats_shifted": aligned["within"] > shifted["within"],
            "aligned_credit_agreement_beats_shifted":
                diag["aligned"]["aligned_credit_agreement"] >
                diag["shifted"]["aligned_credit_agreement"],
        }
        all_sota = all(aligned[name] > SOTA[corpus][name]
                       for name in ("ap", "roc", "within"))
        output["corpora"][corpus] = {
            "arms": arms,
            "aligned_minus_anchor": delta(aligned, anchor),
            "aligned_minus_shifted": delta(aligned, shifted),
            "same_harness_anchor_minus_official_seed234":
                delta(anchor, OFFICIAL_ANCHOR[corpus]),
            "mechanism_diagnostics": diag,
            "mechanism_checks": mechanism,
            "aligned_all_three_sota": all_sota,
        }
        mechanism_rows.extend(mechanism.values())
        performance_rows.append(all_sota)
    output["mechanism_gate"] = {"all_checks_both_corpora": all(mechanism_rows)}
    output["performance_gate"] = {
        "both_corpora_all_three_sota": all(performance_rows)}
    output["decision"] = (
        "EXPAND_TO_MHC_AND_MULTISEED"
        if output["mechanism_gate"]["all_checks_both_corpora"] and
        output["performance_gate"]["both_corpora_all_three_sota"]
        else "STOP_OR_RULE18_POST_TEST_DECISION")
    (root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

