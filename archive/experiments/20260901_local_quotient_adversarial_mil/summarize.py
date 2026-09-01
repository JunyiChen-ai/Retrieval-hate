#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


SOTA = {
    "hatemm": {"ap": .5938315566, "roc": .8161837922, "within": .6315317180},
    "hateclipseg": {"ap": .6193710950, "roc": .6050224699, "within": .5619078936},
}
ANCHOR = {"hatemm": .6284561854116418,
          "hateclipseg": .5237011979483558}


def read_metrics(path):
    row = json.loads(path.read_text())["results"]["score_fused"]
    return {"ap": row["pr_auc"], "roc": row["roc_auc"],
            "within": row["per_video"]["macro_auc"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    out = {"corpora": {}}
    for corpus, sota in SOTA.items():
        mechanism_path = root / corpus / "mechanism.json"
        if not mechanism_path.is_file():
            raise SystemExit(f"missing mechanism diagnostic: {mechanism_path}")
        mechanism = json.loads(mechanism_path.read_text())
        arms = {arm: read_metrics(root / corpus / arm / "metrics.json")
                for arm in ("local_control", "local_adversarial")}
        core = arms["local_adversarial"]
        gains = {"vs_control": core["within"] - arms["local_control"]["within"],
                 "vs_multihateloc_anchor": core["within"] - ANCHOR[corpus]}
        out["corpora"][corpus] = {
            "arms": arms, "core_within_gains": gains,
            "mechanism": mechanism,
            "core_all_sota": all(core[key] > sota[key] for key in sota)}
    gains = [row["core_within_gains"] for row in out["corpora"].values()]
    out["mechanism_gate"] = {
        "core_beats_control_and_anchor_on_both": all(
            row["vs_control"] > 0 and row["vs_multihateloc_anchor"] > 0
            for row in gains),
        "at_least_one_anchor_gain_ge_020": max(
            row["vs_multihateloc_anchor"] for row in gains) >= .020,
        "both_nuisance_probes_reduced_on_both": all(
            row["mechanism"]["arms"]["local_adversarial"][key] <
            row["mechanism"]["arms"]["local_control"][key]
            for row in out["corpora"].values()
            for key in ("video_id_accuracy", "position_bin_accuracy")),
        "positive_high_position_risk_gain_on_both": all(
            row["mechanism"]["position_risk"]["high"]
               ["mean_core_minus_control_within"] > 0
            for row in out["corpora"].values()),
    }
    out["performance_gate"] = {"both_corpora_all_three_sota": all(
        row["core_all_sota"] for row in out["corpora"].values())}
    out["decision"] = ("EXPAND" if all(out["mechanism_gate"].values()) and
                       out["performance_gate"]["both_corpora_all_three_sota"]
                       else "STOP_OR_ONE_POST_TEST_CORRECTION")
    (root / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
