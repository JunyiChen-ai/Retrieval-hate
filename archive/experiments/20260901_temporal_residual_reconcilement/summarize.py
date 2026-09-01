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
        arms = {arm: read_metrics(root / corpus / arm / "metrics.json")
                for arm in ("cyclic_control", "temporal_residual")}
        core = arms["temporal_residual"]
        gains = {"vs_cyclic_control": core["within"] - arms["cyclic_control"]["within"],
                 "vs_multihateloc_anchor": core["within"] - ANCHOR[corpus]}
        out["corpora"][corpus] = {
            "arms": arms, "core_within_gains": gains,
            "core_all_sota": all(core[key] > sota[key] for key in sota)}
    gains = [row["core_within_gains"] for row in out["corpora"].values()]
    out["mechanism_gate"] = {
        "core_beats_control_and_anchor_on_both": all(
            row["vs_cyclic_control"] > 0 and row["vs_multihateloc_anchor"] > 0
            for row in gains),
        "at_least_one_anchor_gain_ge_020": max(
            row["vs_multihateloc_anchor"] for row in gains) >= .020,
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
