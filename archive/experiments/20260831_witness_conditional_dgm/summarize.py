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


def metrics(path):
    payload = json.loads(path.read_text())
    row = payload["results"]["score_fused"]
    return {"ap": row["pr_auc"], "roc": row["roc_auc"],
            "within": row["per_video"]["macro_auc"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", required=True)
    args = parser.parse_args()
    root = Path(args.pilot_dir).resolve()
    out = {"corpora": {}}
    for corpus in SOTA:
        arms = {arm: metrics(root / corpus / arm / "metrics.json")
                for arm in ("anchor", "source_dgm", "witness_dgm")}
        core = arms["witness_dgm"]
        gain = {arm: core["within"] - arms[arm]["within"]
                for arm in ("anchor", "source_dgm")}
        out["corpora"][corpus] = {
            "arms": arms, "within_gain_core": gain,
            "core_all_sota": all(core[key] > SOTA[corpus][key]
                                  for key in SOTA[corpus]),
        }
    all_control_gains = [out["corpora"][corpus]["within_gain_core"][arm]
                         for corpus in SOTA
                         for arm in ("anchor", "source_dgm")]
    anchor_gains = [out["corpora"][corpus]["within_gain_core"]["anchor"]
                    for corpus in SOTA]
    out["mechanism_gate"] = {
        "core_beats_both_controls_on_both_corpora":
            all(x > 0 for x in all_control_gains),
        "at_least_one_core_vs_anchor_gain_ge_020":
            max(anchor_gains) >= .020,
    }
    out["performance_gate"] = {
        "both_corpora_all_three_sota": all(
            row["core_all_sota"] for row in out["corpora"].values())}
    out["decision"] = (
        "EXPAND" if all(out["mechanism_gate"].values()) and
        out["performance_gate"]["both_corpora_all_three_sota"]
        else "STOP_OR_ITERATE_FROM_TEST_ERROR")
    (root / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
