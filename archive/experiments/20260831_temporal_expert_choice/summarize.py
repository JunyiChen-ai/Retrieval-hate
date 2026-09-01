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
ANCHOR_ROOT = Path(
    "/home/jehc223/Retrieval-hate/runs/20260831_witness_conditional_dgm/"
    "pilot_seed234_matched")


def read_metric(path, branch):
    row = json.loads(path.read_text())["results"][branch]
    return {"ap": row["pr_auc"], "roc": row["roc_auc"],
            "within": row["per_video"]["macro_auc"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", required=True)
    args = parser.parse_args()
    root = Path(args.pilot_dir).resolve()
    output = {"corpora": {}}
    for corpus in SOTA:
        anchor = read_metric(ANCHOR_ROOT / corpus / "anchor" / "metrics.json",
                             "score_fused")
        token = read_metric(root / corpus / "token_choice" / "metrics.json",
                            "score_final")
        core = read_metric(root / corpus / "expert_choice" / "metrics.json",
                           "score_final")
        output["corpora"][corpus] = {
            "anchor": anchor, "token_choice": token, "expert_choice": core,
            "within_gain_vs_anchor": core["within"] - anchor["within"],
            "within_gain_vs_token_choice": core["within"] - token["within"],
            "core_all_sota": all(core[key] > SOTA[corpus][key]
                                  for key in SOTA[corpus]),
        }
    rows = output["corpora"].values()
    output["mechanism_gate"] = {
        "core_beats_anchor_and_control_both": all(
            row["within_gain_vs_anchor"] > 0 and
            row["within_gain_vs_token_choice"] > 0 for row in rows),
        "one_core_vs_anchor_gain_ge_020": max(
            row["within_gain_vs_anchor"] for row in rows) >= .020,
    }
    output["performance_gate"] = {
        "both_corpora_all_sota": all(row["core_all_sota"] for row in rows)}
    output["decision"] = (
        "EXPAND" if all(output["mechanism_gate"].values()) and
        output["performance_gate"]["both_corpora_all_sota"]
        else "STOP_OR_ITERATE_FROM_TEST_ERROR")
    (root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

