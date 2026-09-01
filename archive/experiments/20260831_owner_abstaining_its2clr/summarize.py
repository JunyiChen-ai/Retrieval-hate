"""Summarize formal test metrics and apply the frozen gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GATES = {
    "hatemm": {"ap": 0.5938315566, "roc": 0.8161837922,
               "within": 0.6315317180},
    "hateclipseg": {"ap": 0.6193710950, "roc": 0.6050224699,
                    "within": 0.5619078936},
}
ARMS = (
    "anchor", "broadcast", "core", "branch_selector",
    "shuffled_carrier", "abstain_negative", "nonpositive_background",
    "projection_only",
)


def compact(path):
    payload = json.loads(path.read_text())
    result = payload["results"]["score_core"]
    return {"ap": result["pr_auc"], "roc": result["roc_auc"],
            "within": result["per_video"]["macro_auc"],
            "n_videos": result["n_videos"], "n_frames": result["n_frames"]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.root)
    metrics = {
        corpus: {arm: compact(root / corpus / arm / "metrics.json")
                 for arm in ARMS}
        for corpus in GATES
    }
    performance = {
        corpus: all(metrics[corpus]["core"][key] > gate
                    for key, gate in GATES[corpus].items())
        for corpus in GATES
    }
    within_gain = {
        corpus: metrics[corpus]["core"]["within"]
                - metrics[corpus]["broadcast"]["within"]
        for corpus in GATES
    }
    attribution = {
        "core_beats_broadcast_both": all(value > 0 for value in within_gain.values()),
        "at_least_one_gain_ge_020": any(value >= 0.020 for value in within_gain.values()),
        "core_beats_selector_both": all(
            metrics[c]["core"]["within"] > metrics[c]["branch_selector"]["within"]
            for c in GATES),
        "core_beats_shuffled_both": all(
            metrics[c]["core"]["within"] > metrics[c]["shuffled_carrier"]["within"]
            for c in GATES),
        "core_beats_forced_negative_both": all(
            metrics[c]["core"]["within"] > metrics[c]["abstain_negative"]["within"]
            for c in GATES),
        "core_beats_nonpositive_background_both": all(
            metrics[c]["core"]["within"] > metrics[c]["nonpositive_background"]["within"]
            for c in GATES),
        "core_beats_projection_only_both": all(
            metrics[c]["core"]["within"] > metrics[c]["projection_only"]["within"]
            for c in GATES),
    }
    verdict = {
        "split": "test", "evidence_status": "iterative/developmental",
        "metrics": metrics, "sota_gates": GATES,
        "performance_gate_by_corpus": performance,
        "within_gain_core_vs_broadcast": within_gain,
        "attribution_gates": attribution,
        "all_metrics_sota_both": all(performance.values()),
        "mechanism_gate": all(attribution.values()),
    }
    verdict["advance"] = verdict["all_metrics_sota_both"] and verdict["mechanism_gate"]
    (root / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
