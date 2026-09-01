"""Apply the hard two-corpus core gate after performance-triggered early stop."""

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


def compact(path):
    result = json.loads(path.read_text())["results"]["score_core"]
    return {"ap": result["pr_auc"], "roc": result["roc_auc"],
            "within": result["per_video"]["macro_auc"]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.root)
    metrics = {
        corpus: {arm: compact(root / corpus / arm / "metrics.json")
                 for arm in ("anchor", "broadcast", "core")}
        for corpus in GATES
    }
    performance = {
        corpus: all(metrics[corpus]["core"][key] > value
                    for key, value in GATES[corpus].items())
        for corpus in GATES
    }
    gains = {corpus: metrics[corpus]["core"]["within"]
                      - metrics[corpus]["broadcast"]["within"]
             for corpus in GATES}
    payload = {
        "split": "test", "evidence_status": "iterative/developmental",
        "metrics": metrics, "sota_gates": GATES,
        "performance_gate_by_corpus": performance,
        "within_gain_core_vs_broadcast": gains,
        "core_beats_broadcast_both": all(value > 0 for value in gains.values()),
        "at_least_one_gain_ge_020": any(value >= 0.020 for value in gains.values()),
        "all_metrics_sota_both": all(performance.values()),
        "advance": False,
        "early_stop": {
            "reason": "HateMM core failed all three SOTA gates and the frozen mechanism gain; remaining attribution controls cannot change core eligibility",
            "completed_formal_arms": ["anchor", "broadcast", "core"],
            "uncompleted_arms_not_claimed": ["branch_selector", "shuffled_carrier", "abstain_negative", "nonpositive_background", "projection_only"],
        },
    }
    (root / "verdict.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

