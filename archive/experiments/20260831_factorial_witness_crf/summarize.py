"""Create the frozen two-corpus pilot verdict from evaluator outputs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "runs/20260831_factorial_witness_crf/pilot_seed234"
GATES = {
    "hatemm": {"ap": 0.5938315566328208, "roc": 0.8161837922270064, "within": 0.631531717970362},
    "hateclipseg": {"ap": 0.6193710949898349, "roc": 0.6050224699167533, "within": 0.5619078936355938},
}
ARMS = ("core", "zero_transition", "collapsed")


def compact(path):
    result = json.loads(path.read_text())["results"]["score_core"]
    return {
        "pooled_ap": result["pr_auc"],
        "pooled_roc": result["roc_auc"],
        "within_roc": result["per_video"]["macro_auc"],
        "n_videos": result["n_videos"],
        "n_frames": result["n_frames"],
        "missing": result["n_videos_missing_from_scores"],
        "extra": result["n_videos_not_in_gold"],
        "source": str(path),
    }


def main():
    metrics = {
        corpus: {
            arm: compact(ROOT / corpus / arm / "metrics.json") for arm in ARMS
        }
        for corpus in GATES
    }
    sota_pass = {
        corpus: {
            name: metrics[corpus]["core"][metric] > threshold
            for name, metric, threshold in (
                ("pooled_ap", "pooled_ap", gate["ap"]),
                ("pooled_roc", "pooled_roc", gate["roc"]),
                ("within_roc", "within_roc", gate["within"]),
            )
        }
        for corpus, gate in GATES.items()
    }
    mechanism_pass = {
        corpus: all(
            metrics[corpus]["core"]["within_roc"]
            > metrics[corpus][control]["within_roc"]
            for control in ("zero_transition", "collapsed")
        )
        for corpus in GATES
    }
    payload = {
        "method": "factorial_witness_crf",
        "evidence_status": "iterative/developmental test evidence",
        "metrics": metrics,
        "sota_gates": GATES,
        "sota_pass": sota_pass,
        "mechanism_pass": mechanism_pass,
        "all_sota_pass": all(all(values.values()) for values in sota_pass.values()),
        "mechanism_pass_both": all(mechanism_pass.values()),
        "verdict": "FAIL_AND_STOP",
        "expand_to_mhc": False,
        "test_error_analysis": str(ROOT / "test_error_analysis.json"),
    }
    target = ROOT / "verdict.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
