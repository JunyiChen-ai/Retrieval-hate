#!/usr/bin/env python
"""Apply the frozen Stage A performance and mechanism gates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SOTA = {
    "hatemm": {"pr_auc": 0.5938315566, "roc_auc": 0.8161837922,
               "within": 0.6315317180},
    "hateclipseg": {"pr_auc": 0.6193710950, "roc_auc": 0.6050224699,
                    "within": 0.5619078936},
}
EXPECTED = {"hatemm": {"n_videos": 214, "n_frames": 29269},
            "hateclipseg": {"n_videos": 79, "n_frames": 18839}}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    args = ap.parse_args()
    root = Path(args.run_root).resolve()
    rows = {}
    for corpus in SOTA:
        rows[corpus] = {}
        for arm in ("anchor", "core"):
            directory = root / corpus / arm
            metrics = json.loads((directory / "metrics.json").read_text())
            result = metrics["result"]
            log = json.loads((directory / "train_log.json").read_text())
            require(metrics.get("developmental_test_evidence") is True,
                    f"{corpus}/{arm}: missing developmental scope")
            require(metrics.get("corpus") == corpus and metrics.get("arm") == arm,
                    f"{corpus}/{arm}: metrics identity mismatch")
            require(log.get("corpus") == corpus and log.get("arm") == arm,
                    f"{corpus}/{arm}: train identity mismatch")
            fixed = log.get("args", {})
            require(fixed.get("corpus") == corpus and fixed.get("arm") == arm and
                    fixed.get("seed") == 234 and fixed.get("max_epoch") == 100 and
                    fixed.get("batch_size") == 32 and
                    fixed.get("lambda_pr") == 1.0 and
                    fixed.get("device") == "cuda",
                    f"{corpus}/{arm}: frozen training args mismatch")
            require(len(log.get("history", [])) == 100,
                    f"{corpus}/{arm}: incomplete training history")
            require((directory / "scores.jsonl").is_file() and
                    (directory / "model.pt").is_file(),
                    f"{corpus}/{arm}: missing prediction or model")
            require(result.get("n_videos") == EXPECTED[corpus]["n_videos"] and
                    result.get("n_frames") == EXPECTED[corpus]["n_frames"] and
                    result.get("n_videos_missing_from_scores") == 0 and
                    result.get("n_videos_not_in_gold") == 0,
                    f"{corpus}/{arm}: test coverage mismatch")
            for key in ("pr_auc", "roc_auc"):
                require(finite_number(result.get(key)),
                        f"{corpus}/{arm}: non-finite {key}")
            require(finite_number(result.get("per_video", {}).get("macro_auc")),
                    f"{corpus}/{arm}: non-finite within")
            residuals = [row.get("pr_projection_residual_max")
                         for row in log["history"]]
            require(all(finite_number(x) and x <= 2e-4 for x in residuals),
                    f"{corpus}/{arm}: projection residual failure")
            rows[corpus][arm] = {
                "pr_auc": float(result["pr_auc"]),
                "roc_auc": float(result["roc_auc"]),
                "within": float(result["per_video"]["macro_auc"]),
                "selected_epoch": int(log["selected_epoch"]),
                "initial_train_constraints": log["initial_train_constraints"],
                "selected_train_constraints": log["selected_train_constraints"],
                "metrics_path": str(directory / "metrics.json"),
            }
    corpus_gates = {}
    deltas = {}
    for corpus, threshold in SOTA.items():
        core, anchor = rows[corpus]["core"], rows[corpus]["anchor"]
        deltas[corpus] = {key: core[key] - anchor[key]
                          for key in ("pr_auc", "roc_auc", "within")}
        corpus_gates[corpus] = {
            "all_three_sota": bool(core["pr_auc"] >= threshold["pr_auc"] and
                                   core["roc_auc"] >= threshold["roc_auc"] and
                                   core["within"] >= threshold["within"]),
            "within_above_anchor": bool(core["within"] > anchor["within"]),
        }
    one_delta_020 = max(x["within"] for x in deltas.values()) >= 0.020
    mechanism = {}
    for corpus in SOTA:
        core = rows[corpus]["core"]
        before = core["initial_train_constraints"]
        after = core["selected_train_constraints"]
        diagnostic_keys = ("positive_gap_mean", "positive_violation_mean",
                           "negative_high_mean", "negative_violation_mean")
        support_ok = bool(before.get("n_positive_support", 0) > 0 and
                          before.get("n_negative_support", 0) > 0 and
                          after.get("n_positive_support") == before.get("n_positive_support") and
                          after.get("n_negative_support") == before.get("n_negative_support"))
        finite_ok = all(finite_number(before.get(k)) and finite_number(after.get(k))
                        for k in diagnostic_keys)
        positive_decrease = bool(
            finite_ok and before["positive_violation_mean"] > 0 and
            after["positive_violation_mean"] <=
            0.90 * before["positive_violation_mean"])
        negative_decrease = bool(
            finite_ok and before["negative_violation_mean"] > 0 and
            after["negative_violation_mean"] <=
            0.90 * before["negative_violation_mean"])
        mechanism[corpus] = {
            "support_nonzero_and_stable": support_ok,
            "diagnostics_finite": finite_ok,
            "positive_violation_relative_decrease_at_least_010": positive_decrease,
            "negative_violation_relative_decrease_at_least_010": negative_decrease,
        }
        mechanism[corpus]["pass"] = bool(
            support_ok and finite_ok and positive_decrease and negative_decrease)
    performance_pass = bool(
        all(g["all_three_sota"] and g["within_above_anchor"]
            for g in corpus_gates.values()) and one_delta_020)
    mechanism_pass = bool(all(x["pass"] for x in mechanism.values()))
    payload = {"developmental_test_evidence": True, "rows": rows,
               "sota_thresholds": SOTA, "deltas_core_minus_anchor": deltas,
               "corpus_gates": corpus_gates,
               "one_within_delta_at_least_020": one_delta_020,
               "mechanism_gates": mechanism,
               "performance_pass": performance_pass,
               "mechanism_pass": mechanism_pass,
               "stage_a_pass": bool(performance_pass and mechanism_pass)}
    payload["decision"] = (
        "RUN_STAGE_B_CONTROLS" if payload["stage_a_pass"]
        else "STOP_AND_ARCHIVE")
    (root / "stage_a_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"deltas": deltas, "corpus_gates": corpus_gates,
                      "mechanism_gates": mechanism,
                      "decision": payload["decision"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
