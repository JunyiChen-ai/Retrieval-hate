"""Calls-vs-performance curves from saved complete-combination trajectories (CPU, no retraining).

For one checkpoint the 32-call trajectory is fixed, so every budget is a prefix of the same run:
fixed-k uses min(k, nodes) calls per video; qmixSG_rt10 is recalibrated on validation (no labels)
to several average-call targets. Comparing the two at the same test mean calls asks whether the
adaptive allocation reaches the same AP/ROC/within with fewer calls. Scores go through the unified
evaluator. `--results-dir` runs on the host holding the cohort; `--merge` aggregates returned files.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import pickle
import socket
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import policy
from qtl.stop_rules import rule_values, add_quantile_rules, add_relaxed_rules
import hier_evidence_common as hc

RULE = "qmixSG_rt10"
FIXED_K = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32)
TARGETS = (1, 2, 4, 6, 8, 12, 16)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def curve_for_trial(corpus, trial_dir, out_dir, labels, ids, gt):
    summary = json.loads((trial_dir / "summary.json").read_text())
    calibration = json.loads((trial_dir / "calibration.json").read_text())
    runs = pickle.loads((trial_dir / "runs.pkl").read_bytes())["runs"]
    assert set(runs["val"]) == set(ids["val"]) and set(runs["test"]) == set(ids["test"])
    vals = {split: {v: rule_values(r, len(r["scores"][0])) for v, r in runs[split].items()} for split in runs}
    add_quantile_rules(vals)
    add_relaxed_rules(vals)
    val_rule = [vals["val"][v][RULE] for v in ids["val"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    settings = [("fixed", k, None) for k in FIXED_K]
    for t in TARGETS:
        threshold, val_mean = policy.calibrate(val_rule, float(t))
        if t == 8:      # must reproduce the evaluated complete combination exactly
            assert threshold == calibration["threshold"]
        settings.append(("auto", t, (threshold, val_mean)))
    for kind, budget, cal in settings:
        calls = {v: (min(budget, len(r["eig"])) if kind == "fixed"
                     else policy.stop_calls(vals["test"][v][RULE], cal[0])) for v, r in runs["test"].items()}
        name = f"{kind}{budget}"
        sp, mp = out_dir / f"scores_test_{name}.jsonl", out_dir / f"metrics_test_{name}.json"
        hc.write_scores(str(sp), {v: runs["test"][v]["scores"][calls[v]] for v in ids["test"]})
        r = hc.run_evaluator(corpus, "test", str(sp), str(mp))["results"]["score_av"]
        assert r["n_videos_missing_from_scores"] == 0 and r["n_videos_not_in_gold"] == 0
        assert r["n_videos"] == len(ids["test"]) and r["n_frames"] == sum(len(gt["test"][v]) for v in ids["test"])
        values = [r["pr_auc"], r["roc_auc"], r["per_video"]["macro_auc"]]
        assert all(math.isfinite(x) for x in values)
        counts = np.asarray([calls[v] for v in ids["test"]])
        row = {"kind": kind, "budget": budget, "metric_source": str(mp), "ap_roc_within": values,
               "mean_calls": float(counts.mean()),
               "mean_calls_positive": float(np.mean([calls[v] for v in calls if labels[v] == 1])),
               "mean_calls_negative": float(np.mean([calls[v] for v in calls if labels[v] == 0]))}
        if kind == "auto":
            # inf is a valid threshold (only the undefined-statistic first call is asked); JSON needs a string.
            row.update(threshold=cal[0] if math.isfinite(cal[0]) else str(cal[0]), validation_mean_calls=cal[1])
        if (kind, budget) in (("fixed", 0), ("fixed", 8), ("fixed", 32), ("auto", 8)):
            name0 = RULE if kind == "auto" else f"fixed{budget}"
            assert values == summary["results"][name0]["ap_roc_within"]
        rows.append(row)
    report = {"seed": summary["seed"], "trial": summary["trial"], "selections": summary["selections"],
              "source": str(trial_dir), "rows": rows}
    write_json(out_dir / "curve.json", report)
    return report


def interp(x, xs, ys):
    """Piecewise-linear value of the fixed-budget curve at mean calls x (duplicate x keep the first point)."""
    xs, idx = np.unique(np.asarray(xs), return_index=True)
    return float(np.interp(x, xs, np.asarray(ys)[idx]))


def merge(parts, corpus):
    trials = [t for p in parts for t in p["trials"]]
    out = {"corpus": corpus, "fixed_k": FIXED_K, "targets": TARGETS, "selections": {}}
    for selection in ("test_selected", "validation_selected"):
        chosen = sorted((t for t in trials if selection in t["selections"]), key=lambda t: t["seed"])
        assert [t["seed"] for t in chosen] == [234, 2025, 3407]
        block = {"fixed": [], "auto": []}
        for k in FIXED_K:
            rows = [next(r for r in t["rows"] if r["kind"] == "fixed" and r["budget"] == k) for t in chosen]
            v = np.array([r["ap_roc_within"] for r in rows])
            block["fixed"].append({"budget": k, "mean_calls": float(np.mean([r["mean_calls"] for r in rows])),
                                   "mean": v.mean(0).tolist(), "std_population": v.std(0).tolist()})
        for t_ in TARGETS:
            rows, diffs = [], []
            for t in chosen:
                fixed = [r for r in t["rows"] if r["kind"] == "fixed"]
                a = next(r for r in t["rows"] if r["kind"] == "auto" and r["budget"] == t_)
                matched = [interp(a["mean_calls"], [r["mean_calls"] for r in fixed],
                                  [r["ap_roc_within"][m] for r in fixed]) for m in range(3)]
                rows.append(a)
                diffs.append(np.subtract(a["ap_roc_within"], matched))
            v, d = np.array([r["ap_roc_within"] for r in rows]), np.array(diffs)
            block["auto"].append({"target": t_, "mean_calls": float(np.mean([r["mean_calls"] for r in rows])),
                "mean_calls_positive": float(np.mean([r["mean_calls_positive"] for r in rows])),
                "mean_calls_negative": float(np.mean([r["mean_calls_negative"] for r in rows])),
                "mean": v.mean(0).tolist(), "std_population": v.std(0).tolist(),
                "minus_fixed_at_same_calls": {"paired_mean": d.mean(0).tolist(),
                    "paired_std_population": d.std(0).tolist(), "paired_seed_values": d.tolist()}})
        out["selections"][selection] = block
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--results-dir", nargs="+", help="Evaluated results dirs holding seed<s>_trial<k>/runs.pkl")
    ap.add_argument("--out-root", default="runs/20261002_complete_io_method/budget_curve")
    ap.add_argument("--merge", nargs="+", help="Partial curve files to aggregate")
    a = ap.parse_args()
    out_root = (ROOT / a.out_root).resolve() / a.corpus
    if a.merge:
        report = merge([json.loads(Path(p).read_text()) for p in a.merge], a.corpus)
        report["merged_at"] = datetime.now().astimezone().isoformat()
        write_json(out_root / "curve_summary.json", report)
        print(out_root / "curve_summary.json")
        return
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    trials = []
    for results in a.results_dir:
        for trial_dir in sorted(Path(results).resolve().glob("seed*_trial*")):
            trials.append(curve_for_trial(a.corpus, trial_dir, out_root / trial_dir.name, labels, ids, gt))
            print(trial_dir.name, "done", flush=True)
    part = out_root / f"partial_{socket.gethostname()}.json"
    write_json(part, {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(),
                      "trials": trials})
    print(part)


if __name__ == "__main__":
    main()
