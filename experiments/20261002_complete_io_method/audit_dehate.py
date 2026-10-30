"""Audit DeHate full searches and complete-combination evaluations (read-only on runs/).

Re-derives trial selection, stopping threshold, per-video calls and written score prefixes,
then aggregates the evaluator outputs; never reruns training. The DeHate cohort exists only on
the remote hosts, so `--seeds` runs there and `--merge` combines the returned partial files.
"""
import argparse
from datetime import datetime
import json
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
SETTINGS = ("fixed0", "fixed8", "qmixSG_rt10", "fixed32")
HOSTS = {234: "sc474398", 2025: "sc474397", 3407: "sc474398"}
R5 = {234: 8, 2025: 4, 3407: 4}
TRAIN_METRICS = 13


def load(path):
    return json.loads(Path(path).read_text())


def audit_study(run, seed, labels, ids, gt):
    completion = load(run / "completion.json")
    assert completion["status"] == "success" and completion["host"] == HOSTS[seed]
    assert (run / "run.log").read_text().splitlines()[0].startswith(f"host {HOSTS[seed]}")
    budget = load(run / "budget.json")["n_trials"]
    study = load(run / "study_summary.json")
    assert budget == study["n_trials"] == len(study["trials"]) == 20
    assert all(t["state"] == "COMPLETE" for t in study["trials"])
    for t in study["trials"]:
        trial = run / f'trial{t["number"]}'
        s = load(trial / "summary.json")
        assert s["cfg"]["max_epoch"] == 50 and [r["epoch"] for r in s["history"]] == list(range(1, 51))
        assert s["best_epoch"] == t["user_attrs"]["best_epoch"]
        assert len(list(trial.glob("metrics_test_*.json"))) == TRAIN_METRICS and (trial / "model.pth").exists()
    # Selection keys used by the search: test (AP+ROC)/2 and validation (AP+ROC)/2, both no-copy fixed8.
    test_key = lambda t: (t["user_attrs"]["test_pooled_ap"] + t["user_attrs"]["test_pooled_roc"]) / 2
    val_key = lambda t: (t["user_attrs"]["val_pooled_ap"] + t["user_attrs"]["val_pooled_roc"]) / 2
    test_best = max(study["trials"], key=test_key)["number"]
    val_best = max(study["trials"], key=val_key)["number"]
    assert test_best == study["best"]["number"] and val_best == study["validation_selected"]["number"]
    selection = load(run / "results/selection.json")
    expected = {str(run.resolve() / f"trial{test_best}"): "test_selected",
                str(run.resolve() / f"trial{val_best}"): "validation_selected"}
    assert {k: v["selections"] for k, v in selection.items()} == {k: [v] for k, v in expected.items()}
    summary = load(run / "results/summary.json")
    rows = []
    for trial, row in summary["trials"].items():
        dest = Path(row["output"])
        source = load(Path(trial) / "summary.json")
        assert row["checkpoint_epoch"] == source["best_epoch"]
        cfg = load(dest / "config.json")
        assert cfg["stop_rule"] == RULE and cfg["copy_pi"] == "neg" and cfg["copy_mode"] == "both"
        calibration = load(dest / "calibration.json")
        assert calibration["reference_video_ids"] == ids["val"]
        copy_ids = calibration["copy"]["source_ids"]
        assert set(copy_ids) <= set(ids["train"]) and all(labels[v] == 0 for v in copy_ids)
        runs = pickle.loads((dest / "runs.pkl").read_bytes())["runs"]
        assert set(runs["val"]) == set(ids["val"]) and set(runs["test"]) == set(ids["test"])
        vals = {split: {v: rule_values(r, len(r["scores"][0])) for v, r in runs[split].items()} for split in runs}
        add_quantile_rules(vals)
        add_relaxed_rules(vals)
        threshold, val_mean = policy.calibrate([vals["val"][v][RULE] for v in ids["val"]], 8.0)
        assert threshold == calibration["threshold"] and val_mean == calibration["validation_mean_calls"]
        out = {"seed": seed, "trial": row["trial"], "selections": row["selections"],
               "checkpoint_epoch": row["checkpoint_epoch"], "source_trial": trial,
               "calibration_source": str(dest / "calibration.json"), "threshold": threshold,
               "validation_mean_calls": val_mean, "results": {}}
        for name in SETTINGS:
            calls = {v: (min(int(name[5:]), len(r["eig"])) if name.startswith("fixed")
                         else policy.stop_calls(vals["test"][v][RULE], threshold)) for v, r in runs["test"].items()}
            assert calls == row["results"][name]["calls_per_video"]
            written = {}
            for line in (dest / f"scores_test_{name}.jsonl").read_text().splitlines():
                rec = json.loads(line)
                written[rec["video_id"]] = rec["score_av"]
            assert set(written) == set(ids["test"])
            for v in ids["test"]:
                assert written[v] == [round(float(x), 6) for x in runs["test"][v]["scores"][calls[v]]]
            metric = load(dest / f"metrics_test_{name}.json")["results"]["score_av"]
            assert metric["n_videos"] == len(ids["test"]) == 1151
            assert metric["n_frames"] == sum(len(gt["test"][v]) for v in ids["test"])
            assert metric["n_videos_missing_from_scores"] == 0 and metric["n_videos_not_in_gold"] == 0
            values = [metric["pr_auc"], metric["roc_auc"], metric["per_video"]["macro_auc"]]
            assert values == row["results"][name]["ap_roc_within"]
            counts = np.asarray(list(calls.values()))
            out["results"][name] = {"metric_source": str(dest / f"metrics_test_{name}.json"),
                "ap_roc_within": values, "mean_calls": float(counts.mean()),
                "mean_calls_positive": float(np.mean([calls[v] for v in calls if labels[v] == 1])),
                "mean_calls_negative": float(np.mean([calls[v] for v in calls if labels[v] == 0]))}
        rows.append(out)
    return rows, {"seed": seed, "host": HOSTS[seed], "budget": budget, "complete_trials": len(study["trials"]),
                  "failed_trials": 0, "test_selected": test_best, "validation_selected": val_best,
                  "ended": completion["ended"]}


def aggregate(rows):
    report = {}
    for selection in ("test_selected", "validation_selected"):
        chosen = sorted((r for r in rows if selection in r["selections"]), key=lambda r: r["seed"])
        assert [r["seed"] for r in chosen] == [234, 2025, 3407]
        block = {}
        for name in SETTINGS:
            values = np.array([r["results"][name]["ap_roc_within"] for r in chosen])
            calls = np.array([r["results"][name]["mean_calls"] for r in chosen])
            block[name] = {"mean": values.mean(0).tolist(), "std_population": values.std(0).tolist(),
                           "calls_mean": float(calls.mean()), "calls_std_population": float(calls.std())}
        diff = np.array([np.subtract(r["results"][RULE]["ap_roc_within"], r["results"]["fixed8"]["ap_roc_within"])
                         for r in chosen])
        block["auto_minus_same_copy_fixed8"] = {"paired_mean": diff.mean(0).tolist(),
            "paired_std_population": diff.std(0).tolist(), "paired_seed_values": diff.tolist()}
        report[selection] = block
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default="runs/20261002_complete_io_method")
    ap.add_argument("--r5-root", default="runs/20260929_query_paradigm_r5/stop_check/dehate")
    ap.add_argument("--seeds", type=int, nargs="+", choices=(234, 2025, 3407),
                    help="Audit these seeds on the host holding the DeHate cohort; writes a partial file")
    ap.add_argument("--merge", nargs="+", help="Partial audit files to combine into the final analysis")
    a = ap.parse_args()
    root = (ROOT / a.run_root).resolve()
    if a.seeds:
        labels, ids, gt, _ = hc.load_fixed_cohort("dehate")
        rows, studies = [], []
        for seed in a.seeds:
            r, s = audit_study(root / "dehate" / f"seed{seed}", seed, labels, ids, gt)
            rows += r
            studies.append(s)
        out = root / "analysis" / f"dehate_audit_partial_{'_'.join(map(str, a.seeds))}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"audited_at": datetime.now().astimezone().isoformat(),
            "host": socket.gethostname(), "test_frames": sum(len(gt["test"][v]) for v in ids["test"]),
            "studies": studies, "trials": rows}, indent=2) + "\n")
        print(out)
        return
    parts = [load(p) for p in a.merge]
    rows = [r for p in parts for r in p["trials"]]
    studies = sorted((s for p in parts for s in p["studies"]), key=lambda s: s["seed"])
    assert [s["seed"] for s in studies] == [234, 2025, 3407] and len({p["test_frames"] for p in parts}) == 1
    gt_frames = parts[0]["test_frames"]
    agg = aggregate(rows)
    r5_rows = []
    for seed, trial in R5.items():
        source = (ROOT / a.r5_root / f"seed{seed}_trial{trial}_r5rt/metrics_test_qmixSG_rt108.json").resolve()
        m = load(source)["results"]["score_av"]
        r5_rows.append({"seed": seed, "source": str(source),
                        "ap_roc_within": [m["pr_auc"], m["roc_auc"], m["per_video"]["macro_auc"]]})
    r5 = np.array([r["ap_roc_within"] for r in r5_rows])
    report = {"audited_at": datetime.now().astimezone().isoformat(), "corpus": "dehate",
              "status": "complete_and_returned", "test_videos": 1151,
              "test_frames": gt_frames, "test_evaluator_files": len(rows) * len(SETTINGS),
              "partial_audits": [{"host": p["host"], "audited_at": p["audited_at"],
                                  "seeds": [s["seed"] for s in p["studies"]]} for p in parts],
              "checks": ["completion/host/20 complete trials x 50 epochs, 13 training evaluator files each",
                         "test and validation selection re-derived from study trial attributes",
                         "checkpoint epoch, copy train-negative IDs and validation-only quantile reference",
                         "recomputed unchanged stopping threshold and per-video call counts",
                         "exact score prefix after existing six-decimal writer",
                         "all four unified evaluator outputs and complete cohort"],
              "studies": studies, "trials": rows, "aggregate": agg,
              "r5_same_copy_stopping_reference": {"rows": r5_rows, "mean": r5.mean(0).tolist(),
                  "std_population": r5.std(0).tolist(),
                  "integrated_minus_r5": (np.array(agg["test_selected"][RULE]["mean"]) - r5.mean(0)).tolist()},
              "scope": "New-backbone DeHate full 20-trial searches; frozen fixed8 no-copy winners evaluated "
                       "with the complete combination, no integrated-result reselection"}
    out = root / "analysis/dehate_complete_method.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(out), "aggregate": agg,
                      "integrated_minus_r5": report["r5_same_copy_stopping_reference"]["integrated_minus_r5"]},
                     indent=1))


if __name__ == "__main__":
    main()
