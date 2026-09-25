"""DeHate results table (README section 5): four baselines (three-seed retraining of the validation-selected
configuration) and the query-tree method (best trial of each seed's 20-trial search, and the validation-selected
trial), test pooled AP / pooled ROC / within-video ROC, mean and std over seeds 234 / 2025 / 3407.

    python experiments/20260926_dehate_external/summarize.py
Reads runs/20260926_dehate_external/baselines/final/<method>/dehate/seed_<s>/frame_eval.json (evaluator output) and
runs/20260926_dehate_external/qtl/dehate/seed<s>/{study_summary.json, trial<k>/summary.json}; writes
runs/20260926_dehate_external/summary.json.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
R = os.path.join(ROOT, "runs", "20260926_dehate_external")
SEEDS = (234, 2025, 3407)
BRANCH = {"macilsd": "score_av", "multihateloc": "score_fused", "dsanet": "score_mlp",
          "fed_wsvad_3client": "score_align"}          # scripts/reproduction_baselines/aggregate_official_val.py
KEYS = ("pooled_ap", "pooled_roc", "within_roc")
BUDGETS = ("0", "2", "4", "8", "16", "32")


def agg(rows):
    out = {"n_seeds": len(rows)}
    for k in rows[0] if rows else ():
        v = [r[k] for r in rows if r.get(k) is not None]
        if v and all(isinstance(x, (int, float)) for x in v):
            out[k] = float(np.mean(v))
            out[k + "_std"] = float(np.std(v))
    return out


def main():
    res = {"baselines": {}, "query_tree": {}}
    for m, br in BRANCH.items():
        rows = []
        for s in SEEDS:
            p = os.path.join(R, "baselines", "final", m, "dehate", "seed_%d" % s, "frame_eval.json")
            if os.path.exists(p):
                r = json.load(open(p))["results"][br]
                rows.append({"pooled_ap": r["pr_auc"], "pooled_roc": r["roc_auc"],
                             "within_roc": r["per_video"]["macro_auc"]})
        if rows:
            res["baselines"][m] = agg(rows)
    best, vsel, curve = [], [], {B: [] for B in BUDGETS}
    for s in SEEDS:
        p = os.path.join(R, "qtl", "dehate", "seed%d" % s, "study_summary.json")
        if not os.path.exists(p):
            continue
        st = json.load(open(p))
        ua = st["best"]["user_attrs"]
        best.append({"pooled_ap": ua["test_pooled_ap"], "pooled_roc": ua["test_pooled_roc"],
                     "within_roc": ua["test_within_roc"], "calls": ua["test_mean_calls"],
                     "trial": st["best"]["number"]})
        va = st["validation_selected"]["user_attrs"]
        vsel.append({"pooled_ap": va["test_pooled_ap"], "pooled_roc": va["test_pooled_roc"],
                     "within_roc": va["test_within_roc"], "trial": st["validation_selected"]["number"]})
        summ = json.load(open(os.path.join(R, "qtl", "dehate", "seed%d" % s, "trial%d" % st["best"]["number"],
                                           "summary.json")))
        for B in BUDGETS:
            if B in summ["fixed"]:
                t = summ["fixed"][B]["test"]
                curve[B].append({k: t[k] for k in KEYS} | {"calls": summ["fixed"][B]["test_mean_calls"]})
    if best:
        res["query_tree"] = {"best_trial": agg(best), "validation_selected": agg(vsel),
                             "per_seed_best": best, "budget_curve_best_trial": {B: agg(v) for B, v in curve.items()
                                                                                if v}}
    json.dump(res, open(os.path.join(R, "summary.json"), "w"), indent=1)
    fmt = lambda d: "%.4f ± %.4f / %.4f ± %.4f / %.4f" % (d["pooled_ap"], d["pooled_ap_std"], d["pooled_roc"],  # noqa
                                                        d["pooled_roc_std"], d["within_roc"])
    for m, d in res["baselines"].items():
        print("%-18s (%d seeds) %s" % (m, d["n_seeds"], fmt(d)))
    if best:
        print("%-18s (%d seeds) %s" % ("query-tree best", len(best), fmt(res["query_tree"]["best_trial"])))
        print("%-18s (%d seeds) %s" % ("query-tree val-sel", len(vsel), fmt(res["query_tree"]["validation_selected"])))
        for B, d in res["query_tree"]["budget_curve_best_trial"].items():
            print("  B=%-3s %.4f / %.4f / %.4f  calls %.2f" % (B, d["pooled_ap"], d["pooled_roc"], d["within_roc"],
                                                              d["calls"]))


if __name__ == "__main__":
    main()
