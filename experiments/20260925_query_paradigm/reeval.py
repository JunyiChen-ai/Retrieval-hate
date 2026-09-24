"""Re-evaluate a finished trial's checkpoint with the decision-relevant stopping rule (stopping.py; README section
2.4 revision). Loads <trial>/model.pth and config.json, runs the EIG policy with VOI recording on validation and
test, and reports: fixed budgets (must reproduce the trial's summary), adaptive VOI-threshold rule at mean budgets
calibrated on validation, and the old EIG-threshold rule for reference. Test numbers through the shared evaluator.

    python experiments/20260925_query_paradigm/reeval.py --trial runs/20260925_query_paradigm/<corpus>/seed<s>/trial<k>
Output: <trial>/reeval/{summary.json, scores_test_*.jsonl, metrics_test_*.json}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train as T                               # noqa: E402  (sets up the import paths)
import hier_evidence_common as hc               # noqa: E402
import data as qdata                            # noqa: E402
import qtree                                    # noqa: E402
import policy                                   # noqa: E402
import stopping                                 # noqa: E402
from model import PriorNet                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    summ = json.load(open(os.path.join(a.trial, "summary.json")))
    cfg = dict(T.DEFAULTS)
    cfg.update(summ["cfg"])
    corpus = summ["corpus"]
    out = os.path.join(a.trial, "reeval")
    os.makedirs(out, exist_ok=True)
    labels, ids, gt, _ = hc.load_fixed_cohort(corpus)
    answers, T_ans = qdata.load_answers(corpus)
    store = qdata.Store(corpus, ids["train"] + ids["val"] + ids["test"], cfg.get("text_sources", ["bert"]))
    ck = torch.load(os.path.join(a.trial, "model.pth"), map_location=a.device)
    model = PriorNet(cfg).to(a.device)
    model.load_state_dict(ck["model"])
    model.eval()
    th = ck["am"]["theta"]
    am = qtree.AnswerModel(th.cpu().numpy(), 0.0, 1.0, bool(cfg["length_term"]), int(cfg["n_state"]),
                           cfg["categories"]).to(a.device)
    am.load_state_dict(ck["am"], strict=False)
    # length standardisation constants are not parameters: recompute them exactly as train.py did
    loglen = np.log([b - a_ for v in ids["train"] for (a_, b) in answers[v]])
    am.len_mu, am.len_sd = float(loglen.mean()), float(loglen.std())
    max_calls = int(cfg["max_calls"])

    def run(split):
        res = {}
        for v in ids[split]:
            s, g = policy.video_prior(model, store, v, a.device)
            vt = qtree.VideoTree(store.T[v])
            res[v] = stopping.run_video(vt, s, g, policy.Asker(vt, am, cfg["categories"]), answers[v], max_calls)
        return res

    val_runs, test_runs = run("val"), run("test")
    hate_val = {v for v in ids["val"] if labels[v] == 1}

    def test_eval(name, scores):
        sp = os.path.join(out, "scores_test_%s.jsonl" % name)
        hc.write_scores(sp, scores)
        r = hc.run_evaluator(corpus, "test", sp, os.path.join(out, "metrics_test_%s.json" % name))["results"]["score_av"]
        return {"pooled_ap": r["pr_auc"], "pooled_roc": r["roc_auc"], "within_roc": r["per_video"]["macro_auc"]}

    res = {"fixed": {}, "adaptive_voi": {}, "adaptive_eig": {}}
    for B in cfg["fixed_budgets"]:
        B = int(B)
        res["fixed"][str(B)] = {"test": test_eval("fixed%d" % B, T.at_budget(test_runs, B))}
    for key, rule in (("adaptive_voi", "voi"), ("adaptive_eig", "eig")):
        for B in cfg["mean_budgets"]:
            c, vmean = policy.calibrate([r[rule] for r in val_runs.values()], float(B))
            calls = {v: policy.stop_calls(r[rule], c) for v, r in test_runs.items()}
            st = {v: r["scores"][calls[v]] for v, r in test_runs.items()}
            sv = {v: r["scores"][policy.stop_calls(r[rule], c)] for v, r in val_runs.items()}
            cv = np.array(list(calls.values()))
            res[key][str(B)] = {
                "c": c, "val_mean_calls": vmean, "val": hc.frame_metrics(sv, gt["val"], hate_val),
                "test": test_eval("%s%d" % (key, B), st), "test_mean_calls": float(cv.mean()),
                "test_mean_calls_pos": float(np.mean([calls[v] for v in calls if labels[v] == 1])),
                "test_mean_calls_neg": float(np.mean([calls[v] for v in calls if labels[v] == 0])),
                "test_calls_quantiles": [float(x) for x in np.percentile(cv, [0, 25, 50, 75, 100])]}
            r = res[key][str(B)]
            print("%s mean %d | test calls %.2f (pos %.2f neg %.2f) | AP %.4f ROC %.4f within %.4f" % (
                key, B, r["test_mean_calls"], r["test_mean_calls_pos"], r["test_mean_calls_neg"],
                r["test"]["pooled_ap"], r["test"]["pooled_roc"], r["test"]["within_roc"]), flush=True)
    for B in cfg["fixed_budgets"]:
        r = res["fixed"][str(B)]["test"]
        ref = summ["fixed"][str(int(B))]["test"]
        print("fixed %2d | AP %.4f ROC %.4f | trial summary AP %.4f ROC %.4f" % (
            int(B), r["pooled_ap"], r["pooled_roc"], ref["pooled_ap"], ref["pooled_roc"]), flush=True)
    res["asked_length"] = {str(k): float(np.mean([store.T[v] and (qtree.tree(store.T[v])["b"][r["asked"][k]]
                                                               - qtree.tree(store.T[v])["a"][r["asked"][k]])
                                                  for v, r in test_runs.items() if len(r["asked"]) > k]))
                           for k in range(8)}
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
