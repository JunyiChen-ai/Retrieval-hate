"""Revision-4 summary (README section 15): per corpus, the best trial of each seed's search (the Optuna objective,
rule 7), 3-seed means of the budget curve, the pre-registered checks Q1-Q4 of README 15.2 against revision 3, the
ablation table (rule 14(g): an arm is claimable when the 3-seed mean of full_rerun - arm drops AP or ROC by >= .01 on
both corpora) and the default-hyperparameter controls (runs/20260927_query_paradigm_r4/diag).

    python experiments/20260925_query_paradigm/summarize_r4.py [--corpus hatemm hateclipseg] [--root R]
Output: <R>/summary.json (default R = runs/20260927_query_paradigm_r4) and printed tables. All numbers are read
from the evaluator-written summary.json of each run.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
R4 = os.path.join(ROOT, "runs", "20260927_query_paradigm_r4")
R3_SUMMARY = os.path.join(ROOT, "runs", "20260925_query_paradigm_r3", "summary.json")
SEEDS = (234, 2025, 3407)
BUDGETS = ("0", "1", "2", "4", "8", "16", "32")
ARMS = ("no_node", "k30", "a_hate_only", "b_flat", "c_label", "d_bfs", "e_independent", "f_joint", "g_coupled")
CONTROLS = ("abl_full", "abl_no_node", "abl_noback", "abl_lvl4", "abl_lvl8", "abl_lvl16")
KEYS = ("pooled_ap", "pooled_roc", "within_roc")


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)), float(np.std(xs)), len(xs)) if xs else (None, None, 0)


def corpus_summary(root, c, r3):
    seeds = {}
    for s in SEEDS:
        st = load(os.path.join(root, c, "seed%d" % s, "study_summary.json"))
        if st is None or st.get("best") is None:
            continue
        best = st["best"]["number"]
        summ = load(os.path.join(root, c, "seed%d" % s, "trial%d" % best, "summary.json"))
        if summ is None:
            continue
        row = {"trial": best, "params": st["best"]["params"], "extra": st.get("extra"),
               "fixed": {k: v["test"] for k, v in summ["fixed"].items()},
               "fixed_calls": {k: v["test_mean_calls"] for k, v in summ["fixed"].items()}}
        for rule in ("adaptive", "adaptive_voi"):
            row[rule] = {k: dict(v["test"], mean_calls=v["test_mean_calls"]) for k, v in summ.get(rule, {}).items()}
        vs = st.get("validation_selected") or {}
        row["validation_selected"] = {"trial": vs.get("number"),
                                      "test": {k: vs.get("user_attrs", {}).get("test_" + k) for k in KEYS}}
        seeds[str(s)] = row
    agg = {}
    for B in BUDGETS:
        agg["fixed%s" % B] = {k: mean_std([r["fixed"].get(B, {}).get(k) for r in seeds.values()]) for k in KEYS}
    for rule in ("adaptive", "adaptive_voi"):
        for B in ("2", "4", "8"):
            agg["%s%s" % (rule, B)] = {k: mean_std([r[rule].get(B, {}).get(k) for r in seeds.values()])
                                       for k in KEYS + ("mean_calls",)}
    agg["validation_selected"] = {k: mean_std([r["validation_selected"]["test"][k] for r in seeds.values()])
                                  for k in KEYS}
    checks = {}
    f0, f8 = agg["fixed0"], agg["fixed8"]
    if f8["pooled_ap"][0] is not None and r3 is not None:
        r3f8 = r3["corpora"][c]["mean"]["fixed8"]
        checks["Q1"] = {k: {"mean": f8[k][0], "floor": r3f8[k][0] - .005, "pass": f8[k][0] >= r3f8[k][0] - .005}
                        for k in ("pooled_ap", "pooled_roc")}
    if f8["pooled_ap"][0] is not None:
        checks["Q2"] = {"within8": f8["within_roc"][0], "within0": f0["within_roc"][0],
                        "pass": f8["within_roc"][0] >= f0["within_roc"][0]}
        checks["Q3_curve"] = {B: {k: {"mean": agg["fixed%s" % B][k][0], "floor": f8[k][0] - .005,
                                      "pass": agg["fixed%s" % B][k][0] >= f8[k][0] - .005}
                                  for k in ("pooled_ap", "pooled_roc")} for B in ("16", "32")}
        for rule in ("adaptive", "adaptive_voi"):
            a8 = agg["%s8" % rule]
            if a8["pooled_ap"][0] is not None:
                checks["Q3_P2b_" + rule] = {k: {"mean": a8[k][0], "fixed8": f8[k][0],
                                                "pass": a8[k][0] >= f8[k][0] - .005,
                                                "mean_calls": a8["mean_calls"][0]} for k in ("pooled_ap", "pooled_roc")}
    abl = {}
    for arm in ARMS:
        d = []
        for s in SEEDS:
            base = load(os.path.join(root, "ablations", c, "seed%d" % s, "full_rerun", "summary.json"))
            a = load(os.path.join(root, "ablations", c, "seed%d" % s, arm, "summary.json"))
            if base and a:
                d.append({k: base["fixed"]["8"]["test"][k] - a["fixed"]["8"]["test"][k] for k in KEYS})
        abl[arm] = {"n_seeds": len(d), **{k: (float(np.mean([x[k] for x in d])) if d else None) for k in KEYS}}
    reruns = {}
    for s in SEEDS:
        base = load(os.path.join(root, "ablations", c, "seed%d" % s, "full_rerun", "summary.json"))
        if base and str(s) in seeds:
            reruns[str(s)] = {k: base["fixed"]["8"]["test"][k] - seeds[str(s)]["fixed"]["8"][k] for k in KEYS}
    ctrl = {}
    for tag in CONTROLS:
        runs = [load(os.path.join(root, "diag", c, "seed%d" % s, tag, "summary.json")) for s in SEEDS]
        runs = [r for r in runs if r]
        if runs:
            ctrl[tag] = {"n_seeds": len(runs),
                         **{B: {k: float(np.mean([r["fixed"][B]["test"][k] for r in runs])) for k in KEYS}
                            for B in BUDGETS},
                         "mean_calls8": float(np.mean([r["fixed"]["8"]["test_mean_calls"] for r in runs]))}
    return {"seeds": seeds, "mean": agg, "checks": checks, "ablation_full_minus_arm": abl,
            "rerun_minus_best_trial": reruns, "controls": ctrl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", default=["hatemm", "hateclipseg"])
    ap.add_argument("--root", default=R4)
    a = ap.parse_args()
    r3 = load(R3_SUMMARY)
    out = {"corpora": {c: corpus_summary(a.root, c, r3 if c in ("hatemm", "hateclipseg") else None)
                       for c in a.corpus}}
    claims = {}
    for arm in ARMS:
        ok = []
        for c in a.corpus:
            x = out["corpora"][c]["ablation_full_minus_arm"][arm]
            ok.append(x["n_seeds"] == len(SEEDS) and max(x["pooled_ap"], x["pooled_roc"]) >= .01)
        claims[arm] = all(ok)
    out["ablation_claimable_all_listed_corpora"] = claims
    json.dump(out, open(os.path.join(a.root, "summary.json" if len(a.corpus) > 1 else "summary_%s.json" % a.corpus[0]),
                        "w"), indent=2)
    for c in a.corpus:
        o = out["corpora"][c]
        print("== %s (seeds %s)" % (c, sorted(o["seeds"])))
        for s, r in sorted(o["seeds"].items()):
            f = r["fixed"]["8"]
            print("  seed %s trial %d: fixed 8 AP %.4f ROC %.4f within %.4f (calls %.2f)" % (
                s, r["trial"], f["pooled_ap"], f["pooled_roc"], f["within_roc"], r["fixed_calls"]["8"]))
        for B in BUDGETS:
            m = o["mean"]["fixed%s" % B]
            if m["pooled_ap"][0] is not None:
                print("  B=%-2s AP %.4f±%.4f ROC %.4f±%.4f within %.4f±%.4f" % (
                    B, m["pooled_ap"][0], m["pooled_ap"][1], m["pooled_roc"][0], m["pooled_roc"][1],
                    m["within_roc"][0], m["within_roc"][1]))
        v = o["mean"]["validation_selected"]
        if v["pooled_ap"][0] is not None:
            print("  validation-selected trial (test): AP %.4f ROC %.4f within %.4f" % tuple(v[k][0] for k in KEYS))
        for k, x in o["checks"].items():
            print("  %s: %s" % (k, json.dumps(x, default=float)))
        for arm, x in o["ablation_full_minus_arm"].items():
            if x["n_seeds"]:
                print("  ablation %-14s (%d seeds) full - arm: AP %+.4f ROC %+.4f within %+.4f" % (
                    arm, x["n_seeds"], x["pooled_ap"], x["pooled_roc"], x["within_roc"]))
        print("  rerun - best trial:", o["rerun_minus_best_trial"])
        for tag, x in o["controls"].items():
            print("  control %-11s (%d seeds) " % (tag, x["n_seeds"]) + " | ".join(
                "B%s %.3f/%.3f/%.3f" % (B, x[B]["pooled_ap"], x[B]["pooled_roc"], x[B]["within_roc"])
                for B in ("0", "2", "8", "32")) + " | calls8 %.2f" % x["mean_calls8"])
    print("claimable on all listed corpora:", claims)


if __name__ == "__main__":
    main()
