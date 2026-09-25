"""Revision-3 summary (README section 10): per corpus, the best trial of each seed's search (the Optuna objective,
rule 7), 3-seed means of the primary point (fixed 8 questions per video), the pre-registered checks P1, P2, P2b, P3
(README section 5) against it5, and the ablation table (README sections 9, 10; rule 14(g): an arm is a claimable
component when the 3-seed mean of full_rerun - arm drops AP or ROC by >= .01 on both corpora).

    python experiments/20260925_query_paradigm/summarize_r3.py
Output: runs/20260925_query_paradigm_r3/summary.json (and printed tables). All numbers are read from the
evaluator-written summary.json of each run.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
R3 = os.path.join(ROOT, "runs", "20260925_query_paradigm_r3")
SEEDS = (234, 2025, 3407)
CORPORA = ("hatemm", "hateclipseg")
IT5 = {"hatemm": {"pooled_ap": .6444, "pooled_roc": .8500, "within_roc": .6492},
       "hateclipseg": {"pooled_ap": .6837, "pooled_roc": .6809, "within_roc": .5597}}
IT5_SILENT_AP = .495
ARMS = ("a_hate_only", "b_flat", "c_label", "d_bfs", "e_independent", "f_joint", "g_coupled")
KEYS = ("pooled_ap", "pooled_roc", "within_roc")


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)), float(np.std(xs)), len(xs)) if xs else (None, None, 0)


def main():
    out = {"corpora": {}}
    for c in CORPORA:
        seeds = {}
        for s in SEEDS:
            st = load(os.path.join(R3, c, "seed%d" % s, "study_summary.json"))
            if st is None:
                continue
            best = st["best"]["number"]
            summ = load(os.path.join(R3, c, "seed%d" % s, "trial%d" % best, "summary.json"))
            if summ is None:
                continue
            row = {"trial": best, "params": st["best"]["params"],
                   "fixed": {k: v["test"] for k, v in summ["fixed"].items()},
                   "fixed_calls": {k: v["test_mean_calls"] for k, v in summ["fixed"].items()},
                   "silent_group": summ.get("silent_group")}
            for rule in ("adaptive", "adaptive_voi"):
                row[rule] = {k: dict(v["test"], mean_calls=v["test_mean_calls"]) for k, v in summ.get(rule, {}).items()}
            vs = st.get("validation_selected", {})
            row["validation_selected"] = {"trial": vs.get("number"),
                                          "test": {k: vs.get("user_attrs", {}).get("test_" + k) for k in KEYS}}
            seeds[str(s)] = row
        agg = {}
        for B in ("0", "1", "2", "4", "8", "16", "32"):
            agg["fixed%s" % B] = {k: mean_std([r["fixed"].get(B, {}).get(k) for r in seeds.values()]) for k in KEYS}
        for rule in ("adaptive", "adaptive_voi"):
            for B in ("2", "4", "8"):
                agg["%s%s" % (rule, B)] = {k: mean_std([r[rule].get(B, {}).get(k) for r in seeds.values()])
                                           for k in KEYS + ("mean_calls",)}
        agg["validation_selected"] = {k: mean_std([r["validation_selected"]["test"][k] for r in seeds.values()])
                                      for k in KEYS}
        f8 = agg["fixed8"]
        checks = {}
        if f8["pooled_ap"][0] is not None:
            checks["P1"] = {k: {"mean": f8[k][0], "floor": IT5[c][k] - .005, "pass": f8[k][0] >= IT5[c][k] - .005}
                            for k in ("pooled_ap", "pooled_roc")}
            for rule in ("adaptive", "adaptive_voi"):
                a4, a8 = agg["%s4" % rule], agg["%s8" % rule]
                if a4["pooled_ap"][0] is not None:
                    checks["P2_" + rule] = {k: {"mean": a4[k][0], "floor": IT5[c][k] - .01,
                                                "pass": a4[k][0] >= IT5[c][k] - .01,
                                                "mean_calls": a4["mean_calls"][0]} for k in ("pooled_ap", "pooled_roc")}
                    checks["P2b_" + rule] = {k: {"mean": a8[k][0], "fixed8": f8[k][0],
                                                 "pass": a8[k][0] >= f8[k][0] - .005,
                                                 "mean_calls": a8["mean_calls"][0]} for k in ("pooled_ap", "pooled_roc")}
            sil = [r["silent_group"]["ap_primary"] for r in seeds.values() if r.get("silent_group")]
            if c == "hateclipseg" and sil:
                checks["P3"] = {"mean_ap": float(np.mean(sil)), "it5": IT5_SILENT_AP,
                                "pass": float(np.mean(sil)) > IT5_SILENT_AP}
        abl = {}
        for arm in ARMS:
            d = []
            for s in SEEDS:
                base = load(os.path.join(R3, "ablations", c, "seed%d" % s, "full_rerun", "summary.json"))
                a = load(os.path.join(R3, "ablations", c, "seed%d" % s, arm, "summary.json"))
                if base and a:
                    d.append({k: base["fixed"]["8"]["test"][k] - a["fixed"]["8"]["test"][k] for k in KEYS})
            abl[arm] = {"n_seeds": len(d), **{k: (float(np.mean([x[k] for x in d])) if d else None) for k in KEYS}}
        reruns = {}
        for s in SEEDS:
            base = load(os.path.join(R3, "ablations", c, "seed%d" % s, "full_rerun", "summary.json"))
            if base and str(s) in seeds:
                reruns[str(s)] = {k: base["fixed"]["8"]["test"][k] - seeds[str(s)]["fixed"]["8"][k] for k in KEYS}
        out["corpora"][c] = {"seeds": seeds, "mean": agg, "checks": checks, "ablation_full_minus_arm": abl,
                             "rerun_minus_best_trial": reruns}
    claims = {}
    for arm in ARMS:
        ok = []
        for c in CORPORA:
            a = out["corpora"][c]["ablation_full_minus_arm"][arm]
            ok.append(a["n_seeds"] == len(SEEDS) and max(a["pooled_ap"], a["pooled_roc"]) >= .01)
        claims[arm] = all(ok)
    out["ablation_claimable_both_corpora"] = claims
    json.dump(out, open(os.path.join(R3, "summary.json"), "w"), indent=2)
    for c in CORPORA:
        o = out["corpora"][c]
        print("== %s (seeds %s)" % (c, sorted(o["seeds"])))
        for s, r in sorted(o["seeds"].items()):
            f = r["fixed"]["8"]
            print("  seed %s trial %d: fixed 8 AP %.4f ROC %.4f within %.4f (calls %.2f)" % (
                s, r["trial"], f["pooled_ap"], f["pooled_roc"], f["within_roc"], r["fixed_calls"]["8"]))
        f8 = o["mean"]["fixed8"]
        if f8["pooled_ap"][0] is not None:
            print("  mean fixed 8: AP %.4f±%.4f ROC %.4f±%.4f within %.4f±%.4f | it5 %.4f / %.4f / %.4f" % (
                f8["pooled_ap"][0], f8["pooled_ap"][1], f8["pooled_roc"][0], f8["pooled_roc"][1],
                f8["within_roc"][0], f8["within_roc"][1], *IT5[c].values()))
        for k, v in o["checks"].items():
            print("  %s: %s" % (k, json.dumps(v, default=float)))
        for arm, a in o["ablation_full_minus_arm"].items():
            if a["n_seeds"]:
                print("  ablation %-14s (%d seeds) full - arm: AP %+.4f ROC %+.4f within %+.4f" % (
                    arm, a["n_seeds"], a["pooled_ap"], a["pooled_roc"], a["within_roc"]))
        print("  rerun - best trial:", o["rerun_minus_best_trial"])
    print("claimable on both corpora:", claims)


if __name__ == "__main__":
    main()
