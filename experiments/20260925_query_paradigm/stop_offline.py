"""Revision 5 step 4, offline analysis and offline rule evaluation on the runs saved by stop_check.py --dump
(runs.pkl: per video and per call the per-second posterior, P(G = 1), EIG, VOI). Nothing here re-runs the policy.
README section 17.4 "no floor" diagnostics (development evidence on test under rule 10; video labels are training
labels and may be used; per-second labels only in the diagnosis, never in a rule):
  calib   calibration of P(G = 1 | k answers) against the video label, validation and test, k in {0, 1, 2, 4, 8}
  grid    class-budget grid: positives k1 calls, negatives k2 calls (test labels; an upper bound on what any rule
          that only re-allocates calls between the classes can reach at a mean budget)
  rules   offline stopping rules from the saved runs, calibrated on validation by mean calls (policy.calibrate),
          reported on test through the shared metric code (hc.frame_metrics; the evaluator script gives the same
          numbers, see stop_check.py for the file-based evaluation)
  decomp  for one rule at one budget: test metrics with the rule's scores for one class and fixed-8 scores for the other

    python experiments/20260925_query_paradigm/stop_offline.py --corpus hatemm --dumps <dirs with runs.pkl> \\
        --what calib grid rules decomp [--rules hG dlogit ...] [--budgets 4 8 12 16]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import train as TR                              # noqa: E402  (import paths)
import hier_evidence_common as hc               # noqa: E402
import policy                                   # noqa: E402
from stop_check import rule_values, _hb         # noqa: E402

M = ("pooled_ap", "pooled_roc", "within_roc")


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


def offline_values(run, T):
    """All stop_check rules plus offline-only ones (values read before call k+1, k = 0..n-1):
      dlogit  mean over seconds of |logit p_t[k] - logit p_t[k-1]| (the change made by the last call, on the ranking
              scale; +inf before the first call)
      dG      |logit P(G)[k] - logit P(G)[k-1]|
      dlogit2 the larger of the last two dlogit values
      lG      -logit P(G = 1): ask while the video is not yet confidently negative on the log-odds scale (positives:
              value negative, stop at once)
      aG      |logit P(G = 1)| below c means undecided: ask while |logit P(G)| <= c  (implemented as -|logit|)"""
    vals = rule_values(run, T)
    n = len(run["eig"])
    lp = [_logit(run["scores"][k]) for k in range(n + 1)]
    lg = _logit(run["p_G"][: n + 1])
    d = [np.inf] + [float(np.mean(np.abs(lp[k] - lp[k - 1]))) for k in range(1, n + 1)]
    dg = [np.inf] + [float(abs(lg[k] - lg[k - 1])) for k in range(1, n + 1)]
    vals["dlogit"] = d[:n]
    vals["dlogit2"] = [max(d[k], d[k - 1] if k >= 1 else np.inf) for k in range(n)]
    vals["dG"] = dg[:n]
    vals["lG"] = [float(-lg[k]) for k in range(n)]
    vals["aG"] = [float(-abs(lg[k])) for k in range(n)]
    return vals


def metrics(scores, gt, hate_ids):
    m = hc.frame_metrics(scores, gt, hate_ids)
    return np.array([m[k] for k in M])


def fmt(m):
    return "%.3f/%.3f/%.3f" % tuple(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--dumps", nargs="+", required=True, help="directories holding runs.pkl (one per trial)")
    ap.add_argument("--what", nargs="+", default=["calib", "grid", "rules", "decomp"])
    ap.add_argument("--rules", nargs="*", default=["eig", "stab1", "hG", "hT", "hmax", "dlogit", "dlogit2", "dG", "lG", "aG"])
    ap.add_argument("--budgets", nargs="*", type=int, default=[4, 8, 12, 16])
    ap.add_argument("--decomp-rule", default="hG")
    ap.add_argument("--decomp-budget", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    hate = {sp: {v for v in ids[sp] if labels[v] == 1} for sp in ("val", "test")}
    res = {"corpus": a.corpus, "trials": {}}
    agg = {}

    def acc(key, m):
        agg.setdefault(key, []).append(np.asarray(m, dtype=np.float64))

    for d in a.dumps:
        D = pickle.load(open(os.path.join(d, "runs.pkl"), "rb"))
        runs, Ts = D["runs"], D["T"]
        tag = d.rstrip("/").split("/")[-1]
        print("==== %s" % tag, flush=True)
        rt = {}
        gts = {sp: {v: gt[sp][v] for v in runs[sp]} for sp in runs}

        def sc_at(sp, calls):
            return {v: runs[sp][v]["scores"][min(calls[v], len(runs[sp][v]["eig"]))] for v in runs[sp]}

        fixed = {}
        for B in (0, 2, 4, 8, 12, 16, 32):
            m = metrics(sc_at("test", {v: B for v in runs["test"]}), gts["test"], hate["test"])
            fixed[B] = m
            acc(("fixed", B), m)
            print("fixed %2d %s" % (B, fmt(m)))
        rt["fixed"] = {str(B): list(map(float, m)) for B, m in fixed.items()}

        if "calib" in a.what:
            edges = [0, .01, .02, .05, .1, .2, .5, .8, .95, 1.0001]
            for sp in ("val", "test"):
                for k in (0, 1, 2, 4, 8):
                    pg = np.array([runs[sp][v]["p_G"][min(k, len(runs[sp][v]["eig"]))] for v in runs[sp]])
                    y = np.array([labels[v] for v in runs[sp]])
                    row = []
                    for lo, hi in zip(edges[:-1], edges[1:]):
                        s = (pg >= lo) & (pg < hi)
                        row.append("%s:%d/%d" % (("[%g,%g)" % (lo, hi)), int(y[s].sum()), int(s.sum())))
                    print("calib %-4s k=%d  " % (sp, k) + "  ".join(row))
                    # negatives' per-second score level: mean and 90th percentile of the per-second scores
                    neg = np.concatenate([runs[sp][v]["scores"][min(k, len(runs[sp][v]["eig"]))] for v in runs[sp] if labels[v] == 0])
                    pos_h = np.concatenate([np.asarray(runs[sp][v]["scores"][min(k, len(runs[sp][v]["eig"]))])[np.asarray(gt[sp][v]) > 0]
                                            for v in runs[sp] if labels[v] == 1 and np.asarray(gt[sp][v]).sum() > 0])
                    print("      neg seconds mean %.4f p50 %.4f p90 %.4f | hate seconds mean %.3f p10 %.4f p25 %.4f p50 %.3f" % (
                        neg.mean(), np.median(neg), np.percentile(neg, 90), pos_h.mean(), np.percentile(pos_h, 10),
                        np.percentile(pos_h, 25), np.median(pos_h)))

        if "grid" in a.what:
            ks = [0, 2, 4, 6, 8, 12, 16, 24, 32]
            n_pos = sum(labels[v] == 1 for v in runs["test"])
            n_neg = len(runs["test"]) - n_pos
            rt["grid"] = {}
            for k1 in ks:
                for k2 in ks:
                    calls = {v: (k1 if labels[v] == 1 else k2) for v in runs["test"]}
                    m = metrics(sc_at("test", calls), gts["test"], hate["test"])
                    mean_calls = float(np.mean([min(calls[v], len(runs["test"][v]["eig"])) for v in runs["test"]]))
                    rt["grid"]["%d_%d" % (k1, k2)] = {"m": list(map(float, m)), "calls": mean_calls}
                    acc(("grid", k1, k2), np.r_[m, mean_calls])
                    print("grid pos %2d neg %2d | calls %5.2f | %s" % (k1, k2, mean_calls, fmt(m)))

        if "rules" in a.what or "decomp" in a.what:
            vals = {sp: {v: offline_values(runs[sp][v], Ts[v]) for v in runs[sp]} for sp in runs}
            rt["rules"] = {}
            rules = a.rules if "rules" in a.what else [a.decomp_rule]
            for rule in rules:
                rt["rules"][rule] = {}
                for B in a.budgets:
                    c, val_mean = policy.calibrate([vals["val"][v][rule] for v in vals["val"]], float(B))
                    calls = {v: policy.stop_calls(vals["test"][v][rule], c) for v in vals["test"]}
                    st = sc_at("test", calls)
                    m = metrics(st, gts["test"], hate["test"])
                    cv = np.array(list(calls.values()))
                    cp = np.mean([calls[v] for v in calls if labels[v] == 1])
                    cn = np.mean([calls[v] for v in calls if labels[v] == 0])
                    rt["rules"][rule][str(B)] = {"c": float(c), "val_mean_calls": val_mean, "test_mean_calls": float(cv.mean()),
                                                "pos": float(cp), "neg": float(cn), "m": list(map(float, m))}
                    acc(("rule", rule, B), np.r_[m, cv.mean(), cp, cn])
                    print("rule %-8s m%2d (c %.4g) calls %5.2f (pos %5.2f neg %5.2f) %s d8 %s" % (
                        rule, B, c, cv.mean(), cp, cn, fmt(m), fmt(m - fixed[8])))
                    if "decomp" in a.what and rule == a.decomp_rule and B == a.decomp_budget:
                        f8 = sc_at("test", {v: 8 for v in runs["test"]})
                        for cls, name in ((1, "rule on positives, fixed 8 on negatives"), (0, "rule on negatives, fixed 8 on positives")):
                            mix = {v: (st[v] if labels[v] == cls else f8[v]) for v in st}
                            m2 = metrics(mix, gts["test"], hate["test"])
                            acc(("decomp", rule, B, cls), m2)
                            print("decomp %s: %s d8 %s" % (name, fmt(m2), fmt(m2 - fixed[8])))
        res["trials"][tag] = rt

    print("==== means over %d trials" % len(a.dumps))
    ref = np.mean(agg[("fixed", 8)], axis=0)
    for key, xs in agg.items():
        m = np.mean(xs, axis=0)
        if key[0] == "fixed":
            print("fixed %2d %s" % (key[1], fmt(m)))
        elif key[0] == "grid":
            print("grid pos %2d neg %2d | calls %5.2f | %s d8 %s" % (key[1], key[2], m[3], fmt(m[:3]), fmt(m[:3] - ref)))
        elif key[0] == "rule":
            print("rule %-8s m%2d calls %5.2f (pos %5.2f neg %5.2f) %s d8 %s" % (key[1], key[2], m[3], m[4], m[5], fmt(m[:3]), fmt(m[:3] - ref)))
        elif key[0] == "decomp":
            print("decomp %s m%d class %d: %s d8 %s" % (key[1], key[2], key[3], fmt(m), fmt(m - ref)))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
