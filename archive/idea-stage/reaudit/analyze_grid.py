#!/usr/bin/env python
"""RE-AUDIT -- generic powered read-out for a head-level arm grid.

Frozen design: idea-stage/REAUDIT_FREEZE.md.

Parsing, confusion-matrix reconstruction and epoch selection are imported
VERBATIM from idea-stage/r6_audit/analyze_audit.py -- one parser, no fork.

Read-out protocols, both computed from the SAME runs:
  P1 (primary)       epoch = argmax_{e>=5} dev macro-F1 (ties -> earliest);
                     test macro-F1 @ threshold 0.5
  P2 (corroboration) epoch = 29 (last of 30); test macro-F1 @ threshold 0.5

Test labels are read only for the final metric.  No threshold, no epoch rule and
no arm definition is selected on them.

Frozen decision rule (idea-stage/REAUDIT_FREEZE.md section "Decision rule"):
a candidate is REVIVED iff, under P1, EVERY listed contrast has
mean >= +0.005 with its paired-bootstrap 95% CI excluding zero, and P2 agrees in
sign on each.  Anything else is NOT REVIVED.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(ROOT, "idea-stage", "r6_audit"))
from analyze_audit import parse, select_epoch, SPLIT, LAST_EPOCH, WARMUP  # noqa: E402

BAR = 0.005
N_BOOT = 20000
BOOT_SEED = 20260817
PROTOCOLS = ["P1", "P2"]


def paired_bootstrap_ci(d, n_boot, rng):
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float(means.std(ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arms", required=True, help="csv")
    ap.add_argument("--seeds", required=True, help="csv")
    ap.add_argument("--contrasts", required=True,
                    help="csv of LEFT-RIGHT pairs, e.g. OCR-A0,OCR-RAND")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ds = a.dataset
    arms = a.arms.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    pairs = [tuple(c.split("-", 1)) for c in a.contrasts.split(",")]
    rng = np.random.default_rng(BOOT_SEED)

    runs, bad = {}, []
    for arm in arms:
        for s in seeds:
            p = os.path.join(a.logdir, "%s_%s_s%d.trainlog" % (ds, arm, s))
            if not os.path.exists(p):
                bad.append((arm, s, "missing"))
                continue
            r, err = parse(p, ds)
            if r is None:
                bad.append((arm, s, err))
            else:
                runs[(arm, s)] = r
    if bad:
        print("BAD RUNS (%d):" % len(bad), bad[:20])
        raise SystemExit("HALT: incomplete grid")

    f1 = {}
    sel_epoch = {}
    for prot in PROTOCOLS:
        for arm in arms:
            v = []
            e_sel = []
            for s in seeds:
                r = runs[(arm, s)]
                e = select_epoch(r["dev"], prot)
                v.append(r["test"][e]["macro_f1"])
                e_sel.append(e)
            f1[(prot, arm)] = np.asarray(v, dtype=np.float64)
            sel_epoch[(prot, arm)] = e_sel

    res = {"what": "RE-AUDIT powered grid read-out",
           "freeze": "idea-stage/REAUDIT_FREEZE.md",
           "dataset": ds, "arms": arms, "seeds": seeds,
           "n_runs_parsed": len(runs), "n_boot": N_BOOT,
           "bootstrap_rng_seed": BOOT_SEED, "bar": BAR,
           "arm_means": {}, "contrasts": {}}

    for prot in PROTOCOLS:
        for arm in arms:
            v = f1[(prot, arm)]
            res["arm_means"]["%s/%s" % (prot, arm)] = {
                "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "per_seed": [float(x) for x in v],
                "mean_selected_epoch": float(np.mean(sel_epoch[(prot, arm)]))}

    for prot in PROTOCOLS:
        for (L, R) in pairs:
            d = f1[(prot, L)] - f1[(prot, R)]
            lo, hi, se = paired_bootstrap_ci(d, N_BOOT, rng)
            res["contrasts"]["%s/%s-%s" % (prot, L, R)] = {
                "mean": float(d.mean()), "std": float(d.std(ddof=1)),
                "se_boot": se, "ci95": [lo, hi],
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "n_positive": int((d > 0).sum()), "n_seeds": len(d),
                "passes_bar_and_ci": bool(d.mean() >= BAR and lo > 0),
                "per_seed": [float(x) for x in d]}

    p1_pass = all(res["contrasts"]["P1/%s-%s" % p]["passes_bar_and_ci"] for p in pairs)
    p2_sign = all(np.sign(res["contrasts"]["P2/%s-%s" % p]["mean"])
                  == np.sign(res["contrasts"]["P1/%s-%s" % p]["mean"]) for p in pairs)
    res["decision"] = {
        "rule": "REVIVED iff every listed P1 contrast has mean >= +0.005 with "
                "paired-bootstrap 95% CI excluding 0, and P2 agrees in sign on each",
        "p1_all_pass": bool(p1_pass), "p2_sign_agrees": bool(p2_sign),
        "verdict": "REVIVED" if (p1_pass and p2_sign) else "NOT REVIVED"}

    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["decision"], indent=1))
    for k, v in res["contrasts"].items():
        print("%-24s mean=%+.4f ci=[%+.4f, %+.4f] pos=%d/%d"
              % (k, v["mean"], v["ci95"][0], v["ci95"][1],
                 v["n_positive"], v["n_seeds"]))


if __name__ == "__main__":
    main()
