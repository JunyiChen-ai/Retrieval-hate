#!/usr/bin/env python
"""R10-COMBO -- dev-side / epoch-selection panel (REAUDIT_NCA lesson).

DIAGNOSTIC + one demotion clause, both frozen in idea-stage/R10_COMBO_FREEZE.md 3.

The judgement numbers come from idea-stage/reaudit/analyze_grid.py, unchanged.
This script adds the read-outs that grid analyzer does not print and that
REAUDIT_NCA showed are the cheapest tell for a selection-rule-bound effect:

  * dev macro-F1 at the P1-selected epoch, per arm and paired-contrasted
  * mean P1-selected epoch, per arm
  * dev - test gap at the P1-selected epoch, paired-contrasted
  * P2 (final epoch) test macro-F1, paired-contrasted

Parsing and epoch selection are imported VERBATIM from r6_audit/analyze_audit.py,
the same import analyze_grid.py uses.  Bootstrap settings identical (B=20000,
seed 20260817).
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(ROOT, "idea-stage", "r6_audit"))
from analyze_audit import parse, select_epoch  # noqa: E402

N_BOOT = 20000
BOOT_SEED = 20260817


def ci(d, rng):
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    m = d[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--contrasts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = a.arms.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    pairs = [tuple(c.split("-", 1)) for c in a.contrasts.split(",")]
    rng = np.random.default_rng(BOOT_SEED)

    q = {}   # (metric, arm) -> array over seeds
    for arm in arms:
        dv, tv, gp, ep, p2 = [], [], [], [], []
        for s in seeds:
            p = os.path.join(a.logdir, "%s_%s_s%d.trainlog" % (a.dataset, arm, s))
            r, err = parse(p, a.dataset)
            if r is None:
                raise SystemExit("HALT: %s -> %s" % (p, err))
            e = select_epoch(r["dev"], "P1")
            dv.append(r["dev"][e]["macro_f1"])
            tv.append(r["test"][e]["macro_f1"])
            gp.append(r["dev"][e]["macro_f1"] - r["test"][e]["macro_f1"])
            ep.append(e)
            p2.append(r["test"][select_epoch(r["dev"], "P2")]["macro_f1"])
        for k, v in (("dev_mf1_P1", dv), ("test_mf1_P1", tv), ("gap_P1", gp),
                     ("sel_epoch", ep), ("test_mf1_P2", p2)):
            q[(k, arm)] = np.asarray(v, dtype=np.float64)

    res = {"what": "R10-COMBO dev/epoch panel", "dataset": a.dataset,
           "arms": arms, "seeds": seeds, "n_boot": N_BOOT,
           "per_arm": {}, "contrasts": {}}
    for arm in arms:
        res["per_arm"][arm] = {k: {"mean": float(q[(k, arm)].mean()),
                                   "std": float(q[(k, arm)].std(ddof=1))}
                               for k in ("dev_mf1_P1", "test_mf1_P1", "gap_P1",
                                         "sel_epoch", "test_mf1_P2")}
    for k in ("dev_mf1_P1", "test_mf1_P1", "gap_P1", "test_mf1_P2"):
        for (L, R) in pairs:
            d = q[(k, L)] - q[(k, R)]
            lo, hi = ci(d, rng)
            res["contrasts"]["%s/%s-%s" % (k, L, R)] = {
                "mean": float(d.mean()), "ci95": [lo, hi],
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "n_positive": int((d > 0).sum()), "n_seeds": len(d)}

    json.dump(res, open(a.out, "w"), indent=1)
    print("%-6s %9s %9s %9s %9s %6s" % ("arm", "devF1", "testF1", "gap", "P2", "epoch"))
    for arm in arms:
        r = res["per_arm"][arm]
        print("%-6s %9.4f %9.4f %9.4f %9.4f %6.1f"
              % (arm, r["dev_mf1_P1"]["mean"], r["test_mf1_P1"]["mean"],
                 r["gap_P1"]["mean"], r["test_mf1_P2"]["mean"], r["sel_epoch"]["mean"]))
    print()
    for k, v in res["contrasts"].items():
        if k.startswith("dev_mf1_P1"):
            print("%-28s mean=%+.4f ci=[%+.4f,%+.4f] pos=%d/%d"
                  % (k, v["mean"], v["ci95"][0], v["ci95"][1],
                     v["n_positive"], v["n_seeds"]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
