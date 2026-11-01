"""Monotonicity acceptance check (README section 1): fixed k on the grid 0, 1, 2, 4, 8, 16, 32 and the qmixE1 budgets
1-8 (per-video cap 8), 3-trial means of AP / ROC / within; every step down to the next grid point at most TOL lower,
and 16 and 32 calls not lower than 8 calls. Reads summary.json files written by sim_monotone.py / offset_check.py,
or with the source "final" the <summary root>/<corpus>/summary.json of final_curves.py (budgets under "rule").

    python accept_check.py <summary root> <corpus> <source> [<source> ...]
"""
import glob
import json
import sys

import numpy as np

TOL = 0.005
GRID = ("0", "1", "2", "4", "8", "16", "32")
METS = ("pooled_ap", "pooled_roc", "within_roc")


def curve(rows, key, points, m):
    return [float(np.mean([r[key][p][m] for r in rows])) for p in points]


def check(rows):
    fails = []
    for m in METS:
        f = curve(rows, "fixed", GRID, m)
        for i in range(1, len(GRID)):
            if f[i] < f[i - 1] - TOL:
                fails.append("%s fixed %s->%s %.3f->%.3f" % (m, GRID[i - 1], GRID[i], f[i - 1], f[i]))
        for k in ("16", "32"):
            if f[GRID.index(k)] < f[GRID.index("8")]:
                fails.append("%s fixed %s < 8 (%.3f < %.3f)" % (m, k, f[GRID.index(k)], f[GRID.index("8")]))
        bs = [b for b in ("1", "2", "3", "4", "5", "6", "7", "8") if b in rows[0]["rule_cap8"]]
        r = curve(rows, "rule_cap8", bs, m)
        for i in range(1, len(bs)):
            if r[i] < max(r[:i]) - TOL:
                fails.append("%s budget %s %.3f (max before %.3f)" % (m, bs[i], r[i], max(r[:i])))
    return fails


def main():
    root, corpus, sources = sys.argv[1], sys.argv[2], sys.argv[3:]
    files = sorted(glob.glob("%s/%s/*/summary.json" % (root, corpus)))
    for src in sources:
        if src == "final":
            trials = json.load(open("%s/%s/summary.json" % (root, corpus)))["trials"].values()
            rows = [{"fixed": t["fixed"], "rule_cap8": t["rule"]} for t in trials]
        else:
            rows = [json.load(open(f))["sources"][src]["mean"] for f in files
                    if src in json.load(open(f))["sources"]]
        if not rows:
            print(corpus, src, "missing")
            continue
        fails = check(rows)
        f8 = [curve(rows, "fixed", ("8",), m)[0] for m in METS]
        b5 = [curve(rows, "rule_cap8", ("5",), m)[0] for m in METS]
        print("%s %-15s n=%d fixed8 %.3f/%.3f/%.3f budget5 %.3f/%.3f/%.3f  %s" % (
            corpus, src, len(rows), *f8, *b5, "PASS" if not fails else "FAIL: " + "; ".join(fails)))


if __name__ == "__main__":
    main()
