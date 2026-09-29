"""Equal-calls comparison of a stopping rule with the fixed-budget curve (README 17.4): for each calibrated budget of
the rule, the fixed curve (test mean calls -> AP / ROC / within of fixed 0, 2, 4, 8, 12, 16, 24, 32) is interpolated
linearly at the rule's test mean calls and the difference reported; three-trial means. Label-free reference: the
fixed line at the same number of calls (the "mean 8 vs fixed 8" criterion compares 7.1-7.4 rule calls with the
7.6 calls of fixed 8 on corpora with short videos).

    python experiments/20260925_query_paradigm/stop_curve.py --rules qmixSG qmix2G FILE [FILE ...]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

M = ("pooled_ap", "pooled_roc", "within_roc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--rules", nargs="+", default=["qmixSG"])
    a = ap.parse_args()
    for fp in a.files:
        d = json.load(open(fp))
        tr = list(d["trials"].values())
        Bs = sorted({int(b) for t in tr for b in t["fixed"]})
        calls = np.array([np.mean([t["fixed"][str(B)]["test_mean_calls"] for t in tr]) for B in Bs])
        fx = np.array([[np.mean([t["fixed"][str(B)]["test"][m] for t in tr]) for m in M] for B in Bs])
        print("== %s (%s): fixed calls %s" % (fp, d["corpus"], np.round(calls, 2)))
        for r in a.rules:
            if r not in tr[0]["rules"]:
                continue
            for B in sorted(tr[0]["rules"][r], key=int):
                xs = [t["rules"][r][B] for t in tr if B in t["rules"].get(r, {})]
                c = np.mean([x["test_mean_calls"] for x in xs])
                m = np.mean([[x["test"][k] for k in M] for x in xs], 0)
                ref = np.array([np.interp(c, calls, fx[:, j]) for j in range(3)])
                print("  %-11s m%-2s calls %5.2f  rule %.3f/%.3f/%.3f  fixed@same %.3f/%.3f/%.3f  d %+.3f/%+.3f/%+.3f" % (
                    r, B, c, *m, *ref, *(m - ref)))


if __name__ == "__main__":
    main()
