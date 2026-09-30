"""Summary of stop_check.py --select-val outputs (README 17.4 item 11): per corpus and rule, for each validation
objective (AP / ROC / within / sum; exact maximum and smallest budget within .005 of it) the selected validation
budget, the test calls and the three-trial mean test metrics with the difference to fixed 8 of the same setting.

    python experiments/20260925_query_paradigm/stop_select_summary.py FILE [FILE ...]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

M = ("pooled_ap", "pooled_roc", "within_roc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    for fp in a.files:
        d = json.load(open(fp))
        tr = list(d["trials"].values())
        f8 = np.mean([[t["fixed"]["8"]["test"][m] for m in M] for t in tr], 0)
        c8 = np.mean([t["fixed"]["8"]["test_mean_calls"] for t in tr])
        print("== %s (%s, %d trials) fixed 8 (%.2f calls) %.3f/%.3f/%.3f" % (fp, d["corpus"], len(tr), c8, *f8))
        for rule in tr[0].get("select", {}):
            for name in tr[0]["select"][rule]["selected"]:
                xs = [t["select"][rule]["selected"][name] for t in tr]
                m = np.mean([[x["test"][k] for k in M] for x in xs], 0)
                B = [x["B"] for x in xs]
                c = np.mean([[x["test_mean_calls"], x["test_mean_calls_pos"], x["test_mean_calls_neg"]] for x in xs], 0)
                d8 = m - f8
                ok = all(x >= -0.005 for x in d8)
                print("  %-11s %-15s val B %s calls %5.2f (pos %4.1f neg %4.1f) %.3f/%.3f/%.3f d8 %+.3f/%+.3f/%+.3f %s" % (
                    rule, name, B, c[0], c[1], c[2], *m, *d8, "PASS" if ok else ""))


if __name__ == "__main__":
    main()
