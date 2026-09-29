"""Three-corpus tally of the no-floor stopping rules (README 17.4): for every rule present on all corpora, the
three-trial mean difference to fixed 8 of the same setting (AP / ROC / within) per corpus, PASS when every corpus
holds AP, ROC and within >= fixed 8 - tol. Rules recorded in several stop_check outputs of the same trials
(e.g. _r5e, _r5n1, _r5q) are merged per corpus.

    python experiments/20260925_query_paradigm/stop_tally.py --hatemm A.json B.json --hateclipseg ... --dehate ... [--tol .005]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

M = ("pooled_ap", "pooled_roc", "within_roc")


def load(files, B="8"):
    fixed, rules, calls = None, {}, {}
    for fp in files:
        d = json.load(open(fp))
        tr = d["trials"]
        f = np.mean([[t["fixed"][B]["test"][m] for m in M] for t in tr.values()], 0)
        fixed = f if fixed is None else fixed
        for r in list(tr.values())[0]["rules"]:
            xs = [t["rules"][r][B] for t in tr.values() if B in t["rules"].get(r, {})]
            if len(xs) != len(tr):
                continue
            rules[r] = np.mean([[x["test"][m] for m in M] for x in xs], 0)
            calls[r] = np.mean([[x["test_mean_calls"], x["test_mean_calls_pos"], x["test_mean_calls_neg"]] for x in xs], 0)
    return fixed, rules, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hatemm", nargs="+", required=True)
    ap.add_argument("--hateclipseg", nargs="+", required=True)
    ap.add_argument("--dehate", nargs="+", required=True)
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--budget", default="8")
    a = ap.parse_args()
    C = {"hatemm": load(a.hatemm, a.budget), "hateclipseg": load(a.hateclipseg, a.budget), "dehate": load(a.dehate, a.budget)}
    for c, (f, _, _) in C.items():
        print("%-11s fixed %s: %.3f/%.3f/%.3f" % (c, a.budget, *f))
    common = set.intersection(*[set(r) for _, r, _ in C.values()])
    rows = []
    for r in sorted(common):
        dd = {c: C[c][1][r] - C[c][0] for c in C}
        worst = min(min(dd[c]) for c in C)
        worst_w = min(dd[c][2] for c in C)
        ok = all((dd[c] >= -a.tol).all() for c in C)
        rows.append((ok, worst_w, worst, r, dd))
    rows.sort(key=lambda x: (-x[0], -x[1], -x[2]))
    print("rule        | HateMM d8 (calls)        | HCS d8 (calls)           | DeHate d8 (calls)        | min within | min all")
    for ok, ww, w, r, dd in rows:
        cells = []
        for c in C:
            cl = C[c][2][r]
            cells.append("%+.3f/%+.3f/%+.3f (%4.1f)" % (*dd[c], cl[0]))
        print("%-11s | %s | %s | %s | %+.4f | %+.4f %s" % (r, *cells, ww, w, "PASS" if ok else ""))


if __name__ == "__main__":
    main()
