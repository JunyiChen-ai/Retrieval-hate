"""How the cached VLM answers depend on node length and on the true state (README sections 10.4 and 13; DeHate:
experiments/20260926_dehate_external/README.md). Development analysis on test GT (rule 10).

For node-length bins [4,8), [8,16), ... and five groups of answered nodes
    train_neg        all nodes of negative training videos          (the anchored table's harmless row)
    train_pos_root   the root of positive training videos           (the anchored table's harmful row)
    test_harm        test nodes that contain a GT-harmful second
    test_pos_nonharm test nodes of positive videos without a GT-harmful second
    test_neg         nodes of negative test videos
it reports the rate of "some category >= 2" (yes) and of the all-zero answer (zero), with counts; bins with < 15
nodes are left out. The within-video separation logit(yes | test_harm) - logit(yes | test_pos_nonharm) is the
information a short-node answer carries about where the harm is.

    python experiments/20260925_query_paradigm/answer_rates.py --corpus hatemm hateclipseg dehate
Writes runs/20260925_query_paradigm/answer_rates/<corpus>.json and prints the tables.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import data as qdata               # noqa: E402  (sets the shared import paths)
import hier_evidence_common as hc  # noqa: E402

BINS = [4, 8, 16, 32, 64, 128, 256, 100000]
MIN_N = 15


def lg(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def rates(rows):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        r = [(y, z) for l, y, z in rows if lo <= l < hi]
        out.append({"lo": lo, "n": len(r), "yes": float(np.mean([y for y, _ in r])) if len(r) >= MIN_N else None,
                    "zero": float(np.mean([z for _, z in r])) if len(r) >= MIN_N else None})
    allr = {"n": len(rows), "yes": float(np.mean([y for _, y, _ in rows])) if rows else None,
            "zero": float(np.mean([z for _, _, z in rows])) if rows else None}
    return out, allr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "20260925_query_paradigm", "answer_rates"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for corpus in a.corpus:
        labels, ids, gt, _ = hc.load_fixed_cohort(corpus)
        ans, Ta = qdata.load_answers(corpus)
        row = lambda a_, b_, o: (b_ - a_, max(o) >= 2, max(o) == 0)  # noqa: E731
        groups = {"train_neg": [], "train_pos_root": [], "test_harm": [], "test_pos_nonharm": [], "test_neg": []}
        for v in ids["train"]:
            for (x, z), o in ans.get(v, {}).items():
                if o is None:
                    continue
                if labels[v] == 0:
                    groups["train_neg"].append(row(x, z, o))
                elif x == 0 and z == Ta[v]:
                    groups["train_pos_root"].append(row(x, z, o))
        for v in ids["test"]:
            G = np.asarray(gt["test"][v])
            for (x, z), o in ans.get(v, {}).items():
                if o is None:
                    continue
                if labels[v] == 0:
                    groups["test_neg"].append(row(x, z, o))
                elif G[x:z].max() > 0:
                    groups["test_harm"].append(row(x, z, o))
                else:
                    groups["test_pos_nonharm"].append(row(x, z, o))
        res = {"corpus": corpus, "bins": BINS[:-1], "groups": {}}
        print("== %s (yes = some category >= 2; zero = all five 0)" % corpus)
        for g, rows in groups.items():
            per, allr = rates(rows)
            res["groups"][g] = {"per_bin": per, "all": allr}
            print("  %-17s yes  %s | all %.3f (n %d)" % (g, " ".join("%5.3f" % r["yes"] if r["yes"] is not None
                                                                   else "  -  " for r in per), allr["yes"], allr["n"]))
            print("  %-17s zero %s | all %.3f" % ("", " ".join("%5.3f" % r["zero"] if r["zero"] is not None
                                                             else "  -  " for r in per), allr["zero"]))
        sep = []
        for h, p in zip(res["groups"]["test_harm"]["per_bin"], res["groups"]["test_pos_nonharm"]["per_bin"]):
            sep.append(lg(h["yes"]) - lg(p["yes"]) if h["yes"] is not None and p["yes"] is not None else None)
        res["within_separation_logit"] = sep
        print("  within separation logit(yes|harm) - logit(yes|pos non-harm): %s" % " ".join(
            "%5.2f" % s if s is not None else "  -  " for s in sep))
        json.dump(res, open(os.path.join(a.out, "%s.json" % corpus), "w"), indent=1)


if __name__ == "__main__":
    main()
