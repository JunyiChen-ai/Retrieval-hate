"""User-requested controls (README section 12), default hyperparameters, no search, seeds 234 / 2025 / 3407:
abl_full (the method), abl_noback (no content backbone: constant prior), abl_lvl{4,8,16} (questions only from one
tree depth = equal non-overlapping windows of L to 2L seconds, in training and at test).

    python experiments/20260925_query_paradigm/summarize_controls.py [--root R] [--corpus c ...]
Reads <R>/<corpus>/seed<s>/abl_*/summary.json (evaluator-written test metrics; default R =
runs/20260925_query_paradigm_diag, DeHate: runs/20260926_dehate_external/diag); writes <R>/controls_summary.json and
prints 3-seed means.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
D = os.path.join(ROOT, "runs", "20260925_query_paradigm_diag")
SEEDS = (234, 2025, 3407)
TAGS = ("abl_full", "abl_noback", "abl_lvl4", "abl_lvl8", "abl_lvl16")
BUDGETS = ("0", "2", "4", "8", "16", "32")
KEYS = ("pooled_ap", "pooled_roc", "within_roc")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=D)
    ap.add_argument("--corpus", nargs="+", default=["hatemm", "hateclipseg"])
    a = ap.parse_args()
    out = {}
    for c in a.corpus:
        out[c] = {}
        print("== %s" % c)
        for tag in TAGS:
            runs = []
            for s in SEEDS:
                p = os.path.join(a.root, c, "seed%d" % s, tag, "summary.json")
                if os.path.exists(p):
                    runs.append(json.load(open(p)))
            if not runs:
                continue
            row = {"n_seeds": len(runs)}
            for B in BUDGETS:
                vals = [r["fixed"][B] for r in runs if B in r["fixed"]]
                row[B] = {k: float(np.mean([v["test"][k] for v in vals])) for k in KEYS}
                row[B]["mean_calls"] = float(np.mean([v["test_mean_calls"] for v in vals]))
                row[B]["std_ap"] = float(np.std([v["test"]["pooled_ap"] for v in vals]))
            out[c][tag] = row
            print("  %-11s (%d seeds) " % (tag, len(runs)) + " | ".join(
                "%s: %.4f/%.4f w%.3f c%.2f" % (B, row[B]["pooled_ap"], row[B]["pooled_roc"], row[B]["within_roc"],
                                                row[B]["mean_calls"]) for B in BUDGETS))
    json.dump(out, open(os.path.join(a.root, "controls_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
