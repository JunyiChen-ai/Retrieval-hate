"""Diagnostic (README section 1): does the VLM's soft answer on a node depend on how much of the node is hateful,
beyond whether it contains any hate (the OR state z_n that the revision-6 answer model conditions on)?

Reads the cached soft answers (data/vlm_tree/<C>/soft_both_p1*.jsonl, first-token P(Yes)) and the validation / test
per-second GT. Nothing is trained. Per node (queryable, >= 4 s): hate fraction f = mean GT over [a, b), length.
Prints, per split and length bin, mean P(Yes) and mean logit P(Yes) for: negative videos; positive videos with f = 0;
and f in (0, .25], (.25, .5], (.5, .75], (.75, 1); plus, among z_n = 1 nodes of a length bin, the Spearman correlation
of P(Yes) with f and the AUC of P(Yes) for f > .5 vs f <= .5.
"""
import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import hier_evidence_common as hc                                       # noqa: E402

DIRS = {"hatemm": "HateMM", "hateclipseg": "HateClipSeg", "dehate": "DeHate"}
LBINS = [(4, 8), (8, 16), (16, 32), (32, 64), (64, 10 ** 6)]
FBINS = [("neg video", None), ("f=0", (0, 0)), ("(0,.25]", (0, .25)), ("(.25,.5]", (.25, .5)),
         ("(.5,.75]", (.5, .75)), ("(.75,1)", (.75, .999999)), ("f=1", (1, 1))]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+", default=["hatemm", "hateclipseg", "dehate"])
    ap.add_argument("--out", default="runs/20261109_tree_within/diag/answer_fraction.json")
    a = ap.parse_args()
    report = {}
    for corpus in a.corpora:
        labels, ids, gt, _ = hc.load_fixed_cohort(corpus)
        want = {v: s for s in ("val", "test") for v in ids[s]}
        rows = {"val": [], "test": []}
        files = sorted(glob.glob(str(ROOT / "data/vlm_tree" / DIRS[corpus] / "soft_both_p1*.jsonl")))
        for fn in files:
            for line in open(fn):
                r = json.loads(line)
                s = want.get(r["id"])
                if s is None:
                    continue
                y = np.asarray(gt[s][r["id"]], dtype=float)
                neg = labels[r["id"]] == 0
                for nd in r["nodes"]:
                    a_, b_, probs = nd[0], nd[1], nd[2]
                    if b_ - a_ < 4 or b_ > len(y):
                        continue
                    rows[s].append((b_ - a_, -1.0 if neg else float(y[a_:b_].mean()), float(probs[0])))
        report[corpus] = {}
        for s, rr in rows.items():
            R = np.array(rr)
            L, F, P = R[:, 0], R[:, 1], R[:, 2]
            out = {}
            print(corpus, s, "nodes", len(R))
            for lo, hi in LBINS:
                inl = (L >= lo) & (L < hi)
                cells = {}
                for name, fb in FBINS:
                    if fb is None:
                        m = inl & (F < 0)
                    elif fb[0] == fb[1]:
                        m = inl & (F == fb[0])
                    else:
                        m = inl & (F > fb[0]) & (F <= fb[1])
                    if m.sum():
                        cells[name] = {"n": int(m.sum()), "p": float(P[m].mean()), "logit": float(logit(P[m]).mean())}
                z1 = inl & (F > 0)
                stat = {}
                if z1.sum() > 20:
                    stat["spearman_f"] = float(spearmanr(P[z1], F[z1]).correlation)
                    hi_f = F[z1] > .5
                    if 0 < hi_f.sum() < z1.sum():
                        stat["auc_f_gt_half"] = float(roc_auc_score(hi_f, P[z1]))
                out["%d-%d" % (lo, min(hi, 999))] = {"cells": cells, "z1": stat}
                print("  len %3d-%3d" % (lo, min(hi, 999)),
                      " ".join("%s %.2f(%d)" % (k, c["p"], c["n"]) for k, c in cells.items()),
                      "| z=1 spearman(p,f) %.2f auc(f>.5) %.2f" % (stat.get("spearman_f", np.nan),
                                                                   stat.get("auc_f_gt_half", np.nan)))
            report[corpus][s] = out
    o = ROOT / a.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(report, indent=1) + "\n")
    print("DONE", o)


if __name__ == "__main__":
    main()
