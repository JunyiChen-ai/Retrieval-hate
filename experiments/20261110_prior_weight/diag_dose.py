"""Diagnostic (README section 2): does the soft answer P(Yes) of a window grow with HOW MUCH of the window is hateful,
or only with WHETHER it contains hate (what the revision-6 answer model assumes: the answer depends on the node's
"any hateful second" state only)?

Developmental analysis on validation + test (rule 10): per-second GT is read to group the nodes, nothing is fitted
for the method. Per corpus, for every answered queryable node of the val / test videos: x = logit P(Yes) (clipped to
+-10), f = hateful fraction of the node's seconds, length class.
Reported per length class:
  mean x of the nodes of negative videos, of clean nodes (f = 0) of positive videos, and of the hateful nodes by
  fraction bin (0, .25], (.25, .5], (.5, .75], (.75, 1), 1;
  among the hateful nodes (f > 0): Spearman correlation of x with f, and the ROC-AUC of x for f >= .5 against f < .5;
  the ROC-AUC of x for hateful against clean nodes inside positive videos ("any").
Video level (roots of positive videos): Spearman of the root's x with the video's hateful fraction; the same for the
mean P(Yes) of the 8 equal windows (what the naive control paints).
Output runs/20261110_prior_weight/diag/dose_<corpus>.json.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import data as qdata, qtree                                    # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

CLASSES = [("4-8", 4, 8), ("8-16", 8, 16), ("16-32", 16, 32), ("32-64", 32, 64), ("64+", 64, 10 ** 9)]
BINS = [("(0,.25]", 0.0, 0.25), ("(.25,.5]", 0.25, 0.5), ("(.5,.75]", 0.5, 0.75), ("(.75,1)", 0.75, 0.999999)]


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    return np.clip(np.log(p) - np.log1p(-p), -10.0, 10.0)


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(y) and y.min() != y.max() else None


def rho(a, b):
    return float(spearmanr(a, b).correlation) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--out-root", default="runs/20261110_prior_weight/diag")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    P, T_ans, _ = qdata.load_soft_p(a.corpus, "soft_both")
    rows = []                                    # (split, positive video, length, f, x, is root)
    video = []                                   # positive videos: (split, fraction, root x, mean P of 8 windows)
    for split in ("val", "test"):
        for v in ids[split]:
            T = T_ans[v]
            y = np.zeros(T)
            g = np.asarray(gt[split][v])[:T]
            y[:len(g)] = g
            tr = qtree.tree(T)
            for n in np.where(tr["queryable"])[0]:
                p = P[v].get((int(tr["a"][n]), int(tr["b"][n])))
                if p is None:
                    continue
                f = float(y[tr["a"][n]:tr["b"][n]].mean())
                rows.append((split, int(labels[v]), int(tr["b"][n] - tr["a"][n]), f, float(logit(p)), int(n == 0)))
            if labels[v] == 1 and P[v].get((0, T)) is not None:
                w8 = [P[v].get((int(tr["a"][n]), int(tr["b"][n]))) for n in qtree.depth_nodes(T, 3)]
                w8 = [p for p in w8 if p is not None]
                video.append((split, float(y.mean()), float(logit(P[v][(0, T)])), float(np.mean(w8)) if w8 else None))
    R = np.array([(r[1], r[2], r[3], r[4], r[5]) for r in rows], dtype=np.float64)
    out = {"corpus": a.corpus, "host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(),
           "n_nodes": int(len(R)), "classes": {}}
    for name, lo, hi in CLASSES + [("root", None, None)]:
        m = (R[:, 4] == 1) if name == "root" else ((R[:, 1] >= lo) & (R[:, 1] < hi) & (R[:, 4] == 0))
        neg, pos = m & (R[:, 0] == 0), m & (R[:, 0] == 1)
        clean, hatef = pos & (R[:, 2] == 0), pos & (R[:, 2] > 0)
        c = {"n_negative_video": int(neg.sum()), "x_negative_video": float(R[neg, 3].mean()) if neg.any() else None,
             "n_clean": int(clean.sum()), "x_clean": float(R[clean, 3].mean()) if clean.any() else None,
             "n_hateful": int(hatef.sum()), "bins": {}}
        for bname, blo, bhi in BINS:
            k = pos & (R[:, 2] > blo) & (R[:, 2] <= bhi)
            c["bins"][bname] = {"n": int(k.sum()), "x": float(R[k, 3].mean()) if k.any() else None}
        k = pos & (R[:, 2] == 1.0)
        c["bins"]["1"] = {"n": int(k.sum()), "x": float(R[k, 3].mean()) if k.any() else None}
        c["spearman_x_f_hateful"] = rho(R[hatef, 3], R[hatef, 2])
        c["auc_x_f_ge_half_hateful"] = auc(R[hatef, 2] >= 0.5, R[hatef, 3])
        c["auc_x_any_in_positive"] = auc(R[pos, 2] > 0, R[pos, 3])
        c["auc_x_hateful_vs_negative_video"] = auc(np.r_[np.ones(int(hatef.sum())), np.zeros(int(neg.sum()))],
                                                   np.r_[R[hatef, 3], R[neg, 3]])
        out["classes"][name] = c
        print(a.corpus, name, json.dumps({k_: (round(v_, 3) if isinstance(v_, float) else v_)
                                          for k_, v_ in c.items() if k_ != "bins"}),
              " ".join("%s:%s(%d)" % (b_, "%.2f" % d_["x"] if d_["x"] is not None else "-", d_["n"])
                       for b_, d_ in c["bins"].items()), flush=True)
    for split in ("val", "test", "both"):
        vv = [r for r in video if split in ("both", r[0])]
        fr = np.array([r[1] for r in vv])
        out.setdefault("video", {})[split] = {
            "n_positive": len(vv),
            "spearman_root_x_fraction": rho(np.array([r[2] for r in vv]), fr),
            "spearman_mean8_fraction": rho(np.array([r[3] for r in vv if r[3] is not None]),
                                           np.array([r[1] for r in vv if r[3] is not None]))}
        print(a.corpus, "video", split, json.dumps(out["video"][split]), flush=True)
    od = (ROOT / a.out_root).resolve()
    od.mkdir(parents=True, exist_ok=True)
    (od / f"dose_{a.corpus}.json").write_text(json.dumps(out, indent=1) + "\n")
    print("DONE", a.corpus, flush=True)


if __name__ == "__main__":
    main()
