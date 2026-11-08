"""Diagnostic (README section 11): how much does the revision-6 answer table trust an answer, against how well the
answers actually separate hateful from clean nodes inside hateful videos?

Revision-6 test-selected trials, cached yes / no answers (soft_both, 8 training-quantile levels), validation + test
nodes (queryable, >= 4 s). Per length bin (table read at the bin's median length, mean over the trials):
- table AUC: P(o | hateful node) against P(o | no hate) of the anchored table, AUC = P(o1 > o0) + P(o1 = o0) / 2;
- data AUC of the answer level for hateful nodes against (a) nodes of negative videos, (b) clean nodes of hateful
  videos (pooled), (c) per hateful video with both kinds of node, averaged over videos;
- mean answer level of the three node kinds.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import data as qdata, qtree                                   # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

LBINS = [(4, 8), (8, 16), (16, 32), (32, 64)]


def table_auc(p1, p0):
    c0 = np.concatenate([[0.0], np.cumsum(p0)[:-1]])
    return float(np.sum(p1 * (c0 + 0.5 * p0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+", default=["hatemm", "hateclipseg", "dehate"])
    ap.add_argument("--out", default="runs/20261109_tree_within/diag/table_trust.json")
    a = ap.parse_args()
    qdata.configure_source("soft_both", 8)
    report = {}
    for c in a.corpora:
        summ = json.load(open(ROOT / f"runs/20261101_r6_length_table/final_curves/test_selected/{c}/summary.json"))
        answers, _ = qdata.load_answers(c, "soft_both")
        labels, ids, gt, _ = hc.load_fixed_cohort(c)
        loglen = np.log([b - a_ for v in ids["train"] for (a_, b) in answers[v]])
        tabs = []
        for r in summ["trials"].values():
            ck = torch.load(Path(r["trial"]) / "model.pth", map_location="cpu", weights_only=False)
            tabs.append((ck["am"]["theta"].numpy(), ck["am"]["omega"].numpy()))
        assert tabs[0][0].shape[1] == 2, "expects the two-state anchored table"
        vid, R = [], []
        for sp in ("val", "test"):
            for v in ids[sp]:
                if labels[v] == 1 and v not in gt[sp]:
                    continue
                y = np.asarray(gt[sp][v]).astype(int) if labels[v] == 1 else None
                for (a_, b), o in answers[v].items():
                    if o is None or b - a_ < qtree.F_FRAMES or (y is not None and b > len(y)):
                        continue
                    vid.append(v)
                    R.append((0 if y is None else 1 + int(y[a_:b].any()), b - a_, int(o[0])))
        vid, R = np.array(vid), np.array(R)
        res = {}
        for lo, hi in LBINS:
            k = (R[:, 1] >= lo) & (R[:, 1] < hi)
            L = float(np.median(R[k, 1]))
            u = (np.log(L) - loglen.mean()) / loglen.std()
            ta = []
            for th, om in tabs:
                lg = th[0] + om[0] * u                                  # S, L
                p = np.exp(lg - lg.max(1, keepdims=True))
                p /= p.sum(1, keepdims=True)
                ta.append(table_auc(p[1], p[0]))
            h, neg, clean = k & (R[:, 0] == 2), k & (R[:, 0] == 0), k & (R[:, 0] == 1)
            auc = lambda m: float(roc_auc_score(np.r_[np.ones(h.sum()), np.zeros(m.sum())], np.r_[R[h, 2], R[m, 2]]))
            wv = []
            for v in np.unique(vid[k & (R[:, 0] >= 1)]):
                m = k & (vid == v) & (R[:, 0] >= 1)
                if m.sum() >= 3 and len(np.unique(R[m, 0])) == 2:
                    wv.append(roc_auc_score(R[m, 0] == 2, R[m, 2]))
            res[f"{lo}-{hi}"] = {"median_length": L, "table_auc": float(np.mean(ta)),
                                 "data_auc_vs_negative_videos": auc(neg), "data_auc_vs_clean_nodes": auc(clean),
                                 "data_auc_within_video": float(np.mean(wv)), "videos_within": len(wv),
                                 "mean_level_negative_clean_hateful": [float(R[m, 2].mean()) for m in (neg, clean, h)]}
        report[c] = res
        print(c, json.dumps(res), flush=True)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
