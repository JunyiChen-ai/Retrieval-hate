"""Diagnostic (README section 10): do the No / Some / All answers (prompt p3, data/vlm_tree/<C>/ext_both_p3*.jsonl)
carry the information the extent question needs?

Validation and test nodes (queryable, >= 4 s) with both the extent answer and the yes / no answer
(soft_both_p1*, first-token P(Yes)). Nothing is trained. Node state: negative video / none / some / all
(extent.node_extent on the per-second GT). Per split:
1. AUC of presence 1 - P(No) and of P(Yes) for a hateful node (some or all) against every other node;
2. among hateful nodes of a length bin, AUC for all vs some of fullness P(All) / (P(Some) + P(All)), of P(Yes) and of
   presence, and the Pearson correlation of fullness with the node's hate fraction;
3. inside positive videos (val + test pooled), per video and length bin with both classes, AUC of presence and of
   P(Yes) for none vs hateful nodes, averaged over videos.
"""
import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import extent, qtree                                           # noqa: E402
from qtl.data import CORPUS_DIR                                         # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

LBINS = [(4, 8), (8, 16), (16, 32), (32, 64), (64, 10 ** 6)]


def load_yes(corpus):
    yes = {}
    for path in glob.glob(str(ROOT / "data/vlm_tree" / CORPUS_DIR[corpus] / "soft_both_p1*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            yes[r["id"]] = {(int(a), int(b)): (None if o is None else float(o[0])) for a, b, o, _ in r["nodes"]}
    return yes


def node_rows(raw, yes, labels, ids, gt):
    """Per node: state (0 negative video, 1 none, 2 some, 3 all), length, presence, fullness, P(Yes), hate fraction."""
    rows = {}
    for sp in ("val", "test"):
        R = []
        for v in ids[sp]:
            if v not in raw or (labels[v] == 1 and v not in gt[sp]):
                continue
            y = np.asarray(gt[sp][v]).astype(int) if labels[v] == 1 else None
            for (a, b), o in raw[v].items():
                py = yes.get(v, {}).get((a, b))
                if o is None or py is None or b - a < qtree.F_FRAMES or (y is not None and b > len(y)):
                    continue
                s = 0 if y is None else 1 + extent.node_extent(y, a, b)
                R.append((s, b - a, o[0], o[1], py, 0.0 if y is None else float(y[a:b].mean())))
        rows[sp] = np.array(R)
    return rows


def split_report(R):
    hate = R[:, 0] >= 2
    res = {"n_nodes_neg_none_some_all": [int((R[:, 0] == s).sum()) for s in range(4)],
           "auc_hate_vs_rest": {"presence": float(roc_auc_score(hate, R[:, 2])),
                                "p_yes": float(roc_auc_score(hate, R[:, 4]))},
           "mean_presence_by_state": [float(R[R[:, 0] == s, 2].mean()) for s in range(4)],
           "mean_fullness_by_state": [float(R[R[:, 0] == s, 3].mean()) for s in range(4)],
           "all_vs_some": {}}
    for lo, hi in LBINS:
        k = hate & (R[:, 1] >= lo) & (R[:, 1] < hi)
        if k.sum() > 10 and len(np.unique(R[k, 0])) == 2:
            full = R[k, 0] == 3
            res["all_vs_some"][f"{lo}-{hi}"] = {
                "n_all": int(full.sum()), "n_some": int((~full).sum()),
                "auc_fullness": float(roc_auc_score(full, R[k, 3])),
                "auc_p_yes": float(roc_auc_score(full, R[k, 4])),
                "auc_presence": float(roc_auc_score(full, R[k, 2])),
                "corr_fullness_fraction": float(np.corrcoef(R[k, 3], R[k, 5])[0, 1])}
    return res


def within_video(raw, yes, labels, ids, gt):
    out = {}
    for lo, hi in LBINS[:4]:
        ap_pres, ap_yes = [], []
        for sp in ("val", "test"):
            for v in ids[sp]:
                if v not in raw or labels[v] != 1 or v not in gt[sp]:
                    continue
                y = np.asarray(gt[sp][v]).astype(int)
                R = [(y[a:b].any(), o[0], yes[v][(a, b)]) for (a, b), o in raw[v].items()
                     if o is not None and yes.get(v, {}).get((a, b)) is not None and lo <= b - a < hi and b <= len(y)]
                R = np.array(R, dtype=float)
                if len(R) < 3 or len(np.unique(R[:, 0])) < 2:
                    continue
                ap_pres.append(roc_auc_score(R[:, 0], R[:, 1]))
                ap_yes.append(roc_auc_score(R[:, 0], R[:, 2]))
        if ap_pres:
            out[f"{lo}-{hi}"] = {"videos": len(ap_pres), "auc_presence": float(np.mean(ap_pres)),
                                 "auc_p_yes": float(np.mean(ap_yes))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+", default=["hatemm", "hateclipseg", "dehate"])
    ap.add_argument("--out", default="runs/20261109_tree_within/diag/extent_answers.json")
    a = ap.parse_args()
    report = {}
    for c in a.corpora:
        raw, _T, _split = extent.load_extent(c)
        yes = load_yes(c)
        labels, ids, gt, _ = hc.load_fixed_cohort(c)
        rows = node_rows(raw, yes, labels, ids, gt)
        report[c] = {sp: split_report(rows[sp]) for sp in ("val", "test")}
        report[c]["videos_with_extent"] = {sp: sum(v in raw for v in ids[sp]) for sp in ("val", "test")}
        report[c]["videos_in_split"] = {sp: len(ids[sp]) for sp in ("val", "test")}
        report[c]["within_video_none_vs_hate"] = within_video(raw, yes, labels, ids, gt)
        print(c, json.dumps(report[c]), flush=True)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
