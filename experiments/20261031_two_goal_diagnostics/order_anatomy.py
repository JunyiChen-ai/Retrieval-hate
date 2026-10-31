"""Goal 1 diagnostic: which questions each order asks, and where within-video ROC is gained or lost.

Reads the runs.pkl files written by policy_lab.py (test split) and the test GT. For each corpus, tag (answer model)
and order, 3-seed means of:
  - corpus shape: video length, GT harm fraction of positive videos;
  - asked nodes at calls 1-4 and 5-8: node length (seconds and fraction of the video), share asked in positive
    videos, share of positive-video nodes that contain GT harm, mean GT harm fraction inside the node;
  - within-video ROC (positive videos with both classes) at 0 / 4 / 8 calls, overall and by GT harm fraction tercile.
Rule-10 developmental evidence (test labels read; nothing trained or selected).
"""
import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import qtree
import hier_evidence_common as hc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+", default=["hatemm", "hateclipseg"])
    ap.add_argument("--tags", nargs="+", default=["r5", "r5_val3"])
    ap.add_argument("--root", default="runs/20261031_two_goal_diagnostics/policy_lab")
    a = ap.parse_args()
    report = {}
    for corpus in a.corpora:
        labels, ids, gt, _ = hc.load_fixed_cohort(corpus)
        test = ids["test"]
        pos = [v for v in test if labels[v] == 1]
        y = {v: np.asarray(gt["test"][v]).astype(bool) for v in test}
        frac = {v: float(y[v].mean()) for v in pos}
        within_vids = [v for v in pos if 0 < y[v].sum() < len(y[v])]
        terc = np.quantile([frac[v] for v in within_vids], [1 / 3, 2 / 3])
        group = {v: int(np.searchsorted(terc, frac[v], side="right")) for v in within_vids}
        rep = {"n_test": len(test), "n_pos": len(pos), "n_within": len(within_vids),
               "T_median": float(np.median([len(y[v]) for v in test])),
               "harm_fraction_median_pos": float(np.median([frac[v] for v in pos])),
               "harm_fraction_terciles": terc.tolist(), "tags": {}}
        for tag in a.tags:
            base = ROOT / a.root / corpus / tag
            if not base.exists():
                continue
            rep["tags"][tag] = {}
            for order in ("eig", "eer"):
                per_trial = []
                for d in sorted(base.glob(f"*_{order}")):
                    with (d / "runs.pkl").open("rb") as fh:
                        runs = pickle.load(fh)["runs"]["test"]
                    row = {}
                    for lo, hi, name in ((0, 4, "calls1_4"), (4, 8, "calls5_8")):
                        L, R, inpos, hit, hf = [], [], [], [], []
                        for v in test:
                            tr = qtree.tree(len(y[v]))
                            for n in runs[v]["asked"][lo:hi]:
                                s, e = int(tr["a"][n]), int(tr["b"][n])
                                L.append(e - s); R.append((e - s) / len(y[v])); inpos.append(labels[v] == 1)
                                if labels[v] == 1:
                                    hit.append(y[v][s:e].any()); hf.append(y[v][s:e].mean())
                        row[name] = {"len_s_median": float(np.median(L)), "len_frac_mean": float(np.mean(R)),
                                     "share_in_positive": float(np.mean(inpos)),
                                     "pos_node_has_harm": float(np.mean(hit)),
                                     "pos_node_harm_fraction": float(np.mean(hf))}
                    for k in (0, 4, 8):
                        auc = {v: roc_auc_score(y[v], runs[v]["scores"][min(k, len(runs[v]["scores"]) - 1)])
                               for v in within_vids}
                        row[f"within_k{k}"] = float(np.mean(list(auc.values())))
                        for g in range(3):
                            row[f"within_k{k}_tercile{g}"] = float(np.mean([auc[v] for v in within_vids
                                                                            if group[v] == g]))
                    per_trial.append(row)
                rep["tags"][tag][order] = {"n_trials": len(per_trial), "mean": _mean(per_trial)}
        report[corpus] = rep
    out = ROOT / a.root / "order_anatomy.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


def _mean(rows):
    out = {}
    for k, v in rows[0].items():
        out[k] = _mean([r[k] for r in rows]) if isinstance(v, dict) else float(np.mean([r[k] for r in rows]))
    return out


if __name__ == "__main__":
    main()
