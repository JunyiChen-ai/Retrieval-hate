"""Diagnostic (README section 6): does the full tree lose within where the backbone's prior is wrong about where the
hate is? Revision-6 test-selected trials, E3 replays and per-video AUCs (experiments/20261108_astra_review), test GT;
nothing re-run. Per positive test video with both classes and level L: prior within AUC (scores before any call),
tree and single-level within AUC at the E3 matched calls, their difference; mean difference by tercile of the prior
within AUC. Also per video the mean prior P(y_t = 1) on the seconds the tree / the level covers with a node of length
<= 2L + 1 s ("fine coverage") versus the rest.
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
from qtl import qtree                                                   # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

E3 = ROOT / "runs/20261108_astra_review/e3"
REPLAY = ROOT / "runs/20261108_astra_review/replay"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+", default=["hatemm", "hateclipseg", "dehate"])
    ap.add_argument("--levels", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--out", default="runs/20261109_tree_within/diag/prior_trust.json")
    a = ap.parse_args()
    out = {}
    for corpus in a.corpora:
        _labels, _ids, gt, _ = hc.load_fixed_cohort(corpus)
        gt = gt["test"] if isinstance(gt, dict) and "test" in gt else gt
        trials = sorted(p.name for p in (E3 / corpus).iterdir() if p.is_dir())
        out[corpus] = {}
        for L in a.levels:
            rows = []
            for trial in trials:
                full = pickle.load(open(REPLAY / corpus / trial / "runs_L0.pkl", "rb"))["runs"]["test"]
                lev = pickle.load(open(REPLAY / corpus / trial / f"runs_L{L}.pkl", "rb"))["runs"]["test"]
                af = json.load(open(E3 / corpus / trial / f"metrics_test_full_matchL{L}.json"))["results"]["score_av"]
                al = json.load(open(E3 / corpus / trial / f"metrics_test_level{L}.json"))["results"]["score_av"]
                af, al = af["per_video"]["per_video_auc"], al["per_video"]["per_video_auc"]
                for v, y in gt.items():
                    if v not in af or v not in al or v not in full:
                        continue
                    y = np.asarray(y).astype(int)
                    T = len(y)
                    p0 = np.asarray(full[v]["scores"][0])[:T]
                    tr = qtree.tree(T)
                    k = min(8, len(lev[v]["eig"]))

                    def fine(asked):
                        res = np.full(T, float(T))
                        for n in asked:
                            res[tr["a"][n]:tr["b"][n]] = np.minimum(res[tr["a"][n]:tr["b"][n]], tr["b"][n] - tr["a"][n])
                        return res <= 2 * L + 1
                    ft = fine([int(n) for n in full[v]["asked"][:k]])
                    fl = fine([int(n) for n in lev[v]["asked"][:k]])
                    rows.append({"prior_auc": float(roc_auc_score(y, p0)), "diff": af[v] - al[v],
                                 "auc_tree": af[v], "auc_level": al[v],
                                 "frac": float(y.mean()),
                                 "prior_err_tree_fine": float(np.abs(p0 - y)[ft].mean()) if ft.any() else np.nan,
                                 "prior_err_tree_rest": float(np.abs(p0 - y)[~ft].mean()) if (~ft).any() else np.nan,
                                 "cov_tree": float(ft.mean()), "cov_level": float(fl.mean())})
            pa = np.array([r["prior_auc"] for r in rows])
            d = np.array([r["diff"] for r in rows])
            q = np.quantile(pa, [1 / 3, 2 / 3])
            terc = [d[pa <= q[0]], d[(pa > q[0]) & (pa <= q[1])], d[pa > q[1]]]
            res = {"n_rows": len(rows), "prior_auc_mean": float(pa.mean()), "diff_mean": float(d.mean()),
                   "prior_auc_tercile_edges": q.tolist(),
                   "diff_by_prior_auc_tercile": [float(t.mean()) for t in terc],
                   "tree_minus_prior_by_tercile": [float(np.mean([r["auc_tree"] - r["prior_auc"] for r, x in zip(rows, pa)
                                                                  if lo < x <= hi])) for lo, hi in
                                                   ((-1, q[0]), (q[0], q[1]), (q[1], 2))],
                   "level_minus_prior_by_tercile": [float(np.mean([r["auc_level"] - r["prior_auc"] for r, x in zip(rows, pa)
                                                                   if lo < x <= hi])) for lo, hi in
                                                    ((-1, q[0]), (q[0], q[1]), (q[1], 2))],
                   "frac_by_tercile": [float(np.mean([r["frac"] for r, x in zip(rows, pa) if lo < x <= hi]))
                                       for lo, hi in ((-1, q[0]), (q[0], q[1]), (q[1], 2))],
                   "corr_prior_auc_diff": float(np.corrcoef(pa, d)[0, 1]),
                   "prior_err_tree_fine": float(np.nanmean([r["prior_err_tree_fine"] for r in rows])),
                   "prior_err_tree_rest": float(np.nanmean([r["prior_err_tree_rest"] for r in rows])),
                   "cov_tree": float(np.mean([r["cov_tree"] for r in rows])),
                   "cov_level": float(np.mean([r["cov_level"] for r in rows]))}
            out[corpus][str(L)] = res
            print(corpus, L, json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, (list, float)) else v)
                                         for k, v in res.items()}), flush=True)
    Path(ROOT / a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
