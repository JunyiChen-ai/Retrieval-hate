"""Diagnostic (README section 1): why does one tree level beat the full tree on within in E3?

Reads the E3 inputs and outputs of experiments/20261108_astra_review (replays runs_L<level>.pkl of the revision-6
test-selected trials, and the unified evaluator's per-video AUCs metrics_test_{full_matchL<L>,level<L>}.json) and the
test GT. Nothing is trained or re-scored. Per positive test video with both classes and per level L:
  the full tree's first k_v asked nodes (k_v = the E3 matched calls): their lengths, how many are nested with an
  earlier asked node, and the per-second resolution = length of the shortest asked node covering the second
  (T when only the root covers it); the single level's resolution is its window length;
  GT: hate fraction, number of hate segments, video length;
  per-video within AUC of both (evaluator output) and their difference.
Prints, per corpus and level, means over videos and trials, and the mean difference by tercile of each covariate.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import qtree                                                   # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

E3 = ROOT / "runs/20261108_astra_review/e3"
REPLAY = ROOT / "runs/20261108_astra_review/replay"


def segments(y):
    y = np.asarray(y).astype(int)
    return int(((y[1:] == 1) & (y[:-1] == 0)).sum() + (y[0] == 1))


def video_rows(corpus, trial, L, gt, cap=8):
    full = pickle.load(open(REPLAY / corpus / trial / "runs_L0.pkl", "rb"))["runs"]["test"]
    lev = pickle.load(open(REPLAY / corpus / trial / f"runs_L{L}.pkl", "rb"))["runs"]["test"]
    pf = json.load(open(E3 / corpus / trial / f"metrics_test_full_matchL{L}.json"))["results"]["score_av"]
    pl = json.load(open(E3 / corpus / trial / f"metrics_test_level{L}.json"))["results"]["score_av"]
    af, al = pf["per_video"]["per_video_auc"], pl["per_video"]["per_video_auc"]
    rows = []
    for v, y in gt.items():
        if v not in af or v not in al:
            continue
        T = len(y)
        tr = qtree.tree(T)
        k = min(cap, len(lev[v]["eig"]))
        asked = [int(n) for n in full[v]["asked"][:min(k, len(full[v]["eig"]))]]
        lens = np.array([tr["b"][n] - tr["a"][n] for n in asked], dtype=float)
        nested = 0
        for i, n in enumerate(asked):
            for m in asked[:i]:
                if (tr["a"][m] <= tr["a"][n] and tr["b"][n] <= tr["b"][m]) or \
                   (tr["a"][n] <= tr["a"][m] and tr["b"][m] <= tr["b"][n]):
                    nested += 1
                    break
        res = np.full(T, float(T))
        for n in asked:
            a_, b_ = tr["a"][n], tr["b"][n]
            res[a_:b_] = np.minimum(res[a_:b_], b_ - a_)
        lev_nodes = [int(n) for n in lev[v]["asked"][:k]]
        lres = np.full(T, float(T))
        for n in lev_nodes:
            a_, b_ = tr["a"][n], tr["b"][n]
            lres[a_:b_] = np.minimum(lres[a_:b_], b_ - a_)
        rows.append({"v": v, "T": T, "k": k, "frac": float(np.mean(y)), "segs": segments(y),
                     "len_mean": float(lens.mean()) if len(lens) else np.nan,
                     "len_min": float(lens.min()) if len(lens) else np.nan,
                     "nested": nested, "res_tree": float(np.mean(res)), "res_level": float(np.mean(lres)),
                     "cov_tree": float(np.mean(res <= 2 * L + 1)), "cov_level": float(np.mean(lres <= 2 * L + 1)),
                     "auc_tree": af[v], "auc_level": al[v], "diff": af[v] - al[v]})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+", default=["hatemm", "hateclipseg", "dehate"])
    ap.add_argument("--levels", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--out", default="runs/20261109_tree_within/diag/diag_tree_within.json")
    a = ap.parse_args()
    report = {}
    for corpus in a.corpora:
        _, _, gt, _ = hc.load_fixed_cohort(corpus)
        gt = {v: np.asarray(y) for v, y in gt["test"].items()}
        trials = sorted(p.name for p in (E3 / corpus).iterdir() if p.is_dir())
        report[corpus] = {}
        for L in a.levels:
            rows = [dict(r, trial=t) for t in trials for r in video_rows(corpus, t, L, gt)]
            keys = ["T", "k", "frac", "segs", "len_mean", "len_min", "nested", "res_tree", "res_level", "cov_tree",
                    "cov_level", "auc_tree", "auc_level", "diff"]
            mean = {kk: float(np.nanmean([r[kk] for r in rows])) for kk in keys}
            terc = {}
            for cov in ["T", "frac", "segs", "res_tree", "cov_tree", "nested"]:
                x = np.array([r[cov] for r in rows], dtype=float)
                d = np.array([r["diff"] for r in rows])
                qs = np.quantile(x, [1 / 3, 2 / 3])
                g = np.digitize(x, qs, right=True)
                terc[cov] = [{"range": [float(x[g == i].min()), float(x[g == i].max())], "n": int((g == i).sum()),
                              "diff": float(d[g == i].mean()),
                              "res_gap": float(np.mean([r["res_tree"] - r["res_level"] for r, gi in zip(rows, g)
                                                        if gi == i]))}
                             for i in range(3) if (g == i).any()]
            report[corpus][str(L)] = {"n_rows": len(rows), "mean": mean, "terciles": terc}
            print(corpus, "L%d" % L, "videos x trials", len(rows), json.dumps({kk: round(mean[kk], 3) for kk in keys}))
            for cov, ts in terc.items():
                print("   ", cov, " | ".join("%s n=%d diff %+.3f resgap %+.1f" % (
                    np.round(t["range"], 2).tolist(), t["n"], t["diff"], t["res_gap"]) for t in ts))
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print("DONE", out)


if __name__ == "__main__":
    main()
