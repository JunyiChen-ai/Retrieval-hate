"""Where the real answers differ from the anchored answer model (README section 2.3; rule-10 developmental evidence,
validation and test GT read, nothing trained or selected).

For every answered queryable node of the validation and test videos: the soft answer p (first-token P(Yes)), its
level, the node length and its GT state (0 negative video, 1 positive video without GT harm in the node, 2 with GT
harm) and, for state 2, the fraction of GT-harmful seconds in the node. Reports mean p and mean level by state and
length bucket (copy_noise.BUCKETS), by harm-fraction bin for state 2, and the anchored model's expected level for
the same lengths; and the within-video dependence of the answers: the share of the variance of the level residual
(level minus its state-and-length-bucket mean) that lies between videos, per state.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import data as qdata, qtree
from qtl.copy_noise import BUCKETS, bucket_of
import hier_evidence_common as hc

FRAC_BINS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0001)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--out-root", default="runs/20261101_r5_completion/answer_stats")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    qdata.configure_source("soft_both", 8)
    answers, T_ans = qdata.load_answers(a.corpus, "soft_both")
    P, _, _ = qdata.load_soft_p(a.corpus, "soft_both")
    store = qdata.Store(a.corpus, ids["val"] + ids["test"], ["bert"])
    report = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(), "corpus": a.corpus,
              "buckets": [list(map(float, b)) for b in BUCKETS], "frac_bins": list(FRAC_BINS), "splits": {}}
    for split in ("val", "test"):
        rows = []                                        # (video, state, bucket, length, frac, p, level)
        for v in ids[split]:
            T = store.T[v]
            tr = qtree.tree(T)
            y = np.zeros(T)
            g = np.asarray(gt[split][v], dtype=np.float64)[:T]
            y[:len(g)] = g > 0.5
            c = np.concatenate([[0.0], np.cumsum(y)])
            for n in np.where(tr["queryable"])[0]:
                key = (int(tr["a"][n]), int(tr["b"][n]))
                o = answers[v].get(key)
                p = P[v].get(key)
                if o is None or p is None:
                    continue
                L = key[1] - key[0]
                k = c[key[1]] - c[key[0]]
                s = 0 if labels[v] == 0 else (2 if k > 0 else 1)
                rows.append((v, s, bucket_of(float(L)), L, k / L, float(p), int(o[0]), key == (0, T)))
        sp = {}
        st = np.array([r[1] for r in rows])
        bk = np.array([r[2] for r in rows])
        fr = np.array([r[4] for r in rows])
        pp = np.array([r[5] for r in rows])
        lv = np.array([r[6] for r in rows], dtype=np.float64)
        root = np.array([r[7] for r in rows])
        vid = np.array([r[0] for r in rows])
        cell = {}
        for s in (0, 1, 2):
            for b in sorted(set(bk.tolist())):
                m = (st == s) & (bk == b)
                if m.sum():
                    cell["s%d_b%d" % (s, b)] = {"n": int(m.sum()), "mean_p": float(pp[m].mean()),
                                                "mean_level": float(lv[m].mean()),
                                                "p_yes_gt_half": float((pp[m] > 0.5).mean())}
            m = (st == s) & root
            if m.sum():
                cell["s%d_root" % s] = {"n": int(m.sum()), "mean_p": float(pp[m].mean()),
                                        "mean_level": float(lv[m].mean()), "p_yes_gt_half": float((pp[m] > 0.5).mean())}
        sp["by_state_bucket"] = cell
        frac = {}
        for i in range(len(FRAC_BINS) - 1):
            m = (st == 2) & (fr > FRAC_BINS[i] - 1e-12 if i == 0 else fr > FRAC_BINS[i]) & (fr <= FRAC_BINS[i + 1])
            if m.sum():
                frac["(%.2f,%.2f]" % (FRAC_BINS[i], FRAC_BINS[i + 1])] = {
                    "n": int(m.sum()), "mean_p": float(pp[m].mean()), "mean_level": float(lv[m].mean()),
                    "p_yes_gt_half": float((pp[m] > 0.5).mean())}
        sp["state2_by_harm_fraction"] = frac
        # within-video dependence: between-video share of the residual variance per state (one-way ANOVA ICC)
        res = lv.copy()
        for s in (0, 1, 2):
            for b in set(bk.tolist()):
                m = (st == s) & (bk == b)
                if m.sum():
                    res[m] -= lv[m].mean()
        icc = {}
        for s in (0, 1, 2):
            m = st == s
            if m.sum() < 10:
                continue
            groups = {}
            for v_, r_ in zip(vid[m], res[m]):
                groups.setdefault(v_, []).append(r_)
            gs = [np.asarray(x) for x in groups.values() if len(x) >= 2]
            if len(gs) < 5:
                continue
            k_bar = np.mean([len(x) for x in gs])
            msb = np.sum([len(x) * (x.mean() - np.concatenate(gs).mean()) ** 2 for x in gs]) / (len(gs) - 1)
            msw = np.sum([((x - x.mean()) ** 2).sum() for x in gs]) / (sum(len(x) for x in gs) - len(gs))
            icc["s%d" % s] = {"icc": float((msb - msw) / (msb + (k_bar - 1) * msw)), "videos": len(gs),
                              "mean_nodes": float(k_bar)}
        sp["icc_level_residual"] = icc
        report["splits"][split] = sp
        print(a.corpus, split, json.dumps(frac), json.dumps(icc), flush=True)
    out = (ROOT / a.out_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.corpus}.json").write_text(json.dumps(report, indent=1) + "\n")
    print("DONE", a.corpus, flush=True)


if __name__ == "__main__":
    main()
