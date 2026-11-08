"""Diagnostic (README section 1): are the answers of nested nodes more alike than the answer model (answers
independent given the node states) allows?

Cached soft answers (soft_both_p1) and validation / test GT. Residual of a node = logit P(Yes) minus the mean logit
of its cell (corpus, split, state in {negative video, f = 0, z = 1}, length bin of a factor 2). Per positive video,
pairs of nodes in the same state: nested pairs (one contains the other, length ratio 2 or 4) versus non-nested pairs
of the same lengths (disjoint) in the same video. Prints the Pearson correlation of the two residuals for each pair
type, and the same for negative videos. Under the model both correlations are about 0 except for what the shared
video adds (which the disjoint pairs measure).
"""
import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import hier_evidence_common as hc                                       # noqa: E402

DIRS = {"hatemm": "HateMM", "hateclipseg": "HateClipSeg", "dehate": "DeHate"}


def logit(p):
    p = np.clip(float(p), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+", default=["hatemm", "hateclipseg", "dehate"])
    ap.add_argument("--out", default="runs/20261109_tree_within/diag/answer_redundancy.json")
    a = ap.parse_args()
    rep = {}
    for corpus in a.corpora:
        labels, ids, gt, _ = hc.load_fixed_cohort(corpus)
        want = {v: s for s in ("val", "test") for v in ids[s]}
        vids = []
        for fn in sorted(glob.glob(str(ROOT / "data/vlm_tree" / DIRS[corpus] / "soft_both_p1*.jsonl"))):
            for line in open(fn):
                r = json.loads(line)
                s = want.get(r["id"])
                if s is None:
                    continue
                y = np.asarray(gt[s][r["id"]], dtype=float)
                nodes = []
                for a_, b_, pr, _ in r["nodes"]:
                    if b_ - a_ < 4 or b_ > len(y):
                        continue
                    st = "neg" if labels[r["id"]] == 0 else ("z1" if y[a_:b_].any() else "f0")
                    nodes.append((a_, b_, st, int(np.floor(np.log2(b_ - a_))), logit(pr[0])))
                vids.append((s, nodes))
        cell = {}
        for s, nodes in vids:
            for a_, b_, st, lb, x in nodes:
                cell.setdefault((s, st, lb), []).append(x)
        mu = {k: float(np.mean(v)) for k, v in cell.items()}
        pairs = {}
        for s, nodes in vids:
            res = [(a_, b_, st, lb, x - mu[(s, st, lb)]) for a_, b_, st, lb, x in nodes]
            for i, (a1, b1, s1, l1, r1) in enumerate(res):
                for a2, b2, s2, l2, r2 in res[i + 1:]:
                    if s1 != s2 or abs(l1 - l2) not in (1, 2):
                        continue
                    nested = (a1 <= a2 and b2 <= b1) or (a2 <= a1 and b1 <= b2)
                    disjoint = b1 <= a2 or b2 <= a1
                    if not (nested or disjoint):
                        continue
                    key = (s1, "nested" if nested else "disjoint", abs(l1 - l2))
                    pairs.setdefault(key, []).append((r1, r2))
        rep[corpus] = {}
        for key in sorted(pairs):
            P = np.array(pairs[key])
            c = float(np.corrcoef(P[:, 0], P[:, 1])[0, 1])
            rep[corpus]["%s %s ratio%d" % (key[0], key[1], 2 ** key[2])] = {"n": len(P), "corr": c}
        print(corpus, json.dumps({k: "%.2f (%d)" % (v["corr"], v["n"]) for k, v in rep[corpus].items()}, indent=0))
    o = ROOT / a.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(rep, indent=1) + "\n")
    print("DONE", o)


if __name__ == "__main__":
    main()
