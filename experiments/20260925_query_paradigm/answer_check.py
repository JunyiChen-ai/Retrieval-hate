"""Check of the new tree answers on test videos (README section 1, developmental evidence under rule 10; no
training or selection uses it). Per corpus, over the test videos that already have answers:
  - node level: P(category >= 2 | node contains GT hate), P(... | positive video, node without hate),
    P(... | negative video), for each category and for "any category"; node ROC of the max level;
  - the old VLM-silent positives (old K30 fine rate of level >= 2 below .1): how many now get an answer >= 2 in
    any category at the root / at any node, and in the hate category alone.

    python experiments/20260925_query_paradigm/answer_check.py --corpus hateclipseg
Output: runs/20260925_query_paradigm/answer_check/<corpus>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from hate_common import data as hdata  # noqa: E402
import vlm_verdict                     # noqa: E402
import data as qdata                   # noqa: E402
import qtree                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    a = ap.parse_args()
    labels = hdata.load_labels(a.corpus)
    gt = hdata.gt_arrays(a.corpus, "test")
    answers, T = qdata.load_answers(a.corpus)
    old = vlm_verdict.load_verdicts(a.corpus, k=30, tag="qwen")
    ids = [v for v in hdata.load_split(a.corpus, "test") if v in answers and v in gt]
    rows = []                                   # (group, max level per category (5,))
    for v in ids:
        g = np.asarray(gt[v])
        for (s, e), o in answers[v].items():
            if o is None:
                continue
            grp = "neg" if labels[v] == 0 else ("in" if g[s:e].max() > 0 else "pos_out")
            rows.append((grp, o, e - s))
    out = {"n_videos": len(ids), "n_nodes": len(rows), "flag_rate": {}}
    for grp in ("in", "pos_out", "neg"):
        m = np.array([o for gname, o, _ in rows if gname == grp])
        if len(m) == 0:
            continue
        out["flag_rate"][grp] = {c: round(float(np.mean(m[:, k] >= 2)), 4) for k, c in enumerate(qtree.CATEGORIES)}
        out["flag_rate"][grp]["any"] = round(float(np.mean(m.max(1) >= 2)), 4)
        out["flag_rate"][grp]["n"] = int(len(m))
    y = np.array([1 if gname == "in" else 0 for gname, _, _ in rows])
    omax = np.array([o.max() for _, o, _ in rows])
    ohate = np.array([o[0] for _, o, _ in rows])
    out["node_roc_any_max"] = round(float(roc_auc_score(y, omax)), 4)
    out["node_roc_hate"] = round(float(roc_auc_score(y, ohate)), 4)
    silent = [v for v in ids if labels[v] == 1 and v in old and np.mean(old[v] >= 2) < 0.1]
    root = lambda v: answers[v].get((0, T[v]))                          # noqa: E731
    out["old_silent_pos"] = {
        "n": len(silent),
        "root_any_ge2": int(sum(root(v) is not None and root(v).max() >= 2 for v in silent)),
        "root_hate_ge2": int(sum(root(v) is not None and root(v)[0] >= 2 for v in silent)),
        "any_node_any_ge2": int(sum(any(o is not None and o.max() >= 2 for o in answers[v].values()) for v in silent)),
        "any_node_hate_ge2": int(sum(any(o is not None and o[0] >= 2 for o in answers[v].values()) for v in silent)),
    }
    negs = [v for v in ids if labels[v] == 0]
    out["neg_videos"] = {"n": len(negs),
                         "root_any_ge2": int(sum(root(v) is not None and root(v).max() >= 2 for v in negs)),
                         "root_hate_ge2": int(sum(root(v) is not None and root(v)[0] >= 2 for v in negs))}
    pos = [v for v in ids if labels[v] == 1]
    out["pos_videos"] = {"n": len(pos),
                         "root_any_ge2": int(sum(root(v) is not None and root(v).max() >= 2 for v in pos)),
                         "root_hate_ge2": int(sum(root(v) is not None and root(v)[0] >= 2 for v in pos))}
    lv = np.stack([o for _, o, _ in rows])
    out["level_hist"] = {c: np.bincount(lv[:, k], minlength=4).tolist() for k, c in enumerate(qtree.CATEGORIES)}
    d = os.path.join(ROOT, "runs", "20260925_query_paradigm", "answer_check")
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "%s.json" % a.corpus), "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
