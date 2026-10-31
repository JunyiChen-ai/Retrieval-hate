"""Goal 2 diagnostic: how much per-second hate information the fixed input features carry (CPU, no VLM answers).

A per-second classifier is fitted with full per-second supervision on the validation videos (the only split with
per-second GT besides test) and evaluated on test with the unified frame metrics. Inputs are the features r5 reads
(I3D visual 5-crop mean, VGGish audio, BERT text, `qtl.data.Store`). Feature sets:
  - second: x_t;
  - second+video: x_t and the video mean of x;
  - second+video+local: also the means of x over centred windows of 3, 9 and 27 seconds (diagnostic constants).
Classifiers: logistic regression (C by grouped 5-fold CV over validation videos) and a one-hidden-layer MLP.
If a supervised probe on these features localises far better than r5 (within .54 / .77 at 0 calls), the limit of
r5 is the weak supervision and the network, not the features. Validation GT is used only for this diagnostic
(no gradient into any method model). Rule-10 developmental evidence.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import data as qdata
import hier_evidence_common as hc

WINDOWS = (3, 9, 27)


def local_mean(x, w):
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), 0)
    T, r = len(x), w // 2
    lo, hi = np.clip(np.arange(T) - r, 0, T), np.clip(np.arange(T) + r + 1, 0, T)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def features(store, v, kind):
    x = np.concatenate([np.mean([store.visual(v, c) for c in range(5)], 0), store.at[v]], 1).astype(np.float32)
    parts = [x]
    if kind in ("second+video", "second+video+local"):
        parts.append(np.repeat(x.mean(0, keepdims=True), len(x), 0))
    if kind == "second+video+local":
        parts += [local_mean(x, w) for w in WINDOWS]
    return np.concatenate(parts, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg"), required=True)
    ap.add_argument("--out-root", default="runs/20261031_two_goal_diagnostics/feature_ceiling")
    a = ap.parse_args()
    labels, ids, gt_all, _ = hc.load_fixed_cohort(a.corpus)
    vids = {sp: [v for v in ids[sp] if v in gt_all[sp]] for sp in ("val", "test")}
    store = qdata.Store(a.corpus, vids["val"] + vids["test"], ["bert"])
    gt = {v: np.asarray(gt_all[sp][v]).astype(np.int64) for sp in ("val", "test") for v in vids[sp]}
    hate = {v for v in vids["test"] if labels[v] == 1}
    report = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(), "corpus": a.corpus,
              "n_val_seconds": int(sum(len(gt[v]) for v in vids["val"])), "results": {}}
    for kind in ("second", "second+video", "second+video+local"):
        X = {v: features(store, v, kind) for v in vids["val"] + vids["test"]}
        Xtr = np.concatenate([X[v] for v in vids["val"]])
        ytr = np.concatenate([gt[v] for v in vids["val"]])
        grp = np.concatenate([[i] * len(gt[v]) for i, v in enumerate(vids["val"])])
        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr)
        best = None
        for C in (1e-4, 1e-3, 1e-2, 1e-1):
            pr = np.zeros(len(ytr))
            for tr, te in GroupKFold(5).split(Xtr, ytr, grp):
                pr[te] = LogisticRegression(C=C, max_iter=2000).fit(Xtr[tr], ytr[tr]).decision_function(Xtr[te])
            cv = average_precision_score(ytr, pr) + roc_auc_score(ytr, pr)
            if best is None or cv > best[0]:
                best = (cv, C)
        clfs = {"logistic": LogisticRegression(C=best[1], max_iter=2000),
                "mlp": MLPClassifier((256,), alpha=1e-2, early_stopping=True, max_iter=200, random_state=0)}
        for name, clf in clfs.items():
            clf.fit(Xtr, ytr)
            scores = {v: clf.predict_proba(sc.transform(X[v]))[:, 1] for v in vids["test"]}
            m = hc.frame_metrics(scores, {v: gt[v] for v in vids["test"]}, hate)
            pos = [v for v in vids["test"] if labels[v] == 1]
            rho = spearmanr([scores[v].mean() for v in pos], [gt[v].mean() for v in pos]).correlation
            r = {"pooled_ap": float(m["pooled_ap"]), "pooled_roc": float(m["pooled_roc"]),
                 "within_roc": float(m["within_roc"]), "level_fraction_spearman": float(rho)}
            if name == "logistic":
                r["C"] = best[1]
            report["results"][f"{kind}/{name}"] = r
            print(a.corpus, kind, name, json.dumps(r), flush=True)
    out = ROOT / a.out_root
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.corpus}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(out / f"{a.corpus}.json")


if __name__ == "__main__":
    main()
