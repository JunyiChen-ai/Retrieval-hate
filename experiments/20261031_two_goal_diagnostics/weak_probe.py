"""Goal 2 diagnostic: a linear per-second classifier trained with video labels only (CPU, no VLM answers).

Same inputs as r5 (I3D visual 5-crop mean, VGGish audio, BERT text; `qtl.data.Store`). Training videos only, their
video label copied to every second (the plainest weakly supervised per-second classifier); logistic regression on
standardised features, C chosen on the validation videos by video-level (AP + ROC) of the mean second score (video
labels only, no per-second GT). Test: unified frame metrics and the Spearman correlation between a positive
video's mean score and its GT hate fraction. Compared with r5 at 0 calls (content model only), this says whether
r5's network extracts the per-video level better or worse than a linear read-out of the same features.
Rule-10 developmental evidence.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import data as qdata
import hier_evidence_common as hc


def feats(store, v):
    return np.concatenate([np.mean([store.visual(v, c) for c in range(5)], 0), store.at[v]], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg"), required=True)
    ap.add_argument("--max-seconds-per-video", type=int, default=120, help="subsample long training videos")
    ap.add_argument("--out-root", default="runs/20261031_two_goal_diagnostics/weak_probe")
    a = ap.parse_args()
    labels, ids, gt_all, _ = hc.load_fixed_cohort(a.corpus)
    test = [v for v in ids["test"] if v in gt_all["test"]]
    store = qdata.Store(a.corpus, ids["train"] + ids["val"] + test, ["bert"])
    rng = np.random.default_rng(0)
    Xtr, ytr = [], []
    for v in ids["train"]:
        x = feats(store, v)
        if len(x) > a.max_seconds_per_video:
            x = x[np.sort(rng.choice(len(x), a.max_seconds_per_video, replace=False))]
        Xtr.append(x)
        ytr.append(np.full(len(x), labels[v]))
    Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr)
    Xval = {v: sc.transform(feats(store, v)) for v in ids["val"]}
    Xte = {v: sc.transform(feats(store, v)) for v in test}
    yval = [labels[v] for v in ids["val"]]
    report = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(), "corpus": a.corpus,
              "n_train_videos": len(ids["train"]), "n_train_seconds": int(len(ytr)), "by_C": {}}
    best = None
    for C in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        clf = LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr)
        mv = [clf.decision_function(Xval[v]).mean() for v in ids["val"]]
        val = float(average_precision_score(yval, mv) + roc_auc_score(yval, mv))
        scores = {v: clf.predict_proba(Xte[v])[:, 1] for v in test}
        m = hc.frame_metrics(scores, {v: np.asarray(gt_all["test"][v]) for v in test},
                             {v for v in test if labels[v] == 1})
        pos = [v for v in test if labels[v] == 1]
        rho = spearmanr([scores[v].mean() for v in pos], [np.asarray(gt_all["test"][v]).mean() for v in pos]).correlation
        r = {"val_video_ap_plus_roc": val, "pooled_ap": float(m["pooled_ap"]), "pooled_roc": float(m["pooled_roc"]),
             "within_roc": float(m["within_roc"]), "level_fraction_spearman": float(rho)}
        report["by_C"][str(C)] = r
        print(a.corpus, "C", C, json.dumps(r), flush=True)
        if best is None or val > best[0]:
            best = (val, C)
    report["validation_selected_C"] = best[1]
    report["validation_selected"] = report["by_C"][str(best[1])]
    out = ROOT / a.out_root
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.corpus}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(a.corpus, "validation-selected", best[1], json.dumps(report["validation_selected"]))
    print(out / f"{a.corpus}.json")


if __name__ == "__main__":
    main()
