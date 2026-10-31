"""Goal 2 diagnostic: what a plain per-second model learns from the training answers alone (CPU, no GT in training).

r5's content model is trained by the video label and the VLM soft answers on every node of the training videos.
Here the same training signal is used directly: every training second gets the p_yes of the shortest answered node
covering it (the all-answers score of `answer_quality.py`), and a ridge regression on the fixed per-second features
(I3D 5-crop mean, VGGish, BERT; optionally with the video mean and local means over 3, 9, 27 s) is fitted to it.
The ridge alpha is chosen on the validation videos by the agreement (Pearson) with their own all-answers score
(answers only, no GT). Test: unified frame metrics of the predicted per-second scores, at 0 calls, against r5 at 0
calls (.658 / .647 / .539 HateClipSeg, .623 / .819 / .768 HateMM). With --combine-r5, also the combination with
r5's own 0-call scores of each seed's test-selected trial (mean of the two standardised logits per second), the check
for a backbone with both r5's network and a linear path from the raw features. Rule-10 developmental evidence.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import data as qdata
import hier_evidence_common as hc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_ceiling import features                     # noqa: E402  (same feature sets)


def answer_score(nodes, T):
    best, acc, cnt = np.full(T, np.inf), np.zeros(T), np.zeros(T)
    for (s, e), p in nodes.items():
        if p is None:
            continue
        e = min(e, T)
        if e <= s:
            continue
        L = e - s
        b, a_, c = best[s:e], acc[s:e], cnt[s:e]
        shorter, same = L < b, L == b
        a_[shorter], c[shorter], b[shorter] = p, 1, L
        a_[same] += p
        c[same] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def combine(corpus, scores, gt, hate, labels, test):
    """Mean of standardised logits of the probe and of r5's 0-call test scores, per r5 seed; 3-seed mean metrics."""
    lp = {v: logit(scores[v]) for v in test}
    allp = np.concatenate(list(lp.values()))
    rows = []
    for seed in (234, 2025, 3407):
        study = ROOT / "runs/20260929_query_paradigm_r5" / corpus / f"seed{seed}"
        trial = json.loads((study / "study_summary.json").read_text())["best"]["number"]
        r5 = {}
        for line in (study / f"trial{trial}" / "scores_test_fixed0.jsonl").read_text().splitlines():
            r = json.loads(line)
            r5[r["video_id"]] = logit(np.asarray(r["score_av"], dtype=np.float64))
        allr = np.concatenate([r5[v] for v in test])
        comb = {v: 1 / (1 + np.exp(-((lp[v] - allp.mean()) / allp.std() + (r5[v] - allr.mean()) / allr.std()) / 2))
                for v in test}
        m = hc.frame_metrics(comb, gt, hate)
        pos = [v for v in test if labels[v] == 1]
        rows.append([m["pooled_ap"], m["pooled_roc"], m["within_roc"],
                     spearmanr([comb[v].mean() for v in pos], [gt[v].mean() for v in pos]).correlation])
    return dict(zip(("pooled_ap", "pooled_roc", "within_roc", "level_fraction_spearman"),
                    np.mean(rows, 0).tolist()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg"), required=True)
    ap.add_argument("--max-seconds-per-video", type=int, default=150)
    ap.add_argument("--kinds", nargs="+", default=["second", "second+video", "second+video+local"])
    ap.add_argument("--combine-r5", action="store_true")
    ap.add_argument("--out-root", default="runs/20261031_two_goal_diagnostics/answer_probe")
    a = ap.parse_args()
    labels, ids, gt_all, _ = hc.load_fixed_cohort(a.corpus)
    P, _, _ = qdata.load_soft_p(a.corpus, "soft_both")
    test = [v for v in ids["test"] if v in gt_all["test"]]
    train = [v for v in ids["train"] if v in P]
    val = [v for v in ids["val"] if v in P]
    store = qdata.Store(a.corpus, train + val + test, ["bert"])
    rng = np.random.default_rng(0)
    gt = {v: np.asarray(gt_all["test"][v]) for v in test}
    hate = {v for v in test if labels[v] == 1}
    report = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(), "corpus": a.corpus,
              "n_train_videos": len(train), "results": {}}
    for kind in a.kinds:
        Xtr, ytr = [], []
        for v in train:
            x, y = features(store, v, kind), answer_score(P[v], store.T[v])
            keep = np.flatnonzero(np.isfinite(y))
            if len(keep) > a.max_seconds_per_video:
                keep = np.sort(rng.choice(keep, a.max_seconds_per_video, replace=False))
            Xtr.append(x[keep])
            ytr.append(y[keep])
        Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr)
        Xv = {v: sc.transform(features(store, v, kind)) for v in val}
        yv = {v: answer_score(P[v], store.T[v]) for v in val}
        best = None
        for alpha in np.logspace(0, 6, 7):
            reg = Ridge(alpha=alpha).fit(Xtr, ytr)
            pv = np.concatenate([reg.predict(Xv[v])[np.isfinite(yv[v])] for v in val])
            tv = np.concatenate([yv[v][np.isfinite(yv[v])] for v in val])
            r = float(np.corrcoef(pv, tv)[0, 1])
            if best is None or r > best[0]:
                best = (r, alpha, reg)
        reg = best[2]
        scores = {v: reg.predict(sc.transform(features(store, v, kind))) for v in test}
        allv = np.concatenate(list(scores.values()))
        mu, sd = allv.mean(), allv.std() + 1e-12                 # rank-preserving map into (0, 1)
        scores = {v: 1 / (1 + np.exp(-(s - mu) / sd)) for v, s in scores.items()}
        m = hc.frame_metrics(scores, gt, hate)
        pos = [v for v in test if labels[v] == 1]
        res = {"alpha": float(best[1]), "val_pearson_with_answers": best[0],
               "pooled_ap": float(m["pooled_ap"]), "pooled_roc": float(m["pooled_roc"]),
               "within_roc": float(m["within_roc"]),
               "level_fraction_spearman": float(spearmanr([scores[v].mean() for v in pos],
                                                          [gt[v].mean() for v in pos]).correlation)}
        if a.combine_r5:
            res["combined_with_r5"] = combine(a.corpus, scores, gt, hate, labels, test)
        report["results"][kind] = res
        print(a.corpus, kind, json.dumps(res), flush=True)
    out = ROOT / a.out_root
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.corpus}{'_combined' if a.combine_r5 else ''}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(out / f"{a.corpus}.json")


if __name__ == "__main__":
    main()
