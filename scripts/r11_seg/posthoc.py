#!/usr/bin/env python
"""R11-SEG post-hoc DESCRIPTIVE diagnostics. Not in the freeze, not a gate.

Run after run_pilot.py. Answers "why did the temporal operator buy nothing":
  (1) how much within-video variation the test split actually has;
  (2) the ceiling of a predictor with perfect video-level knowledge and zero
      temporal resolution (the landscape's degenerate oracle, on this task);
  (3) within-video AUC of each arm -- the read-out a video-level classifier
      cannot inflate, since a constant-per-video prediction scores exactly 0.5;
  (4) how much of each arm's per-window prediction is video-level (between-video
      variance of the mean logit) vs within-video.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
OUT = ROOT / "idea-stage/r11_seg/out"
K = 30


def macro_f1(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    f1p = 2 * tp / max(2 * tp + fp + fn, 1)
    f1n = 2 * tn / max(2 * tn + fn + fp, 1)
    return 100 * (f1p + f1n) / 2


def ts_macro_f1(y_ts, wot, pred_win, ids):
    tp = fp = fn = tn = 0
    for j, i in enumerate(ids):
        p = pred_win[j][wot[i]]; y = y_ts[i]
        tp += int(((p == 1) & (y == 1)).sum()); fp += int(((p == 1) & (y == 0)).sum())
        fn += int(((p == 0) & (y == 1)).sum()); tn += int(((p == 0) & (y == 0)).sum())
    f1p = 2 * tp / max(2 * tp + fp + fn, 1)
    f1n = 2 * tn / max(2 * tn + fn + fp, 1)
    return 100 * (f1p + f1n) / 2, 100 * (tp + tn) / max(tp + fp + fn + tn, 1)


def auc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    # average ranks for ties
    ss = s[o]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[o[i : j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    g = np.load(OUT / "grid_labels.npz", allow_pickle=True)
    vids = [str(v) for v in g["video_ids"]]
    y_win, y_ts, wot = g["y_win"], list(g["y_ts"]), list(g["win_of_ts"])
    split = json.loads((ROOT / "data/gt/HateClipSeg/p11_split.json").read_text())
    idx = {v: i for i, v in enumerate(vids)}
    te = np.array([idx[v] for v in split["test"]])
    Y = y_win[te]

    rep = {}
    frac = Y.mean(1)
    rep["n_test"] = len(te)
    rep["test_window_base_rate"] = float(Y.mean())
    rep["videos_all_normal"] = int((frac == 0).sum())
    rep["videos_all_offensive"] = int((frac == 1).sum())
    rep["videos_with_within_video_variation"] = int(((frac > 0) & (frac < 1)).sum())
    rep["mean_offensive_fraction"] = float(frac.mean())

    # (2) degenerate oracle: perfect video-level knowledge, zero temporal resolution
    pred = np.repeat((frac >= 0.5).astype(int)[:, None], K, axis=1)
    f, a = ts_macro_f1(y_ts, wot, pred, te)
    rep["oracle_video_broadcast_ts_macro_f1"] = f
    rep["oracle_video_broadcast_ts_acc"] = a
    rep["oracle_video_broadcast_win_macro_f1"] = macro_f1(Y.ravel(), pred.ravel())
    # perfect per-window oracle for reference
    rep["oracle_perwindow_ts_macro_f1"] = ts_macro_f1(y_ts, wot, Y, te)[0]

    # (3)/(4) per-arm read-outs
    arms = {}
    for f_ in sorted(OUT.glob("probs_*.npy")):
        tag = f_.stem[len("probs_"):]
        P = np.load(f_)
        wv, used = [], 0
        for j in range(len(te)):
            if 0 < Y[j].sum() < K:
                v = auc(Y[j], P[j])
                if not np.isnan(v):
                    wv.append(v); used += 1
        mu = P.mean(1)
        arms[tag] = dict(
            wv_auc=float(np.mean(wv)), n_videos_scored=used,
            pooled_auc=float(auc(Y.ravel(), P.ravel())),
            between_video_var=float(mu.var()),
            within_video_var=float((P - mu[:, None]).var()),
            frac_variance_between=float(mu.var() / max(P.var(), 1e-12)),
        )
    rep["arms"] = arms

    (OUT / "posthoc.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "arms"}, indent=2))
    print(f"{'arm':32s} {'wv-AUC':>8s} {'pooledAUC':>10s} {'%var between':>13s}")
    for k, v in sorted(arms.items()):
        print(f"{k:32s} {v['wv_auc']:8.4f} {v['pooled_auc']:10.4f} {100*v['frac_variance_between']:12.1f}%")


if __name__ == "__main__":
    main()
