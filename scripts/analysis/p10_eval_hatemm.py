#!/usr/bin/env python
"""P10 — HateMM span localization eval (calibration set, CPU).

P6-style within-video localization on HateMM's gold hate_snippet spans
(data/gt/HateMM/hate_spans.json). This is the FREE-ITERATION calibration set:
score windows with any MLLM config, map to 1-fps seconds, and measure
within-video AUC (primary) + AP over hateful videos with both-class seconds.

Usage:
  # single config
  p10_eval_hatemm.py --scores_tag segscoreK30_qwen --K 30
  # paired vs a baseline config (anchor) -> wv-AUC delta + bootstrap CI + sign test
  p10_eval_hatemm.py --scores_tag <cfg> --K <k> --baseline_tag segscoreK30_qwen --baseline_K 30
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = "/data/jehc223/RGCL"
SPANS = os.path.join(ROOT, "data/gt/HateMM/hate_spans.json")
MDIR = os.path.join(ROOT, "data/MLLM_scores/HateMM")
SPLITS = ["train", "dev_seen", "test_seen"]


def load_spans():
    d = json.load(open(SPANS))
    return {k: v for k, v in d.items() if v.get("spans")}


def load_scores(tag):
    """Merge per-split <split>_<tag>.jsonl -> {id: [scores]}."""
    S = {}
    for sp in SPLITS:
        p = os.path.join(MDIR, "{}_{}.jsonl".format(sp, tag))
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if line:
                r = json.loads(line)
                S[str(r["id"])] = np.asarray(r.get("scores") or [], dtype=np.float64)
    return S


def sec_labels(v, K):
    """1-fps second labels + window index for one video."""
    D = float(v["duration"])
    spans = v["spans"]
    labs, qs = [], []
    for t in range(int(np.floor(D))):
        mid = t + 0.5
        lab = int(any(s <= mid < e for s, e in spans))
        labs.append(lab)
        qs.append(min(K - 1, int(mid * K / D)))
    return np.array(labs), np.array(qs)


def wv_auc_list(spans, S, K, rand=None):
    """Per-video within-video AUC over hateful videos with both-class seconds.
    rand: RandomState -> random control scores instead of S."""
    aucs, vids = [], []
    for vid, v in sorted(spans.items()):
        if vid not in S and rand is None:
            continue
        lab, qs = sec_labels(v, K)
        if len(lab) == 0 or lab.sum() == 0 or lab.sum() == len(lab):
            continue  # need both classes
        if rand is not None:
            sc = rand.random(len(lab))
        else:
            row = S[vid]
            if len(row) < K:
                continue
            sc = row[qs]
        if np.allclose(sc, sc[0]):
            aucs.append(0.5)
        else:
            aucs.append(float(roc_auc_score(lab, sc)))
        vids.append(vid)
    return np.array(aucs), vids


def ap_hateonly(spans, S, K):
    labs, scs = [], []
    for vid, v in sorted(spans.items()):
        if vid not in S:
            continue
        lab, qs = sec_labels(v, K)
        if len(lab) == 0:
            continue
        row = S[vid]
        if len(row) < K:
            continue
        labs.append(lab); scs.append(row[qs])
    if not labs:
        return None, None
    lab = np.concatenate(labs); sc = np.concatenate(scs)
    if lab.sum() in (0, len(lab)):
        return None, None
    return float(average_precision_score(lab, sc)), float(roc_auc_score(lab, sc))


def boot_ci(a, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    b = np.array([a[rng.randint(0, len(a), len(a))].mean() for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_tag", required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--baseline_tag", default="")
    ap.add_argument("--baseline_K", type=int, default=0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    spans = load_spans()
    S = load_scores(args.scores_tag)
    a, vids = wv_auc_list(spans, S, args.K)
    lo, hi = boot_ci(a)
    gt = int((a > 0.5).sum()); lt = int((a < 0.5).sum())
    p = binomtest(gt, gt + lt, 0.5, alternative="greater").pvalue if gt + lt else 1.0
    apv, aucv = ap_hateonly(spans, S, args.K)
    rnd, _ = wv_auc_list(spans, S, args.K, rand=np.random.RandomState(0))
    print("[{}] K={} scored {} vids | wv-AUC {:.4f} CI[{:.4f},{:.4f}] "
          "(>.5 {}/{}, sign-p {:.3g}) | AP-hateonly {:.4f} AUC {:.4f} | random wv {:.4f}".format(
              args.label or args.scores_tag, args.K, len(a), a.mean(), lo, hi,
              gt, gt + lt, p, apv or -1, aucv or -1, rnd.mean()))
    out = dict(tag=args.scores_tag, K=args.K, n_videos=len(a),
               wv_auc=float(a.mean()), wv_ci=[lo, hi], sign_p=float(p),
               ap_hateonly=apv, auc_hateonly=aucv, random_wv=float(rnd.mean()))

    if args.baseline_tag:
        Sb = load_scores(args.baseline_tag)
        b, _ = wv_auc_list(spans, Sb, args.baseline_K)
        # pair on common video set
        A, V = wv_auc_list(spans, S, args.K)
        Bmap = dict(zip(*[wv_auc_list(spans, Sb, args.baseline_K)[1],
                          wv_auc_list(spans, Sb, args.baseline_K)[0]]))
        pair = [(av, Bmap[v]) for av, v in zip(A, V) if v in Bmap]
        d = np.array([x - y for x, y in pair])
        dlo, dhi = boot_ci(d)
        gt2 = int((d > 1e-9).sum()); lt2 = int((d < -1e-9).sum())
        pp = binomtest(gt2, gt2 + lt2, 0.5, alternative="greater").pvalue if gt2 + lt2 else 1.0
        print("  PAIRED vs {}(K{}): Δwv-AUC {:+.4f} CI[{:+.4f},{:+.4f}] "
              "(excl 0: {}) sign-p {:.3g}  [cfg {:.4f} vs base {:.4f}]".format(
                  args.baseline_tag, args.baseline_K, d.mean(), dlo, dhi,
                  dlo > 0, pp, np.mean([x for x, _ in pair]), np.mean([y for _, y in pair])))
        out["paired_delta"] = float(d.mean())
        out["paired_ci"] = [dlo, dhi]
        out["paired_sign_p"] = float(pp)
        out["baseline_wv"] = float(np.mean([y for _, y in pair]))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
