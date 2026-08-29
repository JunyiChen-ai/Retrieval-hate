#!/usr/bin/env python3
"""C3 pilot: top-k MIL + cross-video kNN pseudo-negatives inside hateful videos.

Selection (frozen, model-independent): per-modality L2-normalized concatenated
features; for each second of each hateful train video, mean cosine distance to
its 5 nearest benign-train seconds; pseudo-negative iff distance in the closest
quartile within its own video AND below the global median, capped at 50% of the
video. Loss = top-k MIL + lambda * BCE(pseudo-negatives -> 0), lambda 1.0.

Arms: xneg (the method), rand (position-matched random pseudo-negatives,
attribution control). Everything else identical to the weak-MIL control.

Output: runs/20260830_xneg_mil_pilot/{results.json,results.md,strata.md}
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = "/home/jehc223/Retrieval-hate"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(REPO, "experiments", "20260830_powa_within_diagnosis"))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from skyline import TemporalConv, load_feats, FEATSETS  # noqa: E402

OUT_DIR = os.path.join(REPO, "runs", "20260830_xneg_mil_pilot")
CORPORA = ("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg")
SEEDS = (234, 2025, 3407)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DIRS = FEATSETS["clip+vgg+bert"]
DIMS = (512, 128, 768)  # clip, vggish, bert
KNN_K, LAMBDA = 5, 1.0


def norm_concat(x):
    """L2-normalize each modality block, then concatenate."""
    parts, off = [], 0
    for d in DIMS:
        b = x[:, off:off + d]
        parts.append(b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8))
        off += d
    return np.concatenate(parts, axis=1).astype(np.float32)


def pack(corpus, split):
    labels = hdata.load_labels(corpus)
    out = []
    for v in hdata.load_split(corpus, split):
        y = labels.get(v)
        if y is None:
            continue
        try:
            T = int(np.load(os.path.join(REPO, "results/reproduction/features/"
                                         "vggish_1s", corpus, v + ".npy"),
                            mmap_mode="r").shape[0])
            x = load_feats(corpus, v, DIRS, T)
        except FileNotFoundError:
            continue
        out.append((v, x, float(y)))
    return out


def select_pseudo_negatives(train, rng, arm):
    """Return {video_index: bool mask over seconds} for hateful train videos."""
    benign = [norm_concat(x) for _, x, y in train if y == 0]
    if not benign:
        return {}
    bank = torch.from_numpy(np.concatenate(benign))              # (N, D) CPU
    masks = {}
    dists_all, per_video = [], {}
    for i, (_, x, y) in enumerate(train):
        if y != 1:
            continue
        q = torch.from_numpy(norm_concat(x))                     # (T, D) CPU
        sim = q @ bank.T                                          # cosine
        d = 1.0 - torch.topk(sim, min(KNN_K, bank.shape[0]), dim=1
                             ).values.mean(1)
        d = d.numpy()
        per_video[i] = d
        dists_all.append(d)
    gmed = float(np.median(np.concatenate(dists_all)))
    for i, d in per_video.items():
        T = len(d)
        n_cap = T // 2
        if arm == "xneg":
            q25 = np.quantile(d, 0.25)
            cand = np.where((d <= q25) & (d < gmed))[0]
            cand = cand[np.argsort(d[cand])][:n_cap]
        else:  # rand: position-matched count, uniform sample
            q25 = np.quantile(d, 0.25)
            n = min(len(np.where((d <= q25) & (d < gmed))[0]), n_cap)
            cand = rng.choice(T, size=n, replace=False)
        m = np.zeros(T, dtype=bool)
        m[cand] = True
        masks[i] = m
    return masks


def run(corpus, seed, arm, epochs=30, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train = pack(corpus, "train")
    gt_te = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    hate_ids = {v for v in gt_te if labels.get(v) == 1}
    masks = select_pseudo_negatives(train, rng, arm) if arm != "mil" else {}
    train_t = [(v, torch.from_numpy(x), y) for v, x, y in train]
    model = TemporalConv(train_t[0][1].shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for i in rng.permutation(len(train_t)):
            _, x, y = train_t[i]
            logits = model(x.to(DEV))
            k = max(1, len(logits) // 8)
            bag = torch.topk(logits, k).values.mean()
            loss = nn.functional.binary_cross_entropy_with_logits(
                bag, torch.tensor(y, device=DEV))
            if i in masks and masks[i].any():
                sel = torch.from_numpy(masks[i]).to(DEV)
                loss = loss + LAMBDA * nn.functional.binary_cross_entropy_with_logits(
                    logits[sel], torch.zeros(int(sel.sum()), device=DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    scores = {}
    with torch.no_grad():
        for v in gt_te:
            T = len(gt_te[v])
            try:
                x = load_feats(corpus, v, DIRS, T)
            except FileNotFoundError:
                continue
            scores[v] = model(torch.from_numpy(x).to(DEV)).cpu().numpy().astype(float)
    m = evaluate_scores(scores, {v: gt_te[v] for v in scores}, hate_ids)
    # high-pos stratum
    hi = []
    for v, auc in m["per_video"]["per_video_auc"].items():
        y = np.asarray(gt_te[v])
        if y.mean() > 0.6:
            hi.append(auc)
    return {"frame_ap": m["pr_auc"], "frame_roc": m["roc_auc"],
            "within_roc_macro": m["per_video"]["macro_auc"],
            "within_n": m["per_video"]["n_videos_both_classes"],
            "hipos_within_roc": float(np.mean(hi)) if hi else None,
            "hipos_n": len(hi),
            "video_roc_max": m["video_level"]["max_roc_auc"],
            "n_scored": len(scores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="xneg,rand")
    ap.add_argument("--corpora", default=",".join(CORPORA))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for corpus in args.corpora.split(","):
        for arm in args.arms.split(","):
            per_seed = [run(corpus, s, arm) for s in SEEDS]
            agg = {}
            for key in per_seed[0]:
                vals = [p[key] for p in per_seed if p[key] is not None]
                if key in ("within_n", "hipos_n", "n_scored"):
                    agg[key] = per_seed[0][key]
                else:
                    agg[key] = {"mean": float(np.mean(vals)),
                                "sd": float(np.std(vals))} if vals else None
            results["%s/%s" % (corpus, arm)] = {"seeds": per_seed, "agg": agg}
            print(corpus, arm, json.dumps(agg), flush=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
        json.dump({"seeds": list(SEEDS), "knn_k": KNN_K, "lambda": LAMBDA,
                   "results": results}, fh, indent=1)
    lines = ["# C3 cross-video pseudo-negative MIL (TEST, 3 seeds)", "",
             "| corpus/arm | frame AP | frame ROC | within-ROC macro (n) | "
             "hi-pos within-ROC (n) |", "|---|---:|---:|---:|---:|"]
    for key, r in results.items():
        a = r["agg"]
        lines.append("| %s | %.4f±%.3f | %.4f±%.3f | %.4f±%.3f (%d) | %.4f (%d) |" % (
            key, a["frame_ap"]["mean"], a["frame_ap"]["sd"],
            a["frame_roc"]["mean"], a["frame_roc"]["sd"],
            a["within_roc_macro"]["mean"], a["within_roc_macro"]["sd"],
            a["within_n"],
            a["hipos_within_roc"]["mean"] if a["hipos_within_roc"] else float("nan"),
            a["hipos_n"]))
    with open(os.path.join(OUT_DIR, "results.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
