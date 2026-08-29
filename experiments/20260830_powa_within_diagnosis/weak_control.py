#!/usr/bin/env python3
"""Weak-MIL control on the exact skyline architecture and features.

Same TemporalConv + CLIP/VGGish/BERT inputs as the supervised skyline, but
trained ONLY with video-level labels via top-k MIL (mean of top-k logits per
video as the bag logit). This anchors the objective-gap measurement: the
distance between this control and skyline_train is attributable to the
supervision signal alone, with architecture and features held fixed.

3 seeds. Test eval. Output: runs/20260830_powa_within_diagnosis/weak_control.*
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = "/home/jehc223/Retrieval-hate"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
sys.path.insert(0, HERE)
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from skyline import TemporalConv, load_feats, FEATSETS  # noqa: E402

OUT_DIR = os.path.join(REPO, "runs", "20260830_powa_within_diagnosis")
CORPORA = ("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg")
SEEDS = (234, 2025, 3407)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DIRS = FEATSETS["clip+vgg+bert"]


def pack(corpus, split):
    labels = hdata.load_labels(corpus)
    out = []
    for v in hdata.load_split(corpus, split):
        y = labels.get(v)
        if y is None:
            continue
        try:
            # length from vggish (the duration reference used by gold building)
            T = int(np.load(os.path.join(REPO, "results/reproduction/features/"
                                         "vggish_1s", corpus, v + ".npy"),
                            mmap_mode="r").shape[0])
            x = load_feats(corpus, v, DIRS, T)
        except FileNotFoundError:
            continue
        out.append((v, torch.from_numpy(x), float(y)))
    return out


def run(corpus, seed, epochs=30, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = pack(corpus, "train")
    gt_te = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    hate_ids = {v for v in gt_te if labels.get(v) == 1}
    dim = train[0][1].shape[1]
    model = TemporalConv(dim).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for i in np.random.permutation(len(train)):
            _, x, y = train[i]
            logits = model(x.to(DEV))
            k = max(1, len(logits) // 8)
            bag = torch.topk(logits, k).values.mean()
            loss = nn.functional.binary_cross_entropy_with_logits(
                bag, torch.tensor(y, device=DEV))
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
    return {"frame_ap": m["pr_auc"], "frame_roc": m["roc_auc"],
            "within_roc_macro": m["per_video"]["macro_auc"],
            "within_n": m["per_video"]["n_videos_both_classes"],
            "video_roc_max": m["video_level"]["max_roc_auc"]}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for corpus in CORPORA:
        per_seed = [run(corpus, s) for s in SEEDS]
        agg = {}
        for key in per_seed[0]:
            if key == "within_n":
                agg[key] = per_seed[0][key]
                continue
            vals = [p[key] for p in per_seed if p[key] is not None]
            agg[key] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}
        results[corpus] = {"seeds": per_seed, "agg": agg}
        print(corpus, json.dumps(agg))
    with open(os.path.join(OUT_DIR, "weak_control.json"), "w") as fh:
        json.dump({"note": "top-k MIL, video labels only, same arch/features as "
                           "skyline; TEST eval", "seeds": list(SEEDS),
                   "results": results}, fh, indent=1)
    lines = ["# Weak top-k MIL control (same arch/features as skyline, TEST)", "",
             "| corpus | frame AP | frame ROC | within-ROC macro (n) | video ROC |",
             "|---|---:|---:|---:|---:|"]
    for corpus, r in results.items():
        a = r["agg"]
        lines.append("| %s | %.4f±%.3f | %.4f±%.3f | %.4f±%.3f (%d) | %.4f |" % (
            corpus, a["frame_ap"]["mean"], a["frame_ap"]["sd"],
            a["frame_roc"]["mean"], a["frame_roc"]["sd"],
            a["within_roc_macro"]["mean"], a["within_roc_macro"]["sd"],
            a["within_n"], a["video_roc_max"]["mean"]))
    with open(os.path.join(OUT_DIR, "weak_control.md"), "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
