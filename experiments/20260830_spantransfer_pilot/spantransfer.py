#!/usr/bin/env python3
"""C5 pilot: LOO aux-span pretraining + rank-preserving weak adaptation.

Arms: loo_zero, loo_adapt, loo_naive, shuf_span (see PILOT_PLAN.md).
3 seeds x 4 corpora, TEST eval via the shared evaluator.

Output: runs/20260830_spantransfer_pilot/{results.json,results.md}
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(REPO, "experiments", "20260830_powa_within_diagnosis"))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
import skyline as S  # noqa: E402

GT_TRAIN = os.path.join(REPO, "runs", "20260830_powa_within_diagnosis",
                        "gt_train_diagnosis_only")
OUT_DIR = os.path.join(REPO, "runs", "20260830_spantransfer_pilot")
CORPORA = ("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg")
SEEDS = (234, 2025, 3407)
DIRS = S.FEATSETS["clip+vgg+bert"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAU, LAMBDA_RANK, N_PAIRS = 0.5, 1.0, 256
PRE_EPOCHS, ADAPT_EPOCHS, LR = 30, 15, 1e-3


def pack_spans(corpus, rng=None, shuffle=False):
    """Auxiliary span-supervised videos of one corpus (train split)."""
    with np.load(os.path.join(GT_TRAIN, corpus + "_train.npz")) as z:
        gt = {k: z[k] for k in z.files}
    out = []
    for v, y in sorted(gt.items()):
        y = np.asarray(y, dtype=np.float32)
        if shuffle and y.sum() > 0:
            y = np.roll(y, int(rng.integers(len(y))))
        try:
            x = S.load_feats(corpus, v, DIRS, len(y))
        except FileNotFoundError:
            continue
        out.append((torch.from_numpy(x), torch.from_numpy(y)))
    return out


def pack_weak(corpus):
    """Target-corpus train videos with video labels only."""
    labels = hdata.load_labels(corpus)
    out = []
    for v in hdata.load_split(corpus, "train"):
        y = labels.get(v)
        if y is None:
            continue
        try:
            T = int(np.load(os.path.join(REPO, "results/reproduction/features/"
                                         "vggish_1s", corpus, v + ".npy"),
                            mmap_mode="r").shape[0])
            x = S.load_feats(corpus, v, DIRS, T)
        except FileNotFoundError:
            continue
        out.append((torch.from_numpy(x), float(y)))
    return out


def pretrain(aux, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = S.TemporalConv(aux[0][0].shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pos = float(np.mean([t[1].mean().item() for t in aux]))
    w = torch.tensor((1 - pos) / max(pos, 1e-6), device=DEV)
    for ep in range(PRE_EPOCHS):
        model.train()
        for i in rng.permutation(len(aux)):
            x, y = aux[i]
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(x.to(DEV)), y.to(DEV), pos_weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def adapt(model, weak, seed, rank_term):
    teacher = [None] * len(weak)
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(weak):
            teacher[i] = model(x.to(DEV)).detach()
    rng = np.random.default_rng(seed + 7)
    opt = torch.optim.Adam(model.parameters(), lr=LR / 3)
    for ep in range(ADAPT_EPOCHS):
        model.train()
        for i in rng.permutation(len(weak)):
            x, y = weak[i]
            logits = model(x.to(DEV))
            k = max(1, len(logits) // 8)
            bag = torch.topk(logits, k).values.mean()
            loss = nn.functional.binary_cross_entropy_with_logits(
                bag, torch.tensor(y, device=DEV))
            if rank_term and len(logits) > 3:
                t = teacher[i]
                a = torch.from_numpy(rng.integers(len(t), size=N_PAIRS)).to(DEV)
                b = torch.from_numpy(rng.integers(len(t), size=N_PAIRS)).to(DEV)
                keep = (t[a] - t[b]).abs() > TAU
                if keep.any():
                    a, b = a[keep], b[keep]
                    sign = torch.sign(t[a] - t[b])
                    loss = loss + LAMBDA_RANK * nn.functional.margin_ranking_loss(
                        logits[a], logits[b], sign, margin=0.0)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def evaluate(model, corpus):
    gt = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    hate_ids = {v for v in gt if labels.get(v) == 1}
    scores = {}
    model.eval()
    with torch.no_grad():
        for v in gt:
            try:
                x = S.load_feats(corpus, v, DIRS, len(gt[v]))
            except FileNotFoundError:
                continue
            scores[v] = model(torch.from_numpy(x).to(DEV)).cpu().numpy().astype(float)
    m = evaluate_scores(scores, {v: gt[v] for v in scores}, hate_ids)
    hi = [auc for v, auc in m["per_video"]["per_video_auc"].items()
          if np.asarray(gt[v]).mean() > 0.6]
    return {"frame_ap": m["pr_auc"], "frame_roc": m["roc_auc"],
            "within_roc_macro": m["per_video"]["macro_auc"],
            "within_n": m["per_video"]["n_videos_both_classes"],
            "hipos_within_roc": float(np.mean(hi)) if hi else None,
            "hipos_n": len(hi),
            "video_roc_max": m["video_level"]["max_roc_auc"]}


def run(target, seed, arm):
    rng = np.random.default_rng(seed)
    shuffle = arm == "shuf_span"
    aux = []
    for c in CORPORA:
        if c == target:
            continue
        aux += pack_spans(c, rng=rng, shuffle=shuffle)
    model = pretrain(aux, seed)
    if arm in ("loo_adapt", "loo_naive"):
        weak = pack_weak(target)
        model = adapt(model, weak, seed, rank_term=(arm == "loo_adapt"))
    return evaluate(model, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="loo_zero,loo_adapt,loo_naive,shuf_span")
    ap.add_argument("--corpora", default=",".join(CORPORA))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "results.json")
    results = json.load(open(path))["results"] if os.path.exists(path) else {}
    for target in args.corpora.split(","):
        for arm in args.arms.split(","):
            key = "%s/%s" % (target, arm)
            if key in results:
                continue
            per_seed = [run(target, s, arm) for s in SEEDS]
            agg = {}
            for k in per_seed[0]:
                vals = [p[k] for p in per_seed if p[k] is not None]
                if k in ("within_n", "hipos_n"):
                    agg[k] = per_seed[0][k]
                else:
                    agg[k] = {"mean": float(np.mean(vals)),
                              "sd": float(np.std(vals))} if vals else None
            results[key] = {"seeds": per_seed, "agg": agg}
            with open(path, "w") as fh:
                json.dump({"seeds": list(SEEDS), "tau": TAU,
                           "lambda_rank": LAMBDA_RANK, "results": results}, fh,
                          indent=1)
            print(key, json.dumps(agg), flush=True)
    lines = ["# C5 span-transfer pilot (TEST, 3 seeds)", "",
             "| target/arm | frame AP | frame ROC | within-ROC macro (n) | "
             "hi-pos (n) |", "|---|---:|---:|---:|---:|"]
    for key in sorted(results):
        a = results[key]["agg"]
        hp = a["hipos_within_roc"]
        lines.append("| %s | %.4f±%.3f | %.4f±%.3f | %.4f±%.3f (%d) | %s (%d) |" % (
            key, a["frame_ap"]["mean"], a["frame_ap"]["sd"],
            a["frame_roc"]["mean"], a["frame_roc"]["sd"],
            a["within_roc_macro"]["mean"], a["within_roc_macro"]["sd"],
            a["within_n"],
            ("%.4f" % hp["mean"]) if hp else "n/a", a["hipos_n"]))
    with open(os.path.join(OUT_DIR, "results.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
