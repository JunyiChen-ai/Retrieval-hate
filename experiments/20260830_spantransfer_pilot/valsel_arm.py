#!/usr/bin/env python3
"""Amendment A1 arm: loo_adapt with val-selected adaptation depth.

Identical pretraining and adaptation to spantransfer.py's loo_adapt, but
checkpoints are taken at epochs {0,1,2,4,8,15}; the deployed depth is the one
with the best VALIDATION within-ROC macro (val frame labels are sanctioned for
checkpoint selection). Epoch 0 = pure zero-shot transfer.

Output: appends target/loo_adapt_valsel keys into runs/.../results.json and
writes valsel.md with the chosen depth per corpus/seed.
"""
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(REPO, "experiments", "20260830_powa_within_diagnosis"))
sys.path.insert(0, os.path.join(REPO, "experiments", "20260830_spantransfer_pilot"))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
import skyline as S  # noqa: E402
import spantransfer as ST  # noqa: E402

OUT_DIR = ST.OUT_DIR
CKPT_EPOCHS = (0, 1, 2, 4, 8, 15)


def eval_split(model, corpus, split):
    gt = hdata.gt_arrays(corpus, split)
    labels = hdata.load_labels(corpus)
    hate_ids = {v for v in gt if labels.get(v) == 1}
    scores = {}
    model.eval()
    with torch.no_grad():
        for v in gt:
            try:
                x = S.load_feats(corpus, v, ST.DIRS, len(gt[v]))
            except FileNotFoundError:
                continue
            scores[v] = model(torch.from_numpy(x).to(ST.DEV)).cpu().numpy().astype(float)
    m = evaluate_scores(scores, {v: gt[v] for v in scores}, hate_ids)
    return m


def run(target, seed):
    rng = np.random.default_rng(seed)
    aux = []
    for c in ST.CORPORA:
        if c == target:
            continue
        aux += ST.pack_spans(c, rng=rng, shuffle=False)
    model = ST.pretrain(aux, seed)
    weak = ST.pack_weak(target)

    teacher = [None] * len(weak)
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(weak):
            teacher[i] = model(x.to(ST.DEV)).detach()
    arng = np.random.default_rng(seed + 7)
    opt = torch.optim.Adam(model.parameters(), lr=ST.LR / 3)
    ckpts = {0: copy.deepcopy(model.state_dict())}
    for ep in range(1, max(CKPT_EPOCHS) + 1):
        model.train()
        for i in arng.permutation(len(weak)):
            x, y = weak[i]
            logits = model(x.to(ST.DEV))
            k = max(1, len(logits) // 8)
            bag = torch.topk(logits, k).values.mean()
            loss = nn.functional.binary_cross_entropy_with_logits(
                bag, torch.tensor(y, device=ST.DEV))
            if len(logits) > 3:
                t = teacher[i]
                a = torch.from_numpy(arng.integers(len(t), size=ST.N_PAIRS)).to(ST.DEV)
                b = torch.from_numpy(arng.integers(len(t), size=ST.N_PAIRS)).to(ST.DEV)
                keep = (t[a] - t[b]).abs() > ST.TAU
                if keep.any():
                    a, b = a[keep], b[keep]
                    sign = torch.sign(t[a] - t[b])
                    loss = loss + ST.LAMBDA_RANK * nn.functional.margin_ranking_loss(
                        logits[a], logits[b], sign, margin=0.0)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in CKPT_EPOCHS:
            ckpts[ep] = copy.deepcopy(model.state_dict())

    best_ep, best_val = None, -1.0
    for ep in CKPT_EPOCHS:
        model.load_state_dict(ckpts[ep])
        val = eval_split(model, target, "val")
        macro = val["per_video"]["macro_auc"]
        macro = -0.5 if macro is None else macro
        if macro > best_val:
            best_val, best_ep = macro, ep
    model.load_state_dict(ckpts[best_ep])
    m = eval_split(model, target, "test")
    gt = hdata.gt_arrays(target, "test")
    hi = [auc for v, auc in m["per_video"]["per_video_auc"].items()
          if np.asarray(gt[v]).mean() > 0.6]
    return {"frame_ap": m["pr_auc"], "frame_roc": m["roc_auc"],
            "within_roc_macro": m["per_video"]["macro_auc"],
            "within_n": m["per_video"]["n_videos_both_classes"],
            "hipos_within_roc": float(np.mean(hi)) if hi else None,
            "hipos_n": len(hi),
            "video_roc_max": m["video_level"]["max_roc_auc"],
            "selected_epoch": best_ep, "val_within": best_val}


def main():
    path = os.path.join(OUT_DIR, "results.json")
    blob = json.load(open(path))
    results = blob["results"]
    sel_lines = ["# A1 val-selected adaptation depth", "",
                 "| target | seed | selected epoch | val within-ROC |",
                 "|---|---:|---:|---:|"]
    for target in ST.CORPORA:
        key = "%s/loo_adapt_valsel" % target
        if key in results:
            continue
        per_seed = [run(target, s) for s in ST.SEEDS]
        for s, p in zip(ST.SEEDS, per_seed):
            sel_lines.append("| %s | %d | %d | %.4f |" % (
                target, s, p["selected_epoch"], p["val_within"]))
        agg = {}
        for k in per_seed[0]:
            vals = [p[k] for p in per_seed if p[k] is not None]
            if k in ("within_n", "hipos_n"):
                agg[k] = per_seed[0][k]
            elif k == "selected_epoch":
                agg[k] = [p[k] for p in per_seed]
            elif k == "val_within":
                continue
            else:
                agg[k] = {"mean": float(np.mean(vals)),
                          "sd": float(np.std(vals))} if vals else None
        results[key] = {"seeds": per_seed, "agg": agg}
        with open(path, "w") as fh:
            json.dump(blob, fh, indent=1)
        print(key, json.dumps(agg), flush=True)
    with open(os.path.join(OUT_DIR, "valsel.md"), "w") as fh:
        fh.write("\n".join(sel_lines))


if __name__ == "__main__":
    main()
