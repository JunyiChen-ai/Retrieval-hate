#!/usr/bin/env python3
"""Recon probe: does within-video ordering transfer across corpora?

Train the skyline TemporalConv on corpus A's TRAIN span rasterization (built
for diagnosis), evaluate within-hate macro ROC on corpus B's TEST. Full 4x4
matrix (diagonal = skyline upper bound, off-diagonal = transfer). CPU-safe
fallback if GPU busy. Purpose: rank the cross-corpus-span-transfer candidate.
"""
import json
import os
import sys

import numpy as np
import torch

REPO = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(REPO, "experiments", "20260830_powa_within_diagnosis"))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
import skyline as S  # noqa: E402

GT_TRAIN = os.path.join(REPO, "runs", "20260830_powa_within_diagnosis",
                        "gt_train_diagnosis_only")
OUT = os.path.join(REPO, "runs", "20260830_xneg_mil_pilot")
CORPORA = ("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg")
DIRS = S.FEATSETS["clip+vgg+bert"]
DEV = "cpu"
SEED = 234


def pack_train(corpus):
    with np.load(os.path.join(GT_TRAIN, corpus + "_train.npz")) as z:
        gt = {k: z[k] for k in z.files}
    out = []
    for v, y in gt.items():
        y = np.asarray(y, dtype=np.float32)
        try:
            x = S.load_feats(corpus, v, DIRS, len(y))
        except FileNotFoundError:
            continue
        out.append((torch.from_numpy(x), torch.from_numpy(y)))
    return out


def train_model(train, epochs=30, lr=1e-3):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = S.TemporalConv(train[0][0].shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos = float(np.mean([t[1].mean().item() for t in train]))
    w = torch.tensor((1 - pos) / max(pos, 1e-6), device=DEV)
    for ep in range(epochs):
        model.train()
        for i in np.random.permutation(len(train)):
            x, y = train[i]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(x.to(DEV)), y.to(DEV), pos_weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_on(model, corpus):
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
            scores[v] = model(torch.from_numpy(x).to(DEV)).numpy().astype(float)
    m = evaluate_scores(scores, {v: gt[v] for v in scores}, hate_ids)
    return m["per_video"]["macro_auc"]


def main():
    os.makedirs(OUT, exist_ok=True)
    mat = {}
    for src in CORPORA:
        train = pack_train(src)
        model = train_model(train)
        for tgt in CORPORA:
            mat["%s->%s" % (src, tgt)] = eval_on(model, tgt)
            print("%s->%s within-ROC %.4f" % (src, tgt, mat["%s->%s" % (src, tgt)]),
                  flush=True)
    with open(os.path.join(OUT, "probe_cross_corpus.json"), "w") as fh:
        json.dump(mat, fh, indent=1)
    lines = ["# Cross-corpus span-transfer probe (within-ROC macro, TEST)", "",
             "| train\\eval | " + " | ".join(CORPORA) + " |",
             "|---|" + "---:|" * len(CORPORA)]
    for src in CORPORA:
        lines.append("| %s | " % src + " | ".join(
            "%.4f" % mat["%s->%s" % (src, t)] for t in CORPORA) + " |")
    with open(os.path.join(OUT, "probe_cross_corpus.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
