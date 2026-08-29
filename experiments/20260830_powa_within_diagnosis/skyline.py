#!/usr/bin/env python3
"""Supervised skyline probe: how far can frozen 1 Hz features localize?

Trains on VAL frame labels (the only frame-labelled non-test split), evaluates
on TEST. Two model sizes (linear per-frame probe; small dilated temporal conv)
x two feature sets (CLIP only; CLIP+VGGish+BERT). If even the skyline stays
near chance on within-video ordering, the features lack the signal; if the
skyline is high, the weak-supervision objective is the gap.

Output: runs/20260830_powa_within_diagnosis/skyline.json / skyline.md
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402

FEAT_ROOT = os.path.join(REPO, "results", "reproduction", "features")
OUT_DIR = os.path.join(REPO, "runs", "20260830_powa_within_diagnosis")
CORPORA = ("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg")
FEATSETS = {"clip": ("clip_b16_1fps",),
            "clip+vgg+bert": ("clip_b16_1fps", "vggish_1s", "bert_sentence_1fps")}
SEED = 234
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_feats(corpus, vid, dirs, T):
    parts = []
    for d in dirs:
        f = np.load(os.path.join(FEAT_ROOT, d, corpus, vid + ".npy"))
        parts.append(np.asarray(f, dtype=np.float32))
    L = min([p.shape[0] for p in parts] + [T])
    x = np.concatenate([p[:L] for p in parts], axis=1)
    if L < T:
        x = np.concatenate([x, np.repeat(x[-1:], T - L, axis=0)], axis=0)
    return x


class TemporalConv(nn.Module):
    def __init__(self, dim, hid=256):
        super().__init__()
        self.inp = nn.Linear(dim, hid)
        self.convs = nn.ModuleList([
            nn.Conv1d(hid, hid, 3, padding=d, dilation=d) for d in (1, 2, 4, 8)])
        self.out = nn.Linear(hid, 1)

    def forward(self, x):          # x: (T, D)
        h = torch.relu(self.inp(x)).T.unsqueeze(0)   # (1, H, T)
        for c in self.convs:
            h = torch.relu(c(h)) + h
        return self.out(h.squeeze(0).T).squeeze(-1)  # (T,)


class Linear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.out = nn.Linear(dim, 1)

    def forward(self, x):
        return self.out(x).squeeze(-1)


def run(corpus, featset, arch, epochs=30, lr=1e-3):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dirs = FEATSETS[featset]
    gt_tr = hdata.gt_arrays(corpus, "val")
    gt_te = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    hate_ids = {v for v in gt_te if labels.get(v) == 1}

    def pack(gt):
        out = []
        for v, y in gt.items():
            y = np.asarray(y, dtype=np.float32)
            try:
                x = load_feats(corpus, v, dirs, len(y))
            except FileNotFoundError:
                continue
            out.append((v, torch.from_numpy(x), torch.from_numpy(y)))
        return out

    train, test = pack(gt_tr), pack(gt_te)
    dim = train[0][1].shape[1]
    model = (TemporalConv(dim) if arch == "tconv" else Linear(dim)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos = float(np.mean([t[2].mean().item() for t in train]))
    w = torch.tensor((1 - pos) / max(pos, 1e-6), device=DEV)
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(len(train))
        for i in order:
            _, x, y = train[i]
            x, y = x.to(DEV), y.to(DEV)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(x), y, pos_weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    scores = {}
    with torch.no_grad():
        for v, x, y in test:
            scores[v] = model(x.to(DEV)).cpu().numpy().astype(float)
    m = evaluate_scores(scores, {v: gt_te[v] for v in scores}, hate_ids)
    return {"n_train_videos": len(train), "n_test_videos": len(test),
            "frame_ap": m["pr_auc"], "frame_roc": m["roc_auc"],
            "within_roc_macro": m["per_video"]["macro_auc"],
            "within_n": m["per_video"]["n_videos_both_classes"]}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for corpus in CORPORA:
        for featset in FEATSETS:
            for arch in ("linear", "tconv"):
                key = f"{corpus}/{featset}/{arch}"
                results[key] = run(corpus, featset, arch)
                print(key, json.dumps(results[key]))
    with open(os.path.join(OUT_DIR, "skyline.json"), "w") as fh:
        json.dump({"note": "trained on VAL frame labels, evaluated on TEST",
                   "seed": SEED, "results": results}, fh, indent=1)
    lines = ["# Supervised skyline (val-frame-trained, TEST eval)", "",
             "| corpus | features | arch | frame AP | frame ROC | within-ROC macro (n) |",
             "|---|---|---|---:|---:|---:|"]
    for key, r in results.items():
        c, f, a = key.split("/")
        lines.append("| %s | %s | %s | %.4f | %.4f | %.4f (%d) |" % (
            c, f, a, r["frame_ap"], r["frame_roc"],
            r["within_roc_macro"], r["within_n"]))
    with open(os.path.join(OUT_DIR, "skyline.md"), "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
