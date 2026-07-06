#!/usr/bin/env python
"""P8 probe gate (CPU, before training).

Per dataset, TRAIN split, no trained head: leave-one-out kNN vote (cosine, k=20) over the
[l2n(img) | l2n(text)] representation the head consumes, for A(floor raw chunk-mean) /
B(summary) / C(first-70-tok). Gate: B LOO-kNN accuracy >= A on that dataset. Reports A/B/C
acc + macro-F1 at k=20 and the k-curve; writes p8_out/probe_gate.json.
"""
import argparse
import json
import os

import numpy as np
import torch
import faiss
from sklearn.metrics import f1_score

MODEL = "openai_clip-vit-large-patch14-336_HF"
ROOT = "/data/jehc223/RGCL"


def rep(img, text):
    """[l2n(img) | l2n(text)] then row-normalise for cosine kNN. float32 numpy."""
    im = torch.nn.functional.normalize(img.float(), dim=1)
    tx = torch.nn.functional.normalize(text.float(), dim=1)
    x = torch.cat([im, tx], dim=1).numpy().astype("float32")
    faiss.normalize_L2(x)
    return x


def loo_knn(x, labels, topk=20):
    """Leave-one-out similarity-weighted arithmetic kNN vote acc + macro-F1."""
    n = x.shape[0]
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    sims, idx = index.search(x, topk + 1)   # +1 to drop self
    lab = labels.astype("float64")
    preds = np.zeros(n, dtype=int)
    for i in range(n):
        nbr, w = [], []
        for j, s in zip(idx[i], sims[i]):
            if j == i:
                continue
            nbr.append(lab[j]); w.append(max(s, 0.0))
            if len(nbr) == topk:
                break
        w = np.asarray(w); nbr = np.asarray(nbr)
        vote = (w * nbr).sum() / w.sum() if w.sum() > 0 else nbr.mean()
        preds[i] = int(vote >= 0.5)
    acc = float((preds == labels).mean())
    mf1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    return round(acc, 4), round(mf1, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="MHC,MHC_zh,HateMM")
    ap.add_argument("--out", default="scripts/analysis/p8_out/probe_gate.json")
    args = ap.parse_args()

    result = {}
    for ds in args.datasets.split(","):
        ds = ds.strip()
        ds_dir = os.path.join(ROOT, "data/CLIP_Embedding", ds)
        floor = torch.load(os.path.join(ds_dir, "train_{}.pt".format(MODEL)), map_location="cpu")
        img = floor["img_feats"]
        labels = floor["labels"].long().numpy()
        texts = {"A": floor["text_feats"]}
        for tag, cond in (("p8sum", "B"), ("p8trunc", "C")):
            p = os.path.join(ds_dir, "train_{}_HF.pt".format(tag))
            texts[cond] = torch.load(p, map_location="cpu")["text_feats"]

        conds = {}
        for cond in ("A", "B", "C"):
            x = rep(img, texts[cond])
            acc20, mf20 = loo_knn(x, labels, 20)
            kc = {k: loo_knn(x, labels, k)[0] for k in (1, 5, 10)}
            conds[cond] = {"acc_k20": acc20, "macro_f1_k20": mf20, "acc_k1_5_10": kc}
        gate_open = conds["B"]["acc_k20"] >= conds["A"]["acc_k20"]
        beats_trunc = conds["B"]["acc_k20"] >= conds["C"]["acc_k20"]
        result[ds] = {"n": int(len(labels)), "pos": int(labels.sum()),
                      "conds": conds, "gate_open": bool(gate_open),
                      "B_beats_C_probe": bool(beats_trunc)}
        print("\n===== P8 probe :: {} (n={}, pos={}) =====".format(ds, len(labels), int(labels.sum())))
        for cond in ("A", "B", "C"):
            c = conds[cond]
            print("  {} ({}): acc@k20={} macroF1={} | k1/5/10={}".format(
                cond, {"A": "floor raw chunk-mean", "B": "summary", "C": "first-70-tok"}[cond],
                c["acc_k20"], c["macro_f1_k20"], c["acc_k1_5_10"]))
        print("  GATE (B>=A @k20): {}   B>=C(probe): {}".format(
            "OPEN" if gate_open else "CLOSED", beats_trunc))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("\n[out] wrote", args.out)


if __name__ == "__main__":
    main()
