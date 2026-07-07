#!/usr/bin/env python
"""P9 C3-knn read-out: kNN vote over the SFT'd LMM embeddings (our retrieval-memory decision).

Consumes the LoRA-adapted feature cache produced by generate_VideoMLLM_embedding_lora_HF.py
(tag e.g. p9c3_mhc_s0): {ids, img_feats[N,3584], text_feats[N,3584], labels} for train/dev/test.
Builds the kNN memory from TRAIN and votes the dev (and, when --test, test) queries with a
similarity-weighted arithmetic vote over [l2n(img)|l2n(text)] — the same decision mechanism as
our frozen-head system, now on the LoRA'd backbone's features. Reports acc / macro-F1 so the
C3-knn read-out can be compared to the in-LMM MLP head (C3-mlp) and the frozen-head floor.

CPU-only, read-only. Prints + writes JSON.
"""
import argparse
import json
import os

import numpy as np
import torch
import faiss
from sklearn.metrics import f1_score

ROOT = "/data/jehc223/RGCL"


def load_cache(ds, split_out, tag):
    p = os.path.join(ROOT, "data/CLIP_Embedding", ds, "{}_{}.pt".format(split_out, tag))
    d = torch.load(p, map_location="cpu")
    ids = [i for sub in d["ids"] for i in sub]
    img = torch.nn.functional.normalize(d["img_feats"].float(), dim=1)
    txt = torch.nn.functional.normalize(d["text_feats"].float(), dim=1)
    rep = torch.cat([img, txt], dim=1).numpy().astype("float32")
    faiss.normalize_L2(rep)
    return ids, rep, d["labels"].long().numpy()


def knn_vote(mem_rep, mem_lab, q_rep, topk=20):
    index = faiss.IndexFlatIP(mem_rep.shape[1])
    index.add(mem_rep)
    sims, idx = index.search(q_rep, topk)
    lab = mem_lab.astype("float64")
    preds = np.zeros(q_rep.shape[0], dtype=int)
    for i in range(q_rep.shape[0]):
        w = np.clip(sims[i], 0, None)
        nb = lab[idx[i]]
        vote = (w * nb).sum() / w.sum() if w.sum() > 0 else nb.mean()
        preds[i] = int(vote >= 0.5)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", required=True, help="feature cache tag, e.g. p9c3_mhc_s0")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--test", action="store_true", help="also vote the test split")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tr_ids, tr_rep, tr_lab = load_cache(args.dataset, "train", args.tag)
    va_ids, va_rep, va_lab = load_cache(args.dataset, "dev_seen", args.tag)

    def evl(q_rep, q_lab):
        pred = knn_vote(tr_rep, tr_lab, q_rep, args.topk)
        return (round(float((pred == q_lab).mean()), 4),
                round(float(f1_score(q_lab, pred, average="macro", zero_division=0)), 4))

    res = {"dataset": args.dataset, "tag": args.tag, "topk": args.topk,
           "mem_n": len(tr_ids)}
    res["dev"] = dict(zip(("acc", "macro_f1"), evl(va_rep, va_lab)))
    print("[{}] C3-knn DEV: acc={} macroF1={} (mem={} train)".format(
        args.dataset, res["dev"]["acc"], res["dev"]["macro_f1"], len(tr_ids)))
    if args.test:
        te_ids, te_rep, te_lab = load_cache(args.dataset, "test_seen", args.tag)
        res["test"] = dict(zip(("acc", "macro_f1"), evl(te_rep, te_lab)))
        print("[{}] C3-knn TEST: acc={} macroF1={}".format(
            args.dataset, res["test"]["acc"], res["test"]["macro_f1"]))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print("[out] wrote", args.out)


if __name__ == "__main__":
    main()
