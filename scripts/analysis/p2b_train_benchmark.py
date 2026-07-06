#!/usr/bin/env python
"""P2b — build the TRAIN-side selectivity benchmark (CPU, labels only, no test contact).

Pre-registration: research-wiki/EXP_p2b_stronger_judge.md. For seed-0 EN + ZH heads,
leave-one-out retrieval over the TRAIN memory (self excluded), gate the bottom-25%-margin
train queries (the train analogue of the gated test set), take their top-20 LOO neighbours,
split into correct-vote (neighbour label == query gold) / wrong-vote (!=), and emit a
balanced ~1.5k-pair subsample per language. Judge configs are then measured on this set by
selectivity lift = drop_rate(wrong-vote) - drop_rate(correct-vote).

Reuses the exact vote code from p2_rerank_eval (bit-identical to the training-log floor).
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis"))

from p2_rerank_eval import (  # noqa: E402  (frozen P2 harness helpers)
    MODEL, TOPK, ALPHA, DEFER_RATE, ckpt_path, augment, sim_vote, retrieve)
from eval_cross_dataset import project_split, build_head            # noqa: E402
from data_loader.dataset import (                                   # noqa: E402
    load_feats_from_CLIP, load_archive_feats_split, resolve_archive_path)
from easydict import EasyDict                                       # noqa: E402

TARGET_PER_GROUP = 750   # -> ~1500 balanced pairs / language


def build_train_bench(ds, seed=0):
    device = "cpu"
    clip_path = os.path.join(ROOT, "data", "CLIP_Embedding")
    train, _, _ = load_feats_from_CLIP(clip_path, ds, MODEL[ds])
    model = build_head(train[1].shape[1], train[2].shape[1], EasyDict(
        eval_dataset=ds, num_layers=3, proj_dim=1024, map_dim=1024,
        fusion_mode="align", dropout=[0.2, 0.4, 0.1], batch_norm=False))
    model.load_state_dict(torch.load(ckpt_path(ds, seed), map_location="cpu"))
    model.eval()
    tr_ids, tr_emb, tr_lab = project_split(model, train, device)
    tr_lab = np.asarray(tr_lab, dtype=int)
    arc = load_archive_feats_split(
        resolve_archive_path("auto", os.path.join(ROOT, "data"), ds, "train"), tr_ids)
    keys = augment(torch.tensor(tr_emb), arc)

    # LOO retrieval: top-21 then drop the self match (rank 0, sim==1).
    D, I = retrieve(keys, tr_lab, tr_ids, keys, TOPK + 1)
    margins = []
    per_q = []
    for i in range(len(tr_ids)):
        nb = [(int(I[i, r]), float(D[i, r])) for r in range(I.shape[1]) if int(I[i, r]) != i][:TOPK]
        labs = [tr_lab[j] for j, _ in nb]
        sims = [s for _, s in nb]
        v = sim_vote(labs, sims)
        margins.append(abs(v))
        per_q.append(nb)
    margins = np.asarray(margins)
    thr = np.quantile(margins, DEFER_RATE)          # bottom-25% margin
    gated = np.where(margins < thr)[0]

    correct, wrong = [], []
    for i in gated:
        gold = int(tr_lab[i])
        for j, sim in per_q[i]:
            rec = dict(query_id=tr_ids[i], neighbor_id=tr_ids[j],
                       query_label=gold, neighbor_label=int(tr_lab[j]),
                       correct_vote=int(tr_lab[j] == gold), sim=round(sim, 4))
            (correct if rec["correct_vote"] else wrong).append(rec)
    rng = random.Random(0)
    rng.shuffle(correct); rng.shuffle(wrong)
    sub = correct[:TARGET_PER_GROUP] + wrong[:TARGET_PER_GROUP]
    rng.shuffle(sub)
    stats = dict(dataset=ds, seed=seed, n_train=len(tr_ids),
                 n_gated=int(len(gated)), gate_thr=float(thr),
                 pairs_correct=len(correct), pairs_wrong=len(wrong),
                 subsample=len(sub),
                 sub_correct=sum(r["correct_vote"] for r in sub),
                 sub_wrong=sum(1 - r["correct_vote"] for r in sub))
    return sub, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="MHC,MHC_zh")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "scripts/analysis/p2_out"))
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)
    allstats = {}
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        sub, stats = build_train_bench(ds)
        p = os.path.join(args.out_dir, "trainbench_{}.jsonl".format(ds))
        with open(p, "w") as f:
            for r in sub:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        allstats[ds] = stats
        print("[{}] gated {}/{} train | pairs corr/wrong {}/{} | subsample {} "
              "(corr {} wrong {}) -> {}".format(
                  ds, stats["n_gated"], stats["n_train"],
                  stats["pairs_correct"], stats["pairs_wrong"], stats["subsample"],
                  stats["sub_correct"], stats["sub_wrong"], p), flush=True)
    with open(os.path.join(args.out_dir, "trainbench_stats.json"), "w") as f:
        json.dump(allstats, f, indent=1)


if __name__ == "__main__":
    main()
