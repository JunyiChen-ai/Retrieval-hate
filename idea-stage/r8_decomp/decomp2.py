#!/usr/bin/env python
"""R8-D2 -- protocol-matched follow-up to decomp.py.

decomp.py compared trajectory-averaged predictions against the FINAL epoch.
The deployed read-out (P1) instead selects the epoch by validation macro-F1, so
the FINAL-epoch baseline is not the one a candidate has to beat.  This script
adds an inner validation split so all three read-outs are paired on identical
runs:

  SEL   epoch = argmax_{e>=5} inner-dev macro-F1  (this is P1)
  FINAL last epoch
  TRAJ  mean probability over epochs 20..29
  TRAJW mean probability over the 5-epoch window centred on the SEL epoch

Outer: stratified 5-fold over train+val (TEST IS NEVER LOADED).
Inner: stratified 1/8 of the outer-train portion, used only to pick the epoch.
Best encoder per dataset only, 5 seeds.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from r4_harness import Head, load_split  # noqa: E402
from decomp import load_pool, macro_f1  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CELLS = {
    "HateMM": "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
    "MHC": "Qwen2.5-VL-7B-Instruct-LoRA_HF",
    "MHC_zh": "Qwen2.5-VL-7B-Instruct-LoRA_HF",
    "ImpliHateVid": "Qwen2.5-VL-7B-Instruct_HF",
}
NFOLD, SEEDS, EPOCHS, LR, BS, WARMUP = 5, [0, 1, 2, 3, 4], 30, 1e-4, 64, 5
TRAJ_LO, HALFW = 20, 2


def run(pool, tri, tei, seed):
    itr, idv = train_test_split(tri, test_size=0.125, random_state=seed,
                                stratify=pool["y"][tri].numpy())
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xi, Xt, Y = (pool["img"][itr].to(DEV), pool["txt"][itr].to(DEV), pool["y"][itr].to(DEV))
    Di, Dt = pool["img"][idv].to(DEV), pool["txt"][idv].to(DEV)
    ydv = pool["y"][idv].numpy()
    Ei, Et = pool["img"][tei].to(DEV), pool["txt"][tei].to(DEV)
    m = Head(Xi.shape[1], Xt.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    n = Y.shape[0]
    g = torch.Generator().manual_seed(seed)
    P, dvf = [], []
    for ep in range(EPOCHS):
        m.train()
        perm = torch.randperm(n, generator=g).to(DEV)
        for k in range(0, n, BS):
            idx = perm[k:k + BS]
            lo = m(Xi[idx], Xt[idx]).squeeze(1)
            loss = nn.functional.binary_cross_entropy_with_logits(lo, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            P.append(torch.sigmoid(m(Ei, Et).squeeze(1)).cpu().numpy())
            dvf.append(macro_f1(ydv, torch.sigmoid(m(Di, Dt).squeeze(1)).cpu().numpy())
                       if ep >= WARMUP else -1.0)
    sel = int(np.argmax(dvf))
    lo_, hi_ = max(WARMUP, sel - HALFW), min(EPOCHS - 1, sel + HALFW)
    return {
        "SEL": P[sel],
        "FINAL": P[-1],
        "TRAJ": np.mean(P[TRAJ_LO:], axis=0),
        "TRAJW": np.mean(P[lo_:hi_ + 1], axis=0),
        "sel_epoch": sel,
    }


def main():
    ARMS = ["SEL", "FINAL", "TRAJ", "TRAJW"]
    out = {"meta": {"nfold": NFOLD, "seeds": SEEDS, "arms": ARMS,
                    "test_touched": False}, "datasets": {}}
    for ds, tag in CELLS.items():
        t0 = time.time()
        pool = load_pool(ds, tag)
        y = pool["y"].numpy()
        n = len(y)
        skf = StratifiedKFold(NFOLD, shuffle=True, random_state=20260817)
        oof = {a: {s: np.zeros(n) for s in SEEDS} for a in ARMS}
        eps = []
        for tri, tei in skf.split(np.zeros(n), y):
            for s in SEEDS:
                r = run(pool, tri, tei, s)
                eps.append(r["sel_epoch"])
                for a in ARMS:
                    oof[a][s][tei] = r[a]
        row = {}
        for a in ARMS:
            f = [macro_f1(y, oof[a][s]) for s in SEEDS]
            rc = [float(roc_auc_score(y, oof[a][s])) for s in SEEDS]
            row[a] = {"macro_f1": float(np.mean(f)), "macro_f1_sd": float(np.std(f)),
                      "roc": float(np.mean(rc)), "per_seed": f}
        row["deltas_vs_SEL"] = {a: {
            "macro_f1": row[a]["macro_f1"] - row["SEL"]["macro_f1"],
            "roc": row[a]["roc"] - row["SEL"]["roc"],
            "seeds_positive": int(sum(np.array(row[a]["per_seed"])
                                      > np.array(row["SEL"]["per_seed"])))}
            for a in ARMS if a != "SEL"}
        row["sel_epoch_mean"] = float(np.mean(eps))
        row["sel_epoch_sd"] = float(np.std(eps))
        row["encoder"] = tag
        out["datasets"][ds] = row
        d = row["deltas_vs_SEL"]
        print(f"{ds} {time.time()-t0:.0f}s SEL={row['SEL']['macro_f1']:.4f} "
              f"FINAL{d['FINAL']['macro_f1']:+.4f} TRAJ{d['TRAJ']['macro_f1']:+.4f}"
              f"({d['TRAJ']['seeds_positive']}/5) TRAJW{d['TRAJW']['macro_f1']:+.4f}"
              f"({d['TRAJW']['seeds_positive']}/5) selep={row['sel_epoch_mean']:.1f}"
              f"+-{row['sel_epoch_sd']:.1f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "results2.json"), "w"), indent=2)
    print("WROTE results2.json")


if __name__ == "__main__":
    main()
