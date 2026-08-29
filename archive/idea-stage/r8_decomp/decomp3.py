#!/usr/bin/env python
"""R8-D3 -- does a ranking gain convert into macro-F1, and where is it lost?

decomp2.py found that trajectory-averaged predictions beat the P1 (dev-selected
epoch) read-out on ROC in 4/4 datasets while producing ~0 macro-F1 at the fixed
0.5 threshold.  Two readings are possible:
  (a) the ranking gain is real but thrown away by the operating point, or
  (b) the ranking gain sits in a region of the ROC curve that no threshold can
      convert.
This script separates them by evaluating every arm at four operating points:

  T05     fixed 0.5 (the deployed read-out)
  TDEV    threshold chosen per fold on the inner dev split by macro-F1
  TPRIOR  quantile threshold matching the training-split positive rate
  TORACLE global threshold fitted on the pooled out-of-fold predictions
          (upper bound; still no test data -- this is train+val CV)

Arms: SEL (P1: dev-selected epoch), FINAL, TRAJ (mean prob over epochs 20-29).
Outer 5-fold over train+val, 5 seeds, best encoder per dataset.
TEST IS NEVER LOADED.
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
from r4_harness import Head  # noqa: E402
from decomp import load_pool  # noqa: E402
from decomp2 import CELLS, NFOLD, SEEDS, EPOCHS, LR, BS, WARMUP, TRAJ_LO  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ARMS = ["SEL", "FINAL", "TRAJ"]
GRID = np.arange(0.02, 0.99, 0.01)


def mf1(y, b):
    return float(f1_score(y, b, average="macro"))


def best_thr(y, p):
    s = [mf1(y, (p >= t).astype(int)) for t in GRID]
    return float(GRID[int(np.argmax(s))]), float(np.max(s))


def run(pool, tri, tei, seed):
    itr, idv = train_test_split(tri, test_size=0.125, random_state=seed,
                                stratify=pool["y"][tri].numpy())
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xi, Xt, Y = pool["img"][itr].to(DEV), pool["txt"][itr].to(DEV), pool["y"][itr].to(DEV)
    Di, Dt = pool["img"][idv].to(DEV), pool["txt"][idv].to(DEV)
    ydv = pool["y"][idv].numpy()
    Ei, Et = pool["img"][tei].to(DEV), pool["txt"][tei].to(DEV)
    m = Head(Xi.shape[1], Xt.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    n = Y.shape[0]
    g = torch.Generator().manual_seed(seed)
    Pt, Pd, dvf = [], [], []
    for ep in range(EPOCHS):
        m.train()
        perm = torch.randperm(n, generator=g).to(DEV)
        for k in range(0, n, BS):
            idx = perm[k:k + BS]
            loss = nn.functional.binary_cross_entropy_with_logits(
                m(Xi[idx], Xt[idx]).squeeze(1), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            Pt.append(torch.sigmoid(m(Ei, Et).squeeze(1)).cpu().numpy())
            pd = torch.sigmoid(m(Di, Dt).squeeze(1)).cpu().numpy()
        Pd.append(pd)
        dvf.append(mf1(ydv, (pd >= 0.5).astype(int)) if ep >= WARMUP else -1.0)
    sel = int(np.argmax(dvf))
    out = {}
    for a, (pt, pd) in (("SEL", (Pt[sel], Pd[sel])), ("FINAL", (Pt[-1], Pd[-1])),
                        ("TRAJ", (np.mean(Pt[TRAJ_LO:], 0), np.mean(Pd[TRAJ_LO:], 0)))):
        tdev, _ = best_thr(ydv, pd)
        prior = float(pool["y"][itr].numpy().mean())
        tprior = float(np.quantile(pt, 1.0 - prior))
        out[a] = {"p": pt, "tdev": tdev, "tprior": tprior}
    return out


def main():
    res = {"meta": {"nfold": NFOLD, "seeds": SEEDS, "arms": ARMS,
                    "test_touched": False}, "datasets": {}}
    for ds, tag in CELLS.items():
        t0 = time.time()
        pool = load_pool(ds, tag)
        y = pool["y"].numpy()
        n = len(y)
        skf = StratifiedKFold(NFOLD, shuffle=True, random_state=20260817)
        P = {a: {s: np.zeros(n) for s in SEEDS} for a in ARMS}
        B = {a: {r: {s: np.zeros(n, dtype=int) for s in SEEDS}
                 for r in ("T05", "TDEV", "TPRIOR")} for a in ARMS}
        for tri, tei in skf.split(np.zeros(n), y):
            for s in SEEDS:
                r = run(pool, tri, tei, s)
                for a in ARMS:
                    P[a][s][tei] = r[a]["p"]
                    B[a]["T05"][s][tei] = (r[a]["p"] >= 0.5).astype(int)
                    B[a]["TDEV"][s][tei] = (r[a]["p"] >= r[a]["tdev"]).astype(int)
                    B[a]["TPRIOR"][s][tei] = (r[a]["p"] >= r[a]["tprior"]).astype(int)
        row = {}
        for a in ARMS:
            cell = {"roc": float(np.mean([roc_auc_score(y, P[a][s]) for s in SEEDS]))}
            for rule in ("T05", "TDEV", "TPRIOR"):
                v = [mf1(y, B[a][rule][s]) for s in SEEDS]
                cell[rule] = {"macro_f1": float(np.mean(v)), "sd": float(np.std(v)),
                              "per_seed": v}
            orc = [best_thr(y, P[a][s]) for s in SEEDS]
            cell["TORACLE"] = {"macro_f1": float(np.mean([o[1] for o in orc])),
                               "thr_mean": float(np.mean([o[0] for o in orc])),
                               "thr_sd": float(np.std([o[0] for o in orc])),
                               "per_seed": [o[1] for o in orc]}
            row[a] = cell
        row["encoder"] = tag
        res["datasets"][ds] = row
        print(f"{ds} {time.time()-t0:.0f}s " + " | ".join(
            f"{a} roc {row[a]['roc']:.4f} @.5 {row[a]['T05']['macro_f1']:.4f} "
            f"dev {row[a]['TDEV']['macro_f1']:.4f} prior {row[a]['TPRIOR']['macro_f1']:.4f} "
            f"orc {row[a]['TORACLE']['macro_f1']:.4f}" for a in ARMS), flush=True)
    json.dump(res, open(os.path.join(HERE, "results3.json"), "w"), indent=2)
    print("WROTE results3.json")


if __name__ == "__main__":
    main()
