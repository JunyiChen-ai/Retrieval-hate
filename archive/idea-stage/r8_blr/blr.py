#!/usr/bin/env python
"""R8-1 BLR -- boundary-localised ranking objective for the frozen-feature head.

Frozen design: idea-stage/R8_BLR_FREEZE.md (committed BEFORE this file was run).

Four arms, identical head / optimiser / schedule / epoch-selection, differing only
in the training objective:

  A0     plain BCE on every training item (the deployed baseline objective).
  BALBCE the anchored pointwise term alone, i.e. balanced BCE over the same sampled
         positive / negative pairs with no ranking term.  Isolates the implicit class
         balancing that a pairwise objective performs for free (2512.01766, 2607.09832).
  PAIRG  global pairwise logistic over uniformly sampled positive x negative pairs,
         plus the anchored pointwise term (the fixed decision threshold treated as a
         virtual item every positive must outrank and every negative be outranked by;
         with tau = 0 that term is exactly balanced BCE):
             mean softplus(-(s_p - s_n)) + 0.5 * mean[softplus(-s_p) + softplus(s_n)]
  PAIRL  the candidate.  Identical to PAIRG except the pairwise term is averaged over
         only the hardest Q fraction of the sampled pairs in each step -- a two-way
         partial-AUC surrogate that concentrates ranking pressure on pairs local to
         the decision boundary.
  RANDL  control for PAIRL.  Identical to PAIRL except the retained subset of pairs is
         drawn uniformly at random instead of by hardness, with the same pair count.
         Isolates "boundary localisation" from "fewer / noisier pairs".

Read-outs, both from the same runs:
  P1 (primary)       epoch = argmax_{e>=warmup} val macro-F1 @0.5, ties -> earliest;
                     test macro-F1 @0.5
  P2 (corroboration) last epoch, test macro-F1 @0.5
Test labels are read only for the final metric; nothing is selected on them.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from r4_harness import Head, load_split  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CELLS = {
    "HateMM": "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
    "MHC": "Qwen2.5-VL-7B-Instruct-LoRA_HF",
    "MHC_zh": "Qwen2.5-VL-7B-Instruct-LoRA_HF",
    "ImpliHateVid": "Qwen2.5-VL-7B-Instruct_HF",
}
ARMS = ["A0", "BALBCE", "PAIRG", "PAIRL", "RANDL"]
EPOCHS, LR, BS, WARMUP = 30, 1e-4, 64, 5
NPAIR = 1024          # pairs sampled per optimisation step
Q = 0.25              # retained fraction for PAIRL / RANDL


def mf1(y, p):
    return float(f1_score(y, (p >= 0.5).astype(int), average="macro"))


def train(tr, va, te, seed, arm, quiet=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xi, Xt, Y = tr["img"].to(DEV), tr["txt"].to(DEV), tr["y"].to(DEV)
    m = Head(Xi.shape[1], Xt.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    n = Y.shape[0]
    pos = torch.nonzero(Y > 0.5).squeeze(1)
    neg = torch.nonzero(Y < 0.5).squeeze(1)
    g = torch.Generator().manual_seed(seed)
    packs = {k: (s["img"].to(DEV), s["txt"].to(DEV), s["y"].numpy())
             for k, s in (("val", va), ("test", te))}
    keep = max(1, int(round(Q * NPAIR)))
    best, hist, losses = (-1.0, None), [], []
    for ep in range(EPOCHS):
        m.train()
        if arm == "A0":
            perm = torch.randperm(n, generator=g).to(DEV)
            for k in range(0, n, BS):
                idx = perm[k:k + BS]
                loss = nn.functional.binary_cross_entropy_with_logits(
                    m(Xi[idx], Xt[idx]).squeeze(1), Y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
        else:
            for _ in range(max(1, n // BS)):
                pi = pos[torch.randint(len(pos), (NPAIR,), generator=g)].to(DEV)
                ni = neg[torch.randint(len(neg), (NPAIR,), generator=g)].to(DEV)
                sp = m(Xi[pi], Xt[pi]).squeeze(1)
                sn = m(Xi[ni], Xt[ni]).squeeze(1)
                per_pair = nn.functional.softplus(-(sp - sn))
                if arm == "BALBCE":
                    rank = torch.zeros((), device=sp.device)
                elif arm == "PAIRG":
                    rank = per_pair.mean()
                elif arm == "PAIRL":
                    rank = torch.topk(per_pair, keep).values.mean()
                elif arm == "RANDL":
                    sel = torch.randperm(NPAIR, generator=g).to(DEV)[:keep]
                    rank = per_pair[sel].mean()
                else:
                    raise ValueError(arm)
                anchor = 0.5 * (nn.functional.softplus(-sp).mean()
                                + nn.functional.softplus(sn).mean())
                loss = rank + anchor
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
        m.eval()
        with torch.no_grad():
            pr = {k: torch.sigmoid(m(a, b).squeeze(1)).cpu().numpy()
                  for k, (a, b, _) in packs.items()}
        hist.append(pr)
        if ep >= WARMUP:
            f = mf1(packs["val"][2], pr["val"])
            if f > best[0]:
                best = (f, ep)
    if quiet:
        return {"loss_first": losses[0], "loss_last": losses[-1],
                "n_steps": len(losses), "nan": bool(np.isnan(losses[-1]))}
    yt = packs["test"][2]
    sel, last = hist[best[1]], hist[-1]
    return {
        "sel_epoch": best[1],
        "P1": mf1(yt, sel["test"]), "P1_roc": float(roc_auc_score(yt, sel["test"])),
        "P2": mf1(yt, last["test"]), "P2_roc": float(roc_auc_score(yt, last["test"])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(range(200, 230)))
    ap.add_argument("--datasets", nargs="+", default=list(CELLS))
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    ap.add_argument("--smoke", action="store_true",
                    help="timing / NaN check only; prints no metric of any arm")
    a = ap.parse_args()
    out = {"meta": {"seeds": a.seeds, "arms": a.arms, "npair": NPAIR, "q": Q,
                    "epochs": EPOCHS, "lr": LR, "bs": BS, "warmup": WARMUP,
                    "freeze": "idea-stage/R8_BLR_FREEZE.md"}, "runs": {}}
    for ds in a.datasets:
        tag = CELLS[ds]
        tr, va, te = (load_split(ds, tag, s) for s in ("train", "val", "test"))
        out["runs"][ds] = {}
        for arm in a.arms:
            rows = []
            t0 = time.time()
            for s in a.seeds:
                rows.append(train(tr, va, te, s, arm, quiet=a.smoke))
            out["runs"][ds][arm] = rows
            if a.smoke:
                print(f"SMOKE {ds:14s} {arm:6s} {(time.time()-t0)/len(a.seeds):6.1f}s/run "
                      f"steps={rows[0]['n_steps']} loss {rows[0]['loss_first']:.4f}"
                      f"->{rows[0]['loss_last']:.4f} nan={rows[0]['nan']}", flush=True)
            else:
                print(f"{ds:14s} {arm:6s} {len(a.seeds)} seeds "
                      f"{(time.time()-t0):6.0f}s", flush=True)
    if not a.smoke:
        json.dump(out, open(a.out, "w"), indent=2)
        print("WROTE", a.out)


if __name__ == "__main__":
    main()
