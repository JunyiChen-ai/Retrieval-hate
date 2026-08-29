#!/usr/bin/env python
"""R8-D1 -- zero-test-contact decomposition of the ensemble gain.

Question this answers, before any round-8 candidate is chosen:
    the recorded +1.3..+5.3 macro-F1 "three-encoder ensemble" gain -- is it
    MEMBER COMPLEMENTARITY (different encoders see different things) or is it
    VARIANCE REDUCTION (averaging away an unstable head optimisation)?

Instrument: stratified 5-fold CV over train+val (TEST IS NEVER LOADED).
Epoch selection is removed as a confound: every read-out is either the final
epoch or a fixed-window trajectory average, so no held-out selection is done.

Arms, all computed from the SAME set of head runs:
  SINGLE   mean over seeds of a single head's fold macro-F1
  SEEDENS  probability-average of 3 heads, SAME encoder, seeds 0,1,2
  CROSSENS probability-average of 1 head per encoder (all encoders), seed-matched,
           then averaged over seeds
  ALLENS   probability-average of every head (encoders x 3 seeds)
Read-outs: FINAL (last epoch) and TRAJ (mean prob over epochs 20-29).

No test file is opened anywhere in this script.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from r4_harness import Head, load_split  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLIP = "openai_clip-vit-large-patch14-336_HF"
QWEN = "Qwen2.5-VL-7B-Instruct_HF"

CELLS = {
    "HateMM": [CLIP, QWEN, "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF"],
    "MHC": [CLIP, QWEN, "Qwen2.5-VL-7B-Instruct-LoRA_HF"],
    "MHC_zh": [CLIP, QWEN, "Qwen2.5-VL-7B-Instruct-LoRA_HF"],
    "ImpliHateVid": [CLIP, QWEN],
}
NFOLD = 5
SEEDS = [0, 1, 2]
EPOCHS, LR, BS = 30, 1e-4, 64
TRAJ_LO = 20


def macro_f1(y, p):
    return float(f1_score(y, (p >= 0.5).astype(int), average="macro"))


def load_pool(ds, tag):
    tr = load_split(ds, tag, "train")
    va = load_split(ds, tag, "val")
    return {
        "ids": list(tr["ids"]) + list(va["ids"]),
        "img": torch.cat([tr["img"], va["img"]], 0),
        "txt": torch.cat([tr["txt"], va["txt"]], 0),
        "y": torch.cat([tr["y"], va["y"]], 0),
    }


def train_fold(pool, tri, tei, seed):
    """Train on tri, return (final_prob, traj_prob) on tei. No selection."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xi = pool["img"][tri].to(DEV)
    Xt = pool["txt"][tri].to(DEV)
    Y = pool["y"][tri].to(DEV)
    Ei = pool["img"][tei].to(DEV)
    Et = pool["txt"][tei].to(DEV)
    m = Head(Xi.shape[1], Xt.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    n = Y.shape[0]
    g = torch.Generator().manual_seed(seed)
    acc = []
    last = None
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
            p = torch.sigmoid(m(Ei, Et).squeeze(1)).cpu().numpy()
        last = p
        if ep >= TRAJ_LO:
            acc.append(p)
    return last, np.mean(acc, axis=0)


def main():
    out = {"meta": {"nfold": NFOLD, "seeds": SEEDS, "epochs": EPOCHS,
                    "traj_window": [TRAJ_LO, EPOCHS - 1],
                    "test_touched": False}, "datasets": {}}
    for ds, tags in CELLS.items():
        t0 = time.time()
        pools = {tag: load_pool(ds, tag) for tag in tags}
        ref = pools[tags[0]]
        y = ref["y"].numpy()
        n = len(y)
        for tag in tags[1:]:
            assert pools[tag]["ids"] == ref["ids"], f"{ds}/{tag} id order mismatch"
        skf = StratifiedKFold(NFOLD, shuffle=True, random_state=20260817)
        # preds[readout][tag][seed] -> full-length OOF prob vector
        preds = {r: {tag: {s: np.zeros(n) for s in SEEDS} for tag in tags}
                 for r in ("FINAL", "TRAJ")}
        for tri, tei in skf.split(np.zeros(n), y):
            for tag in tags:
                for s in SEEDS:
                    fin, traj = train_fold(pools[tag], tri, tei, s)
                    preds["FINAL"][tag][s][tei] = fin
                    preds["TRAJ"][tag][s][tei] = traj
        res = {}
        for r in ("FINAL", "TRAJ"):
            P = preds[r]
            row = {"per_encoder_single": {}, "per_encoder_seedens": {}}
            for tag in tags:
                row["per_encoder_single"][tag] = {
                    "macro_f1": float(np.mean([macro_f1(y, P[tag][s]) for s in SEEDS])),
                    "roc": float(np.mean([roc_auc_score(y, P[tag][s]) for s in SEEDS])),
                    "macro_f1_per_seed": [macro_f1(y, P[tag][s]) for s in SEEDS],
                }
                ens = np.mean([P[tag][s] for s in SEEDS], axis=0)
                row["per_encoder_seedens"][tag] = {
                    "macro_f1": macro_f1(y, ens), "roc": float(roc_auc_score(y, ens))}
            cross = [macro_f1(y, np.mean([P[tag][s] for tag in tags], axis=0)) for s in SEEDS]
            crossr = [float(roc_auc_score(y, np.mean([P[tag][s] for tag in tags], axis=0)))
                      for s in SEEDS]
            allp = np.mean([P[tag][s] for tag in tags for s in SEEDS], axis=0)
            row["crossens"] = {"macro_f1": float(np.mean(cross)),
                               "roc": float(np.mean(crossr)),
                               "macro_f1_per_seed": cross}
            row["allens"] = {"macro_f1": macro_f1(y, allp),
                             "roc": float(roc_auc_score(y, allp))}
            best = max(tags, key=lambda t: row["per_encoder_single"][t]["macro_f1"])
            b = row["per_encoder_single"][best]["macro_f1"]
            row["best_encoder"] = best
            row["gain_seedens"] = row["per_encoder_seedens"][best]["macro_f1"] - b
            row["gain_crossens"] = row["crossens"]["macro_f1"] - b
            row["gain_allens"] = row["allens"]["macro_f1"] - b
            res[r] = row
        res["traj_minus_final_single"] = {
            tag: res["TRAJ"]["per_encoder_single"][tag]["macro_f1"]
                 - res["FINAL"]["per_encoder_single"][tag]["macro_f1"] for tag in tags}
        out["datasets"][ds] = res
        print(f"{ds} done in {time.time()-t0:.0f}s  best={res['FINAL']['best_encoder']}  "
              f"seedens={res['FINAL']['gain_seedens']:+.4f}  "
              f"crossens={res['FINAL']['gain_crossens']:+.4f}  "
              f"allens={res['FINAL']['gain_allens']:+.4f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=2)
    print("WROTE", os.path.join(HERE, "results.json"))


if __name__ == "__main__":
    main()
