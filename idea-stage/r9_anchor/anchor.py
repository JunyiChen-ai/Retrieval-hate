"""R9-1 ANCHOR-INT. Frozen design: idea-stage/R9_PILOT_FREEZE.md (commit 20ab02b).

Runs the full alpha x seed grid and dumps raw per-seed probabilities. No verdict quantity is
computed here; analyze.py renders the frozen decision rule exactly once.

--smoke prints wall-clock, step count and a NaN flag only, never an arm metric.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from r4_harness import load_split, train_head  # noqa: E402

FROZEN = "Qwen2.5-VL-7B-Instruct_HF"
ADAPTED = "Qwen2.5-VL-7B-Instruct-LoRA_HF"
ALPHAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
CELLS = {
    "HateMM": list(range(400, 415)),
    "MHC": list(range(400, 430)),
    "MHC_zh": list(range(400, 430)),
}


def l2(x):
    return torch.nn.functional.normalize(x, p=2, dim=1)


def mix(a, b, alpha):
    return {
        "ids": a["ids"],
        "img": l2((1.0 - alpha) * a["img"] + alpha * b["img"]),
        "txt": l2((1.0 - alpha) * a["txt"] + alpha * b["txt"]),
        "y": a["y"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()

    rows = []
    for ds, seeds in CELLS.items():
        z0 = {s: load_split(ds, FROZEN, s) for s in ("train", "val", "test")}
        z1 = {s: load_split(ds, ADAPTED, s) for s in ("train", "val", "test")}
        for s in ("train", "val", "test"):
            assert z0[s]["ids"] == z1[s]["ids"], f"{ds}/{s} id order mismatch"
        if a.smoke:
            seeds = seeds[:1]
            alphas = [0.5]
        else:
            alphas = ALPHAS
        for alpha in alphas:
            m = {s: mix(z0[s], z1[s], alpha) for s in ("train", "val", "test")}
            for seed in seeds:
                t0 = time.time()
                r = train_head(m["train"], m["val"], m["test"], seed)
                dt = time.time() - t0
                nan = bool(np.isnan(r["test_prob"]).any() or np.isnan(r["val_prob"]).any())
                if a.smoke:
                    print(f"SMOKE {ds} alpha={alpha} seed={seed} wall={dt:.1f}s "
                          f"epochs=30 nan={nan}", flush=True)
                    continue
                rows.append({
                    "dataset": ds, "alpha": alpha, "seed": seed, "epoch": r["epoch"],
                    "val_macro_f1": r["val_macro_f1"], "wall_s": round(dt, 2), "nan": nan,
                    "test_prob": [round(float(v), 6) for v in r["test_prob"]],
                })
                print(f"RUN {ds} alpha={alpha} seed={seed} wall={dt:.1f}s nan={nan}", flush=True)
    if a.smoke:
        return
    ids = {ds: load_split(ds, FROZEN, "test")["ids"] for ds in CELLS}
    labels = {ds: load_split(ds, FROZEN, "test")["y"].numpy().tolist() for ds in CELLS}
    json.dump({"alphas": ALPHAS, "cells": {k: v for k, v in CELLS.items()},
               "ids": ids, "labels": labels, "rows": rows},
              open(os.path.join(HERE, a.out), "w"))
    print("WROTE", a.out, len(rows), "rows")


if __name__ == "__main__":
    main()
