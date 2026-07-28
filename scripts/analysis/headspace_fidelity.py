#!/usr/bin/env python
"""headspace_fidelity.py -- GATE-FID for the HEADSPACE-TRANSFER pregate.

Compares the CPU-minted DEPLOYED-CONFIGURATION proxy head against the banked GPU floor
(job 13241 for HateMM) on the **DEV** retrieval curve.  The dev split is not the test
split, so this costs no test touch; the floor trainlogs are mixed files, so the reader
below is a HARD FILTER that can only ever emit a `Val_Retrieval` line -- `Test_Retrieval`
lines are discarded at the point of read and never enter any data structure.

Record: refine-logs/HEADSPACE_TRANSFER_PREGATE.md §2.1 / §4.1.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))
from headspace_mint import det1_assert, runtime_block, sha256_of  # noqa: E402

VAL_RE = re.compile(
    r"^Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: [\d.]+ macroR: [\d.]+ "
    r"acc: ([\d.]+) roc: ([\d.]+)")

FLOOR = {
    "hatemm": ("slurm/logs/enc3s_HateMM_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF_"
               "seed{seed}_13241.trainlog"),
    "zh": ("slurm/logs/enc3s_MHC_zh_Qwen2.5-VL-7B-Instruct-LoRA_HF_"
           "seed{seed}_13150.trainlog"),
}


def floor_dev_curve(pattern, seed):
    """DEV-ONLY reader.  Any line that is not a Val_Retrieval line is dropped before
    anything is parsed from it, so no test metric can be produced by this function."""
    path = os.path.join(REPO, pattern.format(seed=seed))
    out = {}
    with open(path) as f:
        for line in f:
            m = VAL_RE.match(line.strip())
            if m is None:
                continue                      # <- Test_Retrieval lines die here
            out[int(m.group(1))] = {"acc": float(m.group(3)),
                                    "macroF1": float(m.group(2)),
                                    "roc": float(m.group(4))}
    assert out, "no Val_Retrieval lines parsed from {}".format(path)
    return out, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(FLOOR))
    ap.add_argument("--mintdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    det1_assert("8")
    seeds = [int(s) for s in a.seeds.split(",")]

    rows, curves = [], {}
    for s in seeds:
        z = np.load(os.path.join(a.mintdir, "mint_{}_s{}_ffull.npz".format(a.dataset, s)),
                    allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        proxy = {r["epoch"]: r for r in meta["eval_curve"] if r["split"] == "dev"}
        fl, fpath = floor_dev_curve(FLOOR[a.dataset], s)
        ep = sorted(set(proxy) & set(fl))
        d = np.array([proxy[e]["acc"] - fl[e]["acc"] for e in ep])
        rows.append({
            "seed": s, "floor_log": fpath, "n_epochs_compared": len(ep),
            "floor_dev_acc_ep29": round(fl[29]["acc"], 4),
            "proxy_dev_acc_ep29": round(proxy[29]["acc"], 4),
            "delta_ep29": round(proxy[29]["acc"] - fl[29]["acc"], 4),
            "floor_dev_mF1_ep29": round(fl[29]["macroF1"], 4),
            "proxy_dev_mF1_ep29": round(proxy[29]["macroF1"], 4),
            "mean_abs_delta_over_30_epochs": round(float(np.abs(d).mean()), 4),
            "max_abs_delta_over_30_epochs": round(float(np.abs(d).max()), 4),
            "corr_over_30_epochs": round(float(np.corrcoef(
                [proxy[e]["acc"] for e in ep], [fl[e]["acc"] for e in ep])[0, 1]), 4),
            "mint_secs": meta["secs"], "n_dev": meta["n_dev"],
            "n_train": meta["n_train"], "head_dim": meta["head_dim"],
        })
        curves[str(s)] = {"proxy": [round(proxy[e]["acc"], 4) for e in ep],
                          "floor": [round(fl[e]["acc"], 4) for e in ep]}

    fmean = float(np.mean([r["floor_dev_acc_ep29"] for r in rows]))
    pmean = float(np.mean([r["proxy_dev_acc_ep29"] for r in rows]))
    B_fid = abs(pmean - fmean)
    gate = {
        "floor_dev_acc_ep29_3seedmean": round(fmean, 4),
        "proxy_dev_acc_ep29_3seedmean": round(pmean, 4),
        "delta_3seedmean": round(pmean - fmean, 4),
        "B_fid_abs_3seedmean": round(B_fid, 4),
        "per_seed_abs_delta_max": round(
            max(abs(r["delta_ep29"]) for r in rows), 4),
        "mean_abs_delta_over_30_epochs_3seedmean": round(
            float(np.mean([r["mean_abs_delta_over_30_epochs"] for r in rows])), 4),
        "raw_effect_under_test": 0.0255,
        "STOP_RULE_TRIGGERED": bool(B_fid >= 0.0255),
        "one_dev_item": round(1.0 / rows[0]["n_dev"], 4),
    }
    OUT = {"meta": {"script_sha256": sha256_of(os.path.abspath(__file__)),
                    "dataset": a.dataset, "seeds": seeds,
                    "test_contact": "NONE -- Val_Retrieval-only hard filter; "
                                    "no Test_Retrieval line is parsed or stored",
                    "runtime": runtime_block()},
           "per_seed": rows, "gate": gate, "curves_dev_acc": curves}
    tmp = a.out + ".tmp"
    json.dump(OUT, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
