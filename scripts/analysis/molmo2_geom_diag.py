#!/usr/bin/env python
"""
molmo2_geom_diag.py -- RAW (pre-head) encoder-geometry diagnostics for the Molmo2 swap.

The accuracy probe answers "does the swap convert". This answers the mechanism question
F89 actually posed: **did the embedding geometry itself change?** It runs on the frozen
encoder outputs directly -- no head, no training, no selection -- so it is a property of
the encoder alone.

Per arm x per stream view, on HateMM (bank = train, query = test):

  * raw kNN vote acc / mF1 -- the deployed top-20 rank-weighted signed-cosine operator
    (mechfix_ops.deployed_vote) applied straight to encoder output.
  * top-1 cosine saturation + mean top-20 cosine -- cone collapse.
  * participation ratio and leading-eigenvalue variance share of the key covariance --
    how many directions the representation actually uses.
  * length-organisation rho = spearman(query transcript volume, median volume of its
    top-20 bank neighbours) -- is retrieval organised by transcript length rather than
    by content (the nuisance axis F89 named).

CPU only. Read-only. Selects nothing.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import torch  # noqa: E402
import mechfix_ops as M  # noqa: E402

FEAT = os.path.join(REPO, "data/CLIP_Embedding/{ds}/{split}_{tag}.pt")
GT = os.path.join(REPO, "data/gt/{ds}/{split}.jsonl")

ARMS = {
    "A_molmo2": "Molmo2-8B_HF",
    "B_lora_curric": "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
    "C_frozen_qwen": "Qwen2.5-VL-7B-Instruct_HF",
}


def load_feats(ds, split, tag):
    d = torch.load(FEAT.format(ds=ds, split=split, tag=tag), weights_only=False)
    ids = [i for sub in d["ids"] for i in sub]
    return (
        ids,
        d["img_feats"].float().numpy(),
        d["text_feats"].float().numpy(),
        d["labels"].numpy().astype(int),
    )


def load_gt_text(ds, split):
    out = {}
    with open(GT.format(ds=ds, split=split)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out[str(o["id"])] = "" if o.get("text") is None else str(o["text"])
    return out


def views(img, txt):
    """Stream views of the raw encoder output. Each is L2-normalised before use."""
    return {
        "img": img,
        "text": txt,
        "concat": np.concatenate([img, txt], axis=1),
        "hadamard": img * txt,  # the deployed fusion's raw-space analogue
    }


def cone_stats(bank, query):
    """Top-1 / mean-top-20 cosine saturation, on L2-normalised keys."""
    B = M._norm32(bank.astype("float32"))
    Q = M._norm32(query.astype("float32"))
    sims = Q @ B.T
    part = np.sort(sims, axis=1)[:, ::-1][:, : M.TOPK]
    return float(part[:, 0].mean()), float(part.mean())


def spectrum_stats(keys):
    """Participation ratio (effective #directions) and leading variance share."""
    X = M._norm32(keys.astype("float32")).astype("float64")
    X = X - X.mean(0, keepdims=True)
    ev = np.linalg.svd(X, compute_uv=False) ** 2
    ev = ev[ev > 0]
    pr = float((ev.sum() ** 2) / (ev**2).sum())  # participation ratio
    return round(pr, 3), round(float(ev[0] / ev.sum()), 4)


def length_org(bank, query, tr_vol, te_vol):
    B = M._norm32(bank.astype("float32"))
    Q = M._norm32(query.astype("float32"))
    sims = Q @ B.T
    I = np.argsort(-sims, axis=1)[:, : M.TOPK]
    return round(float(spearmanr(te_vol, np.median(tr_vol[I], axis=1))[0]), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HateMM")
    ap.add_argument("--out", default=os.path.join(REPO, "scripts/analysis/molmo2_geom_diag_OUT.json"))
    args = ap.parse_args()
    ds = args.dataset

    gt_tr, gt_te = load_gt_text(ds, "train"), load_gt_text(ds, "test")
    OUT = {"meta": {"dataset": ds, "cpu_only": True, "gpu_jobs": 0,
                    "note": "RAW pre-head encoder geometry; no head, no training, no selection",
                    "vol_mode": "words", "topk": M.TOPK}, "arms": {}}

    for arm, tag in ARMS.items():
        try:
            tr_ids, tr_img, tr_txt, tr_lab = load_feats(ds, "train", tag)
            te_ids, te_img, te_txt, te_lab = load_feats(ds, "test_seen", tag)
        except FileNotFoundError as e:
            print("[SKIP] {} ({}): {}".format(arm, tag, e), flush=True)
            continue

        tr_vol = np.array([len(gt_tr[i].split()) for i in tr_ids], dtype="float64")
        te_vol = np.array([len(gt_te[i].split()) for i in te_ids], dtype="float64")

        arm_out = {"tag": tag, "dim": int(tr_img.shape[1]), "n_train": len(tr_ids),
                   "n_test": len(te_ids), "views": {}}

        TRV, TEV = views(tr_img, tr_txt), views(te_img, te_txt)
        for v in TRV:
            bank, query = TRV[v], TEV[v]
            vote = M.deployed_vote(bank, tr_lab, query)
            pred = (np.asarray(vote) >= 0).astype(int) if np.ndim(vote) == 1 else np.asarray(vote[1])
            top1, top20 = cone_stats(bank, query)
            pr, lead = spectrum_stats(bank)
            arm_out["views"][v] = {
                "raw_knn_acc": round(float(M.acc(te_lab, pred)), 4),
                "raw_knn_mF1": round(float(M.macro_f1(te_lab, pred)), 4),
                "top1_cos": round(top1, 4),
                "mean_top20_cos": round(top20, 4),
                "participation_ratio": pr,
                "leading_var_share": lead,
                "length_organisation_rho": length_org(bank, query, tr_vol, te_vol),
            }
            print("[{}] {:9s} acc {:.4f} mF1 {:.4f} | top1cos {:.4f} top20 {:.4f} | PR {:8.3f} "
                  "lead {:.4f} | rho {:+.4f}".format(
                      arm, v, arm_out["views"][v]["raw_knn_acc"], arm_out["views"][v]["raw_knn_mF1"],
                      top1, top20, pr, lead, arm_out["views"][v]["length_organisation_rho"]),
                  flush=True)
        OUT["arms"][arm] = arm_out

    json.dump(OUT, open(args.out, "w"), indent=2)
    print("\nwrote {}".format(args.out))


if __name__ == "__main__":
    main()
