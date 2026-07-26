"""MNTP recon: $0 CPU raw-key-space kNN dev screen, causal vs bidir.

Purpose: calibrate the DYNAMIC RANGE of the cheapest possible observable that
distinguishes "MNTP repair working" from "F72 crater persisting", BEFORE any
GPU is spent. Uses ONLY banked feature caches + DEV labels. No head training,
no test read, no GPU.

Operator = the deployed one (src/utils/metrics.py:229-284, use_sim=True,
majority_voting='arithmetic'): L2-normalised keys, top-20 by cosine over the
own-TRAIN memory, labels mapped to +/-1, multiplied by cosine, rank-weighted
[20..1], normalised by sum of weights; decision sigmoid(v)>=0.5 <=> v>=0.
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

ROOT = "/data/jehc223/RGCL"
TOPK = 20


def load_cache(ds, split, tag):
    p = os.path.join(ROOT, "data/CLIP_Embedding", ds, "{}_{}.pt".format(split, tag))
    d = torch.load(p, map_location="cpu", weights_only=False)
    ids = d["ids"]
    if len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    return ids, d["img_feats"].float(), d["text_feats"].float(), d["labels"]


def l2(x):
    return torch.nn.functional.normalize(x, p=2, dim=1)


def knn_vote(Q, M, m_labels, q_labels, topk=TOPK):
    """Deployed rank-weighted signed-cosine vote. Q/M already L2-normed."""
    sims = Q @ M.T                                   # [nq, nm]
    vals, idx = torch.topk(sims, k=topk, dim=1)      # cosine desc
    w = np.arange(1, topk + 1)[::-1].astype(np.float64)   # [20..1]
    lab = m_labels.numpy()[idx.numpy()] * 2 - 1      # map {0,1} -> {-1,+1}
    signed = lab * vals.numpy()                      # times similarity
    vote = (signed * w).sum(1) / w.sum()
    pred = (vote >= 0).astype(int)                   # sigmoid(v)>=0.5 <=> v>=0
    y = q_labels.numpy()
    return (
        float((pred == y).mean()),
        float(f1_score(y, pred, average="macro", zero_division=0)),
        float(roc_auc_score(y, vote)),
    )


def run(ds, causal_tag, bidir_tag):
    out = {}
    for arm, tag in (("causal", causal_tag), ("bidir", bidir_tag)):
        _, tr_i, tr_t, tr_y = load_cache(ds, "train", tag)
        _, dv_i, dv_t, dv_y = load_cache(ds, "dev_seen", tag)
        streams = {
            "img": (l2(tr_i), l2(dv_i)),
            "text": (l2(tr_t), l2(dv_t)),
            "concat": (l2(torch.cat([l2(tr_i), l2(tr_t)], 1)),
                       l2(torch.cat([l2(dv_i), l2(dv_t)], 1))),
        }
        for name, (M, Q) in streams.items():
            out[(arm, name)] = knn_vote(Q, M, tr_y, dv_y)
        # feature-space drift vs causal (id-matched, same order)
        if arm == "bidir":
            _, ctr_i, ctr_t, _ = load_cache(ds, "train", causal_tag)
            _, cdv_i, cdv_t, _ = load_cache(ds, "dev_seen", causal_tag)
            out["drift"] = {
                "train_img_cos": float((l2(tr_i) * l2(ctr_i)).sum(1).mean()),
                "train_text_cos": float((l2(tr_t) * l2(ctr_t)).sum(1).mean()),
                "dev_img_cos": float((l2(dv_i) * l2(cdv_i)).sum(1).mean()),
                "dev_text_cos": float((l2(dv_t) * l2(cdv_t)).sum(1).mean()),
            }
    return out


if __name__ == "__main__":
    cells = [
        ("HateMM", "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
         "Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir_HF"),
        ("MHC_zh", "Qwen2.5-VL-7B-Instruct-LoRA_HF",
         "Qwen2.5-VL-7B-Instruct-LoRA-bidir_HF"),
    ]
    blob = {}
    for ds, ct, bt in cells:
        r = run(ds, ct, bt)
        print("\n=== {} (DEV only, raw untrained key space, no head) ===".format(ds))
        print("{:8s} {:8s} {:>8s} {:>8s} {:>8s}".format("stream", "arm", "acc", "mF1", "roc"))
        for name in ("img", "text", "concat"):
            for arm in ("causal", "bidir"):
                a, f, ro = r[(arm, name)]
                print("{:8s} {:8s} {:8.4f} {:8.4f} {:8.4f}".format(name, arm, a, f, ro))
            da = r[("bidir", name)][0] - r[("causal", name)][0]
            df = r[("bidir", name)][1] - r[("causal", name)][1]
            print("{:8s} {:8s} {:+8.4f} {:+8.4f}".format(name, "DELTA", da, df))
        print("feature drift bidir-vs-causal (mean per-item cosine):", r["drift"])
        blob[ds] = {"{}|{}".format(k[0], k[1]) if isinstance(k, tuple) else k: v
                    for k, v in r.items()}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "mntp_rawkey_devscreen_OUT.json"), "w") as fh:
        json.dump(blob, fh, indent=2)
    print("\nwrote mntp_rawkey_devscreen_OUT.json")
