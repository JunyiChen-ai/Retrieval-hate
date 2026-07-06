#!/usr/bin/env python
"""P7 train-side gate (EXP_p7_score_fusion §0.3) — NO test contact.

LOO kNN vote share on TRAIN with the seed-0 archive-kNN winner head (reuses
p2_rerank_eval's exact head/vote/augment infra), then measures whether the MLLM
channel (bin = P1 verdict, dens = P3 density) corrects the vote's LOO errors
net-of-damage under the two frozen fusion rules R1/R2. Also reports the
decorrelation diagnostics (AUC, point-biserial, corr with vote share).

Promotion bar: some rule achieves net>0 AND net/n_errors >= 0.15 (train LOO).
100% CPU.
"""
import json
import os
import sys

import numpy as np
import torch
import faiss
from sklearn.metrics import roc_auc_score

ROOT = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis"))
from p2_rerank_eval import (  # noqa: E402
    build_head, project_split, ckpt_path, MODEL, sim_vote, vote_pred, augment,
    TOPK, sigmoid, load_feats_from_CLIP, load_archive_feats_split,
    resolve_archive_path)
from easydict import EasyDict  # noqa: E402

DEFER_RATE = 0.25
BOOST = 0.25


def load_channels(ds):
    """id -> channel value in [0,1] for bin (P1 verdict) and dens (P3 density)."""
    hv = json.load(open(os.path.join(
        ROOT, "scripts/analysis/p1_out/harmful_verdicts.json")))[ds]["v2"]
    binc = {vid: (1.0 if v["verdict"] == "HARMFUL" else 0.0)
            for vid, v in hv.items()}
    dens = {}
    p = os.path.join(ROOT, "data/MLLM_scores", ds, "train_segscoreK4_qwen.jsonl")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            dens[str(o["id"])] = float(np.mean(o["scores"])) / 3.0
    return binc, dens


def loo_votes(ds, seed=0):
    device = "cpu"
    train, dev, test = load_feats_from_CLIP(
        os.path.join(ROOT, "data", "CLIP_Embedding"), ds, MODEL[ds])
    model = build_head(train[1].shape[1], train[2].shape[1], EasyDict(
        eval_dataset=ds, num_layers=3, proj_dim=1024, map_dim=1024,
        fusion_mode="align", dropout=[0.2, 0.4, 0.1], batch_norm=False))
    model.load_state_dict(torch.load(ckpt_path(ds, seed), map_location="cpu"))
    model.eval()
    tr_ids, tr_emb, tr_lab = project_split(model, train, device)
    arc = load_archive_feats_split(
        resolve_archive_path("auto", os.path.join(ROOT, "data"), ds, "train"),
        tr_ids)
    mem = augment(torch.tensor(tr_emb), arc)          # [N, d] float32
    tr_lab = np.asarray(tr_lab, dtype=int)
    m = mem.copy()
    faiss.normalize_L2(m)
    index = faiss.IndexFlatIP(m.shape[1])
    index.add(m)
    D, I = index.search(m, TOPK + 1)                  # self included at rank 0
    votes = np.zeros(len(tr_ids))
    for i in range(len(tr_ids)):
        labs, sims = [], []
        for r in range(I.shape[1]):
            j = int(I[i, r])
            if j == i:
                continue                              # drop self
            labs.append(int(tr_lab[j]))
            sims.append(float(D[i, r]))
            if len(labs) >= TOPK:
                break
        votes[i] = sim_vote(labs, sims)
    preds = (sigmoid(votes) >= 0.5).astype(int)
    return list(tr_ids), tr_lab, votes, preds


def rankpct(x):
    """percentile rank in [0,1], mid-rank for ties."""
    from scipy.stats import rankdata
    return (rankdata(x, method="average") - 1) / max(len(x) - 1, 1)


def fuse_R1(sv, c, fpred, ok):
    """rank-average, operating point = floor's #positives (over the ok subset)."""
    pred = fpred.copy()
    idx = np.where(ok)[0]
    rv = rankpct(sv[idx])
    rc = rankpct(c[idx])
    fused = (rv + rc) / 2.0
    n_pos = int(fpred[idx].sum())
    # top n_pos by fused (tie-break by sv)
    order = np.lexsort((sv[idx], fused))[::-1]        # desc by fused, then sv
    newp = np.zeros(len(idx), dtype=int)
    newp[order[:n_pos]] = 1
    pred[idx] = newp
    return pred


def fuse_R2(sv, c, fpred, votes, tau, ok):
    """band-limited veto-boost: only |vote|<tau samples adjusted."""
    pred = fpred.copy()
    band = (np.abs(votes) < tau) & ok
    fs = sv + BOOST * (c - 0.5)
    pred[band] = (fs[band] >= 0.5).astype(int)
    return pred


def net_flips(fpred, newpred, gold, ok):
    idx = np.where(ok)[0]
    f = fpred[idx]
    n = newpred[idx]
    g = gold[idx]
    err = f != g
    corrected = int(np.sum(err & (n == g)))
    damaged = int(np.sum(~err & (n != g)))
    nerr = int(np.sum(err))
    net = corrected - damaged
    rate = net / nerr if nerr else 0.0
    return corrected, damaged, net, nerr, rate


def gate(ds):
    tr_ids, gold, votes, fpred = loo_votes(ds, 0)
    sv = sigmoid(votes)
    tau = float(np.percentile(np.abs(votes), 25))
    floor_acc = float(np.mean(fpred == gold))
    n = len(tr_ids)
    print("=" * 74)
    print("P7 GATE  ds={}  seed0  N_train={}  floor LOO acc={:.4f}  n_errors={}  "
          "band tau(|vote|25pct)={:.4f}".format(
              ds, n, floor_acc, int(np.sum(fpred != gold)), tau))
    print("=" * 74)
    binc, dens = load_channels(ds)
    any_pass = False
    for cname, cmap in [("bin", binc), ("dens", dens)]:
        if not cmap:
            print("  channel {}: (unavailable)".format(cname))
            continue
        c = np.array([cmap.get(v, np.nan) for v in tr_ids])
        ok = ~np.isnan(c)
        nok = int(ok.sum())
        auc = roc_auc_score(gold[ok], c[ok]) if len(set(gold[ok])) > 1 else float("nan")
        pbis = float(np.corrcoef(c[ok], gold[ok])[0, 1])
        corr_sv = float(np.corrcoef(c[ok], sv[ok])[0, 1])
        cov = 100.0 * nok / n
        print("\n  --- channel: {} (coverage {:.0f}% of train) ---".format(cname, cov))
        print("    quality: AUC(c,gold)={:.3f}  point-biserial(c,gold)={:.3f}  "
              "DECORR corr(c,vote_share)={:+.3f}".format(auc, pbis, corr_sv))
        for rname, newpred in [("R1", fuse_R1(sv, c, fpred, ok)),
                               ("R2", fuse_R2(sv, c, fpred, votes, tau, ok))]:
            corr_, dam, net, nerr, rate = net_flips(fpred, newpred, gold, ok)
            passed = (net > 0) and (rate >= 0.15)
            any_pass = any_pass or passed
            newacc = float(np.mean(newpred[ok] == gold[ok]))
            print("    {}: corrected {} / damaged {} / net {} of {} LOO errors "
                  "-> net rate {:+.3f}  (fused LOO acc {:.4f}) -> {}".format(
                      rname, corr_, dam, net, nerr, rate, newacc,
                      "PASS" if passed else "fail"))
    print("\n  GATE {}: {}".format(
        ds, "PASS (>=1 rule x channel cleared +15% net)" if any_pass
        else "FAIL (kill train-side, no test contact)"))
    return any_pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="MHC,MHC_zh")
    a = ap.parse_args()
    for ds in a.datasets.split(","):
        gate(ds.strip())
