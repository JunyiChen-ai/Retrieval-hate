#!/usr/bin/env python
"""READOUT-GRID $0 dev screen (ZERO GPU, ZERO test-touch).

Pre-registered in refine-logs/READOUT_PREREG.md. Decides KS-readout-dead vs promote-one-cell
on the raw fused-key kNN vote over the R0/R1/R2/R3 readout caches. NO head training, NO test
split ever loaded. Machinery (rank-weighted signed-cosine top-20 vote, fused key, two retrieval
arms) is lifted VERBATIM from the deployed $0 gates (cross_channel_router_gate.py:73-79 vote;
LP_GATE_RECORD.md §1 fused key; ISR_PREGATE_RECORD.md §0.2 two arms). This script only computes.

Run: conda activate HateVideo; CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 \
     python scripts/analysis/readout_screen.py
Writes: refine-logs/READOUT_SCREEN_OUT.json
"""
import os
import sys
import json
import hashlib

import numpy as np
import torch
import faiss

faiss.omp_set_num_threads(4)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE = os.path.join(REPO, "data", "CLIP_Embedding")

TOPK = 20
ADVANCE_BAR = 0.020          # winner promotes iff dev Δacc >= +0.02 on >=1 dataset (either arm)
N_PERM = 200                 # permutation-null perms (>=200, LP §4b)
N_BOOT = 1000                # bootstrap resamples (LP §4a discipline)
RNG = 20260725

# R0..R3 cache suffixes (frozen grid, prereg §1).
CELLS = ["ro_L28", "ro_L24", "ro_ow_L28", "ro_ow_L24"]
R0 = "ro_L28"

# dataset -> DEPLOYED base tag (the banked R0 == this cache with NO suffix).
DATASETS = {
    "MHC_zh": "Qwen2.5-VL-7B-Instruct-LoRA_HF",           # ZH primary (B3 generic-LoRA)
    "HateMM": "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",    # HateMM hold (curric)
}
ALLOWED_SPLITS = {"train", "dev_seen"}


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def load_cache(ds, split, tag):
    """Load a cache; HARD-ASSERT split is train/dev_seen (test can never be opened)."""
    assert split in ALLOWED_SPLITS, "test-touch blocked: split={}".format(split)
    path = os.path.join(CACHE, ds, "{}_{}.pt".format(split, tag))
    d = torch.load(path, map_location="cpu")
    ids = d["ids"]
    ids = ids[0] if (isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list)) else ids
    return (list(ids), d["img_feats"].float(), d["text_feats"].float(),
            d["labels"].long().numpy(), path)


def _weighted_signed_vote(nb_lab, nb_sim):
    """arithmetic rank-weighted signed-cosine vote over top-k neighbours (verbatim,
    cross_channel_router_gate.py:73-79)."""
    k = len(nb_lab)
    w = np.arange(1, TOPK + 1)[::-1].astype("float64")[:k]
    lm = (nb_lab.astype("float64") * 2 - 1) * nb_sim
    return float(np.sum(lm * w) / np.sum(w))


def fused_key(img, text):
    """L2-norm each stream row-wise, concat -> 7168, L2-renorm (LP_GATE_RECORD §1)."""
    img = torch.nn.functional.normalize(img, p=2, dim=1)
    text = torch.nn.functional.normalize(text, p=2, dim=1)
    key = torch.cat([img, text], dim=1)
    key = torch.nn.functional.normalize(key, p=2, dim=1)
    return key.numpy().astype("float32")


def knn_votes(mem_feat, mem_lab, q_feat, exclude_self):
    """Per-query rank-weighted signed vote. exclude_self => LOO (q rows ARE mem rows, aligned)."""
    tr = mem_feat.copy()
    q = q_feat.copy()
    faiss.normalize_L2(tr)
    faiss.normalize_L2(q)
    ix = faiss.IndexFlatIP(tr.shape[1])
    ix.add(tr)
    kk = TOPK + (1 if exclude_self else 0)
    D, I = ix.search(q, kk)
    votes = np.zeros(len(q))
    for i in range(len(q)):
        idx, sim = I[i], D[i]
        if exclude_self:
            keep = idx != i
            idx, sim = idx[keep][:TOPK], sim[keep][:TOPK]
        else:
            idx, sim = idx[:TOPK], sim[:TOPK]
        votes[i] = _weighted_signed_vote(mem_lab[idx], sim)
    return votes


def dev_preds(key_train, lab_train, key_dev, lab_dev, arm):
    """Return dev predictions (0/1) under one retrieval arm.

    arm 'loo'      : memory = train ∪ dev, LOO self-exclusion, score dev rows.
    arm 'devtrain' : memory = train only, dev queries (disjoint), score dev rows.
    """
    if arm == "loo":
        key_all = np.concatenate([key_train, key_dev], axis=0)
        lab_all = np.concatenate([lab_train, lab_dev], axis=0)
        v_all = knn_votes(key_all, lab_all, key_all, exclude_self=True)
        v_dev = v_all[len(key_train):]
    elif arm == "devtrain":
        v_dev = knn_votes(key_train, lab_train, key_dev, exclude_self=False)
    else:
        raise ValueError(arm)
    return (v_dev >= 0).astype(int)


def acc(preds, gold):
    return float((preds == gold).mean())


def fix_break_net(cell_preds, r0_preds, gold):
    """Items the cell fixes / breaks / net vs R0 (both against gold)."""
    r0_ok = (r0_preds == gold)
    cl_ok = (cell_preds == gold)
    fixed = int(np.sum((~r0_ok) & cl_ok))
    broken = int(np.sum(r0_ok & (~cl_ok)))
    return fixed, broken, fixed - broken


def perm_null(key_train, lab_train, key_dev, lab_dev, best_key_train, best_lab_train, arm, rng):
    """Null Δacc = bestcell_acc(shuffled mem labels) − R0_acc(shuffled) on dev gold (LP §4b).

    R0 and the winning cell each recompute their vote under the SAME shuffled memory labels;
    dev preds are scored against the TRUE dev gold. Returns the null Δacc distribution.
    """
    n = len(lab_train)
    out = []
    for _ in range(N_PERM):
        perm = rng.permutation(n)
        lt = lab_train[perm]
        # best cell uses its own train labels shuffled by the SAME perm (aligned rows)
        bt = best_lab_train[perm]
        r0 = dev_preds(key_train, lt, key_dev, lab_dev, arm)
        bc = dev_preds(best_key_train, bt, best_key_dev_holder[0], lab_dev, arm)
        out.append(acc(bc, lab_dev) - acc(r0, lab_dev))
    return np.array(out)


# small holder so perm_null can see the winning cell's dev key without a long signature
best_key_dev_holder = [None]


def bootstrap_delta(cell_preds, r0_preds, gold, rng):
    """5th-pct of Δacc(cell − R0) under 1000 dev-resamples (LP §4a)."""
    n = len(gold)
    deltas = []
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        deltas.append(acc(cell_preds[s], gold[s]) - acc(r0_preds[s], gold[s]))
    d = np.array(deltas)
    return float(np.percentile(d, 5)), float(d.mean())


def screen_dataset(ds, base_tag, rng):
    res = {"dataset": ds, "base_tag": base_tag, "cells": {}, "arms": {}, "guards": {}}

    # --- load all cells (train+dev) + the banked deployed cache for the R0 bit-exact gate ---
    loaded = {}
    for suffix in CELLS:
        tag = "{}-{}".format(base_tag, suffix)
        ids_tr, img_tr, txt_tr, lab_tr, p_tr = load_cache(ds, "train", tag)
        ids_dv, img_dv, txt_dv, lab_dv, p_dv = load_cache(ds, "dev_seen", tag)
        loaded[suffix] = dict(img_tr=img_tr, txt_tr=txt_tr, lab_tr=lab_tr,
                              img_dv=img_dv, txt_dv=txt_dv, lab_dv=lab_dv,
                              sha_tr=sha16(p_tr), sha_dv=sha16(p_dv))
    lab_dev = loaded[R0]["lab_dv"]
    lab_train = loaded[R0]["lab_tr"]
    res["n_dev"] = int(len(lab_dev))
    res["n_train"] = int(len(lab_train))

    # --- GUARD 1: R0 bit-exact clobber-guard (ro_L28 == banked deployed cache) ---
    bit_exact = {}
    for split in ("train", "dev_seen"):
        _, bimg, btxt, _, bp = load_cache(ds, split, base_tag)      # banked deployed
        _, rimg, rtxt, _, rp = load_cache(ds, split, base_tag + "-" + R0)  # R0 recompute
        di = float((bimg - rimg).abs().max())
        dt = float((btxt - rtxt).abs().max())
        bit_exact[split] = dict(img_maxabs=di, text_maxabs=dt,
                                exact=bool(di == 0.0 and dt == 0.0),
                                banked_sha=sha16(bp), recompute_sha=sha16(rp))
    res["guards"]["R0_bit_exact"] = bit_exact
    res["guards"]["R0_bit_exact_pass"] = all(v["exact"] for v in bit_exact.values())

    # --- per-cell, per-arm dev acc + Δ vs R0 ---
    r0_preds = {}
    for arm in ("loo", "devtrain"):
        kt = fused_key(loaded[R0]["img_tr"], loaded[R0]["txt_tr"])
        kd = fused_key(loaded[R0]["img_dv"], loaded[R0]["txt_dv"])
        r0_preds[arm] = dev_preds(kt, lab_train, kd, lab_dev, arm)

    def l2_np(feat):
        return torch.nn.functional.normalize(feat, p=2, dim=1).numpy().astype("float32")

    for suffix in CELLS:
        L = loaded[suffix]
        cell_out = {"sha_train": L["sha_tr"], "sha_dev": L["sha_dv"], "arms": {}}
        # key builders per stream: fused (decision object) + img-only/text-only diagnostics
        keys = {
            "fused": (fused_key(L["img_tr"], L["txt_tr"]), fused_key(L["img_dv"], L["txt_dv"])),
            "img":   (l2_np(L["img_tr"]), l2_np(L["img_dv"])),
            "text":  (l2_np(L["txt_tr"]), l2_np(L["txt_dv"])),
        }
        for arm in ("loo", "devtrain"):
            arm_out = {}
            for stream, (kt, kd) in keys.items():
                preds = dev_preds(kt, lab_train, kd, lab_dev, arm)
                a = acc(preds, lab_dev)
                entry = {"acc": a}
                if stream == "fused":
                    fx, bk, net = fix_break_net(preds, r0_preds[arm], lab_dev)
                    entry.update(dacc=a - acc(r0_preds[arm], lab_dev),
                                 fix=fx, brk=bk, net=net, preds=preds.tolist())
                arm_out[stream] = entry
            cell_out["arms"][arm] = arm_out
        res["cells"][suffix] = cell_out

    # R0 absolute acc per arm (context)
    res["arms"] = {arm: dict(R0_acc=acc(r0_preds[arm], lab_dev)) for arm in ("loo", "devtrain")}

    # --- winner = best fused-key dev Δacc over R0 across R1/R2/R3, either arm ---
    best = {"suffix": None, "arm": None, "dacc": -9.9}
    for suffix in CELLS:
        if suffix == R0:
            continue
        for arm in ("loo", "devtrain"):
            dacc = res["cells"][suffix]["arms"][arm]["fused"]["dacc"]
            if dacc > best["dacc"]:
                best = {"suffix": suffix, "arm": arm, "dacc": dacc}
    res["winner"] = best

    # --- perm-null + bootstrap on the winner (fused key, winner's arm) ---
    w = loaded[best["suffix"]]
    arm = best["arm"]
    r0kt = fused_key(loaded[R0]["img_tr"], loaded[R0]["txt_tr"])
    r0kd = fused_key(loaded[R0]["img_dv"], loaded[R0]["txt_dv"])
    wkt = fused_key(w["img_tr"], w["txt_tr"])
    wkd = fused_key(w["img_dv"], w["txt_dv"])
    best_key_dev_holder[0] = wkd
    null = perm_null(r0kt, lab_train, r0kd, lab_dev, wkt, lab_train, arm, rng)
    res["guards"]["perm_null"] = dict(p5=float(np.percentile(null, 5)),
                                      p95=float(np.percentile(null, 95)),
                                      maxv=float(null.max()),
                                      obs=best["dacc"],
                                      obs_gt_p95=bool(best["dacc"] > np.percentile(null, 95)))
    w_preds = res["cells"][best["suffix"]]["arms"][arm]["fused"]["preds"]
    b5, bmean = bootstrap_delta(np.array(w_preds), r0_preds[arm], lab_dev, rng)
    res["guards"]["bootstrap"] = dict(dacc_p5=b5, dacc_mean=bmean)

    # strip bulky preds arrays from the JSON
    for suffix in res["cells"]:
        for arm in res["cells"][suffix]["arms"]:
            res["cells"][suffix]["arms"][arm]["fused"].pop("preds", None)

    res["advance"] = bool(best["dacc"] >= ADVANCE_BAR and res["guards"]["R0_bit_exact_pass"])
    return res


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", \
        "run CPU-only: export CUDA_VISIBLE_DEVICES=''"
    rng = np.random.default_rng(RNG)
    out = {"topk": TOPK, "advance_bar": ADVANCE_BAR, "n_perm": N_PERM, "n_boot": N_BOOT,
           "rng": RNG, "datasets": {}}
    for ds, base in DATASETS.items():
        out["datasets"][ds] = screen_dataset(ds, base, rng)

    any_adv = any(out["datasets"][ds]["advance"] for ds in DATASETS)
    out["verdict"] = "PROMOTE" if any_adv else "KS-readout-dead"
    if any_adv:
        cand = [(ds, out["datasets"][ds]["winner"]) for ds in DATASETS
                if out["datasets"][ds]["advance"]]
        out["promoted"] = cand

    dest = os.path.join(REPO, "refine-logs", "READOUT_SCREEN_OUT.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"verdict": out["verdict"],
                      "winners": {ds: out["datasets"][ds]["winner"] for ds in DATASETS},
                      "R0_bit_exact": {ds: out["datasets"][ds]["guards"]["R0_bit_exact_pass"]
                                       for ds in DATASETS}}, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
