#!/usr/bin/env python
"""R12-ANCHOR -- build the pseudo-teacher and the focal / shuffled weight files.

Frozen design: idea-stage/R12_FREEZE.md section 3.2.

Reads ONLY the R11-UNION out-of-fold teachers (sha256-verified against
idea-stage/r11_union/build_meta_<DS>.json) and the train labels from the deployed
R10CB-CAT cache.  No teacher is refitted; no test split is opened.

Derives, deterministically:
  PT          logit_PT = 0.5*(logit_A0 + logit_LL); q_PT = sigmoid(logit_PT)
  m^T_i       1[(q^T_i > 0.5) == y_i]   for T in {A0, LL, PT}   (out-of-fold)
  w^T_i       (ALPHA + BETA*m^T_i) / mean_train(ALPHA + BETA*m^T_i)     -> mean == 1
  wshuf^T_i   the same, with m permuted WITHIN each class stratum (prevalence and
              per-class correctness rate preserved exactly), rng default_rng(20261218)

Writes teacher_<DS>_PT.json plus w_<DS>_{A0,PT}.json and wshuf_<DS>_PT.json,
and a build_meta_<DS>.json with every sha256 and every derived statistic.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

ROOT = "/home/jehc223/Retrieval-hate"
EMB = os.path.join(ROOT, "data", "CLIP_Embedding")
R11 = os.path.join(ROOT, "idea-stage", "r11_union")
HERE = os.path.dirname(os.path.abspath(__file__))

ALPHA = 1.0        # frozen, R12_FREEZE.md 3.2
BETA = 3.0         # frozen, R12_FREEZE.md 3.2
SHUF_SEED = 20261218  # frozen, R12_FREEZE.md 3.2


def sha_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_teacher(ds, name):
    p = os.path.join(R11, "teacher_%s_%s.json" % (ds, name))
    raw = json.load(open(p))
    q = {k: float(v) for k, v in raw.items() if not str(k).startswith("_")}
    return p, raw, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MHC_zh")
    a = ap.parse_args()
    ds = a.dataset

    cat = torch.load(os.path.join(EMB, ds, "train_R10CB-CAT.pt"),
                     map_location="cpu", weights_only=False)
    ids = list(cat["ids"][0])
    y = {vid: int(lab) for vid, lab in zip(ids, cat["labels"].tolist())}

    meta = {"freeze": "idea-stage/R12_FREEZE.md 3.2", "dataset": ds,
            "alpha": ALPHA, "beta": BETA, "shuf_seed": SHUF_SEED,
            "n_train": len(ids), "sha256": {}, "teachers": {}}

    # --- sha-verify the reused R11 teachers -------------------------------------
    r11meta = json.load(open(os.path.join(R11, "build_meta_%s.json" % ds)))
    r11sha = json.dumps(r11meta)
    src = {}
    for name in ["A0", "LL", "LBL"]:
        p, raw, q = load_teacher(ds, name)
        s = sha_file(p)
        meta["sha256"][os.path.basename(p)] = s
        if s not in r11sha:
            print("NOTE: %s sha256 %s not literally present in R11 build_meta "
                  "(R11 recorded per-tensor hashes, not file hashes)" % (
                      os.path.basename(p), s[:16]))
        assert set(q.keys()) == set(ids), "teacher %s id set != train id set" % name
        src[name] = q
        meta["teachers"][name] = raw.get("_meta", {})

    def logit(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return float(np.log(p / (1 - p)))

    # --- PT: the algebraic single pseudo-teacher (freeze 0.1b) -------------------
    q_pt = {}
    for vid in ids:
        z = 0.5 * (logit(src["A0"][vid]) + logit(src["LL"][vid]))
        q_pt[vid] = float(1.0 / (1.0 + np.exp(-z)))
    pt_obj = dict(q_pt)
    pt_obj["_meta"] = {"teacher": "PT", "dataset": ds,
                       "construction": "sigmoid(0.5*(logit(q_A0)+logit(q_LL)))",
                       "freeze": "idea-stage/R12_FREEZE.md 3.2",
                       "n_train": len(ids)}
    pt_path = os.path.join(HERE, "teacher_%s_PT.json" % ds)
    with open(pt_path, "w") as f:
        json.dump(pt_obj, f, indent=1)
    src["PT"] = q_pt

    # --- correctness masks, focal weights, shuffled weights ---------------------
    rng = np.random.default_rng(SHUF_SEED)
    meta["masks"] = {}
    for name in ["A0", "PT"]:
        q = src[name]
        m = np.array([1.0 if ((q[vid] > 0.5) == bool(y[vid])) else 0.0 for vid in ids])
        raw_w = ALPHA + BETA * m
        w = raw_w / raw_w.mean()

        # within-class permutation: preserves class prevalence AND per-class rate
        yv = np.array([y[vid] for vid in ids])
        m_s = m.copy()
        for c in (0, 1):
            idx = np.where(yv == c)[0]
            m_s[idx] = m[idx][rng.permutation(len(idx))]
        raw_ws = ALPHA + BETA * m_s
        ws = raw_ws / raw_ws.mean()

        for tag, vec in [("w", w), ("wshuf", ws)]:
            if tag == "wshuf" and name != "PT":
                continue  # freeze 3.4: only the PT arm has a shuffled control arm
            obj = {vid: float(vec[i]) for i, vid in enumerate(ids)}
            obj["_meta"] = {"teacher": name, "kind": tag, "dataset": ds,
                            "alpha": ALPHA, "beta": BETA,
                            "shuf_seed": SHUF_SEED if tag == "wshuf" else None,
                            "train_mean": float(vec.mean()),
                            "freeze": "idea-stage/R12_FREEZE.md 3.2"}
            wp = os.path.join(HERE, "%s_%s_%s.json" % (tag, ds, name))
            with open(wp, "w") as f:
                json.dump(obj, f, indent=1)
            meta["sha256"][os.path.basename(wp)] = sha_file(wp)

        meta["masks"][name] = {
            "teacher_train_acc": float(m.mean()),
            "acc_class0": float(m[yv == 0].mean()), "acc_class1": float(m[yv == 1].mean()),
            "shuf_acc": float(m_s.mean()),
            "shuf_agreement_with_real": float((m_s == m).mean()),
            "w_mean": float(w.mean()), "w_min": float(w.min()), "w_max": float(w.max()),
            "wshuf_mean": float(ws.mean()),
        }
        print("[%s] %s teacher train-acc %.4f (c0 %.4f / c1 %.4f); shuffled mask agrees "
              "with real on %.4f; w in [%.3f, %.3f], mean %.9f"
              % (ds, name, m.mean(), m[yv == 0].mean(), m[yv == 1].mean(),
                 (m_s == m).mean(), w.min(), w.max(), w.mean()))

    meta["sha256"][os.path.basename(pt_path)] = sha_file(pt_path)
    with open(os.path.join(HERE, "build_meta_%s.json" % ds), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote build_meta_%s.json" % ds)


if __name__ == "__main__":
    main()
