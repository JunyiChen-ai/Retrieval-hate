#!/usr/bin/env python
"""CAT close-out Leg C -- build the 5x5 repeated stratified CV cell caches.

Frozen design: idea-stage/CAT_CLOSEOUT_FREEZE.md section 4.2.

Population = train + dev_seen rows of the ALREADY-BANKED R10CB-A0 / R10CB-CAT caches
(the exact caches behind the R10-COMBO / R11 / R12 CAT numbers).  Zero new extraction.

The official test_seen split is never loaded except to build the forbidden-id set for
BELT C1, which asserts every fold is disjoint from it.

Per cell (r,f):
  EVAL        = fold f of StratifiedKFold(5, shuffle=True, random_state=20260818+r)
  REST        = pool \\ EVAL
  INNER_DEV   = StratifiedShuffleSplit(1, test_size=round(|REST|*d),
                                       random_state=(1000*(20260818+r)+f) % (2**32-1))
                of REST   [deviation D1]
  INNER_TRAIN = REST \\ INNER_DEV
  d           = |dev_seen| / (|train| + |dev_seen|)

written as {train: INNER_TRAIN, dev_seen: INNER_DEV, test_seen: EVAL} under tags
CCCVr{r}f{f}-A0 and CCCVr{r}f{f}-CAT.
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

ROOT = "/home/jehc223/Retrieval-hate"
EMB = os.path.join(ROOT, "data", "CLIP_Embedding")
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PREFIX = "R10CB"
ARMS = ["A0", "CAT"]
N_REP = 5
N_FOLD = 5
BASE_RNG = 20260818


def load(ds, arm, split):
    return torch.load(os.path.join(EMB, ds, "%s_%s-%s.pt" % (split, SRC_PREFIX, arm)),
                      map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    ds = a.dataset

    pools = {}
    ids_ref = None
    for arm in ARMS:
        tr, dv = load(ds, arm, "train"), load(ds, arm, "dev_seen")
        ids = list(tr["ids"][0]) + list(dv["ids"][0])
        img = torch.cat([tr["img_feats"], dv["img_feats"]], 0).float()
        txt = torch.cat([tr["text_feats"], dv["text_feats"]], 0).float()
        lab = torch.cat([tr["labels"], dv["labels"]], 0).long()
        if ids_ref is None:
            ids_ref, y_ref = ids, lab
            n_train, n_dev = len(tr["ids"][0]), len(dv["ids"][0])
        else:
            assert ids == ids_ref, "arm id order differs"
            assert torch.equal(lab, y_ref), "arm labels differ"
        pools[arm] = {"ids": ids, "img": img, "text": txt, "labels": lab}

    # BELT C1: the official test split must never appear in any fold.
    te = load(ds, ARMS[0], "test_seen")
    forbidden = set(te["ids"][0])
    overlap = forbidden & set(ids_ref)
    if overlap:
        raise SystemExit("HALT: BELT C1 -- %d official test ids are inside the CV pool"
                         % len(overlap))
    print("BELT C1: pool n=%d (train %d + dev %d), official test n=%d, overlap 0"
          % (len(ids_ref), n_train, n_dev, len(forbidden)))

    y = y_ref.numpy()
    d = n_dev / float(n_train + n_dev)
    meta = {"freeze": "idea-stage/CAT_CLOSEOUT_FREEZE.md 4.2", "dataset": ds,
            "pool_n": len(ids_ref), "pool_pos": int(y.sum()),
            "n_train_orig": n_train, "n_dev_orig": n_dev, "inner_dev_frac": d,
            "src_prefix": SRC_PREFIX, "belt_C1_overlap": 0, "cells": {}}

    for r in range(N_REP):
        skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=BASE_RNG + r)
        for f, (rest_idx, eval_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            n_inner_dev = int(round(len(rest_idx) * d))
            # deviation D1: the frozen expression 1000*(BASE_RNG+r)+f exceeds numpy's
            # RandomState domain; it is taken modulo 2**32-1.  See
            # idea-stage/CAT_CLOSEOUT_DEVIATION_D1.md.
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=n_inner_dev,
                random_state=(1000 * (BASE_RNG + r) + f) % (2 ** 32 - 1))
            (itr_rel, idv_rel), = sss.split(np.zeros(len(rest_idx)), y[rest_idx])
            itr = rest_idx[itr_rel]
            idv = rest_idx[idv_rel]
            assert len(set(itr) | set(idv) | set(eval_idx)) == len(y)
            assert not (set(itr) & set(idv)) and not (set(itr) & set(eval_idx))
            assert not (set(idv) & set(eval_idx))

            for arm in ARMS:
                P = pools[arm]
                tag = "CCCVr%df%d-%s" % (r, f, arm)
                for split, sel in (("train", itr), ("dev_seen", idv), ("test_seen", eval_idx)):
                    sel_t = torch.as_tensor(np.asarray(sel), dtype=torch.long)
                    obj = {"ids": [[P["ids"][i] for i in sel]],
                           "img_feats": P["img"][sel_t].contiguous(),
                           "text_feats": P["text"][sel_t].contiguous(),
                           "labels": P["labels"][sel_t]}
                    torch.save(obj, os.path.join(EMB, ds, "%s_%s.pt" % (split, tag)))
            meta["cells"]["r%df%d" % (r, f)] = {
                "n_inner_train": int(len(itr)), "n_inner_dev": int(len(idv)),
                "n_eval": int(len(eval_idx)),
                "pos_inner_train": int(y[itr].sum()), "pos_inner_dev": int(y[idv].sum()),
                "pos_eval": int(y[eval_idx].sum()),
                "head_seed": 1500 + 5 * r + f,
                "eval_ids": [P["ids"][i] for i in eval_idx]}
        print("repeat %d built (5 folds)" % r)

    mp = os.path.join(HERE, "cv_meta_%s.json" % ds)
    json.dump(meta, open(mp, "w"), indent=1)
    c0 = meta["cells"]["r0f0"]
    print("cells: inner_train=%d inner_dev=%d eval=%d (r0f0); wrote %s"
          % (c0["n_inner_train"], c0["n_inner_dev"], c0["n_eval"], mp))


if __name__ == "__main__":
    main()
