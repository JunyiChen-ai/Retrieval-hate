#!/usr/bin/env python
"""R11-UNION -- build the one new arm cache and the three frozen anchor teachers.

Frozen design: idea-stage/R11_UNION_FREEZE.md sections 2 and 3.

Nothing here reads dev or test LABELS.  The MC cache reads only the train-split
mean of the deployed block; the teachers are fitted on the TRAIN split only, and
their probabilities are out-of-fold so they are not a re-statement of the train
labels.  No metric of any candidate arm is computed here.

Outputs
  data/CLIP_Embedding/<DS>/<split>_R11UN-MC.pt      img i28, text n(a28 - mean_train(a28))
  idea-stage/r11_union/teacher_<DS>_A0.json         OOF P(hate) from [i28 | a28]
  idea-stage/r11_union/teacher_<DS>_LL.json         OOF P(hate) from [i28|i24|a28|a24]
  idea-stage/r11_union/teacher_<DS>_LBL.json        hard train labels (control)
  idea-stage/r11_union/build_meta_<DS>.json

Belt: the three reused R10-COMBO caches (A0, LL, CAT) are re-hashed and required
to match idea-stage/r10_combo/build_meta_<DS>.json byte for byte.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = "/home/jehc223/Retrieval-hate"
EMB = os.path.join(ROOT, "data", "CLIP_Embedding")
HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = ["train", "dev_seen", "test_seen"]
PREFIX = "R11UN"
N_FOLDS = 5
FOLD_SEED = 20260818
LOGREG = dict(C=1.0, max_iter=5000, solver="lbfgs", tol=1e-4)
Q_CLIP = 1e-4          # keeps BCEWithLogits finite for the hard-label control


def l2norm(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def sha_tensor(t):
    return hashlib.sha256(np.ascontiguousarray(
        t.detach().cpu().numpy().astype(np.float32)).tobytes()).hexdigest()


def macro_f1(y, p):
    yp = (p >= 0.5).astype(int)
    f1s = []
    for c in (1, 0):
        tp = int(((yp == c) & (y == c)).sum())
        fp = int(((yp == c) & (y != c)).sum())
        fn = int(((yp != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s))


def oof_teacher(X, y, name):
    """5-fold stratified out-of-fold P(y=1).  Deterministic given FOLD_SEED."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    q = np.zeros(len(y), dtype=np.float64)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(**LOGREG)
        clf.fit(X[tr], y[tr])
        q[te] = clf.predict_proba(X[te])[:, 1]
    q = np.clip(q, Q_CLIP, 1.0 - Q_CLIP)
    stats = {"oof_macro_f1": macro_f1(y, q),
             "oof_acc": float(((q >= 0.5).astype(int) == y).mean()),
             "mean_abs_q_minus_y": float(np.abs(q - y).mean()),
             "frac_q_within_0.05_of_label": float((np.abs(q - y) < 0.05).mean()),
             "q_mean": float(q.mean()), "q_std": float(q.std())}
    print("teacher %-4s in_dim=%d  OOF macroF1=%.4f acc=%.4f  mean|q-y|=%.4f  "
          "frac(|q-y|<0.05)=%.3f" % (name, X.shape[1], stats["oof_macro_f1"],
                                     stats["oof_acc"], stats["mean_abs_q_minus_y"],
                                     stats["frac_q_within_0.05_of_label"]))
    return q, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base", required=True)
    a = ap.parse_args()
    DS, BASE = a.dataset, a.base
    meta = {"freeze": "idea-stage/R11_UNION_FREEZE.md 2,3", "dataset": DS,
            "base": BASE, "n_folds": N_FOLDS, "fold_seed": FOLD_SEED,
            "logreg": LOGREG, "reused_caches": {}, "files": {}, "teachers": {}}

    # ---- belt: the three reused R10-COMBO caches are unchanged ----
    r10_meta = json.load(open(os.path.join(
        ROOT, "idea-stage", "r10_combo", "build_meta_%s.json" % DS)))
    for split in SPLITS:
        for arm in ["A0", "LL", "CAT"]:
            key = "%s/%s" % (split, arm)
            ref = r10_meta["files"][key]
            obj = torch.load(os.path.join(EMB, DS, "%s_R10CB-%s.pt" % (split, arm)),
                             map_location="cpu", weights_only=False)
            got_i, got_t = sha_tensor(obj["img_feats"]), sha_tensor(obj["text_feats"])
            ok = (got_i == ref["img_sha256"] and got_t == ref["text_sha256"])
            meta["reused_caches"][key] = {"img_sha256": got_i, "text_sha256": got_t,
                                          "matches_r10_combo": bool(ok)}
            if not ok:
                raise SystemExit("HALT: R10CB-%s on %s no longer matches R10-COMBO"
                                 % (arm, split))
    print("BELT ok: 9 reused R10CB caches match idea-stage/r10_combo/build_meta_%s.json"
          % DS)

    # ---- MC arm: train-mean-removed deployed block, full width ----
    raw = {}
    for split in SPLITS:
        tp = torch.load(os.path.join(EMB, DS, "%s_%s-tp.pt" % (split, BASE)),
                        map_location="cpu", weights_only=False)
        ro28 = torch.load(os.path.join(EMB, DS, "%s_%s-ro_L28.pt" % (split, BASE)),
                          map_location="cpu", weights_only=False)
        ids = tp["ids"][0]
        assert ids == ro28["ids"][0], "id order differs on %s" % split
        raw[split] = {"ids": ids, "labels": ro28["labels"],
                      "a28": l2norm(tp["spans"]["28"]["A0"].float()),
                      "i28": l2norm(ro28["img_feats"].float())}
    mu = raw["train"]["a28"].mean(dim=0, keepdim=True)
    meta["mc_train_mean_norm"] = float(mu.norm())
    for split in SPLITS:
        img = raw[split]["i28"]
        txt = l2norm(raw[split]["a28"] - mu)
        op = os.path.join(EMB, DS, "%s_%s-MC.pt" % (split, PREFIX))
        torch.save({"ids": [raw[split]["ids"]], "img_feats": img.contiguous(),
                    "text_feats": txt.contiguous(), "labels": raw[split]["labels"]}, op)
        meta["files"]["%s/MC" % split] = {
            "path": os.path.relpath(op, ROOT), "rows": int(txt.shape[0]),
            "img_dim": int(img.shape[1]), "text_dim": int(txt.shape[1]),
            "img_sha256": sha_tensor(img), "text_sha256": sha_tensor(txt)}
        print("   %-10s MC  img=%-12s text=%s" % (split, tuple(img.shape),
                                                  tuple(txt.shape)))

    # ---- anchor teachers, TRAIN split only ----
    tr_ids = raw["train"]["ids"]
    y = raw["train"]["labels"].reshape(-1).long().numpy()
    feats = {}
    for arm in ["A0", "LL"]:
        obj = torch.load(os.path.join(EMB, DS, "train_R10CB-%s.pt" % arm),
                         map_location="cpu", weights_only=False)
        assert obj["ids"][0] == tr_ids, "train id order differs for %s" % arm
        feats[arm] = torch.cat([obj["img_feats"].float(),
                                obj["text_feats"].float()], dim=1).numpy()

    for name in ["A0", "LL", "LBL"]:
        if name == "LBL":
            q = np.clip(y.astype(np.float64), Q_CLIP, 1.0 - Q_CLIP)
            stats = {"oof_macro_f1": 1.0, "oof_acc": 1.0,
                     "mean_abs_q_minus_y": float(Q_CLIP),
                     "frac_q_within_0.05_of_label": 1.0,
                     "q_mean": float(q.mean()), "q_std": float(q.std()),
                     "note": "hard-label control, not a teacher"}
            print("teacher LBL  hard train labels (control)")
        else:
            q, stats = oof_teacher(feats[name], y, name)
        out = {vid: float(v) for vid, v in zip(tr_ids, q)}
        out["_meta"] = {"teacher": name, "dataset": DS, "n_train": len(tr_ids),
                        "n_folds": N_FOLDS, "fold_seed": FOLD_SEED,
                        "logreg": LOGREG, **stats}
        p = os.path.join(HERE, "teacher_%s_%s.json" % (DS, name))
        json.dump(out, open(p, "w"), indent=1)
        meta["teachers"][name] = {"path": os.path.relpath(p, ROOT), **stats}
        print("   wrote", p)

    mp = os.path.join(HERE, "build_meta_%s.json" % DS)
    json.dump(meta, open(mp, "w"), indent=1)
    print("wrote", mp)


if __name__ == "__main__":
    main()
