#!/usr/bin/env python
"""Blast-radius audit for the degenerate CLIP feature rows.

Question: could the degenerate rows have flipped any historical A0 conclusion?

Instrument: the project's existing frozen-space A0 head-level OOF harness
(`scripts/ocr_cache/ocr_fusion_pilot.py`, arm 0 = l2(img) + l2(txt), linear head,
inner-4-fold epoch/threshold selection, the frozen HateMM 5-fold OOF split from
`artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf`).  Nothing here
touches dev_seen or test_seen: TRAIN OOF only.

Arms
  PRE   img_feats from the original cache
  POST  img_feats from the *-degenfix1 cache (only rows the fixer repaired differ)

Reported per arm and seed
  * OOF macro-F1 over all 744 train videos (the historical quantity)
  * OOF macro-F1 over the 744 - k non-degenerate videos (does the number survive
    deleting the degenerate rows from the evaluation?)
  * per-degenerate-row OOF correctness and fold id
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ocr_cache import ocr_fusion_pilot as P  # noqa: E402

MODEL = "openai_clip-vit-large-patch14-336_HF"
SEEDS = (20260810, 20260811, 20260812)


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_cache(path):
    d = torch.load(path, map_location="cpu")
    raw = d["ids"]
    ids = list(raw[0]) if (len(raw) == 1 and isinstance(raw[0], list)) else list(raw)
    img = torch.as_tensor(d["img_feats"])
    if img.dim() == 3:
        img = img[0]
    txt = torch.as_tensor(d["text_feats"])
    if txt.dim() == 3:
        txt = txt[0]
    y = torch.as_tensor(d["labels"]).flatten().numpy().astype(np.int64)
    return ids, img.numpy().astype(np.float64), txt.numpy().astype(np.float64), y, d.get("degen_flags", {})


def load_folds(ids):
    run = P.RUN
    folds = []
    for f in range(5):
        tr = json.load(open(run / ("folds/fold_%d/train_ids.json" % f)))
        qu = json.load(open(run / ("folds/fold_%d/query_ids.json" % f)))
        for v in tr + qu:
            if "test" in v.lower():
                raise SystemExit("HALT_TEST_CONTACT:" + v)
        folds.append((sorted(tr), sorted(qu)))
    covered = sorted({v for _, qu in folds for v in qu})
    if covered != sorted(ids):
        raise SystemExit("HALT_FOLD_COVERAGE")
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HateMM")
    ap.add_argument("--split", default="train")
    ap.add_argument("--suffix", default="degenfix1")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--out", default=str(ROOT / "artifacts/degen_feat_fix/oof_audit.json"))
    a = ap.parse_args()
    torch.set_num_threads(P.TORCH_THREADS)

    base = ROOT / "data/CLIP_Embedding" / a.dataset
    pre_p = base / ("%s_%s.pt" % (a.split, MODEL))
    post_p = base / ("%s_%s-%s.pt" % (a.split, MODEL, a.suffix))

    ids, img0, txt0, y, _ = load_cache(pre_p)
    ids1, img1, txt1, y1, flags = load_cache(post_p)
    if ids1 != ids or not (y1 == y).all():
        raise SystemExit("HALT_CACHE_MISMATCH")
    idx = {v: i for i, v in enumerate(ids)}
    folds = load_folds(ids)

    changed = [ids[i] for i in range(len(ids)) if not np.array_equal(img0[i], img1[i])]
    degen = sorted(flags)
    fold_of = {v: f for f, (_, qu) in enumerate(folds) for v in qu}
    log("cache rows=%d  flagged degenerate=%d  rows whose feature changed=%s"
        % (len(ids), len(degen), changed))

    keep = np.array([i for i, v in enumerate(ids) if v not in set(degen)], dtype=np.int64)
    X = {"PRE": np.hstack([P.l2np(img0), P.l2np(txt0)]),
         "POST": np.hstack([P.l2np(img1), P.l2np(txt1)])}

    res = {
        "instrument": "scripts/ocr_cache/ocr_fusion_pilot.py arm-0 head, frozen 5-fold train OOF",
        "dataset": a.dataset, "split": a.split, "n": len(ids), "n_pos": int(y.sum()),
        "test_contact": "none (train OOF only)",
        "caches": {
            "PRE": {"path": str(pre_p), "sha256": sha256_file(pre_p)},
            "POST": {"path": str(post_p), "sha256": sha256_file(post_p)},
        },
        "rows_with_changed_features": changed,
        "degenerate_rows": {v: {"code": flags[v], "label": int(y[idx[v]]),
                                "oof_fold": fold_of.get(v)} for v in degen},
        "seeds": a.seeds,
        "arms": {},
    }

    preds = {}
    for arm in ("PRE", "POST"):
        rows = []
        for seed in a.seeds:
            t0 = time.time()
            # arm id fixed to 0 for BOTH arms so head init / batch shuffling are
            # byte-identical between PRE and POST: the only difference is the
            # feature matrix.  0 is also the historical A0 arm id in the pilot.
            f1, pred, info = P.run_arm(X[arm], y, ids, folds, 0, seed, idx)
            f1_keep = P.macro_f1(y[keep], pred[keep])
            preds[(arm, seed)] = pred
            rows.append({"seed": seed,
                         "oof_macro_f1_all": float(f1),
                         "oof_macro_f1_nondegen": float(f1_keep),
                         "seconds": round(time.time() - t0, 1)})
            log("RESULT arm=%s seed=%d oof_all=%.4f oof_nondegen=%.4f (%.0fs)"
                % (arm, seed, f1, f1_keep, time.time() - t0))
        res["arms"][arm] = {
            "per_seed": rows,
            "mean_all": float(np.mean([r["oof_macro_f1_all"] for r in rows])),
            "std_all": float(np.std([r["oof_macro_f1_all"] for r in rows], ddof=1)),
            "mean_nondegen": float(np.mean([r["oof_macro_f1_nondegen"] for r in rows])),
            "std_nondegen": float(np.std([r["oof_macro_f1_nondegen"] for r in rows], ddof=1)),
        }

    paired = [preds[("POST", s)] for s in a.seeds], [preds[("PRE", s)] for s in a.seeds]
    res["paired_delta_all"] = [
        float(P.macro_f1(y, po) - P.macro_f1(y, pr)) for po, pr in zip(*paired)
    ]
    res["paired_delta_mean_all"] = float(np.mean(res["paired_delta_all"]))
    res["n_pred_disagreements"] = [int((po != pr).sum()) for po, pr in zip(*paired)]

    # how the degenerate rows are actually predicted (PRE arm, majority over seeds)
    per_row = {}
    for v in degen:
        i = idx[v]
        corr = [int(preds[("PRE", s)][i] == y[i]) for s in a.seeds]
        per_row[v] = {"label": int(y[i]), "oof_fold": fold_of.get(v),
                      "pre_correct_per_seed": corr,
                      "post_correct_per_seed": [int(preds[("POST", s)][i] == y[i])
                                                for s in a.seeds]}
    res["degenerate_row_oof"] = per_row
    nd = len(degen)
    if nd:
        acc = np.mean([np.mean(per_row[v]["pre_correct_per_seed"]) for v in degen])
        res["degenerate_row_oof_accuracy_pre"] = float(acc)
        log("degenerate-row OOF accuracy (PRE, mean over seeds) = %.3f over %d rows" % (acc, nd))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    log("wrote %s" % a.out)
    log("SUMMARY  PRE all=%.4f  POST all=%.4f  delta=%+.4f | PRE nondegen=%.4f POST nondegen=%.4f"
        % (res["arms"]["PRE"]["mean_all"], res["arms"]["POST"]["mean_all"],
           res["arms"]["POST"]["mean_all"] - res["arms"]["PRE"]["mean_all"],
           res["arms"]["PRE"]["mean_nondegen"], res["arms"]["POST"]["mean_nondegen"]))


if __name__ == "__main__":
    main()
