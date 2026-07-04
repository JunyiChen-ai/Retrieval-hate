"""
W4 follow-up: threshold-RECALIBRATION adaptation on the temporal splits.

Motivation: the W4 evolving-memory runs showed that ADDING new-period
labelled samples to the kNN memory does not recover the temporal drop
(EN macro-F1 0.7113 random -> 0.6273 temporal), while the test positive
rate drops 34% -> 24% (prior drift). Hypothesis: the dominant component of
"hate evolution" inside the MHClip window is PRIOR drift, so the right
lightweight adaptation is not memory growth but re-calibrating the kNN
vote threshold with the k newly-arrived labelled samples (no memory
update, no retraining).

Mechanism compared against memory augmentation (same k, same pool, same
seeds):
  score s(x) = rank-weighted top-20 neighbour label fraction vs the STATIC
               temporal-train memory (exactly compute_metrics_retrieval's
               majority_voting="arithmetic", use_sim=False; decision
               s >= t, default t=0.5).
  calibrate   t* = argmax over a fixed grid of macro-F1 on the k labelled
               new-period (val) samples' scores; if the k samples contain a
               single class, keep t=0.5 (no evidence to move; count these).
  apply       t* on the temporal-test scores. Test labels are never used
               for selection; the labelled-oracle threshold on test is
               reported separately as a ceiling (diagnostic ONLY).

Everything is CPU-only and read-only reuse of existing modules.

Usage:
  python scripts/analysis/temporal_recalibration.py \
      --dataset MHC_temporal --ckpt <temporal best_model.pt> \
      --out logging/temporal_memory/MHC_temporal_recalibration.json
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "src"))

import argparse
import json

import numpy as np
import torch
import faiss
from easydict import EasyDict
from sklearn.metrics import f1_score, accuracy_score

from eval_cross_dataset import project_split
from eval_temporal_memory import (
    TEMPORAL_BASE, load_temporal_feats, build_head)

GRID = np.linspace(0.0, 1.0, 201)


def rank_weighted_scores(mem_emb, mem_lab, q_emb, topk=20):
    """compute_metrics_retrieval(majority_voting='arithmetic', use_sim=False)
    score: rank-weighted (w = topk..1) neighbour-label fraction in [0,1]."""
    mem = mem_emb.copy()
    qry = q_emb.copy()
    faiss.normalize_L2(mem)
    faiss.normalize_L2(qry)
    index = faiss.IndexFlatIP(mem.shape[1])
    index.add(mem)
    _, I = index.search(qry, min(topk, mem.shape[0]))
    w = np.arange(1, I.shape[1] + 1)[::-1].astype("float64")
    lab = mem_lab[I].astype("float64")
    return (lab * w).sum(axis=1) / w.sum()


def eval_at(scores, labels, t):
    pred = (scores >= t).astype(int)
    return (round(float(f1_score(labels, pred, average="macro",
                                 zero_division=0)), 4),
            round(float(accuracy_score(labels, pred)), 4))


def best_threshold(cal_scores, cal_labels):
    """argmax macro-F1 over GRID; ties -> median of the argmax set.
    Single-class calibration set -> keep the default 0.5 (guarded)."""
    if len(np.unique(cal_labels)) < 2:
        return 0.5, True
    f1s = np.array([f1_score(cal_labels, (cal_scores >= t).astype(int),
                             average="macro", zero_division=0)
                    for t in GRID])
    idxs = np.flatnonzero(f1s == f1s.max())
    return float(GRID[idxs[len(idxs) // 2]]), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True,
                    choices=sorted(TEMPORAL_BASE))
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--model", type=str,
                    default="openai_clip-vit-large-patch14-336_HF")
    ap.add_argument("--path", type=str, default="./data/")
    ap.add_argument("--gt_root", type=str, default="./data/gt")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--k_list", type=str, default="5,10,20,50")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--out", type=str, default="")
    # head architecture (must match the trained checkpoint)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--proj_dim", type=int, default=1024)
    ap.add_argument("--map_dim", type=int, default=1024)
    ap.add_argument("--fusion_mode", type=str, default="align")
    ap.add_argument("--dropout", type=float, nargs=3, default=[0.2, 0.4, 0.1])
    ap.add_argument("--batch_norm", type=lambda x: str(x).lower() == "true",
                    default=False)
    args = ap.parse_args()

    clip_path = os.path.join(args.path, "CLIP_Embedding")
    train, val, test = load_temporal_feats(
        clip_path, args.dataset, args.model, gt_root=args.gt_root)

    model = build_head(train[1].shape[1], train[2].shape[1], args)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    device = "cpu"
    model.to(device)

    tr_ids, tr_emb, tr_lab = project_split(model, train, device)
    va_ids, va_emb, va_lab = project_split(model, val, device)
    te_ids, te_emb, te_lab = project_split(model, test, device)
    assert not (set(te_ids) & set(tr_ids)) and not (set(te_ids) & set(va_ids))

    # scores against the STATIC train memory (memory is never modified here)
    s_val = rank_weighted_scores(tr_emb, tr_lab, va_emb, args.topk)
    s_test = rank_weighted_scores(tr_emb, tr_lab, te_emb, args.topk)

    # baseline t=0.5 must reproduce the static arithmetic-vote numbers
    base_f1, base_acc = eval_at(s_test, te_lab, 0.5)
    print("BASELINE t=0.500  macroF1={:.4f} acc={:.4f}  "
          "(must match static mem=train arith)".format(base_f1, base_acc))

    # labelled-oracle ceiling on test (diagnostic ONLY, never a claim)
    t_oracle, _ = best_threshold(s_test, te_lab)
    orc_f1, orc_acc = eval_at(s_test, te_lab, t_oracle)
    print("ORACLE  t={:.3f}  macroF1={:.4f} acc={:.4f}  "
          "(threshold-only ceiling; uses test labels, diagnostic)".format(
              t_oracle, orc_f1, orc_acc))

    k_list = [int(k) for k in args.k_list.split(",") if k.strip()]
    rows = []
    for k in k_list:
        k = min(k, len(va_ids))
        per_seed = []
        for s in range(args.n_seeds):
            rng = np.random.default_rng(1000 + s)  # same seeds as W4 augment
            sel = np.sort(rng.choice(len(va_ids), size=k, replace=False))
            t_star, guarded = best_threshold(s_val[sel], va_lab[sel])
            f1, acc = eval_at(s_test, te_lab, t_star)
            per_seed.append({"seed": 1000 + s, "t": round(t_star, 3),
                             "single_class_guard": bool(guarded),
                             "macro_f1": f1, "acc": acc,
                             "cal_pos_n": int(va_lab[sel].sum())})
        f1s = [r["macro_f1"] for r in per_seed]
        accs = [r["acc"] for r in per_seed]
        row = {
            "k": k,
            "macro_f1_mean": round(float(np.mean(f1s)), 4),
            "macro_f1_std": round(float(np.std(f1s)), 4),
            "acc_mean": round(float(np.mean(accs)), 4),
            "acc_std": round(float(np.std(accs)), 4),
            "gain_per_sample_macro_f1": round(
                (float(np.mean(f1s)) - base_f1) / k, 5),
            "n_guarded": sum(r["single_class_guard"] for r in per_seed),
            "per_seed": per_seed,
        }
        rows.append(row)
        print("RECAL k={:<3d} macroF1={:.4f}+-{:.4f} acc={:.4f}+-{:.4f} "
              "gain/sample(F1)={:+.5f} guarded={}/{} t={}".format(
                  k, row["macro_f1_mean"], row["macro_f1_std"],
                  row["acc_mean"], row["acc_std"],
                  row["gain_per_sample_macro_f1"],
                  row["n_guarded"], args.n_seeds,
                  [r["t"] for r in per_seed]))

    # all-val calibration (deterministic upper end of the curve)
    t_all, guarded_all = best_threshold(s_val, va_lab)
    all_f1, all_acc = eval_at(s_test, te_lab, t_all)
    print("RECAL k=all({}) t={:.3f} macroF1={:.4f} acc={:.4f}".format(
        len(va_ids), t_all, all_f1, all_acc))

    result = {
        "dataset": args.dataset,
        "ckpt": args.ckpt,
        "topk": args.topk,
        "mechanism": "threshold recalibration only (static train memory, "
                     "no memory update, no retraining)",
        "baseline_t0.5": {"macro_f1": base_f1, "acc": base_acc},
        "oracle_test": {"t": round(t_oracle, 3), "macro_f1": orc_f1,
                        "acc": orc_acc,
                        "note": "uses test labels; ceiling diagnostic only"},
        "curve": rows,
        "all_val": {"k": len(va_ids), "t": round(t_all, 3),
                    "single_class_guard": bool(guarded_all),
                    "macro_f1": all_f1, "acc": all_acc},
        "test_score_stats": {
            "mean": round(float(s_test.mean()), 4),
            "frac_ge_0.5": round(float((s_test >= 0.5).mean()), 4),
            "test_pos_rate": round(float(te_lab.mean()), 4),
        },
        "val_score_stats": {
            "mean": round(float(s_val.mean()), 4),
            "frac_ge_0.5": round(float((s_val >= 0.5).mean()), 4),
            "val_pos_rate": round(float(va_lab.mean()), 4),
        },
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print("[out] wrote {}".format(args.out))
    print("RESULT_JSON {}".format(json.dumps({
        "dataset": args.dataset,
        "baseline_macro_f1": base_f1,
        "recal_all_val_macro_f1": all_f1,
        "oracle_macro_f1": orc_f1,
    })))


if __name__ == "__main__":
    main()
