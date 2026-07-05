#!/usr/bin/env python
"""P1: zero-label drift recalibration via MLLM prior estimation (adjusted classify-count).

research-wiki/EXP_p1_zerolabel_recal.md. CPU-only, read-only reuse of the W4 machinery
(src/eval_temporal_memory.py, scripts/analysis/temporal_recalibration.py).

Pipeline (all quantities are LABEL-FREE for the test era; only train-era labels, which the
method always has, and unlabeled test SCORES are used):

  1. score s(x) = rank-weighted top-20 neighbour-label fraction vs the STATIC temporal-train
     kNN memory (identical to W4; s>=t decision).
  2. MLLM verdict v(x) in {HARMFUL, BENIGN} from the archive alone (judge_archive_harmful.py,
     label NEVER shown). Bias-correct with adjusted classify-and-count:
        TPR = P(v=HARMFUL | gold=1) and FPR = P(v=HARMFUL | gold=0) on the TRAIN era (free),
        CC  = P(v=HARMFUL) on the unlabeled TEST era,
        p_hat = clamp((CC - FPR) / (TPR - FPR), 0, 1).
  3. Recalibrate the vote threshold by QUANTILE MATCHING: pick t so the predicted positive
     rate on the unlabeled test era equals p_hat. Zero labels, zero retraining.
  4. Drift trigger (pre-registered): recalibrate ONLY if the trigger fires; else keep t=0.5.
        primary : |p_hat - train_prior| > tau           (as instructed)
        refined : |p_hat - r0| > tau, r0 = frac(s_test>=0.5)  (operating-point gap; zero-label)

Conditions reported: (a) static t=0.5, (b) k=20 labelled recal [W4 reproduction],
(c) MLLM zero-label recal (ours; gated + forced), (d) oracle-prior recal (true test prior),
(e) uncorrected classify-and-count (no TPR/FPR correction), (f) sensitivity (TPR/FPR from 50%
of the train era). Bootstrap CIs over test samples for the single-threshold conditions.

Test-era gold labels are used ONLY to (i) compute final metrics and (ii) report the oracle
prior / diagnostic verdict quality — never inside the method that picks the threshold.

Usage:
  python scripts/analysis/p1_prior_recal.py --dataset MHC_temporal \
      --ckpt <best_model.pt> --verdicts scripts/analysis/p1_out/harmful_verdicts.json \
      --version v2 --out logging/temporal_memory/MHC_temporal_p1_zerolabel.json
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

from eval_cross_dataset import project_split
from eval_temporal_memory import (
    TEMPORAL_BASE, load_temporal_feats, load_temporal_records, build_head)
# reuse the EXACT W4 scoring + labelled-recal primitives (guarantees (a)/(b) reproduction)
from temporal_recalibration import rank_weighted_scores, eval_at, best_threshold

TAU = 0.05


def clamp01(x):
    return float(min(1.0, max(0.0, x)))


def threshold_for_rate(scores, target_rate):
    """Quantile matching: threshold t minimizing |frac(scores>=t) - target_rate|.
    Deterministic; on exact ties prefer the HIGHER threshold (fewer positives)."""
    uniq = np.unique(scores)
    cands = [uniq[0] - 1e-6] + list((uniq[:-1] + uniq[1:]) / 2.0) + [uniq[-1] + 1e-6]
    best = None  # (err, -t)
    best_t, best_rate = 0.5, None
    for t in sorted(set(cands)):
        rate = float((scores >= t).mean())
        key = (abs(rate - target_rate), -t)
        if best is None or key < best:
            best, best_t, best_rate = key, float(t), rate
    return best_t, best_rate


def bootstrap_f1_ci(scores, gold, t, n_boot=2000, seed=0, alpha=0.05):
    """95% percentile bootstrap CI of macro-F1 at a FIXED threshold t (test-sample noise)."""
    rng = np.random.default_rng(seed)
    n = len(gold)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(f1_score(gold[idx], (scores[idx] >= t).astype(int),
                             average="macro", zero_division=0))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return round(float(lo), 4), round(float(hi), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(TEMPORAL_BASE))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--verdicts", required=True,
                    help="JSON from judge_archive_harmful.py")
    ap.add_argument("--version", default="v2")
    ap.add_argument("--model", default="openai_clip-vit-large-patch14-336_HF")
    ap.add_argument("--path", default="./data/")
    ap.add_argument("--gt_root", default="./data/gt")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--k_recal", type=int, default=20, help="labelled-recal k for (b)")
    ap.add_argument("--n_seeds", type=int, default=5, help="seeds for (b)")
    ap.add_argument("--n_sub_seeds", type=int, default=20, help="subsamples for (f)")
    ap.add_argument("--out", default="")
    # head architecture (must match the trained checkpoint)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--proj_dim", type=int, default=1024)
    ap.add_argument("--map_dim", type=int, default=1024)
    ap.add_argument("--fusion_mode", type=str, default="align")
    ap.add_argument("--dropout", type=float, nargs=3, default=[0.2, 0.4, 0.1])
    ap.add_argument("--batch_norm", type=lambda x: str(x).lower() == "true", default=False)
    args = ap.parse_args()

    base = TEMPORAL_BASE[args.dataset]
    clip_path = os.path.join(args.path, "CLIP_Embedding")

    # ---- features -> projected fused space -> static-memory vote scores ----
    train, val, test = load_temporal_feats(
        clip_path, args.dataset, args.model, gt_root=args.gt_root)
    model = build_head(train[1].shape[1], train[2].shape[1], args)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.to("cpu")
    tr_ids, tr_emb, tr_lab = project_split(model, train, "cpu")
    va_ids, va_emb, va_lab = project_split(model, val, "cpu")
    te_ids, te_emb, te_lab = project_split(model, test, "cpu")
    assert not (set(te_ids) & set(tr_ids)) and not (set(te_ids) & set(va_ids))

    s_val = rank_weighted_scores(tr_emb, tr_lab, va_emb, args.topk)
    s_test = rank_weighted_scores(tr_emb, tr_lab, te_emb, args.topk)
    tr_gold = tr_lab.astype(int)
    te_gold = te_lab.astype(int)
    train_prior = float(tr_gold.mean())
    true_test_prior = float(te_gold.mean())
    r0 = float((s_test >= 0.5).mean())  # current operating-point positive rate (unlabeled)

    # ---- MLLM label-free verdicts -> harmful flags ----
    V = json.load(open(args.verdicts))[base][args.version]

    def flags(ids):
        return np.array([1 if V[i]["verdict"] == "HARMFUL" else 0 for i in ids], dtype=int)

    tr_pred, va_pred, te_pred = flags(tr_ids), flags(va_ids), flags(te_ids)
    # consistency: the verdict file's bookkeeping label must equal the split gold
    for ids, gold in ((tr_ids, tr_gold), (te_ids, te_gold)):
        assert all(int(V[i]["label"]) == int(g) for i, g in zip(ids, gold)), \
            "verdict-file label disagrees with temporal-split gold"

    # ---- adjusted classify-and-count on the TRAIN era (labels are free) ----
    TPR = float(tr_pred[tr_gold == 1].mean())
    FPR = float(tr_pred[tr_gold == 0].mean())
    CC_test = float(te_pred.mean())
    degenerate = (TPR - FPR) < 1e-6
    p_hat = clamp01(CC_test) if degenerate else clamp01((CC_test - FPR) / (TPR - FPR))

    # diagnostic: verdict quality + cross-era TPR/FPR stability (uses test labels; NOT in method)
    TPR_te = float(te_pred[te_gold == 1].mean()) if te_gold.sum() else float("nan")
    FPR_te = float(te_pred[te_gold == 0].mean()) if (te_gold == 0).sum() else float("nan")
    verdict_diag = {
        "train_TPR": round(TPR, 4), "train_FPR": round(FPR, 4),
        "test_TPR_diag": round(TPR_te, 4), "test_FPR_diag": round(FPR_te, 4),
        "train_verdict_acc": round(float((tr_pred == tr_gold).mean()), 4),
        "test_verdict_acc": round(float((te_pred == te_gold).mean()), 4),
        "CC_train": round(float(tr_pred.mean()), 4), "CC_test": round(CC_test, 4),
    }

    # ---------------- conditions ----------------
    # (a) static t=0.5
    a_f1, a_acc = eval_at(s_test, te_gold, 0.5)

    # (b) k=20 labelled recal (EXACT W4 mechanism) -> reproduce 0.7336
    b_seeds = []
    for s in range(args.n_seeds):
        rng = np.random.default_rng(1000 + s)
        sel = np.sort(rng.choice(len(va_ids), size=min(args.k_recal, len(va_ids)),
                                 replace=False))
        t_star, guarded = best_threshold(s_val[sel], va_lab[sel])
        f1, acc = eval_at(s_test, te_gold, t_star)
        b_seeds.append({"seed": 1000 + s, "t": round(t_star, 3),
                        "guarded": bool(guarded), "macro_f1": f1, "acc": acc})
    b_f1 = round(float(np.mean([r["macro_f1"] for r in b_seeds])), 4)
    b_f1_std = round(float(np.std([r["macro_f1"] for r in b_seeds])), 4)
    b_acc = round(float(np.mean([r["acc"] for r in b_seeds])), 4)

    # triggers
    fire_primary = abs(p_hat - train_prior) > args.tau
    fire_refined = abs(p_hat - r0) > args.tau

    # (c) MLLM zero-label recal (ours). gated_* use the trigger; forced ignores it.
    t_forced, rate_forced = threshold_for_rate(s_test, p_hat)
    c_forced_f1, c_forced_acc = eval_at(s_test, te_gold, t_forced)
    t_primary = t_forced if fire_primary else 0.5
    t_refined = t_forced if fire_refined else 0.5
    c_primary_f1, c_primary_acc = eval_at(s_test, te_gold, t_primary)
    c_refined_f1, c_refined_acc = eval_at(s_test, te_gold, t_refined)

    # (d) oracle-prior recal (quantile-match to the TRUE test prior)
    t_oracle, rate_oracle = threshold_for_rate(s_test, true_test_prior)
    d_f1, d_acc = eval_at(s_test, te_gold, t_oracle)

    # (e) uncorrected classify-and-count (quantile-match to raw CC; no bias correction)
    t_cc, rate_cc = threshold_for_rate(s_test, clamp01(CC_test))
    e_f1, e_acc = eval_at(s_test, te_gold, t_cc)

    # (f) sensitivity: TPR/FPR from 50% of the train era
    f_rows = []
    for s in range(args.n_sub_seeds):
        rng = np.random.default_rng(2000 + s)
        idx = rng.choice(len(tr_ids), size=len(tr_ids) // 2, replace=False)
        g, p = tr_gold[idx], tr_pred[idx]
        if g.sum() == 0 or (g == 0).sum() == 0:
            continue
        tpr, fpr = float(p[g == 1].mean()), float(p[g == 0].mean())
        ph = clamp01(CC_test) if (tpr - fpr) < 1e-6 else clamp01((CC_test - fpr) / (tpr - fpr))
        fire = abs(ph - train_prior) > args.tau
        t_s = threshold_for_rate(s_test, ph)[0] if fire else 0.5
        f1_s, acc_s = eval_at(s_test, te_gold, t_s)
        f_rows.append({"seed": 2000 + s, "p_hat": round(ph, 4), "fire": bool(fire),
                       "macro_f1": f1_s, "acc": acc_s})
    f_phats = [r["p_hat"] for r in f_rows]
    f_f1s = [r["macro_f1"] for r in f_rows]
    sens = {
        "n": len(f_rows),
        "p_hat_mean": round(float(np.mean(f_phats)), 4),
        "p_hat_std": round(float(np.std(f_phats)), 4),
        "macro_f1_mean": round(float(np.mean(f_f1s)), 4),
        "macro_f1_std": round(float(np.std(f_f1s)), 4),
        "fire_frac": round(float(np.mean([r["fire"] for r in f_rows])), 3),
        "per_seed": f_rows,
    }

    # bootstrap CIs over test samples (single-threshold conditions)
    ci = {
        "a_static": bootstrap_f1_ci(s_test, te_gold, 0.5),
        "c_forced": bootstrap_f1_ci(s_test, te_gold, t_forced),
        "d_oracle_prior": bootstrap_f1_ci(s_test, te_gold, t_oracle),
        "e_uncorrected_cc": bootstrap_f1_ci(s_test, te_gold, t_cc),
    }

    gap = round(b_f1 - a_f1, 4)
    recovered_primary = round((c_primary_f1 - a_f1) / (b_f1 - a_f1), 3) if gap else None
    recovered_forced = round((c_forced_f1 - a_f1) / (b_f1 - a_f1), 3) if gap else None

    result = {
        "dataset": args.dataset, "ckpt": args.ckpt,
        "verdicts": args.verdicts, "version": args.version,
        "topk": args.topk, "tau": args.tau,
        "priors": {"train_prior": round(train_prior, 4),
                   "true_test_prior": round(true_test_prior, 4),
                   "operating_point_rate_r0": round(r0, 4)},
        "prior_estimation": {
            "TPR": round(TPR, 4), "FPR": round(FPR, 4), "CC_test": round(CC_test, 4),
            "p_hat": round(p_hat, 4), "degenerate": degenerate,
            "abs_err_vs_true_prior": round(abs(p_hat - true_test_prior), 4)},
        "verdict_diag": verdict_diag,
        "triggers": {"fire_primary_|phat-trainprior|>tau": bool(fire_primary),
                     "fire_refined_|phat-r0|>tau": bool(fire_refined),
                     "primary_gap": round(abs(p_hat - train_prior), 4),
                     "refined_gap": round(abs(p_hat - r0), 4)},
        "conditions": {
            "a_static_t0.5": {"macro_f1": a_f1, "acc": a_acc, "ci95": ci["a_static"]},
            "b_labelled_k{}".format(args.k_recal): {
                "macro_f1": b_f1, "macro_f1_std": b_f1_std, "acc": b_acc,
                "per_seed": b_seeds},
            "c_zerolabel_primary_gated": {"macro_f1": c_primary_f1, "acc": c_primary_acc,
                                          "fired": bool(fire_primary), "t": round(t_primary, 4)},
            "c_zerolabel_refined_gated": {"macro_f1": c_refined_f1, "acc": c_refined_acc,
                                          "fired": bool(fire_refined), "t": round(t_refined, 4)},
            "c_zerolabel_forced": {"macro_f1": c_forced_f1, "acc": c_forced_acc,
                                   "t": round(t_forced, 4), "pred_pos_rate": round(rate_forced, 4),
                                   "ci95": ci["c_forced"]},
            "d_oracle_prior": {"macro_f1": d_f1, "acc": d_acc, "t": round(t_oracle, 4),
                               "pred_pos_rate": round(rate_oracle, 4),
                               "ci95": ci["d_oracle_prior"]},
            "e_uncorrected_cc": {"macro_f1": e_f1, "acc": e_acc, "t": round(t_cc, 4),
                                 "pred_pos_rate": round(rate_cc, 4),
                                 "ci95": ci["e_uncorrected_cc"]},
            "f_sensitivity_half_train": sens,
        },
        "gap_b_minus_a": gap,
        "recovered_frac_primary_gated": recovered_primary,
        "recovered_frac_forced": recovered_forced,
    }

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print("[out] wrote", args.out)

    # ---- readable summary ----
    print("\n===== P1 zero-label recal :: {} =====".format(args.dataset))
    print("priors: train={:.4f} true_test={:.4f}  operating_point_r0={:.4f}".format(
        train_prior, true_test_prior, r0))
    print("verdict: trainTPR={:.3f} trainFPR={:.3f} | (diag testTPR={:.3f} testFPR={:.3f}) "
          "acc tr={:.3f} te={:.3f}".format(
              TPR, FPR, TPR_te, FPR_te, verdict_diag["train_verdict_acc"],
              verdict_diag["test_verdict_acc"]))
    print("CC_test={:.4f} -> p_hat={:.4f}  |p_hat-true_prior|={:.4f}".format(
        CC_test, p_hat, abs(p_hat - true_test_prior)))
    print("trigger primary(|phat-trainprior|>{})={}  refined(|phat-r0|>{})={}".format(
        args.tau, fire_primary, args.tau, fire_refined))
    print("(a) static t=0.5        F1={:.4f} acc={:.4f}  CI{}".format(a_f1, a_acc, ci["a_static"]))
    print("(b) labelled k={:<3d}      F1={:.4f}+-{:.4f} acc={:.4f}".format(
        args.k_recal, b_f1, b_f1_std, b_acc))
    print("(c) zero-label forced   F1={:.4f} acc={:.4f} t={:.3f} rate={:.3f}  CI{}".format(
        c_forced_f1, c_forced_acc, t_forced, rate_forced, ci["c_forced"]))
    print("(c) zero-label primary  F1={:.4f} acc={:.4f} (fired={})".format(
        c_primary_f1, c_primary_acc, fire_primary))
    print("(c) zero-label refined  F1={:.4f} acc={:.4f} (fired={})".format(
        c_refined_f1, c_refined_acc, fire_refined))
    print("(d) oracle-prior        F1={:.4f} acc={:.4f} t={:.3f} rate={:.3f}".format(
        d_f1, d_acc, t_oracle, rate_oracle))
    print("(e) uncorrected CC      F1={:.4f} acc={:.4f} t={:.3f} rate={:.3f}".format(
        e_f1, e_acc, t_cc, rate_cc))
    print("(f) half-train sens     p_hat={:.4f}+-{:.4f}  F1={:.4f}+-{:.4f} fire_frac={}".format(
        sens["p_hat_mean"], sens["p_hat_std"], sens["macro_f1_mean"], sens["macro_f1_std"],
        sens["fire_frac"]))
    print("gap(b-a)={:.4f}  recovered: forced={} primary_gated={}".format(
        gap, recovered_forced, recovered_primary))


if __name__ == "__main__":
    main()
