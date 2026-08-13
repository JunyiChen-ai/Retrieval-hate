"""CAD analysis -- frozen readout and frozen decision rule (CAD_FREEZE.md section 5).

Epoch selection and metric readout imported verbatim from
scripts/rgcl_ablation_analyze.py::parse_run (I1 head rung), same as TEXT_MERGE.
Per-class precision/recall on test are recomputed offline from the selected epoch's
checkpoint and cross-checked against the trainlog's own test macro-F1.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from rgcl_ablation_analyze import parse_run  # noqa: E402

ARMS = ["A0", "CAD", "CTRLRAND"]
SEEDS = [0, 1, 2]
GROUP = "CAD_20260813"
PREFIX = "CAD"
THRESH = 0.005


def macro_f1(y, p):
    out = []
    for c in (0, 1):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(out))


def prf(y, p, c=1):
    tp = int(((p == c) & (y == c)).sum())
    fp = int(((p == c) & (y != c)).sum())
    fn = int(((p != c) & (y == c)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0), tp, fp, fn


def exp_dir(arm, seed, group):
    pat = os.path.join(ROOT, "logging", "Retrieval", "HateMM", group,
                       "*_seed{}_hybrid_loss_CAD_{}_cm-none".format(seed, arm))
    d = sorted(glob.glob(pat))
    if len(d) != 1:
        raise SystemExit("exp dir glob %s -> %s" % (pat, d))
    return d[0]


def head_predictions(arm, seed, epoch, group, prefix):
    from data_loader.dataset import load_feats_from_CLIP
    from model.classifier import classifier_hateClipper

    _, _, test = load_feats_from_CLIP(os.path.join(ROOT, "data", "CLIP_Embedding"),
                                      "HateMM", "%s-%s" % (prefix, arm))
    ids = list(test[0])
    img = test[1].float()
    txt = test[2].float()
    labels = np.asarray([int(x) for x in test[3]])

    import run_rac as RR
    argv = sys.argv
    sys.argv = ["run_rac.py", "--batch_size", "64", "--lr", "0.0001", "--epochs", "30",
                "--topk", "20", "--dataset", "HateMM", "--model", "%s-%s" % (prefix, arm),
                "--proj_dim", "1024", "--map_dim", "1024",
                "--dropout", "0.2", "0.4", "0.1", "--fusion_mode", "align",
                "--hard_negatives_loss", "True", "--no_hard_negatives", "1",
                "--final_eval", "False", "--seed", str(seed),
                "--metric", "cos", "--loss", "triplet", "--batch_norm", "False",
                "--hybrid_loss", "True", "--warmup", "5",
                "--majority_voting", "arithmetic", "--no_pseudo_gold_positives", "1",
                "--lambda_seg", "0", "--contrast_mode", "none",
                "--Faiss_GPU", "False"]
    try:
        margs = RR.parse_args()
    finally:
        sys.argv = argv
    margs.device = "cpu"
    model = classifier_hateClipper(
        img.shape[1], txt.shape[1], margs.num_layers, margs.proj_dim, margs.map_dim,
        margs.fusion_mode, dropout=margs.dropout, batch_norm=margs.batch_norm,
        args=margs)
    ck = sorted(glob.glob(os.path.join(glob.escape(exp_dir(arm, seed, group)), "ckpt",
                                       "epoch_model_%d_*.pt" % epoch)))
    if not ck:
        raise SystemExit("no checkpoint for arm=%s seed=%s epoch=%s" % (arm, seed, epoch))
    model.load_state_dict(torch.load(ck[-1], map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits = model(img, txt)
        if isinstance(logits, tuple):
            logits = logits[0]
    return ids, labels, torch.sigmoid(logits).squeeze(-1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="logging/runs/cad/logs")
    ap.add_argument("--out", default="idea-stage/cad/results.json")
    ap.add_argument("--group", default=GROUP)
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--no_subset", action="store_true")
    a = ap.parse_args()

    res, cls = {}, {}
    for arm in ARMS:
        for seed in SEEDS:
            p = os.path.join(ROOT, a.logdir, "%s_s%d.trainlog" % (arm, seed))
            parsed = parse_run(p) if os.path.exists(p) else None
            if parsed is None:
                print("MISSING/UNPARSEABLE", p)
                continue
            res[(arm, seed)] = parsed["I1"]

    import re
    RE_TEST_HEAD = re.compile(
        r"^test Epoch (\d+) acc: ([\d.]+) roc: ([\d.]+) pre: [\d.]+ recall: [\d.]+ "
        r"f1: [\d.]+ \| macroF1: ([\d.]+)", re.M)
    for (arm, seed), r in res.items():
        txt = open(os.path.join(ROOT, a.logdir, "%s_s%d.trainlog" % (arm, seed)),
                   errors="replace").read()
        for m in RE_TEST_HEAD.finditer(txt):
            if int(m.group(1)) == r["epoch"]:
                r["test_roc"] = float(m.group(3))
                r["test_acc"] = float(m.group(2))

    if not a.no_subset:
        for (arm, seed), r in sorted(res.items()):
            ids, y, s = head_predictions(arm, seed, r["epoch"], a.group, a.prefix)
            pred = (s >= 0.5).astype(int)
            chk = macro_f1(y, pred)
            p1, r1, f1, tp, fp, fn = prf(y, pred, 1)
            p0, r0, f0, _, _, _ = prf(y, pred, 0)
            cls[(arm, seed)] = {
                "recomputed_test_macro_f1": chk,
                "trainlog_test_macro_f1": r["test"],
                "match": abs(chk - r["test"]) < 5e-4,
                "hate_precision": p1, "hate_recall": r1, "hate_f1": f1,
                "nonhate_precision": p0, "nonhate_recall": r0, "nonhate_f1": f0,
                "tp": tp, "fp": fp, "fn": fn, "n_test": len(y),
            }
            print("%-9s s%d ep%-3d recomp %.4f vs log %.4f %s  hate P %.4f R %.4f"
                  % (arm, seed, r["epoch"], chk, r["test"],
                     "OK" if cls[(arm, seed)]["match"] else "MISMATCH", p1, r1))

    def vals(arm, k="test"):
        return [res[(arm, s)][k] for s in SEEDS if (arm, s) in res]

    def paired(a1, a2, k="test"):
        d = [res[(a1, s)][k] - res[(a2, s)][k] for s in SEEDS
             if (a1, s) in res and (a2, s) in res]
        return float(np.mean(d)), d

    out = {"per_run": {"%s_s%d" % k: v for k, v in res.items()},
           "per_class": {"%s_s%d" % k: v for k, v in cls.items()},
           "table": {}, "deltas": {}}
    print("\n%-10s %-22s %-22s %s" % ("arm", "test macroF1", "test ROC", "per-seed"))
    for arm in ARMS:
        v = vals(arm)
        if not v:
            continue
        rc = vals(arm, "test_roc")
        out["table"][arm] = {
            "test_macro_f1": v, "test_macro_f1_mean": float(np.mean(v)),
            "test_macro_f1_std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "test_roc": rc, "test_roc_mean": float(np.mean(rc)) if rc else None,
            "val_macro_f1": vals(arm, "val"), "epochs": vals(arm, "epoch"),
        }
        if cls:
            for k in ("hate_precision", "hate_recall"):
                xs = [cls[(arm, s)][k] for s in SEEDS if (arm, s) in cls]
                out["table"][arm][k + "_mean"] = float(np.mean(xs))
                out["table"][arm][k] = xs
        print("%-10s %.4f ± %.4f%9s %.4f%14s %s"
              % (arm, np.mean(v), np.std(v, ddof=1), "",
                 np.mean(rc) if rc else float("nan"), "",
                 " ".join("%.4f" % x for x in v)))

    print("\npaired deltas (test macro-F1)")
    for a1, a2 in [("CAD", "A0"), ("CTRLRAND", "A0"), ("CAD", "CTRLRAND")]:
        if not (vals(a1) and vals(a2)):
            continue
        m, d = paired(a1, a2)
        out["deltas"]["%s-%s" % (a1, a2)] = {"mean": m, "per_seed": d}
        print("  %-16s %+.4f   per-seed %s" % ("%s-%s" % (a1, a2), m,
                                               " ".join("%+.4f" % x for x in d)))

    # ------------------------------------------------------------- frozen verdict
    g = out["deltas"].get("CAD-A0")
    c = out["deltas"].get("CTRLRAND-A0")
    if g and c:
        c1 = g["mean"] >= THRESH
        c2 = len(g["per_seed"]) == 3 and all(x > 0 for x in g["per_seed"])
        c3 = c["mean"] < 0.5 * g["mean"]
        verdict = "GO" if (c1 and c2 and c3) else "KILL"
        out["verdict"] = {
            "clause1_cad_mean_ge_0.005": bool(c1),
            "clause2_3of3_positive": bool(c2),
            "clause3_ctrl_lt_half_cad": bool(c3),
            "note_ctrl_ge_cad_both_positive": bool(
                c["mean"] >= g["mean"] and c["mean"] > 0 and g["mean"] > 0),
            "verdict": verdict,
        }
        print("\nFROZEN VERDICT: %s  %s" % (verdict, json.dumps(out["verdict"])))

    with open(os.path.join(ROOT, a.out), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
