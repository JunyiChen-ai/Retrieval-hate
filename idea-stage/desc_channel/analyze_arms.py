"""DESC_CHANNEL step 2 analysis -- frozen readout and frozen decision rule.

Epoch selection and metric readout are imported verbatim from
scripts/rgcl_ablation_analyze.py::parse_run (I1 head rung), per FREEZE section 6.

Also recomputes per-sample TEST head predictions offline from the selected epoch's
checkpoint, so the DEFECT-subset readout (FREEZE section 7 item 4) can be reported.
The recomputation is verified against the trainlog's own test macro-F1 at that epoch.
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
from defect import is_defect, load_gt  # noqa: E402
from rgcl_ablation_analyze import parse_run  # noqa: E402

ARMS = ["A0", "T", "B", "G", "Bmis", "Gmis", "N"]
SEEDS = [0, 1, 2]
GROUP = "DESC_CHANNEL_20260813"
MODEL = "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF"
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


def roc_auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def exp_dir(arm, seed):
    pat = os.path.join(ROOT, "logging", "Retrieval", "HateMM", GROUP,
                       "*_seed{}_hybrid_loss_LORA_{}{}_cm-none".format(
                           seed, arm, "" if arm == "A0" else "_arc-stream"))
    d = sorted(glob.glob(pat))
    if len(d) != 1:
        raise SystemExit("exp dir glob %s -> %s" % (pat, d))
    return d[0]


def head_predictions(arm, seed, epoch, featsdir):
    """-> (ids, labels, scores) on test_seen from the epoch-`epoch` checkpoint."""
    from data_loader.dataset import load_archive_feats_split, load_feats_from_CLIP
    from model.classifier import classifier_hateClipper
    from run_rac import classifier_hateClipperArchive

    _, _, test = load_feats_from_CLIP(os.path.join(ROOT, "data", "CLIP_Embedding"),
                                      "HateMM", MODEL)
    ids = list(test[0])
    img = test[1].float()
    txt = test[2].float()
    labels = np.asarray([int(x) for x in test[3]])
    archive_dim = 0
    if arm != "A0":
        arc = load_archive_feats_split(
            os.path.join(featsdir, "test_seen_%s.pt" % arm), ids)
        archive_dim = arc.shape[1]
        txt = torch.cat((txt, arc), dim=1)

    import run_rac as RR
    argv = sys.argv
    sys.argv = ["run_rac.py", "--batch_size", "64", "--lr", "0.0001", "--epochs", "30",
                "--topk", "20", "--dataset", "HateMM", "--model", MODEL,
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
    if archive_dim:
        model = classifier_hateClipperArchive(
            img.shape[1], txt.shape[1] - archive_dim, archive_dim, margs.num_layers,
            margs.proj_dim, margs.map_dim, margs.fusion_mode, dropout=margs.dropout,
            batch_norm=margs.batch_norm, args=margs)
    else:
        model = classifier_hateClipper(
            img.shape[1], txt.shape[1], margs.num_layers, margs.proj_dim, margs.map_dim,
            margs.fusion_mode, dropout=margs.dropout, batch_norm=margs.batch_norm,
            args=margs)
    ck = sorted(glob.glob(os.path.join(glob.escape(exp_dir(arm, seed)), "ckpt",
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
    ap.add_argument("--logdir", default="logging/runs/desc_channel/logs")
    ap.add_argument("--featsdir", default="idea-stage/desc_channel/feats")
    ap.add_argument("--out", default="idea-stage/desc_channel/results.json")
    ap.add_argument("--no_subset", action="store_true")
    ap.add_argument("--group", default=GROUP)
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    globals()["GROUP"] = a.group
    arms = [x for x in a.arms.split(",") if x]

    gt = load_gt(ROOT)
    res, subset = {}, {}
    for arm in arms:
        for seed in SEEDS:
            p = os.path.join(a.logdir, "%s_s%d.trainlog" % (arm, seed))
            parsed = parse_run(p) if os.path.exists(p) else None
            if parsed is None:
                print("MISSING/UNPARSEABLE", p)
                continue
            res[(arm, seed)] = parsed["I1"]

    # test ROC at the selected epoch, straight from the trainlog
    import re
    RE_TEST_HEAD = re.compile(
        r"^test Epoch (\d+) acc: ([\d.]+) roc: ([\d.]+) pre: [\d.]+ recall: [\d.]+ "
        r"f1: [\d.]+ \| macroF1: ([\d.]+)", re.M)
    for (arm, seed), r in res.items():
        txt = open(os.path.join(a.logdir, "%s_s%d.trainlog" % (arm, seed)),
                   errors="replace").read()
        for m in RE_TEST_HEAD.finditer(txt):
            if int(m.group(1)) == r["epoch"]:
                r["test_roc"] = float(m.group(3))
                r["test_acc"] = float(m.group(2))

    if not a.no_subset:
        for (arm, seed), r in sorted(res.items()):
            ids, y, s = head_predictions(arm, seed, r["epoch"],
                                         os.path.join(ROOT, a.featsdir))
            pred = (s >= 0.5).astype(int)
            chk = macro_f1(y, pred)
            dm = np.asarray([is_defect(gt[v]["text"]) for v in ids])
            em = np.asarray([not (gt[v]["text"] or "").strip() for v in ids])
            subset[(arm, seed)] = {
                "recomputed_test_macro_f1": chk,
                "trainlog_test_macro_f1": r["test"],
                "match": abs(chk - r["test"]) < 5e-4,
                "n_defect": int(dm.sum()), "n_empty": int(em.sum()),
                "correct_all": int((pred == y).sum()), "n_all": len(y),
                "correct_defect": int((pred[dm] == y[dm]).sum()),
                "correct_empty": int((pred[em] == y[em]).sum()),
                "correct_clean": int((pred[~dm] == y[~dm]).sum()),
                "n_clean": int((~dm).sum()),
                "macro_f1_defect": macro_f1(y[dm], pred[dm]),
                "roc_defect": roc_auc(y[dm], s[dm]),
            }
            print(arm, seed, "ep", r["epoch"], "recomp %.4f vs log %.4f %s"
                  % (chk, r["test"], "OK" if subset[(arm, seed)]["match"] else "MISMATCH"))

    # ------------------------------------------------------------- tables
    def vals(arm, k="test"):
        return [res[(arm, s)][k] for s in SEEDS if (arm, s) in res]

    def paired(a1, a2, k="test"):
        d = [res[(a1, s)][k] - res[(a2, s)][k] for s in SEEDS
             if (a1, s) in res and (a2, s) in res]
        return float(np.mean(d)), d

    out = {"per_run": {"%s_s%d" % k: v for k, v in res.items()},
           "subset": {"%s_s%d" % k: v for k, v in subset.items()},
           "table": {}, "deltas": {}}
    print("\n%-6s %-32s %-32s %s" % ("arm", "test macroF1 mean±std", "test ROC mean±std",
                                     "per-seed test macroF1"))
    for arm in arms:
        v = vals(arm)
        if not v:
            continue
        rc = vals(arm, "test_roc")
        out["table"][arm] = {
            "test_macro_f1": v, "test_macro_f1_mean": float(np.mean(v)),
            "test_macro_f1_std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "test_roc": rc, "test_roc_mean": float(np.mean(rc)) if rc else None,
            "test_roc_std": float(np.std(rc, ddof=1)) if len(rc) > 1 else 0.0,
            "val_macro_f1": vals(arm, "val"), "epochs": vals(arm, "epoch"),
        }
        print("%-6s %.4f ± %.4f%17s %.4f ± %.4f%13s %s"
              % (arm, np.mean(v), np.std(v, ddof=1), "",
                 np.mean(rc) if rc else float("nan"),
                 np.std(rc, ddof=1) if rc else float("nan"), "",
                 " ".join("%.4f" % x for x in v)))

    print("\npaired deltas (test macro-F1)")
    for a1, a2 in [("T", "A0"), ("B", "A0"), ("G", "A0"), ("Bmis", "A0"),
                   ("Gmis", "A0"), ("N", "A0"), ("G", "T"), ("B", "T"), ("G", "B")]:
        if not (vals(a1) and vals(a2)):
            continue
        m, d = paired(a1, a2)
        out["deltas"]["%s-%s" % (a1, a2)] = {"mean": m, "per_seed": d}
        print("  %-8s %+.4f   per-seed %s" % ("%s-%s" % (a1, a2), m,
                                              " ".join("%+.4f" % x for x in d)))

    # ------------------------------------------------------------- verdict
    g = out["deltas"].get("G-A0")
    gm = out["deltas"].get("Gmis-A0")
    n = out["deltas"].get("N-A0")
    b = out["deltas"].get("B-A0")
    gt_ = out["deltas"].get("G-T")
    if g and gm and n:
        c1 = g["mean"] >= THRESH
        c2 = all(x > 0 for x in g["per_seed"]) and len(g["per_seed"]) == 3
        c3 = (gm["mean"] < 0.5 * g["mean"]) and (gm["mean"] < THRESH)
        c4 = n["mean"] < THRESH
        verdict = "GO" if (c1 and c2 and c3 and c4) else "KILL"
        out["verdict"] = {"clause1_mean_ge_0.005": bool(c1), "clause2_3of3_positive": bool(c2),
                          "clause3_mismatch_control": bool(c3), "clause4_noise_control": bool(c4),
                          "verdict": verdict,
                          "clause5_gate_has_no_value": bool(b and b["mean"] >= g["mean"]),
                          "clause6_third_stream_artefact": bool(
                              g["mean"] >= THRESH and gt_ and gt_["mean"] <= 0)}
        print("\nFROZEN VERDICT: %s  %s" % (verdict, json.dumps(out["verdict"])))

    json.dump({k: (v if k != "per_run" and k != "subset" else v) for k, v in out.items()},
              open(os.path.join(ROOT, a.out), "w"), indent=1, default=str)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
