"""ARBITER step 1b -- dump per-video head probabilities on val (dev_seen) and test (test_seen).

For each seed the epoch is selected by the frozen I1 rule imported verbatim from
scripts/rgcl_ablation_analyze.py::parse_run (argmax over epochs >= warmup(5) of
(dev head acc, dev head roc)).  The epoch checkpoint is reloaded and sigmoid(logit) is
written per video.  The recomputed val/test macro-F1 is cross-checked against the trainlog
so the dump is provably the same model the trainlog scored.

Test labels are loaded here only to run that consistency check and to produce the final
report; no selection or tuning reads them (see ARBITER_FREEZE.md section 6).
"""
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from rgcl_ablation_analyze import parse_run  # noqa: E402

GROUP = "ARBITER_20260813"
MODEL = "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF"
SEEDS = [0, 1, 2]
LOGDIR = os.path.join(ROOT, "logging", "runs", "arbiter", "logs")
OUT = os.path.join(HERE, "head_probs.json")


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


def exp_dir(seed):
    pat = os.path.join(ROOT, "logging", "Retrieval", "HateMM", GROUP,
                       "*_seed%d_hybrid_loss_LORA_A0_cm-none" % seed)
    d = sorted(glob.glob(pat))
    if len(d) != 1:
        raise SystemExit("exp dir glob %s -> %s" % (pat, d))
    return d[0]


def build_model(img_dim, txt_dim, seed):
    from model.classifier import classifier_hateClipper
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
    return classifier_hateClipper(img_dim, txt_dim, margs.num_layers, margs.proj_dim,
                                  margs.map_dim, margs.fusion_mode, dropout=margs.dropout,
                                  batch_norm=margs.batch_norm, args=margs)


def main():
    from data_loader.dataset import load_feats_from_CLIP
    _, dev, test = load_feats_from_CLIP(os.path.join(ROOT, "data", "CLIP_Embedding"),
                                        "HateMM", MODEL)
    splits = {"val": dev, "test": test}
    out = {"group": GROUP, "model": MODEL, "seeds": {}}
    for seed in SEEDS:
        p = os.path.join(LOGDIR, "A0_s%d.trainlog" % seed)
        r = parse_run(p)
        if r is None:
            raise SystemExit("unparseable trainlog %s" % p)
        ep = r["I1"]["epoch"]
        ck = sorted(glob.glob(os.path.join(glob.escape(exp_dir(seed)), "ckpt",
                                           "epoch_model_%d_*.pt" % ep)))
        if not ck:
            ck = sorted(glob.glob(os.path.join(glob.escape(exp_dir(seed)), "ckpt",
                                               "best_model_%d_*.pt" % ep)))
        if not ck:
            raise SystemExit("no checkpoint for seed=%s epoch=%s" % (seed, ep))
        rec = {"epoch": ep, "trainlog_val_macro_f1": r["I1"]["val"],
               "trainlog_test_macro_f1": r["I1"]["test"], "ckpt": os.path.basename(ck[-1]),
               "splits": {}}
        model = None
        for name, sp in splits.items():
            ids = list(sp[0])
            img = sp[1].float()
            txt = sp[2].float()
            y = np.asarray([int(x) for x in sp[3]])
            if model is None:
                model = build_model(img.shape[1], txt.shape[1], seed)
                model.load_state_dict(torch.load(ck[-1], map_location="cpu"))
                model.eval()
            with torch.no_grad():
                logits = model(img, txt)
                if isinstance(logits, tuple):
                    logits = logits[0]
            prob = torch.sigmoid(logits).squeeze(-1).numpy()
            f1 = macro_f1(y, (prob >= 0.5).astype(int))
            key = "trainlog_%s_macro_f1" % name
            rec["splits"][name] = {
                "ids": ids, "labels": y.tolist(), "prob": [float(x) for x in prob],
                "recomputed_macro_f1": f1,
                "match": bool(abs(f1 - rec[key]) < 5e-4),
            }
            print("seed=%d %s ep=%d recomputed %.4f vs trainlog %.4f %s"
                  % (seed, name, ep, f1, rec[key],
                     "OK" if rec["splits"][name]["match"] else "MISMATCH"))
        out["seeds"][str(seed)] = rec
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
