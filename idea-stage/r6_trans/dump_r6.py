"""R6-2 step 2 -- dump per-item head probabilities AND fused embeddings on dev_seen
and test_seen at the val-selected epoch.

Approach reused from idea-stage/arbiter/dump_probs.py: re-run the head with
--keep_epoch_ckpts True, pick the epoch on the validation split only, reload that
epoch's checkpoint, forward the frozen features once, and cross-check the recomputed
macro-F1 at threshold 0.5 against the number the trainlog printed for that epoch, so
the dump is provably the same model the trainlog scored.

Difference from arbiter: the R6 freeze (idea-stage/R6_PILOT_FREEZE_2026-08-17.md, head
of file) states the epoch is "selected on val by validation macro-F1", so the selection
key here is the dev head macro-F1 (ties -> earliest epoch), not arbiter's (acc, roc).
Only epochs >= warmup (5) are eligible, matching the training-time warmup floor.

Test labels are loaded only so the final scoring step can use them; nothing in this
file or in em_r6.py selects on them.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

GROUP = "R6TRANS_20260817"
WARMUP = 5
ENC = {
    "HateMM": "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
    "MHC": "Qwen2.5-VL-7B-Instruct_HF",
    "MHC_zh": "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
    "ImpliHateVid": "openai_clip-vit-large-patch14-336_HF",
}
LOGDIR = os.path.join(ROOT, "logging", "runs", "r6_trans", "logs")
OUTDIR = os.path.join(HERE, "dumps")

RE_DEV = re.compile(
    r"^dev\s+Epoch (\d+) acc: [\d.]+ roc: [\d.]+ pre: [\d.]+ recall: [\d.]+ "
    r"f1: [\d.]+ loss: [\d.]+ \| macroF1: ([\d.]+)", re.M)
RE_TEST = re.compile(
    r"^test Epoch (\d+) acc: [\d.]+ roc: [\d.]+ pre: [\d.]+ recall: [\d.]+ "
    r"f1: [\d.]+ \| macroF1: ([\d.]+)", re.M)


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


def exp_dir(dataset, seed):
    pat = os.path.join(ROOT, "logging", "Retrieval", dataset, GROUP,
                       "*_seed%d_hybrid_loss_R6T_cm-none" % seed)
    d = sorted(glob.glob(pat))
    if len(d) != 1:
        raise SystemExit("exp dir glob %s -> %s" % (pat, d))
    return d[0]


def build_model(dataset, model_tag, img_dim, txt_dim, seed):
    from model.classifier import classifier_hateClipper
    import run_rac as RR
    argv = sys.argv
    sys.argv = ["run_rac.py", "--batch_size", "64", "--lr", "0.0001", "--epochs", "30",
                "--topk", "20", "--dataset", dataset, "--model", model_tag,
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


def select_epoch(trainlog):
    txt = open(trainlog, errors="replace").read()
    dev = {int(m.group(1)): float(m.group(2)) for m in RE_DEV.finditer(txt)}
    test = {int(m.group(1)): float(m.group(2)) for m in RE_TEST.finditer(txt)}
    if not dev:
        raise SystemExit("no dev epochs parsed from %s" % trainlog)
    cand = sorted([e for e in dev if e >= WARMUP]) or sorted(dev)
    # argmax of dev macro-F1, ties -> earliest epoch
    best = max(cand, key=lambda e: (dev[e], -e))
    return best, dev[best], test.get(best)


def dump_one(dataset, seed):
    from data_loader.dataset import load_feats_from_CLIP
    model_tag = ENC[dataset]
    trainlog = os.path.join(LOGDIR, "%s_s%d.trainlog" % (dataset, seed))
    ep, dev_f1_log, test_f1_log = select_epoch(trainlog)

    d = exp_dir(dataset, seed)
    ck = sorted(glob.glob(os.path.join(glob.escape(d), "ckpt",
                                       "epoch_model_%d_*.pt" % ep)))
    if not ck:
        ck = sorted(glob.glob(os.path.join(glob.escape(d), "ckpt",
                                           "best_model_%d_*.pt" % ep)))
    if not ck:
        raise SystemExit("no checkpoint for %s seed=%s epoch=%s" % (dataset, seed, ep))

    train, dev, test = load_feats_from_CLIP(
        os.path.join(ROOT, "data", "CLIP_Embedding"), dataset, model_tag)
    y_train = np.asarray([int(x) for x in train[3]])

    model = None
    rec = {"dataset": dataset, "model": model_tag, "seed": seed, "epoch": int(ep),
           "ckpt": os.path.basename(ck[-1]),
           "trainlog_dev_macro_f1": dev_f1_log, "trainlog_test_macro_f1": test_f1_log,
           "train_prior_pos": float(y_train.mean()), "n_train": int(len(y_train))}
    arrays = {"train_labels": y_train}
    for name, sp in (("dev", dev), ("test", test)):
        ids = [str(x) for x in sp[0]]
        img = sp[1].float()
        txt = sp[2].float()
        y = np.asarray([int(x) for x in sp[3]])
        if model is None:
            model = build_model(dataset, model_tag, img.shape[1], txt.shape[1], seed)
            model.load_state_dict(torch.load(ck[-1], map_location="cpu"))
            model.eval()
        with torch.no_grad():
            logits, embed = model(img, txt, return_embed=True)
        prob = torch.sigmoid(logits).squeeze(-1).numpy().astype(np.float64)
        z = embed.numpy().astype(np.float64)
        f1 = macro_f1(y, (prob >= 0.5).astype(int))
        ref = dev_f1_log if name == "dev" else test_f1_log
        rec[name] = {"n": int(len(y)), "dim": int(z.shape[1]),
                     "recomputed_macro_f1_at_0.5": f1,
                     "trainlog_macro_f1": ref,
                     "match": bool(ref is not None and abs(f1 - ref) < 5e-4)}
        arrays["%s_prob" % name] = prob
        arrays["%s_z" % name] = z
        arrays["%s_y" % name] = y
        arrays["%s_ids" % name] = np.asarray(ids)
        print("[dump] %s seed=%d ep=%d %s n=%d d=%d f1@0.5=%.4f trainlog=%s %s"
              % (dataset, seed, ep, name, len(y), z.shape[1], f1, ref,
                 "OK" if rec[name]["match"] else "MISMATCH"))
    os.makedirs(OUTDIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUTDIR, "%s_s%d.npz" % (dataset, seed)), **arrays)
    with open(os.path.join(OUTDIR, "%s_s%d.json" % (dataset, seed)), "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(ENC))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    bad = []
    for ds in a.datasets:
        for s in a.seeds:
            r = dump_one(ds, s)
            for split in ("dev", "test"):
                if not r[split]["match"]:
                    bad.append("%s_s%d/%s" % (ds, s, split))
    print("[dump] DONE mismatches=%s" % (bad or "none"))


if __name__ == "__main__":
    main()
