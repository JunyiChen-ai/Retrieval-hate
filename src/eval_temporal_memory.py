"""
W4: "Evolving-memory" protocol evaluation on the temporal splits (DESIGN_iter3 S3).

Protocol
--------
The head is trained ONCE on temporal-train (past period). At deployment time
we NEVER retrain the head; we only change the kNN memory bank:

  (i)   static memory      = temporal-train fused embeddings (past only)
  (ii)  memory augmentation = temporal-train + k labelled samples drawn from
        the "new-period stream" (= temporal-val, whose upload dates lie
        strictly between train-period and test-period), k in {5,10,20,50}.
        Strategies for picking the k samples:
           random    - uniform from the val pool (n_seeds seeds, mean+-std)
           latest    - k most recent by upload_date
           uncertain - k smallest kNN vote margin |2p-1| vs the static
                       train memory (no test information used)
  (iii) reference           = the same protocol on the RANDOM split
        (run this script with --dataset MHC / MHC_zh and the RAC_video_CLIP
        head; --augment False).

Leak-safety: temporal-test NEVER enters the memory; k is NEVER tuned on test
(we report the full curve over k). Majority-class baselines are recomputed
for every split because the temporal test positive rate (~24-25%) is lower
than train (~34%) (survivor + temporal bias).

Everything is READ-ONLY reuse of existing modules:
  - eval_cross_dataset.project_split / knn_vote  (memory build + kNN eval)
  - model.classifier.classifier_hateClipper      (the trained head)
  - data_loader.dataset.load_feats_from_CLIP     (random-split cache loader)
The temporal split shares the SAME underlying cached CLIP features as the
random split; load_temporal_feats() just re-indexes them by video id.

Usage
-----
  python src/eval_temporal_memory.py \
      --dataset MHC_temporal --ckpt <best_model.pt> \
      --augment True --k_list 5,10,20,50 \
      --strategies random,latest,uncertain --n_seeds 5 \
      --out logging/temporal_memory/MHC_temporal_evolving_memory.json
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json

import numpy as np
import torch
import faiss
from easydict import EasyDict
from sklearn.metrics import f1_score

from model.classifier import classifier_hateClipper
from data_loader.dataset import load_feats_from_CLIP
from eval_cross_dataset import project_split, knn_vote

# temporal split name -> base dataset whose CLIP-feature cache we re-index
TEMPORAL_BASE = {"MHC_temporal": "MHC", "MHC_zh_temporal": "MHC_zh"}


# ------------------------------------------------------------------ #
#  temporal split assembly (read-only re-index of the shared cache)
# ------------------------------------------------------------------ #
def load_temporal_records(gt_root, temporal_dataset, split):
    """Read data/gt/<temporal_dataset>/<split>.jsonl -> list of dicts
    (id / text / label / upload_date)."""
    recs = []
    path = os.path.join(gt_root, temporal_dataset, split + ".jsonl")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def load_temporal_feats(clip_path, temporal_dataset, model, gt_root=None):
    """Assemble the temporal train/val/test feature tuples
    [ids, img_feats, text_feats, labels] by re-indexing the BASE dataset's
    cached random-split CLIP embeddings by video id (the temporal split is a
    re-partition of the same universe, so the union of the three random-split
    caches covers every temporal id).

    Returns (train, val, test), each in the exact format produced by
    data_loader.dataset.load_feats_split (so it is a drop-in replacement for
    load_feats_from_CLIP inside run_rac.main)."""
    base = TEMPORAL_BASE[temporal_dataset]
    if gt_root is None:
        # <path>/CLIP_Embedding -> <path>/gt
        gt_root = os.path.join(os.path.dirname(os.path.normpath(clip_path)), "gt")

    id2row = {}
    imgs, txts, labs = [], [], []
    row = 0
    for split in ("train", "dev_seen", "test_seen"):
        d = torch.load(
            os.path.join(clip_path, base, "{}_{}.pt".format(split, model)),
            map_location="cpu")
        ids = [i for sub in d["ids"] for i in sub]
        assert len(ids) == d["img_feats"].shape[0]
        for vid in ids:
            assert vid not in id2row, "duplicate id in cache: {}".format(vid)
            id2row[vid] = row
            row += 1
        imgs.append(d["img_feats"])
        txts.append(d["text_feats"])
        labs.append(d["labels"])
    imgs = torch.cat(imgs, dim=0)
    txts = torch.cat(txts, dim=0)
    labs = torch.cat(labs, dim=0)

    out = []
    for split in ("train", "val", "test"):
        recs = load_temporal_records(gt_root, temporal_dataset, split)
        idx = [id2row[r["id"]] for r in recs]
        sel_lab = labs[idx]
        gt_lab = torch.tensor([int(r["label"]) for r in recs],
                              dtype=sel_lab.dtype)
        assert torch.equal(sel_lab, gt_lab), \
            "label mismatch between cache and {}/{} gt".format(
                temporal_dataset, split)
        out.append([[r["id"] for r in recs], imgs[idx], txts[idx], sel_lab])
    return out[0], out[1], out[2]


# ------------------------------------------------------------------ #
#  helpers
# ------------------------------------------------------------------ #
def build_head(image_dim, text_dim, ns):
    """Instantiate the exact head architecture used during training."""
    return classifier_hateClipper(
        image_dim, text_dim,
        num_layers=ns.num_layers, proj_dim=ns.proj_dim, map_dim=ns.map_dim,
        fusion_mode=ns.fusion_mode, dropout=ns.dropout,
        batch_norm=ns.batch_norm, args=EasyDict({"dataset": ns.dataset}),
    )


def split_stats(labels, dates=None):
    labels = np.asarray(labels).astype(int)
    n = len(labels)
    pos = int(labels.sum())
    maj_cls = int(np.bincount(labels).argmax())
    maj_pred = np.full(n, maj_cls)
    st = {
        "n": n,
        "pos_n": pos,
        "pos_rate": round(pos / n, 4),
        "majority_class": maj_cls,
        "majority_acc": round(max(pos, n - pos) / n, 4),
        "majority_macro_f1": round(
            float(f1_score(labels, maj_pred, average="macro")), 4),
    }
    if dates:
        dd = [d for d in dates if d]
        if dd:
            st["date_min"] = min(dd)
            st["date_max"] = max(dd)
    return st


def eval_memory(mem_ids, mem_emb, mem_lab, q_emb, q_lab, topk):
    """kNN vote of test queries against a memory bank; both vote variants.
    'arith' (use_sim=False) matches the quoted cross-matrix floors and
    final_evaluation; 'sim' matches the per-epoch run_rac Test line."""
    res = {"mem_N": int(len(mem_ids))}
    for use_sim in (False, True):
        macro, (acc, roc, pre, rec, f1) = knn_vote(
            mem_emb, mem_lab, mem_ids, q_emb, q_lab,
            topk=topk, metric="cos", threshold=-1.0, use_sim=use_sim)
        res["sim" if use_sim else "arith"] = {
            "macro_f1": round(float(macro["macro_f1"]), 4),
            "macro_pre": round(float(macro["macro_pre"]), 4),
            "macro_recall": round(float(macro["macro_recall"]), 4),
            "acc": round(float(macro["acc"]), 4),
            "roc": round(float(macro["roc"]), 4),
            "pos_f1": round(float(f1), 4),
        }
    return res


def knn_margin(mem_emb, mem_lab, q_emb, topk):
    """Arithmetic kNN vote margin |2p-1| of each query vs the memory
    (p = fraction of positive labels among the topk cosine neighbours).
    Small margin = memory is most uncertain about this sample."""
    mem = mem_emb.copy()
    qry = q_emb.copy()
    faiss.normalize_L2(mem)
    faiss.normalize_L2(qry)
    index = faiss.IndexFlatIP(mem.shape[1])
    index.add(mem)
    _, I = index.search(qry, min(topk, mem.shape[0]))
    p = mem_lab[I].astype("float64").mean(axis=1)
    return np.abs(2.0 * p - 1.0)


def select_pool_indices(strategy, k, n_pool, dates=None, margins=None,
                        rng=None):
    if strategy == "random":
        return np.sort(rng.choice(n_pool, size=k, replace=False))
    if strategy == "latest":
        order = np.argsort(np.asarray(dates), kind="stable")[::-1]
        return order[:k]
    if strategy == "uncertain":
        order = np.argsort(margins, kind="stable")
        return order[:k]
    raise ValueError("unknown strategy: {}".format(strategy))


def augment_and_eval(mem_ids, mem_emb, mem_lab,
                     pool_ids, pool_emb, pool_lab, sel,
                     q_emb, q_lab, topk):
    ids = list(mem_ids) + [pool_ids[i] for i in sel]
    emb = np.concatenate([mem_emb, pool_emb[sel]], axis=0)
    lab = np.concatenate([mem_lab, pool_lab[sel]], axis=0)
    return eval_memory(ids, emb, lab, q_emb, q_lab, topk)


# ------------------------------------------------------------------ #
#  main
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True,
                    help="MHC_temporal / MHC_zh_temporal (temporal protocol) "
                         "or MHC / MHC_zh (random-split reference).")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Trained head checkpoint (val-selected, warmup>=5).")
    ap.add_argument("--model", type=str,
                    default="openai_clip-vit-large-patch14-336_HF")
    ap.add_argument("--path", type=str, default="./data/")
    ap.add_argument("--gt_root", type=str, default="./data/gt")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--augment", type=lambda x: str(x).lower() == "true",
                    default=True,
                    help="Run the memory-augmentation curve (temporal only).")
    ap.add_argument("--k_list", type=str, default="5,10,20,50")
    ap.add_argument("--strategies", type=str,
                    default="random,latest,uncertain")
    ap.add_argument("--n_seeds", type=int, default=5,
                    help="Seeds for the random selection strategy.")
    ap.add_argument("--out", type=str, default="",
                    help="Where to dump the JSON results.")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    # head architecture (must match the trained checkpoint)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--proj_dim", type=int, default=1024)
    ap.add_argument("--map_dim", type=int, default=1024)
    ap.add_argument("--fusion_mode", type=str, default="align")
    ap.add_argument("--dropout", type=float, nargs=3, default=[0.2, 0.4, 0.1])
    ap.add_argument("--batch_norm", type=lambda x: str(x).lower() == "true",
                    default=False)
    args = ap.parse_args()

    is_temporal = args.dataset in TEMPORAL_BASE
    clip_path = os.path.join(args.path, "CLIP_Embedding")

    if is_temporal:
        train, val, test = load_temporal_feats(
            clip_path, args.dataset, args.model, gt_root=args.gt_root)
        recs = {s: load_temporal_records(args.gt_root, args.dataset, s)
                for s in ("train", "val", "test")}
        dates = {s: [r.get("upload_date") or "" for r in recs[s]]
                 for s in recs}
    else:
        train, val, test = load_feats_from_CLIP(
            clip_path, args.dataset, args.model)
        dates = None
        if args.augment:
            print("[warn] --augment only defined for temporal splits; "
                  "disabling.")
            args.augment = False

    # ---- head + projection into the trained fused space ------------------
    image_dim = train[1].shape[1]
    text_dim = train[2].shape[1]
    model = build_head(image_dim, text_dim, args)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.to(args.device)

    tr_ids, tr_emb, tr_lab = project_split(model, train, args.device)
    va_ids, va_emb, va_lab = project_split(model, val, args.device)
    te_ids, te_emb, te_lab = project_split(model, test, args.device)

    # ---- leak-safety asserts ---------------------------------------------
    assert not (set(te_ids) & set(tr_ids)), "test leaked into train memory"
    assert not (set(te_ids) & set(va_ids)), "test leaked into val pool"

    result = {
        "dataset": args.dataset,
        "temporal": is_temporal,
        "ckpt": args.ckpt,
        "model": args.model,
        "topk": args.topk,
        "vote": "arithmetic (arith) / similarity-weighted (sim)",
        "splits": {
            "train": split_stats(tr_lab, dates["train"] if dates else None),
            "val": split_stats(va_lab, dates["val"] if dates else None),
            "test": split_stats(te_lab, dates["test"] if dates else None),
        },
    }

    # ---- (i) static memory = train; upper ref = train+val (== k=all) -----
    static_train = eval_memory(tr_ids, tr_emb, tr_lab, te_emb, te_lab,
                               args.topk)
    all_idx = np.arange(len(va_ids))
    static_train_val = augment_and_eval(
        tr_ids, tr_emb, tr_lab, va_ids, va_emb, va_lab, all_idx,
        te_emb, te_lab, args.topk)
    result["static"] = {
        "train_memory": static_train,
        "train_val_memory (k=all={})".format(len(va_ids)): static_train_val,
    }
    base_f1 = static_train["arith"]["macro_f1"]
    base_acc = static_train["arith"]["acc"]
    print("STATIC  mem=train      N={:<4d} macroF1={:.4f} acc={:.4f} "
          "(majority acc={:.4f} macroF1={:.4f})".format(
              static_train["mem_N"], base_f1, base_acc,
              result["splits"]["test"]["majority_acc"],
              result["splits"]["test"]["majority_macro_f1"]))
    print("STATIC  mem=train+val  N={:<4d} macroF1={:.4f} acc={:.4f}".format(
        static_train_val["mem_N"], static_train_val["arith"]["macro_f1"],
        static_train_val["arith"]["acc"]))

    # ---- (ii) memory augmentation from the new-period stream (val) -------
    if args.augment:
        k_list = [int(k) for k in args.k_list.split(",") if k.strip()]
        strategies = [s.strip() for s in args.strategies.split(",")
                      if s.strip()]
        margins = knn_margin(tr_emb, tr_lab, va_emb, args.topk)
        pool_dates = dates["val"]
        rows = []
        for strategy in strategies:
            for k in k_list:
                k = min(k, len(va_ids))
                if strategy == "random":
                    per_seed = []
                    for s in range(args.n_seeds):
                        rng = np.random.default_rng(1000 + s)
                        sel = select_pool_indices("random", k, len(va_ids),
                                                  rng=rng)
                        r = augment_and_eval(
                            tr_ids, tr_emb, tr_lab, va_ids, va_emb, va_lab,
                            sel, te_emb, te_lab, args.topk)
                        r["seed"] = 1000 + s
                        r["selected_ids"] = [va_ids[i] for i in sel]
                        per_seed.append(r)
                    f1s = [r["arith"]["macro_f1"] for r in per_seed]
                    accs = [r["arith"]["acc"] for r in per_seed]
                    row = {
                        "strategy": strategy, "k": k,
                        "macro_f1_mean": round(float(np.mean(f1s)), 4),
                        "macro_f1_std": round(float(np.std(f1s)), 4),
                        "acc_mean": round(float(np.mean(accs)), 4),
                        "acc_std": round(float(np.std(accs)), 4),
                        "gain_per_sample_macro_f1": round(
                            (float(np.mean(f1s)) - base_f1) / k, 5),
                        "gain_per_sample_acc": round(
                            (float(np.mean(accs)) - base_acc) / k, 5),
                        "per_seed": per_seed,
                    }
                    print("AUG strat=random    k={:<3d} macroF1={:.4f}+-{:.4f} "
                          "acc={:.4f}+-{:.4f} gain/sample(F1)={:+.5f}".format(
                              k, row["macro_f1_mean"], row["macro_f1_std"],
                              row["acc_mean"], row["acc_std"],
                              row["gain_per_sample_macro_f1"]))
                    rows.append(row)
                else:
                    sel = select_pool_indices(
                        strategy, k, len(va_ids),
                        dates=pool_dates, margins=margins)
                    r = augment_and_eval(
                        tr_ids, tr_emb, tr_lab, va_ids, va_emb, va_lab,
                        sel, te_emb, te_lab, args.topk)
                    row = {
                        "strategy": strategy, "k": k,
                        "macro_f1_mean": r["arith"]["macro_f1"],
                        "macro_f1_std": 0.0,
                        "acc_mean": r["arith"]["acc"],
                        "acc_std": 0.0,
                        "gain_per_sample_macro_f1": round(
                            (r["arith"]["macro_f1"] - base_f1) / k, 5),
                        "gain_per_sample_acc": round(
                            (r["arith"]["acc"] - base_acc) / k, 5),
                        "selected_ids": [va_ids[i] for i in sel],
                        "detail": r,
                    }
                    print("AUG strat={:<9s} k={:<3d} macroF1={:.4f}        "
                          "acc={:.4f}        gain/sample(F1)={:+.5f}".format(
                              strategy, k, row["macro_f1_mean"],
                              row["acc_mean"],
                              row["gain_per_sample_macro_f1"]))
                    rows.append(row)
        # store the val-pool margins for the record (id -> margin)
        result["val_pool_margins"] = {
            va_ids[i]: round(float(margins[i]), 4)
            for i in range(len(va_ids))}
        result["augment"] = rows

    # ---- dump -------------------------------------------------------------
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print("[out] wrote {}".format(args.out))

    summary = {
        "dataset": args.dataset,
        "static_train_macro_f1": base_f1,
        "static_train_acc": base_acc,
        "static_train_val_macro_f1": static_train_val["arith"]["macro_f1"],
        "static_train_val_acc": static_train_val["arith"]["acc"],
        "test_majority_acc": result["splits"]["test"]["majority_acc"],
        "test_majority_macro_f1":
            result["splits"]["test"]["majority_macro_f1"],
        "test_pos_rate": result["splits"]["test"]["pos_rate"],
    }
    print("RESULT_JSON {}".format(json.dumps(summary)))


if __name__ == "__main__":
    main()
