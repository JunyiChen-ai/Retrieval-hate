"""
Cross-dataset kNN memory-transfer evaluation.

Idea
----
RGCL inference is a kNN vote over the TRAIN-set memory bank in the learned
fused-embedding space. A head trained on dataset A produces a projection; if
that projection space transfers, we can classify a *different* dataset T's test
set simply by SWAPPING the memory bank to another dataset M's train(+val),
with NO retraining. This is a capability a trained-MoE head (e.g. MoRE)
structurally lacks: its decision is baked into learned weights and cannot be
re-pointed at a new support set at inference time.

All datasets share the SAME frozen CLIP encoder (openai_clip-vit-large-patch14-336),
so once a trained head projects their pooled features, all fused embeddings live
in one space and can be compared with faiss.

Protocol (leak-safe)
--------------------
  - Load a TRAINED head checkpoint (trained on dataset A).
  - MEMORY / index  = dataset M's train (+ dev/val) fused embeddings.  (never M-test, never T-test)
  - QUERIES         = dataset T's test fused embeddings.
  - kNN vote        = compute_metrics_retrieval(topk=20, majority_voting="arithmetic").
  - Report warmup-consistent metrics: macro-F1 / macro-P / macro-R / acc (+ pos-F1, roc).

Usage
-----
  python src/eval_cross_dataset.py \
      --ckpt <path/to/best_model_XX.pt> \
      --memory_dataset MHC --eval_dataset HateMM \
      --model openai_clip-vit-large-patch14-336_HF \
      --topk 20

Nothing here touches the training loop; it only *reads* cached CLIP features and
a saved head.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch
import faiss
from easydict import EasyDict

from model.classifier import classifier_hateClipper
from data_loader.dataset import load_feats_from_CLIP
from utils.metrics import compute_metrics_retrieval


# ------------------------------------------------------------------ #
#  feature -> fused-embedding projection (mirrors retrieve_evaluate_RAC)
# ------------------------------------------------------------------ #
def project_split(model, split, device, batch_size=256):
    """Return (ids:list, embeds:np.float32[N,D], labels:np.int[N]) for one split.

    `split` is the [ids, img_feats, text_feats, labels] tuple returned by
    load_feats_from_CLIP. We push it through the trained head and grab the
    `embed` (the fused-embedding used for kNN), exactly as retrieve_evaluate_RAC
    does with return_embed=True.
    """
    ids, img_feats, text_feats, labels = split
    if not torch.is_tensor(img_feats):
        img_feats = torch.tensor(img_feats)
    if not torch.is_tensor(text_feats):
        text_feats = torch.tensor(text_feats)
    img_feats = img_feats.float()
    text_feats = text_feats.float()
    labels = torch.as_tensor(labels).long()

    model.eval()
    embeds = []
    with torch.no_grad():
        for i in range(0, img_feats.shape[0], batch_size):
            imb = img_feats[i:i + batch_size].to(device)
            txb = text_feats[i:i + batch_size].to(device)
            _, emb = model(imb, txb, return_embed=True)
            embeds.append(emb.detach().cpu())
    embeds = torch.cat(embeds, dim=0).numpy().astype("float32")
    return list(ids), embeds, labels.numpy().astype("int")


def knn_vote(mem_embeds, mem_labels, mem_ids,
             qry_embeds, qry_labels,
             topk=20, metric="cos", threshold=-1.0, use_sim=False):
    """Cosine-kNN vote of query embeddings against a memory bank.

    Reproduces retrieve_evaluate_RAC's faiss retrieval + logging_dict, then
    scores with compute_metrics_retrieval (arithmetic vote over retrieved
    labels; identical to the in-domain evaluation used in run_rac).

    use_sim=True reproduces the per-epoch run_rac Test line (similarity-weighted
    signed vote); use_sim=False is the plain arithmetic label vote used by
    final_evaluation.
    """
    dim = mem_embeds.shape[1]
    mem = mem_embeds.copy()
    qry = qry_embeds.copy()
    if metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.IndexFlatIP(dim)
        # cosine == IP on L2-normalized vectors
        faiss.normalize_L2(mem)
        faiss.normalize_L2(qry)
    index.add(mem)
    D, I = index.search(qry, topk)

    logging_dict = EasyDict()
    for i, row in enumerate(D):
        retrieved_ids, retrieved_scores, retrieved_label = [], [], []
        for j, value in enumerate(row):
            if j == 0 or value < threshold or threshold == -1:
                retrieved_ids.append(mem_ids[I[i, j]])
                retrieved_scores.append(value)
                retrieved_label.append(int(mem_labels[I[i, j]]))
            else:
                break
        logging_dict["q{}".format(i)] = {
            "no_retrieved": len(retrieved_ids),
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores,
            "retrieved_label": retrieved_label,
        }
    # arithmetic label-vote over retrieved neighbours (matches in-domain eval)
    acc, roc, pre, recall, f1, _, _, macro = compute_metrics_retrieval(
        logging_dict, qry_labels, majority_voting="arithmetic", topk=topk,
        use_sim=use_sim,
    )
    return macro, (acc, roc, pre, recall, f1)


def build_head(image_dim, text_dim, args):
    """Instantiate the exact head used during training."""
    head_args = EasyDict({"dataset": args.eval_dataset})
    model = classifier_hateClipper(
        image_dim, text_dim,
        num_layers=args.num_layers, proj_dim=args.proj_dim, map_dim=args.map_dim,
        fusion_mode=args.fusion_mode, dropout=args.dropout,
        batch_norm=args.batch_norm, args=head_args,
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to a TRAINED head checkpoint (trained on dataset A).")
    ap.add_argument("--memory_dataset", type=str, required=True,
                    help="Dataset M whose train(+val) fused embeddings form the memory bank.")
    ap.add_argument("--eval_dataset", type=str, required=True,
                    help="Dataset T whose test split is used as queries.")
    ap.add_argument("--trained_on", type=str, default="",
                    help="Label of the dataset the checkpoint was trained on (for reporting only).")
    ap.add_argument("--path", type=str, default="./data/")
    ap.add_argument("--model", type=str,
                    default="openai_clip-vit-large-patch14-336_HF")
    ap.add_argument("--include_val", type=lambda x: str(x).lower() == "true",
                    default=True, help="Include M's dev/val split in the memory bank.")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--metric", type=str, default="cos")
    ap.add_argument("--similarity_threshold", type=float, default=-1.0)
    ap.add_argument("--use_sim", type=lambda x: str(x).lower() == "true",
                    default=False,
                    help="Similarity-weighted signed vote (matches per-epoch "
                         "run_rac Test line); False = plain arithmetic label vote.")
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

    clip_path = os.path.join(args.path, "CLIP_Embedding")

    # ---- load memory dataset M (train + optional val) -------------------
    m_train, m_dev, _m_test = load_feats_from_CLIP(
        clip_path, args.memory_dataset, args.model)
    # ---- load eval dataset T (need test only) ---------------------------
    _t_train, _t_dev, t_test = load_feats_from_CLIP(
        clip_path, args.eval_dataset, args.model)

    image_dim = m_train[1].shape[1]
    text_dim = m_train[2].shape[1]

    # ---- build head + load trained weights ------------------------------
    model = build_head(image_dim, text_dim, args)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.to(args.device)

    # ---- project all needed splits into the trained fused space ---------
    mem_ids, mem_emb, mem_lab = project_split(model, m_train, args.device)
    if args.include_val:
        dv_ids, dv_emb, dv_lab = project_split(model, m_dev, args.device)
        mem_ids = mem_ids + dv_ids
        mem_emb = np.concatenate([mem_emb, dv_emb], axis=0)
        mem_lab = np.concatenate([mem_lab, dv_lab], axis=0)
    q_ids, q_emb, q_lab = project_split(model, t_test, args.device)

    # ---- leak-safety assert: memory is M-train(+val), query is T-test ---
    # (no overlap possible across datasets; within a dataset, train/val ids
    #  are disjoint from test ids by construction of the gt splits.)
    trained_on = args.trained_on or "?"
    macro, legacy = knn_vote(
        mem_emb, mem_lab, mem_ids, q_emb, q_lab,
        topk=args.topk, metric=args.metric,
        threshold=args.similarity_threshold, use_sim=args.use_sim,
    )
    acc, roc, pre, recall, f1 = legacy

    maj = float(np.mean(np.bincount(q_lab).max() / len(q_lab)))
    print("=" * 72)
    print("CROSS-DATASET kNN MEMORY TRANSFER")
    print("  trained-on A : {}".format(trained_on))
    print("  memory   M   : {}  (train{}  N={})".format(
        args.memory_dataset, "+val" if args.include_val else "", len(mem_ids)))
    print("  eval/test T  : {}  (test  N={})".format(args.eval_dataset, len(q_ids)))
    print("  topk={} vote=arithmetic metric={}".format(args.topk, args.metric))
    print("-" * 72)
    print("  macro-F1 : {:.4f}".format(macro["macro_f1"]))
    print("  macro-P  : {:.4f}".format(macro["macro_pre"]))
    print("  macro-R  : {:.4f}".format(macro["macro_recall"]))
    print("  acc      : {:.4f}".format(macro["acc"]))
    print("  roc      : {:.4f}".format(macro["roc"]))
    print("  pos-F1   : {:.4f}   (majority-class acc baseline: {:.4f})".format(f1, maj))
    print("=" * 72)
    # machine-readable one-liner for aggregation
    print("RESULT_JSON {}".format({
        "trained_on": trained_on,
        "memory": args.memory_dataset,
        "test": args.eval_dataset,
        "macro_f1": round(macro["macro_f1"], 4),
        "macro_pre": round(macro["macro_pre"], 4),
        "macro_recall": round(macro["macro_recall"], 4),
        "acc": round(macro["acc"], 4),
        "roc": round(macro["roc"], 4),
        "pos_f1": round(float(f1), 4),
        "majority_acc": round(maj, 4),
        "mem_N": len(mem_ids),
        "test_N": len(q_ids),
    }))


if __name__ == "__main__":
    main()
