#!/usr/bin/env python
"""Role 3 step 1 (CPU): confidence gate over the archive-kNN vote margin.

Selective reasoning: the kNN memory decides WHEN to spend MLLM reasoning
budget. For every val (dev_seen) and test sample of the winning archive-kNN
configs (alpha=0.25, seed0; EN job 12210 / ZH job 12207) we compute the
similarity-signed arithmetic vote (bit-identical to the training-log protocol,
reusing eval_cross_dataset.project_split / build_head and
compute_metrics_retrieval read-only) and define

    margin = |vote|        (pred = sigmoid(vote) >= 0.5  <=>  vote >= 0)

Thresholds are selected ON VAL ONLY, at deferral rates ~{10%, 20%, 30%}
(t = midpoint between the k-th and (k+1)-th smallest val margin,
k = round(rate * N_val)); test never participates in threshold tuning.
Deferral sets are nested (defer <=> margin < t).

Memory variants:
  base  : all train keys (reproduction gate: test acc/macro-F1 must equal the
          training-log numbers bit-for-bit, same gate as DEMO_memory_editing).
  clean : EN only -- the two W2-forensics noisy train entries
          (XScP1AiMkNM, QvPp8Q7QhWE) removed from the memory
          (legitimated by train-side forensics, see DEMO_memory_editing.md).

Output: scripts/role3/out/gate_<ds>_<variant>.json with per-sample records
(vote, margin, kNN pred, defer flags per rate, top-5 neighbour evidence cards
from the train archive JSONL, own archive entry, title+transcript) for BOTH
val and test -- the GPU arbitration step reads the deferred ones.

No src/ file is modified; all imports are read-only.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import faiss
from sklearn.metrics import f1_score

ROOT = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(ROOT, "src"))

from eval_cross_dataset import project_split, build_head  # noqa: E402
from data_loader.dataset import (  # noqa: E402
    load_feats_from_CLIP, load_archive_feats_split, resolve_archive_path)
from utils.metrics import compute_metrics_retrieval, sigmoid  # noqa: E402
from easydict import EasyDict  # noqa: E402

TOPK = 20
ALPHA = 0.25
N_EVIDENCE = 5
RATES = [0.10, 0.20, 0.30]
ARCHIVE_JSONL_TAG = "Qwen2.5-VL-7B-Instruct_archive"
NOISY_IDS = ["XScP1AiMkNM", "QvPp8Q7QhWE"]  # W2 train-side forensics (MHC/EN)

CKPT = {
    "MHC": dict(
        model="Qwen2.5-VL-7B-Instruct_HF",
        job=12210, sel_epoch=24,
        logged=dict(acc=0.8075, macro_f1=0.7626),
        ckpt=os.path.join(
            ROOT, "logging/Retrieval/MHC/RAC_video_archive",
            "RAC_lr0.0001_Bz64_Ep30_cosSim_triplet_drop[0.2, 0.4, 0.1]_topK20_"
            "_PseudoGold_positive_1_hard_negative_1_seed0_hybrid_loss_"
            "Qwen2.5-VL-7B-Instruct_HF_arc-knn-a0.25/ckpt/best_model_24_0.7875.pt"),
    ),
    "MHC_zh": dict(
        model="Qwen2.5-VL-7B-Instruct-LoRA_HF",
        job=12207, sel_epoch=18,
        logged=dict(acc=0.8523, macro_f1=0.8270),
        ckpt=os.path.join(
            ROOT, "logging/Retrieval/MHC_zh/RAC_video_archive",
            "RAC_lr0.0001_Bz64_Ep30_cosSim_triplet_drop[0.2, 0.4, 0.1]_topK20_"
            "_PseudoGold_positive_1_hard_negative_1_seed0_hybrid_loss_"
            "Qwen2.5-VL-7B-Instruct-LoRA_HF_arc-knn-a0.25/ckpt/"
            "best_model_18_0.8717948717948718.pt"),
    ),
}

GT_SPLIT_FILE = {"train": "train", "val": "val", "test": "test"}


# --------------------------------------------------------------------------- #
def load_archive_records(ds):
    """id -> archive JSONL record (all splits; last occurrence wins)."""
    recs = {}
    for sp in ["train", "dev_seen", "test_seen"]:
        p = os.path.join(ROOT, "data/Archive", ds,
                         "{}_{}.jsonl".format(sp, ARCHIVE_JSONL_TAG))
        for line in open(p):
            r = json.loads(line)
            recs[r["id"]] = r
    return recs


def load_gt_texts(ds):
    """id -> title+transcript text from the gt jsonl (all splits)."""
    texts = {}
    for sp in ["train", "val", "test"]:
        p = os.path.join(ROOT, "data/gt", ds, "{}.jsonl".format(sp))
        for line in open(p):
            o = json.loads(line)
            texts[str(o["id"])] = "" if o.get("text") is None else str(o["text"])
    return texts


def archive_card(rec, max_summary=400, max_cue=200):
    """Compact evidence-card fields from one archive JSONL record."""
    if rec is None or not rec.get("archive"):
        return None
    a = rec["archive"]
    cues = a.get("modality_cues") or {}
    trim = lambda s, n: (s or "")[:n]  # noqa: E731
    return dict(
        target_groups=a.get("target_groups") or [],
        mechanism=a.get("mechanism") or [],
        explicitness=a.get("explicitness") or "unknown",
        summary=trim(a.get("neutral_summary"), max_summary),
        visual=trim(cues.get("visual"), max_cue),
        speech=trim(cues.get("speech"), max_cue),
        on_screen_text=trim(cues.get("on_screen_text"), max_cue),
    )


def augment(fused, arc, alpha=ALPHA):
    fused_n = torch.nn.functional.normalize(fused, p=2, dim=1)
    arc_n = torch.nn.functional.normalize(arc.float(), p=2, dim=1)
    return torch.cat((fused_n, alpha * arc_n), dim=1).numpy().astype("float32")


def knn_query(mem_keys, mem_labels, mem_ids, qry_keys, qry_labels):
    """Winning-config vote (faiss cosine topk=20, arithmetic weights,
    similarity-signed use_sim=True). Returns (macro, votes, preds, D, I)."""
    mem = mem_keys.copy()
    qry = qry_keys.copy()
    faiss.normalize_L2(mem)
    faiss.normalize_L2(qry)
    index = faiss.IndexFlatIP(mem.shape[1])
    index.add(mem)
    k = min(TOPK, mem.shape[0])
    D, I = index.search(qry, k)
    logging_dict = EasyDict()
    for i, row in enumerate(D):
        logging_dict["q%d" % i] = {
            "no_retrieved": len(row),
            "retrieved_ids": [mem_ids[j] for j in I[i]],
            "retrieved_scores": [np.float32(v) for v in row],
            "retrieved_label": [int(mem_labels[j]) for j in I[i]],
        }
    acc, roc, pre, recall, f1, votes, _, macro = compute_metrics_retrieval(
        logging_dict, qry_labels, majority_voting="arithmetic", topk=TOPK,
        use_sim=True)
    votes = np.array(votes, dtype=np.float64)
    preds = (sigmoid(votes) >= 0.5).astype(int)
    return macro, votes, preds, D, I


def pick_thresholds(val_margins, rates=RATES):
    """Midpoint threshold between k-th / (k+1)-th smallest val margin,
    k = round(rate*N). Defer <=> margin < t (nested across rates)."""
    m = np.sort(np.asarray(val_margins))
    n = len(m)
    out = {}
    for r in rates:
        k = int(round(r * n))
        k = max(1, min(k, n - 1))
        out["%.2f" % r] = float((m[k - 1] + m[k]) / 2.0)
    return out


def split_records(ids, labels, votes, preds, D, I, mem_ids, mem_labels,
                  thresholds, recs, texts, split_name):
    rows = []
    for i, vid in enumerate(ids):
        margin = float(abs(votes[i]))
        neighbors = []
        for r in range(min(N_EVIDENCE, I.shape[1])):
            j = int(I[i, r])
            nid = mem_ids[j]
            neighbors.append(dict(
                id=nid, sim=float(D[i, r]), label=int(mem_labels[j]),
                card=archive_card(recs.get(nid)),
            ))
        rows.append(dict(
            id=vid, split=split_name, label=int(labels[i]),
            vote=float(votes[i]), margin=margin, pred_knn=int(preds[i]),
            defer={k: bool(margin < t) for k, t in thresholds.items()},
            neighbors=neighbors,
            own_card=archive_card(recs.get(vid)),
            text=texts.get(vid, ""),
        ))
    return rows


def macro_of(preds, labels):
    return dict(acc=float(np.mean(preds == labels)),
                macro_f1=float(f1_score(labels, preds, average="macro",
                                        zero_division=0)))


# --------------------------------------------------------------------------- #
def run(ds, variant, out_dir):
    cfg = CKPT[ds]
    device = "cpu"
    clip_path = os.path.join(ROOT, "data", "CLIP_Embedding")
    train, dev, test = load_feats_from_CLIP(clip_path, ds, cfg["model"])
    model = build_head(train[1].shape[1], train[2].shape[1], EasyDict(
        eval_dataset=ds, num_layers=3, proj_dim=1024, map_dim=1024,
        fusion_mode="align", dropout=[0.2, 0.4, 0.1], batch_norm=False))
    model.load_state_dict(torch.load(cfg["ckpt"], map_location="cpu"))

    tr_ids, tr_emb, tr_lab = project_split(model, train, device)
    dv_ids, dv_emb, dv_lab = project_split(model, dev, device)
    te_ids, te_emb, te_lab = project_split(model, test, device)

    arc = {}
    for name, ids in [("train", tr_ids), ("dev_seen", dv_ids),
                      ("test_seen", te_ids)]:
        arc[name] = load_archive_feats_split(
            resolve_archive_path("auto", os.path.join(ROOT, "data"), ds, name),
            ids)
    mem_keys = augment(torch.tensor(tr_emb), arc["train"])
    val_keys = augment(torch.tensor(dv_emb), arc["dev_seen"])
    tst_keys = augment(torch.tensor(te_emb), arc["test_seen"])

    removed = []
    keep = np.ones(len(tr_ids), dtype=bool)
    if variant == "clean":
        assert ds == "MHC", "clean variant is only legitimated for MHC (EN)"
        keep = np.array([i not in NOISY_IDS for i in tr_ids])
        removed = [i for i in tr_ids if i in NOISY_IDS]
        assert len(removed) == 2, removed
    kidx = np.where(keep)[0]
    m_keys = mem_keys[kidx]
    m_lab = tr_lab[kidx]
    m_ids = [tr_ids[j] for j in kidx]

    # ---- baseline reproduction gate (base variant, test split) -------------
    te_macro, te_votes, te_preds, te_D, te_I = knn_query(
        m_keys, m_lab, m_ids, tst_keys, te_lab)
    repro = dict(test_acc=round(float(te_macro["acc"]), 4),
                 test_macro_f1=round(float(te_macro["macro_f1"]), 4),
                 logged=cfg["logged"])
    if variant == "base":
        assert repro["test_acc"] == cfg["logged"]["acc"], repro
        assert repro["test_macro_f1"] == cfg["logged"]["macro_f1"], repro
        repro["gate"] = "PASS (bit-identical to training log)"
    else:
        repro["gate"] = "n/a (edited memory; base gate holds on gate_%s_base)" % ds

    dv_macro, dv_votes, dv_preds, dv_D, dv_I = knn_query(
        m_keys, m_lab, m_ids, val_keys, dv_lab)

    # ---- thresholds from VAL margins only ----------------------------------
    thresholds = pick_thresholds(np.abs(dv_votes))

    recs = load_archive_records(ds)
    texts = load_gt_texts(ds)
    samples = (
        split_records(dv_ids, dv_lab, dv_votes, dv_preds, dv_D, dv_I,
                      m_ids, m_lab, thresholds, recs, texts, "val") +
        split_records(te_ids, te_lab, te_votes, te_preds, te_D, te_I,
                      m_ids, m_lab, thresholds, recs, texts, "test"))

    defer_counts = {}
    for sp, n in [("val", len(dv_ids)), ("test", len(te_ids))]:
        defer_counts[sp] = {
            k: dict(n=sum(1 for s in samples
                          if s["split"] == sp and s["defer"][k]),
                    total=n)
            for k in thresholds}

    out = dict(
        dataset=ds, variant=variant,
        config=dict(model=cfg["model"], job=cfg["job"],
                    sel_epoch=cfg["sel_epoch"], ckpt=cfg["ckpt"],
                    topk=TOPK, alpha=ALPHA, n_evidence=N_EVIDENCE,
                    removed_ids=removed, mem_n=len(m_ids)),
        repro=repro,
        metrics_full=dict(
            val=macro_of(dv_preds, dv_lab), test=macro_of(te_preds, te_lab)),
        thresholds=thresholds,
        defer_counts=defer_counts,
        samples=samples,
    )
    path = os.path.join(out_dir, "gate_{}_{}.json".format(ds, variant))
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[{}/{}] repro: {}  | val {}  test {}".format(
        ds, variant, repro,
        out["metrics_full"]["val"], out["metrics_full"]["test"]))
    print("  thresholds:", thresholds)
    print("  defer_counts:", json.dumps(defer_counts))
    print("  wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "scripts/role3/out"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)
    for ds, variant in [("MHC", "base"), ("MHC", "clean"), ("MHC_zh", "base")]:
        run(ds, variant, args.out_dir)


if __name__ == "__main__":
    main()
