#!/usr/bin/env python
"""Train one strict train-only SSR OOF comparator and emit exact rankings."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from easydict import EasyDict

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from data_loader.rac_dataloader import CLIP2Dataloader  # noqa: E402
from model.classifier import classifier_hateClipper  # noqa: E402
from model.loss import compute_loss  # noqa: E402
from utils.metrics import compute_metrics_retrieval  # noqa: E402
from ssr_common import (  # noqa: E402
    atomic_write_json, atomic_write_jsonl, canonical_json, exact_vote,
    load_config, resolve, sha256_file, sha256_obj,
)


def flatten_ids(cache):
    ids = cache["ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        return [str(x) for batch in ids for x in batch]
    return [str(x) for x in ids]


def load_train_cache(cfg, dataset):
    path = resolve(cfg, "clip") / dataset / "train_{}.pt".format(
        cfg["comparator"]["model"])
    cache = torch.load(path, map_location="cpu")
    ids = flatten_ids(cache)
    img = torch.as_tensor(cache["img_feats"]).float()
    txt = torch.as_tensor(cache["text_feats"]).float()
    labels = torch.as_tensor(cache["labels"]).long().reshape(-1)
    if not (len(ids) == img.shape[0] == txt.shape[0] == labels.shape[0]):
        raise RuntimeError("train cache row mismatch")
    return path, ids, img, txt, labels


def take_dataset(ids, img, txt, labels, selected_ids):
    row = {vid: i for i, vid in enumerate(ids)}
    if len(row) != len(ids):
        raise RuntimeError("duplicate IDs in train cache")
    missing = [x for x in selected_ids if x not in row]
    if missing:
        raise RuntimeError("selected IDs absent from cache: {}".format(missing[:10]))
    idx = torch.as_tensor([row[x] for x in selected_ids], dtype=torch.long)
    return [list(selected_ids), img.index_select(0, idx), txt.index_select(0, idx),
            labels.index_select(0, idx)], idx


def make_segment_cache(cfg, dataset, full_ids, full_labels, memory, memory_full_idx):
    c = cfg["comparator"]
    path = resolve(cfg, "clip") / dataset / "train_subclipK{}_{}.pt".format(
        c["num_subclips"], c["model"])
    sc = torch.load(path, map_location="cpu")
    old_parent = torch.as_tensor(sc["subclip_parent"]).long().reshape(-1)
    sc_img = torch.as_tensor(sc["subclip_img_feats"]).float()
    sc_labels = torch.as_tensor(sc["labels"]).long().reshape(-1)
    if not (len(old_parent) == sc_img.shape[0] == len(sc_labels)):
        raise RuntimeError("subclip cache row mismatch")
    if int(old_parent.min()) < 0 or int(old_parent.max()) >= len(full_ids):
        raise RuntimeError("subclip parent outside full train cache")
    inherited_bad = sc_labels != full_labels.index_select(0, old_parent)
    if bool(inherited_bad.any()):
        raise RuntimeError("subclip inherited parent video labels mismatch")

    old_to_new = {int(old): new for new, old in enumerate(memory_full_idx.tolist())}
    keep = torch.as_tensor([int(x) in old_to_new for x in old_parent.tolist()],
                           dtype=torch.bool)
    kept_old_parent = old_parent[keep]
    new_parent = torch.as_tensor([old_to_new[int(x)] for x in kept_old_parent],
                                 dtype=torch.long)
    kept_labels = sc_labels[keep]
    memory_labels = torch.as_tensor(memory[3]).long()
    if not torch.equal(kept_labels, memory_labels.index_select(0, new_parent)):
        raise RuntimeError("fold-local subclip labels are not inherited video labels")
    return path, {
        "subclip_img_feats": sc_img[keep],
        "subclip_parent": new_parent,
        "labels": kept_labels,
        "parent_id_to_row": {vid: i for i, vid in enumerate(memory[0])},
        "video_text_feats": torch.as_tensor(memory[2]).float(),
    }, {
        "n_subclips": int(keep.sum()),
        "n_parents": len(memory[0]),
        "label_source": "inherited_parent_video_label_not_segment_gold",
        "sha256": sha256_file(path),
    }


def train_args(cfg, dataset):
    c = cfg["comparator"]
    d = cfg["datasets"][dataset]
    return EasyDict({
        "dataset": dataset, "device": "cuda", "batch_size": c["batch_size"],
        "lr": c["lr"], "proj_dim": c["proj_dim"], "metric": c["metric"],
        "loss": c["loss"], "triplet_margin": c["triplet_margin"],
        "norm_feats_loss": False, "l2_sqrt": False,
        "hybrid_loss": c["hybrid_loss"], "ce_weight": c["ce_weight"],
        "pos_weight_value": None, "hard_negatives_loss": c["hard_negatives_loss"],
        "no_hard_negatives": c["no_hard_negatives"],
        "no_hard_positives": 0,
        "no_pseudo_gold_positives": c["no_pseudo_gold_positives"],
        "hard_negatives_multiple": 12, "sparse_dictionary": None,
        "sparse_topk": None, "Faiss_GPU": False, "grad_clip": c["grad_clip"],
        "lambda_seg": d["lambda_seg"], "seg_mode": d["seg_mode"],
        "cf_negs": False, "lambda_aux": 0.0,
    })


def train_model(model, train_dl, train_set, segment_cache, args, epochs, log_path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        for epoch in range(epochs):
            train_feats = train_labels = None
            losses = []
            outer_batch_ids = []
            for step, batch in enumerate(train_dl):
                outer_batch_ids.extend(str(x) for x in batch["ids"])
                out = compute_loss(
                    batch, train_dl, model, args, train_set=train_set,
                    sparse_retrieval_dictionary=None, train_feats=train_feats,
                    train_labels=train_labels, segment_cache=segment_cache,
                    aux_pack=None, cf_pack=None)
                total_loss, _, _, _, _, train_feats, train_labels = out
                if torch.is_tensor(train_feats):
                    train_feats = train_feats.detach()
                if torch.is_tensor(train_labels):
                    train_labels = train_labels.detach()
                if not bool(torch.isfinite(total_loss)):
                    raise RuntimeError("non-finite loss epoch={} step={}".format(epoch, step))
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                losses.append(float(total_loss.detach().cpu()))
            rec = {"epoch_index": epoch, "mean_loss": float(np.mean(losses)),
                   "steps": len(losses),
                   "outer_batch_order_sha256": sha256_obj(outer_batch_ids),
                   "outer_batch_rows": len(outer_batch_ids),
                   "wall_s": round(time.time() - start, 3)}
            history.append(rec)
            log.write(canonical_json(rec) + "\n")
            log.flush()
            print(canonical_json(rec), flush=True)
    return history


@torch.no_grad()
def project(model, dataset, device="cuda", batch_size=256):
    ids, img, txt, labels = dataset
    model.eval()
    out = []
    for i in range(0, len(ids), batch_size):
        _, z = model(img[i:i + batch_size].to(device),
                     txt[i:i + batch_size].to(device), return_embed=True)
        out.append(z.detach().cpu())
    return torch.cat(out).numpy().astype("float32"), np.asarray(labels, dtype="int64")


def make_rankings(memory_ids, memory_z, memory_labels,
                  query_ids, query_z, query_labels, fold, topk):
    mem = memory_z.copy()
    qry = query_z.copy()
    faiss.normalize_L2(mem)
    faiss.normalize_L2(qry)
    index = faiss.IndexFlatIP(mem.shape[1])
    index.add(mem)
    dists, inds = index.search(qry, len(memory_ids))
    rows, preds = [], []
    repo_logging = EasyDict()
    for qi, qid in enumerate(query_ids):
        # FAISS supplies exact IP values; canonical ID resolves all equal-IP ties.
        raw = [(int(inds[qi, r]), float(dists[qi, r]))
               for r in range(len(memory_ids))]
        raw.sort(key=lambda x: (-x[1], memory_ids[x[0]]))
        ranking = [{"rank": r + 1, "id": memory_ids[idx],
                    "label": int(memory_labels[idx]), "cosine": sim}
                   for r, (idx, sim) in enumerate(raw)]
        vote, pred, denom = exact_vote(ranking, topk=topk)
        rows.append({"query_id": qid, "query_label": int(query_labels[qi]),
                     "outer_fold": int(fold), "memory_n": len(memory_ids),
                     "ranking": ranking})
        preds.append({"query_id": qid, "query_label": int(query_labels[qi]),
                      "outer_fold": int(fold), "prediction": pred,
                      "vote": vote, "vote_abs_margin": abs(vote),
                      "vote_abs_denom": denom,
                      "normalized_abs_margin": abs(vote) / max(denom, 1e-12),
                      "baseline_error": int(pred != int(query_labels[qi]))})
        repo_logging[qid] = {
            "no_retrieved": topk,
            "retrieved_ids": [x["id"] for x in ranking[:topk]],
            "retrieved_scores": [np.float32(x["cosine"]) for x in ranking[:topk]],
            "retrieved_label": [x["label"] for x in ranking[:topk]],
        }
    repo = compute_metrics_retrieval(
        repo_logging, np.asarray(query_labels), majority_voting="arithmetic",
        topk=topk, use_sim=True)
    repo_scores = repo[5]
    repo_preds = [int(float(x) >= 0.0) for x in repo_scores]
    ours = [x["prediction"] for x in preds]
    if repo_preds != ours:
        raise RuntimeError("exact vote disagrees with repository compute_metrics_retrieval")
    return rows, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True, choices=["MHC", "MHC_zh"])
    ap.add_argument("--fold", required=True, type=int, choices=range(5))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--force", action="store_true")
    args_cli = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("SSR computation must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("expected conda HateVideo")
    cfg = load_config(args_cli.config)
    fold_path = resolve(cfg, "artifacts") / "folds" / "{}.json".format(args_cli.dataset)
    folds = json.load(open(fold_path, encoding="utf-8"))
    if folds["config_sha256"] != cfg["computed_config_sha256"]:
        raise RuntimeError("fold/config hash mismatch")
    records = folds["records"]
    query_ids = sorted(x["id"] for x in records if x["fold"] == args_cli.fold)
    memory_ids = sorted(x["id"] for x in records if x["fold"] != args_cli.fold)
    if set(query_ids) & set(memory_ids) or set(query_ids) | set(memory_ids) != {
            x["id"] for x in records}:
        raise RuntimeError("fold-local memory/query partition invalid")

    out_dir = resolve(cfg, "artifacts") / "oof" / args_cli.dataset / "fold{}".format(args_cli.fold)
    if out_dir.exists() and any(out_dir.iterdir()) and not args_cli.force:
        raise RuntimeError("refusing to overwrite {}".format(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["comparator"]["seed"])
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cache_path, full_ids, img, txt, labels = load_train_cache(cfg, args_cli.dataset)
    frozen_train = folds["split_assertions"]["clip_cache"]["train"]
    if sha256_file(cache_path) != frozen_train["sha256"]:
        raise RuntimeError("train cache changed after fold freeze")
    fold_by_id = {x["id"]: x for x in records}
    if set(full_ids) != set(fold_by_id) or any(
            int(labels[i]) != int(fold_by_id[vid]["label"])
            for i, vid in enumerate(full_ids)):
        raise RuntimeError("train cache IDs/labels disagree with frozen folds")
    memory, memory_idx = take_dataset(full_ids, img, txt, labels, memory_ids)
    query, _ = take_dataset(full_ids, img, txt, labels, query_ids)
    (train_dl, _query_dl), (train_set, _query_set) = CLIP2Dataloader(
        memory, query, batch_size=int(cfg["comparator"]["batch_size"]),
        return_dataset=True, normalize=False)
    seg_path, segment_cache, seg_manifest = make_segment_cache(
        cfg, args_cli.dataset, full_ids, labels, memory, memory_idx)
    frozen_subclip = folds["split_assertions"]["subclip_cache"]
    if sha256_file(seg_path) != frozen_subclip["sha256"]:
        raise RuntimeError("subclip cache changed after fold freeze")
    targs = train_args(cfg, args_cli.dataset)
    model = classifier_hateClipper(
        int(img.shape[1]), int(txt.shape[1]),
        cfg["comparator"]["num_layers"], cfg["comparator"]["proj_dim"],
        cfg["comparator"]["map_dim"], cfg["comparator"]["fusion_mode"],
        dropout=cfg["comparator"]["dropout"],
        batch_norm=cfg["comparator"]["batch_norm"], args=targs).cuda()
    epoch_index = int(cfg["datasets"][args_cli.dataset]["epoch_index"])
    history = train_model(model, train_dl, train_set, segment_cache, targs,
                          epochs=epoch_index + 1, log_path=out_dir / "train.jsonl")
    ckpt_path = out_dir / "checkpoint_epoch{}.pt".format(epoch_index)
    torch.save(model.state_dict(), ckpt_path)

    memory_z, memory_y = project(model, memory)
    query_z, query_y = project(model, query)
    np.savez_compressed(out_dir / "embeddings.npz",
                        memory_ids=np.asarray(memory_ids), memory_z=memory_z,
                        memory_labels=memory_y, query_ids=np.asarray(query_ids),
                        query_z=query_z, query_labels=query_y)
    rankings, predictions = make_rankings(
        memory_ids, memory_z, memory_y, query_ids, query_z, query_y,
        args_cli.fold, int(cfg["comparator"]["topk"]))
    atomic_write_jsonl(out_dir / "ranking.jsonl", rankings)
    atomic_write_json(out_dir / "predictions.json", predictions)
    manifest = {
        "run_id": args_cli.run_id, "status": "COMPLETED",
        "dataset": args_cli.dataset, "outer_fold": args_cli.fold,
        "config_sha256": cfg["computed_config_sha256"],
        "fold_artifact_sha256": sha256_file(fold_path),
        "query_ids_sha256": sha256_obj(query_ids),
        "memory_ids_sha256": sha256_obj(memory_ids),
        "query_n": len(query_ids), "memory_n": len(memory_ids),
        "query_memory_overlap": [], "dev_or_test_endpoint_count": 0,
        "fixed_epoch_index": epoch_index, "epochs_trained": epoch_index + 1,
        "only_gold_supervision": "video_level_binary_label",
        "segment_gold_exists": False,
        "subclip_contract": seg_manifest,
        "source_train_cache": {"path": str(cache_path.relative_to(ROOT)),
                               "sha256": sha256_file(cache_path)},
        "source_subclip_cache": str(seg_path.relative_to(ROOT)),
        "outputs": {name: sha256_file(out_dir / name) for name in (
            "train.jsonl", ckpt_path.name, "embeddings.npz", "ranking.jsonl",
            "predictions.json")},
        "training_last": history[-1],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_write_json(out_dir / "manifest.json", manifest)
    print(canonical_json({"status": "COMPLETED", "run_id": args_cli.run_id,
                          "query_n": len(query_ids), "memory_n": len(memory_ids)}))


if __name__ == "__main__":
    main()
