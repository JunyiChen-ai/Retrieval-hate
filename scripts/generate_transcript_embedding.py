#!/usr/bin/env python
"""W4 ablation: long-context TRANSCRIPT embeddings as kNN-key augmentation.

Decisive ablation for the MLLM structured-archive (E0b) contribution:
is the archive gain structured distillation, or merely a fix for the CLIP
text tower's 77-token transcript truncation (EN 56% / ZH 79% truncated)?

This script encodes the FULL title+transcript text (same source as
data/gt/<dataset>/<split>.jsonl, i.e. what prep_mhc.py wrote) with a
long-context multilingual sentence encoder and stores per-split .pt caches
in EXACTLY the archive-cache format {ids, text_feats [N, D], labels}, so
the untouched --archive_feats / archive_mode=knn pipeline consumes them
without any src/ change.

Encoder: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  (XLM-R base, mean pooling, output dim 768 == archive dim; default
  max_seq_length is 128 -> we raise it to 512, the XLM-R position limit).
  Chinese is encoded directly (multilingual model, no translation, to avoid
  introducing a new variable).

Alignment: ids + labels are taken verbatim from the EXISTING archive .pt of
the same split (the same ids the main Qwen feature caches use), texts are
looked up from the union of gt jsonl files by id. STRICT: any missing id or
label mismatch raises.

Modes:
  encode  (default)  build <split>_transcript_<tag>.pt for each dataset
  concat             build <split>_arctrans_<tag>.pt = [l2n(archive) | l2n(transcript)]
                     (offline double-key; downstream normalisation turns the
                     augmented-key similarity into the equal-weight blend
                     (cos_archive + cos_transcript)/2 inside the alpha channel)

CPU is fine (a few minutes for ~800 texts per dataset). No SLURM needed.
"""
import argparse
import json
import os

import torch

REPO = "/data/jehc223/RGCL"
EMB_ROOT = os.path.join(REPO, "data", "CLIP_Embedding")
GT_ROOT = os.path.join(REPO, "data", "gt")
ARCHIVE_TAG = "archive_openai_clip-vit-large-patch14-336_HF"
SPLITS = ["train", "dev_seen", "test_seen"]
ENCODER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def flat_ids(ids):
    if len(ids) > 0 and isinstance(ids[0], (list, tuple)):
        return [i for sub in ids for i in sub]
    return list(ids)


def load_gt_texts(dataset):
    """id -> text over the union of gt splits (train/val/test)."""
    id2text = {}
    id2label = {}
    for sp in ("train", "val", "test"):
        p = os.path.join(GT_ROOT, dataset, sp + ".jsonl")
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                id2text[r["id"]] = r["text"]
                id2label[r["id"]] = int(r["label"])
    return id2text, id2label


def cmd_encode(args):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ENCODER)
    model.max_seq_length = args.max_seq_length
    tok = model.tokenizer
    print(f"[encoder] {ENCODER} max_seq_length={model.max_seq_length}")

    for dataset in args.datasets:
        id2text, id2label = load_gt_texts(dataset)
        for split in SPLITS:
            arc_path = os.path.join(
                EMB_ROOT, dataset, f"{split}_{ARCHIVE_TAG}.pt")
            arc = torch.load(arc_path, map_location="cpu")
            ids = flat_ids(arc["ids"])
            labels = arc["labels"]

            missing = [i for i in ids if i not in id2text]
            if missing:
                raise ValueError(
                    f"{dataset}/{split}: {len(missing)} ids missing from gt "
                    f"jsonl (first: {missing[:5]})")
            # label sanity vs gt
            mism = [i for k, i in enumerate(ids)
                    if id2label[i] != int(labels[k])]
            if mism:
                raise ValueError(
                    f"{dataset}/{split}: label mismatch vs gt for {mism[:5]}")

            texts = [id2text[i] for i in ids]
            tl = [len(tok(t, truncation=False)["input_ids"]) for t in texts]
            n = len(tl)
            over512 = sum(1 for x in tl if x > args.max_seq_length)
            over77 = sum(1 for x in tl if x > 77)
            print(f"[{dataset}/{split}] n={n} tok-len med={sorted(tl)[n//2]} "
                  f"max={max(tl)} | >{args.max_seq_length}tok: {over512} "
                  f"({100*over512/n:.1f}%) | >77tok: {over77} "
                  f"({100*over77/n:.1f}%)")

            feats = model.encode(
                texts, batch_size=args.batch_size,
                convert_to_tensor=True, show_progress_bar=False,
                normalize_embeddings=False).cpu().float()
            assert feats.shape == (n, 768), feats.shape

            out = os.path.join(
                EMB_ROOT, dataset, f"{split}_transcript_{args.tag}.pt")
            torch.save({"ids": ids, "text_feats": feats,
                        "labels": labels}, out)
            print(f"  -> {out}  feats {tuple(feats.shape)}")


def cmd_concat(args):
    """[l2n(archive) | l2n(transcript)] per row, id-aligned, saved as new .pt."""
    for dataset in args.datasets:
        for split in SPLITS:
            arc_path = os.path.join(
                EMB_ROOT, dataset, f"{split}_{ARCHIVE_TAG}.pt")
            trs_path = os.path.join(
                EMB_ROOT, dataset, f"{split}_transcript_{args.tag}.pt")
            arc = torch.load(arc_path, map_location="cpu")
            trs = torch.load(trs_path, map_location="cpu")
            aids, tids = flat_ids(arc["ids"]), flat_ids(trs["ids"])
            row_of = {v: r for r, v in enumerate(tids)}
            rows = torch.tensor([row_of[v] for v in aids], dtype=torch.long)
            a = torch.nn.functional.normalize(
                arc["text_feats"].float(), p=2, dim=1)
            t = torch.nn.functional.normalize(
                trs["text_feats"].float().index_select(0, rows), p=2, dim=1)
            feats = torch.cat((a, t), dim=1)
            out = os.path.join(
                EMB_ROOT, dataset, f"{split}_arctrans_{args.tag}.pt")
            torch.save({"ids": aids, "text_feats": feats,
                        "labels": arc["labels"]}, out)
            print(f"[{dataset}/{split}] -> {out}  feats {tuple(feats.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["encode", "concat"], default="encode")
    ap.add_argument("--datasets", nargs="+", default=["MHC", "MHC_zh"])
    ap.add_argument("--tag", default="mpnet512_HF")
    ap.add_argument("--max_seq_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    if args.mode == "encode":
        cmd_encode(args)
    else:
        cmd_concat(args)


if __name__ == "__main__":
    main()
