#!/usr/bin/env python3
"""Build per-second OCR sentence embeddings (ocr_bert_1fps).

For each video: K=30 OCR windows (data/OCR/<dir>/ocr_windows_K30*.jsonl);
second t maps to window k = min(int(t/D*30), 29); texts with conf >= .5 are
concatenated (cap 400 chars) and embedded with bert-base-uncased CLS (768-d;
bert-base-chinese for mhclip_zh). Empty -> zero vector. T comes from the
vggish duration reference (same as every other channel).

Usage: python build_ocr_features.py --corpus hateclipseg
Output: results/reproduction/features/ocr_bert_1fps/<corpus>/<vid>.npy
        + coverage report runs/20260831_ocr_channel/build_<corpus>.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
from hate_common import data as hdata  # noqa: E402

OCR_DIRS = {"hatemm": "HateMM", "mhclip_en": "MHC", "mhclip_zh": "MHC_zh",
            "hateclipseg": "HateClipSeg"}
OUT_ROOT = os.path.join(REPO, "results", "reproduction", "features",
                        "ocr_bert_1fps")
RUN_DIR = os.path.join(REPO, "runs", "20260831_ocr_channel")
VGGISH = os.path.join(REPO, "results", "reproduction", "features", "vggish_1s")
K = 30
CONF, CAP = 0.5, 400


def load_windows(corpus):
    d = OCR_DIRS[corpus]
    table = {}
    for suffix in ("", "_test"):
        path = os.path.join(REPO, "data", "OCR", d,
                            "ocr_windows_K30%s.jsonl" % suffix)
        if not os.path.exists(path):
            continue
        for line in open(path):
            row = json.loads(line)
            texts = " ".join(t["text"] for t in row.get("texts", [])
                             if t.get("conf", 0) >= CONF).strip()[:CAP]
            table.setdefault(row["video_id"], {})[row["window_k"]] = texts
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    args = ap.parse_args()
    corpus = args.corpus
    os.makedirs(os.path.join(OUT_ROOT, corpus), exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer
    model_id = ("bert-base-chinese" if corpus == "mhclip_zh"
                else "bert-base-uncased")
    tok = AutoTokenizer.from_pretrained(model_id)
    enc = AutoModel.from_pretrained(model_id).to("cuda").eval()

    windows = load_windows(corpus)
    vids = set()
    for split in ("train", "val", "test"):
        try:
            vids |= set(hdata.load_split(corpus, split))
        except Exception:
            pass

    cache = {}

    def embed(text):
        if text not in cache:
            with torch.no_grad():
                inp = tok(text, return_tensors="pt", truncation=True,
                          max_length=128).to("cuda")
                cache[text] = enc(**inp).last_hidden_state[0, 0].cpu().numpy()
        return cache[text]

    n_ok = n_missing = n_empty_all = 0
    for i, vid in enumerate(sorted(vids)):
        out_path = os.path.join(OUT_ROOT, corpus, vid + ".npy")
        if os.path.exists(out_path):
            continue
        try:
            T = int(np.load(os.path.join(VGGISH, corpus, vid + ".npy"),
                            mmap_mode="r").shape[0])
        except FileNotFoundError:
            continue
        feats = np.zeros((T, 768), dtype=np.float32)
        wins = windows.get(vid)
        if wins is None:
            n_missing += 1
        else:
            any_text = False
            for t in range(T):
                k = min(int(t / max(T, 1) * K), K - 1)
                text = wins.get(k, "")
                if text:
                    feats[t] = embed(text)
                    any_text = True
            n_ok += 1
            if not any_text:
                n_empty_all += 1
        np.save(out_path, feats)
        if (i + 1) % 100 == 0:
            print("PROGRESS %s %d/%d" % (corpus, i + 1, len(vids)), flush=True)
    report = {"corpus": corpus, "videos": len(vids), "with_ocr": n_ok,
              "missing_from_cache": n_missing, "all_empty": n_empty_all,
              "conf": CONF, "cap": CAP, "encoder": model_id}
    with open(os.path.join(RUN_DIR, "build_%s.json" % corpus), "w") as fh:
        json.dump(report, fh, indent=1)
    print("DONE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
