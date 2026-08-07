#!/usr/bin/env python
"""I5 redundancy gate: is the OCR retrieval key redundant with the transcript key?

HateMM-train only (744 videos, frozen 5-fold folds from the Gate-0 run). Zero test contact:
no file under data/gt/HateMM/test.jsonl is opened and no test video is read.

=========================== FROZEN DECISION RULES (written before any number was computed) ===========================
Setting: fold-internal kNN memory. For a query video in outer fold f, the memory is that fold's
train_ids (frozen in artifacts/tera_gate0/.../folds/fold_f/train_ids.json). Keys are L2-normalised
768-d CLIP text-tower pooler embeddings (openai/clip-vit-large-patch14-336, truncation=True), the
same recipe that produced the project's existing whole-video `text_feats`:
  Key_T = CLIP(ASR transcript)      [existing cache, data/CLIP_Embedding/HateMM/train_..._HF.pt]
  Key_O = CLIP(video-level OCR text)[this cache,     data/OCR/HateMM/ocr_video.jsonl]
Query population Q = train videos whose OCR text has >= 20 characters after a conf >= 0.5 filter.

R1 (PRIMARY, the parent question "are the two neighbour sets the same?"):
    ov@10 = mean over Q of |topk_T(10) ∩ topk_O(10)| / 10   (self excluded; memory = fold train ids)
    - REDUNDANT      if ov@10 >= 0.50   -> I5 downgraded: the OCR key retrieves the same neighbours
    - COMPLEMENTARY  if ov@10 <= 0.25   -> the OCR key buys a genuinely different neighbourhood
    - AMBIGUOUS      otherwise
    Chance level (k / |memory|) is reported alongside and must be << the observed value for the
    measurement to mean anything.

R2 (SECONDARY, the IDEA_REPORT I5 gate, transcript-only retrieval vs the Gate-C census):
    purity_gap = purity@20(Key_T | on_screen_text NOT required) - purity@20(Key_T | on_screen_text required)
    recovery   = fraction of census false negatives that are OCR-required-and-speech-not-required
                 whose top-20 Key_T neighbour majority vote is label 1
    - GO-REDUNDANT      if recovery >= 0.30 and purity_gap >= 0.10
    - GO-COMPLEMENTARY  if recovery >= 0.30 and purity_gap <= 0.02
    - NO-GO             if recovery <  0.15
    - AMBIGUOUS         otherwise

R3 (context, no gate): purity@20 under Key_O vs Key_T on Q, and the Key_O recovery on the same
    census subgroup. Reported for interpretation only; it does not change the verdict.
=====================================================================================================================
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/jehc223/Retrieval-hate")
RUN = ROOT / "artifacts/tera_gate0/tera-gate0-20260807T000625Z-7ba80eaf"
WHOLE = ROOT / "data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt"
OCRV = ROOT / "data/OCR/HateMM/ocr_video.jsonl"
OUT = ROOT / "data/OCR/HateMM/i5_redundancy_gate.json"
CLIP_MODEL = "openai/clip-vit-large-patch14-336"
MIN_CONF = 0.5
MIN_CHARS = 20
K1, K2 = 10, 20


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def encode_texts(texts):
    from transformers import CLIPTextModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    mdl = CLIPTextModel.from_pretrained(CLIP_MODEL).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(dev)
    out = []
    B = 64
    with torch.no_grad():
        for i in range(0, len(texts), B):
            batch = [t if t.strip() else " " for t in texts[i:i + B]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(dev) for k, v in enc.items()}
            out.append(mdl(**enc).pooler_output.float().cpu())
    return torch.cat(out).numpy().astype(np.float64)


def topk_neighbors(Kq, Kmem, mem_idx, k, self_pos):
    """cosine top-k over the memory, excluding the query itself."""
    sim = Kq @ Kmem.T
    if self_pos is not None and self_pos >= 0:
        sim[self_pos] = -np.inf
    order = np.argsort(-sim)[:k]
    return mem_idx[order]


def main():
    who = torch.load(WHOLE, map_location="cpu")
    ids = who["ids"][0] if (len(who["ids"]) == 1 and isinstance(who["ids"][0], list)) else who["ids"]
    ids = list(ids)
    idx = {v: i for i, v in enumerate(ids)}
    y = who["labels"].numpy().astype(int)
    KT = l2(who["text_feats"].numpy().astype(np.float64))

    # ---- OCR video-level text -> Key_O (same CLIP text tower / same truncation recipe)
    ocr_raw = {}
    with open(OCRV) as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                ocr_raw[r["video_id"]] = r
    # rebuild the text under the conf filter from the window file
    win = ROOT / "data/OCR/HateMM/ocr_windows_K30.jsonl"
    buf = {v: [] for v in ids}
    with open(win) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["video_id"] not in buf:
                continue
            keep = [d["text"].strip() for d in r["texts"] if d["conf"] >= MIN_CONF and d["text"].strip()]
            if keep:
                buf[r["video_id"]].append(" ".join(keep))
    ocr_text = [" ".join(buf[v]) for v in ids]
    n_chars = np.array([len(t) for t in ocr_text])
    KO = l2(encode_texts(ocr_text))

    folds = []
    for f in range(5):
        tr = json.load(open(RUN / f"folds/fold_{f}/train_ids.json"))
        qu = json.load(open(RUN / f"folds/fold_{f}/query_ids.json"))
        folds.append((np.array([idx[v] for v in tr]), np.array([idx[v] for v in qu])))
    fold_of = np.full(len(ids), -1, dtype=int)
    for f, (_, qu) in enumerate(folds):
        fold_of[qu] = f

    Q = np.where(n_chars >= MIN_CHARS)[0]
    res = {
        "n_videos": len(ids),
        "min_conf": MIN_CONF,
        "min_chars": MIN_CHARS,
        "n_query_population": int(len(Q)),
        "frac_query_population": round(float(len(Q) / len(ids)), 4),
        "mean_ocr_chars_all": float(n_chars.mean()),
    }

    # ---------------------------------------------------------------- R1 + R3
    ov1, ov2, purT, purO = [], [], [], []
    mem_sizes = []
    for i in Q:
        f = fold_of[i]
        if f < 0:
            continue
        mem = folds[f][0]
        self_pos = int(np.where(mem == i)[0][0]) if (mem == i).any() else -1
        mem_sizes.append(len(mem) - (1 if self_pos >= 0 else 0))
        nT1 = topk_neighbors(KT[i], KT[mem], mem, K1, self_pos)
        nO1 = topk_neighbors(KO[i], KO[mem], mem, K1, self_pos)
        ov1.append(len(set(nT1) & set(nO1)) / K1)
        nT2 = topk_neighbors(KT[i], KT[mem], mem, K2, self_pos)
        nO2 = topk_neighbors(KO[i], KO[mem], mem, K2, self_pos)
        ov2.append(len(set(nT2) & set(nO2)) / K2)
        purT.append(float((y[nT2] == y[i]).mean()))
        purO.append(float((y[nO2] == y[i]).mean()))
    ov1 = np.array(ov1); ov2 = np.array(ov2)
    chance1 = K1 / float(np.mean(mem_sizes))
    chance2 = K2 / float(np.mean(mem_sizes))
    res["R1"] = {
        "overlap_at_10_mean": round(float(ov1.mean()), 4),
        "overlap_at_10_median": round(float(np.median(ov1)), 4),
        "overlap_at_20_mean": round(float(ov2.mean()), 4),
        "chance_overlap_at_10": round(float(chance1), 4),
        "chance_overlap_at_20": round(float(chance2), 4),
        "frac_queries_zero_overlap_at_10": round(float((ov1 == 0).mean()), 4),
    }
    res["R1"]["verdict"] = ("REDUNDANT" if ov1.mean() >= 0.50
                            else "COMPLEMENTARY" if ov1.mean() <= 0.25
                            else "AMBIGUOUS")
    res["R3"] = {
        "purity_at_20_key_T": round(float(np.mean(purT)), 4),
        "purity_at_20_key_O": round(float(np.mean(purO)), 4),
        "delta_O_minus_T": round(float(np.mean(purO) - np.mean(purT)), 4),
    }

    # ---------------------------------------------------------------- R2
    rows = [json.loads(l) for l in open(RUN / "gate_c_audit.jsonl")]
    final = {}
    for r in rows:
        if r["coder_id"].endswith("c1"):
            final[r["video_id"]] = r
    for r in rows:
        if r["coder_id"].endswith("adj"):
            final[r["video_id"]] = r
    sample = json.load(open(RUN / "gate_c_sample.json"))
    fn_ids = set(sample["audit_fn"])

    def knn(i, key, k):
        f = fold_of[i]
        mem = folds[f][0]
        self_pos = int(np.where(mem == i)[0][0]) if (mem == i).any() else -1
        return topk_neighbors(key[i], key[mem], mem, k, self_pos)

    aud = [v for v in final if v in idx and fold_of[idx[v]] >= 0]
    req = {v: set(final[v]["required_modalities"]) for v in aud}
    grpA = [v for v in aud if "on_screen_text" in req[v]]        # OCR required
    grpB = [v for v in aud if "on_screen_text" not in req[v]]    # OCR not required
    def purity(vs, key):
        if not vs:
            return float("nan")
        return float(np.mean([(y[knn(idx[v], key, K2)] == y[idx[v]]).mean() for v in vs]))
    pA, pB = purity(grpA, KT), purity(grpB, KT)
    gap = pB - pA

    census_fn_ocr_nospeech = [v for v in aud
                              if v in fn_ids and "on_screen_text" in req[v] and "speech" not in req[v]]
    def recovery(vs, key):
        if not vs:
            return float("nan")
        hits = 0
        for v in vs:
            nb = knn(idx[v], key, K2)
            hits += int(y[nb].mean() >= 0.5)
        return hits / len(vs)
    recT = recovery(census_fn_ocr_nospeech, KT)
    recO = recovery(census_fn_ocr_nospeech, KO)

    res["R2"] = {
        "n_census": len(aud),
        "n_ocr_required": len(grpA),
        "n_ocr_not_required": len(grpB),
        "purity_at_20_ocr_required": round(pA, 4),
        "purity_at_20_ocr_not_required": round(pB, 4),
        "purity_gap": round(gap, 4),
        "n_census_fn_ocr_required_no_speech": len(census_fn_ocr_nospeech),
        "recovery_key_T": round(recT, 4) if recT == recT else None,
    }
    if recT != recT:
        v2 = "UNDEFINED (empty subgroup)"
    elif recT >= 0.30 and gap >= 0.10:
        v2 = "GO-REDUNDANT"
    elif recT >= 0.30 and gap <= 0.02:
        v2 = "GO-COMPLEMENTARY"
    elif recT < 0.15:
        v2 = "NO-GO"
    else:
        v2 = "AMBIGUOUS"
    res["R2"]["verdict"] = v2
    res["R3"]["recovery_key_O_same_subgroup"] = round(recO, 4) if recO == recO else None

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
