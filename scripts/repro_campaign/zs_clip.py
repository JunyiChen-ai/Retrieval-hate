#!/usr/bin/env python
"""REPRO campaign, Wave 0 item 1 — ZS-CLIP frame-similarity floor, four datasets.

Protocol: `idea-stage/REPRO_CAMPAIGN_FREEZE.md` (frozen 74b9d87). Assets: Phase A
(`idea-stage/repro_campaign/PHASE_A_STATUS.md`).

Method (LAVAD, CVPR 2024, baseline row "ZS CLIP"):
    for each frame f and a prompt pair (p_normal, p_anomalous),
        s_n = cos(img(f), txt(p_normal)),  s_a = cos(img(f), txt(p_anomalous))
        score(f) = softmax_a([s_n, s_a] * logit_scale) = sigmoid(logit_scale * (s_a - s_n))
    The LAVAD repo (third_party/lavad) ships no ZS-CLIP script — it is a paper
    baseline row only — so the formula above is our reading of the paper text.
    NOTE, recorded because it removes the ambiguity entirely: softmax over the
    two prompts and the raw difference (s_a - s_n) are related by a strictly
    increasing map, so **frame ROC-AUC and PR-AUC are bit-identical under either
    normalisation**. The implementation choice is therefore not load-bearing for
    any number in the campaign table. We store the difference and report the
    softmax probability.

Image side: the Phase A cache stores CLIPVisionModel `pooler_output` (1024-d,
pre-projection). `CLIPModel.get_image_features` is exactly
`visual_projection(pooler_output)`, so applying the frozen 1024->768 projection
to the cache reconstructs the joint-space image embedding without re-decoding a
single frame. `--check-projection` verifies this against a live decode.

Stages (each idempotent — an existing output file is skipped):
  1  prompts    text-tower encode of every prompt in PROMPT_SETS         [GPU, seconds]
  2  visual     per-video (n_sets, T) score-difference arrays            [CPU, I/O bound]
  3  text       text-tower encode of ASR+OCR window strings (+text)      [GPU, minutes]
  4  eval       pooled frame ROC-AUC / PR-AUC + controls + tables        [CPU]

Usage
  python scripts/repro_campaign/zs_clip.py --stage all
  python scripts/repro_campaign/zs_clip.py --stage eval --datasets HateMM
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
OUT = ROOT / "idea-stage/repro_zs_clip"
CLIP_MODEL = "openai/clip-vit-large-patch14-336"
DATASETS = ["HateMM", "MHC", "MHC_zh", "HateClipSeg"]
DS_LABEL = {"HateMM": "HateMM", "MHC": "MHC-EN", "MHC_zh": "MHC-ZH", "HateClipSeg": "HateClipSeg"}
FPS = 4.0
K_TEXT = 30  # the OCR cache grid; also the +text native window (freeze §13)

# ---------------------------------------------------------------- prompt sets
# `main` is the row fixed by the campaign brief. Everything else is the
# prompt-sensitivity appendix, reported in full; no prompt set is selected on
# any metric (freeze §10 red line 1).
PROMPT_SETS = {
    "main":     ("a normal video frame", "a hateful video frame"),
    "sens_hs":  ("a normal video frame", "a video frame containing hate speech"),
    "sens_off": ("a normal video frame", "a video frame with offensive content"),
    "sens_lavad": ("a normal video frame", "an anomalous video frame"),  # LAVAD's own wording
}
# HateClipSeg per-class appendix: 5 toxic classes of the released 6-way label
# (0 normal is the shared negative prompt), aggregated by max = "any toxic".
HCS_CLASS_PROMPTS = {
    "hateful":   "a hateful video frame",
    "insulting": "an insulting video frame",
    "sexual":    "a sexually explicit video frame",
    "violence":  "a violent video frame",
    "harm":      "a self-harm or harmful video frame",
}
HCS_NEG = "a normal video frame"

LOGIT_SCALE = 100.0  # CLIP's exp(logit_scale) at the released checkpoint

ASR_GLOB = {
    "HateMM": ["data/ASR/HateMM/*_asrK4_whisper-large-v3.jsonl"],
    "MHC": ["data/ASR/MHC/*_asrK4_whisper-large-v3.jsonl"],
    "MHC_zh": ["data/ASR/MHC_zh/*_asrK4_whisper-large-v3.jsonl"],
    "HateClipSeg": ["data/ASR/HateClipSeg/*_asrK4_whisper-large-v3.jsonl"],
}
OCR_FILES = {
    "HateMM": ["data/OCR/HateMM/ocr_windows_K30.jsonl",
               "data/OCR/HateMM/ocr_windows_K30_test.jsonl"],
    "MHC": ["data/OCR/MHC/ocr_windows_K30.jsonl",
            "data/OCR/MHC_test/ocr_windows_K30.jsonl"],
    "MHC_zh": ["data/OCR/MHC_zh/ocr_windows_K30.jsonl",
               "data/OCR/MHC_zh_test/ocr_windows_K30.jsonl"],
    "HateClipSeg": ["data/OCR/HateClipSeg/ocr_windows_K30.jsonl"],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def feat_dir(ds: str) -> Path:
    return ROOT / f"data/CLIP_Embedding/{ds}/dense4fps_clipL336"


# ------------------------------------------------------------------ stage 1/3
def load_clip(device="cuda"):
    import torch
    from transformers import CLIPModel, CLIPTokenizerFast
    tok = CLIPTokenizerFast.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL, torch_dtype=torch.float32).to(device).eval()
    return tok, model


def encode_texts(texts, tok, model, device="cuda", bs=256, tag="") -> np.ndarray:
    import torch
    out = np.zeros((len(texts), model.config.projection_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            enc = tok(batch, padding=True, truncation=True, max_length=77, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            f = model.get_text_features(**enc)
            f = f / f.norm(dim=-1, keepdim=True)
            out[i:i + len(batch)] = f.float().cpu().numpy()
            if tag and (i // bs) % 40 == 0:
                log(f"  [text {tag}] {i + len(batch)}/{len(texts)}")
    return out


def stage_prompts(device="cuda") -> None:
    p = OUT / "prompt_emb.npz"
    if p.exists():
        log("stage prompts: exists, skip")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    tok, model = load_clip(device)
    names, texts = [], []
    for k, (neg, pos) in PROMPT_SETS.items():
        names += [f"{k}|neg", f"{k}|pos"]
        texts += [neg, pos]
    for k, pos in HCS_CLASS_PROMPTS.items():
        names += [f"hcs_{k}|pos"]
        texts += [pos]
    names += ["hcs|neg"]
    texts += [HCS_NEG]
    emb = encode_texts(texts, tok, model, device)
    # also persist the visual projection so stage 2 needs no GPU / no model
    W = model.visual_projection.weight.detach().cpu().numpy()  # (768, 1024)
    np.savez(p, names=np.array(names), emb=emb, visual_projection=W, texts=np.array(texts))
    log(f"stage prompts: wrote {p} ({len(names)} prompts)")


def check_projection(device="cuda") -> None:
    """Verify visual_projection(cached pooler_output) == get_image_features(frame)."""
    import torch
    ds, vid = "HateClipSeg", None
    for f in sorted(feat_dir(ds).glob("*.npy")):
        vid = f.stem
        break
    vpath = None
    for ext in (".mp4", ".webm", ".mkv"):
        c = ROOT / f"data/video/HateClipSeg/All/{vid}{ext}"
        if c.exists():
            vpath = c
            break
    if vpath is None:
        log("check-projection: source video not found, skipped")
        return
    SIZE = 336
    vf = (f"fps=4,scale=w={SIZE}:h={SIZE}:force_original_aspect_ratio=increase,"
          f"crop={SIZE}:{SIZE}")
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(vpath), "-map", "0:v:0",
           "-vf", vf, "-frames:v", "4", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (SIZE * SIZE * 3)
    frames = np.frombuffer(raw, np.uint8).reshape(n, SIZE, SIZE, 3).astype(np.float32) / 255.0
    MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
    x = ((frames - MEAN) / STD).transpose(0, 3, 1, 2)
    _, model = load_clip(device)
    with torch.no_grad():
        live = model.get_image_features(pixel_values=torch.from_numpy(x).to(device)).cpu().numpy()
    cached = np.load(feat_dir(ds) / f"{vid}.npy")[:n]
    W = model.visual_projection.weight.detach().cpu().numpy()
    recon = cached @ W.T
    ln = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
    cos = float((ln(live) * ln(recon)).sum(-1).mean())
    log(f"check-projection {vid}: n={n} mean cosine(live, projection(cache)) = {cos:.6f}, "
        f"max|diff| = {np.abs(live - recon).max():.4f}")


# -------------------------------------------------------------------- stage 2
def stage_visual(ds: str) -> None:
    p = OUT / f"scores_visual_{ds}.npz"
    if p.exists():
        log(f"stage visual {ds}: exists, skip")
        return
    pr = np.load(OUT / "prompt_emb.npz", allow_pickle=True)
    names = list(pr["names"])
    emb, W = pr["emb"], pr["visual_projection"]
    idx = {n: i for i, n in enumerate(names)}

    set_names = list(PROMPT_SETS)
    negs = np.stack([emb[idx[f"{k}|neg"]] for k in set_names])
    poss = np.stack([emb[idx[f"{k}|pos"]] for k in set_names])
    if ds == "HateClipSeg":
        hcs_names = list(HCS_CLASS_PROMPTS)
        negs = np.concatenate([negs, np.stack([emb[idx["hcs|neg"]]] * len(hcs_names))])
        poss = np.concatenate([poss, np.stack([emb[idx[f"hcs_{k}|pos"]] for k in hcs_names])])
        set_names = set_names + [f"hcs_{k}" for k in hcs_names]
    D = (poss - negs).astype(np.float32)  # (S, 768); score diff = <img_hat, pos-neg>

    z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
    vids = [str(v) for v in z["video_ids"]]
    res, missing, t0 = {}, [], time.time()
    for i, vid in enumerate(vids):
        f = feat_dir(ds) / f"{vid}.npy"
        if not f.exists():
            missing.append(vid)
            continue
        a = np.load(f).astype(np.float32)
        img = a @ W.T
        img /= np.linalg.norm(img, axis=-1, keepdims=True) + 1e-12
        res[vid] = (img @ D.T).T.astype(np.float32)  # (S, T)
        if (i + 1) % 100 == 0:
            log(f"  [visual {ds}] {i + 1}/{len(vids)} videos, {time.time() - t0:.0f}s")
    np.savez_compressed(p, set_names=np.array(set_names), missing=np.array(missing), **res)
    log(f"stage visual {ds}: wrote {p}, {len(res)} videos, {len(missing)} missing, "
        f"{time.time() - t0:.0f}s")


# -------------------------------------------------------------------- stage 3
_WS = re.compile(r"\s+")


def load_asr(ds: str) -> dict:
    """video_id -> list of (start, end, word)."""
    import glob
    out = {}
    for pat in ASR_GLOB[ds]:
        for f in glob.glob(str(ROOT / pat)):
            for line in open(f):
                r = json.loads(line)
                ch = r.get("chunks") or []
                items = []
                for c in ch:
                    if not isinstance(c, (list, tuple)) or len(c) < 3:
                        continue
                    s, e, t = c[0], c[1], c[2]
                    if s is None:
                        continue
                    if e is None:
                        e = s
                    items.append((float(s), float(e), str(t)))
                out.setdefault(str(r["id"]), []).extend(items)
    return out


def load_ocr(ds: str) -> dict:
    """video_id -> {window_k: text}."""
    out = {}
    for rel in OCR_FILES[ds]:
        f = ROOT / rel
        if not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            txt = " ".join(t["text"] for t in (r.get("texts") or []) if t.get("text"))
            txt = _WS.sub(" ", txt).strip()
            out.setdefault(str(r["video_id"]), {})[int(r["window_k"])] = txt
    return out


def build_window_strings(ds: str):
    """Per video, K_TEXT window strings following freeze §8.

    The +text native window is the OCR cache's own K=30 grid, window k covering
    [k*D/K, (k+1)*D/K).  String = ASR words overlapping the window, in time
    order, then the OCR window text, joined by " | ".  Empty when neither
    channel has text.
    """
    z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
    vids = [str(v) for v in z["video_ids"]]
    dur = {v: float(d) for v, d in zip(vids, z["duration"])}
    asr, ocr = load_asr(ds), load_ocr(ds)
    per_video = {}
    for v in vids:
        D = max(dur[v], 1e-6)
        edges = np.arange(K_TEXT + 1) * D / K_TEXT
        buckets = [[] for _ in range(K_TEXT)]
        for s, e, t in asr.get(v, []):
            k0 = int(np.clip(np.searchsorted(edges, s, "right") - 1, 0, K_TEXT - 1))
            k1 = int(np.clip(np.searchsorted(edges, max(e, s), "right") - 1, 0, K_TEXT - 1))
            for k in range(k0, k1 + 1):
                buckets[k].append(t)
        ow = ocr.get(v, {})
        strings = []
        for k in range(K_TEXT):
            a = _WS.sub(" ", "".join(buckets[k])).strip()
            o = ow.get(k, "")
            parts = [x for x in (a, o) if x]
            strings.append(" | ".join(parts))
        per_video[v] = strings
    return vids, per_video


def stage_text(ds: str, device="cuda") -> None:
    p = OUT / f"scores_text_{ds}.npz"
    if p.exists():
        log(f"stage text {ds}: exists, skip")
        return
    pr = np.load(OUT / "prompt_emb.npz", allow_pickle=True)
    names = list(pr["names"])
    emb = pr["emb"]
    idx = {n: i for i, n in enumerate(names)}
    set_names = list(PROMPT_SETS)
    negs = np.stack([emb[idx[f"{k}|neg"]] for k in set_names])
    poss = np.stack([emb[idx[f"{k}|pos"]] for k in set_names])
    if ds == "HateClipSeg":
        hn = list(HCS_CLASS_PROMPTS)
        negs = np.concatenate([negs, np.stack([emb[idx["hcs|neg"]]] * len(hn))])
        poss = np.concatenate([poss, np.stack([emb[idx[f"hcs_{k}|pos"]] for k in hn])])
        set_names = set_names + [f"hcs_{k}" for k in hn]
    D = (poss - negs).astype(np.float32)

    t0 = time.time()
    vids, per_video = build_window_strings(ds)
    log(f"  [text {ds}] built window strings for {len(vids)} videos in {time.time() - t0:.0f}s")

    flat, owner = [], []
    for v in vids:
        for k, s in enumerate(per_video[v]):
            if s:
                flat.append(s)
                owner.append((v, k))
    log(f"  [text {ds}] {len(flat)} non-empty windows of {len(vids) * K_TEXT}")
    tok, model = load_clip(device)
    E = encode_texts(flat, tok, model, device, tag=ds) if flat else np.zeros((0, 768), np.float32)
    del model
    import torch
    torch.cuda.empty_cache()

    diffs = E @ D.T if len(flat) else np.zeros((0, len(set_names)), np.float32)
    res = {}
    for v in vids:
        res[v] = np.zeros((len(set_names), K_TEXT), np.float32)
        res[v + "//mask"] = np.zeros(K_TEXT, np.int8)
    for j, (v, k) in enumerate(owner):
        res[v][:, k] = diffs[j]
        res[v + "//mask"][k] = 1
    np.savez_compressed(p, set_names=np.array(set_names), **res)
    log(f"stage text {ds}: wrote {p}, {time.time() - t0:.0f}s")


# -------------------------------------------------------------------- stage 4
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def eval_pool(y, s):
    from sklearn.metrics import average_precision_score, roc_auc_score
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def stage_eval(datasets) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score
    ctrl = json.load(open(ROOT / "idea-stage/repro_campaign/gt_controls.json"))
    rows = []
    for ds in datasets:
        z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
        vids = [str(v) for v in z["video_ids"]]
        split = {v: str(s) for v, s in zip(vids, z["split"])}
        nspans = {v: int(n) for v, n in zip(vids, z["n_spans"])}
        y4 = {v: np.asarray(a).astype(np.int8) for v, a in zip(vids, z["y4"])}
        yvid = {v: int(a) for v, a in zip(vids, z["y_video"])}

        V = np.load(OUT / f"scores_visual_{ds}.npz", allow_pickle=True)
        set_names = [str(s) for s in V["set_names"]]
        T = np.load(OUT / f"scores_text_{ds}.npz", allow_pickle=True)
        have = [v for v in vids if v in V.files]
        log(f"[eval {ds}] {len(have)} videos with features")

        # ---- assemble pooled arrays, per (split, stratum)
        def pools():
            yield "full", lambda v: True
            yield "test", lambda v: split[v] == "test"

        strata = [("all", lambda v: True)]
        if ds in ("HateMM", "MHC", "MHC_zh"):
            strata += [("single_span", lambda v: nspans[v] <= 1),
                       ("multi_span", lambda v: nspans[v] == 0 or nspans[v] >= 2)]

        for split_name, split_ok in pools():
            for st_name, st_ok in strata:
                sel = [v for v in have if split_ok(v) and st_ok(v)]
                if not sel:
                    continue
                ys, vs, ts, ms, bc = [], [], [], [], []
                for v in sel:
                    yv = y4[v]
                    sv = V[v]                       # (S, T_feat)
                    n = min(len(yv), sv.shape[1])
                    if n == 0:
                        continue
                    ys.append(yv[:n])
                    vs.append(sv[:, :n])
                    # text window index for each frame: k = floor(t / (D/K))
                    tw = T[v] if v in T.files else None
                    if tw is not None:
                        Tg = len(yv)
                        k = np.minimum((np.arange(n) * K_TEXT) // max(Tg, 1), K_TEXT - 1)
                        ts.append(tw[:, k])
                        ms.append(T[v + "//mask"][k])
                    bc.append(np.full(n, float(yvid[v])))
                if not ys:
                    continue
                y = np.concatenate(ys)
                if y.min() == y.max():
                    # a stratum can be single-class (e.g. MHC-ZH test has no
                    # multi-span video, so that pool is all-negative). ROC/AP are
                    # undefined; the slice is reported as n/a, never dropped
                    # silently.
                    log(f"  [eval {ds}] {split_name}/{st_name}: single-class pool "
                        f"({len(y)} frames, {len(sel)} videos), metrics n/a")
                    rows.append(dict(method="ZS-CLIP", variant="base", prompt_set="all",
                                     dataset=DS_LABEL[ds], split=split_name, stratum=st_name,
                                     n_frames=int(len(y)), base_rate=float(y.mean()),
                                     n_videos=len(sel), roc=None, ap=None, ap_norm=None,
                                     note="single-class pool, metrics undefined"))
                    continue
                Sv = np.concatenate(vs, axis=1)
                St = np.concatenate(ts, axis=1) if ts else None
                Mt = np.concatenate(ms) if ms else None
                broadcast = np.concatenate(bc)
                n_frames = len(y)
                base = float(y.mean())
                ap_bc = float(average_precision_score(y, broadcast))
                roc_bc = float(roc_auc_score(y, broadcast))
                # random floor, 20 seeds (freeze §3)
                rr, ra = [], []
                for k in range(20):
                    rng = np.random.default_rng(20250819 + k)
                    r = rng.random(n_frames)
                    rr.append(roc_auc_score(y, r))
                    ra.append(average_precision_score(y, r))
                ap_rand = float(np.mean(ra))
                norm = lambda ap: (ap - ap_rand) / (ap_bc - ap_rand) if ap_bc > ap_rand else float("nan")

                common = dict(dataset=DS_LABEL[ds], split=split_name, stratum=st_name,
                              n_frames=n_frames, base_rate=base, n_videos=len(sel))
                rows.append(dict(method="GOLD_BROADCAST", variant="control", prompt_set="n/a",
                                 roc=roc_bc, ap=ap_bc, ap_norm=1.0, **common))
                rows.append(dict(method="RANDOM_UNIFORM", variant="control", prompt_set="n/a",
                                 roc=float(np.mean(rr)), ap=ap_rand, ap_norm=0.0,
                                 roc_sd=float(np.std(rr)), ap_sd=float(np.std(ra)), **common))

                for si, sn in enumerate(set_names):
                    if sn.startswith("hcs_") and st_name != "all":
                        continue
                    sc = sigmoid(LOGIT_SCALE * Sv[si])
                    roc, ap = eval_pool(y, sc)
                    rows.append(dict(method="ZS-CLIP", variant="base", prompt_set=sn,
                                     roc=roc, ap=ap, ap_norm=norm(ap), **common))
                    if St is not None:
                        tsc = sigmoid(LOGIT_SCALE * St[si])
                        fused = np.where(Mt > 0, 0.5 * sc + 0.5 * tsc, sc)
                        roc2, ap2 = eval_pool(y, fused)
                        rows.append(dict(method="ZS-CLIP", variant="+text (ours)", prompt_set=sn,
                                         roc=roc2, ap=ap2, ap_norm=norm(ap2),
                                         text_cov=float((Mt > 0).mean()), **common))
                # HateClipSeg 6-class any-toxic aggregation
                hcs_idx = [i for i, s in enumerate(set_names) if s.startswith("hcs_")]
                if hcs_idx:
                    sc = sigmoid(LOGIT_SCALE * Sv[hcs_idx].max(axis=0))
                    roc, ap = eval_pool(y, sc)
                    rows.append(dict(method="ZS-CLIP", variant="base", prompt_set="hcs6_anytoxic_max",
                                     roc=roc, ap=ap, ap_norm=norm(ap), **common))
                    if St is not None:
                        tsc = sigmoid(LOGIT_SCALE * St[hcs_idx].max(axis=0))
                        fused = np.where(Mt > 0, 0.5 * sc + 0.5 * tsc, sc)
                        roc2, ap2 = eval_pool(y, fused)
                        rows.append(dict(method="ZS-CLIP", variant="+text (ours)",
                                         prompt_set="hcs6_anytoxic_max",
                                         roc=roc2, ap=ap2, ap_norm=norm(ap2),
                                         text_cov=float((Mt > 0).mean()), **common))
                log(f"  [eval {ds}] {split_name}/{st_name}: {n_frames} frames done")
        del V, T
    p = OUT / "results.json"
    json.dump(rows, open(p, "w"), indent=1)
    log(f"stage eval: wrote {p}, {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "prompts", "visual", "text", "eval", "check"])
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dss = [d for d in a.datasets.split(",") if d]
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("all", "prompts"):
        stage_prompts(a.device)
    if a.stage == "check":
        stage_prompts(a.device)
        check_projection(a.device)
        return
    if a.stage in ("all", "visual"):
        for d in dss:
            stage_visual(d)
    if a.stage in ("all", "text"):
        for d in dss:
            stage_text(d, a.device)
    if a.stage in ("all", "eval"):
        stage_eval(dss)


if __name__ == "__main__":
    main()
