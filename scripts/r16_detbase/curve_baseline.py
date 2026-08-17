#!/usr/bin/env python
"""R16-DETBASE: the project's own per-window score-curve test bed, run on TEST.

The 16.0 / 23.8 F1@tIoU0.5 figures this project quotes were measured out-of-fold inside
train (rounds 13-15) or on val with val-based epoch selection (round 14 recon).  To decompose
the gap against ActionFormer honestly we need the same test bed measured on the same 119-video
test split, under the same GT convention and the same matcher.  That is what this produces.

Head, optimiser, epoch budget and feature blocks are R11/R14/R15's, unchanged
(`scripts/r11_seg/run_pilot.py:PerWin`, 40 epochs, AdamW 1e-3/1e-2).  The decoder grid is swept
on VAL only; test is scored once with the val-selected configuration.

Outputs `idea-stage/r16_detbase/out/curve_baseline.json` and, for the diagnostics, the
per-window test scores + window bounds.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "scripts/r11_seg"))
sys.path.insert(0, str(ROOT / "scripts/r16_detbase"))
import run_pilot as RP                                    # noqa: E402
from eval_f1 import match_prf                             # noqa: E402

K = 30
EPOCHS = 40
OUT = ROOT / "idea-stage/r16_detbase/out"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# decoder grid, swept on val only
G_Q = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]     # score quantile -> threshold
G_W = [1, 3, 5]                                      # running-mean width
G_GAP = [0.0, 5.0, 10.0]                             # merge gap (s)
G_MIN = [0.0, 6.0, 12.0, 20.0]                       # min interval length (s)


def l2(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def blocks_of(segs):
    out, cur = [], None
    for s, e, mh in segs:
        if sum(mh[1:]) > 0:
            cur = [s, e] if cur is None else [cur[0], e]
        else:
            if cur:
                out.append(tuple(cur))
                cur = None
    if cur:
        out.append(tuple(cur))
    return out


def build_blocks(vids):
    ct = torch.load(ROOT / "data/CLIP_Embedding/HateClipSeg/"
                    "test_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt",
                    map_location="cpu")
    assert list(ct["video_ids"]) == vids
    V = l2(ct["subclip_img_feats"].float().numpy().reshape(len(vids), K, -1))
    t = np.load(ROOT / "idea-stage/r11_seg/out/text_feats.npz")
    assert [str(x) for x in t["video_ids"]] == vids
    Tc, Oc = l2(t["asr_feat"]), l2(t["ocr_feat"])
    Tm = t["asr_mask"].astype(np.float32)[..., None]
    Om = t["ocr_mask"].astype(np.float32)[..., None]
    a = np.load(ROOT / "idea-stage/r11_seg/out/audio_feats.npz", allow_pickle=True)
    assert [str(x) for x in a["video_ids"]] == vids
    A = l2(a["w2v"])
    return dict(V=V.astype(np.float32), T=Tc.astype(np.float32), O=Oc.astype(np.float32),
                A=A.astype(np.float32), M=np.concatenate([Tm, Om], -1).astype(np.float32))


def decode(scores, bounds, ids, thr, w, gap, minlen):
    preds = {}
    for j, v in enumerate(ids):
        s = scores[j].astype(float)
        if w > 1:
            s = uniform_filter1d(s, size=w, mode="nearest")
        wb = bounds[j]
        out, cur = [], None
        for k in range(K):
            if s[k] >= thr:
                cur = [wb[k, 0], wb[k, 1]] if cur is None else [cur[0], wb[k, 1]]
            else:
                if cur:
                    out.append(cur)
                    cur = None
        if cur:
            out.append(cur)
        m = []
        for iv in out:
            if m and iv[0] - m[-1][1] <= gap:
                m[-1][1] = iv[1]
            else:
                m.append(list(iv))
        preds[v] = [(x[0], x[1], 1.0) for x in m if x[1] - x[0] >= minlen]
    return preds


def fit(X, y, tr, seed):
    mu = X[tr].reshape(-1, X.shape[-1]).mean(0)
    sd = np.maximum(X[tr].reshape(-1, X.shape[-1]).std(0), 1e-6)
    Xn = ((X - mu) / sd).astype(np.float32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = RP.PerWin(Xn.shape[-1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    Xtr = torch.tensor(Xn[tr]).to(DEV)
    ytr = torch.tensor(y[tr]).to(DEV)
    for _ in range(EPOCHS):
        m.train()
        opt.zero_grad()
        loss = F.cross_entropy(m(Xtr).reshape(-1, 2), ytr.reshape(-1))
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        P = torch.softmax(m(torch.tensor(Xn).to(DEV)), -1)[..., 1].cpu().numpy()
    return P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ALL", choices=["ALL", "VIS"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[5100, 5101, 5102])
    ap.add_argument("--gt", default="blocks", choices=["blocks", "rawseg"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    split = json.loads((ROOT / "data/gt/HateClipSeg/p11_split.json").read_text())
    g = np.load(ROOT / "idea-stage/r11_seg/out/grid_labels.npz", allow_pickle=True)
    vids = [str(v) for v in g["video_ids"]]
    y = g["y_win"].astype(np.int64)
    bounds = g["bounds"]
    idx = {v: i for i, v in enumerate(vids)}
    pos = {s: np.array([idx[v] for v in split[s]]) for s in ("train", "val", "test")}

    gold = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    gtfn = blocks_of if args.gt == "blocks" else (
        lambda segs: [(a, b) for a, b, mh in segs if sum(mh[1:]) > 0])
    G = {s: {v: gtfn(gold[v]["segments"]) for v in split[s]} for s in ("val", "test")}

    B = build_blocks(vids)
    spec = ["V", "T", "O", "A", "M"] if args.arm == "ALL" else ["V"]
    X = np.concatenate([B[b] for b in spec], -1).astype(np.float32)
    print(f"[cfg] arm={args.arm} width={X.shape[-1]} seeds={args.seeds}", flush=True)

    res = {}
    for seed in args.seeds:
        t0 = time.time()
        P = fit(X, y, pos["train"], seed)
        best = None
        for q in G_Q:
            thr = float(np.quantile(P[pos["train"]], q))
            for w in G_W:
                for gp in G_GAP:
                    for mn in G_MIN:
                        pv = decode(P[pos["val"]], bounds[pos["val"]], split["val"],
                                    thr, w, gp, mn)
                        f1 = match_prf(pv, G["val"], 0.5)["F1"]
                        if best is None or f1 > best[0]:
                            best = (f1, q, thr, w, gp, mn)
        f1v, q, thr, w, gp, mn = best
        pt = decode(P[pos["test"]], bounds[pos["test"]], split["test"], thr, w, gp, mn)
        m = {str(t): match_prf(pt, G["test"], t) for t in (0.3, 0.5, 0.7)}
        res[str(seed)] = dict(val_f1_50=f1v, cfg=dict(q=q, thr=thr, w=w, gap=gp, minlen=mn),
                              test=m, wall_s=time.time() - t0)
        print(f"[seed {seed}] val F1@0.5={f1v:.2f} cfg(q={q},w={w},gap={gp},min={mn}) -> "
              + " ".join(f"test t{t}: F1={m[str(t)]['F1']:.2f} P={m[str(t)]['P']:.2f} "
                         f"R={m[str(t)]['R']:.2f}" for t in (0.3, 0.5, 0.7)), flush=True)
        np.savez(OUT / f"curve_scores_{args.arm}_{args.gt}_s{seed}.npz",
                 test_scores=P[pos["test"]], test_bounds=bounds[pos["test"]],
                 test_ids=np.array(split["test"]), val_scores=P[pos["val"]],
                 val_bounds=bounds[pos["val"]], val_ids=np.array(split["val"]),
                 cfg=np.array([q, thr, w, gp, mn]))

    for t in (0.3, 0.5, 0.7):
        f = [res[s]["test"][str(t)]["F1"] for s in res]
        p = [res[s]["test"][str(t)]["P"] for s in res]
        r = [res[s]["test"][str(t)]["R"] for s in res]
        print(f"[SUMMARY {args.arm}/{args.gt}] test tIoU {t}: F1 {np.mean(f):.2f}+-{np.std(f):.2f}  "
              f"P {np.mean(p):.2f}+-{np.std(p):.2f}  R {np.mean(r):.2f}+-{np.std(r):.2f}",
              flush=True)
    out = OUT / f"curve_baseline_{args.arm}_{args.gt}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"[write] {out}", flush=True)


if __name__ == "__main__":
    main()
