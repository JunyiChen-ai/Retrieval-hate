#!/usr/bin/env python
"""R17-OCRV P2: the extent-conditioned proposal re-ranking panel.

Frozen in `idea-stage/R17_OCRV_FREEZE.md` §4 (commit 1e268c6), written after it.

Runs entirely on the out-of-fold proposal pools P1 dumped for the `VAT` arm; no detector is
trained here.  Nested cross-fitting: for held-out fold f the re-ranker is fitted on the
out-of-fold proposals of the other two folds and applied to fold f.  Every arm sees the same
partition, the same proposals and the same capacity (one hidden layer, 256 units, 40 epochs,
Adam 1e-3); only the input vector differs.

Endpoint: F1@tIoU0.5 keeping exactly the top 22 proposals per video, pooled over the 237
out-of-fold videos, averaged over 3 detector seeds x 3 re-ranker seeds.

Gates (frozen):
  G1  R3 - max(R0, R2) >= +2.0 with video-clustered bootstrap LCB95 > 0
  G2  R4 - R3          >= +1.0 with LCB95 > 0
  G3  R5 - R0          <= 0.25 * (R3 - R0)     (only evaluated if G1 passes)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "scripts/r16_detbase"))
from eval_f1 import iou_1d  # noqa: E402

EMB = ROOT / "data/CLIP_Embedding/HateClipSeg"
OUT = ROOT / "idea-stage/r17_ocrv/out"
DET_SEEDS = (6200, 6201, 6202)
RR_SEEDS = (6210, 6211, 6212)
NFOLD = 3
TOPK = 22
FPS = 4.0
NBOOT = 10000
BSEED = 6299
ARMS = ("R0", "R1", "R2", "R3", "R4", "R5")


def rawseg_of(segs):
    return [(s, e) for s, e, mh in segs if sum(mh[1:]) > 0]


def pool_feat(X, s, e, T):
    a = int(np.clip(round(s * FPS), 0, T - 1))
    b = int(np.clip(round(e * FPS), a + 1, T))
    return X[a:b].mean(0)


def build(vids, pools, G, feat_cache):
    """Return per-video arrays: geom (6), span content (V/A/T), ring content, ocr, target."""
    rows = {}
    for v in vids:
        ps = pools[v]
        VAT = feat_cache[v]["vat"]
        O = feat_cache[v]["ocr"]
        T = VAT.shape[0]
        dur_v = T / FPS
        n = len(ps)
        geom = np.zeros((n, 6), np.float32)
        span = np.zeros((n, VAT.shape[1]), np.float32)
        ringL = np.zeros((n, VAT.shape[1]), np.float32)
        ringR = np.zeros((n, VAT.shape[1]), np.float32)
        ocr = np.zeros((n, O.shape[1]), np.float32)
        for i, (s, e, sc) in enumerate(ps):
            L = max(e - s, 1e-3)
            nov = sum(1 for q in ps if iou_1d((s, e), q[:2]) >= 0.5)
            geom[i] = (sc, L, (s + e) / 2 / max(dur_v, 1e-3), s / max(dur_v, 1e-3),
                       e / max(dur_v, 1e-3), nov)
            span[i] = pool_feat(VAT, s, e, T)
            ringL[i] = pool_feat(VAT, s - 0.5 * L, s, T)
            ringR[i] = pool_feat(VAT, e, e + 0.5 * L, T)
            ocr[i] = pool_feat(O, s, e, T)
        # target: matched at tIoU >= 0.5 by the greedy score-ordered matcher
        y = np.zeros(n, np.float32)
        order = np.argsort([-p[2] for p in ps])
        used = set()
        for i in order:
            best, bi = -1.0, -1
            for j, q in enumerate(G[v]):
                if j in used:
                    continue
                o = iou_1d(ps[i][:2], q)
                if o > best:
                    best, bi = o, j
            if best >= 0.5:
                y[i] = 1.0
                used.add(bi)
        rows[v] = dict(geom=geom, span=span, ringL=ringL, ringR=ringR, ocr=ocr, y=y,
                       segs=np.array([[p[0], p[1]] for p in ps], np.float32),
                       score=np.array([p[2] for p in ps], np.float32))
    return rows


def arm_input(r, arm, rng=None):
    g = r["geom"]
    if arm == "R1":
        return g[:, 1:2]
    if arm == "R2":
        return g
    content = np.concatenate([r["span"], r["ringL"], r["ringR"]], 1)
    if arm == "R5":
        # permute the content block within video, within duration decile
        L = g[:, 1]
        dec = np.digitize(L, np.quantile(L, np.linspace(0.1, 0.9, 9)))
        perm = np.arange(len(L))
        for d in np.unique(dec):
            m = np.where(dec == d)[0]
            perm[m] = rng.permutation(m)
        content = content[perm]
        return np.concatenate([g, content], 1)
    if arm == "R3":
        return np.concatenate([g, content], 1)
    if arm == "R4":
        return np.concatenate([g, content, r["ocr"]], 1)
    raise ValueError(arm)


def fit_score(Xtr, ytr, Xte, seed):
    dev = "cuda"
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Linear(256, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(ytr, device=dev)[:, None]
    lossf = nn.BCEWithLogitsLoss()
    n = len(xt)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(40):
        idx = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, 512):
            b = idx[i:i + 512]
            opt.zero_grad()
            lossf(net(xt[b]), yt[b]).backward()
            opt.step()
    with torch.no_grad():
        return net(torch.tensor(Xte, device=dev)).squeeze(1).cpu().numpy()


def counts_topk(sel, G, vids):
    out = {}
    for v in vids:
        P = sel[v]
        used, tp = set(), 0
        for s, e in P:
            best, bi = -1.0, -1
            for j, q in enumerate(G[v]):
                if j in used:
                    continue
                o = iou_1d((s, e), q)
                if o > best:
                    best, bi = o, j
            if best >= 0.5:
                tp += 1
                used.add(bi)
        out[v] = (tp, len(P), len(G[v]))
    return out


def f1_from(c, vids):
    tp = sum(c[v][0] for v in vids)
    npd = sum(c[v][1] for v in vids)
    ng = sum(c[v][2] for v in vids)
    P, R = tp / max(npd, 1), tp / max(ng, 1)
    return 100 * 2 * P * R / max(P + R, 1e-12)


def main() -> None:
    t0 = time.time()
    gold = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    split = json.loads((ROOT / "data/gt/HateClipSeg/p11_split.json").read_text())
    test_ids = set(split["test"])
    vids = sorted(split["train"])
    assert not (set(vids) & test_ids)
    print(f"[guard] P2 touches {len(vids)} train videos only; 0 val/test ids", flush=True)
    G = {v: rawseg_of(gold[v]["segments"]) for v in vids}
    folds = [[v for i, v in enumerate(vids) if i % NFOLD == f] for f in range(NFOLD)]

    feat_cache = {v: dict(vat=np.load(EMB / f"dense4fps_vat/{v}.npy").astype(np.float32),
                          ocr=np.load(EMB / f"dense4fps_ocrbert/{v}.npy").astype(np.float32))
                  for v in vids}
    print(f"[feat] loaded {len(feat_cache)} videos  t={time.time()-t0:.0f}s", flush=True)

    C = {}          # (arm, det_seed, rr_seed) -> per-video counts
    for ds in DET_SEEDS:
        pools = {}
        for f in range(NFOLD):
            for v, ps in json.loads((OUT / f"pool_VAT_s{ds}_f{f}.json").read_text()).items():
                pools[v] = [tuple(p) for p in ps]
        rows = build(vids, pools, G, feat_cache)
        print(f"[build] det seed {ds}  t={time.time()-t0:.0f}s", flush=True)

        for rs in RR_SEEDS:
            rng = np.random.default_rng(rs)
            sel = {a: {} for a in ARMS}
            for f in range(NFOLD):
                te = folds[f]
                tr = [v for g in range(NFOLD) if g != f for v in folds[g]]
                for arm in ARMS:
                    if arm == "R0":
                        for v in te:
                            o = np.argsort(-rows[v]["score"])[:TOPK]
                            sel[arm][v] = rows[v]["segs"][o].tolist()
                        continue
                    Xtr = np.concatenate([arm_input(rows[v], arm, rng) for v in tr], 0)
                    ytr = np.concatenate([rows[v]["y"] for v in tr], 0)
                    Xte_l = [arm_input(rows[v], arm, rng) for v in te]
                    s_te = fit_score(Xtr, ytr, np.concatenate(Xte_l, 0), rs)
                    p = 0
                    for v, x in zip(te, Xte_l):
                        sc = s_te[p:p + len(x)]
                        p += len(x)
                        o = np.argsort(-sc)[:TOPK]
                        sel[arm][v] = rows[v]["segs"][o].tolist()
            for arm in ARMS:
                C[(arm, ds, rs)] = counts_topk(sel[arm], G, vids)
            print(f"  [rerank] det {ds} rr {rs} " + "  ".join(
                f"{a}:{f1_from(C[(a, ds, rs)], vids):.2f}" for a in ARMS)
                + f"   t={time.time()-t0:.0f}s", flush=True)

    cells = [(ds, rs) for ds in DET_SEEDS for rs in RR_SEEDS]

    def mean_f1(arm, vv):
        return float(np.mean([f1_from(C[(arm, ds, rs)], vv) for ds, rs in cells]))

    print("\n=== P2 point estimates (top-22/video, 237 OOF videos, 9 cells) ===")
    pt = {}
    for arm in ARMS:
        xs = [f1_from(C[(arm, ds, rs)], vids) for ds, rs in cells]
        pt[arm] = float(np.mean(xs))
        print(f"  {arm}  F1@0.5 {np.mean(xs):6.2f} +- {np.std(xs):.2f}")

    rng = np.random.default_rng(BSEED)
    n = len(vids)
    boot_idx = [rng.integers(0, n, n) for _ in range(NBOOT)]

    def boot(fn):
        obs = fn(vids)
        d = np.array([fn([vids[j] for j in idx]) for idx in boot_idx])
        lo, hi = np.percentile(d, [2.5, 97.5])
        return float(obs), float(lo), float(hi)

    base = "R2" if pt["R2"] >= pt["R0"] else "R0"
    g1 = boot(lambda vv: mean_f1("R3", vv) - max(mean_f1("R0", vv), mean_f1("R2", vv)))
    g2 = boot(lambda vv: mean_f1("R4", vv) - mean_f1("R3", vv))
    g1p = "PASS" if (g1[0] >= 2.0 and g1[1] > 0) else "FAIL"
    g2p = "PASS" if (g2[0] >= 1.0 and g2[1] > 0) else "FAIL"
    print(f"\n=== frozen gates ===")
    print(f"  G1  R3 - max(R0,R2)[={base}] = {g1[0]:+.2f} [{g1[1]:+.2f}, {g1[2]:+.2f}] "
          f"(bar +2.00) -> {g1p}")
    print(f"  G2  R4 - R3               = {g2[0]:+.2f} [{g2[1]:+.2f}, {g2[2]:+.2f}] "
          f"(bar +1.00) -> {g2p}")
    g3 = None
    if g1p == "PASS":
        lhs = pt["R5"] - pt["R0"]
        rhs = 0.25 * (pt["R3"] - pt["R0"])
        g3 = "PASS" if lhs <= rhs else "FAIL (gain survives permutation -> G1 overridden)"
        print(f"  G3  R5-R0 = {lhs:+.2f}  vs  0.25*(R3-R0) = {rhs:+.2f}  -> {g3}")
    else:
        print("  G3  not evaluated (G1 failed)")

    rep = dict(point=pt, G1=dict(delta=g1[0], lo=g1[1], hi=g1[2], bar=2.0, verdict=g1p,
                                 baseline=base),
               G2=dict(delta=g2[0], lo=g2[1], hi=g2[2], bar=1.0, verdict=g2p),
               G3=g3, topk=TOPK, cells=len(cells))
    (OUT / "analysis_p2.json").write_text(json.dumps(rep, indent=1))
    print(f"\n[write] {OUT/'analysis_p2.json'}  wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
