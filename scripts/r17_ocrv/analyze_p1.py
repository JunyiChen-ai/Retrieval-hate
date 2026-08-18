#!/usr/bin/env python
"""R17-OCRV P1 analysis: the frozen contrasts and the video-clustered paired bootstrap.

Frozen in `idea-stage/R17_OCRV_FREEZE.md` §3:
  primary   D1 = F1@0.5(VATO) - F1@0.5(VAT),  delta = +1.5, PASS iff D1 >= 1.5 and LCB95 > 0
  secondary D2 = F1@0.5(VATO) - F1@0.5(VATO_SHUF)
  diagnostic: proposal-pool recall @0.5 per arm; F1 at 0.3 and 0.7.
Bootstrap: video-clustered paired, 10 000 resamples, seed 6299, corpus-level F1 recomputed
inside every resample, seeds pooled.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "scripts/r16_detbase"))
from eval_f1 import iou_1d  # noqa: E402

OUT = ROOT / "idea-stage/r17_ocrv/out"
ARMS = ("VAT", "VATO", "VATO_SHUF")
SEEDS = (6200, 6201, 6202)
NFOLD = 3
NBOOT = 10000
BSEED = 6299
TIOUS = (0.3, 0.5, 0.7)


def rawseg_of(segs):
    return [(s, e) for s, e, mh in segs if sum(mh[1:]) > 0]


def counts(preds, golds, vids, tiou):
    """Per-video (tp, n_pred, n_gold) under the frozen greedy score-ordered matcher."""
    out = {}
    for v in vids:
        P = sorted(preds.get(v, []), key=lambda x: -x[2])
        G = golds[v]
        used, tp = set(), 0
        for p in P:
            best, bi = -1.0, -1
            for i, q in enumerate(G):
                if i in used:
                    continue
                o = iou_1d(p, q)
                if o > best:
                    best, bi = o, i
            if best >= tiou:
                tp += 1
                used.add(bi)
        out[v] = (tp, len(P), len(G))
    return out


def f1_from(c, vids):
    tp = sum(c[v][0] for v in vids)
    npd = sum(c[v][1] for v in vids)
    ng = sum(c[v][2] for v in vids)
    P = tp / max(npd, 1)
    R = tp / max(ng, 1)
    return 100 * 2 * P * R / max(P + R, 1e-12)


def main() -> None:
    gold = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    split = json.loads((ROOT / "data/gt/HateClipSeg/p11_split.json").read_text())
    vids = sorted(split["train"])
    G = {v: rawseg_of(gold[v]["segments"]) for v in vids}
    res = json.loads((OUT / "res_p1.json").read_text())

    print("=== P1 point estimates (237 out-of-fold train videos, 3 seeds) ===")
    pt = {}
    for arm in ARMS:
        for t in TIOUS:
            xs = [res[f"{arm}|{s}"]["oof"][str(t)]["F1"] for s in SEEDS]
            pt[(arm, t)] = float(np.mean(xs))
            if t == 0.5:
                p = np.mean([res[f"{arm}|{s}"]["oof"]["0.5"]["P"] for s in SEEDS])
                r = np.mean([res[f"{arm}|{s}"]["oof"]["0.5"]["R"] for s in SEEDS])
                npd = np.mean([res[f"{arm}|{s}"]["n_pred"] for s in SEEDS])
                print(f"  {arm:10s} tIoU 0.5: F1 {np.mean(xs):6.2f} +- {np.std(xs):.2f}  "
                      f"P {p:5.2f} R {r:5.2f}  n_pred {npd:.0f} "
                      f"({', '.join(f'{x:.2f}' for x in xs)})")
    for arm in ARMS:
        print(f"  {arm:10s} tIoU 0.3 {pt[(arm,0.3)]:6.2f}   tIoU 0.7 {pt[(arm,0.7)]:6.2f}")

    # ---- proposal-pool recall (whole 200-proposal pool, before thresholding)
    print("\n=== pool recall @tIoU 0.5 (whole 200/video pool, out-of-fold) ===")
    poolrec = {}
    for arm in ARMS:
        rs = []
        for s in SEEDS:
            pool = {}
            for f in range(NFOLD):
                pool.update({v: [tuple(p) for p in ps] for v, ps in
                             json.loads((OUT / f"pool_{arm}_s{s}_f{f}.json").read_text()).items()})
            hit = tot = 0
            for v in vids:
                for q in G[v]:
                    tot += 1
                    if any(iou_1d(p, q) >= 0.5 for p in pool.get(v, [])):
                        hit += 1
            rs.append(100.0 * hit / tot)
        poolrec[arm] = (float(np.mean(rs)), float(np.std(rs)))
        print(f"  {arm:10s} {np.mean(rs):.2f} +- {np.std(rs):.2f}")

    # ---- video-clustered paired bootstrap on the pooled-seed counts
    rng = np.random.default_rng(BSEED)
    C = {}
    for arm in ARMS:
        for s in SEEDS:
            preds = json.loads((OUT / f"preds_oof_{arm}_s{s}.json").read_text())
            preds = {v: [tuple(p) for p in ps] for v, ps in preds.items()}
            C[(arm, s)] = counts(preds, G, vids, 0.5)

    def boot(a, b):
        obs = np.mean([f1_from(C[(a, s)], vids) - f1_from(C[(b, s)], vids) for s in SEEDS])
        d = np.empty(NBOOT)
        n = len(vids)
        for i in range(NBOOT):
            idx = rng.integers(0, n, n)
            vv = [vids[j] for j in idx]
            d[i] = np.mean([f1_from(C[(a, s)], vv) - f1_from(C[(b, s)], vv) for s in SEEDS])
        lo, hi = np.percentile(d, [2.5, 97.5])
        return float(obs), float(lo), float(hi)

    print("\n=== frozen contrasts (video-clustered paired bootstrap, 10 000 resamples) ===")
    d1, l1, h1 = boot("VATO", "VAT")
    d2, l2, h2 = boot("VATO", "VATO_SHUF")
    d3, l3, h3 = boot("VATO_SHUF", "VAT")
    print(f"  D1  VATO - VAT        = {d1:+.2f}  [{l1:+.2f}, {h1:+.2f}]   "
          f"(delta = +1.50)  -> {'PASS' if (d1 >= 1.5 and l1 > 0) else 'KILL'}")
    print(f"  D2  VATO - VATO_SHUF  = {d2:+.2f}  [{l2:+.2f}, {h2:+.2f}]")
    print(f"  D3  VATO_SHUF - VAT   = {d3:+.2f}  [{l3:+.2f}, {h3:+.2f}]   (descriptive)")

    rep = dict(point={f"{a}|{t}": pt[(a, t)] for a in ARMS for t in TIOUS},
               pool_recall_50=poolrec,
               D1=dict(delta=d1, lo=l1, hi=h1, bar=1.5,
                       verdict="PASS" if (d1 >= 1.5 and l1 > 0) else "KILL"),
               D2=dict(delta=d2, lo=l2, hi=h2),
               D3=dict(delta=d3, lo=l3, hi=h3))
    (OUT / "analysis_p1.json").write_text(json.dumps(rep, indent=1))
    print(f"\n[write] {OUT/'analysis_p1.json'}")


if __name__ == "__main__":
    main()
