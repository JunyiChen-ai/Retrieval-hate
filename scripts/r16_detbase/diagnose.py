#!/usr/bin/env python
"""R16-DETBASE post-hoc diagnostics: what is the ActionFormer-vs-score-curve gap made of?

All of this is DESCRIPTIVE and runs after the frozen primary number exists.  Nothing here
feeds a decision; it decomposes one.

Pieces, all on the 119-video test split, primary GT convention (merged offensive blocks):

  P0  the two systems' operating points, side by side.
  P1  PROPOSAL POOL RECALL — the fraction of gold blocks that *some* proposal in the pool
      localizes at tIoU t, with precision ignored.  This is the ceiling each system's candidate
      generator imposes, before any scoring or thresholding.
  P2  GRID SNAP — ActionFormer's proposals with both boundaries snapped to the nearest edge of
      the same 30-window grid the score curve is forced to use.  Isolates how much of the
      detector's advantage is sub-window boundary precision.
  P3  BOUNDARY ORACLE — the score curve's decoded intervals, each snapped to its best-overlap
      gold block (only where overlap > 0).  The ceiling of pure boundary repair on our substrate:
      what the curve would score if every interval it emits had perfect boundaries.
  P4  COUNT / LENGTH statistics of the two prediction sets against gold.
  P5  SCORING — hold the pool fixed and trace F1 as a function of how many proposals per video
      are kept, for both systems.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "scripts/r16_detbase"))
from eval_f1 import match_prf, iou_1d                       # noqa: E402

OUT = ROOT / "idea-stage/r16_detbase/out"
TIOUS = (0.3, 0.5, 0.7)


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


def pool_recall(preds, golds, tiou):
    hit = tot = 0
    for v, G in golds.items():
        P = preds.get(v, [])
        for q in G:
            tot += 1
            if any(iou_1d(p, q) >= tiou for p in P):
                hit += 1
    return 100.0 * hit / max(tot, 1)


def snap_to_grid(preds, bounds_by_vid):
    out = {}
    for v, ps in preds.items():
        edges = np.unique(np.concatenate([bounds_by_vid[v][:, 0], bounds_by_vid[v][:, 1]]))
        res = []
        for s, e, sc in ps:
            s2 = float(edges[np.argmin(np.abs(edges - s))])
            e2 = float(edges[np.argmin(np.abs(edges - e))])
            if e2 <= s2:
                j = int(np.argmin(np.abs(edges - e)))
                e2 = float(edges[min(j + 1, len(edges) - 1)])
            if e2 > s2:
                res.append((s2, e2, sc))
        out[v] = res
    return out


def oracle_snap(preds, golds):
    out = {}
    for v, ps in preds.items():
        G = golds.get(v, [])
        res = []
        for s, e, sc in ps:
            best, bq = 0.0, None
            for q in G:
                o = iou_1d((s, e), q)
                if o > best:
                    best, bq = o, q
            res.append((bq[0], bq[1], sc) if bq is not None else (s, e, sc))
        out[v] = res
    return out


def topk(preds, k):
    return {v: sorted(ps, key=lambda x: -x[2])[:k] for v, ps in preds.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="blocks", choices=["blocks", "rawseg"])
    args = ap.parse_args()
    GTC = args.gt
    gtfn = blocks_of if GTC == "blocks" else (
        lambda segs: [(a, b) for a, b, mh in segs if sum(mh[1:]) > 0])
    split = json.loads((ROOT / "data/gt/HateClipSeg/p11_split.json").read_text())
    gold = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    G = {v: gtfn(gold[v]["segments"]) for v in split["test"]}

    res_af = json.loads((OUT / f"res_v_{GTC}.json").read_text())
    seeds = sorted(res_af.keys())
    rep = {}

    # ---- score-curve side: rebuild its decoded test intervals per seed
    from curve_baseline import decode                                    # noqa: E402
    curve_preds, curve_pool = {}, {}
    bounds_by_vid = None
    for s in seeds:
        z = np.load(OUT / f"curve_scores_ALL_{GTC}_s{s}.npz", allow_pickle=True)
        ids = [str(x) for x in z["test_ids"]]
        q, thr, w, gp, mn = z["cfg"].tolist()
        curve_preds[s] = decode(z["test_scores"], z["test_bounds"], ids, thr, int(w), gp, mn)
        # candidate POOL: every interval the decoder could ever emit, over the whole
        # threshold / width / gap / min-length grid it is allowed to choose from
        pool = {v: set() for v in ids}
        for qq in (0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
            th = float(np.quantile(z["test_scores"], qq))
            for ww in (1, 3, 5):
                for gg in (0.0, 5.0, 10.0):
                    d = decode(z["test_scores"], z["test_bounds"], ids, th, ww, gg, 0.0)
                    for v, ps in d.items():
                        for a, b, _ in ps:
                            pool[v].add((round(a, 3), round(b, 3)))
        curve_pool[s] = {v: [(a, b, 1.0) for a, b in sorted(p)] for v, p in pool.items()}
        if bounds_by_vid is None:
            bounds_by_vid = {v: z["test_bounds"][i] for i, v in enumerate(ids)}

    af_preds, af_pool = {}, {}
    for s in seeds:
        raw = json.loads((OUT / f"preds_test_v_{GTC}_s{s}.json").read_text())
        raw = {v: [tuple(p) for p in ps] for v, ps in raw.items()}
        af_pool[s] = raw
        thr = res_af[s]["thr"]
        af_preds[s] = {v: [p for p in ps if p[2] >= thr] for v, ps in raw.items()}

    def agg(fn):
        vals = [fn(s) for s in seeds]
        return float(np.mean(vals)), float(np.std(vals))

    print(f"=== P0 operating points (test, {GTC} GT) ===")
    for name, pr in (("score-curve ALL", curve_preds), ("ActionFormer V", af_preds)):
        row = []
        for t in TIOUS:
            f, fs = agg(lambda s, t=t, pr=pr: match_prf(pr[s], G, t)["F1"])
            p, _ = agg(lambda s, t=t, pr=pr: match_prf(pr[s], G, t)["P"])
            r, _ = agg(lambda s, t=t, pr=pr: match_prf(pr[s], G, t)["R"])
            row.append(f"t{t}: F1 {f:.2f}+-{fs:.2f} P {p:.2f} R {r:.2f}")
        n, _ = agg(lambda s, pr=pr: sum(len(x) for x in pr[s].values()))
        print(f"  {name:16s} " + " | ".join(row) + f"  [n_pred {n:.0f}, n_gold "
              f"{sum(len(g) for g in G.values())}]")
        rep[name] = {str(t): dict(zip(("F1", "P", "R"),
                                      [agg(lambda s, t=t, pr=pr, k=k: match_prf(pr[s], G, t)[k])[0]
                                       for k in ("F1", "P", "R")])) for t in TIOUS}

    print("\n=== P1 proposal-pool recall (precision ignored) ===")
    for name, pool in (("score-curve pool", curve_pool), ("ActionFormer pool", af_pool)):
        row = []
        for t in TIOUS:
            r, rs = agg(lambda s, t=t, pool=pool: pool_recall(pool[s], G, t))
            row.append(f"t{t}: {r:.2f}+-{rs:.2f}")
        n, _ = agg(lambda s, pool=pool: sum(len(x) for x in pool[s].values()))
        print(f"  {name:18s} " + " | ".join(row) + f"  [pool size {n:.0f}]")
        rep[name] = {str(t): agg(lambda s, t=t, pool=pool: pool_recall(pool[s], G, t))[0]
                     for t in TIOUS}

    print("\n=== P2 ActionFormer proposals snapped to the 30-window grid ===")
    snapped = {s: snap_to_grid(af_preds[s], bounds_by_vid) for s in seeds}
    row = []
    for t in TIOUS:
        f, fs = agg(lambda s, t=t: match_prf(snapped[s], G, t)["F1"])
        row.append(f"t{t}: F1 {f:.2f}+-{fs:.2f}")
    print("  " + " | ".join(row))
    rep["af_gridsnap"] = {str(t): agg(lambda s, t=t: match_prf(snapped[s], G, t)["F1"])[0]
                          for t in TIOUS}

    print("\n=== P3 score-curve intervals with ORACLE boundaries ===")
    osnap = {s: oracle_snap(curve_preds[s], G) for s in seeds}
    row = []
    for t in TIOUS:
        f, fs = agg(lambda s, t=t: match_prf(osnap[s], G, t)["F1"])
        row.append(f"t{t}: F1 {f:.2f}+-{fs:.2f}")
    print("  " + " | ".join(row))
    rep["curve_oracle_boundary"] = {
        str(t): agg(lambda s, t=t: match_prf(osnap[s], G, t)["F1"])[0] for t in TIOUS}

    print("\n=== P4 prediction geometry ===")
    gl = [e - s for g in G.values() for s, e in g]
    print(f"  gold      : {sum(len(g) for g in G.values())} inst, "
          f"{np.mean([len(g) for g in G.values()]):.2f}/video, len mean {np.mean(gl):.1f}s "
          f"med {np.median(gl):.1f}s")
    for name, pr in (("score-curve", curve_preds), ("ActionFormer", af_preds)):
        ln = [p[1] - p[0] for sd in seeds for ps in pr[sd].values() for p in ps]
        cnt = [len(ps) for sd in seeds for ps in pr[sd].values()]
        if not ln:
            ln = [0.0]
        print(f"  {name:11s}: {np.mean(cnt):.2f}/video, len mean {np.mean(ln):.1f}s "
              f"med {np.median(ln):.1f}s")

    print("\n=== P5 F1@0.5 vs proposals kept per video ===")
    for name, pool in (("score-curve pool", curve_pool), ("ActionFormer pool", af_pool)):
        row = []
        for k in (1, 2, 3, 5, 10, 20):
            f, _ = agg(lambda s, k=k, pool=pool: match_prf(topk(pool[s], k), G, 0.5)["F1"])
            row.append(f"k={k}: {f:.1f}")
        print(f"  {name:18s} " + "  ".join(row))
        rep[f"{name} topk F1@0.5"] = {
            str(k): agg(lambda s, k=k, pool=pool: match_prf(topk(pool[s], k), G, 0.5)["F1"])[0]
            for k in (1, 2, 3, 5, 10, 20)}

    print("\n=== P6 same pool, model ranking vs ORACLE ranking (top-k/video, F1@0.5) ===")
    for name, pool in (("score-curve pool", curve_pool), ("ActionFormer pool", af_pool)):
        rowm, rowo = [], []
        for k in (1, 2, 3, 5):
            m, _ = agg(lambda s, k=k, pool=pool: match_prf(topk(pool[s], k), G, 0.5)["F1"])

            def orank(s, k=k, pool=pool):
                ok = {}
                for v, ps in pool[s].items():
                    Gv = G.get(v, [])
                    sc = sorted(((max((iou_1d(p, q) for q in Gv), default=0.0), p)
                                 for p in ps), key=lambda x: -x[0])
                    ok[v] = [(p[0], p[1], 1.0 - i * 1e-6) for i, (_, p) in enumerate(sc[:k])]
                return match_prf(ok, G, 0.5)["F1"]
            o, _ = agg(orank)
            rowm.append(f"k={k}: {m:.1f}")
            rowo.append(f"k={k}: {o:.1f}")
            rep.setdefault(f"{name} oracle-rank F1@0.5", {})[str(k)] = o
        print(f"  {name:18s} model  " + "  ".join(rowm))
        print(f"  {'':18s} oracle " + "  ".join(rowo))

    (OUT / f"diagnostics_{GTC}.json").write_text(json.dumps(rep, indent=1))
    print(f"\n[write] {OUT/f'diagnostics_{GTC}.json'}")


if __name__ == "__main__":
    main()
