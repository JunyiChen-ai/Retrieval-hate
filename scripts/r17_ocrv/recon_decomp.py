#!/usr/bin/env python
"""R17 recon 2 (pre-freeze, descriptive): decompose the proposal-ranking gap on val.

For the VAT/rawseg detector at its own threshold, classify each KEPT proposal by its best
tIoU with an unmatched gold segment, and each MISSED gold by whether the pool contained a
>=0.5 proposal that the model ranked out.  Also: does the model's score correlate with tIoU
within a video at all?
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "scripts/r16_detbase"))
from eval_f1 import match_prf, iou_1d  # noqa: E402
OUT = ROOT / "idea-stage/r16_detbase/out"


def rawseg_of(segs):
    return [(a, b) for a, b, mh in segs if sum(mh[1:]) > 0]


def main():
    split = json.loads((ROOT / "data/gt/HateClipSeg/p11_split.json").read_text())
    gold = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    G = {v: rawseg_of(gold[v]["segments"]) for v in split["val"]}
    res = json.loads((OUT / "res_vat_rawseg.json").read_text())
    agg = {}
    rhos, rhos_len = [], []
    for s in sorted(res.keys()):
        fp = OUT / f"preds_val_vat_rawseg_s{s}.json"
        pool = {v: [tuple(p) for p in ps] for v, ps in json.loads(fp.read_text()).items() if v in G}
        thr = res[s]["thr"]
        kept = {v: [p for p in ps if p[2] >= thr] for v, ps in pool.items()}
        c = dict(kept_tp=0, kept_partial=0, kept_zero=0, n_kept=0,
                 miss_inpool=0, miss_notinpool=0, n_gold=0)
        lens_fp0, lens_tp = [], []
        for v, Gv in G.items():
            P = sorted(kept.get(v, []), key=lambda x: -x[2])
            used = set()
            for p in P:
                c["n_kept"] += 1
                best, bi = 0.0, -1
                for i, q in enumerate(Gv):
                    if i in used:
                        continue
                    o = iou_1d(p, q)
                    if o > best:
                        best, bi = o, i
                if best >= 0.5:
                    c["kept_tp"] += 1; used.add(bi); lens_tp.append(p[1] - p[0])
                elif best > 0.0:
                    c["kept_partial"] += 1
                else:
                    c["kept_zero"] += 1; lens_fp0.append(p[1] - p[0])
            c["n_gold"] += len(Gv)
            for i, q in enumerate(Gv):
                if i in used:
                    continue
                if any(iou_1d(p, q) >= 0.5 for p in pool.get(v, [])):
                    c["miss_inpool"] += 1
                else:
                    c["miss_notinpool"] += 1
            # within-video correlation between model score and oracle tIoU, over the FULL pool
            ps = pool.get(v, [])
            if len(ps) > 5 and Gv:
                t = [max((iou_1d(p, q) for q in Gv), default=0.0) for p in ps]
                sc = [p[2] for p in ps]
                ln = [p[1] - p[0] for p in ps]
                if np.std(t) > 0:
                    rhos.append(spearmanr(sc, t).correlation)
                    rhos_len.append(spearmanr(ln, t).correlation)
        for k, val in c.items():
            agg.setdefault(k, []).append(val)
        agg.setdefault("len_tp", []).append(float(np.mean(lens_tp)) if lens_tp else 0.0)
        agg.setdefault("len_fp0", []).append(float(np.mean(lens_fp0)) if lens_fp0 else 0.0)
    m = {k: float(np.mean(v)) for k, v in agg.items()}
    print("=== VAT/rawseg val, kept proposals at the model threshold (mean over 3 seeds) ===")
    print(f"  kept total          {m['n_kept']:.0f}")
    print(f"    matched (tIoU>=.5) {m['kept_tp']:.0f}   ({100*m['kept_tp']/m['n_kept']:.1f}%)")
    print(f"    partial (0<tIoU<.5){m['kept_partial']:.0f}   ({100*m['kept_partial']/m['n_kept']:.1f}%)")
    print(f"    zero overlap       {m['kept_zero']:.0f}   ({100*m['kept_zero']/m['n_kept']:.1f}%)")
    print(f"  gold total          {m['n_gold']:.0f}")
    print(f"    missed, but a >=.5 proposal WAS in the pool  {m['miss_inpool']:.0f} "
          f"({100*m['miss_inpool']/m['n_gold']:.1f}% of gold)")
    print(f"    missed, not in pool at all                   {m['miss_notinpool']:.0f}")
    print(f"  mean length: matched {m['len_tp']:.1f}s   zero-overlap FP {m['len_fp0']:.1f}s")
    print(f"\n  within-video Spearman(model score, oracle tIoU) over full 200-pool: "
          f"{np.mean(rhos):.3f} (n={len(rhos)} video-seed cells)")
    print(f"  within-video Spearman(proposal length, oracle tIoU):                 "
          f"{np.mean(rhos_len):.3f}")


if __name__ == "__main__":
    main()
