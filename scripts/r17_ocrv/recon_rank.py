#!/usr/bin/env python
"""R17 recon (pre-freeze, descriptive): how big is the proposal RE-RANKING headroom on the
R16 ActionFormer base, priced at the system's ACTUAL operating point rather than at a fixed
small k?

Runs on the VAL split (39 videos) only.  Val was already the selection surface in R16, so no
new test contact.  Nothing here fixes a decision rule; it prices a candidate family.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "scripts/r16_detbase"))
from eval_f1 import match_prf, iou_1d  # noqa: E402

OUT = ROOT / "idea-stage/r16_detbase/out"


def blocks_of(segs):
    out, cur = [], None
    for s, e, mh in segs:
        if sum(mh[1:]) > 0:
            cur = [s, e] if cur is None else [cur[0], e]
        else:
            if cur:
                out.append(tuple(cur)); cur = None
    if cur:
        out.append(tuple(cur))
    return out


def rawseg_of(segs):
    return [(a, b) for a, b, mh in segs if sum(mh[1:]) > 0]


def oracle_rerank(pool, G, n_keep):
    """Keep n_keep[v] proposals per video, chosen by oracle max-tIoU against gold."""
    out = {}
    for v, ps in pool.items():
        Gv = G.get(v, [])
        sc = sorted(((max((iou_1d(p, q) for q in Gv), default=0.0), p) for p in ps),
                    key=lambda x: -x[0])
        k = n_keep.get(v, 0)
        out[v] = [(p[0], p[1], 1.0 - i * 1e-6) for i, (_, p) in enumerate(sc[:k])]
    return out


def oracle_binary(pool, G, tiou=0.5):
    """The true re-ranking ceiling: keep exactly the proposals that would match, greedily,
    one per gold instance, and nothing else.  Precision 100%, recall = pool recall."""
    out = {}
    for v, ps in pool.items():
        Gv = G.get(v, [])
        used, keep = set(), []
        cand = sorted(((max(((iou_1d(p, q), j) for j, q in enumerate(Gv)), default=(0.0, -1)), p)
                       for p in ps), key=lambda x: -x[0][0])
        for (o, j), p in cand:
            if o >= tiou and j not in used:
                used.add(j); keep.append((p[0], p[1], 1.0))
        out[v] = keep
    return out


def main():
    split = json.loads((ROOT / "data/gt/HateClipSeg/p11_split.json").read_text())
    gold = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    for GTC, arm in (("rawseg", "vat"), ("rawseg", "v"), ("blocks", "v")):
        gtfn = blocks_of if GTC == "blocks" else rawseg_of
        for SPLIT in ("val",):
            G = {v: gtfn(gold[v]["segments"]) for v in split[SPLIT]}
            res = json.loads((OUT / f"res_{arm}_{GTC}.json").read_text())
            seeds = sorted(res.keys())
            rows = []
            for s in seeds:
                fp = OUT / f"preds_{SPLIT}_{arm}_{GTC}_s{s}.json"
                if not fp.exists():
                    continue
                pool = {v: [tuple(p) for p in ps] for v, ps in json.loads(fp.read_text()).items()}
                pool = {v: ps for v, ps in pool.items() if v in G}
                thr = res[s]["thr"]
                model = {v: [p for p in ps if p[2] >= thr] for v, ps in pool.items()}
                n_keep = {v: len(ps) for v, ps in model.items()}
                m = match_prf(model, G, 0.5)
                orr = match_prf(oracle_rerank(pool, G, n_keep), G, 0.5)
                # oracle rerank at a per-video budget equal to the true gold count
                n_gold = {v: len(G[v]) for v in G}
                org = match_prf(oracle_rerank(pool, G, n_gold), G, 0.5)
                ob = match_prf(oracle_binary(pool, G, 0.5), G, 0.5)
                poolrec = 100.0 * sum(
                    1 for v in G for q in G[v]
                    if any(iou_1d(p, q) >= 0.5 for p in pool.get(v, []))) / max(
                    sum(len(g) for g in G.values()), 1)
                rows.append((m["F1"], m["P"], m["R"], np.mean(list(n_keep.values())),
                             orr["F1"], org["F1"], ob["F1"], ob["P"], ob["R"], poolrec,
                             np.mean([len(ps) for ps in pool.values()])))
            if not rows:
                continue
            a = np.array(rows)
            mu, sd = a.mean(0), a.std(0)
            print(f"\n=== {arm.upper()} / {GTC} / {SPLIT} (n={len(G)} videos, "
                  f"{sum(len(g) for g in G.values())} gold) ===")
            print(f"  model @thr        F1 {mu[0]:6.2f}+-{sd[0]:.2f}  P {mu[1]:5.2f} R {mu[2]:5.2f}"
                  f"  n_pred/video {mu[3]:.1f}")
            print(f"  ORACLE rerank, same budget      F1 {mu[4]:6.2f}+-{sd[4]:.2f}")
            print(f"  ORACLE rerank, budget=n_gold    F1 {mu[5]:6.2f}+-{sd[5]:.2f}")
            print(f"  ORACLE binary (perfect verifier) F1 {mu[6]:6.2f}+-{sd[6]:.2f}"
                  f"  P {mu[7]:5.2f} R {mu[8]:5.2f}")
            print(f"  pool recall@0.5 {mu[9]:.2f}   pool size/video {mu[10]:.1f}")


if __name__ == "__main__":
    main()
