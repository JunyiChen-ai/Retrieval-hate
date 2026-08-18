#!/usr/bin/env python
"""R18-FEATBASE analysis: the frozen decision rule, plus the declared descriptive diagnostics.

Decision rule (R18_FEATBASE_FREEZE.md §5): for X in {mae_vat, maeclip_vat}, the seed-paired
difference d_s = F1@0.5(X, s) - F1@0.5(clip_vat, s) over seeds 7300/7301/7302; report mean and
the Student-t 95% CI (t_{0.975,2} = 4.303).  Bar: mean >= +2.0 AND CI lower bound > 0.

Descriptive (§6): proposal-pool recall at tIoU 0.3/0.5/0.7 per arm, pool size, and prediction
geometry, all on the 119-video test split under the rawseg convention.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "scripts/r16_detbase"))
from eval_f1 import iou_1d                                     # noqa: E402

AF = ROOT / "third_party/actionformer"
OUT = ROOT / "idea-stage/r18_featbase/out"
SEEDS = ["7300", "7301", "7302"]
TIOUS = (0.3, 0.5, 0.7)
T_CRIT_2 = 4.302652729911275                                   # t_{0.975, df=2}
ARMS = ["clip_vat", "mae_vat", "maeclip_vat", "clip_v", "mae_v"]


def gold_test():
    db = json.loads((AF / "data/hateclipseg/hateclipseg_rawseg.json").read_text())["database"]
    return {v: [tuple(a["segment"]) for a in d["annotations"]]
            for v, d in db.items() if d["subset"] == "test"}


def pool_recall(preds, golds, tiou):
    hit = tot = 0
    for v, G in golds.items():
        P = preds.get(v, [])
        for q in G:
            tot += 1
            if any(iou_1d((p[0], p[1]), q) >= tiou for p in P):
                hit += 1
    return 100.0 * hit / max(tot, 1)


def main():
    res = {a: json.loads((OUT / f"res_{a}.json").read_text()) for a in ARMS
           if (OUT / f"res_{a}.json").exists()}
    golds = gold_test()
    n_gold = sum(len(v) for v in golds.values())
    report = {"n_test_videos": len(golds), "n_gold_instances": n_gold, "arms": {}, "contrasts": {}}

    print(f"gold: {len(golds)} test videos, {n_gold} rawseg instances\n")
    print(f"{'arm':<14} {'F1@0.3':>14} {'F1@0.5':>14} {'F1@0.7':>14} {'P@0.5':>7} {'R@0.5':>7}")
    for a in ARMS:
        if a not in res:
            continue
        row = {}
        for t in TIOUS:
            xs = [res[a][s]["test"][str(t)]["F1"] for s in SEEDS]
            row[str(t)] = dict(F1=float(np.mean(xs)), sd=float(np.std(xs)),
                               per_seed=[float(x) for x in xs],
                               P=float(np.mean([res[a][s]["test"][str(t)]["P"] for s in SEEDS])),
                               R=float(np.mean([res[a][s]["test"][str(t)]["R"] for s in SEEDS])))
        row["val_f1_50"] = float(np.mean([res[a][s]["val"]["0.5"]["F1"] for s in SEEDS]))
        row["best_epoch"] = [res[a][s]["best_epoch"] for s in SEEDS]
        row["thr"] = [round(res[a][s]["thr"], 4) for s in SEEDS]
        report["arms"][a] = row
        print(f"{a:<14} " + " ".join(
            f"{row[str(t)]['F1']:8.2f}±{row[str(t)]['sd']:.2f}" for t in TIOUS) +
            f" {row['0.5']['P']:7.2f} {row['0.5']['R']:7.2f}")

    # ---- frozen decision rule ----
    print("\n=== frozen decision rule: seed-paired d = X - clip_vat, test F1@tIoU 0.5 ===")
    for base, cands in (("clip_vat", ["mae_vat", "maeclip_vat"]),
                        ("clip_v", ["mae_v"])):
        if base not in res:
            continue
        b = np.array([res[base][s]["test"]["0.5"]["F1"] for s in SEEDS])
        for c in cands:
            if c not in res:
                continue
            x = np.array([res[c][s]["test"]["0.5"]["F1"] for s in SEEDS])
            d = x - b
            m, sd = float(d.mean()), float(d.std(ddof=1))
            half = T_CRIT_2 * sd / np.sqrt(len(d))
            passed = bool(m >= 2.0 and (m - half) > 0)
            report["contrasts"][f"{c}-{base}"] = dict(
                per_seed=[float(v) for v in d], mean=m, sd=sd,
                ci95=[m - half, m + half], primary=(base == "clip_vat"),
                passes_bar=passed)
            tag = "PRIMARY" if base == "clip_vat" else "secondary(descriptive)"
            print(f"{c:>12} - {base:<10} d = {m:+6.2f} [{m-half:+6.2f}, {m+half:+6.2f}]  "
                  f"per-seed {[round(float(v),2) for v in d]}  "
                  f"{'PASS' if passed else 'fail'}  [{tag}]")

    # ---- descriptive: proposal pool ----
    print("\n=== proposal pool (test, rawseg, seed 7300) ===")
    print(f"{'arm':<14} {'recall@.3':>10} {'recall@.5':>10} {'recall@.7':>10} {'pool':>8} "
          f"{'kept/vid':>9} {'med len':>8}")
    for a in ARMS:
        f = OUT / f"preds_test_{a}_s7300.json"
        if not f.exists():
            continue
        preds = json.loads(f.read_text())
        rec = {str(t): pool_recall(preds, golds, t) for t in TIOUS}
        pool = sum(len(v) for v in preds.values())
        thr = res[a]["7300"]["thr"]
        kept = {v: [p for p in ps if p[2] >= thr] for v, ps in preds.items()}
        nk = sum(len(v) for v in kept.values())
        lens = np.array([p[1] - p[0] for v in kept.values() for p in v])
        geo = dict(pool=pool, kept=nk, kept_per_video=nk / len(preds),
                   median_kept_len=float(np.median(lens)) if lens.size else None)
        report["arms"].setdefault(a, {})["pool_recall"] = rec
        report["arms"][a]["geometry"] = geo
        print(f"{a:<14} {rec['0.3']:10.2f} {rec['0.5']:10.2f} {rec['0.7']:10.2f} {pool:8d} "
              f"{geo['kept_per_video']:9.2f} {geo['median_kept_len'] or 0:8.2f}")

    gl = np.array([e - s for v in golds.values() for s, e in v])
    report["gold_geometry"] = dict(per_video=n_gold / len(golds),
                                   median_len=float(np.median(gl)))
    print(f"{'gold':<14} {'':>10} {'':>10} {'':>10} {'':>8} "
          f"{n_gold/len(golds):9.2f} {np.median(gl):8.2f}")

    (OUT / "analysis.json").write_text(json.dumps(report, indent=1))
    print(f"\nwritten {OUT/'analysis.json'}")


if __name__ == "__main__":
    sys.exit(main())
