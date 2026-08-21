#!/usr/bin/env python
"""REPRO campaign Wave 2 — T3AL, mean +- sd over the three frozen seeds.

Reads idea-stage/repro_t3al/eval/test_s<SEED>.json (one shared-evaluator result
file per seed, freeze §6) and writes a markdown table plus a merged JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
OUT = ROOT / "idea-stage/repro_t3al"
SEEDS = [20250819, 20250820, 20250821]
POOLED = ["frame_ROC_AUC", "frame_PR_AUC", "AP_norm", "base_rate", "coverage",
          "n_frames", "n_videos", "AP_norm_denom", "frame_PR_AUC_trapz"]
IVKEYS = ["F1@0.3", "F1@0.5", "F1@0.7", "P@0.5", "R@0.5", "n_pred", "n_gold"]
STRATA = ["strat_single_span", "strat_multi_span"]


def fmt(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return "n/a"
    if len(v) == 1:
        return f"{v[0]:.4f}" if isinstance(v[0], float) else str(v[0])
    m, s = float(np.mean(v)), float(np.std(v, ddof=1))
    return f"{m:.4f} ± {s:.4f}"


def main() -> int:
    per_seed = {}
    for s in SEEDS:
        f = OUT / f"eval/test_s{s}.json"
        if f.exists():
            per_seed[s] = json.loads(f.read_text())
    if not per_seed:
        raise SystemExit("no per-seed evaluator output found")
    keys = sorted({(r["dataset"], r["variant"]) for rs in per_seed.values() for r in rs})
    merged, lines = [], []
    lines.append("| dataset | variant | seeds | frame_ROC_AUC | frame_PR_AUC | "
                 "F1@0.3 | F1@0.5 | F1@0.7 | AP_norm | base_rate | n_frames | "
                 "coverage | n_videos | missing |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ds, var in keys:
        rows = {s: next((r for r in rs if r["dataset"] == ds and r["variant"] == var), None)
                for s, rs in per_seed.items()}
        rows = {s: r for s, r in rows.items() if r}
        pool = {k: fmt([r["pooled"].get(k) for r in rows.values()]) for k in POOLED}
        iv = {k: fmt([r.get("intervals", {}).get(k) for r in rows.values()]) for k in IVKEYS}
        miss = fmt([r["n_videos_missing"] for r in rows.values()])
        lines.append(f"| {ds} | {var} | {len(rows)} | {pool['frame_ROC_AUC']} | "
                     f"{pool['frame_PR_AUC']} | {iv['F1@0.3']} | {iv['F1@0.5']} | "
                     f"{iv['F1@0.7']} | {pool['AP_norm']} | {pool['base_rate']} | "
                     f"{pool['n_frames']} | {pool['coverage']} | {pool['n_videos']} | {miss} |")
        entry = dict(dataset=ds, variant=var, n_seeds=len(rows), pooled=pool, intervals=iv,
                     n_videos_missing=miss)
        for st in STRATA:
            if any(st in r for r in rows.values()):
                entry[st] = {k: fmt([r.get(st, {}).get(k) for r in rows.values()])
                             for k in POOLED}
                entry[st + "_intervals"] = {
                    k: fmt([r.get(st + "_intervals", {}).get(k) for r in rows.values()])
                    for k in IVKEYS}
        merged.append(entry)
    (OUT / "eval/test_agg.json").write_text(json.dumps(merged, indent=1))
    (OUT / "eval/test_agg.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[written] {OUT/'eval/test_agg.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
