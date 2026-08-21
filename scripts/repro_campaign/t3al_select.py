#!/usr/bin/env python
"""REPRO campaign Wave 2 — T3AL preset selection on the val split.

The rule is frozen in idea-stage/repro_t3al/RUN_RECORD.md before the sweep runs:

    choose the preset with the highest **mean over the four datasets** of the
    val-split pooled frame PR-AUC of variant `main`; a dataset a preset produced
    no row for counts as its dataset's positive base rate (the random floor);
    ties broken by the preset's order in run_t3al.PRESETS.

No test-split number is read here.  Writes idea-stage/repro_t3al/preset_chosen.json.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/jehc223/Retrieval-hate")
OUT = ROOT / "idea-stage/repro_t3al"
PRESETS = ["A_thumos", "B_thumos_rescaled", "C_thumos_15steps", "D_anet"]
DATASETS = ["HateMM", "MHC", "MHC_zh", "HateClipSeg"]


def main() -> int:
    table, means = {}, {}
    for p in PRESETS:
        f = OUT / f"eval/val_{p}.json"
        if not f.exists():
            continue
        rows = {r["dataset"]: r for r in json.loads(f.read_text())
                if r.get("variant") == "main"}
        per = {}
        for ds in DATASETS:
            r = rows.get(ds, {})
            pool = r.get("pooled", {})
            ap = pool.get("frame_PR_AUC")
            if ap is None:
                ap = pool.get("base_rate", 0.0)
            per[ds] = float(ap)
        table[p] = per
        means[p] = sum(per.values()) / len(DATASETS)
    if not means:
        raise SystemExit("no val sweep results found")
    best = max(PRESETS, key=lambda p: (means.get(p, -1), -PRESETS.index(p)))
    rec = dict(rule="max mean val frame_PR_AUC of variant `main` over the four datasets",
               per_preset=table, mean=means, preset=best)
    (OUT / "preset_chosen.json").write_text(json.dumps(rec, indent=1))
    for p in PRESETS:
        print(f"{p:<20} mean={means.get(p)} {table.get(p)}")
    print(f"[chosen] {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
