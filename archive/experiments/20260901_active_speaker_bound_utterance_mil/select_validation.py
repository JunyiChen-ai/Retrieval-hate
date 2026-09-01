#!/usr/bin/env python
"""Select relation weight and checkpoint strictly on validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pooled-tolerance", type=float, default=.005)
    args = ap.parse_args()
    root = Path(args.corpus_dir)
    rows = []
    for path in sorted(root.glob("*/train_meta.json")):
        meta = json.loads(path.read_text())
        cfg = meta["config"]
        rows.append({"trial": path.parent.name,
                     "path": str(path.parent.resolve()),
                     "arm": meta["arm"], "lr": float(cfg["lr"]),
                     "relation_weight": float(cfg["relation_weight"]),
                     "selected_epoch": int(meta["selected_epoch"]),
                     "validation": meta["selected_validation"]})
    anchors = {row["lr"]: row for row in rows if row["arm"] == "anchor"}
    core = [row for row in rows if row["arm"] == "core"]
    if len(anchors) != 2 or len(core) != 6:
        raise RuntimeError(
            f"incomplete search: anchors={len(anchors)} core={len(core)}")
    for row in core:
        base = anchors[row["lr"]]["validation"]
        val = row["validation"]
        row["delta_vs_anchor"] = {
            key: float(val[key] - base[key])
            for key in ("pooled_ap", "pooled_roc", "within_roc")}
        row["pooled_feasible"] = bool(
            row["delta_vs_anchor"]["pooled_ap"] >= -args.pooled_tolerance and
            row["delta_vs_anchor"]["pooled_roc"] >= -args.pooled_tolerance)
    feasible = [row for row in core if row["pooled_feasible"]]
    if feasible:
        selected = max(feasible, key=lambda row: (
            row["validation"]["within_roc"], row["validation"]["pooled_ap"],
            row["validation"]["pooled_roc"], -row["relation_weight"]))
        mode = "pooled-feasible then within"
    else:
        selected = max(core, key=lambda row: (
            min(row["delta_vs_anchor"]["pooled_ap"],
                row["delta_vs_anchor"]["pooled_roc"]),
            row["validation"]["within_roc"], row["validation"]["pooled_ap"],
            row["validation"]["pooled_roc"]))
        mode = "fallback max-min pooled delta then within"
    payload = {"selection_split": "validation", "selection_mode": mode,
               "pooled_tolerance": args.pooled_tolerance,
               "selected": selected, "matched_anchor": anchors[selected["lr"]],
               "trials": rows, "test_predictions_seen": False}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"selected": selected, "mode": mode}, indent=2))


if __name__ == "__main__":
    main()
