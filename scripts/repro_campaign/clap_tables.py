#!/usr/bin/env python
"""REPRO campaign -- CLAP: aggregate the evaluator's per-seed rows into the
freeze §14 table shape, with mean ± sd across the three seeds where the run is
stochastic.  Prints markdown; never edits the results file.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path("/home/jehc223/Retrieval-hate")
DS_LABEL = {"HateMM": "HateMM", "MHC": "MHC-EN", "MHC_zh": "MHC-ZH",
            "HateClipSeg": "HateClipSeg"}
DS_ORDER = ["HateMM", "MHC", "MHC_zh", "HateClipSeg"]
HEADER = ("| method | wave | dataset | split | supervision | variant | config | "
          "native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | "
          "F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | "
          "gt_convention | run_dir | notes |")
SEP = "|" + "---|" * 21
CONFIG = {"fedavg11": "collaborative (11 clients, FedAvg)",
          "central": "centralized (1 client)",
          "normality": "normal-Gaussian score, no MLP (ablation)"}


def ms(vals):
    import statistics
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"


def controls(split):
    ctrl = json.loads((ROOT / "idea-stage/repro_campaign/gt_controls.json").read_text())
    out = []
    for ds in DS_ORDER:
        c = ctrl[ds][f"split_{split}_4fps"]
        out.append("| " + " | ".join([
            "GOLD_BROADCAST", "—", DS_LABEL[ds], split, "control", "control", "n/a",
            "video", f"{c['broadcast_ROC_AUC']:.4f}", f"{c['broadcast_AP']:.4f}",
            "n/a", "n/a", "n/a", "1.0000", str(c["n_frames"]),
            f"{c['base_rate']:.4f}", "1", "n/a", "§4+D1",
            "idea-stage/repro_campaign/",
            "zero-temporal-resolution ceiling, full GT pool"]) + " |")
        out.append("| " + " | ".join([
            "RANDOM_UNIFORM", "—", DS_LABEL[ds], split, "control", "control", "n/a",
            "4 fps",
            f"{c['random_ROC_AUC_mean']:.4f} ± {c['random_ROC_AUC_sd']:.4f}",
            f"{c['random_AP_mean']:.4f} ± {c['random_AP_sd']:.4f}",
            "n/a", "n/a", "n/a", "0.0000", str(c["n_frames"]),
            f"{c['base_rate']:.4f}", "20", "n/a", "§4",
            "idea-stage/repro_campaign/",
            "U(0,1) per frame, 20 seeds, full GT pool"]) + " |")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--stratum", default="")
    args = ap.parse_args()
    rows = json.loads(Path(args.json).read_text())
    rec = json.loads((ROOT / "idea-stage/repro_clap/run_record.json").read_text())

    key = f"strat_{args.stratum}" if args.stratum else "pooled"
    agg = {}
    for r in rows:
        base = re.sub(r"_s\d$", "", r["variant"])
        p = r.get(key, {})
        if not p or p.get("n_videos", 0) == 0:
            continue
        agg.setdefault((r["dataset"], base), []).append((p, r))

    print(HEADER)
    print(SEP)
    if args.controls:
        for c in controls(args.split):
            print(c)
    for ds in DS_ORDER:
        for base in ["fedavg11", "central", "normality"]:
            got = agg.get((ds, base))
            if not got:
                continue
            ps = [g[0] for g in got]
            r0 = got[0][1]
            single = (ps[0].get("base_rate") in (0.0, 1.0))
            note = f"E={rec['frozen_rounds'][ds]} global rounds (val-selected)" \
                if base != "normality" else "no training; CLAP's coarse-to-fine normal model only"
            if args.stratum:
                note += f"; stratum={args.stratum}"
            if r0["n_videos_missing"]:
                note += (f"; missing {r0['n_videos_missing']}/"
                         f"{r0['n_videos_in_split']} ({r0['missing_frac']:.1%}) "
                         "dropped, not interpolated")
            if single:
                note += "; single-class pool, metrics undefined"
            print("| " + " | ".join([
                "CLAP", "2", DS_LABEL[ds], args.split, "unlabelled", "base",
                CONFIG[base], "2 fps",
                "n/a" if single else ms([p["frame_ROC_AUC"] for p in ps]),
                "n/a" if single else ms([p["frame_PR_AUC"] for p in ps]),
                "n/a", "n/a", "n/a",
                "n/a" if single else ms([p["AP_norm"] for p in ps]),
                str(ps[0]["n_frames"]), f"{ps[0]['base_rate']:.4f}",
                "1" if base == "normality" else "3",
                "n/a", "§4", "idea-stage/repro_clap/", note]) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
