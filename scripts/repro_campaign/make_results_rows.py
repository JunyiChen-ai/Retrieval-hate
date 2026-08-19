#!/usr/bin/env python
"""Turn the evaluator's JSON into REPRO_CAMPAIGN_RESULTS.md rows (freeze §14 schema).

Prints markdown; it never edits the results file itself, so two agents writing
different sections of that file cannot collide through this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/home/jehc223/Retrieval-hate")
DS_LABEL = {"HateMM": "HateMM", "MHC": "MHC-EN", "MHC_zh": "MHC-ZH",
            "HateClipSeg": "HateClipSeg"}

# Column layout follows the ZS-CLIP section already in REPRO_CAMPAIGN_RESULTS.md:
# freeze §14 plus the `query_set` column that section added, so one file has one table shape.
HEADER = ("| method | wave | dataset | split | supervision | variant | query_set | "
          "native_rate | frame_ROC_AUC | frame_PR_AUC | F1@0.3 | F1@0.5 | "
          "F1@0.7 | AP_norm | n_frames | base_rate | seeds | transplant | "
          "gt_convention | run_dir | notes |")
SEP = "|" + "---|" * 21


def fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def row(r, run_dir, notes, transplant="n/a", gt_conv="§4", seeds="1", stratum=None):
    p = r.get("pooled", {}) if stratum is None else r.get(f"strat_{stratum}", {})
    iv = r.get("intervals") if stratum is None else r.get(f"strat_{stratum}_intervals")
    f = (lambda k: fmt(iv[k])) if iv else (lambda k: "n/a")
    rate = r["native_rate"]
    rate_s = f"{rate:g} fps"
    return ("| " + " | ".join([
        r["method"], str(r["wave"]), DS_LABEL[r["dataset"]], r["split"],
        r["supervision"], "base", r["variant"], rate_s,
        fmt(p.get("frame_ROC_AUC")), fmt(p.get("frame_PR_AUC")),
        f("F1@0.3"), f("F1@0.5"), f("F1@0.7"),
        fmt(p.get("AP_norm")), str(p.get("n_frames")), fmt(p.get("base_rate")),
        seeds, transplant, gt_conv, run_dir, notes,
    ]) + " |")


def control_rows(split="test"):
    ctrl = json.loads((ROOT / "idea-stage/repro_campaign/gt_controls.json").read_text())
    out = []
    for ds in ["HateMM", "MHC", "MHC_zh", "HateClipSeg"]:
        c = ctrl[ds][f"split_{split}_4fps"]
        out.append("| " + " | ".join([
            "GOLD_BROADCAST", "—", DS_LABEL[ds], split, "control", "control", "n/a",
            "video",
            fmt(c["broadcast_ROC_AUC"]), fmt(c["broadcast_AP"]), "n/a", "n/a", "n/a",
            "1.0000", str(c["n_frames"]), fmt(c["base_rate"]), "1", "n/a", "§4+D1",
            "idea-stage/repro_campaign/", "zero-temporal-resolution ceiling, full GT pool",
        ]) + " |")
        out.append("| " + " | ".join([
            "RANDOM_UNIFORM", "—", DS_LABEL[ds], split, "control", "control", "n/a",
            "4 fps",
            f"{c['random_ROC_AUC_mean']:.4f} ± {c['random_ROC_AUC_sd']:.4f}",
            f"{c['random_AP_mean']:.4f} ± {c['random_AP_sd']:.4f}",
            "n/a", "n/a", "n/a", "0.0000", str(c["n_frames"]), fmt(c["base_rate"]),
            "20", "n/a", "§4", "idea-stage/repro_campaign/",
            "U(0,1) per frame, 20 seeds, full GT pool",
        ]) + " |")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--run-dir", default="`idea-stage/repro_campaign/`")
    ap.add_argument("--notes", default="")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--split", default="test", help="which split the control rows describe")
    ap.add_argument("--strata", action="store_true",
                    help="emit the single_span / multi_span rows instead of the pooled row")
    args = ap.parse_args()

    print(HEADER)
    print(SEP)
    if args.controls:
        for r in control_rows(args.split):
            print(r)
    for jf in args.json:
        for r in json.loads(Path(jf).read_text()):
            if r.get("pooled", {}).get("n_videos", 0) == 0:
                continue
            note = args.notes
            if r["n_videos_missing"]:
                note = (note + "; " if note else "") + \
                    f"missing {r['n_videos_missing']}/{r['n_videos_in_split']} " \
                    f"({r['missing_frac']:.1%}) dropped, not interpolated"
            if args.strata:
                for st in ("single_span", "multi_span"):
                    q = r.get(f"strat_{st}")
                    if not q or q.get("n_videos", 0) == 0:
                        continue
                    n = (note + "; " if note else "") + f"stratum={st}"
                    if q.get("frame_ROC_AUC") is None or q.get("base_rate") in (0.0, 1.0):
                        n += " single-class pool, metrics undefined"
                    print(row(r, args.run_dir, n, stratum=st))
            else:
                print(row(r, args.run_dir, note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
