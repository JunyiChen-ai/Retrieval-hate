#!/usr/bin/env python
"""REPRO campaign — build the Wave 0/1/2 master table straight from the raw result files.

Nothing in this script re-computes a metric.  It reads the JSON each method's run
already wrote through the one shared evaluator (`eval_frame.py`) and lays the
numbers out as one table, so the summary document cannot drift from the evidence
by a transcription error.

Every method contributes the variant its own section designated as the headline
BEFORE any number existed; the extra variants that a section discusses are listed
in SECONDARY and printed underneath, clearly marked.  No variant is picked here
by looking at a metric.

  python scripts/repro_campaign/summary_table.py            # markdown to stdout
  python scripts/repro_campaign/summary_table.py --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path("/home/jehc223/Retrieval-hate")
EVAL = ROOT / "idea-stage/repro_campaign"
DS = ["HateMM", "MHC", "MHC_zh", "HateClipSeg"]
DS_LABEL = {"HateMM": "HateMM", "MHC": "MHC-EN", "MHC_zh": "MHC-ZH",
            "HateClipSeg": "HateClipSeg"}
# ZS-CLIP wrote its own result file before the shared evaluator existed and
# names the two MultiHateClip halves by their display label, not their cache key.
DS_ALIAS = {"MHC-EN": "MHC", "MHC-ZH": "MHC_zh"}

# (display name, wave, supervision, eval-json stem, method key in the json,
#  variant key in the json, native rate, run dir)
# `headline` rows are the pre-fixed primary variant of each method.
HEADLINE = [
    ("ZS-CLIP", 0, "label-free", "ZS_CLIP_RESULTS", "ZS-CLIP", "base (prompt=main)",
     "4 fps", "idea-stage/repro_zs_clip/"),
    ("ZS-ImageBind (image)", 0, "label-free", "eval_imagebind",
     "ZS-ImageBind (image)", "base", "4 fps", "idea-stage/repro_campaign/"),
    ("ZS-ImageBind (video)", 0, "label-free", "eval_imagebind",
     "ZS-ImageBind (video)", "base", "0.5 fps", "idea-stage/repro_campaign/"),
    ("ZS-ImageBind (audio)", 0, "label-free", "eval_imagebind",
     "ZS-ImageBind (audio)", "base", "0.5 fps", "idea-stage/repro_campaign/"),
    ("Qwen2.5-VL-7B grounding", 0, "label-free", "eval_qwen_grounding",
     "Qwen2.5-VL-7B grounding", "query=main", "interval", "idea-stage/repro_qwen_ground/"),
    ("LAVAD", 1, "label-free", "eval_LAVAD", "LAVAD", "base",
     "1 fps", "idea-stage/repro_lavad/"),
    ("URF-HVAA", 1, "label-free", "eval_URF-HVAA", "URF-HVAA", "base",
     "0.1 fps", "idea-stage/repro_urf/"),
    ("LaGoVAD", 1, "aux-temporal-pretrain", "eval_lagovad", "LaGoVAD", "main",
     "0.5 fps", "idea-stage/repro_lagovad/"),
    ("AV²A", 1, "label-free", "eval_AV2A", "AV2A", "sim_combined",
     "1 fps / 0.1 fps", "idea-stage/repro_av2a/"),
    ("UniTime", 1, "aux-temporal-pretrain", "eval_UniTime", "UniTime", "window",
     "interval", "idea-stage/repro_unitime/"),
    ("MULDE", 2, "one-class", "eval_MULDE", "MULDE", "clipL336",
     "4 fps", "idea-stage/repro_mulde/"),
    ("CLAP", 2, "unlabelled", "eval_CLAP", "CLAP", "main",
     "32 seg/video", "idea-stage/repro_clap/"),
    ("T3AL", 2, "label-free", "eval_T3AL", "T3AL", "main",
     "interval", "idea-stage/repro_t3al/"),
    ("SeViLA Localizer", 2, "aux-temporal-pretrain", "eval_SeViLA Localizer",
     "SeViLA Localizer", "main", "1 fps", "idea-stage/repro_sevila/"),
]

# Variants a method's own section calls out as materially different from its
# headline.  Printed as a clearly-labelled appendix, never mixed into the main table.
SECONDARY = [
    ("LaGoVAD (bin, text-free head)", 1, "aux-temporal-pretrain", "eval_lagovad",
     "LaGoVAD", "bin", "0.5 fps", "idea-stage/repro_lagovad/"),
    ("LAVAD (raw, pre-refinement)", 1, "label-free", "eval_LAVAD", "LAVAD", "raw",
     "1 fps", "idea-stage/repro_lavad/"),
    ("URF-HVAA (round1, pre-refinement)", 1, "label-free", "eval_URF-HVAA",
     "URF-HVAA", "round1", "0.1 fps", "idea-stage/repro_urf/"),
    ("AV²A (sim_video)", 1, "label-free", "eval_AV2A", "AV2A", "sim_video",
     "1 fps", "idea-stage/repro_av2a/"),
    ("AV²A (sim_audio)", 1, "label-free", "eval_AV2A", "AV2A", "sim_audio",
     "0.1 fps", "idea-stage/repro_av2a/"),
    ("UniTime (mr_seg)", 1, "aux-temporal-pretrain", "eval_UniTime", "UniTime",
     "seg", "segment", "idea-stage/repro_unitime/"),
]


def load_eval(stem: str, split: str) -> list:
    p = EVAL / f"{stem}_{split}.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def zs_clip_rows(split: str) -> dict:
    """ZS-CLIP wrote its own result list, one flat record per row."""
    p = ROOT / "idea-stage/repro_zs_clip/results.json"
    if not p.exists():
        return {}
    out = {}
    for r in json.loads(p.read_text()):
        if r.get("stratum") != "all" or r.get("split") != split:
            continue
        ds = DS_ALIAS.get(r["dataset"], r["dataset"])
        key = (r["method"], f"{r['variant']} (prompt={r['prompt_set']})", ds)
        out[key] = dict(frame_ROC_AUC=round(r["roc"], 4), frame_PR_AUC=round(r["ap"], 4),
                        AP_norm=round(r["ap_norm"], 4) if r.get("ap_norm") is not None else None,
                        base_rate=round(r["base_rate"], 4), n_frames=r["n_frames"],
                        coverage=1.0, n_videos=r.get("n_videos"))
    return out


def collect(split: str) -> dict:
    """(method, variant, dataset) -> pooled metric dict, from every result file."""
    cells = {}
    cells.update(zs_clip_rows("test" if split == "test" else "full"))
    for stem in sorted({h[3] for h in HEADLINE + SECONDARY}):
        if stem == "ZS_CLIP_RESULTS":
            continue
        for r in load_eval(stem, split):
            p = r.get("pooled", {})
            if not p:
                continue
            cell = dict(p)
            iv = r.get("intervals")
            if iv:
                for t in ("0.3", "0.5", "0.7"):
                    cell[f"F1@{t}"] = iv.get(f"F1@{t}")
            cell["n_videos_missing"] = r.get("n_videos_missing")
            cell["missing_frac"] = r.get("missing_frac")
            cells[(r["method"], r["variant"], r["dataset"])] = cell
    return cells


def controls(split: str) -> dict:
    """The two frozen control rows, from the ZS-CLIP result file that computed them
    on the same evaluator (freeze §3)."""
    p = ROOT / "idea-stage/repro_zs_clip/results.json"
    out = {}
    if not p.exists():
        return out
    for r in json.loads(p.read_text()):
        if r.get("stratum") != "all":
            continue
        if r.get("split") != ("test" if split == "test" else "full"):
            continue
        if r["method"] in ("GOLD_BROADCAST", "RANDOM_UNIFORM"):
            out.setdefault(r["method"], {})[DS_ALIAS.get(r["dataset"], r["dataset"])] = (
                round(r["roc"], 4), round(r["ap"], 4), round(r["base_rate"], 4))
    return out


def fmt(v):
    return "—" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def build(split: str) -> tuple[str, list]:
    cells = collect(split)
    ctl = controls(split)
    lines, csvrows = [], []

    head = (["method", "wave", "supervision", "variant", "native_rate"]
            + [f"{DS_LABEL[d]} ROC / AP" for d in DS])
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))

    for name in ("GOLD_BROADCAST", "RANDOM_UNIFORM"):
        row = [f"**{name}**", "—", "control", "control", "video" if "GOLD" in name else "4 fps"]
        for d in DS:
            c = ctl.get(name, {}).get(d)
            row.append(f"{c[0]:.4f} / {c[1]:.4f}" if c else "—")
        lines.append("| " + " | ".join(row) + " |")

    def emit(spec, bold=False):
        name, wave, sup, stem, mkey, vkey, rate, rundir = spec
        row = [f"**{name}**" if bold else name, str(wave), sup, vkey, rate]
        any_cell = False
        for d in DS:
            c = cells.get((mkey, vkey, d))
            if c is None:
                row.append("not run")
                continue
            any_cell = True
            roc, ap = c.get("frame_ROC_AUC"), c.get("frame_PR_AUC")
            row.append(f"{fmt(roc)} / {fmt(ap)}")
            csvrows.append(dict(split=split, method=name, wave=wave, supervision=sup,
                                variant=vkey, native_rate=rate, dataset=DS_LABEL[d],
                                run_dir=rundir, **{k: c.get(k) for k in (
                                    "frame_ROC_AUC", "frame_PR_AUC", "AP_norm",
                                    "AP_norm_reliable", "base_rate", "n_frames",
                                    "n_videos", "coverage", "n_videos_missing",
                                    "F1@0.3", "F1@0.5", "F1@0.7")}))
        lines.append("| " + " | ".join(row) + " |")
        return any_cell

    for spec in HEADLINE:
        emit(spec, bold=True)
    lines.append("")
    lines.append("Secondary variants each method's own section calls out "
                 "(never a substitute for the headline row above):")
    lines.append("")
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for spec in SECONDARY:
        emit(spec)
    return "\n".join(lines), csvrows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "all"])
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    md, rows = build(args.split)
    print(md)
    if args.csv and rows:
        keys = list(rows[0])
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[csv] {args.csv}  rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
