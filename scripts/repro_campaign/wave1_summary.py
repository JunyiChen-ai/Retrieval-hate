#!/usr/bin/env python
"""One command that builds the cross-method Wave 1 table from whatever eval JSONs exist.

Written so the final summary does not depend on hand-assembly, and so it can be
produced the moment the last run lands rather than only while someone is watching.
Methods that have not finished are listed as PENDING rather than omitted, because
a table that silently drops a missing method reads like a complete one.

Reads `idea-stage/repro_campaign/eval_*.json`, picks each method's headline
variant, and emits markdown.

    python scripts/repro_campaign/wave1_summary.py            # test split
    python scripts/repro_campaign/wave1_summary.py --split all
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path("/home/jehc223/Retrieval-hate")
EVAL = ROOT / "idea-stage/repro_campaign"
DS_ORDER = ["HateMM", "MHC", "MHC_zh", "HateClipSeg"]
DS_LABEL = {"HateMM": "HateMM", "MHC": "MHC-EN", "MHC_zh": "MHC-ZH",
            "HateClipSeg": "HateClipSeg"}

# The headline variant per method, fixed here rather than chosen by score.
HEADLINE = {
    "ZS-CLIP": "main",
    "ZS-ImageBind (image)": "base",
    "Qwen2.5-VL-7B grounding": "query=main",
    "LaGoVAD": "main",
    "UniTime": "window",
    "LAVAD": "base",
    "URF-HVAA": "base",
    "AV2A (combined)": None,          # first variant found
}
WAVE = {"ZS-CLIP": 0, "ZS-ImageBind (image)": 0, "Qwen2.5-VL-7B grounding": 0,
        "LaGoVAD": 1, "UniTime": 1, "LAVAD": 1, "URF-HVAA": 1, "AV2A (combined)": 1}


def load_zs_clip(split):
    """ZS-CLIP predates the shared `eval_*.json` convention and writes its own
    results file. Read it rather than reporting a finished method as pending."""
    f = ROOT / "idea-stage/repro_zs_clip/results.json"
    if not f.exists():
        return {}
    want = "full" if split == "all" else split
    out = {}
    for r in json.loads(f.read_text()):
        if (r.get("method") != "ZS-CLIP" or r.get("variant") != "base"
                or r.get("prompt_set") != "main" or r.get("split") != want
                or r.get("stratum") != "all"):
            continue
        # that file labels datasets MHC-EN / MHC-ZH; the shared convention is
        # MHC / MHC_zh. Map rather than silently dropping two of four datasets.
        ds = {"MHC-EN": "MHC", "MHC-ZH": "MHC_zh"}.get(r["dataset"], r["dataset"])
        if ds not in DS_ORDER:
            continue
        out.setdefault("ZS-CLIP", {})[ds] = {
            "pooled": {"frame_ROC_AUC": r["roc"], "frame_PR_AUC": r["ap"]}}
    return out


def load(split):
    out = {}
    for f in sorted(glob.glob(str(EVAL / "eval_*.json"))):
        if f.endswith(f"_{split}.json") is False:
            continue
        try:
            rows = json.loads(Path(f).read_text())
        except Exception:
            continue
        for r in rows:
            m, ds, var = r.get("method"), r.get("dataset"), r.get("variant")
            if not m or ds not in DS_ORDER:
                continue
            want = HEADLINE.get(m, "__none__")
            if want is not None and want != "__none__" and var != want:
                continue
            out.setdefault(m, {}).setdefault(ds, r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    got = load(a.split)
    got.update(load_zs_clip(a.split))

    print(f"### Wave 1 cross-method summary — {a.split} split, headline variant per method\n")
    print("| method | wave | " + " | ".join(DS_LABEL[d] for d in DS_ORDER) + " |")
    print("|---|---|" + "---|" * len(DS_ORDER))
    for m in HEADLINE:
        if m not in got:
            print(f"| {m} | {WAVE.get(m,'?')} | " +
                  " | ".join(["_pending_"] * len(DS_ORDER)) + " |")
            continue
        cells = []
        for d in DS_ORDER:
            r = got[m].get(d)
            if not r:
                cells.append("—")
                continue
            p = r["pooled"]
            roc, apv = p.get("frame_ROC_AUC"), p.get("frame_PR_AUC")
            cells.append(f"{roc:.4f} / {apv:.4f}" if roc is not None else "n/a")
        print(f"| {m} | {WAVE.get(m,'?')} | " + " | ".join(cells) + " |")

    print("\nCells are `frame_ROC_AUC / frame_PR_AUC`. Controls, for reference:\n")
    ctrl = json.loads((EVAL / "gt_controls.json").read_text())
    print("| control | " + " | ".join(DS_LABEL[d] for d in DS_ORDER) + " |")
    print("|---|" + "---|" * len(DS_ORDER))
    for name, key in [("GOLD_BROADCAST", "broadcast"), ("random floor (= base rate)", "base")]:
        cells = []
        for d in DS_ORDER:
            c = ctrl[d][f"split_{a.split}_4fps"]
            cells.append(f"{c['broadcast_ROC_AUC']:.4f} / {c['broadcast_AP']:.4f}"
                         if key == "broadcast" else f"0.5000 / {c['base_rate']:.4f}")
        print(f"| {name} | " + " | ".join(cells) + " |")

    print("\n**`AP_norm` is deliberately not shown.** On HateClipSeg its normalising gap "
          "(broadcast − base) is only ~0.07 against 0.34–0.65 elsewhere, so it amplifies "
          "noise ~14× there and is not comparable across datasets. Read PR-AUC against the "
          "base rate instead. See the global caveat at the head of REPRO_CAMPAIGN_RESULTS.md.")

    missing = [m for m in HEADLINE if m not in got]
    if missing:
        print(f"\n**Pending ({len(missing)}):** " + ", ".join(missing) +
              " — table is incomplete and must not be read as final.")


if __name__ == "__main__":
    raise SystemExit(main())
