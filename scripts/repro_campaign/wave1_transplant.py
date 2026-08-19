#!/usr/bin/env python
"""REPRO campaign Wave 1 — the freeze §7 transplant-fidelity table.

The only pass/fail judgement in the campaign: a reproduction is a successful
transplant if it lands within **±0.03 absolute** of LELA's published number on
**both** metrics.  Out of tolerance is reported and investigated, never hidden
and never a reason to drop the row.

LELA does not say which MultiHateClip language it pooled, so both are compared
against the same target (freeze §7's recorded caveat).  It also does not say
which PR-AUC convention it used; LAVAD's own `src/eval.py`, which LELA's port
would have inherited, reports `auc(recall, precision)` rather than average
precision, so both of our numbers are printed against the target and the verdict
is taken on the freeze's own convention (average precision) with the trapezoid
figure shown beside it.

  python scripts/repro_campaign/wave1_transplant.py \
      --lavad idea-stage/repro_campaign/eval_LAVAD_test.json \
      --urf   idea-stage/repro_campaign/eval_URF-HVAA_test.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TOL = 0.03
# freeze §7
TARGETS = {
    ("LAVAD", "HateMM"): (0.5781, 0.6163),
    ("LAVAD", "MHC"): (0.5865, 0.6302),
    ("LAVAD", "MHC_zh"): (0.5865, 0.6302),
    ("URF-HVAA", "HateMM"): (0.6239, 0.5674),
    ("URF-HVAA", "MHC"): (0.6147, 0.5626),
    ("URF-HVAA", "MHC_zh"): (0.6147, 0.5626),
}
DS_LABEL = {"HateMM": "HateMM", "MHC": "MHC-EN", "MHC_zh": "MHC-ZH",
            "HateClipSeg": "HateClipSeg"}
NOTE = {"MHC": "LELA's 'MultiHateClip' column, language unstated",
        "MHC_zh": "LELA's 'MultiHateClip' column, language unstated",
        "HateMM": ""}

HEADER = ("| method | dataset | LELA PR-AUC | ours PR-AUC (AP) | |diff| | "
          "LELA ROC-AUC | ours ROC-AUC | |diff| | verdict | ours PR-AUC "
          "(trapezoid, LAVAD's own convention) | note |")
SEP = "|" + "---|" * 11


def rows(path: Path, method: str, variant: str):
    out = []
    for r in json.loads(path.read_text()):
        if r["variant"] != variant:
            continue
        key = (method, r["dataset"])
        if key not in TARGETS:
            continue
        t_pr, t_roc = TARGETS[key]
        p = r["pooled"]
        pr, roc = p["frame_PR_AUC"], p["frame_ROC_AUC"]
        d_pr, d_roc = abs(pr - t_pr), abs(roc - t_roc)
        ok = d_pr <= TOL and d_roc <= TOL
        out.append("| " + " | ".join([
            method, DS_LABEL[r["dataset"]], f"{t_pr:.4f}", f"{pr:.4f}",
            f"{d_pr:.4f}", f"{t_roc:.4f}", f"{roc:.4f}", f"{d_roc:.4f}",
            "**OK**" if ok else "**OUT_OF_TOLERANCE**",
            f"{p.get('frame_PR_AUC_trapz', float('nan')):.4f}",
            NOTE.get(r["dataset"], ""),
        ]) + " |")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lavad")
    ap.add_argument("--urf")
    ap.add_argument("--variant", default="base")
    args = ap.parse_args()
    print(HEADER)
    print(SEP)
    for path, name in ((args.lavad, "LAVAD"), (args.urf, "URF-HVAA")):
        if not path or not Path(path).exists():
            continue
        for r in rows(Path(path), name, args.variant):
            print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
