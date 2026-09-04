"""Module 1 elicitation-level gate: per-second 2x2 cells of (K30 fired, K4 fired) with GT hate rate,
K30 recall/precision, and K30-vs-K4 agreement, for a given K30 verdict tag.  Uses val/test GT only
for analysis (developmental evidence, never for training).  Writes
runs/20260903_hier_evidence_mil/module1_elicitation/<corpus>/cells_<tag>.json

    python experiments/20260903_hier_evidence_mil/module1_cells.py --corpus hatemm --fine-tag qwen
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hate_common import data as hdata   # noqa: E402
import vlm_verdict                       # noqa: E402
import verdict_hmm                       # noqa: E402

K, J = vlm_verdict.GRANULARITIES


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fine-tag", default="qwen")
    ap.add_argument("--coarse-tag", default="qwen")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--out-root", default=os.path.join(
        ROOT, "runs", "20260903_hier_evidence_mil", "module1_elicitation"))
    args = ap.parse_args(argv)
    VF = vlm_verdict.load_verdicts(args.corpus, k=K, tag=args.fine_tag)
    VC = vlm_verdict.load_verdicts(args.corpus, k=J, tag=args.coarse_tag)
    out = {"corpus": args.corpus, "fine_tag": args.fine_tag,
           "coarse_tag": args.coarse_tag, "splits": {}}
    # window-level agreement over every video with both verdicts (no GT needed)
    n_f = n_fc = 0
    for v in VF:
        if v not in VC:
            continue
        bf = verdict_hmm.binarize(VF[v])
        bc = verdict_hmm.binarize(VC[v])[verdict_hmm._block_map(K, J)]
        n_f += int(bf.sum())
        n_fc += int((bf * bc).sum())
    out["fine_windows_fired"] = n_f
    out["fine_fired_with_coarse_fired_rate"] = (n_fc / n_f) if n_f else None
    for split in args.splits.split(","):
        gt = hdata.gt_arrays(args.corpus, split)
        bf_all, bc_all, y_all = [], [], []
        missing = 0
        for vid in sorted(gt):
            if vid not in VF or vid not in VC:
                missing += 1
                continue
            n = len(gt[vid])
            bf_all.append(verdict_hmm.rows_from_windows(
                verdict_hmm.binarize(VF[vid]), n, K))
            bc_all.append(verdict_hmm.rows_from_windows(
                verdict_hmm.binarize(VC[vid]), n, J))
            y_all.append(np.asarray(gt[vid]))
        bf = np.concatenate(bf_all)
        bc = np.concatenate(bc_all)
        y = np.concatenate(y_all)
        cells = {}
        for a in (0, 1):
            for b in (0, 1):
                sel = (bf == a) & (bc == b)
                cells["fine%d_coarse%d" % (a, b)] = {
                    "n": int(sel.sum()),
                    "gt_rate": float(y[sel].mean()) if sel.sum() else None}
        pos = y == 1
        out["splits"][split] = {
            "n_seconds": int(len(y)), "n_missing_videos": missing, "cells": cells,
            "fine_recall": float(bf[pos].mean()),
            "fine_precision": float(y[bf == 1].mean()) if (bf == 1).any() else None,
            "coarse_recall": float(bc[pos].mean()),
            "coarse_precision": float(y[bc == 1].mean()) if (bc == 1).any() else None,
        }
    d = os.path.join(args.out_root, args.corpus)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "cells_%s.json" % args.fine_tag)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
