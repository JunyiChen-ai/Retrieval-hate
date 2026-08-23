#!/usr/bin/env python
"""Collect A0 +- OCR end-to-end results per idea-stage/A0_OCR_E2E_FREEZE.md.

Applies the pipeline's own model-selection rule (best epoch >= warmup by
Val_Retrieval acc, tie-broken by Val_Retrieval roc) and reports the val macro-F1
and val acc at that epoch. Test lines are never parsed -- under --val_only_eval
they are dev duplicates, and reading them would be meaningless anyway.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

VAL_MACRO = re.compile(
    r"^Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) "
    r"macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)", re.M)

GO_T, AMBIG_T = 0.010, 0.003
FROZEN_SPACE_DELTA = 0.0094


def parse(path, warmup):
    txt = Path(path).read_text()
    rows = {}
    for m in VAL_MACRO.finditer(txt):
        e = int(m.group(1))
        rows[e] = {"macro_f1": float(m.group(2)), "acc": float(m.group(5)),
                   "roc": float(m.group(6))}
    if not rows:
        raise SystemExit("NO_VAL_LINES in %s" % path)
    elig = [e for e in rows if e >= warmup] or list(rows)
    sel = max(elig, key=lambda e: (rows[e]["acc"], rows[e]["roc"]))
    return sel, rows[sel], len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="logging/runs/a0_ocr_e2e/trainlogs")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="idea-stage/a0_ocr_e2e.json")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    res = {"A": {}, "B": {}}
    for arm in ("A", "B"):
        for s in seeds:
            p = Path(a.logdir) / ("arm%s_seed%d.trainlog" % (arm, s))
            sel, r, n = parse(p, a.warmup)
            res[arm][s] = {"selected_epoch": sel, "n_epochs": n, **r}

    out = {"per_run": res, "seeds": seeds, "warmup": a.warmup}
    for metric in ("macro_f1", "acc"):
        A = [res["A"][s][metric] for s in seeds]
        B = [res["B"][s][metric] for s in seeds]
        d = [b - x for x, b in zip(A, B)]
        out[metric] = {
            "A": A, "B": B, "delta": d,
            "A_mean": st.mean(A), "A_std": st.pstdev(A) if len(A) < 2 else st.stdev(A),
            "B_mean": st.mean(B), "B_std": st.pstdev(B) if len(B) < 2 else st.stdev(B),
            "delta_mean": st.mean(d),
            "delta_std": st.pstdev(d) if len(d) < 2 else st.stdev(d),
            "n_positive_seeds": sum(1 for x in d if x > 0),
        }

    dm = out["macro_f1"]["delta_mean"]
    npos = out["macro_f1"]["n_positive_seeds"]
    if dm >= GO_T and npos == len(seeds):
        verdict = "GO"
    elif dm <= AMBIG_T:
        verdict = "NO-GO"
    else:
        verdict = "AMBIGUOUS"
    ratio = dm / FROZEN_SPACE_DELTA
    if ratio >= 1.5:
        amp = "AMPLIFIED"
    elif ratio >= 0.5:
        amp = "UNCHANGED"
    else:
        amp = "SHRUNK"
    out["verdict"] = verdict
    out["frozen_space_delta"] = FROZEN_SPACE_DELTA
    out["ratio_to_frozen_space"] = ratio
    out["amplification"] = amp

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
