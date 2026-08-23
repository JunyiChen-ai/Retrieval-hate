#!/usr/bin/env python
"""Parse the RGCL component-ablation grid trainlogs and apply the frozen decision
rules from idea-stage/RGCL_ABLATION_FREEZE.md.

Readout (FREEZE section 3), per run, per inference head, independently:
  I2 (kNN vote):  select epoch = argmax_{e>=warmup} (Val_Retrieval acc, Val_Retrieval roc)
  I1 (head):      select epoch = argmax_{e>=warmup} (dev head acc, dev head roc)
  report          test macro-F1 at that epoch (primary) + val macro-F1 at that epoch.

Decision (FREEZE section 4), on TEST, with VAL as a robustness mirror:
  cell supports  <=>  paired-by-seed mean diff >= +0.005 AND >= 2/3 seeds positive
  ALIVE          <=>  supporting cells >= ceil(n_cells / 2)
"""
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

LOGDIR = "logging/runs/rgcl_ablation/logs"
WARMUP = 5
THRESH = 0.005

RE_VAL_RET = re.compile(
    r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) "
    r"acc: ([\d.]+) roc: ([\d.]+)")
RE_TEST_RET = re.compile(
    r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) "
    r"acc: ([\d.]+) roc: ([\d.]+)")
RE_DEV_HEAD = re.compile(
    r"^dev\s+Epoch (\d+) acc: ([\d.]+) roc: ([\d.]+) pre: [\d.]+ recall: [\d.]+ "
    r"f1: [\d.]+ loss: [\d.]+ \| macroF1: ([\d.]+)", re.M)
RE_TEST_HEAD = re.compile(
    r"^test Epoch (\d+) acc: ([\d.]+) roc: ([\d.]+) pre: [\d.]+ recall: [\d.]+ "
    r"f1: [\d.]+ \| macroF1: ([\d.]+)", re.M)


def parse_run(path):
    """-> {'I1': {'epoch','val','test'}, 'I2': {...}} or None if unparseable."""
    txt = open(path, errors="replace").read()
    val_ret, test_ret, dev_head, test_head = {}, {}, {}, {}
    for m in RE_VAL_RET.finditer(txt):
        e = int(m.group(1))
        val_ret[e] = dict(macro_f1=float(m.group(2)), acc=float(m.group(5)),
                          roc=float(m.group(6)))
    for m in RE_TEST_RET.finditer(txt):
        e = int(m.group(1))
        test_ret[e] = dict(macro_f1=float(m.group(2)), acc=float(m.group(5)),
                           roc=float(m.group(6)))
    for m in RE_DEV_HEAD.finditer(txt):
        e = int(m.group(1))
        dev_head[e] = dict(acc=float(m.group(2)), roc=float(m.group(3)),
                           macro_f1=float(m.group(4)))
    for m in RE_TEST_HEAD.finditer(txt):
        e = int(m.group(1))
        test_head[e] = dict(acc=float(m.group(2)), roc=float(m.group(3)),
                            macro_f1=float(m.group(4)))
    if not (val_ret and test_ret and dev_head and test_head):
        return None

    def select(val_d, test_d):
        cand = [e for e in val_d if e >= WARMUP] or list(val_d)
        best = max(cand, key=lambda e: (val_d[e]["acc"], val_d[e]["roc"]))
        if best not in test_d:
            return None
        return dict(epoch=best, val=val_d[best]["macro_f1"],
                    test=test_d[best]["macro_f1"])

    i2 = select(val_ret, test_ret)
    i1 = select(dev_head, test_head)
    if i1 is None or i2 is None:
        return None
    return {"I1": i1, "I2": i2}


def load_grid():
    """-> results[(enc, ds, loss, inf)][seed] = {'val':.., 'test':.., 'epoch':..}"""
    res = defaultdict(dict)
    missing = []
    for enc in ("CLIP", "QWEN", "LORA"):
        for ds in ("HateMM", "MHC", "MHC_zh", "ImpliHateVid"):
            for loss in ("L1", "L2", "L3"):
                for seed in (0, 1, 2):
                    tag = "{}_{}_{}_s{}".format(enc, ds, loss, seed)
                    path = os.path.join(LOGDIR, tag + ".trainlog")
                    if not os.path.exists(path):
                        missing.append(tag)
                        continue
                    parsed = parse_run(path)
                    if parsed is None:
                        missing.append(tag + " (UNPARSEABLE)")
                        continue
                    for inf in ("I1", "I2"):
                        res[(enc, ds, loss, inf)][seed] = parsed[inf]
    return res, missing


def agg(res, key, split):
    """mean/std over seeds of `split` metric, or None."""
    d = res.get(key, {})
    vals = [d[s][split] for s in sorted(d)]
    if len(vals) == 0:
        return None
    return dict(mean=float(np.mean(vals)), std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                n=len(vals), vals=vals)


def paired_diff(res, key_a, key_b, split):
    """Per-seed paired difference a - b. -> (mean, [per-seed diffs]) or None."""
    da, db = res.get(key_a, {}), res.get(key_b, {})
    seeds = sorted(set(da) & set(db))
    if not seeds:
        return None
    diffs = [da[s][split] - db[s][split] for s in seeds]
    return float(np.mean(diffs)), diffs


def cell_supports(mean_diff, diffs):
    pos = sum(1 for d in diffs if d > 0)
    return (mean_diff >= THRESH) and (pos >= math.ceil(2 * len(diffs) / 3.0))


def component_verdict(rows):
    """rows = list of (label, mean_diff, diffs). -> (verdict, n_support, n_cells)"""
    live = [r for r in rows if cell_supports(r[1], r[2])]
    n = len(rows)
    need = math.ceil(n / 2.0)
    return ("ALIVE" if len(live) >= need else "DECORATIVE"), len(live), n, need


def main():
    os.chdir("/home/jehc223/Retrieval-hate")
    res, missing = load_grid()
    out = {"missing": missing}

    cells = [(enc, ds) for enc in ("CLIP", "QWEN", "LORA")
             for ds in ("HateMM", "MHC", "MHC_zh", "ImpliHateVid")
             if (enc, ds, "L3", "I2") in res]

    # ---- full grid table ----
    grid = {}
    for enc, ds in cells:
        for loss in ("L1", "L2", "L3"):
            for inf in ("I1", "I2"):
                k = (enc, ds, loss, inf)
                grid["|".join(k)] = {
                    "test": agg(res, k, "test"), "val": agg(res, k, "val"),
                    "epochs": {s: res[k][s]["epoch"] for s in sorted(res.get(k, {}))},
                }
    out["grid"] = grid

    # ---- component contributions ----
    comp = {}
    for split in ("test", "val"):
        # kNN readout: I2 - I1, cells = (enc, ds, loss)
        rows = []
        for enc, ds in cells:
            for loss in ("L1", "L2", "L3"):
                pd = paired_diff(res, (enc, ds, loss, "I2"), (enc, ds, loss, "I1"), split)
                if pd:
                    rows.append(("{}/{}/{}".format(enc, ds, loss), pd[0], pd[1]))
        comp[("knn_readout", split)] = rows
        # retrieval guidance: L3 - L2, cells = (enc, ds, inf)
        rows = []
        for enc, ds in cells:
            for inf in ("I1", "I2"):
                pd = paired_diff(res, (enc, ds, "L3", inf), (enc, ds, "L2", inf), split)
                if pd:
                    rows.append(("{}/{}/{}".format(enc, ds, inf), pd[0], pd[1]))
        comp[("retrieval_guidance", split)] = rows
        # contrastive regularisation: L2 - L1
        rows = []
        for enc, ds in cells:
            for inf in ("I1", "I2"):
                pd = paired_diff(res, (enc, ds, "L2", inf), (enc, ds, "L1", inf), split)
                if pd:
                    rows.append(("{}/{}/{}".format(enc, ds, inf), pd[0], pd[1]))
        comp[("contrastive_reg", split)] = rows

    out["components"] = {}
    for (name, split), rows in comp.items():
        verdict, nlive, ntot, need = component_verdict(rows)
        out["components"]["{}|{}".format(name, split)] = {
            "verdict": verdict, "support": nlive, "cells": ntot, "need": need,
            "mean_over_cells": float(np.mean([r[1] for r in rows])) if rows else None,
            "rows": [{"cell": r[0], "mean": r[1], "diffs": r[2],
                      "supports": cell_supports(r[1], r[2])} for r in rows],
        }

    # ---- interaction: [(L3,I2)-(L3,I1)] - [(L1,I2)-(L1,I1)] ----
    for split in ("test", "val"):
        rows = []
        for enc, ds in cells:
            a = paired_diff(res, (enc, ds, "L3", "I2"), (enc, ds, "L3", "I1"), split)
            b = paired_diff(res, (enc, ds, "L1", "I2"), (enc, ds, "L1", "I1"), split)
            if not (a and b):
                continue
            # recompute paired-by-seed difference-of-differences
            sa = set(res[(enc, ds, "L3", "I2")]) & set(res[(enc, ds, "L3", "I1")])
            sb = set(res[(enc, ds, "L1", "I2")]) & set(res[(enc, ds, "L1", "I1")])
            seeds = sorted(sa & sb)
            dd = [(res[(enc, ds, "L3", "I2")][s][split] - res[(enc, ds, "L3", "I1")][s][split])
                  - (res[(enc, ds, "L1", "I2")][s][split] - res[(enc, ds, "L1", "I1")][s][split])
                  for s in seeds]
            rows.append(("{}/{}".format(enc, ds), float(np.mean(dd)), dd,
                         a[0], b[0]))
        live = [r for r in rows if cell_supports(r[1], r[2])]
        need = math.ceil(len(rows) / 2.0)
        out.setdefault("interaction", {})["{}".format(split)] = {
            "verdict": "RGCL_STORY_HOLDS" if len(live) >= need else "RGCL_STORY_FAILS",
            "support": len(live), "cells": len(rows), "need": need,
            "rows": [{"cell": r[0], "dd_mean": r[1], "dd": r[2],
                      "knn_gain_at_L3": r[3], "knn_gain_at_L1": r[4],
                      "supports": cell_supports(r[1], r[2])} for r in rows],
        }

    json.dump(out, open("logging/runs/rgcl_ablation/analysis.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "grid"}, indent=1)[:200])
    print("wrote logging/runs/rgcl_ablation/analysis.json ; cells={} missing={}".format(
        len(cells), len(missing)))


if __name__ == "__main__":
    main()
