#!/usr/bin/env python3
"""Amendment A2: joint val selection over {source set} x {adaptation depth}.

Per target and seed: for each candidate source set (union of others + each
single aux corpus), pretrain, adapt with checkpoints at {0,1,2,4,8,15}, record
val within-ROC of every (source, depth) pair, deploy the best pair, save its
dense TEST scores and metrics.

Output: keys <target>/full/5seed in scale_results.json, dense scores under
scores/<corpus>/full_seed*.jsonl, selection log a2_selection.md.
"""
import json
import os
import sys

import numpy as np
import torch

REPO = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(REPO, "experiments", "20260830_powa_within_diagnosis"))
sys.path.insert(0, os.path.join(REPO, "experiments", "20260830_spantransfer_pilot"))
import spantransfer as ST  # noqa: E402
import scale_up as SU  # noqa: E402

RESULTS = SU.RESULTS
SEEDS5 = SU.SEEDS5
CKPT_EPOCHS = SU.CKPT_EPOCHS


def run_full(target, seed):
    others = [c for c in ST.CORPORA if c != target]
    source_sets = [tuple(others)] + [(c,) for c in others]
    weak = ST.pack_weak(target)
    best = None  # (val, srcs, ep, state_dict)
    log = []
    for srcs in source_sets:
        rng = np.random.default_rng(seed)
        aux = []
        for c in srcs:
            aux += ST.pack_spans(c, rng=rng, shuffle=False)
        model = ST.pretrain(aux, seed)
        ckpts = SU.adapt_with_ckpts(model, weak, seed, ST.TAU, ST.LAMBDA_RANK, 8)
        for ep in CKPT_EPOCHS:
            model.load_state_dict(ckpts[ep])
            val = SU.eval_val_within(model, target)
            log.append((seed, "+".join(srcs), ep, val))
            if best is None or val > best[0]:
                best = (val, srcs, ep,
                        {k: v.clone() for k, v in ckpts[ep].items()})
    model.load_state_dict(best[3])
    out = SU.score_test(model, target, "full", seed, save=True)
    out["selected_sources"] = "+".join(best[1])
    out["selected_epoch"] = best[2]
    out["val_within"] = best[0]
    return out, log


def main():
    blob = json.load(open(RESULTS))
    res = blob["results"]
    sel_lines = ["# A2 selection log", "",
                 "| target | seed | selected sources | epoch | val within |",
                 "|---|---:|---|---:|---:|"]
    for target in ST.CORPORA:
        key = "%s/full/5seed" % target
        if key in res:
            continue
        per_seed = []
        for s in SEEDS5:
            out, _ = run_full(target, s)
            per_seed.append(out)
            sel_lines.append("| %s | %d | %s | %d | %.4f |" % (
                target, s, out["selected_sources"], out["selected_epoch"],
                out["val_within"]))
        agg = {}
        for k in per_seed[0]:
            if k in ("within_n",):
                agg[k] = per_seed[0][k]
            elif k in ("selected_sources", "selected_epoch"):
                agg[k] = [p[k] for p in per_seed]
            elif k == "val_within":
                continue
            elif k == "per_video_auc":
                continue
            else:
                vals = [p[k] for p in per_seed if p[k] is not None]
                agg[k] = ({"mean": float(np.mean(vals)),
                           "sd": float(np.std(vals))} if vals else None)
        res[key] = agg
        with open(RESULTS, "w") as fh:
            json.dump(blob, fh, indent=1)
        print(key, json.dumps(agg), flush=True)
    with open(os.path.join(ST.OUT_DIR, "a2_selection.md"), "w") as fh:
        fh.write("\n".join(sel_lines))
    print("A2_DONE", flush=True)


if __name__ == "__main__":
    main()
