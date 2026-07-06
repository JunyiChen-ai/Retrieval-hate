#!/usr/bin/env python
"""P5 quality gate (CPU, run BEFORE training).

(a) Self-verdict flip: from data/Counterfactual/<DS>/train_twins.jsonl, the conditional flip
    rate P(sanitized==BENIGN | original==HARMFUL) and overall retention. Gate: conditional
    flip rate >= 0.80.
(b) Hardness: each VERIFIED twin's sanitized text embedding must be closer to its anchor's
    original text embedding than the MEDIAN benign TRAIN video is (a near miss). Uses the CLIP
    text space (original text_feats from the whole-video cache; twin text_feats from the twin
    cache). Reports the near-miss fraction + distribution.

Reads only the twin jsonl, twin cache, and CLIP train cache; writes only --out.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/data/jehc223/RGCL"
MODEL = "openai_clip-vit-large-patch14-336"


def load_twins_jsonl(ds):
    rows = []
    with open(os.path.join(ROOT, "data/Counterfactual", ds, "train_twins.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="MHC,MHC_zh")
    ap.add_argument("--out", default="scripts/analysis/p5_out/quality_gate.json")
    args = ap.parse_args()

    result = {}
    for ds in args.datasets.split(","):
        ds = ds.strip()
        rows = load_twins_jsonl(ds)
        n = len(rows)
        n_orig_harm = sum(r["orig_verdict"] == "HARMFUL" for r in rows)
        n_flip = sum(r["flipped"] for r in rows)
        n_regen = sum(r.get("regen_used", False) for r in rows)
        cond_flip = (n_flip / n_orig_harm) if n_orig_harm else 0.0
        retention = n_flip / n if n else 0.0

        # ---- hardness in CLIP text space ----
        clip = torch.load("{}/data/CLIP_Embedding/{}/train_{}_HF.pt".format(ROOT, ds, MODEL),
                          map_location="cpu")
        cids = [i for sub in clip["ids"] for i in sub]
        ctext = F.normalize(clip["text_feats"].float(), dim=1)
        clab = clip["labels"].long().numpy()
        id2row = {vid: r for r, vid in enumerate(cids)}
        benign_rows = np.where(clab == 0)[0]
        benign_mat = ctext[benign_rows]                       # [Nb, Dt]

        twin = torch.load("{}/data/CLIP_Embedding/{}/train_cftwin_{}_HF.pt".format(
            ROOT, ds, MODEL), map_location="cpu")
        tids = [i for sub in twin["ids"] for i in sub]
        ttext = F.normalize(twin["text_feats"].float(), dim=1)
        tflip = twin["flipped"].bool().numpy()

        near_miss, twin_sims, med_benign_sims = 0, [], []
        n_eval = 0
        for k, vid in enumerate(tids):
            if not tflip[k] or vid not in id2row:
                continue
            anchor = ctext[id2row[vid]]                        # [Dt]
            s_twin = float(anchor @ ttext[k])
            s_benign = benign_mat @ anchor                     # [Nb]
            med = float(s_benign.median())
            twin_sims.append(s_twin)
            med_benign_sims.append(med)
            near_miss += int(s_twin > med)
            n_eval += 1
        near_frac = near_miss / n_eval if n_eval else 0.0

        gate_flip = cond_flip >= 0.80
        gate_hard = near_frac >= 0.50   # majority of twins are near misses
        result[ds] = {
            "n_anchors": n, "n_orig_harmful": n_orig_harm, "n_verified_flips": n_flip,
            "n_regen": n_regen,
            "conditional_flip_rate": round(cond_flip, 4),
            "overall_retention": round(retention, 4),
            "hardness_n_eval": n_eval,
            "near_miss_fraction": round(near_frac, 4),
            "twin_sim_mean": round(float(np.mean(twin_sims)), 4) if twin_sims else None,
            "twin_sim_median": round(float(np.median(twin_sims)), 4) if twin_sims else None,
            "median_benign_sim_mean": round(float(np.mean(med_benign_sims)), 4) if med_benign_sims else None,
            "gate": {"flip_pass": bool(gate_flip), "hardness_pass": bool(gate_hard),
                     "open": bool(gate_flip and gate_hard)},
        }
        print("\n===== P5 quality gate :: {} =====".format(ds))
        print("(a) flip: orig_HARMFUL={}/{}  verified_flips={}  cond_flip={:.3f} (>=0.80 {}) "
              "retention={:.3f} regen={}".format(
                  n_orig_harm, n, n_flip, cond_flip, "PASS" if gate_flip else "FAIL",
                  retention, n_regen))
        print("(b) hardness: near-miss {}/{} = {:.3f} (>=0.50 {}) | twin_sim mean={} vs "
              "median-benign mean={}".format(
                  near_miss, n_eval, near_frac, "PASS" if gate_hard else "FAIL",
                  result[ds]["twin_sim_mean"], result[ds]["median_benign_sim_mean"]))
        print("GATE {}".format("OPEN" if result[ds]["gate"]["open"] else "CLOSED"))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\n[out] wrote", args.out)


if __name__ == "__main__":
    main()
