#!/usr/bin/env python
"""P8b probe gate (CPU) — B_vision vs A(floor) and C(naive-trunc).

Reuses P8's exact rep() + loo_knn() so the numbers are apples-to-apples with the
text-only P8 gate. TRAIN split, LOO kNN over [l2n(img)|l2n(text)] @k20.

Gate (higher bar than P8-EN, per team-lead): B_vision must beat BOTH
  A (floor raw chunk-mean) AND C (first-70-token naive truncation)
on the ZH train probe before any training.
"""
import argparse
import json
import os
import sys

import torch

ROOT = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis"))
from p8_probe_gate import rep, loo_knn, MODEL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="MHC_zh")
    ap.add_argument("--vsum_tag", default="p8vsum", help="p8vsum=P8b English, p8vsumzh=P8c CN.")
    ap.add_argument("--label", default="B_vision", help="condition label for the vsum arm.")
    ap.add_argument("--out", default="scripts/analysis/p8_out/p8b_probe_gate.json")
    a = ap.parse_args()
    result = {}
    for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
        ds_dir = os.path.join(ROOT, "data/CLIP_Embedding", ds)
        floor = torch.load(os.path.join(ds_dir, "train_{}.pt".format(MODEL)), map_location="cpu")
        img = floor["img_feats"]
        labels = floor["labels"].long().numpy()
        texts = {"A": floor["text_feats"]}
        for tag, cond in (("p8trunc", "C"), (a.vsum_tag, a.label)):
            p = os.path.join(ds_dir, "train_{}_HF.pt".format(tag))
            if not os.path.exists(p):
                print("[WARN] missing {}".format(p)); texts[cond] = None; continue
            texts[cond] = torch.load(p, map_location="cpu")["text_feats"]
        conds = {}
        for cond, tf in texts.items():
            if tf is None:
                continue
            x = rep(img, tf)
            acc20, mf20 = loo_knn(x, labels, 20)
            kc = {k: loo_knn(x, labels, k)[0] for k in (1, 5, 10)}
            conds[cond] = {"acc_k20": acc20, "macro_f1_k20": mf20, "acc_k1_5_10": kc}
        a_acc = conds.get("A", {}).get("acc_k20")
        c_acc = conds.get("C", {}).get("acc_k20")
        b_acc = conds.get(a.label, {}).get("acc_k20")
        beats_A = (b_acc is not None and a_acc is not None and b_acc >= a_acc)
        beats_C = (b_acc is not None and c_acc is not None and b_acc >= c_acc)
        gate = bool(beats_A and beats_C)
        result[ds] = {"n": int(len(labels)), "pos": int(labels.sum()), "conds": conds,
                      "{}_beats_A".format(a.label): bool(beats_A),
                      "{}_beats_C".format(a.label): bool(beats_C),
                      "GATE_open": gate}
        print("\n===== P8b/c probe :: {} (n={}, pos={}) =====".format(ds, len(labels), int(labels.sum())))
        for cond in ("A", "C", a.label):
            if cond in conds:
                c = conds[cond]
                print("  {:11s}: acc@k20={} macroF1={} | k1/5/10={}".format(
                    cond, c["acc_k20"], c["macro_f1_k20"], c["acc_k1_5_10"]))
        print("  {0} >= A: {1}  |  {0} >= C(0.791 bar): {2}  ->  GATE {3}".format(
            a.label, beats_A, beats_C, "OPEN" if gate else "CLOSED"))
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    with open(os.path.join(ROOT, a.out), "w") as f:
        json.dump(result, f, indent=2)
    print("\n[out] wrote", a.out)


if __name__ == "__main__":
    main()
