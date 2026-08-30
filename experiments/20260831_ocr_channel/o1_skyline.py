#!/usr/bin/env python3
"""O1 kill check: does the OCR channel lift the HCS supervised ceiling?

Train-span-trained TemporalConv, TEST eval, 3 seeds, with and without the OCR
channel. Gate: 4-modal HCS within-ROC >= .62.
"""
import json
import os
import sys

import numpy as np

REPO = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(REPO, "experiments", "20260830_powa_within_diagnosis"))
import skyline as S  # noqa: E402
from hate_common import data as hdata  # noqa: E402

S.FEATSETS["4modal"] = ("clip_b16_1fps", "vggish_1s", "bert_sentence_1fps",
                        "ocr_bert_1fps")
GT_TRAIN = os.path.join(REPO, "runs", "20260830_powa_within_diagnosis",
                        "gt_train_diagnosis_only")
OUT = os.path.join(REPO, "runs", "20260831_ocr_channel")
SEEDS = (234, 2025, 3407)


def main():
    orig = hdata.gt_arrays

    def patched(corpus, split="test"):
        if split == "val":
            with np.load(os.path.join(GT_TRAIN, corpus + "_train.npz")) as z:
                return {k: z[k] for k in z.files}
        return orig(corpus, split)

    hdata.gt_arrays = patched
    S.hdata.gt_arrays = patched

    results = {}
    for featset in ("clip+vgg+bert", "4modal"):
        per_seed = []
        for seed in SEEDS:
            S.SEED = seed
            per_seed.append(S.run("hateclipseg", featset, "tconv"))
            print(featset, seed, json.dumps(per_seed[-1]), flush=True)
        results[featset] = {
            "within_mean": float(np.mean([p["within_roc_macro"] for p in per_seed])),
            "within_sd": float(np.std([p["within_roc_macro"] for p in per_seed])),
            "frame_ap_mean": float(np.mean([p["frame_ap"] for p in per_seed])),
            "seeds": per_seed}
    with open(os.path.join(OUT, "o1_skyline.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    for k, v in results.items():
        print("%s: within %.4f±%.3f frameAP %.4f" % (
            k, v["within_mean"], v["within_sd"], v["frame_ap_mean"]))
    gate = results["4modal"]["within_mean"]
    print("O1 GATE (>= .62):", "PASS" if gate >= .62 else "FAIL", "%.4f" % gate)


if __name__ == "__main__":
    main()
