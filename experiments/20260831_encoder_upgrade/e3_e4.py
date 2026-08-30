#!/usr/bin/env python3
"""E3 ceiling probes + E4 LOCO-ST on the CLIP-L stack (hatemm & hcs only).

E3: train-span skyline (3 seeds) with l14+vggish+bert; gates HCS >= .62,
HateMM >= .77 (at least one must pass to run E4 for the passing corpora).
E4: valsel + loo_zero, single aux source (the other corpus), 5 seeds, dense
test scores saved as l14_valsel/_zero, appended to scale_results.json.
"""
import json
import os
import sys

import numpy as np

REPO = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(REPO, "experiments", "20260830_powa_within_diagnosis"))
sys.path.insert(0, os.path.join(REPO, "experiments", "20260830_spantransfer_pilot"))
import skyline as S  # noqa: E402
from hate_common import data as hdata  # noqa: E402

L14SET = ("clip_l14_336_1fps", "vggish_1s", "bert_sentence_1fps")
S.FEATSETS["l14"] = L14SET
GT_TRAIN = os.path.join(REPO, "runs", "20260830_powa_within_diagnosis",
                        "gt_train_diagnosis_only")
RUN_DIR = os.path.join(REPO, "runs", "20260831_encoder_upgrade")
PAIR = {"hatemm": "hateclipseg", "hateclipseg": "hatemm"}
SEEDS3 = (234, 2025, 3407)
SEEDS5 = (234, 2025, 3407, 42, 20260830)
GATES = {"hatemm": .77, "hateclipseg": .62}


def patch_gt():
    orig = hdata.gt_arrays

    def patched(corpus, split="test"):
        if split == "__train_spans__":
            with np.load(os.path.join(GT_TRAIN, corpus + "_train.npz")) as z:
                return {k: z[k] for k in z.files}
        return orig(corpus, split)

    hdata.gt_arrays = patched
    S.hdata.gt_arrays = patched
    return patched


def e3():
    # skyline.run trains on split "val"; patch it to train spans for E3 only
    orig = hdata.gt_arrays

    def patched(corpus, split="test"):
        if split == "val":
            with np.load(os.path.join(GT_TRAIN, corpus + "_train.npz")) as z:
                return {k: z[k] for k in z.files}
        return orig(corpus, split)

    hdata.gt_arrays = patched
    S.hdata.gt_arrays = patched
    out = {}
    for corpus in ("hateclipseg", "hatemm"):
        ws = []
        for seed in SEEDS3:
            S.SEED = seed
            r = S.run(corpus, "l14", "tconv")
            ws.append(r["within_roc_macro"])
            print("E3", corpus, seed, json.dumps(r), flush=True)
        out[corpus] = {"within_mean": float(np.mean(ws)),
                       "within_sd": float(np.std(ws)),
                       "gate": GATES[corpus],
                       "pass": bool(np.mean(ws) >= GATES[corpus])}
        print("E3 %s: %.4f±%.3f gate %.2f -> %s" % (
            corpus, out[corpus]["within_mean"], out[corpus]["within_sd"],
            GATES[corpus], "PASS" if out[corpus]["pass"] else "FAIL"),
            flush=True)
    hdata.gt_arrays = orig
    S.hdata.gt_arrays = orig
    with open(os.path.join(RUN_DIR, "e3_ceilings.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def e4(targets):
    import spantransfer as ST
    import scale_up as SU
    ST.DIRS = L14SET
    SU  # noqa
    blob_path = SU.RESULTS
    blob = json.load(open(blob_path))
    res = blob["results"]
    for target in targets:
        for arm in ("l14_valsel", "l14_zero"):
            key = "%s/%s/5seed" % (target, arm)
            if key in res:
                continue
            per_seed = []
            for s in SEEDS5:
                if arm == "l14_valsel":
                    out = SU.run_valsel(target, s, arm=arm,
                                        sources=[PAIR[target]], save=True)
                else:
                    out = SU.run_zero(target, s, sources=[PAIR[target]],
                                      arm=arm, save=True)
                per_seed.append(out)
            res[key] = SU.agg(per_seed)
            with open(blob_path, "w") as fh:
                json.dump(blob, fh, indent=1)
            print("E4", key, json.dumps(res[key]), flush=True)
    print("E4_DONE", flush=True)


def main():
    ceilings = e3()
    targets = [c for c in ("hateclipseg", "hatemm") if ceilings[c]["pass"]]
    if not targets:
        print("E3 BOTH FAIL -> round killed", flush=True)
        return
    e4(targets)


if __name__ == "__main__":
    main()
