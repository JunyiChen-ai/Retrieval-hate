#!/usr/bin/env python3
"""True supervised ceiling: rasterize TRAIN spans (diagnosis only) and retrain
the skyline probes on them. Train frame arrays are written under runs/ so the
frozen protocol gt/ tree stays untouched; weak methods must never load them.
"""
import ast
import csv
import json
import os
import sys

import numpy as np

REPO = "/home/jehc223/Retrieval-hate"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "scripts", "duplex"))
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
sys.path.insert(0, HERE)

from frame_eval_common import build_gt_array  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from mhclip_span_gold import parse_spans  # noqa: E402
import skyline  # noqa: E402

OUT_DIR = os.path.join(REPO, "runs", "20260830_powa_within_diagnosis")
GT_DIR = os.path.join(OUT_DIR, "gt_train_diagnosis_only")
VGGISH = os.path.join(REPO, "results", "reproduction", "features", "vggish_1s")


def duration(corpus, vid):
    return float(np.load(os.path.join(VGGISH, corpus, vid + ".npy"),
                         mmap_mode="r").shape[0])


def clock(text):
    fields = [float(x) for x in str(text).split(":")]
    return sum(value * 60 ** power for power, value in enumerate(fields[::-1]))


def hatemm():
    ids = set(hdata.load_split("hatemm", "train"))
    arrays = {}
    with open(os.path.join(REPO, "data/gt/HateMM/HateMM_annotation.csv"),
              newline="") as fh:
        for row in csv.DictReader(fh):
            vid = row["video_file_name"].rsplit(".", 1)[0]
            if vid not in ids:
                continue
            raw = row["hate_snippet"].strip()
            spans = ([(clock(a), clock(b)) for a, b in ast.literal_eval(raw)]
                     if raw else [])
            try:
                arrays[vid] = build_gt_array(spans, duration("hatemm", vid))
            except FileNotFoundError:
                continue
    return arrays


def mhclip(corpus, language):
    ids = set(hdata.load_split(corpus, "train"))
    arrays = {}
    with open(os.path.join(REPO, "data/gt/mhc_votes/mhc_%s_train.tsv" % language),
              newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            vid = row["Video_ID"]
            if vid not in ids:
                continue
            positive = row["Majority_Voting"] in ("Hateful", "Offensive")
            spans = parse_spans(row["Duration"]) if positive else []
            if positive and not spans:
                continue
            try:
                arrays[vid] = build_gt_array(spans, duration(corpus, vid))
            except FileNotFoundError:
                continue
    return arrays


def hateclipseg():
    ids = set(hdata.load_split("hateclipseg", "train"))
    arrays = {}
    csv.field_size_limit(1 << 30)
    with open("/home/jehc223/data/HateClipSeg/Dataset/"
              "segment_level_annotation.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vid = row["Video Id"].strip()
            if vid not in ids:
                continue
            labels = ast.literal_eval(row["Segment-Level Label"])
            spans = ast.literal_eval(row["Segment Timestamp"])
            if len(labels) != len(spans):
                continue
            positive = [(float(a), float(b)) for lab, (a, b) in zip(labels, spans)
                        if any(int(x) == 1 for x in lab[1:6])]
            try:
                arrays[vid] = build_gt_array(positive,
                                             duration("hateclipseg", vid))
            except FileNotFoundError:
                continue
    return arrays


def main():
    os.makedirs(GT_DIR, exist_ok=True)
    groups = {"hatemm": hatemm(),
              "mhclip_en": mhclip("mhclip_en", "English"),
              "mhclip_zh": mhclip("mhclip_zh", "Chinese"),
              "hateclipseg": hateclipseg()}
    for corpus, arrays in groups.items():
        np.savez_compressed(os.path.join(GT_DIR, corpus + "_train.npz"), **arrays)
        print(corpus, "train videos:", len(arrays),
              "pos rate: %.3f" % (sum(int(x.sum()) for x in arrays.values()) /
                                  max(1, sum(len(x) for x in arrays.values()))))

    # patch gt_arrays so skyline.run() trains on the train rasterization
    orig = hdata.gt_arrays

    def patched(corpus, split="test"):
        if split == "val":
            with np.load(os.path.join(GT_DIR, corpus + "_train.npz")) as z:
                return {k: z[k] for k in z.files}
        return orig(corpus, split)

    hdata.gt_arrays = patched
    skyline.hdata.gt_arrays = patched

    results = {}
    for corpus in skyline.CORPORA:
        for featset in skyline.FEATSETS:
            for arch in ("linear", "tconv"):
                key = f"{corpus}/{featset}/{arch}"
                results[key] = skyline.run(corpus, featset, arch)
                print(key, json.dumps(results[key]))
    with open(os.path.join(OUT_DIR, "skyline_train.json"), "w") as fh:
        json.dump({"note": "trained on TRAIN span rasterization (diagnosis only), "
                           "evaluated on TEST", "results": results}, fh, indent=1)
    lines = ["# Supervised skyline v2 (train-span-trained, TEST eval)", "",
             "| corpus | features | arch | frame AP | frame ROC | within-ROC macro (n) |",
             "|---|---|---|---:|---:|---:|"]
    for key, r in results.items():
        c, f, a = key.split("/")
        lines.append("| %s | %s | %s | %.4f | %.4f | %.4f (%d) |" % (
            c, f, a, r["frame_ap"], r["frame_roc"],
            r["within_roc_macro"], r["within_n"]))
    with open(os.path.join(OUT_DIR, "skyline_train.md"), "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
