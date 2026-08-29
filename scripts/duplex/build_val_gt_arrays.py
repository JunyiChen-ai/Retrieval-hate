#!/usr/bin/env python3
"""Build 1-fps frame gold for frozen official validation manifests."""

import ast
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))

from frame_eval_common import build_gt_array  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from mhclip_span_gold import parse_spans  # noqa: E402

OUT = os.path.join(ROOT, "results", "reproduction", "gt")
VGGISH = os.path.join(ROOT, "results", "reproduction", "features", "vggish_1s")


def duration(corpus, vid):
    return float(np.load(os.path.join(VGGISH, corpus, vid + ".npy"),
                         mmap_mode="r").shape[0])


def clock(text):
    fields = [float(x) for x in str(text).split(":")]
    return sum(value * 60 ** power for power, value in enumerate(fields[::-1]))


def hatemm():
    ids = set(hdata.load_split("hatemm", "val"))
    path = "/home/jehc223/Retrieval-hate/data/gt/HateMM/HateMM_annotation.csv"
    arrays = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            vid = row["video_file_name"].rsplit(".", 1)[0]
            if vid not in ids:
                continue
            raw = row["hate_snippet"].strip()
            spans = ([(clock(a), clock(b)) for a, b in ast.literal_eval(raw)]
                     if raw else [])
            arrays[vid] = build_gt_array(spans, duration("hatemm", vid))
    return arrays


def mhclip(corpus, language):
    ids = set(hdata.load_split(corpus, "val"))
    path = ("/home/jehc223/Retrieval-hate/data/gt/mhc_votes/"
            "mhc_%s_valid.tsv" % language)
    arrays = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            vid = row["Video_ID"]
            if vid not in ids:
                continue
            positive = row["Majority_Voting"] in ("Hateful", "Offensive")
            spans = parse_spans(row["Duration"]) if positive else []
            # Same test protocol: a positive without a temporal span has no
            # localization gold and is excluded rather than made negative.
            if positive and not spans:
                continue
            arrays[vid] = build_gt_array(spans, duration(corpus, vid))
    return arrays


def hateclipseg():
    ids = set(hdata.load_split("hateclipseg", "val"))
    path = "/home/jehc223/data/HateClipSeg/Dataset/segment_level_annotation.csv"
    arrays = {}
    csv.field_size_limit(1 << 30)
    with open(path, newline="", encoding="utf-8") as fh:
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
            arrays[vid] = build_gt_array(positive,
                                         duration("hateclipseg", vid))
    return arrays


def main():
    os.makedirs(OUT, exist_ok=True)
    groups = {"hatemm": hatemm(),
              "mhclip_en": mhclip("mhclip_en", "English"),
              "mhclip_zh": mhclip("mhclip_zh", "Chinese"),
              "hateclipseg": hateclipseg()}
    report = {}
    for corpus, arrays in groups.items():
        np.savez_compressed(os.path.join(OUT, corpus + "_val.npz"), **arrays)
        report[corpus] = {"videos": len(arrays),
                          "frames": sum(len(x) for x in arrays.values()),
                          "positive_frames": sum(int(x.sum()) for x in arrays.values())}
    with open(os.path.join(OUT, "validation_gold_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
