#!/usr/bin/env python3
"""Lexically defined validation subset with target and hostility in separate chunks."""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from hate_common import data as hdata  # noqa: E402

EN_TARGET = ("black", "white", "muslim", "jew", "jewish", "christian",
             "gay", "lesbian", "women", "woman", "men", "man", "immigrant",
             "race", "religion", "gender", "trans", "asian", "african",
             "disabled", "nationality")
EN_HOSTILE = ("hate", "kill", "stupid", "idiot", "disgusting", "inferior",
              "evil", "terrorist", "trash", "scum", "fuck", "bitch",
              "destroy", "attack", "die", "dirty")
ZH_TARGET = ("黑人", "白人", "穆斯林", "犹太", "基督", "同性恋", "女人",
             "男人", "女性", "男性", "移民", "种族", "宗教", "性别",
             "跨性别", "亚洲人", "非洲人", "残疾", "国籍")
ZH_HOSTILE = ("仇恨", "杀", "愚蠢", "白痴", "恶心", "低等", "邪恶",
              "恐怖分子", "垃圾", "去死", "肮脏", "攻击", "婊子")


def rows(path):
    return {r["video_id"]: r for r in map(json.loads, open(path))}


def asynchronous_ids(corpus):
    path = os.path.join("results", "reproduction", "asr", corpus + "_all",
                        "timestamped_chunks.jsonl")
    records = rows(path)
    target, hostile = ((ZH_TARGET, ZH_HOSTILE) if corpus == "mhclip_zh"
                       else (EN_TARGET, EN_HOSTILE))
    selected = []
    for vid in hdata.load_split(corpus, "val"):
        target_chunks, hostile_chunks, cooccurring = [], [], []
        for i, chunk in enumerate(records.get(vid, {}).get("chunks", [])):
            text = (chunk.get("text") or "").lower()
            has_target = any(term in text for term in target)
            has_hostile = any(term in text for term in hostile)
            if has_target:
                target_chunks.append(i)
            if has_hostile:
                hostile_chunks.append(i)
            if has_target and has_hostile:
                cooccurring.append(i)
        if target_chunks and hostile_chunks and not cooccurring:
            selected.append(vid)
    return selected


def metric(score_rows, gt, ids):
    ids = [v for v in ids if v in score_rows and v in gt.files]
    if not ids:
        return {"n_videos": 0, "n_frames": 0, "n_positive": 0,
                "frame_ap": None, "frame_roc": None}
    y = np.concatenate([gt[v] for v in ids])
    s = np.concatenate([score_rows[v]["score_powa"] for v in ids])
    return {"n_videos": len(ids), "n_frames": len(y), "n_positive": int(y.sum()),
            "frame_ap": float(average_precision_score(y, s)),
            "frame_roc": (float(roc_auc_score(y, s))
                          if len(np.unique(y)) == 2 else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-dir", required=True)
    ap.add_argument("--same-time-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = {"definition": "target cue and hostile cue occur in distinct ASR chunks, with no chunk containing both", "corpora": {}}
    for corpus in hdata.CORPORA:
        ids = asynchronous_ids(corpus)
        gt = np.load(os.path.join("results", "reproduction", "gt",
                                  corpus + "_val.npz"))
        full = rows(os.path.join(args.full_dir, corpus, "val_scores.jsonl"))
        same = rows(os.path.join(args.same_time_dir, corpus, "val_scores.jsonl"))
        report["corpora"][corpus] = {"video_ids": ids,
                                     "full_awb": metric(full, gt, ids),
                                     "same_time": metric(same, gt, ids)}
        gt.close()
    nonempty = [x for x in report["corpora"].values()
                if x["full_awb"]["frame_ap"] is not None]
    report["macro_over_nonempty_corpora"] = {
        "n_corpora": len(nonempty),
        "n_videos": sum(x["full_awb"]["n_videos"] for x in nonempty),
        "full_awb_frame_ap": float(np.mean(
            [x["full_awb"]["frame_ap"] for x in nonempty])),
        "same_time_frame_ap": float(np.mean(
            [x["same_time"]["frame_ap"] for x in nonempty]))}
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
