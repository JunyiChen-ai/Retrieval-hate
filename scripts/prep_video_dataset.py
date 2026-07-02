#!/usr/bin/env python
"""Prepare hateful-video datasets into the RGCL ground-truth + media layout.

Config-driven generalization of scripts/prep_mhc.py. For each configured
dataset it writes data/gt/<NAME>/<split>.jsonl (one JSON object per line) and,
for datasets whose videos live on local disk, creates absolute symlinks under
data/video/<NAME>/All/<id>.mp4.

For datasets whose videos live on Backblaze B2 (not local), it instead writes
data/video/<NAME>/_id2b2path.tsv mapping each kept id to its B2 relative path,
and creates an empty data/video/<NAME>/All directory (filled at mount time).

Record schema (identical to prep_mhc.py):
  {"id": <Video_ID>,
   "text": (Title.strip() + " . " + Transcript.strip()).strip(),
   "label": <int 0/1>}
  - if Title empty -> just Transcript
  - if both empty  -> a single space " "

CPU only. No GPU / SLURM. Deterministic across reruns.
"""
import argparse
import json
import os
from collections import Counter

# ----------------------------------------------------------------------------
# Fixed repo paths
# ----------------------------------------------------------------------------
REPO_ROOT = "/data/jehc223/RGCL"
GT_ROOT = os.path.join(REPO_ROOT, "data", "gt")
VIDEO_ROOT = os.path.join(REPO_ROOT, "data", "video")


# ----------------------------------------------------------------------------
# Dataset configs
# ----------------------------------------------------------------------------
# Each config:
#   name        : output dataset name (data/gt/<name>, data/video/<name>)
#   src_root    : source dataset root
#   annotation  : path to annotation(new).json
#   splits_dir  : dir with newline-separated Video_ID split files (no header)
#   split_files : {train/val/test -> filename}
#   label_map   : {annotation Label str -> int 0/1}
#   video_mode  : "local"  -> require <video_dir>/<id>.mp4, symlink into All/
#                 "b2"     -> no local check; emit _id2b2path.tsv mapping
#   video_dir   : (local mode) dir holding <id>.mp4 source files
#   b2_prefix_folders : (b2 mode) {id-prefix -> B2 folder name} chosen by
#                       the substring before the first "_" in the Video_ID
DATASETS = {
    "HateMM": {
        "name": "HateMM",
        "src_root": "/data/jehc223/HateMM",
        "annotation": "/data/jehc223/HateMM/annotation(new).json",
        "splits_dir": "/data/jehc223/HateMM/splits",
        "split_files": {
            "train": "train_clean.csv",
            "val": "valid.csv",
            "test": "test_clean.csv",
        },
        "label_map": {"Hate": 1, "Non Hate": 0},
        "video_mode": "local",
        "video_dir": "/data/jehc223/HateMM/video",
    },
    "MHC_zh": {
        "name": "MHC_zh",
        "src_root": "/data/jehc223/Multihateclip/Chinese",
        "annotation": "/data/jehc223/Multihateclip/Chinese/annotation(new).json",
        "splits_dir": "/data/jehc223/Multihateclip/Chinese/splits",
        "split_files": {
            "train": "train_clean.csv",
            "val": "valid.csv",
            "test": "test_clean.csv",
        },
        "label_map": {"Hateful": 1, "Offensive": 1, "Normal": 0},
        "video_mode": "local",
        "video_dir": "/data/jehc223/Multihateclip/Chinese/video",
    },
    "ImpliHateVid": {
        "name": "ImpliHateVid",
        "src_root": "/data/jehc223/ImpliHateVid",
        "annotation": "/data/jehc223/ImpliHateVid/annotation(new).json",
        "splits_dir": "/data/jehc223/ImpliHateVid/splits",
        "split_files": {
            "train": "train_clean.csv",
            "val": "val.csv",
            "test": "test_clean.csv",
        },
        "label_map": {"Hateful": 1, "Normal": 0},
        "video_mode": "b2",
        "b2_prefix_folders": {
            "EX": "Explicit Hate Videos",
            "IM": "Implicit Hate Videos",
            "NH": "Non Hate Videos",
        },
    },
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_annotation(path):
    with open(path, "r") as f:
        data = json.load(f)
    by_id = {}
    for entry in data:
        vid = entry.get("Video_ID")
        if vid is not None:
            by_id[vid] = entry
    return by_id


def read_split_ids(splits_dir, filename):
    """Read newline-separated Video_IDs (no header), preserving order."""
    path = os.path.join(splits_dir, filename)
    ids = []
    with open(path, "r") as f:
        for line in f:
            vid = line.strip()
            if vid:
                ids.append(vid)
    return ids


def build_text(entry):
    title = (entry.get("Title") or "").strip()
    transcript = (entry.get("Transcript") or "").strip()
    if title and transcript:
        text = (title + " . " + transcript).strip()
    elif title:
        text = title
    elif transcript:
        text = transcript
    else:
        text = " "
    if not text:
        text = " "
    return text


def force_symlink(src, dst):
    """Create an absolute symlink dst -> src, overwriting any existing one."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def label_dist(records):
    return dict(sorted(Counter(r["label"] for r in records).items()))


def b2_path_for_id(vid, prefix_folders):
    prefix = vid.split("_", 1)[0]
    folder = prefix_folders.get(prefix)
    if folder is None:
        return None
    return folder + "/" + vid + ".mp4"


# ----------------------------------------------------------------------------
# Core build
# ----------------------------------------------------------------------------
def build_split_records(cfg, split_name, ann):
    """Return (kept_records, dropped_missing_ann, dropped_missing_video).

    kept_records carry internal fields (_mp4 for local, _b2 for b2) that are
    stripped before jsonl is written.
    """
    filename = cfg["split_files"][split_name]
    ids = read_split_ids(cfg["splits_dir"], filename)
    label_map = cfg["label_map"]
    mode = cfg["video_mode"]

    records = []
    dropped_ann = []
    dropped_video = []
    for vid in ids:
        if vid not in ann:
            dropped_ann.append(vid)
            continue
        entry = ann[vid]
        rec = {
            "id": vid,
            "text": build_text(entry),
            "label": int(label_map[entry["Label"]]),
        }
        if mode == "local":
            mp4 = os.path.join(cfg["video_dir"], vid + ".mp4")
            if not os.path.exists(mp4):
                dropped_video.append(vid)
                continue
            rec["_mp4"] = mp4
        elif mode == "b2":
            b2 = b2_path_for_id(vid, cfg["b2_prefix_folders"])
            if b2 is None:
                # unknown prefix -> cannot map to a B2 folder
                dropped_video.append(vid)
                continue
            rec["_b2"] = b2
        records.append(rec)
    return records, dropped_ann, dropped_video


def emit_dataset(cfg, splits):
    """Write jsonl + media artifacts for a dataset. splits: {split: [records]}.

    Returns a dict of created info for the summary.
    """
    name = cfg["name"]
    mode = cfg["video_mode"]
    created_paths = []
    all_ids = {}  # id -> internal media info (dedup across splits, first wins)

    for split_name, records in splits.items():
        gt_path = os.path.join(GT_ROOT, name, split_name + ".jsonl")
        clean = [
            {"id": r["id"], "text": r["text"], "label": r["label"]} for r in records
        ]
        write_jsonl(gt_path, clean)
        created_paths.append(gt_path)
        for r in records:
            if r["id"] not in all_ids:
                all_ids[r["id"]] = r

    link_dir = os.path.join(VIDEO_ROOT, name, "All")
    info = {"created_paths": created_paths, "link_dir": link_dir}

    if mode == "local":
        n_links = 0
        for vid, r in all_ids.items():
            force_symlink(r["_mp4"], os.path.join(link_dir, vid + ".mp4"))
            n_links += 1
        info["n_links"] = n_links
        info["tsv_path"] = None
    elif mode == "b2":
        # empty All/ dir (filled at mount time), plus id->b2path mapping tsv
        os.makedirs(link_dir, exist_ok=True)
        tsv_path = os.path.join(VIDEO_ROOT, name, "_id2b2path.tsv")
        # deterministic order: sort by id
        with open(tsv_path, "w") as f:
            for vid in sorted(all_ids.keys()):
                f.write(vid + "\t" + all_ids[vid]["_b2"] + "\n")
        info["n_links"] = 0
        info["tsv_path"] = tsv_path
        info["n_tsv"] = len(all_ids)

    return info


# ----------------------------------------------------------------------------
def process_dataset(cfg):
    name = cfg["name"]
    print("\n" + "#" * 74)
    print(f"# DATASET: {name}  (mode={cfg['video_mode']})")
    print("#" * 74)
    print(f"Label map: {cfg['label_map']}")
    ann = load_annotation(cfg["annotation"])
    print(f"Loaded annotation: {len(ann)} entries")

    splits = {}
    dropped = {}
    for split_name in ("train", "val", "test"):
        recs, d_ann, d_vid = build_split_records(cfg, split_name, ann)
        splits[split_name] = recs
        dropped[split_name] = {"missing_ann": d_ann, "missing_video": d_vid}

    info = emit_dataset(cfg, splits)

    # --- per-split summary ---
    print("-" * 70)
    for split_name in ("train", "val", "test"):
        recs = splits[split_name]
        d = dropped[split_name]
        n_in = len(read_split_ids(cfg["splits_dir"], cfg["split_files"][split_name]))
        drop_key = "dropped_missing_video" if cfg["video_mode"] == "local" \
            else "dropped_unknown_prefix"
        print(
            f"  {split_name:5s} [{cfg['split_files'][split_name]}]: "
            f"input={n_in} kept={len(recs)} "
            f"dropped_missing_ann={len(d['missing_ann'])} "
            f"{drop_key}={len(d['missing_video'])} "
            f"labels={label_dist(recs)}"
        )
        if d["missing_ann"]:
            print(f"        dropped(no annotation): {d['missing_ann']}")
        if d["missing_video"]:
            tag = "no mp4" if cfg["video_mode"] == "local" else "unknown prefix"
            print(f"        dropped({tag}): {d['missing_video']}")

    if cfg["video_mode"] == "local":
        print(f"  symlinks created under {info['link_dir']}: {info['n_links']}")
    else:
        print(f"  empty dir created: {info['link_dir']}")
        print(f"  tsv written: {info['tsv_path']}  ({info['n_tsv']} lines)")
    print(f"  gt files: {info['created_paths']}")

    return splits, info


def verify_dataset(cfg, splits, info):
    name = cfg["name"]
    print("\n" + "=" * 70)
    print(f"VERIFY: {name}")
    print("=" * 70)
    train_jsonl = os.path.join(GT_ROOT, name, "train.jsonl")
    print(f"First 2 lines of {train_jsonl}:")
    with open(train_jsonl) as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            print("  " + line.rstrip())

    if cfg["video_mode"] == "local":
        if splits["train"]:
            sample_id = splits["train"][0]["id"]
            sample_link = os.path.join(info["link_dir"], sample_id + ".mp4")
            print(
                f"Symlink check: {sample_link}\n"
                f"  islink={os.path.islink(sample_link)} "
                f"exists(resolves)={os.path.exists(sample_link)} "
                f"-> {os.path.realpath(sample_link)}"
            )
    else:
        tsv_path = info["tsv_path"]
        with open(tsv_path) as f:
            lines = [ln.rstrip("\n") for ln in f]
        print(f"_id2b2path.tsv total lines: {len(lines)}")
        # print a sample line for each prefix (EX_, IM_, NH_)
        print("Sample tsv lines (one per prefix):")
        seen = set()
        for ln in lines:
            vid = ln.split("\t", 1)[0]
            pref = vid.split("_", 1)[0]
            if pref not in seen:
                seen.add(pref)
                print("  " + ln)
            if len(seen) >= len(cfg["b2_prefix_folders"]):
                break


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
        help="which dataset configs to build",
    )
    args = ap.parse_args()

    results = {}
    for ds in args.datasets:
        cfg = DATASETS[ds]
        splits, info = process_dataset(cfg)
        results[ds] = (cfg, splits, info)

    for ds in args.datasets:
        cfg, splits, info = results[ds]
        verify_dataset(cfg, splits, info)

    # --- global summary ---
    print("\n" + "=" * 70)
    print("GLOBAL SUMMARY")
    print("=" * 70)
    for ds in args.datasets:
        cfg, splits, info = results[ds]
        total = sum(len(v) for v in splits.values())
        line = f"  {cfg['name']:14s}: total_kept={total}"
        if cfg["video_mode"] == "b2":
            line += f"  tsv_lines={info['n_tsv']}"
        else:
            line += f"  symlinks={info['n_links']}"
        print(line)
        for split_name in ("train", "val", "test"):
            print(
                f"      {split_name:5s}: kept={len(splits[split_name])} "
                f"labels={label_dist(splits[split_name])}"
            )


if __name__ == "__main__":
    main()
