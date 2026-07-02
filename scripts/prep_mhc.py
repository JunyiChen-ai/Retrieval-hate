#!/usr/bin/env python
"""Prepare MultiHateClip-English into the RGCL ground-truth + media layout.

Builds two dataset names under data/:
  - MHC      : full dataset (uses *_clean.csv subsets where available)
  - MHCsmoke : tiny class-balanced subset for end-to-end smoke tests

For each split it writes data/gt/<NAME>/<split>.jsonl (one JSON object per
line) and creates absolute symlinks under data/video/<NAME>/All/<id>.mp4.

CPU only. No GPU / SLURM. Deterministic across reruns.
"""
import argparse
import json
import os
import random
from collections import Counter

# ----------------------------------------------------------------------------
# Fixed paths
# ----------------------------------------------------------------------------
REPO_ROOT = "/data/jehc223/RGCL"
SRC_ROOT = "/data/jehc223/Multihateclip/English"
ANNOTATION = os.path.join(SRC_ROOT, "annotation(new).json")
SPLITS_DIR = os.path.join(SRC_ROOT, "splits")
VIDEO_MP4_DIR = os.path.join(SRC_ROOT, "video_mp4")

GT_ROOT = os.path.join(REPO_ROOT, "data", "gt")
VIDEO_ROOT = os.path.join(REPO_ROOT, "data", "video")

# split-file -> output split name.  Prefer the *_clean.csv where it exists.
SPLIT_FILES = {
    "train": "train_clean.csv",
    "val": "valid.csv",
    "test": "test_clean.csv",
}

LABEL_SCHEMES = {
    "harmful_vs_normal": {"Hateful": 1, "Offensive": 1, "Normal": 0},
    "hateful_vs_rest": {"Hateful": 1, "Offensive": 0, "Normal": 0},
}

SMOKE_SIZES = {"train": (8, 8), "val": (4, 4), "test": (4, 4)}  # (n_label1, n_label0)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_annotation():
    with open(ANNOTATION, "r") as f:
        data = json.load(f)
    by_id = {}
    for entry in data:
        vid = entry.get("Video_ID")
        if vid is not None:
            by_id[vid] = entry
    return by_id


def read_split_ids(filename):
    """Read newline-separated Video_IDs (no header), preserving order."""
    path = os.path.join(SPLITS_DIR, filename)
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


# ----------------------------------------------------------------------------
# Core build
# ----------------------------------------------------------------------------
def build_split_records(split_name, ann, label_map):
    """Return (kept_records, dropped_missing_ann, dropped_missing_video)."""
    filename = SPLIT_FILES[split_name]
    ids = read_split_ids(filename)
    records = []
    dropped_ann = []
    dropped_video = []
    for vid in ids:
        if vid not in ann:
            dropped_ann.append(vid)
            continue
        mp4 = os.path.join(VIDEO_MP4_DIR, vid + ".mp4")
        if not os.path.exists(mp4):
            dropped_video.append(vid)
            continue
        entry = ann[vid]
        label = label_map[entry["Label"]]
        records.append(
            {
                "id": vid,
                "text": build_text(entry),
                "label": int(label),
                "_mp4": mp4,  # internal, stripped before writing
            }
        )
    return records, dropped_ann, dropped_video


def emit_dataset(name, splits, dropped_info):
    """Write jsonl + symlinks for a dataset name. splits: {split: [records]}."""
    created_paths = []
    all_link_targets = {}  # id -> mp4 (dedup across splits)
    for split_name, records in splits.items():
        gt_path = os.path.join(GT_ROOT, name, split_name + ".jsonl")
        clean = [{"id": r["id"], "text": r["text"], "label": r["label"]} for r in records]
        write_jsonl(gt_path, clean)
        created_paths.append(gt_path)
        for r in records:
            all_link_targets[r["id"]] = r["_mp4"]

    # symlinks (one per unique id across all splits)
    link_dir = os.path.join(VIDEO_ROOT, name, "All")
    for vid, mp4 in all_link_targets.items():
        force_symlink(mp4, os.path.join(link_dir, vid + ".mp4"))

    return created_paths, link_dir, len(all_link_targets)


def label_dist(records):
    return dict(sorted(Counter(r["label"] for r in records).items()))


def build_smoke(mhc_splits):
    """Class-balanced disjoint-where-possible subset drawn from kept MHC ids."""
    rng = random.Random(42)
    used = set()
    smoke = {}
    for split_name in ("train", "val", "test"):
        n1, n0 = SMOKE_SIZES[split_name]
        recs = mhc_splits[split_name]
        pos = sorted([r for r in recs if r["label"] == 1], key=lambda r: r["id"])
        neg = sorted([r for r in recs if r["label"] == 0], key=lambda r: r["id"])
        rng.shuffle(pos)
        rng.shuffle(neg)

        def pick(pool, n):
            chosen = []
            # prefer unused ids
            for r in pool:
                if len(chosen) >= n:
                    break
                if r["id"] not in used:
                    chosen.append(r)
            # backfill if not enough unique-across-splits ids available
            if len(chosen) < n:
                for r in pool:
                    if len(chosen) >= n:
                        break
                    if r not in chosen:
                        chosen.append(r)
            for r in chosen:
                used.add(r["id"])
            return chosen

        chosen = pick(pos, n1) + pick(neg, n0)
        chosen = sorted(chosen, key=lambda r: r["id"])
        smoke[split_name] = chosen
    return smoke


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--label_scheme",
        default="harmful_vs_normal",
        choices=list(LABEL_SCHEMES.keys()),
    )
    args = ap.parse_args()
    label_map = LABEL_SCHEMES[args.label_scheme]

    print(f"Label scheme: {args.label_scheme} -> {label_map}")
    ann = load_annotation()
    print(f"Loaded annotation: {len(ann)} entries")

    # ----- build MHC -----
    mhc_splits = {}
    mhc_dropped = {}
    for split_name in ("train", "val", "test"):
        recs, d_ann, d_vid = build_split_records(split_name, ann, label_map)
        mhc_splits[split_name] = recs
        mhc_dropped[split_name] = {"missing_ann": d_ann, "missing_video": d_vid}

    mhc_paths, mhc_link_dir, mhc_links = emit_dataset("MHC", mhc_splits, mhc_dropped)

    # ----- build MHCsmoke -----
    smoke_splits = build_smoke(mhc_splits)
    smoke_paths, smoke_link_dir, smoke_links = emit_dataset("MHCsmoke", smoke_splits, None)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY: MHC (full)")
    print("=" * 70)
    for split_name in ("train", "val", "test"):
        recs = mhc_splits[split_name]
        d = mhc_dropped[split_name]
        n_in = len(read_split_ids(SPLIT_FILES[split_name]))
        print(
            f"  {split_name:5s} [{SPLIT_FILES[split_name]}]: "
            f"input={n_in} kept={len(recs)} "
            f"dropped_missing_ann={len(d['missing_ann'])} "
            f"dropped_missing_video={len(d['missing_video'])} "
            f"labels={label_dist(recs)}"
        )
        if d["missing_ann"]:
            print(f"        dropped(no annotation): {d['missing_ann']}")
        if d["missing_video"]:
            print(f"        dropped(no mp4): {d['missing_video']}")
    print(f"  symlinks created under {mhc_link_dir}: {mhc_links}")
    print(f"  gt files: {mhc_paths}")

    print("\n" + "=" * 70)
    print("SUMMARY: MHCsmoke")
    print("=" * 70)
    for split_name in ("train", "val", "test"):
        recs = smoke_splits[split_name]
        print(
            f"  {split_name:5s}: kept={len(recs)} labels={label_dist(recs)} "
            f"ids={[r['id'] for r in recs]}"
        )
    print(f"  symlinks created under {smoke_link_dir}: {smoke_links}")
    print(f"  gt files: {smoke_paths}")

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERIFY")
    print("=" * 70)
    train_jsonl = os.path.join(GT_ROOT, "MHC", "train.jsonl")
    print(f"First 2 lines of {train_jsonl}:")
    with open(train_jsonl) as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            print("  " + line.rstrip())

    # confirm a symlink resolves
    sample_id = mhc_splits["train"][0]["id"]
    sample_link = os.path.join(mhc_link_dir, sample_id + ".mp4")
    print(
        f"Symlink check: {sample_link}\n"
        f"  islink={os.path.islink(sample_link)} "
        f"exists(resolves)={os.path.exists(sample_link)} "
        f"-> {os.path.realpath(sample_link)}"
    )

    total_mhc = sum(len(v) for v in mhc_splits.values())
    total_smoke = sum(len(v) for v in smoke_splits.values())
    print(f"Total kept records: MHC={total_mhc}  MHCsmoke={total_smoke}")


if __name__ == "__main__":
    main()
