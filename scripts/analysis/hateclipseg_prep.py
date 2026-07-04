#!/usr/bin/env python
"""HateClipSeg zero-training localization eval — data prep (CPU only).

Produces (all under RGCL, nothing in src/ touched, no HateClipSeg label is used
for any training/calibration — gold labels are written ONLY for evaluation):

  data/gt/HateClipSeg/video_durations.jsonl  ffprobe duration cache (resumable)
  data/gt/HateClipSeg/gold_segments.json     cleaned machine-readable gold
        {vid: {duration, platform, n_segments, segments: [[s, e, [6-multihot]]],
               notes: [...]}}
  data/gt/HateClipSeg/test.jsonl             id/text/label rows for the subclip
        extractor (label is a DUMMY 0 for every video: the extractor stores an
        inherited label field that our scoring never reads; keeping it constant
        makes the "no HateClipSeg label enters the pipeline" claim auditable).
  data/video/HateClipSeg/All/<vid>.mp4       symlinks to the real files
        (extractor hardcodes <video_dir>/<ds>/All/<id>.mp4; decord/PyAV sniff
        the container, so a .mp4-named symlink to webm/mkv decodes fine).

Cleaning rules (all 22 anomalies are degenerate FINAL segments whose 'end'
equals the true file duration while their 'start' overshoots it):
  duration D := ffprobe container duration;
  every segment is clipped to [0, D]; segments with s >= D or e <= s after
  clipping are dropped (note recorded per video).
"""
import ast
import csv
import json
import os
import subprocess

HCS = "/data/jehc223/HateClipSeg"
ROOT = "/data/jehc223/RGCL"
GT_DIR = os.path.join(ROOT, "data/gt/HateClipSeg")
LINK_DIR = os.path.join(ROOT, "data/video/HateClipSeg/All")
DUR_CACHE = os.path.join(GT_DIR, "video_durations.jsonl")
GOLD_OUT = os.path.join(GT_DIR, "gold_segments.json")
JSONL_OUT = os.path.join(GT_DIR, "test.jsonl")

os.makedirs(GT_DIR, exist_ok=True)
os.makedirs(LINK_DIR, exist_ok=True)

files = {f.rsplit(".", 1)[0]: f for f in os.listdir(os.path.join(HCS, "videos"))}
print("video files:", len(files))

# ---------------------------------------------------------------- durations
durs = {}
if os.path.exists(DUR_CACHE):
    with open(DUR_CACHE) as f:
        for line in f:
            o = json.loads(line)
            durs[o["id"]] = o["duration"]
with open(DUR_CACHE, "a") as f:
    for vid, fn in sorted(files.items()):
        if vid in durs:
            continue
        p = subprocess.run(
            ["/data/jehc223/miniconda3/bin/ffprobe",
             "-v", "quiet", "-print_format", "json", "-show_format",
             os.path.join(HCS, "videos", fn)],
            capture_output=True, text=True)
        d = float(json.loads(p.stdout)["format"]["duration"])
        durs[vid] = d
        f.write(json.dumps({"id": vid, "duration": d}) + "\n")
print("durations cached:", len(durs))

# ---------------------------------------------------------------- gold
rows = list(csv.DictReader(open(os.path.join(HCS, "Dataset/segment_level_annotation.csv"))))
gold = {}
n_drop, n_clip, n_seg = 0, 0, 0
for r in rows:
    vid = r["Video Id"]
    if vid not in files:
        continue
    D = durs[vid]
    labs = ast.literal_eval(r["Segment-Level Label"])
    ts = ast.literal_eval(r["Segment Timestamp"])
    assert len(labs) == len(ts)
    segs, notes = [], []
    for i, (l, (s, e)) in enumerate(zip(labs, ts)):
        s, e = float(s), float(e)
        assert len(l) == 6 and all(x in (0, 1) for x in l)
        if s >= D or e <= s:
            notes.append("dropped_seg{}:[{:.2f},{:.2f}] D={:.2f}".format(i, s, e, D))
            n_drop += 1
            continue
        if e > D + 0.01:
            notes.append("clipped_seg{}:end {:.2f}->{:.2f}".format(i, e, D))
            e = D
            n_clip += 1
        segs.append([round(s, 2), round(e, 2), list(map(int, l))])
    n_seg += len(segs)
    gold[vid] = {
        "duration": round(D, 2),
        "platform": vid.split("_", 1)[0],
        "n_segments": len(segs),
        "segments": segs,
        "notes": notes,
    }
json.dump(gold, open(GOLD_OUT, "w"))
print("gold videos:", len(gold), " segments kept:", n_seg,
      " dropped:", n_drop, " end-clipped:", n_clip)

# ---------------------------------------------------------------- test.jsonl (dummy labels)
with open(JSONL_OUT, "w") as f:
    for vid in sorted(gold):
        f.write(json.dumps({"id": vid, "text": "", "label": 0}) + "\n")
print("wrote", JSONL_OUT, "(label=0 dummy for all rows)")

# ---------------------------------------------------------------- symlinks
made = 0
for vid, fn in files.items():
    link = os.path.join(LINK_DIR, vid + ".mp4")
    tgt = os.path.join(HCS, "videos", fn)
    if os.path.islink(link):
        if os.readlink(link) == tgt:
            continue
        os.remove(link)
    os.symlink(tgt, link)
    made += 1
print("symlinks ensured:", len(files), "(new/updated {})".format(made))
