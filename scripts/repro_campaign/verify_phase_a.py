#!/usr/bin/env python
"""REPRO campaign — Phase A acceptance gates G1-G5 (REPRO_CAMPAIGN_FREEZE §11).

Reads only what Phase A produced; computes no method metric.
Writes idea-stage/repro_campaign/phase_a_gates.json and prints a summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
OUT = ROOT / "idea-stage/repro_campaign/phase_a_gates.json"
EXPECT_N = {"HateMM": 1083, "MHC": 792, "MHC_zh": 814, "HateClipSeg": 395}

# Deviation D2: source files that carry no video stream at all (ffprobe shows an
# audio stream only), so no visual frame exists to encode.  Neither is in any
# frozen project split, so no headline table is affected.
NO_VIDEO_STREAM = {"HateMM": ["hate_video_147", "hate_video_292"]}


def dir_stats(p: Path) -> dict:
    if not p.exists():
        return {"exists": False, "n_files": 0, "bytes": 0}
    fs = [f for f in p.glob("*.npy") if not f.name.startswith(".")]
    return {"exists": True, "n_files": len(fs),
            "bytes": int(sum(f.stat().st_size for f in fs))}


def check_shapes(ds: str, chan: str, n: int = 25) -> dict:
    """Sample n videos: dtype, 2-D (T,1024), T vs the GT grid length."""
    d = ROOT / f"data/CLIP_Embedding/{ds}/dense4fps_{chan}"
    z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
    gt = {str(v): len(y) for v, y in zip(z["video_ids"], z["y4"])}
    fs = sorted(f for f in d.glob("*.npy") if not f.name.startswith("."))
    if not fs:
        return {"sampled": 0}
    rng = np.random.default_rng(20250819)
    pick = [fs[i] for i in rng.choice(len(fs), size=min(n, len(fs)), replace=False)]
    bad_dtype, bad_dim, diffs = [], [], []
    for f in pick:
        a = np.load(f, mmap_mode="r")
        if a.dtype != np.float32:
            bad_dtype.append(f.stem)
        if a.ndim != 2 or a.shape[1] != 1024:
            bad_dim.append([f.stem, list(a.shape)])
        if f.stem in gt:
            diffs.append(int(a.shape[0]) - gt[f.stem])
    return {"sampled": len(pick), "bad_dtype": bad_dtype, "bad_dim": bad_dim,
            "T_minus_Tgt_min": int(min(diffs)) if diffs else None,
            "T_minus_Tgt_max": int(max(diffs)) if diffs else None,
            "T_minus_Tgt_absgt8": int(sum(abs(x) > 8 for x in diffs))}


def main() -> None:
    ctrl = json.loads((ROOT / "idea-stage/repro_campaign/gt_controls.json").read_text())
    gates: dict = {}

    gates["G1_landscape_broadcast"] = {
        ds: {"target": ctrl[ds]["G1_target_AP_1fps"],
             "measured": round(ctrl[ds]["G1_measured_AP_1fps"], 4),
             "abs_diff": round(ctrl[ds]["G1_abs_diff"], 4),
             "pass": ctrl[ds]["G1_pass"]} for ds in EXPECT_N}
    gates["G1_pass"] = all(ctrl[ds]["G1_pass"] for ds in EXPECT_N)

    for gate, chan in (("G2_dense_clipL336", "clipL336"), ("G4_dense_w2vemo", "w2vemo")):
        per = {}
        for ds, n in EXPECT_N.items():
            st = dir_stats(ROOT / f"data/CLIP_Embedding/{ds}/dense4fps_{chan}")
            exc = NO_VIDEO_STREAM.get(ds, [])
            st["expected"] = n
            st["excluded_no_video_stream"] = exc
            st["complete"] = st["n_files"] >= n
            st["complete_or_explained"] = st["n_files"] >= n - len(exc)
            st["GiB"] = round(st["bytes"] / 2**30, 2)
            st.update(check_shapes(ds, chan))
            per[ds] = st
        gates[gate] = per
        gates[gate + "_pass"] = all(v["complete_or_explained"] for v in per.values())
        gates[gate + "_exact"] = all(v["complete"] for v in per.values())

    g3p = ROOT / "idea-stage/repro_campaign/g3_pipeline_consistency.json"
    if g3p.exists():
        g3 = json.loads(g3p.read_text())
        gates["G3_pipeline_drift"] = g3
        gates["G3_pass"] = all(v["bit_identical"] for v in g3.values())

    # G5 cache backfill
    g5 = {}
    for p, key in (
        ("data/ASR/MHC/dev_seen_asrK4_whisper-large-v3.jsonl", "MHC_dev_ASR"),
        ("data/ASR/MHC/test_seen_asrK4_whisper-large-v3.jsonl", "MHC_test_ASR"),
        ("data/OCR/MHC/ocr_windows_K30.jsonl", "MHC_traindev_OCR"),
        ("data/OCR/MHC_zh/ocr_windows_K30.jsonl", "MHC_zh_traindev_OCR"),
        ("data/CLIP_Embedding/HateMM/test_seen_subclipK30_"
         "openai_clip-vit-large-patch14-336_HF.pt", "HateMM_test_subclipK30"),
    ):
        f = ROOT / p
        e = {"path": p, "exists": f.exists()}
        if f.exists():
            e["bytes"] = f.stat().st_size
            if f.suffix == ".jsonl":
                e["n_lines"] = sum(1 for _ in open(f))
        g5[key] = e
    gates["G5_cache_backfill"] = g5
    gates["G5_pass"] = all(v["exists"] for v in g5.values())

    OUT.write_text(json.dumps(gates, indent=1))
    for k in sorted(k for k in gates if k.endswith("_pass")):
        print(f"{k:26s} {gates[k]}")
    for gate in ("G2_dense_clipL336", "G4_dense_w2vemo"):
        for ds, v in gates[gate].items():
            ex = f" (-{len(v['excluded_no_video_stream'])} no-video-stream)" \
                if v["excluded_no_video_stream"] else ""
            print(f"  {gate:20s} {ds:12s} {v['n_files']:>5d}/{v['expected']:<5d}{ex} "
                  f"{v['GiB']:>6.2f} GiB  dT[{v.get('T_minus_Tgt_min')},"
                  f"{v.get('T_minus_Tgt_max')}] out-of-2s={v.get('T_minus_Tgt_absgt8')}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
