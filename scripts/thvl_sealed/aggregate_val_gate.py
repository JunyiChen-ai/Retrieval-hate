#!/usr/bin/env python3
"""Emit label-safe aggregate validation gate statistics only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.thvl_sealed.acquire import load_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--val-media-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = {x["canonical_id"]: x["split"] for x in json.loads(args.private_map.read_text())["records"]}
    val = [row for row in load_rows(args.csv) if split[row["canonical_id"]] == "validation"]
    media = json.loads(args.val_media_manifest.read_text())
    duration_lower = {
        row["opaque_id"]: int(float(row["hf_path"].split("/", 1)[0].split("-", 1)[0]) * 60)
        for row in media["records"]
    }
    canonical_to_opaque = {
        x["canonical_id"]: x["hashed_id"]
        for x in json.loads(args.private_map.read_text())["records"]
        if x["split"] == "validation"
    }
    positive = 0
    negative = 0
    segment_mixed = 0
    provable_frame_mixed = 0
    unresolved_frame_mixed = 0
    for row in val:
        relevant = [x for x in row["segments"] if x["relevant"]]
        irrelevant = [x for x in row["segments"] if not x["relevant"]]
        positive += bool(relevant)
        negative += not relevant
        segment_mixed += bool(relevant and irrelevant)
        if relevant:
            lower = duration_lower[canonical_to_opaque[row["canonical_id"]]]
            grid = [0] * lower
            for item in relevant:
                for t in range(lower):
                    if t < item["end"] and t + 1 > item["start"]:
                        grid[t] = 1
            if any(grid) and not all(grid):
                provable_frame_mixed += 1
            else:
                unresolved_frame_mixed += 1
    payload = {
        "dataset": "THVL-Bench",
        "split": "validation",
        "target": "Verbal Abuse OR Hate OR Bias",
        "n_videos": len(val),
        "target_positive_videos": positive,
        "target_negative_videos": negative,
        "annotated_segment_mixed_videos": segment_mixed,
        "provably_frame_mixed_videos_using_public_duration_bin_lower_bound": provable_frame_mixed,
        "positive_videos_with_frame_mixed_status_pending_exact_duration": unresolved_frame_mixed,
        "exact_frame_mixed_count_status": "BLOCKED until label-independent exact media durations are frozen",
        "contains_ids_or_labels": False,
        "validation_media_manifest_sha256": hashlib.sha256(args.val_media_manifest.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

