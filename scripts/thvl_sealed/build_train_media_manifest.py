#!/usr/bin/env python3
"""Build THVL train media manifest without reading temporal annotations."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from scripts.thvl_sealed.build_val_media_manifest import MEDIA_SUFFIXES, candidates
from scripts.thvl_sealed.core import FIXED_REMOTE_IDENTITY, canonical_json


def head_metadata(url: str) -> tuple[int, int | None, str | None]:
    response = requests.head(url, allow_redirects=False, timeout=30)
    if response.status_code not in (200, 302, 307):
        raise RuntimeError(f"pinned HF media HEAD failed: {response.status_code}")
    linked_size = response.headers.get("x-linked-size")
    linked_etag = response.headers.get("x-linked-etag", "").strip('"') or None
    return response.status_code, int(linked_size) if linked_size else None, linked_etag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--opaque-train", type=Path, required=True)
    parser.add_argument("--hf-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    private = json.loads(args.private_map.read_text())
    opaque = json.loads(args.opaque_train.read_text())
    api = json.loads(args.hf_metadata.read_text())
    revision = FIXED_REMOTE_IDENTITY["revision"]
    if private["remote_identity"] != FIXED_REMOTE_IDENTITY:
        raise RuntimeError("private-map remote identity mismatch")
    if api["sha"] != revision or opaque["revision"] != revision or opaque["split"] != "train":
        raise RuntimeError("frozen train/HF revision mismatch")

    media: dict[str, list[str]] = {}
    for sibling in api["siblings"]:
        path = sibling["rfilename"]
        if Path(path).suffix.lower() in MEDIA_SUFFIXES:
            media.setdefault(Path(path).stem, []).append(path)

    weak = {row["hashed_id"]: row for row in opaque["records"]}
    source = [row for row in private["records"] if row["split"] == "train"]
    if len(source) != 314 or len(weak) != 314:
        raise RuntimeError("train cohort must contain exactly 314 unique records")

    staged = []
    for item in source:
        if item["hashed_id"] not in weak:
            raise RuntimeError("private/opaque train cohort mismatch")
        hits: list[str] = []
        rule = None
        for index, stem in enumerate(candidates(item["raw_id"])):
            if media.get(stem):
                hits = media[stem]
                rule = "exact_stripped" if index == 0 else "optional_yt_prefix_removed_or_added"
                break
        if len(hits) != 1:
            raise RuntimeError("train media path must reconcile uniquely without labels")
        platform, platform_id = item["canonical_id"].split(":", 1)
        path = hits[0]
        url = f"https://huggingface.co/datasets/THVL/THVL-Bench/resolve/{revision}/{path}"
        staged.append((item, weak[item["hashed_id"]], platform, platform_id, path, url, rule))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        heads = list(pool.map(lambda row: head_metadata(row[5]), staged))

    rows = []
    for staged_row, head in zip(staged, heads, strict=True):
        item, weak_row, platform, platform_id, path, url, rule = staged_row
        status, size, etag = head
        rows.append({
            "opaque_id": item["hashed_id"],
            "duplicate_group_hash": weak_row["duplicate_group_hash"],
            "platform": platform,
            "id": platform_id,
            "split": "train",
            "weak_video_label": weak_row["weak_video_label"],
            "hf_revision": revision,
            "hf_path": path,
            "hf_pinned_url": url,
            "filename_reconciliation": rule,
            "expected_repo_size_bytes": size,
            "expected_repo_etag": etag,
            "expected_repo_sha256": etag if etag and len(etag) == 64 else None,
            "head_status": status,
        })
    rows.sort(key=lambda row: row["opaque_id"])
    if len(rows) != 314 or len({row["opaque_id"] for row in rows}) != 314:
        raise RuntimeError("train manifest identity invariant failed")
    if set(weak) != {row["opaque_id"] for row in rows}:
        raise RuntimeError("train manifest does not exactly cover frozen opaque cohort")

    payload = {
        "schema_version": 1,
        "dataset": "THVL-Bench",
        "purpose": "train media acquisition with permitted weak video labels; no segments, frame GT, or target timestamps",
        "remote_identity": FIXED_REMOTE_IDENTITY,
        "source_opaque_manifest_sha256": hashlib.sha256(args.opaque_train.read_bytes()).hexdigest(),
        "source_private_map_sha256": hashlib.sha256(args.private_map.read_bytes()).hexdigest(),
        "source_hf_metadata_sha256": hashlib.sha256(args.hf_metadata.read_bytes()).hexdigest(),
        "n_records": len(rows),
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(payload) + b"\n")
    print(json.dumps({
        "n_records": len(rows),
        "available_count": sum(row["head_status"] in (200, 302, 307) for row in rows),
        "target_positive": sum(row["weak_video_label"] == 1 for row in rows),
        "target_negative": sum(row["weak_video_label"] == 0 for row in rows),
        "reconciled_by_optional_prefix": sum(row["filename_reconciliation"] != "exact_stripped" for row in rows),
        "total_expected_bytes": sum(row["expected_repo_size_bytes"] or 0 for row in rows),
        "missing_size_count": sum(row["expected_repo_size_bytes"] is None for row in rows),
        "manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
