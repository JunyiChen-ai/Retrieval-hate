#!/usr/bin/env python3
"""Build the label-free development-side THVL validation media manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests

from scripts.thvl_sealed.core import FIXED_REMOTE_IDENTITY, canonical_json

MEDIA_SUFFIXES = {".mp4", ".mov"}


def candidates(raw_id: str) -> list[str]:
    value = raw_id.strip()
    ordered = [value]
    if value.startswith("yt_"):
        ordered.append(value[3:])
    else:
        ordered.append("yt_" + value)
    return list(dict.fromkeys(ordered))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--hf-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    private_rows = json.loads(args.private_map.read_text())["records"]
    api = json.loads(args.hf_metadata.read_text())
    if api["sha"] != FIXED_REMOTE_IDENTITY["revision"]:
        raise RuntimeError("HF revision mismatch")
    media = {}
    for sibling in api["siblings"]:
        path = sibling["rfilename"]
        if Path(path).suffix.lower() in MEDIA_SUFFIXES:
            media.setdefault(Path(path).stem, []).append(path)

    rows = []
    for item in private_rows:
        if item["split"] != "validation":
            continue
        hits = []
        rule = None
        for index, stem in enumerate(candidates(item["raw_id"])):
            current = media.get(stem, [])
            if current:
                hits = current
                rule = "exact_stripped" if index == 0 else "optional_yt_prefix_removed_or_added"
                break
        if len(hits) != 1:
            raise RuntimeError("validation media path must reconcile uniquely without labels")
        path = hits[0]
        url = f"https://huggingface.co/datasets/THVL/THVL-Bench/resolve/{FIXED_REMOTE_IDENTITY['revision']}/{path}"
        response = requests.head(url, allow_redirects=False, timeout=20)
        if response.status_code not in (200, 302, 307):
            raise RuntimeError("pinned HF media HEAD failed")
        linked_etag = response.headers.get("x-linked-etag", "").strip('"') or None
        linked_size = response.headers.get("x-linked-size")
        platform, platform_id = item["canonical_id"].split(":", 1)
        rows.append({
            "opaque_id": item["hashed_id"],
            "platform": platform,
            "id": platform_id,
            "split": "validation",
            "hf_revision": FIXED_REMOTE_IDENTITY["revision"],
            "hf_path": path,
            "hf_pinned_url": url,
            "filename_reconciliation": rule,
            "expected_repo_size_bytes": int(linked_size) if linked_size else None,
            "expected_repo_etag": linked_etag,
            "expected_repo_sha256": linked_etag if linked_etag and len(linked_etag) == 64 else None,
            "head_status": response.status_code,
        })
    rows.sort(key=lambda x: x["opaque_id"])
    if len(rows) != 32 or len({x["opaque_id"] for x in rows}) != len(rows):
        raise RuntimeError("validation cohort invariant failed")
    payload = {
        "schema_version": 1,
        "dataset": "THVL-Bench",
        "purpose": "label-free validation media acquisition; contains no labels or segments",
        "remote_identity": FIXED_REMOTE_IDENTITY,
        "n_records": len(rows),
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(payload) + b"\n")
    print(json.dumps({
        "n_records": len(rows),
        "unique_reconciled": sum(x["filename_reconciliation"] != "exact_stripped" for x in rows),
        "all_head_resolved": all(x["head_status"] in (200, 302, 307) for x in rows),
        "manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

