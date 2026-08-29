#!/usr/bin/env python3
"""HEAD-only source-platform availability probe; does not access annotations."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def probe(item: dict) -> dict:
    canonical = item["canonical_id"]
    platform, source = canonical.split(":", 1)
    if platform == "youtube":
        source = source[:11] if len(source) > 11 and source[11:12] == "_" else source
        url = "https://www.youtube.com/watch?v=" + source
    elif platform == "bilibili":
        source = source.rsplit("_", 1)[0] if source.rsplit("_", 1)[-1].isdigit() else source
        url = "https://www.bilibili.com/video/" + source
    else:
        raise ValueError("unsupported platform")
    try:
        response = requests.head(url, allow_redirects=True, timeout=12, headers={"User-Agent": "Mozilla/5.0 THVL-metadata-audit"})
        status = int(response.status_code)
        result = "reachable" if 200 <= status < 400 else ("blocked_or_denied" if status in (401, 403, 405, 412, 429) else "http_error")
    except requests.RequestException:
        status, result = None, "network_error"
    return {"hashed_id": item["hashed_id"], "platform": platform, "status_code": status, "head_result": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--private-out", type=Path, required=True)
    parser.add_argument("--aggregate-out", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.private_map.read_text())["records"]
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(probe, records))
    rows.sort(key=lambda x: x["hashed_id"])
    args.private_out.write_text(json.dumps({"probe": "HEAD only", "records": rows}, sort_keys=True) + "\n")
    args.private_out.chmod(0o600)
    counts = Counter((r["platform"], r["head_result"]) for r in rows)
    status_counts = Counter((r["platform"], str(r["status_code"])) for r in rows)
    aggregate = {
        "probe": "source-platform HTTP HEAD only; no media body downloaded",
        "n_videos": len(rows),
        "result_counts": {f"{p}:{s}": n for (p, s), n in sorted(counts.items())},
        "status_counts": {f"{p}:{s}": n for (p, s), n in sorted(status_counts.items())},
        "private_probe_sha256": hashlib.sha256(args.private_out.read_bytes()).hexdigest(),
    }
    args.aggregate_out.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

