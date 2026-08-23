from __future__ import annotations

import argparse
import platform
from pathlib import Path

from .actions import split_rows, video_path
from .common import ContactLedger, atomic_json, sha256_file


def entry(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--code", type=Path, action="append", required=True)
    ap.add_argument("--role", action="append", choices=("train", "val"), required=True)
    args = ap.parse_args()
    ledger = ContactLedger()
    sources = []
    for role in args.role:
        for row in sorted(split_rows(role, ledger), key=lambda x: str(x["id"])):
            path = video_path(str(row["id"]))
            if path is None:
                raise RuntimeError("HALT_MISSING_SOURCE:" + str(row["id"]))
            sources.append(entry(path))
    payload = {"schema": "cvoi-start-manifest/1", "python": platform.python_version(),
               "roles": args.role, "code": [entry(p) for p in args.code],
               "sources": sources, "source_count": len(sources),
               "contact": ledger.snapshot()}
    atomic_json(args.out, payload)


if __name__ == "__main__":
    main()
