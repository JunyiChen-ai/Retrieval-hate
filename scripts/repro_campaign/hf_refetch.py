#!/usr/bin/env python
"""Re-fetch corrupted HuggingFace cache blobs, with sha256 verification.

Why this exists.  `hf_fetch.py`'s stall watchdog kills `hf_transfer` mid-flight
and then resumes in plain mode.  `hf_transfer` writes a file through many
parallel range requests, so the on-disk size of a killed download is *not* the
number of contiguous valid bytes; resuming from that offset produces a file of
exactly the right length whose middle is wrong.  Seven checkpoints in this
cache were damaged that way (see `audit_hf_cache.sh` and
`idea-stage/repro_campaign/hf_cache_audit.txt`).  The damage is silent: the
corrupt `openai/clip-vit-base-patch16` loaded without error and returned the
same image embedding for every frame of every video.

What this does instead: fetch each file with N independent `curl` range
requests writing to N separate part files, concatenate, and verify the sha256
against the blob's own filename (an HF cache blob is stored under its LFS oid,
i.e. its sha256).  A file is only put into the cache if it hashes correctly, so
a wrong download can never be mistaken for a good one again.

Usage
  hf_refetch.py --audit idea-stage/repro_campaign/hf_cache_audit.txt   # all CORRUPT rows
  hf_refetch.py --blob <path-to-blob>                                  # one file
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

CHUNK = 32 * 1024 * 1024
WORKERS = 16


def find_source(blob: Path):
    """(repo_id, revision, filename) of a blob, from the snapshot symlink to it."""
    repo_dir = blob.parent.parent
    for snap in (repo_dir / "snapshots").iterdir():
        for f in snap.rglob("*"):
            if f.is_symlink() and os.path.realpath(f) == str(blob.resolve()):
                name = repo_dir.name
                kind, rest = ("dataset", name[len("datasets--"):]) if name.startswith(
                    "datasets--") else ("model", name[len("models--"):])
                return rest.replace("--", "/"), snap.name, str(f.relative_to(snap)), kind
    return None


def total_size(url: str) -> int:
    out = subprocess.run(["curl", "-sLI", url], capture_output=True, text=True).stdout
    for line in reversed(out.splitlines()):
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"no content-length for {url}")


def fetch_range(url: str, out: Path, start: int, end: int, tries: int = 6) -> int:
    for k in range(tries):
        r = subprocess.run(["curl", "-sL", "--fail", "--max-time", "900",
                            "-r", f"{start}-{end}", url, "-o", str(out)])
        if r.returncode == 0 and out.exists() and out.stat().st_size == end - start + 1:
            return out.stat().st_size
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"range {start}-{end} failed after {tries} tries")


def refetch(blob: Path, workdir: Path) -> bool:
    want = blob.name
    src = find_source(blob)
    if src is None:
        print(f"[skip] {blob}: no snapshot symlink points at it", flush=True)
        return False
    repo, rev, fname, kind = src
    base = "https://huggingface.co" + ("/datasets" if kind == "dataset" else "")
    url = f"{base}/{repo}/resolve/{rev}/{fname}"
    size = total_size(url)
    print(f"[get] {repo}@{rev[:8]} {fname} {size/2**30:.2f} GiB", flush=True)

    parts = workdir / want[:12]
    parts.mkdir(parents=True, exist_ok=True)
    jobs = [(i, s, min(s + CHUNK - 1, size - 1))
            for i, s in enumerate(range(0, size, CHUNK))]
    t0 = time.time()
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fetch_range, url, parts / f"{i:05d}", s, e): i
                for i, s, e in jobs}
        done = 0
        for f in cf.as_completed(futs):
            f.result()
            done += 1
            if done % WORKERS == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"PROGRESS {repo} {done}/{len(jobs)} chunks "
                      f"{done*CHUNK/2**20/max(el,1e-9):.1f} MiB/s", flush=True)

    h = hashlib.sha256()
    tmp = workdir / (want + ".assembled")
    with open(tmp, "wb") as out:
        for i, _s, _e in jobs:
            b = (parts / f"{i:05d}").read_bytes()
            h.update(b)
            out.write(b)
    got = h.hexdigest()
    for i, _s, _e in jobs:
        (parts / f"{i:05d}").unlink()
    parts.rmdir()
    if got != want:
        print(f"[FAIL] {repo}/{fname}: sha256 {got} != {want}", flush=True)
        tmp.unlink()
        return False
    os.replace(tmp, blob)
    print(f"[OK] {repo}/{fname} verified, {time.time()-t0:.0f}s", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=None)
    ap.add_argument("--blob", nargs="*", default=[])
    ap.add_argument("--workdir", default="/home/jehc223/Retrieval-hate/.hf_refetch")
    args = ap.parse_args()

    blobs = [Path(b) for b in args.blob]
    if args.audit:
        for line in Path(args.audit).read_text().splitlines():
            if line.startswith("CORRUPT\t"):
                blobs.append(Path(line.split("\t")[2]))
    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    ok = bad = 0
    for b in blobs:
        try:
            if refetch(b, wd):
                ok += 1
            else:
                bad += 1
        except Exception as e:
            print(f"[ERR] {b}: {type(e).__name__}:{e}"[:300], flush=True)
            bad += 1
    print(f"[done] repaired={ok} failed={bad}", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
