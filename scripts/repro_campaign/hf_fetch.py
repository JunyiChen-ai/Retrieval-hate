#!/usr/bin/env python
"""Resilient HF downloader for the repro campaign.

huggingface-cli + hf_transfer reaches ~20 MB/s from this host but hangs
outright every few GB (no bytes written, process alive).  Plain requests mode
never hangs but is capped near 1 MB/s per connection.  This wrapper keeps
hf_transfer and adds a stall watchdog: a child process does the download, the
parent samples the cache size and kills + restarts the child whenever it makes
no progress for --stall seconds.  huggingface-cli resumes from partial blobs,
so restarts are cheap.

Usage:
  hf_fetch.py <repo_id> [--include PAT ...] [--exclude PAT ...]
              [--local-dir DIR] [--type model|dataset]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def dir_size(p: Path) -> int:
    tot = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                tot += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--include", nargs="*", default=[])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--type", default="model", choices=["model", "dataset"])
    ap.add_argument("--stall", type=int, default=90, help="seconds without growth -> restart")
    ap.add_argument("--max-restarts", type=int, default=40)
    args = ap.parse_args()

    if args.local_dir:
        watch = Path(args.local_dir)
    else:
        hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
        watch = hub / ("models--" + args.repo.replace("/", "--") if args.type == "model"
                       else "datasets--" + args.repo.replace("/", "--"))
    watch.mkdir(parents=True, exist_ok=True)

    cmd = [shutil.which("huggingface-cli") or "huggingface-cli", "download", args.repo]
    if args.type == "dataset":
        cmd += ["--repo-type", "dataset"]
    if args.include:
        cmd += ["--include"] + args.include
    if args.exclude:
        cmd += ["--exclude"] + args.exclude
    if args.local_dir:
        cmd += ["--local-dir", args.local_dir]

    t_start = time.time()

    for attempt in range(args.max_restarts):
        # hf_transfer is ~20x faster but reliably hangs on the tail of a file
        # (observed: stops 800 KB short of a 599 MB blob and never returns).
        # Plain mode is ~1 MB/s but resumes byte-exactly and never hangs, so
        # alternate: odd attempts finish what hf_transfer left behind.
        use_ht = (attempt % 2 == 0)
        env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="1" if use_ht else "0")
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        last_size, last_change = dir_size(watch), time.time()
        while True:
            rc = proc.poll()
            if rc is not None:
                if rc == 0:
                    mb = dir_size(watch) / 2**20
                    print(f"OK {args.repo} {mb:.0f} MiB in "
                          f"{time.time()-t_start:.0f}s ({attempt+1} attempt(s))", flush=True)
                    return 0
                print(f"retry {args.repo}: exit {rc}", flush=True)
                break
            time.sleep(15)
            cur = dir_size(watch)
            if cur > last_size:
                last_size, last_change = cur, time.time()
            elif time.time() - last_change > (args.stall if use_ht else args.stall * 4):
                print(f"stall {args.repo} at {cur/2**20:.0f} MiB -> restart", flush=True)
                proc.kill()
                proc.wait()
                break
        time.sleep(3)

    print(f"GIVE-UP {args.repo}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
