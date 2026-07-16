#!/usr/bin/env python3
"""Modal path-adapter for the FROZEN w2a_probe.py — PLUMBING ONLY, changes NO probe logic.

Why this exists: w2a_probe.py (hash-frozen r2b af4a2f9f…) hardcodes
`REPO = "/data/jehc223/RGCL"` and takes NO data-root argument (unlike its W2-B
sibling which had `--path`). On Modal the synced float caches live on the volume
mounted at /root/data and the repo code at /root/src + /root/scripts/analysis. This
shim (a) verifies the frozen probe is byte-identical, (b) symlinks
/data/jehc223/RGCL/{data,src} -> /root/{data,src} so the probe's absolute paths
resolve to the mounted volume, then (c) runpy-execs the probe UNCHANGED with its
outputs + resumable K9 checkpoint written onto /root/data (the committed volume, so
`modal volume get` can retrieve them). It alters NO threshold, seed, datum, or code
path in the probe; CI_NSEED stays the prereg-mandated 150 for the real run.

Invocation (via scripts/cloud/modal_probe_runner.py::run --script <this> --args ...):
  --args ""            -> REAL run: CI_NSEED=150 (prereg), outputs on /root/data/W2A_PROBE_*
  --args "DRYRUN <n>"  -> THROWAWAY plumbing/timing check: CI_NSEED=<n> (<150, numbers
                          DISCARDED — for pipeline validation only), separate _w2a_dry_* paths
"""
import hashlib
import os
import runpy
import sys

FROZEN_SHA = "af4a2f9f5b35461173fd82c176bd52c6fc84bf8fc0d09736f938d38d8f6fe06d"
PROBE = "/root/scripts/analysis/w2a_probe.py"
REPO = "/data/jehc223/RGCL"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    got = _sha256(PROBE)
    print("[adapter] frozen probe sha256 = {}".format(got), flush=True)
    assert got == FROZEN_SHA, \
        "[adapter] FROZEN PROBE HASH MISMATCH: {} != {} — refusing to run".format(got, FROZEN_SHA)

    os.makedirs(REPO, exist_ok=True)
    for name, target in (("data", "/root/data"), ("src", "/root/src")):
        link = os.path.join(REPO, name)
        if not os.path.lexists(link):
            os.symlink(target, link)
        print("[adapter] symlink {} -> {} (resolves: {})".format(
            link, os.readlink(link) if os.path.islink(link) else "?", os.path.exists(link)), flush=True)

    dry = len(sys.argv) >= 3 and sys.argv[1] == "DRYRUN"
    if dry:
        os.environ["CI_NSEED"] = str(int(sys.argv[2]))
        out_md = "/root/data/_w2a_dry_results.md"
        out_json = "/root/data/_w2a_dry_results.json"
        ci_ckpt = "/root/data/_w2a_dry_ckpt.json"
        print("[adapter] *** DRYRUN (THROWAWAY) CI_NSEED={} — numbers discarded ***".format(
            os.environ["CI_NSEED"]), flush=True)
    else:
        # CI_NSEED intentionally UNSET -> probe default 150 (prereg §16 mandated minimum).
        out_md = "/root/data/W2A_PROBE_RESULTS.md"
        out_json = "/root/data/w2a_probe_results.json"
        ci_ckpt = "/root/data/w2a_ci_ckpt.json"
        print("[adapter] REAL run — CI_NSEED default (150)", flush=True)

    sys.argv = ["w2a_probe.py",
                "--datasets", "HateMM,MHC",
                "--num_frames", "8",
                "--n_boot", "1000",
                "--out_md", out_md,
                "--out_json", out_json,
                "--ci_ckpt", ci_ckpt]
    print("[adapter] exec probe argv: {}".format(" ".join(sys.argv)), flush=True)
    runpy.run_path(PROBE, run_name="__main__")


if __name__ == "__main__":
    main()
