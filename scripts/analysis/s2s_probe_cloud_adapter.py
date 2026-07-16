#!/usr/bin/env python3
"""Modal path-adapter for the FROZEN s2s_probe.py — PLUMBING ONLY, changes NO probe logic.

Why this exists: s2s_probe.py (hash-frozen r4 141a0441…) hardcodes
`REPO = "/data/jehc223/RGCL"` and takes NO data-root argument. On Modal the synced
float caches live on the volume mounted at /root/data and the repo code at /root/src +
/root/scripts/analysis. This shim (a) verifies the frozen probe is byte-identical, (b)
symlinks /data/jehc223/RGCL/{data,src} -> /root/{data,src} so the probe's absolute paths
resolve to the mounted volume, then (c) runpy-execs the probe UNCHANGED with its outputs
written onto /root/data (the committed volume, so `modal volume get` can retrieve them).
It alters NO threshold, seed, datum, or code path in the probe.

Direct mirror of the approved W2-A adapter (scripts/analysis/w2a_probe_cloud_adapter.py,
sha fb609d4b…), same plumbing pattern.

Invocation (via scripts/cloud/modal_probe_runner.py::run --script <this> --args ...):
  --args ""            -> REAL run: prereg config (n_boot=1000, n_perframe_null=100),
                          outputs on /root/data/S2S_PROBE_*
  --args "DRYRUN <b>"  -> THROWAWAY plumbing/timing check: n_boot=<b>, n_perframe_null=0
                          (numbers DISCARDED — pipeline validation only), separate
                          _s2s_dry_* paths
"""
import hashlib
import os
import runpy
import sys

FROZEN_SHA = "141a0441845d6175646d642a57b4534f78a48d96521ef3dc3a2d9fcf0f2301b3"
PROBE = "/root/scripts/analysis/s2s_probe.py"
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
        n_boot = str(int(sys.argv[2]))
        n_perframe = "0"
        out_md = "/root/data/_s2s_dry_results.md"
        out_json = "/root/data/_s2s_dry_results.json"
        print("[adapter] *** DRYRUN (THROWAWAY) n_boot={} n_perframe_null=0 — numbers "
              "discarded ***".format(n_boot), flush=True)
    else:
        # prereg-mandated config: n_boot=1000, n_perframe_null=100 (probe defaults).
        n_boot = "1000"
        n_perframe = "100"
        out_md = "/root/data/S2S_PROBE_RESULTS.md"
        out_json = "/root/data/s2s_probe_results.json"
        print("[adapter] REAL run — prereg config n_boot=1000 n_perframe_null=100", flush=True)

    sys.argv = ["s2s_probe.py",
                "--datasets", "HateMM,MHC",
                "--frameset_dir", "frameset_qwen7b_8f",
                "--n_boot", n_boot,
                "--n_perframe_null", n_perframe,
                "--out_md", out_md,
                "--out_json", out_json]
    print("[adapter] exec probe argv: {}".format(" ".join(sys.argv)), flush=True)
    runpy.run_path(PROBE, run_name="__main__")


if __name__ == "__main__":
    main()
