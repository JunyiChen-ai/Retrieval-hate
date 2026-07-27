#!/usr/bin/env python
"""
vga_pregate_runner.py -- ORCHESTRATION ONLY for the VGA/VNQ $0 pregate.

Why this file exists: the login node reaps long-lived non-SLURM processes -- F95 lost
part of a run to SIGTERM (exit 143) that way (MECHNOV_PAIRVERIFY_PREGATE §3). The full
permutation budget (N_PERM=200 over 36 dataset x block x feature-set x model cells) is
well past that horizon in one process, so this runner drives the SAME frozen analysis
module one DATASET at a time in a short-lived process and serialises each immediately.
A reap then costs at most one dataset.

IT CHANGES NO ARM. It imports `vga_pregate_gate` (sha asserted below, unmodified) and
calls its frozen `run_dataset`. Same gate sets, same targets, same nesting, same inner
folds, same seeds, same permutation budget, same bars. Only the process boundary and
the file granularity differ -- exactly the precedent set by
`scripts/analysis/mechnov_pairverify_runner.py` for the F95 arms module.
"""
import argparse
import json
import os
import sys

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))
import vga_pregate_gate as G      # noqa: E402  FROZEN -- not modified

FROZEN_GATE_SHA = "ea37c57b382b9bb0d1c3a87e9302bac7e52071b8cfe85126e96e26eb524f4e34"
PARTS = os.path.join(REPO, "scripts/analysis/vga_parts")
KEYS = ["hatemm", "zh", "en"]


def run_one(key):
    os.makedirs(PARTS, exist_ok=True)
    out = os.path.join(PARTS, f"{key}.json")
    logp = os.path.join(PARTS, f"{key}.log")
    logf = open(logp, "w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    emit = os.path.join(REPO, f"scripts/analysis/vga_emit_{key}_OUT.json")
    log(f"[{key}] reading {emit}")
    R = G.run_dataset(key, emit, log)
    json.dump(R, open(out, "w"), indent=1)
    log(f"[{key}] CELL DONE -> {out}")
    logf.close()


def merge(outp):
    OUT = {"meta": {"script": os.path.join(REPO, "scripts/analysis/vga_pregate_gate.py"),
                    "script_sha256": G.sha256_of(
                        os.path.join(REPO, "scripts/analysis/vga_pregate_gate.py")),
                    "runner": os.path.abspath(__file__),
                    "assembly": ("per-dataset cells written by vga_pregate_runner.py; "
                                 "frozen analysis module unmodified, sha "
                                 + FROZEN_GATE_SHA),
                    "frozen": dict(INNER_FOLDS=G.INNER_FOLDS, INNER_SEED=G.INNER_SEED,
                                   LOGIT_C=G.LOGIT_C, GBM_N=G.GBM_N,
                                   GBM_DEPTH=G.GBM_DEPTH, GBM_LR=G.GBM_LR,
                                   GBM_SEED=G.GBM_SEED, N_PERM=G.N_PERM,
                                   PERM_SEED=G.PERM_SEED, TARGET_GAIN=G.TARGET_GAIN,
                                   PRIMARY_ADJ=G.PRIMARY_ADJ,
                                   PRIMARY_GATE_MODEL=G.PRIMARY_GATE_MODEL),
                    "test_contact": "NONE"}}
    for k in KEYS:
        p = os.path.join(PARTS, f"{k}.json")
        if os.path.exists(p):
            OUT[k] = json.load(open(p))
    json.dump(OUT, open(outp, "w"), indent=1)
    print(f"merged datasets={[k for k in KEYS if k in OUT]} -> {outp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=KEYS)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", default=os.path.join(
        REPO, "scripts/analysis/vga_pregate_OUT.json"))
    a = ap.parse_args()
    assert G.sha256_of(os.path.join(REPO, "scripts/analysis/vga_pregate_gate.py")) \
        == FROZEN_GATE_SHA, "FROZEN GATE MODULE HAS CHANGED -- refusing to run"
    if a.merge:
        merge(a.out)
    else:
        run_one(a.dataset)


if __name__ == "__main__":
    main()
