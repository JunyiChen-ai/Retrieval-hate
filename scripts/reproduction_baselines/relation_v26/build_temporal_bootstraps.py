#!/usr/bin/env python3
"""Build the three frozen V26 validation video-cluster bootstrap arrays."""
import argparse
from pathlib import Path
import sys

import numpy as np

from artifacts import atomic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "relation_v25"))
from steward_val_artifact import decrypt_and_verify


def arrays(n, seed, labels=None):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(2000):
        for attempt in range(100):
            ix = rng.integers(0, n, n).tolist()
            if labels is None or len({labels[i] for i in ix}) == 2:
                out.append(ix)
                break
        else:
            raise RuntimeError("bootstrap class redraw exhausted")
    return out


def run(a):
    payload = decrypt_and_verify(a.encrypted_dir, a.key, a.val_manifest, a.qc,
                                 a.taxonomy, a.private_source, a.raw_id_map)
    gt = payload["records"]
    ids = sorted(gt)
    positive = [v for v in ids if any(gt[v]["target_1hz"])]
    mixed = [v for v in ids if len(set(gt[v]["target_1hz"])) == 2]
    video_y = [int(any(gt[v]["target_1hz"])) for v in ids]
    specs = [
        ("all32", 26031, ids, video_y, a.all_out),
        ("positive", 26032, positive, None, a.positive_out),
        ("mixed", 26033, mixed, None, a.mixed_out),
    ]
    for cohort, seed, cohort_ids, labels, out in specs:
        atomic(out, {"schema": "v26_bootstrap_indices_v1", "cohort": cohort,
                     "seed": seed, "B": 2000, "ids": cohort_ids,
                     "arrays": arrays(len(cohort_ids), seed, labels)})


def main():
    p = argparse.ArgumentParser()
    for name in ("encrypted-dir", "key", "val-manifest", "qc", "taxonomy",
                 "private-source", "raw-id-map", "all-out", "positive-out",
                 "mixed-out"):
        p.add_argument("--" + name, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
