#!/usr/bin/env python3
"""Materialize split-scoped labels for fail-closed training processes.

This preprocessing command may read the corpus annotation source. Relation-V2
training never imports ``load_labels`` and opens only the train/val JSON files
created here, so test labels cannot accidentally enter its process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from hate_common import data as hdata  # noqa: E402


def digest_ids(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=os.path.join(
        hdata.SPLIT_ROOT, "scoped_labels"))
    args = ap.parse_args(argv)
    os.makedirs(args.out_root, exist_ok=True)
    for corpus in hdata.CORPORA:
        labels = hdata.load_labels(corpus)
        split_sets = {s: hdata.load_split(corpus, s)
                      for s in ("train", "val", "test")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = set(split_sets[a]) & set(split_sets[b])
            if overlap:
                raise RuntimeError("%s %s/%s overlap: %s" %
                                   (corpus, a, b, sorted(overlap)[:5]))
        for split, ids in split_sets.items():
            missing = [v for v in ids if v not in labels]
            if missing:
                raise RuntimeError("%s/%s missing labels: %s" %
                                   (corpus, split, missing[:5]))
            payload = {
                "corpus": corpus, "split": split,
                "manifest_sha256": digest_ids(ids),
                "labels": {v: int(labels[v]) for v in ids},
            }
            path = os.path.join(args.out_root, "%s_%s.json" %
                                (corpus, split))
            temporary = path + ".tmp"
            with open(temporary, "w") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(temporary, path)
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
