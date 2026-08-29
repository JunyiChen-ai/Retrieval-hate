#!/usr/bin/env python3
"""CPU protocol regression tests; no model training."""

from __future__ import annotations

import json
import os
import tempfile

from hate_common import data as hdata
from relation_v2.protocol import (assert_single_corpus, checkpoint_corpus,
                                  frozen_splits, scoped_labels,
                                  verify_teacher)


def must_fail(fn, *args):
    try:
        fn(*args)
    except (ValueError, RuntimeError, FileNotFoundError):
        return
    raise AssertionError("expected failure: %s%r" % (fn.__name__, args))


def main():
    must_fail(assert_single_corpus, ["hatemm", "mhclip_en"])
    must_fail(assert_single_corpus, "unknown")
    for corpus in hdata.CORPORA:
        splits = frozen_splits(corpus)
        assert not (set(splits["train"]) & set(splits["val"]))
        assert not (set(splits["train"]) & set(splits["test"]))
        for split in ("train", "val", "test"):
            labels, _ = scoped_labels(corpus, split)
            assert set(labels) == set(splits[split])
    must_fail(checkpoint_corpus, {"corpus": "hatemm"}, "mhclip_en")

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        vid = frozen_splits("mhclip_en")["train"][0]
        with open(path, "w") as fh:
            fh.write(json.dumps({"corpus": "hatemm", "video_id": vid}) + "\n")
        must_fail(verify_teacher, "mhclip_en", path)
        test_vid = frozen_splits("mhclip_en")["test"][0]
        with open(path, "w") as fh:
            fh.write(json.dumps({"corpus": "mhclip_en",
                                 "video_id": test_vid}) + "\n")
        must_fail(verify_teacher, "mhclip_en", path)
    finally:
        os.unlink(path)
    print("Relation-V2 protocol: PASS")


if __name__ == "__main__":
    main()
