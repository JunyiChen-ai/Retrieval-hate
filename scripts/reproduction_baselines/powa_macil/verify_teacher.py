#!/usr/bin/env python3
"""Coverage/schema audit for a completed POWA train-only teacher file."""

import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from hate_common import data as hdata  # noqa: E402

AXES = ("hostile", "target", "violence", "sexual", "self_harm", "context")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--max-chunks", type=int, default=2)
    ap.add_argument("--allow-missing", type=int, default=0)
    args = ap.parse_args()
    rows, errors, seen = [], [], set()
    with open(args.teacher, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append("line %d invalid JSON: %s" % (line_no, exc))
                continue
            key = (rec.get("corpus"), rec.get("video_id"))
            if key in seen:
                errors.append("duplicate %s/%s" % key)
            seen.add(key)
            chunks, values = rec.get("chunks") or [], rec.get("primitive_prob") or []
            if not 1 <= len(chunks) <= args.max_chunks or len(chunks) != len(values):
                errors.append("%s/%s bad chunk count %d/%d" %
                              (*key, len(chunks), len(values)))
            for value in values:
                if set(value) != set(AXES):
                    errors.append("%s/%s bad axes" % key)
                    continue
                v = np.asarray([value[x] for x in AXES], dtype=float)
                if not np.isfinite(v).all() or (v < 0).any() or (v > 1).any():
                    errors.append("%s/%s invalid probability" % key)
            rows.append(rec)
    expected = set()
    for corpus in hdata.CORPORA:
        expected.update((corpus, v) for v in hdata.load_split(corpus, "train"))
    missing, extra = sorted(expected - seen), sorted(seen - expected)
    if len(missing) > args.allow_missing:
        errors.append("missing %d train videos" % len(missing))
    if extra:
        errors.append("%d records outside train" % len(extra))
    summary = {"records": len(rows), "unique": len(seen),
               "expected_train": len(expected), "missing": len(missing),
               "extra": len(extra),
               "by_corpus": dict(collections.Counter(x[0] for x in seen)),
               "errors": errors[:20], "status": "PASS" if not errors else "FAIL"}
    print(json.dumps(summary, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
