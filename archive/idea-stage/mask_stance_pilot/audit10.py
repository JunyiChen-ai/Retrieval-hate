"""MASK STANCE PILOT -- frozen 10-item extraction audit (freeze doc §5.2).

Draws 10 items with numpy.default_rng(20260812) over the lexicographically sorted eval id list
and dumps, for each, the original transcript, the extracted spans and the masked transcript,
for hand-reading. Selection is deterministic and independent of any result.
"""
import json
import os
import sys
import textwrap

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SP = os.path.join(ROOT, "idea-stage", "stance_pilot")


def main(tag):
    ids = sorted(f"{x['dataset']}::{x['id']}" for x in
                 json.load(open(os.path.join(SP, "sample.json")))["eval"])
    pick = set(np.array(ids)[np.default_rng(20260812).choice(len(ids), 10, replace=False)]) \
        if hasattr(np, "default_rng") else None
    rng = np.random.default_rng(20260812)
    pick = set(np.array(ids)[rng.choice(len(ids), 10, replace=False)].tolist())
    print("AUDIT SELECTION (rng 20260812):")
    for k in sorted(pick):
        print("  ", k)
    print()
    M = {}
    for line in open(os.path.join(HERE, f"masked_{tag}.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        M[f"{r['dataset']}::{r['id']}"] = r
    E = {}
    for line in open(os.path.join(HERE, f"extract_{tag}.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        E[f"{r['dataset']}::{r['id']}"] = r
    for k in sorted(pick):
        r, e = M[k], E.get(k, {})
        rep = r["report"]
        print("=" * 100)
        print(k, r["group"], "| spans", rep["n_spans"], "matched", rep["matched"],
              "unmatched", rep["unmatched"], "placeholders", rep["n_placeholders"],
              "masked_frac", rep["masked_frac"], "leaks", rep.get("residual_leaks"))
        print("-- EXTRACTED SPANS --")
        for sp in ((e.get("parsed") or {}).get("spans") or []):
            print("   *", json.dumps(sp, ensure_ascii=False)[:300])
        if rep["unmatched_spans"]:
            print("   UNMATCHED:", rep["unmatched_spans"])
        print("-- ORIGINAL --")
        print(textwrap.fill(r["orig"][:1800], 120))
        print("-- MASKED --")
        print(textwrap.fill(r["masked"][:1800], 120))
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m1")
