#!/usr/bin/env python
"""G0: build the HateMM GT-target map for the TARC (target-aware
retrieval-contrastive) oracle probe (exp-tarc-t0.md, B-line).

Reads data/gt/HateMM/HateMM_annotation.csv (header
`video_file_name,label,hate_snippet,target`) and writes
data/gt/HateMM/target_map.json.

Output schema (top-level keys are video *stems*; the leading `_meta`
key carries the class code dictionary and provenance, since JSON has no
comments -- any consumer must skip keys starting with `_`):

    {
      "_meta": { "code_dict": {"Blacks":0, ...}, ... },
      "hate_video_1":     {"targets": ["Blacks"],        "primary": 0},
      "non_hate_video_1": {"targets": ["Others"],        "primary": 3},
      ...
    }

`targets` = the normalised list of content-community labels for that
video (may be empty). `primary` = the integer code of the FIRST-listed
target under CODE_DICT, or -1 when the video has no target.

Normalisation (exp-tarc-t0.md §4): a `target` cell starting with '['
is parsed with ast.literal_eval (list literal, e.g. "['Blacks','Jews']");
otherwise it is split on ','. Every element is stripped and de-duplicated
(order preserved). The 8 codes are fixed in the pre-registration order.

This is a pure data-prep step: NO GPU, NO MLLM. The resulting map is read
by the training pipeline ONLY on the `--tarc_target_source gt_oracle`
(oracle ceiling) path (exp-tarc-t0.md §5).
"""

import ast
import csv
import json
import os
from collections import Counter, OrderedDict

# Repo root = two levels up from scripts/analysis/.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(REPO, "data", "gt", "HateMM", "HateMM_annotation.csv")
OUT_PATH = os.path.join(REPO, "data", "gt", "HateMM", "target_map.json")
SPLIT_DIR = os.path.join(REPO, "data", "gt", "HateMM")

# Fixed 8-class code dictionary, in the exp-tarc-t0.md pre-registration order.
CODE_DICT = OrderedDict([
    ("Blacks", 0),
    ("Jews", 1),
    ("Whites", 2),
    ("Others", 3),
    ("LGBTQ", 4),
    ("Muslims", 5),
    ("Sexits", 6),
    ("Asian", 7),
])

# Expected hate-video per-target presence (a hate video counted once per
# distinct target), from exp-tarc-t0.md §4 -- verified live 2026-07-13.
EXPECTED_HATE_PRESENCE = {
    "Blacks": 329, "Jews": 90, "Whites": 18, "Others": 12,
    "LGBTQ": 12, "Muslims": 10, "Sexits": 5, "Asian": 1,
}
EXPECTED_SPLIT_N = {"train": 744, "val": 107, "test": 215}


def normalise_target(raw):
    """Return the normalised, de-duplicated list of target strings for one cell."""
    t = (raw or "").strip()
    if t == "":
        return []
    if t.startswith("["):
        parsed = ast.literal_eval(t)
        elems = []
        # target literals are flat lists; be robust to an accidental nesting.
        def _flatten(x):
            if isinstance(x, (list, tuple)):
                for y in x:
                    _flatten(y)
            else:
                elems.append(str(x).strip())
        _flatten(parsed)
    else:
        elems = [p.strip() for p in t.split(",")]
    out = []
    for e in elems:
        if e and e not in out:
            out.append(e)
    return out


def main():
    rows = list(csv.DictReader(open(CSV_PATH)))
    print("[G0] CSV data rows: {}".format(len(rows)))

    target_map = OrderedDict()
    unknown = Counter()
    hate_presence = Counter()
    nonhate_presence = Counter()
    hate_primary = Counter()
    hate_multi = hate_empty = 0

    for r in rows:
        stem = r["video_file_name"].rsplit(".", 1)[0]
        is_hate = (r["label"].strip() == "Hate")
        names = normalise_target(r["target"])
        for n in names:
            if n not in CODE_DICT:
                unknown[n] += 1
        primary = CODE_DICT.get(names[0], -1) if names else -1
        target_map[stem] = {"targets": names, "primary": primary}

        # --- audit accumulation ---
        if is_hate:
            if not names:
                hate_empty += 1
            else:
                if len(names) > 1:
                    hate_multi += 1
                for n in set(names):
                    hate_presence[n] += 1
                hate_primary[names[0]] += 1
        else:
            for n in set(names):
                nonhate_presence[n] += 1

    if unknown:
        raise SystemExit(
            "[G0] FATAL: target strings outside the 8-class dict: {}".format(dict(unknown)))

    # --- jsonl stem-join coverage check (744/107/215 must all hit) ---
    split_ok = True
    split_report = {}
    for sp in ("train", "val", "test"):
        path = os.path.join(SPLIT_DIR, "{}.jsonl".format(sp))
        ids = [json.loads(l)["id"] for l in open(path)]
        miss = [i for i in ids if i not in target_map]
        split_report[sp] = (len(ids), len(ids) - len(miss), miss)
        if len(miss) != 0 or len(ids) != EXPECTED_SPLIT_N[sp]:
            split_ok = False

    meta = OrderedDict([
        ("_note", "TARC GT-target oracle map (exp-tarc-t0.md G0). Keys are video "
                  "stems; skip keys starting with '_'. GT target -- read ONLY on the "
                  "--tarc_target_source gt_oracle probe path (never a main-table run)."),
        ("code_dict", dict(CODE_DICT)),
        ("num_targets", len(CODE_DICT)),
        ("primary_rule", "first-listed normalised target; -1 == no target"),
        ("source_csv", os.path.relpath(CSV_PATH, REPO)),
        ("n_videos", len(target_map)),
    ])
    out = OrderedDict()
    out["_meta"] = meta
    out.update(target_map)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=0)
    print("[G0] wrote {} ({} video stems)".format(
        os.path.relpath(OUT_PATH, REPO), len(target_map)))

    # ------------------------- self-check report -------------------------
    print("\n[G0] === jsonl stem-join coverage ===")
    for sp in ("train", "val", "test"):
        n, hit, miss = split_report[sp]
        print("  {:5s}: {}/{} hit (expected {}) {}".format(
            sp, hit, n, EXPECTED_SPLIT_N[sp],
            "OK" if (hit == n and n == EXPECTED_SPLIT_N[sp]) else "MISMATCH {}".format(miss[:5])))

    print("\n[G0] === hate-video per-target presence  (cross-check vs exp-tarc-t0.md §4) ===")
    print("  {:8s} {:>6s} {:>8s} {:>4s}".format("target", "count", "expected", "ok"))
    counts_ok = True
    for name in CODE_DICT:
        got = hate_presence.get(name, 0)
        exp = EXPECTED_HATE_PRESENCE[name]
        ok = (got == exp)
        counts_ok = counts_ok and ok
        print("  {:8s} {:6d} {:8d}  {}".format(name, got, exp, "OK" if ok else "XX"))
    print("  hate multi-target videos: {} (expected 42)".format(hate_multi))
    print("  hate empty-target videos: {} (expected 1)".format(hate_empty))
    print("  hate primary (first-listed): {}".format(
        {k: hate_primary[k] for k in CODE_DICT if hate_primary.get(k)}))
    print("  non-hate presence (target on both classes): {}".format(
        {k: nonhate_presence[k] for k in CODE_DICT if nonhate_presence.get(k)}))

    all_ok = counts_ok and split_ok and hate_multi == 42 and hate_empty == 1
    print("\n[G0] SELF-CHECK: {}".format("PASS" if all_ok else "FAIL -- investigate above"))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
