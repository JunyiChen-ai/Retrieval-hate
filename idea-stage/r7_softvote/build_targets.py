#!/usr/bin/env python
"""R7-1 -- build the annotator-vote soft targets and the entropy-matched
label-smoothing controls.  CPU only, zero GPU, zero API.

Frozen design: idea-stage/R7_SOFTVOTE_FREEZE.md.

TRAIN SPLIT ONLY.  The val and test vote files are never opened -- an explicit
guard aborts if any path outside `*_train.tsv` is requested.

Soft target for video i with annotator vote multiset V_i (three-way labels
Normal / Offensive / Hateful; the undocumented `Counter Narrative` value is
counted as NOT positive, i.e. it lands in the denominator only):

    p_i(w) = ( #Hateful + w * #Offensive ) / |V_i|,      w in {1.0, 0.5}

Label-smoothing control epsilon is chosen so the CONSTANT smoothed target has
the same mean binary entropy as the soft targets it is matched against:

    H(eps) = mean_i H(p_i(w)),   H(q) = -q ln q - (1-q) ln(1-q),  eps in (0, 0.5)

solved by bisection (H is strictly increasing on (0, 0.5)).
"""
import ast
import collections
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd

ROOT = "/home/jehc223/Retrieval-hate"
VOTES = os.path.join(ROOT, "data", "gt", "mhc_votes")
GT = os.path.join(ROOT, "data", "gt")
HERE = os.path.dirname(os.path.abspath(__file__))

DATASETS = {"MHC_zh": "Chinese", "MHC": "English"}
WEIGHTS = {"SOFT10": 1.0, "SOFT05": 0.5}
POS_CLASSES = ("Hateful", "Offensive")


def guard(path):
    base = os.path.basename(path)
    if not base.endswith("_train.tsv"):
        raise SystemExit("HALT: R7-1 may only read *_train.tsv vote files; got %s" % base)
    return path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def binent(q):
    q = min(max(q, 1e-12), 1 - 1e-12)
    return -(q * math.log(q) + (1 - q) * math.log(1 - q))


def solve_eps(target_H):
    lo, hi = 1e-9, 0.5 - 1e-9
    if target_H <= binent(lo):
        return lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if binent(mid) < target_H:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    meta = {"freeze": "idea-stage/R7_SOFTVOTE_FREEZE.md",
            "rule": "p = (#Hateful + w*#Offensive) / n_annotators; "
                    "Counter Narrative counted as non-positive (denominator only)",
            "train_only_guard": "only *_train.tsv opened",
            "datasets": {}}
    for ds, lang in DATASETS.items():
        vp = guard(os.path.join(VOTES, "mhc_%s_train.tsv" % lang))
        df = pd.read_csv(vp, sep="\t")
        votes = {r["Video_ID"]: ast.literal_eval(r["Label"]) for _, r in df.iterrows()}

        gtp = os.path.join(GT, ds, "train.jsonl")
        rows = [json.loads(l) for l in open(gtp)]
        ids = [r["id"] for r in rows]
        hard = np.array([int(r["label"]) for r in rows], dtype=float)

        missing = [i for i in ids if i not in votes]
        if missing:
            raise SystemExit("HALT: %d train ids without votes, e.g. %s"
                             % (len(missing), missing[:5]))

        vocab = collections.Counter(v for i in ids for v in votes[i])
        nann = collections.Counter(len(votes[i]) for i in ids)

        d = {"vote_tsv": os.path.relpath(vp, ROOT), "vote_tsv_sha256": sha(vp),
             "gt_jsonl": os.path.relpath(gtp, ROOT), "n_train": len(ids),
             "hard_pos_rate": float(hard.mean()),
             "vote_vocab": dict(vocab), "annotators_per_video": dict(nann),
             "arms": {}}

        for arm, w in WEIGHTS.items():
            p = np.array([
                (sum(1 for x in votes[i] if x == "Hateful")
                 + w * sum(1 for x in votes[i] if x == "Offensive")) / len(votes[i])
                for i in ids], dtype=float)
            H = np.array([binent(x) for x in p])
            eps = solve_eps(float(H.mean()))
            out = {"targets": {i: float(x) for i, x in zip(ids, p)}}
            op = os.path.join(HERE, "targets_%s_%s.json" % (ds, arm))
            with open(op, "w") as f:
                json.dump(out, f, indent=0, sort_keys=True)
            d["arms"][arm] = {
                "w": w,
                "target_json": os.path.relpath(op, ROOT),
                "target_json_sha256": sha(op),
                "mean_target": float(p.mean()),
                "frac_non_degenerate": float(np.mean((p > 0) & (p < 1))),
                "n_distinct_values": int(len(set(np.round(p, 6)))),
                "distinct_values": sorted(set(float(x) for x in np.round(p, 6))),
                "corr_with_hard_label": float(np.corrcoef(p, hard)[0, 1]),
                "mean_binary_entropy_nats": float(H.mean()),
                "matched_LS_epsilon": float(eps),
                "LS_entropy_check_nats": float(binent(eps)),
                "n_disagree_with_hard_label_side":
                    int(np.sum(((p > 0.5) != (hard > 0.5)) & (p != 0.5))),
            }
            print("%-8s %-7s w=%.1f  mean=%.4f  nondeg=%.3f  meanH=%.4f  "
                  "matched_eps=%.5f  distinct=%d"
                  % (ds, arm, w, p.mean(), np.mean((p > 0) & (p < 1)), H.mean(),
                     eps, len(set(np.round(p, 6)))))
        meta["datasets"][ds] = d

    mp = os.path.join(HERE, "build_meta.json")
    with open(mp, "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", mp)


if __name__ == "__main__":
    main()
