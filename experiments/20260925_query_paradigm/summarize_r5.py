"""Revision-5 search summary (README section 17.1): per corpus of a search root, the best trial of each seed
(study_summary.json "best", the test objective as in revision 4) and the validation-selected trial
("validation_selected"), the fixed-budget test curve (mean +- std over seeds of trial<k>/summary.json), the
validation-selected fixed 8 line and the adaptive (EIG / VOI threshold) lines. Prints the text that the
summary_<corpus>.txt files of runs/20260929_query_paradigm_r5 hold.

    python experiments/20260925_query_paradigm/summarize_r5.py --root runs/20260929_query_paradigm_r5_soft5 --corpus hatemm
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

M = ("pooled_ap", "pooled_roc", "within_roc")
SEEDS = (234, 2025, 3407)


def load(root, corpus):
    out = {}
    for s in SEEDS:
        p = os.path.join(root, corpus, "seed%d" % s, "study_summary.json")
        if os.path.exists(p):
            out[s] = json.load(open(p))
    return out


def trial_summary(root, corpus, seed, k):
    return json.load(open(os.path.join(root, corpus, "seed%d" % seed, "trial%d" % k, "summary.json")))


def fmt(xs):
    a = np.asarray(xs, dtype=np.float64)
    if a.ndim == 2 and a.shape[0] > 1:
        return " / ".join("%.4f±%.3f" % (m, s) for m, s in zip(a.mean(0), a.std(0)))
    return " / ".join("%.4f" % x for x in np.atleast_2d(a).mean(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--corpus", required=True)
    a = ap.parse_args()
    st = load(a.root, a.corpus)
    best = {s: int(st[s]["best"]["number"]) for s in st}
    vsel = {s: int(st[s]["validation_selected"]["number"]) for s in st if st[s].get("validation_selected")}
    src = next(iter(st.values())).get("extra", {})
    print("%s %s %s: best trials %s, validation-selected %s" % (a.corpus, os.path.basename(a.root.rstrip("/")),
                                                                src.get("answer_source", "?"), best, vsel))
    summ = {s: trial_summary(a.root, a.corpus, s, best[s]) for s in best}
    for B in ("0", "1", "2", "4", "8", "16", "32"):
        rows = [[summ[s]["fixed"][B]["test"][m] for m in M] for s in summ if B in summ[s]["fixed"]]
        if rows:
            print("fixed %2s: %s" % (B, fmt(rows)))
    if vsel:
        vs = {s: trial_summary(a.root, a.corpus, s, vsel[s]) for s in vsel}
        rows = [[vs[s]["fixed"]["8"]["test"][m] for m in M] for s in vs]
        print("validation-selected fixed 8: %s" % fmt(rows))
    for key in ("adaptive", "adaptive_voi"):
        for B in ("4", "8"):
            rows = [[summ[s][key][B]["test"][m] for m in M] for s in summ if B in summ[s].get(key, {})]
            calls = [summ[s][key][B].get("test_mean_calls", float("nan")) for s in summ if B in summ[s].get(key, {})]
            if rows:
                print("%s mean %s: %s | calls %.2f" % (key, B, fmt(rows), float(np.mean(calls))))


if __name__ == "__main__":
    main()
