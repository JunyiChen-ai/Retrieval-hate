"""Ablation and control table of revision 6 (rule 14(g)): for each corpus and arm, the 3-seed mean test AP / ROC /
within at the primary operating point (fixed 8 calls, the training summary's "test"), and the drop from the full method
(the rule-7 best trial of the same seed). A component counts as useful when the mean AP or ROC drop is >= .01 on every
main corpus (user ruling 2026-09-05 / 09-06, three corpora since 2026-11-01).

    python experiments/20261101_r6_length_table/arms_table.py [--corpora hatemm hateclipseg dehate]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/20261101_r6_length_table"
SEEDS = (234, 2025, 3407)
METS = ("pooled_ap", "pooled_roc", "within_roc")


def full(corpus, seed):
    st = json.load(open(RUNS / corpus / f"seed{seed}" / "study_summary.json"))
    s = json.load(open(RUNS / corpus / f"seed{seed}" / f"trial{st['best']['number']}" / "summary.json"))
    return [s["test"][m] for m in METS]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+", default=["hatemm", "hateclipseg", "dehate"])
    a = ap.parse_args()
    table = {}
    for c in a.corpora:
        ref = np.array([full(c, s) for s in SEEDS])
        arms = sorted({p.name for p in (RUNS / "arms" / c).glob("seed*/*") if p.is_dir()})
        table[c] = {"full": ref.mean(0).tolist(), "arms": {}}
        print("%s full %s" % (c, "/".join("%.3f" % x for x in ref.mean(0))))
        for arm in arms:
            rows = [json.load(open(f))["test"] for s in SEEDS
                    for f in [RUNS / "arms" / c / f"seed{s}" / arm / "summary.json"] if f.exists()]
            if len(rows) < len(SEEDS):
                print("  %-14s %d/%d seeds" % (arm, len(rows), len(SEEDS)))
                continue
            m = np.array([[r[k] for k in METS] for r in rows]).mean(0)
            d = ref.mean(0) - m
            table[c]["arms"][arm] = {"mean": m.tolist(), "drop": d.tolist()}
            print("  %-14s %s  drop %s" % (arm, "/".join("%.3f" % x for x in m), "/".join("%+.3f" % x for x in d)))
    useful = {}
    for arm in sorted({k for c in table for k in table[c]["arms"]}):
        if all(arm in table[c]["arms"] for c in a.corpora):
            useful[arm] = all(max(table[c]["arms"][arm]["drop"][:2]) >= 0.01 for c in a.corpora)
    print("useful (AP or ROC drop >= .01 on every corpus):", {k: v for k, v in useful.items()})
    (RUNS / "arms" / "table.json").write_text(json.dumps({"corpora": a.corpora, "table": table, "useful": useful},
                                                         indent=1) + "\n")


if __name__ == "__main__":
    main()
