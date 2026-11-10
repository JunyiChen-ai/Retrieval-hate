"""Summary of stage_a_all_trials.py: per variant, the best trial of each seed by (test AP + test ROC) / 2 (rule 7's
selection), the 3-seed mean and standard deviation of its AP / ROC / within, and the mean over all trials.
    python experiments/20261110_prior_weight/all_trials_table.py [--root runs/20261110_prior_weight/all_trials]
Output <root>/table.json and a printed table.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METS = ("pooled_ap", "pooled_roc", "within_roc")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/20261110_prior_weight/all_trials")
    a = ap.parse_args()
    root = (ROOT / a.root).resolve()
    table = {}
    for cd in sorted(p for p in root.iterdir() if p.is_dir()):
        rows = {}                                # variant -> seed -> list of (trial, metrics)
        n_files = {}
        for sd in sorted(cd.glob("seed*")):
            files = sorted(sd.glob("trial*.json"), key=lambda p: int(p.stem[5:]))
            n_files[sd.name] = len(files)
            for f in files:
                d = json.loads(f.read_text())
                for var, r in d["variants"].items():
                    rows.setdefault(var, {}).setdefault(sd.name, []).append((f.stem, r))
                r6 = d["r6_summary_test"]
                rows.setdefault("(r6 search summary)", {}).setdefault(sd.name, []).append((f.stem, r6))
        out = {}
        for var, by_seed in rows.items():
            best, allm, cs = {}, [], []
            for seed, lst in by_seed.items():
                t, r = max(lst, key=lambda x: x[1]["pooled_ap"] + x[1]["pooled_roc"])
                best[seed] = {"trial": t, **{m: r[m] for m in METS}, "n_trials": len(lst)}
                allm += [[r[m] for m in METS] for _, r in lst]
                cs += [r["weights"]["c"] for _, r in lst if r.get("weights")]
            sel = np.array([[best[s][m] for m in METS] for s in best])
            out[var] = {"selected": best, "selected_mean": dict(zip(METS, sel.mean(0).tolist())),
                        "selected_std": dict(zip(METS, sel.std(0).tolist())),
                        "all_trials_mean": dict(zip(METS, np.mean(allm, 0).tolist())), "n_seeds": len(best),
                        "c_median": float(np.median(cs)) if cs else None}
        table[cd.name] = {"variants": out, "files": n_files}
        print("== %s (trial files per seed %s)" % (cd.name, n_files))
        for var, r in out.items():
            print("  %-22s selected %s +- %s | all trials %s | trials %s%s" % (
                var, "/".join("%.3f" % r["selected_mean"][m] for m in METS),
                "/".join("%.3f" % r["selected_std"][m] for m in METS),
                "/".join("%.3f" % r["all_trials_mean"][m] for m in METS),
                ",".join(r["selected"][s]["trial"][5:] for s in sorted(r["selected"])),
                "" if r["c_median"] is None else " | median c %.2f" % r["c_median"]))
    (root / "table.json").write_text(json.dumps(table, indent=1) + "\n")


if __name__ == "__main__":
    main()
