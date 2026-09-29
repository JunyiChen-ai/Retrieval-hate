"""Summarise stop_check.py outputs (README section 17.4): per corpus, the fixed-budget lines and every stopping rule at
each mean budget, three-seed means of test AP / ROC / within, mean calls (positive / negative videos), and the
difference to the fixed budget 8 of the same setting. PASS = AP and ROC >= fixed 8 - .005 AND within >= fixed 8 - .005
(the within condition is the user's 2026-09-29 ruling: within must not drop; a floor is not allowed, so the
"<rule>_f<k>" lines are reference only).

    python experiments/20260925_query_paradigm/stop_summary.py runs/20260929_query_paradigm_r5/stop_check/hatemm_r5state.json [...]
        [--rules hG hT hmax vsum vmean eig stab1] [--budgets 8 12] [--tol 0.005]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

M = ("pooled_ap", "pooled_roc", "within_roc")


def mean_over_trials(trials, getter):
    xs = [getter(t) for t in trials.values()]
    xs = [x for x in xs if x is not None]
    return None if not xs else np.mean(np.asarray(xs, dtype=np.float64), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--rules", nargs="*", default=None)
    ap.add_argument("--budgets", nargs="*", default=["8", "12"])
    ap.add_argument("--tol", type=float, default=0.005)
    a = ap.parse_args()
    for fp in a.files:
        d = json.load(open(fp))
        trials = d["trials"]
        print("== %s (%s, %d trials, copy_pi %s)" % (fp, d["corpus"], len(trials), d.get("copy_pi")))
        fixed = {}
        for B in sorted({int(b) for t in trials.values() for b in t["fixed"]}):
            m = mean_over_trials(trials, lambda t: [t["fixed"][str(B)]["test"][k] for k in M] if str(B) in t["fixed"] else None)
            c = mean_over_trials(trials, lambda t: [t["fixed"][str(B)]["test_mean_calls"]] if str(B) in t["fixed"] else None)
            fixed[B] = m
            print("  fixed %2d (%5.2f) %.3f/%.3f/%.3f" % (B, c[0], *m))
        ref = fixed.get(8)
        rules = [r for r in list(trials.values())[0]["rules"] if a.rules is None or r in a.rules]
        for rule in rules:
            for B in a.budgets:
                m = mean_over_trials(trials, lambda t: [t["rules"][rule][B]["test"][k] for k in M] if B in t["rules"].get(rule, {}) else None)
                if m is None:
                    continue
                c = mean_over_trials(trials, lambda t: [t["rules"][rule][B][k] for k in ("test_mean_calls", "test_mean_calls_pos", "test_mean_calls_neg", "val_mean_calls")])
                dd = m - ref if ref is not None else np.zeros(3)
                ok = ref is not None and all(x >= -a.tol for x in dd)
                print("  %-10s m%2s calls %5.2f (pos %5.1f neg %5.1f, val %5.2f) %.3f/%.3f/%.3f d8 %+.3f/%+.3f/%+.3f %s" % (
                    rule, B, c[0], c[1], c[2], c[3], *m, *dd, "PASS" if ok else ""))


if __name__ == "__main__":
    main()
