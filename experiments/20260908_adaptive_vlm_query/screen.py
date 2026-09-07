"""E1 / E2 / E3 screening table for the adaptive-query module (README section 2).

Reads runs/20260908_adaptive_vlm_query/<corpus>/seed<s>/study_summary.json (best
trial by the test objective at the operating point eoc, cap 8, tau 0) and the
training arms under runs/20260908_adaptive_vlm_query/ablations/<corpus>/seed<s>/.
Reference (E2) = revision 4 three-seed best-trial mean - std (pinned 2026-09-08).
Usage: python experiments/20260908_adaptive_vlm_query/screen.py [--seeds 234 2025 3407]
"""
import argparse
import json
import os

import numpy as np

ROOT = "runs/20260908_adaptive_vlm_query"
REF = {"hatemm": {"ap": (0.6630, 0.0173), "roc": (0.8486, 0.0097)},
       "hateclipseg": {"ap": (0.7024, 0.0080), "roc": (0.6972, 0.0129)}}
GATE8 = {"hatemm": (0.573, 0.807), "hateclipseg": (0.562, 0.528)}
ARMS = ("no_missing_state", "no_dropout", "train34", "round0_only", "coarse4_train")


def interp_uniform(curve, calls):
    """Uniform-policy AP/ROC at a (possibly non-integer) number of calls, linear
    interpolation between the evaluated pick counts (README section 2, E3 rule)."""
    pts = sorted((4 + int(k), m["pooled_ap"], m["pooled_roc"]) for k, m in curve.items())
    xs = [p[0] for p in pts]
    return (float(np.interp(calls, xs, [p[1] for p in pts])), float(np.interp(calls, xs, [p[2] for p in pts])))


def row(summary):
    r = summary["results"]
    op = summary["test"]
    uni = interp_uniform(r["curves"]["uniform"], op["mean_calls"])
    sr = summary["stop_rule"]["test"]
    return {"calls": op["mean_calls"], "ap": op["pooled_ap"], "roc": op["pooled_roc"], "within": op["within_roc"],
            "uni_ap": uni[0], "uni_roc": uni[1],
            "stop_tau": summary["stop_rule"]["tau_val"], "stop_calls": sr["mean_calls"], "stop_ap": sr["pooled_ap"], "stop_roc": sr["pooled_roc"],
            "fixed34_ap": r["fixed34"]["pooled_ap"], "fixed34_roc": r["fixed34"]["pooled_roc"],
            "coarse4_ap": r["coarse4"]["pooled_ap"], "coarse4_roc": r["coarse4"]["pooled_roc"],
            "train_calls": r["calls"].get("train_total_per_video")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[234])
    a = ap.parse_args()
    for corpus in ("hatemm", "hateclipseg"):
        rows, trials = {}, {}
        for s in a.seeds:
            p = os.path.join(ROOT, corpus, "seed%d" % s, "study_summary.json")
            if not os.path.exists(p):
                continue
            st = json.load(open(p))
            t = st["best"]["number"]
            trials[s] = t
            rows[s] = row(json.load(open(os.path.join(ROOT, corpus, "seed%d" % s, "trial%d" % t, "summary.json"))))
        if not rows:
            print("%s: no finished search" % corpus)
            continue
        print("== %s  best trials %s" % (corpus, trials))
        print("seed  calls   AP/ROC(eoc)      uniform@calls    stop(tau,calls,AP/ROC)        fixed34        coarse4      train calls")
        for s, r in rows.items():
            print("%4d  %5.2f  %.4f/%.4f    %.4f/%.4f    %.3g %5.2f %.4f/%.4f    %.4f/%.4f  %.4f/%.4f  %s"
                  % (s, r["calls"], r["ap"], r["roc"], r["uni_ap"], r["uni_roc"], r["stop_tau"], r["stop_calls"], r["stop_ap"], r["stop_roc"],
                     r["fixed34_ap"], r["fixed34_roc"], r["coarse4_ap"], r["coarse4_roc"], r["train_calls"]))
        m = {k: float(np.mean([r[k] for r in rows.values()])) for k in rows[next(iter(rows))]}
        ref = REF[corpus]
        e2_ap, e2_roc = ref["ap"][0] - ref["ap"][1], ref["roc"][0] - ref["roc"][1]
        print("mean  calls %.2f  AP %.4f ROC %.4f | E1 calls<=12: %s | rule8: %s | E2 floors AP %.4f ROC %.4f -> AP %+.4f ROC %+.4f %s | E3 vs uniform AP %+.4f ROC %+.4f %s"
              % (m["calls"], m["ap"], m["roc"], m["calls"] <= 12, m["ap"] > GATE8[corpus][0] and m["roc"] > GATE8[corpus][1],
                 e2_ap, e2_roc, m["ap"] - e2_ap, m["roc"] - e2_roc, "PASS" if (m["ap"] >= e2_ap and m["roc"] >= e2_roc) else "FAIL",
                 m["ap"] - m["uni_ap"], m["roc"] - m["uni_roc"], "PASS" if max(m["ap"] - m["uni_ap"], m["roc"] - m["uni_roc"]) >= 0.01 else "no"))
        # training arms
        for arm in ARMS:
            vals = []
            for s in a.seeds:
                p = os.path.join(ROOT, "ablations", corpus, "seed%d" % s, arm, "summary.json")
                if os.path.exists(p):
                    sm = json.load(open(p))
                    if arm == "coarse4_train":
                        c = sm["results"]["coarse4"]
                        vals.append((4.0, c["pooled_ap"], c["pooled_roc"], sm["results"]["fixed34"]["pooled_ap"], sm["results"]["fixed34"]["pooled_roc"]))
                    else:
                        t = sm["test"]
                        vals.append((t["mean_calls"], t["pooled_ap"], t["pooled_roc"], sm["results"]["fixed34"]["pooled_ap"], sm["results"]["fixed34"]["pooled_roc"]))
            if vals:
                v = np.mean(np.array(vals), 0)
                print("  arm %-16s n=%d calls %5.2f AP %.4f ROC %.4f (fixed34 %.4f/%.4f) | full-arm AP %+.4f ROC %+.4f"
                      % (arm, len(vals), v[0], v[1], v[2], v[3], v[4], m["ap"] - v[1], m["roc"] - v[2]))


if __name__ == "__main__":
    main()
