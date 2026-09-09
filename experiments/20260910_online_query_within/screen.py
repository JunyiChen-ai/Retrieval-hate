"""P1 / W1 / E1 screening table for the online-query module (README section 2).

Reads runs/20260910_online_query_within<suffix>/<corpus>/seed<s>/study_summary.json
(best trial by the test objective at the operating point eoc, cap 4, tau 0 = 8
calls) and, when present, diagnostic arms under .../ablations/<corpus>/seed<s>/.
Reference = module-1 iteration-1 three-seed best-trial means (12 calls), pinned
2026-09-10. Usage: python experiments/20260910_online_query_within/screen.py [--seeds 234 2025 3407] [--suffix _it1]
"""
import argparse
import json
import os

import numpy as np

ROOT = "runs/20260910_online_query_within"
REF = {"hatemm": {"ap": (0.6674, 0.0130), "roc": (0.8487, 0.0067), "within": 0.6279},
       "hateclipseg": {"ap": (0.6903, 0.0043), "roc": (0.6873, 0.0050), "within": 0.5367}}
BASE_WITHIN = {"hatemm": 0.632, "hateclipseg": 0.562}      # MultiHateLoc; VERA (DSANet .545 = strongest trained)
GATE8 = {"hatemm": (0.573, 0.807), "hateclipseg": (0.562, 0.528)}
ARMS = ("no_missing_state", "no_window_loss", "hmm_weight", "regimes3", "window_target_posterior", "fixed_uniform_train")


def interp_uniform(curve, calls):
    pts = sorted((4 + int(k), m["pooled_ap"], m["pooled_roc"], m["within_roc"]) for k, m in curve.items())
    xs = [p[0] for p in pts]
    return tuple(float(np.interp(calls, xs, [p[i] for p in pts])) for i in (1, 2, 3))


def row(summary):
    r = summary["results"]
    op = summary["test"]
    uni = interp_uniform(r["curves"]["uniform"], op["mean_calls"])
    sr = summary["stop_rule"]["test"]
    return {"calls": op["mean_calls"], "ap": op["pooled_ap"], "roc": op["pooled_roc"], "within": op["within_roc"],
            "uni_ap": uni[0], "uni_roc": uni[1], "uni_within": uni[2],
            "stop_tau": summary["stop_rule"]["tau_val"], "stop_calls": sr["mean_calls"], "stop_ap": sr["pooled_ap"],
            "stop_roc": sr["pooled_roc"], "stop_within": sr["within_roc"],
            "fixed34": (r["fixed34"]["pooled_ap"], r["fixed34"]["pooled_roc"], r["fixed34"]["within_roc"]),
            "coarse4": (r["coarse4"]["pooled_ap"], r["coarse4"]["pooled_roc"], r["coarse4"]["within_roc"]),
            "train_calls": r["calls"].get("train_total_per_video"), "epoch": summary["selected_epoch"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[234])
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    root = ROOT + a.suffix
    for corpus in ("hatemm", "hateclipseg"):
        rows, trials = {}, {}
        for s in a.seeds:
            p = os.path.join(root, corpus, "seed%d" % s, "study_summary.json")
            if not os.path.exists(p):
                continue
            st = json.load(open(p))
            t = st["best"]["number"]
            trials[s] = t
            rows[s] = row(json.load(open(os.path.join(root, corpus, "seed%d" % s, "trial%d" % t, "summary.json"))))
        if not rows:
            print("%s: no finished search" % corpus)
            continue
        print("== %s  best trials %s" % (corpus, trials))
        print("seed  calls  AP/ROC/within(eoc)        uniform@calls            stop(tau,calls,AP/ROC/within)      fixed34                coarse4               train calls  epoch")
        for s, r in rows.items():
            print("%4d  %5.2f  %.4f/%.4f/%.4f   %.4f/%.4f/%.4f   %.3g %5.2f %.4f/%.4f/%.4f   %.4f/%.4f/%.4f   %.4f/%.4f/%.4f   %s  %d"
                  % (s, r["calls"], r["ap"], r["roc"], r["within"], r["uni_ap"], r["uni_roc"], r["uni_within"],
                     r["stop_tau"], r["stop_calls"], r["stop_ap"], r["stop_roc"], r["stop_within"],
                     *r["fixed34"], *r["coarse4"], r["train_calls"], r["epoch"]))
        m = {k: float(np.mean([r[k] for r in rows.values()])) for k in
             ("calls", "ap", "roc", "within", "uni_ap", "uni_roc", "uni_within", "stop_calls", "stop_ap", "stop_roc", "stop_within")}
        sd = {k: float(np.std([r[k] for r in rows.values()])) for k in ("ap", "roc", "within")}
        ref = REF[corpus]
        p1_ap, p1_roc = ref["ap"][0] - max(ref["ap"][1], 0.005), ref["roc"][0] - max(ref["roc"][1], 0.005)
        w1 = max(ref["within"] + 0.01, BASE_WITHIN[corpus])
        print("mean  calls %5.2f  AP %.4f±%.4f ROC %.4f±%.4f within %.4f±%.4f | stop %.2f calls %.4f/%.4f/%.4f"
              % (m["calls"], m["ap"], sd["ap"], m["roc"], sd["roc"], m["within"], sd["within"],
                 m["stop_calls"], m["stop_ap"], m["stop_roc"], m["stop_within"]))
        print("  rule8 %s | P1 floors AP %.4f ROC %.4f -> %+.4f %+.4f %s | W1 floor within %.4f -> %+.4f %s (stop point %+.4f) | E1 calls<=8: fixed %s stop %s | vs uniform AP %+.4f within %+.4f"
              % (m["ap"] >= GATE8[corpus][0] and m["roc"] >= GATE8[corpus][1],
                 p1_ap, p1_roc, m["ap"] - p1_ap, m["roc"] - p1_roc, "PASS" if (m["ap"] >= p1_ap and m["roc"] >= p1_roc) else "FAIL",
                 w1, m["within"] - w1, "PASS" if m["within"] >= w1 else "FAIL", m["stop_within"] - w1,
                 m["calls"] <= 8.0, m["stop_calls"] <= 8.0, m["ap"] - m["uni_ap"], m["within"] - m["uni_within"]))
        for arm in ARMS:
            vals = []
            for s in a.seeds:
                p = os.path.join(root, "ablations", corpus, "seed%d" % s, arm, "summary.json")
                if os.path.exists(p):
                    t = json.load(open(p))["test"]
                    vals.append((t["mean_calls"], t["pooled_ap"], t["pooled_roc"], t["within_roc"]))
            if vals:
                v = np.mean(vals, 0)
                print("  arm %-24s n=%d calls %5.2f AP %.4f ROC %.4f within %.4f | full-arm AP %+.4f ROC %+.4f within %+.4f"
                      % (arm, len(vals), v[0], v[1], v[2], v[3], m["ap"] - v[1], m["roc"] - v[2], m["within"] - v[3]))


if __name__ == "__main__":
    main()
