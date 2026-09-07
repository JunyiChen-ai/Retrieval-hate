"""P1 / P2 decision for revision 4 (README section 2, pre-registered 2026-09-08).

Reads runs/<rev4>/<corpus>/seed<s>/study_summary.json (best trial by the test
objective) for the three seeds, prints the three-seed table next to revision 3,
and writes runs/<rev4>/p2_decision.json.  Usage:
  python experiments/20260908_c3_rev4_rev2_backbone_interval_hmm/p2_decision.py
"""
import json
import os

import numpy as np

REV4 = "runs/20260908_c3_rev4_rev2_backbone_interval_hmm"
REV3 = "runs/20260907_c3_rev3_interval_evidence"
SEEDS = (234, 2025, 3407)
GATE8 = {"hatemm": (0.573, 0.807), "hateclipseg": (0.562, 0.528)}
REV2_HATEMM_AP_MINUS_STD = 0.6581


def best_rows(root, corpus):
    rows = {}
    for s in SEEDS:
        p = os.path.join(root, corpus, "seed%d" % s, "study_summary.json")
        if not os.path.exists(p):
            continue
        st = json.load(open(p))
        assert len(st["trials"]) == st["n_trials"], p
        ua = st["best"]["user_attrs"]
        rows[s] = {"trial": st["best"]["number"], "ap": ua["test_pooled_ap"],
                   "roc": ua["test_pooled_roc"], "within": ua["test_within_roc"]}
    return rows


def stats(rows, key):
    v = np.array([r[key] for r in rows.values()])
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")


def main():
    out = {"rule": "README section 2 P2 (2026-09-08)", "corpora": {}}
    p2 = True
    for corpus in ("hatemm", "hateclipseg"):
        r4, r3 = best_rows(REV4, corpus), best_rows(REV3, corpus)
        c = {"rev4": r4, "rev3": r3}
        for name, rows in (("rev4", r4), ("rev3", r3)):
            for k in ("ap", "roc", "within"):
                c["%s_%s_mean" % (name, k)], c["%s_%s_std" % (name, k)] = stats(rows, k)
        if 234 in r4:
            c["P1_pass"] = bool(r4[234]["ap"] > GATE8[corpus][0] and r4[234]["roc"] > GATE8[corpus][1])
        if len(r4) == 3:
            c["ap_floor"] = c["rev3_ap_mean"] - max(c["rev3_ap_std"], 0.005)
            c["roc_floor"] = c["rev3_roc_mean"] - max(c["rev3_roc_std"], 0.005)
            c["ap_ok"] = bool(c["rev4_ap_mean"] >= c["ap_floor"])
            c["roc_ok"] = bool(c["rev4_roc_mean"] >= c["roc_floor"])
            if corpus == "hatemm":
                c["rev2_ok"] = bool(c["rev4_ap_mean"] >= REV2_HATEMM_AP_MINUS_STD)
            p2 = p2 and c["ap_ok"] and c["roc_ok"] and c.get("rev2_ok", True)
        else:
            p2 = False
        out["corpora"][corpus] = c
        print("%s: rev4 %s" % (corpus, {s: (r["trial"], round(r["ap"], 4), round(r["roc"], 4), round(r["within"], 4)) for s, r in r4.items()}))
        if len(r4) == 3:
            print("  rev4 mean AP %.4f±%.4f ROC %.4f±%.4f within %.4f±%.4f | rev3 AP %.4f±%.4f ROC %.4f±%.4f within %.4f±%.4f"
                  % (c["rev4_ap_mean"], c["rev4_ap_std"], c["rev4_roc_mean"], c["rev4_roc_std"], c["rev4_within_mean"], c["rev4_within_std"],
                     c["rev3_ap_mean"], c["rev3_ap_std"], c["rev3_roc_mean"], c["rev3_roc_std"], c["rev3_within_mean"], c["rev3_within_std"]))
            print("  floors AP %.4f ROC %.4f -> ap_ok %s roc_ok %s rev2_ok %s" % (c["ap_floor"], c["roc_floor"], c["ap_ok"], c["roc_ok"], c.get("rev2_ok", "-")))
    out["P2_pass"] = p2
    out["starting_point"] = "rev4" if p2 else "rev3"
    out["backbone_config"] = ({"bias_mode": "key", "ctx_mode": "rep"} if p2 else {"bias_mode": "gated", "ctx_mode": "logit"})
    print("P2_pass", p2, "starting_point", out["starting_point"], out["backbone_config"])
    os.makedirs(REV4, exist_ok=True)
    with open(os.path.join(REV4, "p2_decision.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
