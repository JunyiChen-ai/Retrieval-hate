#!/usr/bin/env python
"""R11-UNION -- mechanical application of the frozen decision rule.

idea-stage/R11_UNION_FREEZE.md section 5.  Reads the two read-out jsons and
prints STANDS / DOES NOT STAND per candidate mechanism.  No new statistic is
computed here; only the frozen clauses are evaluated.

Candidate mechanisms: AVG, WAVG, SEL (decision-level family), ANCA, ANCL
(anchor-training family).  Reference arm: CAT.

A mechanism STANDS iff there is an ordering (D1, D2) of (MHC_zh, HateMM) with,
under P1 on TEST:
  (a) mean(arm - CAT) on D1 >= +0.005 and the paired-bootstrap 95% CI excludes
      zero on the positive side;
  (b) mean(arm - CAT) on D2 >= -0.002;
  (c) the family control clause on D1:
        decision-level family: mean(arm - ECTL) >= +0.005, CI excluding zero;
        anchor family:         mean(arm - LBL)  >= +0.005, CI excluding zero;
  (d) P2 agrees in sign with (a) on D1: mean(arm - CAT) > 0 under P2.
"""
import argparse
import json

BAR = 0.005
HARM = -0.002
CANDS = [("AVG", "ECTL"), ("WAVG", "ECTL"), ("SEL", "ECTL"),
         ("ANCA", "LBL"), ("ANCL", "LBL")]


def g(res, prot, split, L, R):
    return res["contrasts"]["%s/%s/%s-%s" % (prot, split, L, R)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh", required=True)
    ap.add_argument("--hm", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = {"MHC_zh": json.load(open(a.zh)), "HateMM": json.load(open(a.hm))}

    out = {"what": "R11-UNION frozen verdict", "bar": BAR, "harmless_bar": HARM,
           "per_candidate": {}, "stands": []}
    for cand, ctl in CANDS:
        rec = {"control_arm": ctl, "orderings": {}}
        stands = False
        for d1, d2 in (("MHC_zh", "HateMM"), ("HateMM", "MHC_zh")):
            c1 = g(R[d1], "P1", "test", cand, "CAT")
            c2 = g(R[d2], "P1", "test", cand, "CAT")
            cc = g(R[d1], "P1", "test", cand, ctl)
            p2 = g(R[d1], "P2", "test", cand, "CAT")
            cl = {
                "a_gain_on_D1": bool(c1["mean"] >= BAR and c1["ci95"][0] > 0),
                "b_harmless_on_D2": bool(c2["mean"] >= HARM),
                "c_control_on_D1": bool(cc["mean"] >= BAR and cc["ci95"][0] > 0),
                "d_P2_sign_on_D1": bool(p2["mean"] > 0)}
            ok = all(cl.values())
            stands = stands or ok
            rec["orderings"]["D1=%s" % d1] = {
                "clauses": cl, "all_pass": ok,
                "vs_CAT_D1": [c1["mean"], c1["ci95"]],
                "vs_CAT_D2": [c2["mean"], c2["ci95"]],
                "vs_%s_D1" % ctl: [cc["mean"], cc["ci95"]],
                "P2_vs_CAT_D1": p2["mean"]}
        rec["STANDS"] = stands
        out["per_candidate"][cand] = rec
        if stands:
            out["stands"].append(cand)
        print("%-5s  STANDS=%s" % (cand, stands))
        for k, v in rec["orderings"].items():
            print("   %-14s %s  vsCAT=%+.4f %s  vs%s=%+.4f %s  P2=%+.4f"
                  % (k, v["clauses"], v["vs_CAT_D1"][0], v["vs_CAT_D1"][1],
                     ctl, v["vs_%s_D1" % ctl][0], v["vs_%s_D1" % ctl][1],
                     v["P2_vs_CAT_D1"]))

    out["conclusion"] = ("union partially purchasable: " + ",".join(out["stands"])
                         if out["stands"] else
                         "no mechanism stands -- the union is not purchasable by "
                         "these mechanisms; CAT alone remains the entry")
    print("\nCONCLUSION:", out["conclusion"])
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
