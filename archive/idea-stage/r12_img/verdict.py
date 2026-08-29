#!/usr/bin/env python
"""R12-IMG -- mechanical application of the frozen decision rule.

idea-stage/R12_FREEZE.md section 2.5, applied verbatim, no judgement calls.

A candidate C in {ISPLIT, I2M} STANDS iff ALL FOUR hold on BOTH datasets:
  1. C - I0       P1 mean >= +0.005 and 95% CI excludes zero
  2. C - IRW      P1 mean >= +0.005 and 95% CI excludes zero   (not width)
  3. C - IRSPLIT  P1 mean >= +0.005 and 95% CI excludes zero   (not "any second view")
  4. C - I0       P2 CI lower bound >= -0.005                  (P2 must not support harm)

Demotion clause: an arm that is dev-negative against I0 with the CI excluding zero
while test-positive cannot STAND, whatever clauses 1-4 say.

Tie-break if both stand: (i) fewer distinct pooling operations, (ii) smaller total
width, (iii) ISPLIT.  Frozen; no selection by test number.
"""
import argparse
import json

BAR = 0.005
P2_HARM = -0.005
# frozen tie-break key: (n distinct pooling ops, total img width, declaration order)
TIEBREAK = {"ISPLIT": (2, 7168, 0), "I2M": (2, 7168, 1)}


def get(g, prot, name):
    return g["contrasts"]["%s/%s" % (prot, name)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh", required=True)
    ap.add_argument("--hm", required=True)
    ap.add_argument("--zhdev", required=True)
    ap.add_argument("--hmdev", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    grids = {"MHC_zh": json.load(open(a.zh)), "HateMM": json.load(open(a.hm))}
    devs = {"MHC_zh": json.load(open(a.zhdev)), "HateMM": json.load(open(a.hmdev))}

    res = {"freeze": "idea-stage/R12_FREEZE.md 2.5", "bar": BAR,
           "p2_harm_bound": P2_HARM, "candidates": {}}

    standing = []
    for cand in ["ISPLIT", "I2M"]:
        per_ds = {}
        for ds, g in grids.items():
            c1 = get(g, "P1", "%s-I0" % cand)
            c2 = get(g, "P1", "%s-IRW" % cand)
            c3 = get(g, "P1", "%s-IRSPLIT" % cand)
            p2 = get(g, "P2", "%s-I0" % cand)
            dv = devs[ds].get("contrasts", {}).get("dev_mf1_P1/%s-I0" % cand)
            demoted = bool(dv is not None and dv["mean"] < 0
                           and dv.get("ci_excludes_zero") and c1["mean"] > 0)
            per_ds[ds] = {
                "clause1_vs_I0": {"mean": c1["mean"], "ci95": c1["ci95"],
                                  "pass": bool(c1["mean"] >= BAR and c1["ci95"][0] > 0)},
                "clause2_vs_IRW": {"mean": c2["mean"], "ci95": c2["ci95"],
                                   "pass": bool(c2["mean"] >= BAR and c2["ci95"][0] > 0)},
                "clause3_vs_IRSPLIT": {"mean": c3["mean"], "ci95": c3["ci95"],
                                       "pass": bool(c3["mean"] >= BAR and c3["ci95"][0] > 0)},
                "clause4_P2_no_harm": {"mean": p2["mean"], "ci95": p2["ci95"],
                                       "pass": bool(p2["ci95"][0] >= P2_HARM)},
                "dev_contrast": (None if dv is None
                                 else {"mean": dv["mean"], "ci95": dv["ci95"]}),
                "dev_demotion_fired": demoted,
            }
        all_pass = all(
            per_ds[d]["clause%d%s" % (i, k)]["pass"]
            for d in per_ds
            for i, k in [(1, "_vs_I0"), (2, "_vs_IRW"), (3, "_vs_IRSPLIT"),
                         (4, "_P2_no_harm")])
        demoted_any = any(per_ds[d]["dev_demotion_fired"] for d in per_ds)
        verdict = "STANDS" if (all_pass and not demoted_any) else "DOES NOT STAND"
        if verdict == "STANDS":
            standing.append(cand)
        res["candidates"][cand] = {"per_dataset": per_ds, "all_clauses_pass": bool(all_pass),
                                   "dev_demotion_fired": bool(demoted_any),
                                   "verdict": verdict}

    if not standing:
        res["entry"] = ("none -- the image stream's flat prefix mean is not improved by "
                        "a semantic positional split or a second-moment readout at this "
                        "sample size; the last never-varied readout is measured and closed")
    elif len(standing) == 1:
        res["entry"] = standing[0]
    else:
        res["entry"] = sorted(standing, key=lambda c: TIEBREAK[c])[0]
        res["tiebreak_applied"] = True

    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({c: res["candidates"][c]["verdict"] for c in res["candidates"]}
                     | {"entry": res["entry"]}, indent=1))


if __name__ == "__main__":
    main()
