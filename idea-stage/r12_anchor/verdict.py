#!/usr/bin/env python
"""R12-ANCHOR -- mechanical application of the frozen decision rule.

idea-stage/R12_FREEZE.md section 3.5, applied verbatim, no judgement calls.

AF_PT STANDS iff ALL THREE hold on BOTH datasets, under P1:
  1. gain            AF_PT - CAT      mean >= +0.005 and 95% CI excludes zero
  2. filter          AF_PT - AU_PT    mean >   0     and 95% CI excludes zero
  3. semantics       AF_PT - AF_SHUF  mean >= +0.005 and 95% CI excludes zero

AF_A0 is judged by the same three clauses with AU_A0 substituted in clause 2 and
AF_SHUF kept in clause 3.  It is SECONDARY: it becomes the entry only if AF_PT
does not stand.

Demotion clause (freeze 3.5): an arm that is dev-negative with the CI excluding
zero while test-positive cannot STAND, whatever clauses 1-3 say.

Outcomes:
  STANDS
  BANKABLE_CONFIGURATION_MECHANISM_NOT_IDENTIFIED   (clause 1 only)
  KILL                                              (clause 1 fails anywhere)
"""
import argparse
import json

BAR = 0.005


def get(g, prot, name):
    return g["contrasts"]["%s/%s" % (prot, name)]


def clause(c, bar, strict_gt=False):
    ok_ci = bool(c["ci_excludes_zero"] and c["mean"] > 0)
    ok_mean = (c["mean"] > 0) if strict_gt else (c["mean"] >= bar)
    return bool(ok_ci and ok_mean)


def judge(grids, cand, uniform_ref):
    out = {}
    for ds, g in grids.items():
        c1 = get(g, "P1", "%s-CAT" % cand)
        c2 = get(g, "P1", "%s-%s" % (cand, uniform_ref))
        c3 = get(g, "P1", "%s-AF_SHUF" % cand)
        out[ds] = {
            "clause1_gain": {"contrast": "%s-CAT" % cand, "mean": c1["mean"],
                             "ci95": c1["ci95"], "pass": clause(c1, BAR)},
            "clause2_filter": {"contrast": "%s-%s" % (cand, uniform_ref),
                               "mean": c2["mean"], "ci95": c2["ci95"],
                               "pass": clause(c2, BAR, strict_gt=True)},
            "clause3_semantics": {"contrast": "%s-AF_SHUF" % cand, "mean": c3["mean"],
                                  "ci95": c3["ci95"], "pass": clause(c3, BAR)},
        }
    return out


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

    res = {"freeze": "idea-stage/R12_FREEZE.md 3.5", "bar": BAR, "candidates": {}}

    for cand, uref in [("AF_PT", "AU_PT"), ("AF_A0", "AU_A0")]:
        per_ds = judge(grids, cand, uref)
        c1 = all(per_ds[d]["clause1_gain"]["pass"] for d in per_ds)
        c2 = all(per_ds[d]["clause2_filter"]["pass"] for d in per_ds)
        c3 = all(per_ds[d]["clause3_semantics"]["pass"] for d in per_ds)

        # demotion clause: dev-negative with CI excluding zero AND test-positive
        demoted = {}
        for ds in per_ds:
            key = "%s-CAT" % cand
            dv = devs[ds].get("contrasts", {}).get("dev_mf1_P1/%s" % key)
            tv = get(grids[ds], "P1", key)
            if dv is None:
                demoted[ds] = None
                continue
            demoted[ds] = bool(dv["mean"] < 0 and dv.get("ci_excludes_zero")
                               and tv["mean"] > 0)
        any_demoted = any(v is True for v in demoted.values())

        if not c1 or any_demoted:
            verdict = "KILL"
        elif c1 and c2 and c3:
            verdict = "STANDS"
        else:
            verdict = "BANKABLE_CONFIGURATION_MECHANISM_NOT_IDENTIFIED"

        res["candidates"][cand] = {
            "per_dataset": per_ds,
            "clause1_all": bool(c1), "clause2_all": bool(c2), "clause3_all": bool(c3),
            "dev_demotion_fired": demoted, "verdict": verdict}

    af_pt = res["candidates"]["AF_PT"]["verdict"]
    af_a0 = res["candidates"]["AF_A0"]["verdict"]
    if af_pt == "STANDS":
        res["entry"] = "AF_PT"
    elif af_a0 == "STANDS":
        res["entry"] = "AF_A0 (pseudo-teacher unnecessary)"
    elif "BANKABLE" in af_pt or "BANKABLE" in af_a0:
        res["entry"] = "bankable configuration only, mechanism not identified"
    else:
        res["entry"] = ("none -- reference-correctness gating does not break the "
                        "retention/breakage exchange rate on this substrate")

    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({"AF_PT": af_pt, "AF_A0": af_a0, "entry": res["entry"]}, indent=1))


if __name__ == "__main__":
    main()
