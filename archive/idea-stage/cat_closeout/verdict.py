#!/usr/bin/env python
"""CAT close-out -- mechanical application of the frozen decision rules.

Frozen design: idea-stage/CAT_CLOSEOUT_FREEZE.md sections 2.4, 3.3, 4.3.
No rule is computed here that is not written in that file.
"""
import argparse
import json
import os

BAR = 0.005


def get(path, key="CAT-A0", proto="P1"):
    if not os.path.exists(path):
        return None
    return json.load(open(path))["contrasts"][key][proto]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legA", required=True)
    ap.add_argument("--legB", required=True)
    ap.add_argument("--legC_zh", required=True)
    ap.add_argument("--legC_hm", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = {"freeze": "idea-stage/CAT_CLOSEOUT_FREEZE.md", "bar": BAR, "legs": {}}

    c = get(a.legA)
    out["legs"]["A_MHC_zh_reextraction"] = {
        "rule": "freeze 2.4: REPRODUCED iff P1 mean >= +0.005 and 95% CI excludes zero",
        "contrast": "CAT-A0", "stats": c,
        "verdict": None if c is None else
        ("REPRODUCED" if (c["mean"] >= BAR and c["ci_lo"] > 0)
         else "NOT REPRODUCED AT THE FROZEN BAR")}

    c = get(a.legB)
    out["legs"]["B_MHC_EN_transport"] = {
        "rule": "freeze 3.3: TRANSPORTS iff P1 mean >= +0.005 and 95% CI excludes zero",
        "contrast": "CAT-A0", "stats": c,
        "verdict": None if c is None else
        ("TRANSPORTS" if (c["mean"] >= BAR and c["ci_lo"] > 0)
         else "DOES NOT TRANSPORT AT THE FROZEN BAR")}

    for name, path in (("C_cv_MHC_zh", a.legC_zh), ("C_cv_HateMM", a.legC_hm)):
        c = get(path)
        out["legs"][name] = {
            "rule": "freeze 4.3: CV-SUPPORTED iff P1 mean over cells > 0 and 95% CI "
                    "excludes zero; clearing +0.005 reported without gate status",
            "contrast": "CAT-A0", "stats": c,
            "verdict": None if c is None else
            ("CV-SUPPORTED" if (c["mean"] > 0 and c["ci_lo"] > 0) else "NOT CV-SUPPORTED"),
            "also_clears_bar": None if c is None else bool(c["mean"] >= BAR)}

    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in out["legs"].items():
        s = v["stats"]
        print("%-24s %-32s %s" % (k, v["verdict"],
                                  "" if s is None else
                                  "mean=%+.4f CI[%+.4f,%+.4f] %d/%d>0"
                                  % (s["mean"], s["ci_lo"], s["ci_hi"], s["n_pos"],
                                     len(s["per_unit"]))))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
