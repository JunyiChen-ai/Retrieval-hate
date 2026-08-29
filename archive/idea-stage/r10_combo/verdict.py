#!/usr/bin/env python
"""R10-COMBO -- mechanical application of the frozen decision rule
(idea-stage/R10_COMBO_FREEZE.md section 3.2).  Reads only the JSON produced by
the frozen analyzers; makes no new metric.
"""
import json

BASE = "/home/jehc223/Retrieval-hate/idea-stage/r10_combo/"
CANDS = ["K1", "K2", "K3", "K4", "K5", "K6"]
PCA_FAM = {"K1", "K2", "K6"}
BAR = 0.005
HARM = -0.002
DS = [("MHC_zh", "zh"), ("HateMM", "hm")]

grid = {d: json.load(open(BASE + "%s_grid.json" % p)) for d, p in DS}
dev = {d: json.load(open(BASE + "%s_devpanel.json" % p)) for d, p in DS}


def c(d, prot, a, b):
    return grid[d]["contrasts"]["%s/%s-%s" % (prot, a, b)]


def m(d, prot, arm):
    return grid[d]["arm_means"]["%s/%s" % (prot, arm)]["mean"]


print("=" * 100)
print("ARM MEANS (P1 test macro-F1)")
arms = grid[DS[0][0]]["arms"]
print("%-5s %-24s %-24s" % ("arm", "MHC_zh", "HateMM"))
for a in arms:
    print("%-5s %8.4f +- %-11.4f %8.4f +- %-11.4f"
          % (a, m("MHC_zh", "P1", a),
             grid["MHC_zh"]["arm_means"]["P1/%s" % a]["std"],
             m("HateMM", "P1", a),
             grid["HateMM"]["arm_means"]["P1/%s" % a]["std"]))

REF = {}
for d, _ in DS:
    REF[d] = "LL" if m(d, "P1", "LL") >= m(d, "P1", "CAT") else "CAT"
print("\nREF (better of LL/CAT):", REF)

print("\n" + "=" * 100)
print("FROZEN RULE, per candidate")
stands = {}
for K in CANDS:
    lines = []
    passed_on = []
    for d, _ in DS:
        v = {}
        ok = True
        for ref in ("LL", "CAT"):
            x = c(d, "P1", K, ref)
            p2 = c(d, "P2", K, ref)
            cl = (x["mean"] >= BAR and x["ci_excludes_zero"] and x["mean"] > 0
                  and (p2["mean"] > 0) == (x["mean"] > 0))
            v[ref] = (x, p2, cl)
            ok = ok and cl
        pc = None
        if K in PCA_FAM:
            pc = c(d, "P1", K, "PC0")
            ok = ok and pc["mean"] >= BAR and pc["ci_excludes_zero"] and pc["mean"] > 0
        dv = dev[d]["contrasts"]["dev_mf1_P1/%s-%s" % (K, REF[d])]
        demote = dv["mean"] < 0 and dv["ci_excludes_zero"]
        lines.append("  %-7s vs LL %+0.4f ci[%+0.4f,%+0.4f] P2 %+0.4f | vs CAT %+0.4f "
                     "ci[%+0.4f,%+0.4f] P2 %+0.4f%s | dev vs %s %+0.4f%s -> %s"
                     % (d, v["LL"][0]["mean"], v["LL"][0]["ci95"][0], v["LL"][0]["ci95"][1],
                        v["LL"][1]["mean"], v["CAT"][0]["mean"], v["CAT"][0]["ci95"][0],
                        v["CAT"][0]["ci95"][1], v["CAT"][1]["mean"],
                        ("" if pc is None else " | vs PC0 %+0.4f ci[%+0.4f,%+0.4f]"
                         % (pc["mean"], pc["ci95"][0], pc["ci95"][1])),
                        REF[d], dv["mean"], " (DEMOTE)" if demote else "",
                        "PASS" if ok and not demote else
                        ("DEMOTED" if ok else "fail")))
        if ok and not demote:
            passed_on.append(d)
    harmless = True
    for d, _ in DS:
        if d in passed_on:
            continue
        x = c(d, "P1", K, REF[d])
        harmless = harmless and x["mean"] >= HARM
        lines.append("  %-7s harmlessness vs REF(%s) %+0.4f (need >= %+0.4f) -> %s"
                     % (d, REF[d], x["mean"], HARM, "ok" if x["mean"] >= HARM else "FAIL"))
    verdict = "STANDS" if (passed_on and harmless) else "does not stand"
    stands[K] = (verdict, passed_on)
    print("\n%s: %s" % (K, verdict))
    for l in lines:
        print(l)

print("\n" + "=" * 100)
print("SUMMARY:", {k: v[0] for k, v in stands.items()})
winners = [k for k, v in stands.items() if v[0] == "STANDS"]
if winners:
    key = {k: min(c(d, "P1", k, REF[d])["mean"] for d, _ in DS) for k in winners}
    print("winner (largest min-over-datasets margin over REF):",
          max(key, key=key.get), key)
else:
    print("nothing stands -> frozen fallback: token axis and layer axis are "
          "substitutes; default = cheapest.")
    print("indistinguishable-from-REF arms (mean(K-REF) >= -0.002 on BOTH datasets):")
    def vs_ref(d, a):
        if a == REF[d]:
            return 0.0
        k = "P1/%s-%s" % (a, REF[d])
        if k in grid[d]["contrasts"]:
            return grid[d]["contrasts"][k]["mean"]
        return m(d, "P1", a) - m(d, "P1", REF[d])   # unpaired fallback (LL vs CAT)

    for a in arms:
        if a == "A0":
            continue
        ds_ok = all(vs_ref(d, a) >= HARM for d, _ in DS)
        print("   %-4s %-3s  (%s)" % (a, "yes" if ds_ok else "no",
              ", ".join("%s %+0.4f" % (d, vs_ref(d, a)) for d, _ in DS)))
