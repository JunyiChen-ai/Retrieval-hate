#!/usr/bin/env python
"""
Mechanically apply the PRE-DECLARED decision rule (LSMI_GATE_RECORD.md §2.5, amended §2.6) to
refine-logs/LSMI_GATE_OUT.json and write the verdict block back into it. No estimation here --
this file only evaluates thresholds, so the verdict is computed by code, not by prose.

Rule (verbatim from §2.5):
  M1 power     : G1 XOR at our n must give S >= 0.30 nats AND S_share >= 0.50 on the primary arm
                 for all three sample sizes           -> else LSMI_MEASUREMENT_INVALID
  M2 specificity: C1 duplicate-stream must give S_share <= 0.20
  S_floor      := max( q95(S | permutation null), S from C1 duplicate-stream )
  (i)  FUSION_CAPPED   : S <= S_floor AND S_share <= 0.10 AND R > U1 + U2
  (ii) SYNERGY_PRESENT : S >  S_floor AND S_share >= 0.20, replicated on dev and on A2
  (iii) INDETERMINATE  : anything else
AMD-3: S_share is None when I12 < 0.05 nats -> the magnitude clause falls back to S vs S_floor.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(ROOT, "refine-logs", "LSMI_GATE_OUT.json")
R = json.load(open(P))
PRIMARY = sys.argv[1] if len(sys.argv) > 1 else "A1"
READ = "train_crossfit"

V = {"primary_arm": PRIMARY, "primary_read": READ, "rule_source": "LSMI_GATE_RECORD.md 2.5/2.6"}

# ---- M1 power gate -------------------------------------------------------------------
m1 = {}
for k, v in R.get("controls", {}).items():
    if not k.startswith("G1"):
        continue
    d = v[READ]
    m1[k] = {"S": d["S"], "S_share": d["S_share"], "I12": d["I12"],
             "acc_joint_oof": d["acc_joint"], "acc_img_oof": d["acc_img"], "acc_text_oof": d["acc_text"],
             "pass": bool(d["S"] >= 0.30 and (d["S_share"] is not None and d["S_share"] >= 0.50))}
V["M1_power"] = m1
at_our_n = {k: x for k, x in m1.items() if k.startswith("G1_xor_") or k.startswith("G1f_")}
V["M1_pass_at_our_n"] = bool(at_our_n) and all(x["pass"] for x in at_our_n.values())

# ---- M2 specificity ------------------------------------------------------------------
m2 = {}
for ds, e in R.get("datasets", {}).items():
    for key in e:
        if key.startswith("C1_dup"):
            d = e[key][READ]
            m2[f"{ds}/{key}"] = {"S": d["S"], "S_share": d["S_share"], "I12": d["I12"],
                                 "pass": bool(d["S_share"] is None or d["S_share"] <= 0.20)}
V["M2_specificity"] = m2
V["M2_pass"] = all(x["pass"] for x in m2.values()) if m2 else None

# ---- per-dataset decision ------------------------------------------------------------
per = {}
for ds, e in R.get("datasets", {}).items():
    if PRIMARY not in e:
        continue
    a = e[PRIMARY]
    d = a[READ]
    null_q95 = a.get("perm_null_" + READ, {}).get("S", {}).get("q95")
    dup = None
    for key in e:
        if key.startswith("C1_dup"):
            dup = e[key][READ]["S"]
    floor = max([x for x in (null_q95, dup) if x is not None], default=None)
    share = d["S_share"]
    cond_capped = (floor is not None and d["S"] <= floor
                   and (share is None or share <= 0.10) and d["R"] > d["U1"] + d["U2"])
    cond_present = (floor is not None and d["S"] > floor and share is not None and share >= 0.20)
    rep_dev = a["dev"]["S"] > (a.get("perm_null_dev", {}).get("S", {}).get("q95") or float("inf"))
    rep_a2 = None
    if "A2" in e:
        rep_a2 = e["A2"][READ]["S"] > (floor if floor is not None else float("inf"))
    verdict = ("FUSION_CAPPED" if cond_capped else
               "SYNERGY_PRESENT" if (cond_present and rep_dev and rep_a2) else "INDETERMINATE")
    per[ds] = {"S": d["S"], "R": d["R"], "U1": d["U1"], "U2": d["U2"], "I12": d["I12"],
               "S_share": share, "SmR_estimator_free": d["SmR_estimator_free"],
               "perm_null_q95_S": null_q95, "C1_dup_S": dup, "S_floor": floor,
               "R_gt_U1U2": d["R_gt_U1U2"], "cond_capped": cond_capped,
               "cond_present": cond_present, "replicated_dev": bool(rep_dev),
               "replicated_A2": rep_a2, "verdict": verdict,
               "acc_oof": [d["acc_img"], d["acc_text"], d["acc_joint"]]}
V["per_dataset"] = per

if V["M1_pass_at_our_n"] is False:
    V["VERDICT"] = "LSMI_MEASUREMENT_INVALID"
    V["VERDICT_REASON"] = ("M1 power gate failed at our sample sizes: the estimator does not "
                           "recover a construction whose synergy is known and maximal, so a null "
                           "on the real datasets is uninterpretable as evidence of no synergy.")
elif V["M2_pass"] is False:
    V["VERDICT"] = "LSMI_MEASUREMENT_INVALID"
    V["VERDICT_REASON"] = "M2 specificity gate failed: a provably synergy-free pair reads synergistic."
else:
    vs = {ds: x["verdict"] for ds, x in per.items()}
    V["VERDICT"] = ("FUSION_CAPPED" if all(v == "FUSION_CAPPED" for v in vs.values())
                    else "SYNERGY_PRESENT" if any(v == "SYNERGY_PRESENT" for v in vs.values())
                    else "INDETERMINATE")
    V["VERDICT_REASON"] = json.dumps(vs)

R["verdict"] = V
json.dump(R, open(P, "w"), indent=1)
print(json.dumps({k: V[k] for k in ("VERDICT", "VERDICT_REASON", "M1_pass_at_our_n", "M2_pass")}, indent=1))
for ds, x in per.items():
    print(f"{ds:8s} S={x['S']:+.4f} floor={x['S_floor']} share={x['S_share']} -> {x['verdict']}")
