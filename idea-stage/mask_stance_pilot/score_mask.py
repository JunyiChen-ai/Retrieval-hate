"""MASK STANCE PILOT -- frozen scorer (rules from idea-stage/MASK_STANCE_PILOT_FREEZE.md §5).

The gold collapse, the three views and the P1/P2/P3 formulas are copied VERBATIM from
idea-stage/stance_pilot/score.py so that the numbers are comparable line for line with the
0.257 / 0.167 baselines. The only additions are:
  * the verdict is P1 AND P2 (P3 is computed and printed but does not gate), per this pilot's
    task specification;
  * the frozen descriptive strata of freeze doc §5.2 (>=1 masked span vs 0 masked spans);
  * a per-item join against the previous pilot's two arms.
Nothing here may be edited after the eval batch is fetched.
"""
import argparse
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SP = os.path.join(ROOT, "idea-stage", "stance_pilot")

N_SET = {"quotes_mentions", "condemns", "reports"}
DISTANCING = {"quotes_mentions", "condemns"}
STANCES = ["endorses", "quotes_mentions", "condemns", "reports", "no_hate_content"]
FRAME_DS = ["HateMM", "MHC", "MHC_zh"]
TEXT_DS = ["ImpliHateVid"]
ALL_DS = FRAME_DS + TEXT_DS
P1_BAR, P2_BAR = 0.70, 0.15


def population_counts():
    A = json.load(open(os.path.join(ROOT, "idea-stage", "r5_phase_a.json")))["A2_error_attribution"]
    out = {}
    for ds in ALL_DS:
        err = set(A[ds]["err_ids"])
        pos = neg = 0
        for line in open(os.path.join(ROOT, "data", "gt", ds, "test.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["id"] in err:
                continue
            pos += int(r["label"]) == 1
            neg += int(r["label"]) == 0
        out[ds] = (pos, neg)
    return out


def read_pred(path, sample):
    rows = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        k = (r["dataset"], r["id"])
        if k not in sample:
            continue
        p = r.get("parsed") or {}
        s = p.get("stance")
        rows[k] = {"stance": s if s in STANCES else None, "voice": p.get("primary_voice"),
                   "surface": p.get("hate_surface_present"), "evidence": p.get("evidence"),
                   "parse": r["parse"], "usage": r.get("usage")}
    return rows


def load(tag):
    sample = {(x["dataset"], x["id"]): x for x in
              json.load(open(os.path.join(SP, "sample.json")))["eval"]}
    mask = {}
    for line in open(os.path.join(HERE, f"masked_{tag}.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        mask[(r["dataset"], r["id"])] = r
    cur = read_pred(os.path.join(HERE, f"pred_{tag}.jsonl"), sample)
    b1 = read_pred(os.path.join(SP, "pred_strong.jsonl"), sample)
    b2 = read_pred(os.path.join(SP, "pred_fb2.jsonl"), sample)
    rows = []
    for k, v in cur.items():
        m = mask.get(k, {})
        rep = m.get("report", {})
        rows.append({**sample[k], **v,
                     "n_placeholders": rep.get("n_placeholders", 0),
                     "masked_frac": rep.get("masked_frac", 0.0),
                     "n_spans": rep.get("n_spans", 0),
                     "unmatched": rep.get("unmatched", 0),
                     "extract_parse": m.get("extract_parse"),
                     "base_r1": (b1.get(k) or {}).get("stance"),
                     "base_fb2": (b2.get(k) or {}).get("stance")})
    return rows


def correct(r, key="stance"):
    if r["group"] == "S_FP":
        return r[key] in N_SET
    if r["group"] == "S_FN":
        return r[key] == "endorses"
    return None


def push(stance):
    if stance == "endorses":
        return +1
    if stance in N_SET or stance == "no_hate_content":
        return -1
    return 0


def view(rows, datasets, pop, name):
    rows = [r for r in rows if r["dataset"] in datasets]
    g = defaultdict(list)
    for r in rows:
        g[r["group"]].append(r)
    S = g["S_FP"] + g["S_FN"]
    ch, cn = g["CTRL_HATE"], g["CTRL_NONHATE"]
    R = {"view": name, "datasets": datasets, "n": len(rows), "n_S": len(S),
         "n_S_FP": len(g["S_FP"]), "n_S_FN": len(g["S_FN"]),
         "n_ctrl_hate": len(ch), "n_ctrl_nonhate": len(cn),
         "parse_ok": sum(1 for r in rows if r["parse"] == "ok" and r["stance"])}

    R["P1_acc_S"] = sum(1 for r in S if correct(r)) / max(1, len(S))
    R["P1_acc_S_FP"] = sum(1 for r in g["S_FP"] if correct(r)) / max(1, len(g["S_FP"]))
    R["P1_acc_S_FN"] = sum(1 for r in g["S_FN"] if correct(r)) / max(1, len(g["S_FN"]))
    R["P1_pass"] = R["P1_acc_S"] >= P1_BAR

    R["P2_false_distancing"] = sum(1 for r in ch if r["stance"] in DISTANCING) / max(1, len(ch))
    R["P2_pass"] = R["P2_false_distancing"] <= P2_BAR
    R["ctrl_nonhate_false_endorse"] = sum(1 for r in cn if r["stance"] == "endorses") / max(1, len(cn))

    gains = (sum(1 for r in g["S_FP"] if push(r["stance"]) == -1)
             + sum(1 for r in g["S_FN"] if push(r["stance"]) == +1))
    dmg, per_ds = 0.0, {}
    for ds in datasets:
        h = [r for r in ch if r["dataset"] == ds]
        n = [r for r in cn if r["dataset"] == ds]
        rh = sum(1 for r in h if push(r["stance"]) == -1) / max(1, len(h))
        rn = sum(1 for r in n if push(r["stance"]) == +1) / max(1, len(n))
        d = rh * pop[ds][0] + rn * pop[ds][1]
        per_ds[ds] = {"loss_rate_hate": rh, "loss_rate_nonhate": rn,
                      "projected_damage": round(d, 2)}
        dmg += d
    R["P3_gains"] = gains
    R["P3_projected_damage"] = round(dmg, 2)
    R["P3_net"] = round(gains - dmg, 2)
    R["P3_pass"] = R["P3_net"] > 0
    R["P3_per_dataset"] = per_ds
    R["verdict"] = "PASS" if (R["P1_pass"] and R["P2_pass"]) else "FAIL"     # P1 AND P2 only

    R["stance_hist"] = {k: dict(Counter(r["stance"] for r in v)) for k, v in g.items()}
    R["baseline_r1_hist"] = {k: dict(Counter(r["base_r1"] for r in v)) for k, v in g.items()}
    R["per_dataset_P1"] = {}
    for ds in datasets:
        s = [r for r in S if r["dataset"] == ds]
        if s:
            R["per_dataset_P1"][ds] = {
                "n": len(s), "acc": round(sum(1 for r in s if correct(r)) / len(s), 3),
                "acc_r1": round(sum(1 for r in s if correct(r, "base_r1")) / len(s), 3),
                "acc_fb2": round(sum(1 for r in s if correct(r, "base_fb2")) / len(s), 3)}

    # baselines recomputed on EXACTLY this row set (so denominators match)
    for key, tag in (("base_r1", "r1"), ("base_fb2", "fb2")):
        have = [r for r in S if r[key] is not None]
        R[f"P1_acc_S_{tag}_same_rows"] = (sum(1 for r in have if correct(r, key)) / max(1, len(have)))
        R[f"n_S_{tag}"] = len(have)
        ch_have = [r for r in ch if r[key] is not None]
        R[f"P2_{tag}_same_rows"] = sum(1 for r in ch_have if r[key] in DISTANCING) / max(1, len(ch_have))

    # frozen descriptive strata (freeze §5.2)
    strata = {}
    for lab, sel in (("with_placeholder", lambda r: r["n_placeholders"] > 0),
                     ("no_placeholder", lambda r: r["n_placeholders"] == 0)):
        sub = [r for r in S if sel(r)]
        strata[lab] = {
            "n": len(sub),
            "acc": round(sum(1 for r in sub if correct(r)) / max(1, len(sub)), 3),
            "acc_r1": round(sum(1 for r in sub if correct(r, "base_r1")) / max(1, len(sub)), 3),
            "n_S_FP": sum(1 for r in sub if r["group"] == "S_FP"),
            "n_S_FN": sum(1 for r in sub if r["group"] == "S_FN"),
            "hist": dict(Counter(r["stance"] for r in sub))}
    R["strata_by_masking"] = strata

    R["flips"] = {"S_wrong_to_right": sorted(r["id"] for r in S
                                             if correct(r) and not correct(r, "base_r1")),
                  "S_right_to_wrong": sorted(r["id"] for r in S
                                             if not correct(r) and correct(r, "base_r1"))}
    tin = sum((r["usage"] or {}).get("prompt_tokens", 0) for r in rows)
    tout = sum((r["usage"] or {}).get("completion_tokens", 0) for r in rows)
    R["tokens_stance_step"] = {"in": tin, "out": tout,
                               "in_per_item": round(tin / max(1, len(rows)), 1)}
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    rows = load(a.tag)
    pop = population_counts()
    out = {"tag": a.tag, "n_rows": len(rows),
           "A_frames_primary": view(rows, FRAME_DS, pop, "A_frames_primary"),
           "B_all99_as_frozen": view(rows, ALL_DS, pop, "B_all99_as_frozen"),
           "C_textonly_descriptive": view(rows, TEXT_DS, pop, "C_textonly_descriptive")}
    out["VERDICT"] = out["A_frames_primary"]["verdict"]
    # extraction-quality roll-up over the eval batch
    ns = sum(r["n_spans"] for r in rows)
    out["extraction"] = {
        "n_items": len(rows),
        "items_with_placeholder": sum(1 for r in rows if r["n_placeholders"] > 0),
        "total_spans": ns,
        "unmatched_spans": sum(r["unmatched"] for r in rows),
        "unmatched_rate": round(sum(r["unmatched"] for r in rows) / max(1, ns), 4),
        "mean_masked_frac_when_masked": round(
            sum(r["masked_frac"] for r in rows if r["n_placeholders"] > 0)
            / max(1, sum(1 for r in rows if r["n_placeholders"] > 0)), 3),
        "extract_parse_fail": sum(1 for r in rows if r["extract_parse"] != "ok")}
    out["per_item"] = sorted(
        [{"id": r["id"], "ds": r["dataset"], "grp": r["group"], "pl": r["n_placeholders"],
          "mfrac": r["masked_frac"], "masked": r["stance"], "r1": r["base_r1"],
          "fb2": r["base_fb2"], "ok": correct(r)} for r in rows],
        key=lambda x: (x["grp"], x["ds"], x["id"]))
    json.dump(out, open(os.path.join(HERE, f"score_{a.tag}.json"), "w"), indent=1)

    for k in ("A_frames_primary", "B_all99_as_frozen", "C_textonly_descriptive"):
        v = out[k]
        print("=" * 78)
        print(k, "n=", v["n"], "parse_ok=", v["parse_ok"])
        print(f"  P1 acc_S={v['P1_acc_S']:.3f} (FP {v['P1_acc_S_FP']:.3f} n={v['n_S_FP']} / "
              f"FN {v['P1_acc_S_FN']:.3f} n={v['n_S_FN']})  bar 0.70 -> {v['P1_pass']}")
        print(f"     baselines on the same rows: r1={v['P1_acc_S_r1_same_rows']:.3f} "
              f"(n={v['n_S_r1']})  fb2={v['P1_acc_S_fb2_same_rows']:.3f} (n={v['n_S_fb2']})")
        print(f"  P2 false_distancing={v['P2_false_distancing']:.3f} bar 0.15 -> {v['P2_pass']}"
              f"   [r1={v['P2_r1_same_rows']:.3f}; ctrl_nonhate false_endorse="
              f"{v['ctrl_nonhate_false_endorse']:.3f}]")
        print(f"  P3 (reference only) gains={v['P3_gains']} damage={v['P3_projected_damage']} "
              f"net={v['P3_net']}")
        print("  VERDICT (P1 AND P2):", v["verdict"])
        print("  per-dataset P1:", json.dumps(v["per_dataset_P1"]))
        print("  stance hist  :", json.dumps(v["stance_hist"], ensure_ascii=False))
        print("  r1 hist      :", json.dumps(v["baseline_r1_hist"], ensure_ascii=False))
        print("  strata       :", json.dumps(v["strata_by_masking"], ensure_ascii=False))
        print("  flips        :", json.dumps(v["flips"]))
        print("  tokens(step2):", json.dumps(v["tokens_stance_step"]))
    print("\nextraction:", json.dumps(out["extraction"]))
    print("\nPRIMARY VERDICT (view A, frame-bearing):", out["VERDICT"])


if __name__ == "__main__":
    main()
