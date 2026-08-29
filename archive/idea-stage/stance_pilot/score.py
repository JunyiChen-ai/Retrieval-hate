"""STANCE PILOT -- frozen scorer (rules from idea-stage/STANCE_PILOT_FREEZE.md §6 + §6.1 D1).

Three views, per deviation D1 (user ruling 2026-08-12, recorded before any batch result existed):
  A  frame-bearing subset  (HateMM + MHC + MHC_zh, n=72)  -> THE VERDICT
  B  as-frozen all 99 (adds the 27 transcript-only ImpliHateVid items) -> reference
  C  text-only ImpliHateVid subset (n=27) -> descriptive only

The three thresholds and the gold collapse are frozen and must not be edited.
"""
import argparse
import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

N_SET = {"quotes_mentions", "condemns", "reports"}
DISTANCING = {"quotes_mentions", "condemns"}
STANCES = ["endorses", "quotes_mentions", "condemns", "reports", "no_hate_content"]
FRAME_DS = ["HateMM", "MHC", "MHC_zh"]
TEXT_DS = ["ImpliHateVid"]
ALL_DS = FRAME_DS + TEXT_DS

P1_BAR, P2_BAR = 0.70, 0.15


def population_counts():
    """per-dataset (#correct hate, #correct non-hate) over the test splits."""
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


def load(tag):
    sample = {(x["dataset"], x["id"]): x for x in
              json.load(open(os.path.join(HERE, "sample.json")))["eval"]}
    rows = []
    for line in open(os.path.join(HERE, f"pred_{tag}.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        it = sample[(r["dataset"], r["id"])]
        p = r.get("parsed") or {}
        s = p.get("stance")
        rows.append({**it, "stance": s if s in STANCES else None,
                     "voice": p.get("primary_voice"), "surface": p.get("hate_surface_present"),
                     "target": p.get("target"), "evidence": p.get("evidence"),
                     "parse": r["parse"], "usage": r.get("usage")})
    return rows


def correct(r):
    """frozen gold collapse (freeze doc §5). None = no defined gold (control groups)."""
    if r["group"] == "S_FP":
        return r["stance"] in N_SET
    if r["group"] == "S_FN":
        return r["stance"] == "endorses"
    return None


def push(stance):
    """frozen naive decision rule (§6 P3): +1 towards hate, -1 towards non-hate."""
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
    R = {"view": name, "datasets": datasets, "n": len(rows),
         "n_S": len(S), "n_S_FP": len(g["S_FP"]), "n_S_FN": len(g["S_FN"]),
         "n_ctrl_hate": len(ch), "n_ctrl_nonhate": len(cn),
         "parse_ok": sum(1 for r in rows if r["parse"] == "ok" and r["stance"])}

    R["P1_acc_S"] = sum(1 for r in S if correct(r)) / len(S)
    R["P1_acc_S_FP"] = sum(1 for r in g["S_FP"] if correct(r)) / max(1, len(g["S_FP"]))
    R["P1_acc_S_FN"] = sum(1 for r in g["S_FN"] if correct(r)) / max(1, len(g["S_FN"]))
    R["P1_pass"] = R["P1_acc_S"] >= P1_BAR

    R["P2_false_distancing"] = sum(1 for r in ch if r["stance"] in DISTANCING) / len(ch)
    R["P2_pass"] = R["P2_false_distancing"] <= P2_BAR
    R["ctrl_nonhate_false_endorse"] = sum(1 for r in cn if r["stance"] == "endorses") / len(cn)

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
                      "n_correct_hate": pop[ds][0], "n_correct_nonhate": pop[ds][1],
                      "projected_damage": round(d, 2)}
        dmg += d
    R["P3_gains"] = gains
    R["P3_projected_damage"] = round(dmg, 2)
    R["P3_net"] = round(gains - dmg, 2)
    R["P3_pass"] = R["P3_net"] > 0
    R["P3_per_dataset"] = per_ds
    R["verdict"] = "PASS" if (R["P1_pass"] and R["P2_pass"] and R["P3_pass"]) else "FAIL"

    R["stance_hist"] = {k: dict(Counter(r["stance"] for r in v)) for k, v in g.items()}
    R["per_dataset_P1"] = {}
    for ds in datasets:
        s = [r for r in S if r["dataset"] == ds]
        if s:
            R["per_dataset_P1"][ds] = {"n": len(s),
                                       "acc": round(sum(1 for r in s if correct(r)) / len(s), 3)}
    tin = sum((r["usage"] or {}).get("prompt_tokens", 0) for r in rows)
    tout = sum((r["usage"] or {}).get("completion_tokens", 0) for r in rows)
    R["tokens"] = {"in": tin, "out": tout, "in_per_item": round(tin / max(1, len(rows)), 1),
                   "out_per_item": round(tout / max(1, len(rows)), 1)}
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
    json.dump(out, open(os.path.join(HERE, f"score_{a.tag}.json"), "w"), indent=1)
    for k in ("A_frames_primary", "B_all99_as_frozen", "C_textonly_descriptive"):
        v = out[k]
        print("=" * 70)
        print(k, "n=", v["n"], "parse_ok=", v["parse_ok"])
        print(f"  P1 acc_S={v['P1_acc_S']:.3f} (FP {v['P1_acc_S_FP']:.3f} n={v['n_S_FP']} / "
              f"FN {v['P1_acc_S_FN']:.3f} n={v['n_S_FN']})  bar 0.70 -> {v['P1_pass']}")
        print(f"  P2 false_distancing={v['P2_false_distancing']:.3f} bar 0.15 -> {v['P2_pass']}"
              f"   [ctrl_nonhate false_endorse={v['ctrl_nonhate_false_endorse']:.3f}]")
        print(f"  P3 gains={v['P3_gains']} damage={v['P3_projected_damage']} "
              f"net={v['P3_net']} -> {v['P3_pass']}")
        print("  VERDICT", v["verdict"])
        print("  per-dataset P1:", json.dumps(v["per_dataset_P1"]))
        print("  stance hist:", json.dumps(v["stance_hist"], ensure_ascii=False))
        print("  P3 per ds:", json.dumps(v["P3_per_dataset"]))
        print("  tokens:", json.dumps(v["tokens"]))
    print("\nPRIMARY VERDICT (view A, frame-bearing 72):", out["VERDICT"])


if __name__ == "__main__":
    main()
