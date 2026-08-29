"""CONTRAST STANCE PILOT -- reporting tables (pure formatting of score_<tag>.json)."""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, f"score_{sys.argv[1]}.json")))


def f(x, n=3):
    return "n/a" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


A = S["A_frames_primary"]
print("## headline (view A, variant 1, smoke items excluded)")
print("| metric | value | n | bar | pass |")
print("|---|---|---|---|---|")
print(f"| M1 binary stance acc on S | {f(A['M1_acc_S'])} | {A['M1_n']} | >= 0.70 | {A['M1_pass']} |")
print(f"| M2 CTRL_HATE answered OPPOSE | {f(A['M2_ctrl_hate_said_OPPOSE'])} | {A['M2_n']} | <= 0.15 | {A['M2_pass']} |")
print(f"| verdict | {A['VERDICT']} | | | |")
print(f"\np(M1 vs two-way chance 0.5) = {A['M1_p_vs_chance_0.5']}")

print("\n## three methods, same rows")
print("| method | task | chance | acc on S |")
print("|---|---|---|---|")
print(f"| round 1 direct classification | 5-way | 0.20 | {f(A['r1_5way_acc_S'])} (n={A['r1_5way_n']}) |")
print(f"| round 2 masked classification | 5-way | 0.20 | {f(A['mask_5way_acc_S'])} (n={A['mask_5way_n']}) |")
print(f"| round 1, binarised | 2-way | 0.50 | {f(A['r1_binary_acc_S'])} (n={A['r1_binary_n']}) |")
print(f"| round 2, binarised | 2-way | 0.50 | {f(A['mask_binary_acc_S'])} (n={A['mask_binary_n']}) |")
print(f"| **this round, pinned-comment contrast** | 2-way | 0.50 | **{f(A['M1_acc_S'])}** (n={A['M1_n']}) |")

print("\n## cells / strata (view A, variant 1)")
print(f"S_FP {f(A['acc_S_FP'])} (n={A['n_S_FP']})   S_FN {f(A['acc_S_FN'])} (n={A['n_S_FN']})")
print("\nby hand-coded voice form:")
print("| voice | meaning | n | acc | acc (round 1 binarised) |")
print("|---|---|---|---|---|")
for k, lab in (("OWN", "author speaks (作者有话)"), ("NOT_OWN", "archive/third party (作者无话)"),
               ("UNDET", "undeterminable"), ("UNCODED", "not hand-coded")):
    v = A["by_voice"][k]
    print(f"| {k} | {lab} | {v['n']} | {f(v['acc'])} | {f(v['acc_r1_binary'])} |")
print("\nby dataset:", json.dumps(A["by_dataset"]))
print("call histogram:", json.dumps(A["call_hist"]))

print("\n## the 5 template pairs")
print("| pair | acc on S | endorse rate (all items) |")
print("|---|---|---|")
for p in sorted(A["per_pair_acc_on_S"], key=int):
    print(f"| {p} | {A['per_pair_acc_on_S'][p]['acc_on_S']} | "
          f"{A['per_pair_endorse_rate'][p]['endorse_rate_all_items']} |")
print("vote splits on S:", json.dumps(A["vote_split_hist_on_S"]),
      "acc by split:", json.dumps(A["acc_by_split"]))
print("position bias, slot-A rate:", A["position_bias_slotA_rate"], "over", A["position_bias_n"], "calls")

print("\n## lexical overlap")
print(json.dumps(A["overlap"], indent=1))

print("\n## variant 1 vs 2, paired on target-bearing frame-bearing rows")
for k in ("A_variant1_on_v2_rows", "A_variant2"):
    v = S[k]
    print(f"{k:<26} acc_S={f(v['M1_acc_S'])} n={v['M1_n']}  "
          f"S_FP={f(v['acc_S_FP'])}/{v['n_S_FP']} S_FN={f(v['acc_S_FN'])}/{v['n_S_FN']}  "
          f"M2={f(v['M2_ctrl_hate_said_OPPOSE'])}")
per = {(x["id"]): x for x in S["per_item"]}
agree = Counter()
for x in S["per_item"]:
    if x["v2"] and x["v1"]:
        agree[(x["v1"], x["v2"])] += 1
print("v1 x v2 agreement over all items with both:", json.dumps({f"{a}->{b}": c for (a, b), c in agree.items()}))

print("\n## other views")
for k in ("A_frames_primary__incl_smoke_items", "B_all99", "C_textonly"):
    v = S[k]
    print(f"{k:<38} acc_S={f(v['M1_acc_S'])} n={v['M1_n']}  M2={f(v['M2_ctrl_hate_said_OPPOSE'])}"
          f"  ctrl_nonhate->ENDORSE={f(v['ctrl_nonhate_said_ENDORSE'])}")

print("\n## cost / losses")
print(json.dumps(S["cost"]))
print("losses:", json.dumps(S["losses"]))
