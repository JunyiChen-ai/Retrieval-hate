"""MASK STANCE PILOT -- render the reporting tables asked for by the task, from score_<tag>.json.

Pure offline formatting; computes nothing that score_mask.py did not already compute, except the
cost roll-up, which sums the `usage` fields of every cached call this pilot made.
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SP = os.path.join(ROOT, "idea-stage", "stance_pilot")
STANCES = ["endorses", "quotes_mentions", "condemns", "reports", "no_hate_content"]


def tokens(path):
    i = o = n = 0
    if not os.path.exists(path):
        return 0, 0, 0
    for line in open(path, encoding="utf-8"):
        u = (json.loads(line).get("usage") or {})
        i += u.get("prompt_tokens", 0) or 0
        o += u.get("completion_tokens", 0) or 0
        n += 1
    return i, o, n


def main(tag):
    S = json.load(open(os.path.join(HERE, f"score_{tag}.json")))
    A, B, C = S["A_frames_primary"], S["B_all99_as_frozen"], S["C_textonly_descriptive"]

    print("## 1. Headline — masked vs unmasked, same items, same model\n")
    print("| view | n_S | masked P1 | round-1 P1 | fallback-② P1 | Δ vs r1 | bar |")
    print("|---|---|---|---|---|---|---|")
    for v, nm in ((A, "A frame-bearing (primary)"), (B, "B all 99"), (C, "C text-only")):
        print(f"| {nm} | {v['n_S']} | **{v['P1_acc_S']:.3f}** | {v['P1_acc_S_r1_same_rows']:.3f} | "
              f"{v['P1_acc_S_fb2_same_rows']:.3f} | "
              f"{v['P1_acc_S'] - v['P1_acc_S_r1_same_rows']:+.3f} | ≥0.70 |")

    print("\n## 2. By S sub-bucket, view A\n")
    print("| cell | n | masked | round 1 |")
    print("|---|---|---|---|")
    print(f"| S_FP | {A['n_S_FP']} | {A['P1_acc_S_FP']:.3f} | — |")
    print(f"| S_FN | {A['n_S_FN']} | {A['P1_acc_S_FN']:.3f} | — |")

    print("\n## 3. Per dataset, S-bucket accuracy\n")
    print("| dataset | n_S | masked | round 1 | fallback ② |")
    print("|---|---|---|---|---|")
    for ds, d in B["per_dataset_P1"].items():
        print(f"| {ds} | {d['n']} | {d['acc']:.3f} | {d['acc_r1']:.3f} | {d['acc_fb2']:.3f} |")

    print("\n## 4. Control damage (view A)\n")
    print(f"P2 false_distancing on CTRL_HATE = **{A['P2_false_distancing']:.3f}** "
          f"(round 1 {A['P2_r1_same_rows']:.3f}), bar ≤ 0.15 -> pass={A['P2_pass']}")
    print(f"CTRL_NONHATE false `endorses` = {A['ctrl_nonhate_false_endorse']:.3f}")

    print("\n## 5. Five-class stance distribution, view A (masked -> round 1)\n")
    print("| group | " + " | ".join(STANCES) + " |")
    print("|---" * (len(STANCES) + 1) + "|")
    for g in ("S_FP", "S_FN", "CTRL_HATE", "CTRL_NONHATE"):
        cells = []
        for s in STANCES:
            m = A["stance_hist"].get(g, {}).get(s, 0)
            r = A["baseline_r1_hist"].get(g, {}).get(s, 0)
            cells.append(f"{m} ({r})")
        print(f"| {g} | " + " | ".join(cells) + " |")
    tot_m = Counter()
    tot_r = Counter()
    for g in ("S_FP", "S_FN", "CTRL_HATE", "CTRL_NONHATE"):
        tot_m.update(A["stance_hist"].get(g, {}))
        tot_r.update(A["baseline_r1_hist"].get(g, {}))
    print("| **total** | " + " | ".join(f"**{tot_m.get(s,0)}** ({tot_r.get(s,0)})"
                                        for s in STANCES) + " |")

    print("\n## 6. Frozen stratification by whether masking could act (view A, S bucket)\n")
    print("| stratum | n | n_S_FP | n_S_FN | masked P1 | round-1 P1 | stance hist |")
    print("|---|---|---|---|---|---|---|")
    for k, v in A["strata_by_masking"].items():
        print(f"| {k} | {v['n']} | {v['n_S_FP']} | {v['n_S_FN']} | {v['acc']:.3f} | "
              f"{v['acc_r1']:.3f} | {json.dumps(v['hist'], ensure_ascii=False)} |")

    print("\n## 7. Item-level flips (view A, S bucket)\n")
    print("wrong->right:", A["flips"]["S_wrong_to_right"])
    print("right->wrong:", A["flips"]["S_right_to_wrong"])

    print("\n## 8. Extraction / masking quality over the 99-item batch\n")
    print(json.dumps(S["extraction"], indent=1))

    print("\n## 9. Cost (measured tokens)\n")
    rows = [("smoke step 1 extraction (8 items, realtime)", "extract_s_e1.jsonl", HERE),
            ("smoke step 2 masked stance (8 items, realtime)", "pred_s_e1.jsonl", HERE),
            ("eval step 1 extraction (99 items, Batch)", "extract_m1.jsonl", HERE),
            ("eval step 2 masked stance (99 items, Batch)", "pred_m1.jsonl", HERE)]
    ti = to = 0
    print("| run | endpoint | items | input tok | output tok |")
    print("|---|---|---|---|---|")
    for nm, f, d in rows:
        i, o, n = tokens(os.path.join(d, f))
        ti += i
        to += o
        print(f"| {nm} | {'Batch' if 'm1' in f else 'realtime'} | {n} | {i:,} | {o:,} |")
    print(f"| synthetic reachability probe (5 items x 2 calls) | realtime | 5 | 9,172 | 558 |")
    ti += 9172
    to += 558
    print(f"| **total this pilot** | | | **{ti:,}** | **{to:,}** |")
    print(f"\nAt the same assumed list price used in STANCE_PILOT_RESULT §7 "
          f"(¥0.002/1K in, ¥0.008/1K out, Batch at 50%): upper bound "
          f"¥{ti/1000*0.002 + to/1000*0.008:.2f} ≈ USD {(ti/1000*0.002 + to/1000*0.008)/7.1:.3f}")

    print("\n## VERDICT:", S["VERDICT"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "m1")
