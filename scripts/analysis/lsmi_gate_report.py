#!/usr/bin/env python
"""Render the LSMI gate tables from refine-logs/LSMI_GATE_OUT.json (no computation)."""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = json.load(open(os.path.join(ROOT, "refine-logs", "LSMI_GATE_OUT.json")))


def f(x, n=4):
    return "n/a" if x is None else (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x))


def row(name, d):
    return (f"| {name} | {d['n']} | {f(d['I1'])} | {f(d['I2'])} | {f(d['I12'])} | {f(d['R'])} | "
            f"{f(d['U1'])} | {f(d['U2'])} | **{f(d['S'])}** | {f(d['S_share'],3)} | "
            f"{f(d['SmR_estimator_free'])} | {d['acc_img']:.3f}/{d['acc_text']:.3f}/{d['acc_joint']:.3f} |")


HDR = ("| cell | n | I1 | I2 | I12 | R | U1 | U2 | **S** | S_share | S−R | acc i/t/j |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|")

print("### CONTROLS / MACHINERY GATES\n")
print(HDR)
for k, v in R.get("controls", {}).items():
    for sp in ("train_crossfit", "train_insample", "dev"):
        if sp in v:
            print(row(f"{k} [{sp}]", v[sp]))
print()

print("### DATASETS\n")
for ds, e in R.get("datasets", {}).items():
    print(f"\n**{ds}** — {e.get('lineage','')} — n {e.get('n')}\n")
    print(HDR)
    for arm in ("A1", "A2", "A3", "A4", "A5", "C1_dup_img", "C2_splithalf_img"):
        if arm not in e:
            continue
        for sp in ("train_crossfit", "train_insample", "dev"):
            if sp in e[arm]:
                print(row(f"{arm} [{sp}]", e[arm][sp]))
    a1 = e.get("A1", {})
    for nm in ("perm_null_train_crossfit", "perm_null_dev"):
        if nm in a1:
            p = a1[nm]
            print(f"\n`{nm}` (B={p['S']['n']}): S mean {f(p['S']['mean'])} sd {f(p['S']['sd'])} "
                  f"q95 **{f(p['S']['q95'])}** max {f(p['S']['max'])} | "
                  f"I12 mean {f(p['I12']['mean'])} | R mean {f(p['R']['mean'])}")
    if "fidelity_maxabs_vs_released" in a1:
        print(f"\nfidelity vs released LSMI_estimation (A1 in-sample): maxabs = {a1['fidelity_maxabs_vs_released']:.2e}")
