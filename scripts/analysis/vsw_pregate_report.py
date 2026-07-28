#!/usr/bin/env python
"""
vsw_pregate_report.py -- REPORTING ONLY for the VSW $0 pregate.
Record: refine-logs/VSW_PREGATE_RECORD.md.

It computes NO arm and fits NOTHING.  It merges the frozen script's per-stage outputs
(`vsw_main_{ds}_OUT.json`, `vsw_perm_{ds}_OUT.json`, `vsw_selftest_OUT.json`) into
`vsw_pregate_OUT.json`, derives the permutation p-values from the stored null draws,
renders every frozen bar's verdict, and prints every number at 4 dp.  Nothing here is
part of the freeze.
"""
import json
import os
import sys

import numpy as np

REPO = "/data/jehc223/RGCL"
AN = os.path.join(REPO, "scripts/analysis")
KEYS = ["hatemm", "zh", "en"]
NICE = {"hatemm": "HateMM", "zh": "MHC-ZH", "en": "MHC-EN"}
FAMS = ["pow", "exp", "lin"]
PRIMARY = "pow"
BAR_MAIN = 0.030          # K-VSW-1
BAR_INTEREST = 0.010      # K-VSW-0
ER_BAR = 1.2              # K-VSW-2 outcome (a)
MIN_CHANGED = 20          # K-VSW-2 outcome (a) small-count guard
DEG_KILL = 0.95


def pval(nulls, obs):
    n = np.asarray(nulls, dtype="float64")
    return round(float((1 + int((n >= obs - 1e-12).sum())) / (len(n) + 1)), 4)


def lamstr(r):
    return "inf" if r["lam_is_inf"] else f"{r['lam']:g}"


def main():
    OUT = {"datasets": {}, "meta": {"report_script": os.path.abspath(__file__)}}
    st = os.path.join(AN, "vsw_selftest_OUT.json")
    if os.path.exists(st):
        OUT["selftest"] = json.load(open(st))["selftest"]

    for k in KEYS:
        mp = os.path.join(AN, f"vsw_main_{k}_OUT.json")
        if not os.path.exists(mp):
            print(f"[{k}] MAIN MISSING")
            continue
        Mn = json.load(open(mp))
        D = {"meta": Mn["meta"], "per_fold": Mn["per_fold"],
             "parity_lambda0": Mn["parity_lambda0"],
             "parity_f95": {"n_gates": Mn["parity_f95"]["n_gates"],
                            "n_pass": Mn["parity_f95"]["n_pass"]},
             "result": Mn["result"]}
        pp = os.path.join(AN, f"vsw_perm_{k}_OUT.json")
        if os.path.exists(pp):
            PR = json.load(open(pp))["draws"]
            D["n_perm_draws"] = len(PR)
            D["perm"] = {}
            for fam in FAMS:
                obs = Mn["result"]["arms"][f"VSW_{fam}"]["dacc"]
                nl = [r["arms"][f"VSW_{fam}"]["dacc"] for r in PR]
                D["perm"][f"VSW_{fam}"] = {
                    "observed": obs, "p": pval(nl, obs), "n": len(nl),
                    "null_mean": round(float(np.mean(nl)), 4),
                    "null_sd": round(float(np.std(nl)), 4),
                    "null_max": round(float(np.max(nl)), 4),
                    "null_q95": round(float(np.quantile(nl, 0.95)), 4),
                    "null_frac_ge_zero": round(float(np.mean(np.asarray(nl) >= 0)), 4)}
            # per-lambda nulls on the PRIMARY family (the whole K-VSW-2 curve)
            cur = Mn["result"]["curve"][PRIMARY]
            D["perm"]["curve_" + PRIMARY] = []
            for i, row in enumerate(cur):
                nl = [r["curve"][PRIMARY][i]["dacc"] for r in PR]
                D["perm"]["curve_" + PRIMARY].append(
                    {"lam": row["lam"], "lam_is_inf": row["lam_is_inf"],
                     "observed": row["dacc"], "p": pval(nl, row["dacc"]),
                     "null_mean": round(float(np.mean(nl)), 4)})
        OUT["datasets"][k] = D

    # ---------------------------------------------------------------- verdicts
    V = {}
    ds = OUT["datasets"]
    for fam in FAMS:
        hits = [k for k in ds if ds[k]["result"]["arms"][f"VSW_{fam}"]["dacc"] >= BAR_MAIN]
        V[f"K-VSW-1_{fam}"] = {"datasets_at_bar": hits, "n": len(hits),
                               "PASS": bool(len(hits) >= 2)}
    V["K-VSW-1"] = {"PASS": bool(any(V[f"K-VSW-1_{f}"]["PASS"] for f in FAMS)),
                    "bar": BAR_MAIN,
                    "best_dacc_any_family_any_dataset": (
                        round(max(ds[k]["result"]["arms"][f"VSW_{f}"]["dacc"]
                                  for k in ds for f in FAMS), 4) if ds else None)}
    k0 = []
    for k in ds:
        a = ds[k]["result"]["arms"][f"VSW_{PRIMARY}"]
        fs = a["foldsigns"]
        if (a["dacc"] >= BAR_INTEREST and fs.count("-") == 0 and fs.count("+") >= 3):
            k0.append(k)
    V["K-VSW-0"] = {"bar": BAR_INTEREST, "datasets": k0, "PASS": bool(len(k0) >= 1)}

    # K-VSW-2 outcome
    out_a = []
    for k in ds:
        rows = [r for r in ds[k]["result"]["curve"][PRIMARY]
                if r["changed"] >= MIN_CHANGED and r["exchange_rate"] is not None
                and r["exchange_rate"] > ER_BAR]
        if rows:
            out_a.append({"dataset": k,
                          "lams": [lamstr(r) for r in rows],
                          "max_er": round(max(r["exchange_rate"] for r in rows), 4)})
    V["K-VSW-2"] = {
        "er_bar": ER_BAR, "min_changed": MIN_CHANGED,
        "datasets_with_regime": [x["dataset"] for x in out_a], "detail": out_a,
        "OUTCOME": ("a_sharpness_regime_found" if len(out_a) >= 2
                    else "b_bounded_axis_closed"),
        "max_er_at_min_changed": {k: (round(max(
            [r["exchange_rate"] for r in ds[k]["result"]["curve"][PRIMARY]
             if r["changed"] >= MIN_CHANGED and r["exchange_rate"] is not None] or [0.0]), 4))
            for k in ds},
        "max_er_anywhere": {k: (round(max(
            [r["exchange_rate"] for r in ds[k]["result"]["curve"][PRIMARY]
             if r["exchange_rate"] is not None] or [0.0]), 4)) for k in ds}}
    V["DEG-A"] = {k: ds[k]["result"]["degeneracy"]["A_agree_threshold_shift"] for k in ds}
    V["DEG-A_FIRES"] = {k: ds[k]["result"]["degeneracy"]["A_FIRES"] for k in ds}
    V["DEG-B"] = {k: ds[k]["result"]["degeneracy"]["B_agree_fixk_max"] for k in ds}
    V["DEG-B_argmax_k"] = {k: ds[k]["result"]["degeneracy"]["B_argmax_k"] for k in ds}
    V["DEG-B_FIRES"] = {k: ds[k]["result"]["degeneracy"]["B_FIRES"] for k in ds}
    V["DEG-D_agree"] = {k: ds[k]["result"]["degeneracy"]["D_agree_ctrl_cos"] for k in ds}
    V["DEG-D_ctrl_ge_vsw"] = {
        k: bool(ds[k]["result"]["arms"]["CTRL_cos_pow"]["dacc"]
                >= ds[k]["result"]["arms"][f"VSW_{PRIMARY}"]["dacc"]) for k in ds}
    V["DEG-D_FIRES"] = bool(sum(V["DEG-D_ctrl_ge_vsw"].values()) >= 2)
    V["CLASS_BALANCE"] = {k: ds[k]["result"]["class_balance"] for k in ds}
    V["CLASS_BALANCE_PASS"] = {
        k: ds[k]["result"]["class_balance"]["PRIMARY_PASS"] for k in ds}
    OUT["verdicts"] = V

    with open(os.path.join(AN, "vsw_pregate_OUT.json"), "w") as f:
        json.dump(OUT, f, indent=1)

    # ------------------------------------------------------------------- print
    W = sys.stdout.write
    W("\n===== GATES =====\n")
    for k in ds:
        d = ds[k]
        W(f"{NICE[k]:8s} PARITY-lam0 {d['parity_lambda0']['n_pass']}/"
          f"{d['parity_lambda0']['n_gates']}   F95-PARITY "
          f"{d['parity_f95']['n_pass']}/{d['parity_f95']['n_gates']}   "
          f"n={d['meta']['n_train_items']}  bank pos {d['meta']['pos_rate']}  "
          f"dep acc {d['result']['acc_deployed']}  perm draws "
          f"{d.get('n_perm_draws', 0)}\n")

    W("\n===== K-VSW-1 / K-VSW-0 : lambda selected on inner folds =====\n")
    W(f"{'dataset':9s} {'family':6s} {'dacc':>8s} {'mF1d':>8s} {'signs':7s} "
      f"{'fix':>4s} {'brk':>4s} {'ER':>7s} {'chg':>4s} {'posrate':>8s} {'p':>7s} "
      f"lambda*\n")
    for k in ds:
        r = ds[k]["result"]
        for fam in FAMS:
            a = r["arms"][f"VSW_{fam}"]
            p = ds[k].get("perm", {}).get(f"VSW_{fam}", {}).get("p", None)
            lam = ",".join("inf" if x["lam_is_inf"] else f"{x['lam']:g}"
                           for x in a["lambda_per_fold"])
            er = "n/a" if a["exchange_rate"] is None else f"{a['exchange_rate']:.4f}"
            W(f"{NICE[k]:9s} {fam:6s} {a['dacc']:+8.4f} "
              f"{a['mF1'] - r['mF1_deployed']:+8.4f} {a['foldsigns']:7s} "
              f"{a['fixed']:4d} {a['broke']:4d} {er:>7s} {a['changed']:4d} "
              f"{a['posrate']:8.4f} {('n/a' if p is None else f'{p:.4f}'):>7s} {lam}\n")

    W("\n===== K-VSW-2 : exchange rate vs aggregation sharpness (PRIMARY family 'pow') "
      "=====\n")
    for k in ds:
        r = ds[k]["result"]
        pc = {lamstr(x): x for x in ds[k].get("perm", {}).get("curve_" + PRIMARY, [])}
        W(f"\n-- {NICE[k]}  (deployed acc {r['acc_deployed']}, "
          f"n={ds[k]['meta']['n_train_items']})\n")
        W(f"{'lambda':>8s} {'dacc':>8s} {'fixed':>6s} {'broken':>7s} {'ER':>8s} "
          f"{'changed':>8s} {'posrate':>8s} {'signs':>7s} {'p':>7s}\n")
        for x in r["curve"][PRIMARY]:
            er = "n/a" if x["exchange_rate"] is None else f"{x['exchange_rate']:.4f}"
            pv = pc.get(lamstr(x), {}).get("p", None)
            W(f"{lamstr(x):>8s} {x['dacc']:+8.4f} {x['fixed']:6d} {x['broke']:7d} "
              f"{er:>8s} {x['changed']:8d} {x['posrate']:8.4f} {x['foldsigns']:>7s} "
              f"{('n/a' if pv is None else f'{pv:.4f}'):>7s}\n")

    W("\n===== SECONDARY families: curve extremes =====\n")
    for fam in ("exp", "lin"):
        for k in ds:
            rows = ds[k]["result"]["curve"][fam]
            best = max(rows, key=lambda x: x["dacc"])
            ers = [x for x in rows if x["changed"] >= MIN_CHANGED
                   and x["exchange_rate"] is not None]
            be = max(ers, key=lambda x: x["exchange_rate"]) if ers else None
            be_er = "n/a" if be is None else f"{be['exchange_rate']:.4f}"
            be_lam = "n/a" if be is None else lamstr(be)
            W(f"{NICE[k]:9s} {fam:4s} best dacc {best['dacc']:+.4f} @lam "
              f"{lamstr(best)} (chg {best['changed']}); best ER at >={MIN_CHANGED} "
              f"changed {be_er} @lam {be_lam}\n")

    W("\n===== DEG-D : cosine twin (same machinery, NO verifier information) =====\n")
    for k in ds:
        r = ds[k]["result"]
        a, c = r["arms"][f"VSW_{PRIMARY}"], r["arms"]["CTRL_cos_pow"]
        W(f"{NICE[k]:9s} VSW {a['dacc']:+.4f} (ER "
          f"{'n/a' if a['exchange_rate'] is None else a['exchange_rate']}) vs "
          f"CTRL_cos {c['dacc']:+.4f} (ER "
          f"{'n/a' if c['exchange_rate'] is None else c['exchange_rate']})  "
          f"agreement {r['degeneracy']['D_agree_ctrl_cos']}  "
          f"ctrl>=vsw {V['DEG-D_ctrl_ge_vsw'][k]}\n")

    W("\n===== DEG-A / DEG-B / class balance / references =====\n")
    for k in ds:
        r = ds[k]["result"]
        g = r["degeneracy"]
        W(f"{NICE[k]:9s} DEG-A {g['A_agree_threshold_shift']:.4f} "
          f"{'FIRES' if g['A_FIRES'] else 'ok':6s}  DEG-B "
          f"{g['B_agree_fixk_max']:.4f} (k={g['B_argmax_k']}) "
          f"{'FIRES' if g['B_FIRES'] else 'ok':6s}  agree-deployed "
          f"{g['agree_deployed']:.4f}  posrate {r['arms'][f'VSW_{PRIMARY}']['posrate']:.4f} "
          f"vs bank {r['posrate_bank']:.4f} dev "
          f"{r['class_balance']['primary_deviation']:.4f} "
          f"{'PASS' if r['class_balance']['PRIMARY_PASS'] else 'FAIL'}\n")
        W(f"{'':9s} THRESH_best {r['arms']['THRESH_best']['dacc']:+.4f}  "
          f"ORACLE_lambda {r['arms']['ORACLE_lambda_pow']['dacc']:+.4f}  "
          + "  ".join(f"FIXK_{kk}={r['arms'][f'FIXK_{kk}']['dacc']:+.4f}"
                      for kk in (1, 3, 10, 15, 20)) + "\n")

    W("\n===== VERDICT SUMMARY =====\n")
    W(f"K-VSW-1 (>= +{BAR_MAIN:.3f} on >=2/3, one family): "
      f"{'PASS' if V['K-VSW-1']['PASS'] else 'FAIL'}   best dacc anywhere "
      f"{V['K-VSW-1']['best_dacc_any_family_any_dataset']}\n")
    W(f"K-VSW-0 (>= +{BAR_INTEREST:.3f} on >=1, 5/5 signs>=0, >=3/5 strict): "
      f"{'PASS' if V['K-VSW-0']['PASS'] else 'FAIL'} {V['K-VSW-0']['datasets']}\n")
    W(f"K-VSW-2 OUTCOME: {V['K-VSW-2']['OUTCOME']}   "
      f"max ER (>= {MIN_CHANGED} changed) {V['K-VSW-2']['max_er_at_min_changed']}   "
      f"max ER anywhere {V['K-VSW-2']['max_er_anywhere']}\n")
    W(f"DEG-A {V['DEG-A']} FIRES {V['DEG-A_FIRES']}\n")
    W(f"DEG-B {V['DEG-B']} k {V['DEG-B_argmax_k']} FIRES {V['DEG-B_FIRES']}\n")
    W(f"DEG-D FIRES {V['DEG-D_FIRES']}  ctrl>=vsw {V['DEG-D_ctrl_ge_vsw']}\n")
    W(f"CLASS BALANCE PASS {V['CLASS_BALANCE_PASS']}\n")
    W(f"\nmerged -> {os.path.join(AN, 'vsw_pregate_OUT.json')}\n")


if __name__ == "__main__":
    main()
