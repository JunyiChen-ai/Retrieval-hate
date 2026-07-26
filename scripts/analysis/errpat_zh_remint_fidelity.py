#!/usr/bin/env python
"""ERRPAT MHC-ZH: measure how faithful the CPU re-mint is to the banked job-13150 floor.

The re-mint is a PROXY (head ckpts deleted, F78). This script prices the proxy: per-epoch
dev/test accuracy agreement, plus the two protocol readouts that the report depends on.
"""
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path("/data/jehc223/RGCL")
DUMPS = ROOT / "scripts/analysis/errpat_remint_dumps"
CURVES = json.load(open(ROOT / "scripts/analysis/errpat_zh_curves_OUT.json"))
OUT = ROOT / "scripts/analysis/errpat_zh_fidelity_OUT.json"
WARMUP, N_DEV, N_TEST = 5, 78, 149


def load(seed):
    with open(DUMPS / f"errpat_zh_remint_seed{seed}.pkl", "rb") as f:
        d = pickle.load(f)
    rec = {}
    for r in d["records"]:
        rec[(r["split"], r["epoch"])] = r
    assert len(rec) == 60, len(rec)
    return rec


def val_sel_epoch(vacc, vroc):
    cand = range(WARMUP, 30)
    return max(cand, key=lambda e: (vacc[e], vroc[e], -e))


def main():
    res = {"note": "re-mint = same-recipe CPU re-run of job 13150; NOT bit-exact", "seeds": {}}
    for s in (0, 1, 2):
        rec = load(s)
        bank = CURVES["seeds"][f"seed{s}"]
        rm_v = [rec[("dev", e)]["acc"] for e in range(30)]
        rm_vroc = [rec[("dev", e)]["roc"] for e in range(30)]
        rm_t = [rec[("test", e)]["acc"] for e in range(30)]
        rm_tm = [rec[("test", e)]["macroF1"] for e in range(30)]
        bk_v, bk_t = bank["val_acc_curve"], bank["test_acc_curve"]
        vs = val_sel_epoch(rm_v, rm_vroc)
        dv = np.array(rm_v) - np.array(bk_v)
        dt = np.array(rm_t) - np.array(bk_t)
        res["seeds"][f"seed{s}"] = {
            "remint_val_sel_epoch": vs, "banked_val_sel_epoch": bank["val_sel_epoch"],
            "remint_val_sel_test_acc": round(rm_t[vs], 4),
            "banked_val_sel_test_acc": bank["val_sel_test_acc"],
            "remint_final_test_acc": round(rm_t[29], 4),
            "banked_final_test_acc": bank["final_test_acc"],
            "remint_final_test_mF1": round(rm_tm[29], 4),
            "banked_final_test_mF1": bank["final_test_mF1"],
            "remint_gap_final_minus_valsel_acc": round(rm_t[29] - rm_t[vs], 4),
            "banked_gap_final_minus_valsel_acc": bank["gap_test_acc_final_minus_valsel"],
            "dev_curve_mean_abs_diff_items": round(float(np.mean(np.abs(dv))) * N_DEV, 3),
            "dev_curve_max_abs_diff_items": round(float(np.max(np.abs(dv))) * N_DEV, 2),
            "test_curve_mean_abs_diff_items": round(float(np.mean(np.abs(dt))) * N_TEST, 3),
            "test_curve_max_abs_diff_items": round(float(np.max(np.abs(dt))) * N_TEST, 2),
            "test_curve_spearman": None,
        }
        from scipy.stats import spearmanr
        res["seeds"][f"seed{s}"]["test_curve_spearman"] = round(
            float(spearmanr(rm_t[WARMUP:], bk_t[WARMUP:]).statistic), 4)
        res["seeds"][f"seed{s}"]["dev_curve_spearman"] = round(
            float(spearmanr(rm_v[WARMUP:], bk_v[WARMUP:]).statistic), 4)

    def m(k):
        return round(float(np.mean([res["seeds"][f"seed{s}"][k] for s in (0, 1, 2)])), 4)

    res["3seed"] = {
        "remint_val_sel_acc": m("remint_val_sel_test_acc"),
        "banked_val_sel_acc": m("banked_val_sel_test_acc"),
        "remint_final_acc": m("remint_final_test_acc"),
        "banked_final_acc": m("banked_final_test_acc"),
        "remint_final_mF1": m("remint_final_test_mF1"),
        "banked_final_mF1": m("banked_final_test_mF1"),
        "remint_gap": m("remint_gap_final_minus_valsel_acc"),
        "banked_gap": m("banked_gap_final_minus_valsel_acc"),
    }
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["3seed"], indent=1))
    for s in (0, 1, 2):
        d = res["seeds"][f"seed{s}"]
        print(f"\nseed{s}: val-sel ep re-mint {d['remint_val_sel_epoch']} vs banked {d['banked_val_sel_epoch']}")
        print(f"  val-sel test acc  {d['remint_val_sel_test_acc']:.4f} vs {d['banked_val_sel_test_acc']:.4f}")
        print(f"  final   test acc  {d['remint_final_test_acc']:.4f} vs {d['banked_final_test_acc']:.4f}")
        print(f"  gap (final-valsel) {d['remint_gap_final_minus_valsel_acc']:+.4f} vs {d['banked_gap_final_minus_valsel_acc']:+.4f}")
        print(f"  curve agreement: dev mean|d| {d['dev_curve_mean_abs_diff_items']:.2f} items "
              f"(max {d['dev_curve_max_abs_diff_items']:.0f}), test mean|d| "
              f"{d['test_curve_mean_abs_diff_items']:.2f} items (max {d['test_curve_max_abs_diff_items']:.0f})")
        print(f"  curve spearman: dev {d['dev_curve_spearman']:+.4f}, test {d['test_curve_spearman']:+.4f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
