#!/usr/bin/env python
"""
aggnet_pregate_report.py -- MERGE + REPORT for the AGGNET (C3) $0 pregate.
Reporting only: it computes no new treatment quantity. It merges the per-dataset
run files into scripts/analysis/aggnet_pregate_OUT.json and re-reads every number at
4 dp for refine-logs/AGGNET_PREGATE_RECORD.md.

The main battery (3 spaces, no permutation) and the permutation battery (fused only)
were run as separate short-lived processes per dataset, reap-safe, F95 runner
precedent. Because the harness is deterministic, the fused-space cells must agree
BIT-FOR-BIT between the two runs; that agreement is asserted here and is a free
determinism check on the whole pipeline.
"""
import json
import os
import sys

REPO = "/data/jehc223/RGCL"
AN = os.path.join(REPO, "scripts/analysis")
DS = ("hatemm", "zh", "en")
NAME = {"hatemm": "HateMM", "zh": "MHC-ZH", "en": "MHC-EN"}
VGA_BENCH = {"hatemm": 0.0269, "zh": 0.0104, "en": 0.0182}   # VGA_PREGATE_RECORD 4.3


def merge():
    OUT = {"meta": None, "datasets": {}}
    for d in DS:
        m = json.load(open(os.path.join(AN, f"aggnet_main_{d}_OUT.json")))
        p = json.load(open(os.path.join(AN, f"aggnet_perm_{d}_OUT.json")))
        if OUT["meta"] is None:
            OUT["meta"] = m["meta"]
        assert m["meta"]["script_sha256"] == p["meta"]["script_sha256"], d
        md, pd_ = m["datasets"][d], p["datasets"][d]
        # determinism gate: the fused cell must be identical across the two runs
        a, b = md["spaces"]["fused"]["pooled"], pd_["spaces"]["fused"]["pooled"]
        for k in ("acc_deployed", "mF1_deployed", "n_deployed_wrong", "n_pathology",
                  "frac_class_mixed_top20", "frac_nonmonotone_C3net"):
            assert a[k] == b[k], (d, k, a[k], b[k])
        for arm in ("C3_net", "C3_metak", "FIXBEST_mono", "THRESH_best", "DIRECT_logit"):
            for k in ("acc", "d_acc", "mF1", "d_mF1", "fold_signs", "n_fixed",
                      "n_broken", "exchange_rate", "posrate"):
                assert a[arm][k] == b[arm][k], (d, arm, k, a[arm][k], b[arm][k])
        assert a["DEG"] == b["DEG"], d
        md["perm"] = pd_["perm"]
        md["determinism_gate"] = {"fused_cell_identical_across_runs": True,
                                  "n_asserted": 6 + 5 * 9 + 1}
        OUT["datasets"][d] = md
    json.dump(OUT, open(os.path.join(AN, "aggnet_pregate_OUT.json"), "w"), indent=1)
    return OUT


def f(x, n=4):
    return "n/a" if x is None else f"{x:+.{n}f}" if isinstance(x, float) else str(x)


def main():
    O = merge()
    w = sys.stdout.write
    w(f"script sha256 {O['meta']['script_sha256']}\n")
    w(f"frozen: {json.dumps(O['meta']['frozen'])}\n\n")

    w("=== PARITY (81 cells) + coverage ===\n")
    for d in DS:
        D = O["datasets"][d]
        for sp in ("fused", "text", "img"):
            P = D["spaces"][sp]["parity"]
            assert P["PASS"], (d, sp)
        p = D["spaces"]["fused"]["pooled"]
        w(f"{NAME[d]:8s} n={D['n_items']:4d} parity 27/27 PASS  floor "
          f"{p['acc_deployed']:.4f}/{p['mF1_deployed']:.4f}  wrong={p['n_deployed_wrong']} "
          f"patho={p['n_pathology']}  class-mixed top20 = {p['frac_class_mixed_top20']:.4f} "
          f"({p['n_class_mixed_top20']}/{D['n_items']})\n")

    w("\n=== PRIMARY READ (fused), all arms ===\n")
    hdr = f"{'dataset':8s} {'arm':16s} {'dacc':>8s} {'dmF1':>8s} {'signs':>7s} " \
          f"{'>=0':>3s} {'fix/brk':>9s} {'ER':>7s} {'chg':>5s} {'posrate':>8s}\n"
    w(hdr)
    for d in DS:
        p = O["datasets"][d]["spaces"]["fused"]["pooled"]
        for arm in ("C3_net", "C3_net_s1", "C3_net_s2", "C3_net_dlen", "C3_metak",
                    "FIXBEST_mono", "FIXBEST_oracle", "THRESH_best", "DIRECT_logit"):
            e = p[arm]
            fb = f"{e['n_fixed']}/{e['n_broken']}"
            er = f"{e['exchange_rate']:.4f}" if e['exchange_rate'] else "n/a"
            w(f"{NAME[d]:8s} {arm:16s} {e['d_acc']:+8.4f} {e['d_mF1']:+8.4f} "
              f"{e['fold_signs']:>7s} {e['n_folds_ge0']:>3d} "
              f"{fb:>9s} {er:>7s} "
              f"{e['n_changed']:>5d} {e['posrate']:>8.4f}  (bank {p['posrate_bank']:.4f})\n")
        w("\n")

    w("=== BAR 2 (vs best fixed monotone profile) and the VGA benchmark ===\n")
    for d in DS:
        p = O["datasets"][d]["spaces"]["fused"]["pooled"]
        c3, fb = p["C3_net"]["d_acc"], p["FIXBEST_mono"]["d_acc"]
        w(f"{NAME[d]:8s} C3 {c3:+.4f}  FIXBEST_mono {fb:+.4f}  C3-FIXBEST {c3 - fb:+.4f}  "
          f"oracle-fixed {p['FIXBEST_oracle']['d_acc']:+.4f}  "
          f"VGA F47 gate {VGA_BENCH[d]:+.4f}  C3-VGA {c3 - VGA_BENCH[d]:+.4f}\n")
        sel = [x["fit_selected"] for x in p["fixed_profile_selection"]]
        orc = [x["oracle_selected"] for x in p["fixed_profile_selection"]]
        w(f"{'':8s} fixed profile chosen on fit folds: {sel}   oracle: {orc}\n")
        lam = {k: [x["lam"] for x in v] for k, v in p["lambda_selection"].items()}
        w(f"{'':8s} lambda selected per fold: {json.dumps(lam)}\n")

    w("\n=== DEGENERACY CONTROLS (pooled agreement with C3_net) ===\n")
    for d in DS:
        p = O["datasets"][d]["spaces"]["fused"]["pooled"]
        g = p["DEG"]
        w(f"{NAME[d]:8s} DEG-A threshold-shift {g['A_agree_threshold_shift']:.4f}  "
          f"DEG-B max fixed-k {g['B_agree_fixk_max']:.4f} (k={g['B_argmax_k']})  "
          f"DEG-C direct readout {g['C_agree_direct_logit']:.4f}  "
          f"| deployed {g['agree_deployed']:.4f}  fixbest {g['agree_fixbest_mono']:.4f}\n")
        w(f"{'':8s} per-k: {json.dumps(g['B_agree_fixk'])}\n")

    w("\n=== BAR 3 (non-monotonicity) ===\n")
    for d in DS:
        p = O["datasets"][d]["spaces"]["fused"]["pooled"]
        nm = p["nonmono_read"]
        w(f"{NAME[d]:8s} frac non-monotone {p['frac_nonmonotone_C3net']:.4f} "
          f"({nm['n_nonmono']}/{nm['n_nonmono'] + nm['n_mono']})  "
          f"dacc|nonmono {f(nm['d_acc_nonmono'])}  dacc|mono {f(nm['d_acc_mono'])}  "
          f"changed {nm['n_changed_nonmono']}/{nm['n_changed_mono']}  "
          f"mean rise/max {nm['mean_rise_over_max']:.4f}\n")
        w(f"{'':8s} mean learned profile: {nm['mean_G_profile']}\n")

    w("\n=== PERMUTATION NULL (C3_net, fused, label-shuffled fitting targets) ===\n")
    for d in DS:
        pm = O["datasets"][d]["perm"]
        w(f"{NAME[d]:8s} n_perm={pm['n_perm']} observed {pm['observed']:+.4f}  "
          f"null {pm['null_mean']:+.4f} +- {pm['null_sd']:.4f}  q95 {pm['null_q95']:+.4f}  "
          f"max {pm['null_max']:+.4f}  p={pm['p']:.4f}\n")

    w("\n=== SECONDARY SPACES (C3_net) ===\n")
    for d in DS:
        for sp in ("fused", "text", "img"):
            p = O["datasets"][d]["spaces"][sp]["pooled"]
            e = p["C3_net"]
            er = f"{e['exchange_rate']:.4f}" if e['exchange_rate'] else "n/a"
            w(f"{NAME[d]:8s} {sp:6s} floor {p['acc_deployed']:.4f}  C3 {e['d_acc']:+.4f} "
              f"signs {e['fold_signs']}  ER {er}  "
              f"DEG-A {p['DEG']['A_agree_threshold_shift']:.4f} "
              f"DEG-B {p['DEG']['B_agree_fixk_max']:.4f} "
              f"DEG-C {p['DEG']['C_agree_direct_logit']:.4f}\n")

    w("\n=== BEST C3 CELL ANYWHERE (all datasets x all spaces x all C3 arms) ===\n")
    best = []
    for d in DS:
        for sp in ("fused", "text", "img"):
            p = O["datasets"][d]["spaces"][sp]["pooled"]
            for arm in ("C3_net", "C3_net_s1", "C3_net_s2", "C3_net_dlen", "C3_metak"):
                best.append((p[arm]["d_acc"], NAME[d], sp, arm, p[arm]["fold_signs"],
                             p[arm]["exchange_rate"]))
    best.sort(reverse=True)
    for b in best[:6]:
        w(f"  {b[0]:+.4f}  {b[1]} x {b[2]} x {b[3]}  signs {b[4]}  ER {b[5]}\n")
    w(f"  ... {len(best)} C3 cells total; "
      f"{sum(1 for x in best if x[0] > 0)} positive, {sum(1 for x in best if x[0] <= 0)} <= 0\n")


if __name__ == "__main__":
    main()
