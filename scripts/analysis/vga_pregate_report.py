#!/usr/bin/env python
"""
vga_pregate_report.py -- REPORTING ONLY for the VGA/VNQ $0 pregate.
Re-reads vga_pregate_gate.py's OUT json at 4 dp and prints the record's tables.
Computes no new quantity, fits nothing, touches no feature cache and no test split.
"""
import argparse
import json

DS = [("hatemm", "HateMM"), ("zh", "MHC-ZH"), ("en", "MHC-EN")]
ARMS = ["verifier:logistic", "verifier:gbm", "f47ctrl:logistic", "f47ctrl:gbm",
        "f47ctrl_full:logistic", "f47ctrl_full:gbm", "oracle", "fire_all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    a = ap.parse_args()
    D = json.load(open(a.inp))

    print("\n=== §1 ARITHMETIC RECOMPUTED (primary adjudicator: fused x MLP x max) ===")
    print(f"{'dataset':9s} {'n':>5s} {'dep acc':>8s} {'ungated':>8s} {'N':>4s} "
          f"{'fix':>4s} {'brk':>4s} {'baserate':>9s} {'oracle':>8s} {'breakeven':>10s}")
    for k, nm in DS:
        p = D[k]["primary"]
        print(f"{nm:9s} {D[k]['meta']['n']:5d} {p['acc_deployed']:8.4f} "
              f"{p['dacc_ungated']:+8.4f} {p['gate_set_N']:4d} {p['n_fix']:4d} "
              f"{p['n_break']:4d} {p['base_rate_fix']:9.4f} "
              f"{p['oracle_gate_dacc']:+8.4f} "
              f"{p['breakeven_precision_full_coverage']:10.4f}")

    for sec in ("primary", "secondary"):
        print(f"\n=== C1 / VGA END-TO-END — {sec.upper()} adjudicator "
              f"({D['hatemm'][sec]['adjudicator']}) ===")
        print(f"{'dataset':9s} {'arm':24s} {'fired':>5s} {'prec':>7s} {'req@m':>7s} "
              f"{'fire%N':>7s} {'dacc':>8s} {'dmF1':>8s} {'signs':>6s} {'posrate':>8s}")
        for k, nm in DS:
            for arm in ARMS:
                r = D[k][sec]["arms"][arm]
                pr = "  n/a" if r["gate_precision"] is None else f"{r['gate_precision']:7.4f}"
                rq = "  n/a" if r["required_precision_at_this_fire_count"] is None \
                    else f"{r['required_precision_at_this_fire_count']:7.4f}"
                fr = "  n/a" if r["fire_rate_of_gateset"] is None \
                    else f"{r['fire_rate_of_gateset']:7.4f}"
                print(f"{nm:9s} {arm:24s} {r['n_fired']:5d} {pr} {rq} {fr} "
                      f"{r['dacc_vs_deployed']:+8.4f} {r['dmF1_vs_deployed']:+8.4f} "
                      f"{r['fold_signs']:>6s} {r['posrate_emitted']:8.4f}")
            print()

    print("=== K-VGA-2 PERMUTATION NULL (primary adjudicator, N_PERM per meta) ===")
    print(f"{'dataset':9s} {'arm':24s} {'obs':>8s} {'nullmean':>9s} {'nullsd':>8s} "
          f"{'null q95':>9s} {'p':>7s}")
    for k, nm in DS:
        for arm in [x for x in ARMS if x not in ("oracle", "fire_all")]:
            q = D[k]["primary"]["perm"][arm]
            print(f"{nm:9s} {arm:24s} {q['observed_dacc']:+8.4f} "
                  f"{q['null_mean']:+9.4f} {q['null_sd']:8.4f} {q['null_q95']:+9.4f} "
                  f"{q['p_value']:7.4f}")
        print()

    print("=== K-VGA-4 CLASS BALANCE (primary adjudicator) ===")
    print(f"{'dataset':9s} {'bank':>7s} {'deployed':>9s} "
          + " ".join(f"{a.split(':')[0][:9]+':'+a.split(':')[1][:3]:>14s}"
                     for a in ARMS[:2]))
    for k, nm in DS:
        b = D[k]["meta"]["posrate_bank"]
        dep = D[k]["primary"]["arms"]["fire_all"]
        print(f"{nm:9s} {b:7.4f} {'':>9s} "
              + " ".join(f"{D[k]['primary']['arms'][a]['posrate_emitted']:14.4f}"
                         for a in ARMS[:2])
              + f"   (ungated adj posrate {dep['posrate_emitted']:.4f})")

    print("\n=== C2 / VNQ SELECTIVE PREDICTION (AUGRC lower = better) ===")
    print(f"{'dataset':9s} {'arm':16s} {'AUGRC':>8s} {'AURC':>8s} {'AUROC_err':>10s}")
    for k, nm in DS:
        c2 = D[k]["c2_vnq"]
        for arm, r in c2["arms"].items():
            print(f"{nm:9s} {arm:16s} {r['AUGRC']:8.4f} {r['AURC']:8.4f} "
                  f"{r['AUROC_error_detection']:10.4f}")
        for base in ("knnue_fitted", "vote_margin"):
            v = c2[f"vnq_vs_{base}"]
            print(f"{nm:9s} {'VNQ vs '+base:16s} dAUGRC "
                  f"{v['dAUGRC_pooled_improvement']:+.4f} signs {v['fold_signs']} "
                  f"({v['n_folds_positive']}/5)")
        print()


if __name__ == "__main__":
    main()
