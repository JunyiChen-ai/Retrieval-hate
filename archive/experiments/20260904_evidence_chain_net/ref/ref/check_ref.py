"""Exhaustive consistency check of evidence_chain_ref.forward_backward against brute-force
enumeration (T <= 8, J = 2, random inputs), plus the pure-prior identity.  Exit code 0 = pass.

    python experiments/20260904_evidence_chain_net/ref/check_ref.py [--trials 300] [--seed 0]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evidence_chain_ref as ref  # noqa: E402

TOL = 1e-8


def random_case(rng, T, J):
    # non-decreasing block index covering blocks 0..J-1 (a block may be empty only if J > T)
    cuts = np.sort(rng.choice(np.arange(1, T), size=min(J - 1, T - 1), replace=False))
    block = np.zeros(T, int)
    for c in cuts:
        block[c:] += 1
    u = rng.normal(0, 2.0, T)
    phi_f = rng.normal(0, 2.0, T)
    phi_c = rng.normal(0, 2.0, J)
    gamma_f = rng.uniform(0, 1, T)
    gamma_c = rng.uniform(0, 1, J)
    d = rng.uniform(0.02, 0.98)
    a = rng.uniform(0.02, 0.98)
    return dict(u=u, phi_f=phi_f, phi_c=phi_c, gamma_f=gamma_f, gamma_c=gamma_c,
                d=d, a=a, block_of_t=block)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    worst = {"log_Z": 0.0, "log_Z0": 0.0, "p_y1": 0.0, "post": 0.0}
    n_fail = 0
    for k in range(args.trials):
        T = int(rng.integers(1, 9))
        J = 2 if T >= 2 else 1
        case = random_case(rng, T, J)
        fb = ref.forward_backward(**case)
        bf = ref.brute_force(**case)
        errs = {key: abs(fb[key] - bf[key]) for key in ("log_Z", "log_Z0", "p_y1")}
        errs["post"] = float(np.max(np.abs(fb["post"] - bf["post"])))
        for key, e in errs.items():
            worst[key] = max(worst[key], e)
        if max(errs.values()) >= TOL:
            n_fail += 1
            print("FAIL trial %d T=%d block=%s errs=%s" % (k, T, case["block_of_t"].tolist(), errs))
        # posterior sanity: state marginals sum to 1, p_y1 in [0,1]
        assert np.allclose(fb["post_states"].sum(1), 1.0, atol=1e-9)
        assert -1e-12 <= fb["p_y1"] <= 1.0 + 1e-12
    # pure-prior identity: u = phi = gamma = 0  ->  P(y=1) = 1 - (1-d) * (1 - a d)^(T-1)
    n_fail_prior = 0
    for k in range(50):
        T = int(rng.integers(1, 9))
        J = 2 if T >= 2 else 1
        case = random_case(rng, T, J)
        case["u"][:] = 0.0; case["phi_f"][:] = 0.0; case["phi_c"][:] = 0.0
        case["gamma_f"][:] = 0.0; case["gamma_c"][:] = 0.0
        fb = ref.forward_backward(**case)
        d, a = case["d"], case["a"]
        expect = 1.0 - (1.0 - d) * (1.0 - a * d) ** (T - 1)
        if abs(fb["p_y1"] - expect) >= TOL or abs(fb["log_Z"]) >= TOL:
            n_fail_prior += 1
            print("FAIL prior trial %d T=%d p_y1=%.12f expect=%.12f log_Z=%.3e"
                  % (k, T, fb["p_y1"], expect, fb["log_Z"]))
    print("brute-force trials %d, failures %d; worst abs errors %s"
          % (args.trials, n_fail, {k: "%.2e" % v for k, v in worst.items()}))
    print("pure-prior trials 50, failures %d" % n_fail_prior)
    ok = n_fail == 0 and n_fail_prior == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
