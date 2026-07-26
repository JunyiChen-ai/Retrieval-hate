#!/usr/bin/env python3
"""CLAP lane — MACHINERY PARITY CHECK against the published F64 / LAUD verdict.

Feeds the FORKED CLAP gate (scripts/analysis/clap_g0cond_gate.py) the already-banked
Whisper-large-v3 block as its aux and verifies it reproduces the PUBLISHED HateMM numbers from
refine-logs/LAUD_GATE_RECORD.md section 3 bit-exactly at 4 dp, on BOTH Z arms.

Why this matters: the whole value of holding the house +0.040 bar and every machinery constant
fixed is that the CLAP number lands directly comparable to the F41 (-0.0038) and F64 (+0.0014)
graveyard entries. That comparability is only real if the fork introduced no drift. This check
proves it did not, using the exact code path point_arms() takes.

No CLAP data is touched. Moves no bar. Zero test-touch (train U val only). CPU, ~3 min.

NOTE (documented so nobody repeats it): arm_cor_allk fits ONE PCA at n_components = max(ks) and
slices it, and sklearn's PCA picks a randomized SVD solver whose result depends on n_components.
So the k8/k16 point estimates are NOT invariant to whether the call passes KS_DECISION (kmax=16)
or KS_REPORT (kmax=64). point_arms() -- like LAUD -- passes KS_REPORT, and parity only holds on
that path. A first attempt at this check passed KS_DECISION and produced a spurious +0.0019 on the
deployed arm. The sensitivity is ~5e-4, i.e. two orders of magnitude below the +0.040 bar and
immaterial to any verdict, but it is real and worth knowing.
"""
import importlib.util
import json
import time

import numpy as np
import torch

REPO = '/data/jehc223/RGCL'
OUT = f'{REPO}/refine-logs/CLAP_MACHINERY_PARITY_OUT.json'

# Published verbatim from refine-logs/LAUD_GATE_RECORD.md section 3 (HateMM rows).
PUBLISHED = {
    'deployed_7168': {'accZ': 0.8712, 'best_k': 16, 'dacc': 0.0014,
                      'ci': [-0.0075, 0.0103], 'full': 0.0063},
    'strict_8960':   {'accZ': 0.8383, 'best_k': 8,  'dacc': 0.0014,
                      'ci': [-0.0073, 0.0106], 'full': 0.0052},
}
TOL = 5e-5   # 4 dp


def main():
    t0 = time.time()
    spec = importlib.util.spec_from_file_location('g', f'{REPO}/scripts/analysis/clap_g0cond_gate.py')
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

    wh = torch.load(f'{REPO}/data/audio/HateMM/whisper_whisper-large-v3_trainval.pt',
                    map_location='cpu', weights_only=False)
    ids = list(wh['ids'][0])
    y = wh['labels'].numpy().astype(int)
    A = wh['emb'].numpy().astype(np.float64)

    res = {'source_record': 'refine-logs/LAUD_GATE_RECORD.md section 3 (HateMM rows)',
           'aux_block': 'whisper-large-v3 encoder mean(+)max 2560-d (the F64-killed block)',
           'tol_4dp': TOL, 'arms': {}}

    for z_arm in ['deployed_7168', 'strict_8960']:
        t = time.time()
        if z_arm == 'deployed_7168':
            dep = g._pooled_id2row(g.DEPLOY)
            Z = np.stack([dep[i][0] for i in ids])
        else:
            cl = g._pooled_id2row(g.CLIP); qw = g._pooled_id2row(g.FROZEN_QWEN)
            Z = np.stack([np.concatenate([cl[i][0], qw[i][0]]) for i in ids])

        C_Z, _ = g.pick_C(Z, y)
        C_full, _ = g.pick_C_combined(Z, A, y)
        base = g.baseline_cor(Z, y, C_Z); accZ = float(base.mean())
        accZA = float(g.oracle_cor(Z, y, C_Z).mean())
        rk = g.arm_cor_allk(Z, y, C_Z, A, g.KS_REPORT)      # kmax=64 — the point_arms/LAUD path
        arms = {}
        for k in g.KS_REPORT:
            m, ci = g.boot_ci(rk[k], base, g.BOOT_SEED + k)
            arms[f'k{k}'] = {'dacc': m, 'ci': ci}
        bk = max(g.KS_DECISION, key=lambda k: arms[f'k{k}']['dacc'])
        best = arms[f'k{bk}']
        full = g.full_cor(Z, A, y, C_full)
        fm, _ = g.boot_ci(full, base, g.BOOT_SEED + 999)

        p = PUBLISHED[z_arm]
        checks = {
            'accZ': abs(accZ - p['accZ']) < TOL,
            'best_k': bk == p['best_k'],
            'best_dacc': abs(best['dacc'] - p['dacc']) < TOL,
            'ci_low': abs(best['ci'][0] - p['ci'][0]) < TOL,
            'ci_high': abs(best['ci'][1] - p['ci'][1]) < TOL,
            'full_cvC': abs(fm - p['full']) < TOL,
        }
        ok = all(checks.values())
        res['arms'][z_arm] = {'accZ': accZ, 'calib_accZA': accZA, 'C_Z': C_Z, 'C_full': C_full,
                              'best_k': bk, 'best_dacc': best['dacc'], 'best_ci': best['ci'],
                              'full_cvC': fm, 'all_k': arms, 'published': p,
                              'checks': checks, 'PARITY_4DP': ok, 'elapsed_s': round(time.time() - t, 1)}
        print(f"[{z_arm}] accZ={accZ:.4f} (pub {p['accZ']:.4f}) calib={accZA:.4f}")
        print(f"    best-k{bk} (pub k{p['best_k']}) dacc={best['dacc']:+.4f} (pub {p['dacc']:+.4f}) "
              f"CI[{best['ci'][0]:+.4f},{best['ci'][1]:+.4f}] (pub [{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}])")
        print(f"    full_cvC={fm:+.4f} (pub {p['full']:+.4f})   PARITY_4DP="
              f"{'PASS' if ok else 'FAIL'}  [{time.time()-t:.0f}s]")

    res['ALL_ARMS_PARITY'] = all(v['PARITY_4DP'] for v in res['arms'].values())
    res['elapsed_s'] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f"\nALL_ARMS_PARITY = {res['ALL_ARMS_PARITY']}  -> {OUT}")


if __name__ == '__main__':
    main()
