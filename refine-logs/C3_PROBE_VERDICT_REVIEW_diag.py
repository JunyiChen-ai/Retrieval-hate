#!/usr/bin/env python3
"""Independent verdict-review diagnostic for C3_G0COND_ORACLE_PROBE.

Adjudicates the label-oracle anomaly (why gold-label-as-feature gives only
+0.0473/+0.0140 Deltaacc instead of ~+0.17/+0.16) and re-runs the TARGET arm
under corrected machinery. CPU-only, HateMM train (744), read-only.
"""
import json, sys, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import torch
warnings.filterwarnings('ignore')

RNG = np.random.default_rng(20260714)
N_SPLITS, N_REPEATS = 5, 5
EPS = 1e-12
DS = 'HateMM'
ENCODERS = [('CLIP', 'openai_clip-vit-large-patch14-336_HF'),
            ('Qwen', 'Qwen2.5-VL-7B-Instruct_HF')]


def load_features(enc):
    o = torch.load(f'data/CLIP_Embedding/{DS}/train_{enc}.pt', map_location='cpu', weights_only=False)
    ids = o['ids'][0]
    Z = np.concatenate([o['img_feats'].numpy(), o['text_feats'].numpy()], axis=1).astype(np.float64)
    y = o['labels'].numpy().astype(int)
    return ids, Z, y


def load_target(ids):
    tm = json.load(open(f'data/gt/{DS}/target_map.json'))
    n_tgt = tm['_meta']['num_targets']
    vids = {k: v for k, v in tm.items() if not k.startswith('_')}
    prim = np.array([vids[s]['primary'] for s in ids], dtype=int)
    n = len(ids)
    A = np.zeros((n, n_tgt + 1), dtype=np.float64)
    for i, p in enumerate(prim):
        A[i, p if p != -1 else n_tgt] = 1.0
    return A, prim


# ------- CV evaluators -------
def cv_std(X, y, C, n_repeats=N_REPEATS, max_iter=2000):
    """Original machinery: StandardScaler over ALL columns (incl. A), shared C."""
    n = len(y); nll = np.zeros(n); cor = np.zeros(n); cnt = np.zeros(n)
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=1000 + rep)
        for tr, te in skf.split(X, y):
            pipe = Pipeline([('sc', StandardScaler()),
                             ('lr', LogisticRegression(C=C, max_iter=max_iter))])
            pipe.fit(X[tr], y[tr])
            p = np.clip(pipe.predict_proba(X[te])[:, 1], EPS, 1 - EPS)
            pt = np.where(y[te] == 1, p, 1 - p)
            nll[te] += -np.log2(pt); cor[te] += ((p >= 0.5).astype(int) == y[te]); cnt[te] += 1
    return nll / cnt, cor / cnt


def cv_freeA(Z, A, y, C, scaleA=1.0, penalA=True, n_repeats=N_REPEATS, max_iter=5000):
    """Corrected machinery: standardize Z only (fit on train fold), append RAW A*scaleA.
    Z stays regularized exactly as original (same C); A columns kept at native scale
    (scaleA>1 makes A effectively unpenalized without touching Z's treatment).
    If penalA=False, uses per-column penalty=0 emulation via large scaleA is preferred;
    here we simply rely on scaleA. Baseline g(Z) uses the SAME standardize-Z + C."""
    n = len(y); nll = np.zeros(n); cor = np.zeros(n); cnt = np.zeros(n)
    nllb = np.zeros(n); corb = np.zeros(n)
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=1000 + rep)
        for tr, te in skf.split(Z, y):
            sc = StandardScaler().fit(Z[tr])
            Ztr, Zte = sc.transform(Z[tr]), sc.transform(Z[te])
            # baseline Z-only
            lrb = LogisticRegression(C=C, max_iter=max_iter).fit(Ztr, y[tr])
            pb = np.clip(lrb.predict_proba(Zte)[:, 1], EPS, 1 - EPS)
            ptb = np.where(y[te] == 1, pb, 1 - pb)
            nllb[te] += -np.log2(ptb); corb[te] += ((pb >= 0.5).astype(int) == y[te])
            # [Z, A*scaleA]
            Xtr = np.concatenate([Ztr, A[tr] * scaleA], 1)
            Xte = np.concatenate([Zte, A[te] * scaleA], 1)
            lr = LogisticRegression(C=C, max_iter=max_iter).fit(Xtr, y[tr])
            p = np.clip(lr.predict_proba(Xte)[:, 1], EPS, 1 - EPS)
            pt = np.where(y[te] == 1, p, 1 - p)
            nll[te] += -np.log2(pt); cor[te] += ((p >= 0.5).astype(int) == y[te]); cnt[te] += 1
    return (nllb / cnt, corb / cnt), (nll / cnt, cor / cnt)


def boot_ci(delta, B=5000):
    n = len(delta)
    means = np.array([delta[RNG.integers(0, n, n)].mean() for _ in range(B)])
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    out = {}
    # target-alone Bayes (machinery independent), computed once on shared ids
    ids0, _, y0 = load_features(ENCODERS[0][1])
    _, prim0 = load_target(ids0)
    cats = {}
    for p, yy in zip(prim0, y0):
        cats.setdefault(int(p), [0, 0])[int(yy)] += 1
    maj_correct = sum(max(c) for c in cats.values())
    out['target_alone_bayes'] = {
        'per_cat_nonhate_hate': {str(k): v for k, v in sorted(cats.items())},
        'majority_vote_acc': maj_correct / len(y0),
        'n': len(y0), 'base_rate_pos': float(y0.mean())}
    print('TARGET-ALONE majority-vote acc =', maj_correct / len(y0))
    print('  per-cat [nonhate,hate]:', {k: cats[k] for k in sorted(cats)})

    for enc_name, enc in ENCODERS:
        ids, Z, y = load_features(enc)
        A_tgt, prim = load_target(ids)
        A_lab = np.zeros((len(y), 2)); A_lab[np.arange(len(y)), y] = 1.0
        ecell = {'Z_dim': int(Z.shape[1])}

        # ---- (1) reproduce original C=0.001 (StandardScaler over all cols) ----
        nllz, accz = cv_std(Z, y, 0.001)
        nl_lab, ac_lab = cv_std(np.concatenate([Z, A_lab], 1), y, 0.001)
        nl_tgt, ac_tgt = cv_std(np.concatenate([Z, A_tgt], 1), y, 0.001)
        ecell['repro_C0.001'] = {
            'accZ': float(accz.mean()),
            'label_dacc': float((ac_lab - accz).mean()),
            'target_dacc': float((ac_tgt - accz).mean())}
        print(f'\n[{enc_name}] REPRO C=0.001  accZ={accz.mean():.4f}  '
              f'label_dacc={(ac_lab-accz).mean():+.4f}  target_dacc={(ac_tgt-accz).mean():+.4f}')

        # ---- (2) C-sweep (StandardScaler-over-all, shared C) : mechanism ----
        sweep = []
        for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            nz, az = cv_std(Z, y, C, n_repeats=3)
            _, al = cv_std(np.concatenate([Z, A_lab], 1), y, C, n_repeats=3)
            _, at = cv_std(np.concatenate([Z, A_tgt], 1), y, C, n_repeats=3)
            sweep.append({'C': C, 'accZ': float(az.mean()),
                          'label_dacc': float((al - az).mean()),
                          'target_dacc': float((at - az).mean())})
            print(f'    C={C:<7} accZ={az.mean():.4f}  label_dacc={(al-az).mean():+.4f}  '
                  f'target_dacc={(at-az).mean():+.4f}')
        ecell['C_sweep'] = sweep

        # ---- (3) corrected machinery: standardize Z at Z-optimal C=0.001, A free (scaleA large) ----
        # This changes ONLY the treatment of A (removes the artificial shrink), Z identical to original.
        for scaleA in [50.0]:
            (nlb, acb), (nl_l, ac_l) = cv_freeA(Z, A_lab, y, 0.001, scaleA=scaleA)
            (_, acb2), (nl_t, ac_t) = cv_freeA(Z, A_tgt, y, 0.001, scaleA=scaleA)
            dbits_l = boot_ci(nlb - nl_l); dacc_l = boot_ci(ac_l - acb)
            dbits_t = boot_ci(nlb - nl_t); dacc_t = boot_ci(ac_t - acb)
            ecell[f'corrected_Zc0.001_Afree_s{scaleA:g}'] = {
                'accZ': float(acb.mean()),
                'label_accZA': float(ac_l.mean()), 'label_dacc': dacc_l, 'label_dbits': dbits_l,
                'target_accZA': float(ac_t.mean()), 'target_dacc': dacc_t, 'target_dbits': dbits_t}
            print(f'  CORRECTED (Z@C=0.001, A free scale={scaleA:g})  accZ={acb.mean():.4f}')
            print(f'    label : accZA={ac_l.mean():.4f}  dacc={dacc_l[0]:+.4f} CI[{dacc_l[1]:+.4f},{dacc_l[2]:+.4f}]  '
                  f'dbits={dbits_l[0]:+.4f} CI[{dbits_l[1]:+.4f},{dbits_l[2]:+.4f}]')
            print(f'    target: accZA={ac_t.mean():.4f}  dacc={dacc_t[0]:+.4f} CI[{dacc_t[1]:+.4f},{dacc_t[2]:+.4f}]  '
                  f'dbits={dbits_t[0]:+.4f} CI[{dbits_t[1]:+.4f},{dbits_t[2]:+.4f}]')

        # ---- (4) corrected machinery variant: per-arm best-C (extended grid), StandardScaler-all ----
        grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        def bestC(X):
            best = (grid[0], -1)
            for C in grid:
                _, a = cv_std(X, y, C, n_repeats=3)
                if a.mean() > best[1]:
                    best = (C, a.mean())
            return best[0]
        Cz = bestC(Z); Cl = bestC(np.concatenate([Z, A_lab], 1)); Ct = bestC(np.concatenate([Z, A_tgt], 1))
        # honest marginal: compare [Z,A] at its bestC vs Z at the SAME C
        nzl, azl = cv_std(Z, y, Cl); _, all_ = cv_std(np.concatenate([Z, A_lab], 1), y, Cl)
        nzt, azt = cv_std(Z, y, Ct); _, att = cv_std(np.concatenate([Z, A_tgt], 1), y, Ct)
        dacc_l2 = boot_ci(all_ - azl); dacc_t2 = boot_ci(att - azt)
        ecell['perarm_bestC'] = {
            'Cz': Cz, 'Clabel': Cl, 'Ctarget': Ct,
            'label_accZ_atCl': float(azl.mean()), 'label_dacc': dacc_l2,
            'target_accZ_atCt': float(azt.mean()), 'target_dacc': dacc_t2}
        print(f'  PER-ARM bestC  Cz={Cz} Clabel={Cl} Ctarget={Ct}')
        print(f'    label : accZ@Cl={azl.mean():.4f}  dacc={dacc_l2[0]:+.4f} CI[{dacc_l2[1]:+.4f},{dacc_l2[2]:+.4f}]')
        print(f'    target: accZ@Ct={azt.mean():.4f}  dacc={dacc_t2[0]:+.4f} CI[{dacc_t2[1]:+.4f},{dacc_t2[2]:+.4f}]')

        out[enc_name] = ecell

    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'c3_review_diag_out.json', 'w'), indent=1)
    print('\nwrote', sys.argv[1] if len(sys.argv) > 1 else 'c3_review_diag_out.json')


if __name__ == '__main__':
    main()
