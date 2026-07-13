#!/usr/bin/env python3
"""C3 NON-TARGET content pilot — corrected-machinery conditional-information probe (CPU).

Pre-registration: refine-logs/C3_NONTARGET_PILOT_DESIGN.md (frozen before any probe number).
Machinery mirrors scripts/analysis/c3_real_predictor_probe.py (the corrected machinery from
refine-logs/C3_PROBE_VERDICT_REVIEW.md): Z standardized ALONE at its Z-only inner-CV-optimal C;
the appended auxiliary BLOCK is raw/standardized x s (s large => effectively UN-penalized, so a
shared heavy L2 cannot crush it). RepeatedStratifiedKFold 5x5, MDL held-out bits (-log2 p_true),
example-clustered (per-video) bootstrap B=5000, Fano bits->acc projection, bar +0.040.

Dense-A handling (design doc §5, calibration-consistent): un-penalizing a full 3584-dim block on
~240 train rows would OVERFIT (not crush). So A_text enters the SAME un-penalized appending path
as a TRAIN-FOLD PCA block (PCA fit on the train fold only -> leak-free), k swept {8,16,32,64},
gate reads best-k per cell. The label-oracle (a 2-col one-hot block, raw x s) uses the identical
appending path and MUST hit accZA ~ 1.0 (mandatory calibration). Secondary robustness:
full-3584-dim A_text under a combined inner-CV-tuned C. Null: PCA block of ROW-SHUFFLED A_text.

Arms per cell ({HateMM,MHC} x {CLIP,Qwen}):
  baseline            g(Z)
  text_pca_k{8,16,32,64}   g'([Z, PCA_k(A_text) x s])            <- C3 non-target signal (gate)
  label_oracle        g'([Z, onehot(gold label) x s])           <- CALIBRATION (must ~1.0)
  shuffled_text_k*     g'([Z, PCA_k(shuffled A_text) x s])       <- null control
  text_full_cvC       g'([Z_std, A_text_std], combined CV C)     <- secondary full-dim read

Gold label used PROBE-ONLY. CPU-only, no GPU/SLURM/network. Not committed.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
import json
import sys
import warnings

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

REPO = '/data/jehc223/RGCL'
ART = os.path.join(REPO, 'artifacts/c3_nontarget')
RNG = np.random.default_rng(20260714)
N_SPLITS, N_REPEATS = 5, 5
C_GRID = [0.001, 0.01, 0.1, 1.0]
B_BOOT = 5000
EPS = 1e-12
SHUFFLE_SEED = 12345
SCALE_A = 50.0
MAX_ITER = 2000
BAR = 0.040
KS = [8, 16, 32, 64]

CLIP = 'openai_clip-vit-large-patch14-336_HF'
QWEN = 'Qwen2.5-VL-7B-Instruct_HF'
ENCODERS = [('CLIP', CLIP), ('Qwen', QWEN)]
DATASETS = ['HateMM', 'MHC']


# ---------------- data ----------------
def load_sample(ds):
    m = json.load(open(f'{ART}/{ds}_sample300.json'))
    ids = list(m['ids'])
    y = np.array([int(m['labels'][i]) for i in ids], dtype=int)
    return ids, y


def load_Z(ds, enc, ids):
    o = torch.load(f'{REPO}/data/CLIP_Embedding/{ds}/train_{enc}.pt', map_location='cpu', weights_only=False)
    cache_ids = o['ids'][0]
    pos = {s: i for i, s in enumerate(cache_ids)}
    idx = [pos[s] for s in ids]
    Z = np.concatenate([o['img_feats'].numpy()[idx], o['text_feats'].numpy()[idx]], axis=1).astype(np.float64)
    ycache = o['labels'].numpy().astype(int)[idx]
    return Z, ycache


def load_Atext(ds, ids):
    vecs = []; nz = 0
    for s in ids:
        v = np.load(f'{ART}/{ds}/emb/{s}.npy').astype(np.float64)
        if not np.any(v):
            nz += 1
        vecs.append(v)
    return np.stack(vecs, axis=0), nz


# ---------------- probe helpers ----------------
def pick_C(Z, y):
    best_c, best_acc = C_GRID[0], -1.0
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    for c in C_GRID:
        accs = []
        for tr, te in skf.split(Z, y):
            sc = StandardScaler().fit(Z[tr])
            lr = LogisticRegression(C=c, max_iter=MAX_ITER).fit(sc.transform(Z[tr]), y[tr])
            accs.append((lr.predict(sc.transform(Z[te])) == y[te]).mean())
        if np.mean(accs) > best_acc:
            best_acc, best_c = float(np.mean(accs)), c
    return best_c, best_acc


def pick_C_combined(Z, A, y):
    """Combined CV C for the secondary full-dim arm: standardize Z and A separately, concat."""
    best_c, best_acc = C_GRID[0], -1.0
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    for c in C_GRID:
        accs = []
        for tr, te in skf.split(Z, y):
            scz = StandardScaler().fit(Z[tr]); sca = StandardScaler().fit(A[tr])
            Xtr = np.concatenate([scz.transform(Z[tr]), sca.transform(A[tr])], axis=1)
            Xte = np.concatenate([scz.transform(Z[te]), sca.transform(A[te])], axis=1)
            lr = LogisticRegression(C=c, max_iter=MAX_ITER).fit(Xtr, y[tr])
            accs.append((lr.predict(Xte) == y[te]).mean())
        if np.mean(accs) > best_acc:
            best_acc, best_c = float(np.mean(accs)), c
    return best_c, best_acc


def _fit_eval(Xtr, ytr, Xte, yte, C):
    lr = LogisticRegression(C=C, max_iter=MAX_ITER).fit(Xtr, ytr)
    p = np.clip(lr.predict_proba(Xte)[:, 1], EPS, 1 - EPS)
    pt = np.where(yte == 1, p, 1 - p)
    return -np.log2(pt), ((p >= 0.5).astype(int) == yte).astype(float)


def cv_all(Z, A, y, C_Z, C_full):
    """One CV pass computing per-video (nll, cor) for baseline and every arm on the SAME folds.
    Returns {arm: (nll[n], cor[n])} with baseline keyed 'baseline'. Per-video averaged over reps."""
    n = len(y)
    A_shuf = A[np.random.default_rng(SHUFFLE_SEED).permutation(n)]
    A_lab = np.zeros((n, 2)); A_lab[np.arange(n), y] = 1.0
    arms = ['baseline', 'label_oracle', 'text_full_cvC'] + \
           [f'text_pca_k{k}' for k in KS] + [f'shuffled_text_k{k}' for k in KS]
    nll = {a: np.zeros(n) for a in arms}
    cor = {a: np.zeros(n) for a in arms}
    cnt = np.zeros(n)

    for rep in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=1000 + rep)
        for tr, te in skf.split(Z, y):
            scZ = StandardScaler().fit(Z[tr]); Ztr, Zte = scZ.transform(Z[tr]), scZ.transform(Z[te])
            cnt[te] += 1
            # baseline
            b_nll, b_cor = _fit_eval(Ztr, y[tr], Zte, y[te], C_Z)
            nll['baseline'][te] += b_nll; cor['baseline'][te] += b_cor
            # label oracle (raw one-hot x s, unpenalized)
            Xtr = np.concatenate([Ztr, A_lab[tr] * SCALE_A], axis=1)
            Xte = np.concatenate([Zte, A_lab[te] * SCALE_A], axis=1)
            l_nll, l_cor = _fit_eval(Xtr, y[tr], Xte, y[te], C_Z)
            nll['label_oracle'][te] += l_nll; cor['label_oracle'][te] += l_cor
            # full-dim A_text (secondary): standardize A on tr, combined C
            scA = StandardScaler().fit(A[tr]); Atr_s, Ate_s = scA.transform(A[tr]), scA.transform(A[te])
            f_nll, f_cor = _fit_eval(np.concatenate([Ztr, Atr_s], axis=1), y[tr],
                                     np.concatenate([Zte, Ate_s], axis=1), y[te], C_full)
            nll['text_full_cvC'][te] += f_nll; cor['text_full_cvC'][te] += f_cor
            # PCA block (fit on tr, kmax then slice) for real + shuffled
            kmax = min(max(KS), len(tr) - 1, A.shape[1])
            for src, tag in [(A, 'text_pca_k'), (A_shuf, 'shuffled_text_k')]:
                scS = StandardScaler().fit(src[tr])
                Str, Ste = scS.transform(src[tr]), scS.transform(src[te])
                pca = PCA(n_components=kmax, random_state=0).fit(Str)
                Ptr_all, Pte_all = pca.transform(Str), pca.transform(Ste)
                for k in KS:
                    kk = min(k, kmax)
                    scP = StandardScaler().fit(Ptr_all[:, :kk])
                    Btr = scP.transform(Ptr_all[:, :kk]) * SCALE_A
                    Bte = scP.transform(Pte_all[:, :kk]) * SCALE_A
                    a_nll, a_cor = _fit_eval(np.concatenate([Ztr, Btr], axis=1), y[tr],
                                             np.concatenate([Zte, Bte], axis=1), y[te], C_Z)
                    nll[f'{tag}{k}'][te] += a_nll; cor[f'{tag}{k}'][te] += a_cor
    for a in arms:
        nll[a] /= cnt; cor[a] /= cnt
    return nll, cor


# ---------------- Fano ----------------
def hb(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def hb_inv(L):
    L = min(max(L, 0.0), 1.0); lo, hi = 0.0, 0.5
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if hb(mid) < L: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def fano_ceiling(mb):
    return 1.0 - hb_inv(mb)


def boot(nllz, corz, nllza, corza, B=B_BOOT):
    n = len(nllz)
    dacc = corza - corz; dbits = nllz - nllza
    m_dacc = float(dacc.mean()); m_dbits = float(dbits.mean())
    m_fano = float(fano_ceiling(nllza.mean()) - fano_ceiling(nllz.mean()))
    ba = np.empty(B); bb = np.empty(B); bf = np.empty(B)
    for b in range(B):
        idx = RNG.integers(0, n, n)
        ba[b] = dacc[idx].mean(); bb[b] = dbits[idx].mean()
        bf[b] = fano_ceiling(nllza[idx].mean()) - fano_ceiling(nllz[idx].mean())
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {'dacc_mean': m_dacc, 'dacc_ci': ci(ba),
            'dbits_mean': m_dbits, 'dbits_ci': ci(bb),
            'fano_dacc_mean': m_fano, 'fano_dacc_ci': ci(bf),
            'acc_Z': float(corz.mean()), 'acc_ZA': float(corza.mean()),
            'bits_Z': float(nllz.mean()), 'bits_ZA': float(nllza.mean())}


def main():
    outpath = sys.argv[1] if len(sys.argv) > 1 else f'{REPO}/refine-logs/C3_NONTARGET_PILOT_OUT.json'
    out = {'design': {'n_splits': N_SPLITS, 'n_repeats': N_REPEATS, 'C_grid': C_GRID,
                      'B_boot': B_BOOT, 'scale_A': SCALE_A, 'ks': KS, 'bar_dacc': BAR,
                      'machinery': 'corrected (Z std alone @ C_Z; aux block raw/std x s unpenalized; A_text via train-fold PCA)',
                      'rng_seed': 20260714, 'shuffle_seed': SHUFFLE_SEED},
           'datasets': {}}
    if os.path.exists(outpath):
        out = json.load(open(outpath)); print(f'[resume] {outpath}', flush=True)

    for ds in DATASETS:
        ids, y = load_sample(ds)
        A, nz = load_Atext(ds, ids)
        dcell = out['datasets'].get(ds, {'n': len(ids), 'n_pos': int(y.sum()),
                                         'n_zero_Atext': int(nz), 'encoders': {}})
        out['datasets'][ds] = dcell
        for enc_name, enc in ENCODERS:
            if enc_name in dcell['encoders']:
                print(f'[skip] {ds}/{enc_name}', flush=True); continue
            Z, ycache = load_Z(ds, enc, ids)
            assert np.array_equal(ycache, y), f'{ds}/{enc_name} label order mismatch'
            C_Z, cacc = pick_C(Z, y)
            C_full, cfacc = pick_C_combined(Z, A, y)
            nll, cor = cv_all(Z, A, y, C_Z, C_full)
            base = (nll['baseline'], cor['baseline'])
            arms = {}
            for a in nll:
                if a == 'baseline':
                    continue
                arms[a] = boot(base[0], base[1], nll[a], cor[a])
                arms[a]['arm'] = a
            accZ = float(cor['baseline'].mean())
            lab = arms['label_oracle']
            headroom = 1.0 - accZ
            cell = {'encoder_file': f'data/CLIP_Embedding/{ds}/train_{enc}.pt',
                    'Z_dim': int(Z.shape[1]), 'C_Z': C_Z, 'C_Z_cv_acc': float(cacc),
                    'C_full': C_full, 'C_full_cv_acc': float(cfacc),
                    'baseline_acc_Z': accZ,
                    'calibration': {'label_accZA': lab['acc_ZA'],
                                    'headroom_1_minus_accZ': float(headroom),
                                    'label_dacc': lab['dacc_mean'],
                                    'fano_headroom_fraction': float(lab['dacc_mean'] / headroom) if headroom > 0 else float('nan'),
                                    'PASS': bool(lab['acc_ZA'] >= 0.99)},
                    'arms': arms}
            dcell['encoders'][enc_name] = cell
            json.dump(out, open(outpath, 'w'), indent=1)
            cal = cell['calibration']
            print(f'\n[{ds}/{enc_name}] C_Z={C_Z} C_full={C_full} Zdim={cell["Z_dim"]} '
                  f'accZ={accZ:.4f} label_accZA={cal["label_accZA"]:.4f} '
                  f'(headroom_frac={cal["fano_headroom_fraction"]:.3f}) CALIB_PASS={cal["PASS"]}', flush=True)
            order = ['label_oracle'] + [f'text_pca_k{k}' for k in KS] + \
                    [f'shuffled_text_k{k}' for k in KS] + ['text_full_cvC']
            for a in order:
                r = arms[a]
                print(f'   {a:20s} accZA={r["acc_ZA"]:.4f} '
                      f'dacc={r["dacc_mean"]:+.4f} CI[{r["dacc_ci"][0]:+.4f},{r["dacc_ci"][1]:+.4f}] '
                      f'dbits={r["dbits_mean"]:+.4f} fano={r["fano_dacc_mean"]:+.4f} '
                      f'CI[{r["fano_dacc_ci"][0]:+.4f},{r["fano_dacc_ci"][1]:+.4f}]', flush=True)

    # ---- verdict ----
    calib_ok = all(c['calibration']['PASS']
                   for ds in out['datasets'].values() for c in ds['encoders'].values())
    best = {'direct': (None, -9), 'fano': (None, -9)}
    proceed = False
    for ds_name, ds in out['datasets'].items():
        for enc_name, c in ds['encoders'].items():
            for k in KS:
                r = c['arms'][f'text_pca_k{k}']
                for metric, key in [('direct', ('dacc_mean', 'dacc_ci')), ('fano', ('fano_dacc_mean', 'fano_dacc_ci'))]:
                    pt = r[key[0]]; lo = r[key[1]][0]
                    if pt > best[metric][1]:
                        best[metric] = (f'{ds_name}/{enc_name}/k{k}', pt)
                    if pt >= BAR and lo > 0:
                        proceed = True
    if not calib_ok:
        verdict = 'MACHINERY_INVALID'
    elif proceed:
        verdict = 'C3_NONTARGET_PROCEED'
    else:
        verdict = 'C3_NONTARGET_DEAD_AT_G0'
    out['verdict'] = {'calibration_all_pass': bool(calib_ok),
                      'best_direct_dacc': best['direct'], 'best_fano_dacc': best['fano'],
                      'proceed_condition_met': bool(proceed), 'VERDICT': verdict}
    json.dump(out, open(outpath, 'w'), indent=1)
    print(f'\n==== calib_all_pass={calib_ok} best_direct={best["direct"]} '
          f'best_fano={best["fano"]} => VERDICT={verdict} ====', flush=True)
    print(f'wrote {outpath}', flush=True)


if __name__ == '__main__':
    main()
