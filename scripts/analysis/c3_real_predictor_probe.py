#!/usr/bin/env python3
"""C3 REAL-PREDICTOR conditional-information probe — HateMM (744) + MHC-EN (549).

Decides the C3 target-content channel on a REAL predictor (Qwen2.5-VL-7B target
predictions from the TARC campaign), NOT the gold oracle. Follows the CORRECTED
machinery adjudicated in refine-logs/C3_PROBE_VERDICT_REVIEW.md after the shared-L2
crush bug (a heavy shared C over StandardScaler-standardized [Z,A] crushes the
low-dim auxiliary one-hot). Corrected fix (verbatim from the review §3 / diag script):
  - standardize Z ALONE (fit on train fold), keep Z regularized at its CV-optimal C;
  - append A as RAW one-hot x s (s large => A effectively UN-penalized so it cannot
    be crushed). s from the review's stable range (s>=50).

MANDATORY CALIBRATION ARM (REFLECTION_mllm_integration_failures.md §4, 2026-07-14):
  the label-oracle arm (gold LABEL one-hot as A) MUST reach ~full Fano headroom
  (accZA ~ 1.0). If it does not, the machinery is INVALID and NO verdict may stand.

Arms (per encoder Z in {CLIP, Qwen}; datasets HateMM & MHC-EN):
  (1) baseline        g(Z)                                   [Z-only reference]
  (2) pred_target     g'([Z, A_pred])   A = Qwen-7B PREDICTED 9-way one-hot(8 tgt + none/unparsed)
  (3) gold_target     g'([Z, A_gold])   HateMM ONLY (reproduce review's ~+0.0487 consistency check)
  (4) label_oracle    g'([Z, A_label])  2-way one-hot(gold LABEL) [CALIBRATION — must hit ~1.0]
  (5) shuffled_pred   g'([Z, A_shuf])   rows of A_pred permuted (null control; expect ~0)

Design mirrors scripts/analysis/c3_g0cond_oracle_probe.py: RepeatedStratifiedKFold
5x5, MDL held-out codelength (bits = -log2 p_true), example-clustered (per-video)
bootstrap B=5000, Fano bits->acc projection, decision bar +0.040.

Gold usage is PROBE-ONLY (target/label as appended feature + probe target; never
in-method). MHC has NO gold target_map => no gold_target arm for MHC (fine; this is
a real-predictor-only read). CPU-only, no GPU/SLURM/network, not committed.
"""
import os
# Cap BLAS/OMP threads: keep this CPU-only job modest so the login-node reaper
# leaves it alone (fits are ~1s regardless; see probe log). Set BEFORE numpy import.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
import json, sys, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import torch
warnings.filterwarnings('ignore')

RNG = np.random.default_rng(20260714)
N_SPLITS, N_REPEATS = 5, 5           # 25 fits/model; >= 5 seeds (satisfies protocol)
C_GRID = [0.001, 0.01, 0.1, 1.0]     # Z-only C picked by inner CV (as in the original probe/review)
B_BOOT = 5000
EPS = 1e-12
SHUFFLE_SEED = 12345
SCALE_A = 50.0                        # review's primary stable scale (label oracle hits exactly 1.0)
SCALE_A_ROBUST = [100.0]             # pred_target robustness (s-artifact check)
MAX_ITER = 2000                      # lbfgs converges in ~20-35 iters at C=0.001 (review); 2000 ample
BAR = 0.040

CLIP = 'openai_clip-vit-large-patch14-336_HF'
QWEN = 'Qwen2.5-VL-7B-Instruct_HF'
ENCODERS = [('CLIP', CLIP), ('Qwen', QWEN)]
DATASETS = ['HateMM', 'MHC']          # HateMM train 744, MHC-EN train 549


# ---------------- data ----------------
def load_features(ds, enc):
    o = torch.load(f'data/CLIP_Embedding/{ds}/train_{enc}.pt', map_location='cpu', weights_only=False)
    ids = o['ids'][0]
    Z = np.concatenate([o['img_feats'].numpy(), o['text_feats'].numpy()], axis=1).astype(np.float64)
    y = o['labels'].numpy().astype(int)
    return ids, Z, y


def load_pred_onehot(ds, ids):
    """A_pred = Qwen-7B PREDICTED 9-way one-hot: cols 0..7 named targets, col 8 = none/unparsed.
    predicted primary==-1 (model says 'None') AND any train id missing from the pred file
    (unparsed) both map to col 8. Returns (A, prim_arr, coverage_facts)."""
    pred = json.load(open(f'data/gt/{ds}/target_pred_qwen7b.json'))
    pred = {k: v for k, v in pred.items() if not k.startswith('_')}
    n = len(ids); n_tgt = 8
    A = np.zeros((n, n_tgt + 1), dtype=np.float64)
    prim = np.full(n, -2, dtype=int)         # -2 == unparsed/missing sentinel
    n_missing = 0
    for i, s in enumerate(ids):
        if s in pred:
            p = int(pred[s]['primary'])
            prim[i] = p
            A[i, p if p != -1 else n_tgt] = 1.0
        else:
            n_missing += 1
            A[i, n_tgt] = 1.0                # unparsed -> none/unparsed column
    facts = {
        'pred_file': f'data/gt/{ds}/target_pred_qwen7b.json',
        'n_train': int(n), 'n_covered': int(n - n_missing),
        'coverage': float((n - n_missing) / n), 'n_unparsed_missing': int(n_missing),
        'n_pred_none(-1)': int((prim == -1).sum()),
    }
    return A, prim, facts


def load_gold_onehot(ds, ids):
    """A_gold = 9-way one-hot(gold primary target) with dedicated none(-1) col. HateMM only."""
    tm = json.load(open(f'data/gt/{ds}/target_map.json'))
    code_dict = tm['_meta']['code_dict']; n_tgt = tm['_meta']['num_targets']
    vids = {k: v for k, v in tm.items() if not k.startswith('_')}
    missing = [s for s in ids if s not in vids]
    assert not missing, f'{len(missing)} train ids missing in gold target_map: {missing[:5]}'
    prim = np.array([vids[s]['primary'] for s in ids], dtype=int)
    n = len(ids)
    A = np.zeros((n, n_tgt + 1), dtype=np.float64)
    for i, p in enumerate(prim):
        A[i, p if p != -1 else n_tgt] = 1.0
    return A, prim, code_dict


# ---------------- probe machinery (CORRECTED: standardize Z only, A raw x s) ----------------
def pick_C(Z, y):
    """Z-only inner-CV C pick, exactly as the original probe/review."""
    best_c, best_acc = C_GRID[0], -1.0
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    for c in C_GRID:
        accs = []
        sc = None
        for tr, te in skf.split(Z, y):
            scaler = StandardScaler().fit(Z[tr])
            lr = LogisticRegression(C=c, max_iter=MAX_ITER).fit(scaler.transform(Z[tr]), y[tr])
            accs.append((lr.predict(scaler.transform(Z[te])) == y[te]).mean())
        if np.mean(accs) > best_acc:
            best_acc, best_c = float(np.mean(accs)), c
    return best_c, best_acc


def cv_corrected(Z, A, y, C, scaleA):
    """Corrected machinery. Returns paired per-video arrays:
       (nll_z, cor_z) for baseline g(Z) and (nll_za, cor_za) for g'([Z, A*scaleA]).
       Standardize Z ONLY (fit train fold); Z regularized at C; A appended raw*scaleA
       (large scaleA => A effectively unpenalized). Baseline uses identical Z pipeline/folds."""
    n = len(y)
    nllz = np.zeros(n); corz = np.zeros(n); cnt = np.zeros(n)
    nllza = np.zeros(n); corza = np.zeros(n)
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=1000 + rep)
        for tr, te in skf.split(Z, y):
            sc = StandardScaler().fit(Z[tr])
            Ztr, Zte = sc.transform(Z[tr]), sc.transform(Z[te])
            # baseline Z-only
            lrb = LogisticRegression(C=C, max_iter=MAX_ITER).fit(Ztr, y[tr])
            pb = np.clip(lrb.predict_proba(Zte)[:, 1], EPS, 1 - EPS)
            ptb = np.where(y[te] == 1, pb, 1 - pb)
            nllz[te] += -np.log2(ptb); corz[te] += ((pb >= 0.5).astype(int) == y[te])
            # [Z, A*scaleA]
            Xtr = np.concatenate([Ztr, A[tr] * scaleA], axis=1)
            Xte = np.concatenate([Zte, A[te] * scaleA], axis=1)
            lr = LogisticRegression(C=C, max_iter=MAX_ITER).fit(Xtr, y[tr])
            p = np.clip(lr.predict_proba(Xte)[:, 1], EPS, 1 - EPS)
            pt = np.where(y[te] == 1, p, 1 - p)
            nllza[te] += -np.log2(pt); corza[te] += ((p >= 0.5).astype(int) == y[te])
            cnt[te] += 1
    return nllz / cnt, corz / cnt, nllza / cnt, corza / cnt


# ---------------- Fano bits->acc ----------------
def hb(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def hb_inv(L):
    L = min(max(L, 0.0), 1.0)
    lo, hi = 0.0, 0.5
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if hb(mid) < L:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fano_ceiling(mean_bits):
    return 1.0 - hb_inv(mean_bits)


# ---------------- clustered bootstrap over the 3 accuracy/bits deltas ----------------
def boot_all(nllz, corz, nllza, corza, B=B_BOOT):
    """Per-video (example-clustered) percentile bootstrap. Returns dict with mean+CI for:
       dacc  (direct held-out acc gain), dbits (bits saved/video),
       fano_dacc (Fano bits->acc PROJECTED gain). All paired on the same resample."""
    n = len(nllz)
    dacc = corza - corz
    dbits = nllz - nllza
    m_dacc = float(dacc.mean()); m_dbits = float(dbits.mean())
    # Fano PROJECTED acc gain = ceiling(ZA) - ceiling(Z): adding A lowers cross-entropy
    # => raises the accuracy ceiling (matches c3_g0cond_oracle_probe: fano_za - fano_z).
    m_fano = float(fano_ceiling(nllza.mean()) - fano_ceiling(nllz.mean()))
    bd_acc = np.empty(B); bd_bits = np.empty(B); bd_fano = np.empty(B)
    for b in range(B):
        idx = RNG.integers(0, n, n)
        bd_acc[b] = dacc[idx].mean()
        bd_bits[b] = dbits[idx].mean()
        bd_fano[b] = fano_ceiling(nllza[idx].mean()) - fano_ceiling(nllz[idx].mean())
    def ci(a): return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {
        'dacc_mean': m_dacc, 'dacc_ci': ci(bd_acc),
        'dbits_mean': m_dbits, 'dbits_ci': ci(bd_bits),
        'fano_dacc_mean': m_fano, 'fano_dacc_ci': ci(bd_fano),
        'acc_Z': float(corz.mean()), 'acc_ZA': float(corza.mean()),
        'bits_Z': float(nllz.mean()), 'bits_ZA': float(nllza.mean()),
    }


def run_arm(name, Z, A, y, C, scaleA):
    nllz, corz, nllza, corza = cv_corrected(Z, A, y, C, scaleA)
    r = boot_all(nllz, corz, nllza, corza)
    r['arm'] = name; r['A_dim'] = int(A.shape[1]); r['scaleA'] = scaleA
    return r


# ---------------- driver (per-cell checkpoint + resume; survives reaper kills) ----------------
def compute_cell(ds, enc, Z, y, A_pred, A_shuf, A_gold):
    C, cacc = pick_C(Z, y)
    A_lab = np.zeros((len(y), 2)); A_lab[np.arange(len(y)), y] = 1.0
    arms = {}
    arms['label_oracle'] = run_arm('label_oracle (CALIBRATION)', Z, A_lab, y, C, SCALE_A)
    arms['pred_target'] = run_arm('pred_target (REAL Qwen-7B)', Z, A_pred, y, C, SCALE_A)
    arms['shuffled_pred'] = run_arm('shuffled_pred (null control)', Z, A_shuf, y, C, SCALE_A)
    if A_gold is not None:
        arms['gold_target'] = run_arm('gold_target (oracle consistency)', Z, A_gold, y, C, SCALE_A)
    robust = {}
    for s in SCALE_A_ROBUST:
        robust[f'pred_s{s:g}'] = run_arm(f'pred_target s={s:g}', Z, A_pred, y, C, s)
    accZ = arms['label_oracle']['acc_Z']; lab_accZA = arms['label_oracle']['acc_ZA']
    headroom = 1.0 - accZ
    fano_frac = (arms['label_oracle']['dacc_mean'] / headroom) if headroom > 0 else float('nan')
    return {
        'encoder_file': f'data/CLIP_Embedding/{ds}/train_{enc}.pt',
        'Z_dim': int(Z.shape[1]), 'C': C, 'C_cv_acc': float(cacc),
        'baseline_acc_Z': float(accZ),
        'calibration': {
            'label_accZA': float(lab_accZA), 'headroom_1_minus_accZ': float(headroom),
            'label_dacc': arms['label_oracle']['dacc_mean'],
            'fano_headroom_fraction': float(fano_frac),
            'PASS': bool(lab_accZA >= 0.99)},
        'arms': arms, 'robustness': robust}


def main():
    outpath = sys.argv[1] if len(sys.argv) > 1 else 'c3_real_predictor_out.json'
    if os.path.exists(outpath):
        out = json.load(open(outpath))
        print(f'[resume] loaded {outpath}', flush=True)
    else:
        out = {'design': {
            'n_splits': N_SPLITS, 'n_repeats': N_REPEATS, 'C_grid': C_GRID,
            'B_boot': B_BOOT, 'scale_A': SCALE_A, 'scale_A_robust': SCALE_A_ROBUST,
            'max_iter': MAX_ITER, 'bar_dacc': BAR,
            'machinery': 'CORRECTED (standardize Z only; A raw one-hot x s, effectively unpenalized)',
            'rng_seed': 20260714, 'shuffle_seed': SHUFFLE_SEED}, 'datasets': {}}

    for ds in DATASETS:
        dcell = out['datasets'].get(ds, {'encoders': {}})
        # dataset-level shared arrays + facts (cheap; rebuilt each run)
        ids_ref, _, y0 = load_features(ds, ENCODERS[0][1])
        A_pred, prim_pred, pred_facts = load_pred_onehot(ds, ids_ref)
        perm = np.random.default_rng(SHUFFLE_SEED).permutation(len(ids_ref))
        A_shuf = A_pred[perm]
        try:
            A_gold, _, code_dict = load_gold_onehot(ds, ids_ref); gold_ok = True
        except FileNotFoundError:
            A_gold, code_dict, gold_ok = None, None, False
        cats = {}
        for p, yy in zip(prim_pred, y0):
            cats.setdefault(int(p), [0, 0])[int(yy)] += 1
        dcell['pred_coverage'] = pred_facts
        dcell['base_rate_pos'] = float(y0.mean())
        dcell['pred_target_alone_bayes'] = float(sum(max(c) for c in cats.values()) / len(y0))
        dcell['pred_primary_vs_label'] = {str(k): cats[k] for k in sorted(cats)}
        dcell['has_gold_target'] = gold_ok
        if gold_ok:
            dcell['gold_code_dict'] = code_dict
        out['datasets'][ds] = dcell

        for enc_name, enc in ENCODERS:
            if enc_name in dcell['encoders']:
                print(f'[skip] {ds}/{enc_name} already done', flush=True)
                continue
            ids, Z, y = load_features(ds, enc)
            assert ids == ids_ref, f'{ds} encoder ids order mismatch'
            cell = compute_cell(ds, enc, Z, y, A_pred, A_shuf, A_gold)
            dcell['encoders'][enc_name] = cell
            json.dump(out, open(outpath, 'w'), indent=1)   # CHECKPOINT after each cell
            cal = cell['calibration']
            print(f'\n[{ds}/{enc_name}] C={cell["C"]} Zdim={cell["Z_dim"]} '
                  f'accZ={cell["baseline_acc_Z"]:.4f} label_accZA={cal["label_accZA"]:.4f} '
                  f'(headroom_frac={cal["fano_headroom_fraction"]:.3f}) PASS={cal["PASS"]}', flush=True)
            for k in ['label_oracle', 'gold_target', 'pred_target', 'shuffled_pred']:
                if k not in cell['arms']:
                    continue
                r = cell['arms'][k]
                print(f'   {r["arm"]:34s} accZA={r["acc_ZA"]:.4f} '
                      f'dacc={r["dacc_mean"]:+.4f} CI[{r["dacc_ci"][0]:+.4f},{r["dacc_ci"][1]:+.4f}] '
                      f'dbits={r["dbits_mean"]:+.4f} CI[{r["dbits_ci"][0]:+.4f},{r["dbits_ci"][1]:+.4f}] '
                      f'fano_dacc={r["fano_dacc_mean"]:+.4f} CI[{r["fano_dacc_ci"][0]:+.4f},{r["fano_dacc_ci"][1]:+.4f}]',
                      flush=True)

    json.dump(out, open(outpath, 'w'), indent=1)
    print(f'\nwrote {outpath}', flush=True)


if __name__ == '__main__':
    main()
