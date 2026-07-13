#!/usr/bin/env python3
"""C3-NONTARGET FUSION probe — G0-cond gate on the BEST banked configuration (CPU, checkpointed).

Prescription: refine-logs/C3_NONTARGET_VERDICT_REVIEW.md section 4 (the MANDATORY prereg endpoint).
Design frozen in refine-logs/C3_FUSION_PROBE_RECORD.md BEFORE any number here was computed.

Question: does the generated-text channel A_text add conditional information ON TOP OF the pipeline's
best banked configuration Z_best = concat(CLIP img+text, Qwen img+text)?  (Pilot signal lived only at
MHC/CLIP and did NOT cross the MHC/Qwen baseline; honest prior on a Z_best gain ~ 0.)

Machinery mirrors scripts/analysis/c3_nontarget_probe.py + refine-logs/c3nt_verdict_review_diag.py
VERBATIM: Z standardized ALONE at its Z-only CV-optimal C_Z; auxiliary block appended standardized x
s=50 (effectively un-penalized), refit at C_Z; A_text via train-fold PCA (leak-free), k sliced from a
kmax PCA; 5x5 RepeatedStratifiedKFold rs=1000+rep; per-video correctness averaged over reps;
example-clustered (per-video) bootstrap B=5000 on Delta-acc; permutation null as a DISTRIBUTION over
>=100 fresh permutations of A across videos (all-k + max-over-k family correction).

Cells:
  MHC   x {Z_best(8960), Qwen(7168)}   -- MHC/Z_best = PRIMARY DECISION cell
  HateMM x {Z_best(8960), Qwen(7168)}  -- HateMM/Z_best = no-harm advisory
Arms per cell: baseline, label_oracle(calibration, must ~1.0), text_pca_k{8,16,32,64}, text_full_cvC,
shuffled_text seed 12345 (continuity only). Permutation null (>=100 seeds) run on both Z_best cells.

Gold label used PROBE-ONLY (calibration arm + targets). CPU-only, no GPU/SLURM/network. Not committed.
"""
import os, json, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
import numpy as np
import torch
import warnings
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

REPO = '/data/jehc223/RGCL'
ART = f'{REPO}/artifacts/c3_nontarget'
OUT = f'{REPO}/refine-logs/C3_FUSION_PROBE_OUT.json'
N_SPLITS, N_REPEATS = 5, 5
C_GRID = [0.001, 0.01, 0.1, 1.0]
B_BOOT = 5000
EPS = 1e-12
SCALE_A = 50.0
MAX_ITER = 2000
BAR = 0.040
SHUFFLE_SEED = 12345
KS_REPORT = [8, 16, 32, 64]      # point estimates reported (k32/k64 = context only)
KS_DECISION = [8, 16]            # pre-declared decision family + max-over-k correction family
BOOT_SEED = 20260714
NSEED = int(os.environ.get('NSEED', '150'))   # >=100 mandated; 150 mirrors the verdict review
CLIP = 'openai_clip-vit-large-patch14-336_HF'
QWEN = 'Qwen2.5-VL-7B-Instruct_HF'


# ---------------- data ----------------
def load_sample(ds):
    m = json.load(open(f'{ART}/{ds}_sample300.json'))
    ids = list(m['ids'])
    y = np.array([int(m['labels'][i]) for i in ids], dtype=int)
    return ids, y


def _load_enc(ds, enc, ids):
    o = torch.load(f'{REPO}/data/CLIP_Embedding/{ds}/train_{enc}.pt', map_location='cpu', weights_only=False)
    cache_ids = o['ids'][0]
    pos = {s: i for i, s in enumerate(cache_ids)}
    idx = [pos[s] for s in ids]
    img = o['img_feats'].numpy()[idx].astype(np.float64)
    txt = o['text_feats'].numpy()[idx].astype(np.float64)
    yc = o['labels'].numpy().astype(int)[idx]
    return np.concatenate([img, txt], axis=1), yc


def load_Z(ds, variant, ids, y):
    """variant 'Zbest' = concat(CLIP,Qwen); 'Qwen' = Qwen alone."""
    Zc, yc = _load_enc(ds, CLIP, ids)
    Zq, yq = _load_enc(ds, QWEN, ids)
    assert np.array_equal(yc, y) and np.array_equal(yq, y), f'{ds} label order mismatch'
    return np.concatenate([Zc, Zq], axis=1) if variant == 'Zbest' else Zq


def load_Atext(ds, ids):
    vecs = []; nz = 0
    for s in ids:
        v = np.load(f'{ART}/{ds}/emb/{s}.npy').astype(np.float64)
        if not np.any(v):
            nz += 1
        vecs.append(v)
    return np.stack(vecs, axis=0), nz


# ---------------- probe machinery (verbatim from the verdict-review diag) ----------------
def pick_C(Z, y):
    best_c, best = C_GRID[0], -1.0
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=0)
    for c in C_GRID:
        a = []
        for tr, te in skf.split(Z, y):
            sc = StandardScaler().fit(Z[tr])
            lr = LogisticRegression(C=c, max_iter=MAX_ITER).fit(sc.transform(Z[tr]), y[tr])
            a.append((lr.predict(sc.transform(Z[te])) == y[te]).mean())
        if np.mean(a) > best:
            best, best_c = float(np.mean(a)), c
    return best_c, best


def pick_C_combined(Z, A, y):
    best_c, best = C_GRID[0], -1.0
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=0)
    for c in C_GRID:
        a = []
        for tr, te in skf.split(Z, y):
            scz = StandardScaler().fit(Z[tr]); sca = StandardScaler().fit(A[tr])
            Xtr = np.concatenate([scz.transform(Z[tr]), sca.transform(A[tr])], axis=1)
            Xte = np.concatenate([scz.transform(Z[te]), sca.transform(A[te])], axis=1)
            lr = LogisticRegression(C=c, max_iter=MAX_ITER).fit(Xtr, y[tr])
            a.append((lr.predict(Xte) == y[te]).mean())
        if np.mean(a) > best:
            best, best_c = float(np.mean(a)), c
    return best_c, best


def _fit_cor(Xtr, ytr, Xte, yte, C):
    lr = LogisticRegression(C=C, max_iter=MAX_ITER).fit(Xtr, ytr)
    return ((lr.predict_proba(Xte)[:, 1] >= 0.5).astype(int) == yte).astype(float)


def baseline_cor(Z, y, C_Z):
    n = len(y); cor = np.zeros(n); cnt = np.zeros(n)
    for rep in range(N_REPEATS):
        for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=1000 + rep).split(Z, y):
            scZ = StandardScaler().fit(Z[tr]); cnt[te] += 1
            cor[te] += _fit_cor(scZ.transform(Z[tr]), y[tr], scZ.transform(Z[te]), y[te], C_Z)
    return cor / cnt


def oracle_cor(Z, y, C_Z):
    """label-oracle calibration arm: append 2-col one-hot(y) x s (raw, unpenalized)."""
    n = len(y); A_lab = np.zeros((n, 2)); A_lab[np.arange(n), y] = 1.0
    cor = np.zeros(n); cnt = np.zeros(n)
    for rep in range(N_REPEATS):
        for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=1000 + rep).split(Z, y):
            scZ = StandardScaler().fit(Z[tr]); Ztr, Zte = scZ.transform(Z[tr]), scZ.transform(Z[te]); cnt[te] += 1
            Xtr = np.concatenate([Ztr, A_lab[tr] * SCALE_A], axis=1)
            Xte = np.concatenate([Zte, A_lab[te] * SCALE_A], axis=1)
            cor[te] += _fit_cor(Xtr, y[tr], Xte, y[te], C_Z)
    return cor / cnt


def full_cor(Z, A, y, C_full):
    """full-dim capacity-matched secondary arm: [Z_std, A_std] at combined CV-tuned C."""
    n = len(y); cor = np.zeros(n); cnt = np.zeros(n)
    for rep in range(N_REPEATS):
        for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=1000 + rep).split(Z, y):
            scZ = StandardScaler().fit(Z[tr]); scA = StandardScaler().fit(A[tr]); cnt[te] += 1
            Xtr = np.concatenate([scZ.transform(Z[tr]), scA.transform(A[tr])], axis=1)
            Xte = np.concatenate([scZ.transform(Z[te]), scA.transform(A[te])], axis=1)
            cor[te] += _fit_cor(Xtr, y[tr], Xte, y[te], C_full)
    return cor / cnt


def arm_cor_allk(Z, y, C_Z, src, ks):
    """per-video cor for every k in ks in ONE CV pass (PCA fit to max(ks) once, sliced)."""
    n = len(y); cor = {k: np.zeros(n) for k in ks}; cnt = np.zeros(n); kmax = max(ks)
    for rep in range(N_REPEATS):
        for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=1000 + rep).split(Z, y):
            scZ = StandardScaler().fit(Z[tr]); Ztr, Zte = scZ.transform(Z[tr]), scZ.transform(Z[te]); cnt[te] += 1
            scS = StandardScaler().fit(src[tr]); Str, Ste = scS.transform(src[tr]), scS.transform(src[te])
            kk = min(kmax, len(tr) - 1, src.shape[1]); pca = PCA(n_components=kk, random_state=0).fit(Str)
            Ptr, Pte = pca.transform(Str), pca.transform(Ste)
            for k in ks:
                j = min(k, kk); scP = StandardScaler().fit(Ptr[:, :j])
                Btr = scP.transform(Ptr[:, :j]) * SCALE_A; Bte = scP.transform(Pte[:, :j]) * SCALE_A
                cor[k][te] += _fit_cor(np.concatenate([Ztr, Btr], 1), y[tr],
                                       np.concatenate([Zte, Bte], 1), y[te], C_Z)
    return {k: cor[k] / cnt for k in ks}


def dmean(cor_arm, cor_base):
    return float((cor_arm - cor_base).mean())


def boot_ci(cor_arm, cor_base, seed):
    """example-clustered (per-video) bootstrap of Delta-acc; each row = one video."""
    d = cor_arm - cor_base; n = len(d); rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B_BOOT)])
    return float(d.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


# ---------------- driver ----------------
def point_arms(ds, variant):
    """Compute baseline + calibration + text arms (point Delta-acc + per-video-clustered CI).
    Returns a dict; also caches base cor / C_Z for the permutation-null stage."""
    ids, y = load_sample(ds); n = len(y)
    Z = load_Z(ds, variant, ids, y); A, nz = load_Atext(ds, ids)
    C_Z, cacc = pick_C(Z, y); C_full, cfacc = pick_C_combined(Z, A, y)
    base = baseline_cor(Z, y, C_Z); accZ = float(base.mean())
    orac = oracle_cor(Z, y, C_Z); accZA_lab = float(orac.mean())
    headroom = 1.0 - accZ
    cal = {'label_accZA': accZA_lab, 'headroom_1_minus_accZ': float(headroom),
           'label_dacc': float((orac - base).mean()),
           'headroom_fraction': float((orac - base).mean() / headroom) if headroom > 0 else float('nan'),
           'PASS': bool(accZA_lab >= 0.99)}
    rk = arm_cor_allk(Z, y, C_Z, A, KS_REPORT)
    full = full_cor(Z, A, y, C_full)
    A_shuf = A[np.random.default_rng(SHUFFLE_SEED).permutation(n)]
    sk = arm_cor_allk(Z, y, C_Z, A_shuf, KS_DECISION)
    arms = {}
    for k in KS_REPORT:
        m, ci = boot_ci(rk[k], base, BOOT_SEED + k)
        arms[f'text_pca_k{k}'] = {'accZA': float(rk[k].mean()), 'dacc': m, 'ci': ci}
    m, ci = boot_ci(full, base, BOOT_SEED + 999)
    arms['text_full_cvC'] = {'accZA': float(full.mean()), 'dacc': m, 'ci': ci}
    for k in KS_DECISION:
        arms[f'shuffled_text_k{k}'] = {'dacc': dmean(sk[k], base)}
    return {'ds': ds, 'variant': variant, 'n': n, 'n_pos': int(y.sum()), 'n_zero_Atext': int(nz),
            'Z_dim': int(Z.shape[1]), 'C_Z': C_Z, 'C_Z_cv_acc': float(cacc),
            'C_full': C_full, 'C_full_cv_acc': float(cfacc), 'baseline_accZ': accZ,
            'calibration': cal, 'arms': arms,
            'real_max_over_kdec': float(max(arms[f'text_pca_k{k}']['dacc'] for k in KS_DECISION))}


def perm_null(ds, variant, base_dacc_maxdec, existing):
    """>=NSEED fresh permutations of A across videos; all-k(decision) + max-over-k distribution."""
    ids, y = load_sample(ds); n = len(y)
    Z = load_Z(ds, variant, ids, y); A, _ = load_Atext(ds, ids)
    C_Z, _ = pick_C(Z, y); base = baseline_cor(Z, y, C_Z)
    perk = {str(k): existing.get('perk', {}).get(str(k), []) for k in KS_DECISION}
    maxk = existing.get('maxk', [])
    for si in range(len(maxk), NSEED):
        perm = np.random.default_rng(70000 + si).permutation(n)
        c = arm_cor_allk(Z, y, C_Z, A[perm], KS_DECISION)
        dk = {k: dmean(c[k], base) for k in KS_DECISION}
        for k in KS_DECISION:
            perk[str(k)].append(dk[k])
        maxk.append(float(max(dk.values())))
        if si % 10 == 9 or si == NSEED - 1:
            yield _perm_stats(perk, maxk, base_dacc_maxdec)
    yield _perm_stats(perk, maxk, base_dacc_maxdec)


def _perm_stats(perk, maxk, real_maxdec):
    am = np.array(maxk); st = {'n_seed': len(maxk), 'perk': perk, 'maxk': maxk, 'perk_stats': {}}
    for k, vals in perk.items():
        a = np.array(vals)
        st['perk_stats'][k] = {'mean': float(a.mean()), 'sd': float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                               'max': float(a.max()), 'q': [float(np.percentile(a, q)) for q in (2.5, 50, 97.5)]}
    st['maxk_mean'] = float(am.mean()); st['maxk_sd'] = float(am.std(ddof=1)) if len(am) > 1 else 0.0
    st['maxk_max'] = float(am.max()); st['maxk_q'] = [float(np.percentile(am, q)) for q in (2.5, 50, 97.5)]
    st['p_realmax_ge_permmax'] = float((am >= real_maxdec).mean())
    st['real_beats_all_permmax'] = bool(real_maxdec > am.max())
    st['p_permmax_ge_bar'] = float((am >= BAR).mean())
    return st


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {
        'design': {'Z_best': 'concat(CLIP img+text, Qwen img+text)=8960d', 'Z_secondary': 'Qwen alone=7168d',
                   'ks_decision': KS_DECISION, 'ks_report': KS_REPORT, 'bar': BAR, 'scale_A': SCALE_A,
                   'n_splits': N_SPLITS, 'n_repeats': N_REPEATS, 'B_boot': B_BOOT, 'nseed_perm': NSEED,
                   'machinery': 'Z std alone @ C_Z; aux block std x s=50 unpenalized; A via train-fold PCA'},
        'cells': {}}
    # Cell order: decision first, then secondary/advisory.
    CELLS = [('MHC', 'Zbest'), ('MHC', 'Qwen'), ('HateMM', 'Zbest'), ('HateMM', 'Qwen')]
    PERM_CELLS = {('MHC', 'Zbest'), ('HateMM', 'Zbest')}  # rule/no-harm are on Z_best
    for ds, variant in CELLS:
        key = f'{ds}|{variant}'
        cell = out['cells'].get(key, {})
        if 'point' not in cell:
            t = time.time(); cell['point'] = point_arms(ds, variant)
            out['cells'][key] = cell; json.dump(out, open(OUT, 'w'), indent=1)
            p = cell['point']; cal = p['calibration']
            print(f"[{key}] Zdim={p['Z_dim']} C_Z={p['C_Z']} accZ={p['baseline_accZ']:.4f} "
                  f"label_accZA={cal['label_accZA']:.4f} (hfrac={cal['headroom_fraction']:.3f}) "
                  f"CALIB_PASS={cal['PASS']}  [{time.time()-t:.0f}s]", flush=True)
            for k in KS_REPORT:
                a = p['arms'][f'text_pca_k{k}']
                print(f"   text_pca_k{k:<2d} accZA={a['accZA']:.4f} dacc={a['dacc']:+.4f} "
                      f"CI[{a['ci'][0]:+.4f},{a['ci'][1]:+.4f}]", flush=True)
            a = p['arms']['text_full_cvC']
            print(f"   text_full_cvC accZA={a['accZA']:.4f} dacc={a['dacc']:+.4f} "
                  f"CI[{a['ci'][0]:+.4f},{a['ci'][1]:+.4f}]", flush=True)
            print(f"   shuffled(seed12345) k8={p['arms']['shuffled_text_k8']['dacc']:+.4f} "
                  f"k16={p['arms']['shuffled_text_k16']['dacc']:+.4f}  "
                  f"real_max_over_kdec={p['real_max_over_kdec']:+.4f}", flush=True)
        if (ds, variant) in PERM_CELLS:
            real_maxdec = cell['point']['real_max_over_kdec']
            existing = cell.get('perm_null', {})
            if existing.get('n_seed', 0) < NSEED:
                t = time.time()
                for st in perm_null(ds, variant, real_maxdec, existing):
                    cell['perm_null'] = st; out['cells'][key] = cell
                    json.dump(out, open(OUT, 'w'), indent=1)
                st = cell['perm_null']
                print(f"[{key}] PERM n={st['n_seed']} k8 mean={st['perk_stats']['8']['mean']:+.4f} "
                      f"sd={st['perk_stats']['8']['sd']:.4f} max={st['perk_stats']['8']['max']:+.4f} | "
                      f"maxk mean={st['maxk_mean']:+.4f} max={st['maxk_max']:+.4f} | "
                      f"p(realmax>=permmax)={st['p_realmax_ge_permmax']:.4f} "
                      f"real_beats_all={st['real_beats_all_permmax']}  [{time.time()-t:.0f}s]", flush=True)

    # ---- verdict (decision cell = MHC|Zbest) ----
    dcell = out['cells']['MHC|Zbest']; p = dcell['point']; pn = dcell['perm_null']
    calib_pass = p['calibration']['PASS']
    best_k = max(KS_DECISION, key=lambda k: p['arms'][f'text_pca_k{k}']['dacc'])
    best = p['arms'][f'text_pca_k{best_k}']
    C1 = bool(best['dacc'] >= BAR)
    C2 = bool(best['ci'][0] > 0)
    C3 = bool(pn['real_beats_all_permmax'])
    if not calib_pass:
        verdict = 'MACHINERY_INVALID'
    elif C1 and C2 and C3:
        verdict = 'C3_FUSION_PROCEED'
    else:
        verdict = 'C3_NONTARGET_DEAD_AT_FUSION'
    out['verdict'] = {'decision_cell': 'MHC|Zbest', 'calib_pass': bool(calib_pass),
                      'best_decision_k': best_k, 'best_dacc': best['dacc'], 'best_ci': best['ci'],
                      'C1_point_ge_040': C1, 'C2_ci_low_gt_0': C2,
                      'C3_real_beats_all_permmax': C3,
                      'perm_maxk_mean': pn['maxk_mean'], 'perm_maxk_max': pn['maxk_max'],
                      'p_realmax_ge_permmax': pn['p_realmax_ge_permmax'],
                      'VERDICT': verdict}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\n==== VERDICT={verdict} | calib={calib_pass} C1={C1} C2={C2} C3={C3} "
          f"best k{best_k} dacc={best['dacc']:+.4f} CI[{best['ci'][0]:+.4f},{best['ci'][1]:+.4f}] "
          f"perm_maxk_max={pn['maxk_max']:+.4f} ====", flush=True)
    print(f'wrote {OUT}', flush=True)


if __name__ == '__main__':
    t0 = time.time(); main(); print(f'elapsed {time.time()-t0:.0f}s', flush=True)
