#!/usr/bin/env python3
"""APX G0-cond conditional-information gate (Wave-3 candidate #3) -- NON-BINDING $0 audio screen.

Design frozen in refine-logs/WAVE3_CANDIDATES.md CANDIDATE 3 sections (d)/(e) (commit 0ee06df),
followed VERBATIM. Pre-ceremony cheap-kill screen (precedent: W2-C CLIP-K4 pre-check, ad48dcc):
NON-binding, no prereg freeze, full raw-only discipline.

Machinery reuses scripts/analysis/c3_fusion_probe.py VERBATIM (C3-template conditional-info probe with
Amendment-1 Z_best): Z standardized ALONE at its Z-only inner-CV-optimal C_Z; auxiliary block appended
standardized x s=50 (effectively un-penalized), refit at C_Z; aux via train-fold PCA (leak-free), k
sliced from a kmax PCA; 5x5 RepeatedStratifiedKFold rs=1000+rep; per-video correctness averaged;
example-clustered (per-video) bootstrap B=5000 on Delta-acc; mandatory label-oracle calibration arm
(accZA >= 0.99 or MACHINERY_INVALID); permutation null as a DISTRIBUTION over >=150 fresh permutations
(only needed to confirm a would-be pass; K-APX-0 as written is a 2-condition OR-kill).

Question: does the whole-video openSMILE eGeMAPSv02 prosodic embedding (88-d: pitch, loudness,
jitter/shimmer, spectral) carry conditional label information OVER Z_best = concat(CLIP img+text,
Qwen img+text) (8960-d)?  The honest hazard (F31) is that the whisper-large-v3 transcript already banks
spoken-hate content, so I(label; audio | Z_best) may be < +0.040 -- this classical-prosody probe
upper-bounds the cheap realization and de-risks the acoustic axis at ~$0 before any download question.

Cell: HateMM ONLY (design is HateMM-primary; MHC-EN is data-limited per WAVE3 section 0). N=851 train U val.
Zero test-touch. Gold labels used PROBE-ONLY (calibration arm + CV stratification). CPU-only.
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
OUT = f'{REPO}/refine-logs/APX_G0COND_GATE_OUT.json'
EGEMAPS_CACHE = f'{REPO}/data/audio/HateMM/egemaps_v02_trainval.pt'
# ---- constants: identical to c3_fusion_probe.py (verbatim machinery) ----
N_SPLITS, N_REPEATS = 5, 5
C_GRID = [0.001, 0.01, 0.1, 1.0]
B_BOOT = 5000
SCALE_A = 50.0
MAX_ITER = 2000
BAR = 0.040
SHUFFLE_SEED = 12345
KS_REPORT = [8, 16, 32, 64]      # point estimates reported (k32/k64 = context only)
KS_DECISION = [8, 16]            # pre-declared decision family + max-over-k correction family
BOOT_SEED = 20260714
NSEED = int(os.environ.get('NSEED', '150'))   # >=150 (only computed on a would-be pass)
CLIP = 'openai_clip-vit-large-patch14-336_HF'
QWEN = 'Qwen2.5-VL-7B-Instruct_HF'
DS = 'HateMM'


# ---------------- data (APX-specific loader; features-only, train U val, HateMM) ----------------
def _pooled_id2row(ds, enc):
    d = {}
    for split in ('train', 'dev_seen'):
        o = torch.load(f'{REPO}/data/CLIP_Embedding/{ds}/{split}_{enc}.pt',
                       map_location='cpu', weights_only=False)
        ids = o['ids'][0]
        img = o['img_feats'].numpy().astype(np.float64)
        txt = o['text_feats'].numpy().astype(np.float64)
        lab = o['labels'].numpy().astype(int)
        for i, s in enumerate(ids):
            d[s] = (np.concatenate([img[i], txt[i]]), int(lab[i]))
    return d


def load_cell():
    """Canonical order = the eGeMAPS cache id order (= frameset train (+) dev_seen). Build Z_best(8960)
    aligned to it, assert label agreement. Zero test-touch."""
    e = torch.load(EGEMAPS_CACHE, map_location='cpu', weights_only=False)
    ids = list(e['ids'][0])
    A = e['egemaps'].numpy().astype(np.float64)          # [N, 88]
    y = e['labels'].numpy().astype(int)                  # [N]
    clip = _pooled_id2row(DS, CLIP); qwen = _pooled_id2row(DS, QWEN)
    N = len(ids); Z = np.zeros((N, 8960), dtype=np.float64)
    for i, s in enumerate(ids):
        zc, yc = clip[s]; zq, yq = qwen[s]
        assert yc == yq == int(y[i]), f'{DS} label mismatch at {s}: clip={yc} qwen={yq} egemaps={y[i]}'
        Z[i] = np.concatenate([zc, zq])
    assert Z.shape[1] == 8960 and A.shape[1] == 88, (Z.shape, A.shape)
    n_zero = int((~A.any(axis=1)).sum())
    return ids, y, Z, A, {'N': N, 'n_pos': int(y.sum()), 'n_zero_audio_rows': n_zero,
                          'opensmile_version': e.get('opensmile_version'), 'feature_set': e.get('feature_set')}


# ---------------- probe machinery (VERBATIM from scripts/analysis/c3_fusion_probe.py) ----------------
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
    d = cor_arm - cor_base; n = len(d); rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B_BOOT)])
    return float(d.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


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


# ---------------- driver ----------------
def point_arms():
    ids, y, Z, A, meta = load_cell()
    n = len(y)
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
        arms[f'audio_pca_k{k}'] = {'accZA': float(rk[k].mean()), 'dacc': m, 'ci': ci}
    m, ci = boot_ci(full, base, BOOT_SEED + 999)
    arms['audio_full_cvC'] = {'accZA': float(full.mean()), 'dacc': m, 'ci': ci}
    for k in KS_DECISION:
        arms[f'shuffled_audio_k{k}'] = {'dacc': dmean(sk[k], base)}
    return {'ds': DS, 'n': n, 'n_pos': int(y.sum()), 'meta': meta,
            'Z_dim': int(Z.shape[1]), 'aux_dim': int(A.shape[1]),
            'C_Z': C_Z, 'C_Z_cv_acc': float(cacc), 'C_full': C_full, 'C_full_cv_acc': float(cfacc),
            'baseline_accZ': accZ, 'calibration': cal, 'arms': arms,
            'real_max_over_kdec': float(max(arms[f'audio_pca_k{k}']['dacc'] for k in KS_DECISION)),
            '_cache': {'C_Z': C_Z}}, (ids, y, Z, A, base, C_Z)


def perm_null(state, real_maxdec, existing):
    ids, y, Z, A, base, C_Z = state; n = len(y)
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
            yield _perm_stats(perk, maxk, real_maxdec)
    yield _perm_stats(perk, maxk, real_maxdec)


def eval_gate(cell):
    """K-APX-0 (2-condition OR-kill) + K-APX-1 (calibration). Binding point = best of decision {k8,k16}."""
    p = cell['point']; cal = p['calibration']
    calib_pass = bool(cal['PASS'])
    best_k = max(KS_DECISION, key=lambda k: p['arms'][f'audio_pca_k{k}']['dacc'])
    best = p['arms'][f'audio_pca_k{best_k}']
    C1 = bool(best['dacc'] >= BAR)          # point >= +0.040
    C2 = bool(best['ci'][0] > 0)            # bootstrap CI-lower > 0
    pn = cell.get('perm_null')
    C3 = bool(pn['real_beats_all_permmax']) if pn else None
    if not calib_pass:
        verdict = 'MACHINERY_INVALID'       # K-APX-1
    elif not C1 or not C2:
        verdict = 'KILL'                    # K-APX-0 OR-kill (point<0.040 OR CI-low<=0)
    else:
        verdict = 'PASS_SIDE'               # clears K-APX-0's 2 conditions (perm-null confirmation optional)
    return {'calib_pass': calib_pass, 'best_decision_k': best_k, 'best_dacc': best['dacc'],
            'best_ci': best['ci'], 'C1_point_ge_040': C1, 'C2_ci_low_gt_0': C2,
            'C3_real_beats_all_permmax': C3, 'verdict': verdict}


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {
        'design': {'gate': 'APX G0-cond audio (Wave-3 #3)', 'aux': 'openSMILE eGeMAPSv02 88-d whole-video prosody',
                   'Z_best': 'concat(CLIP img+text, Qwen img+text)=8960d', 'dataset': 'HateMM (primary)',
                   'ks_decision': KS_DECISION, 'ks_report': KS_REPORT, 'bar': BAR, 'scale_A': SCALE_A,
                   'n_splits': N_SPLITS, 'n_repeats': N_REPEATS, 'B_boot': B_BOOT, 'nseed_perm': NSEED,
                   'kill_switch': 'K-APX-0: point<+0.040 OR CI-lower<=0 -> KILL; K-APX-1: calib accZA<0.99 -> MACHINERY_INVALID',
                   'machinery': 'VERBATIM c3_fusion_probe.py', 'NON_BINDING': True},
        'cell': {}}
    if 'point' not in out['cell']:
        t = time.time(); p, state = point_arms(); out['cell']['point'] = p
        json.dump(out, open(OUT, 'w'), indent=1)
        cal = p['calibration']
        print(f"[{DS}] Zdim={p['Z_dim']} auxdim={p['aux_dim']} C_Z={p['C_Z']} accZ={p['baseline_accZ']:.4f} "
              f"label_accZA={cal['label_accZA']:.4f} (hfrac={cal['headroom_fraction']:.3f}) "
              f"CALIB_PASS={cal['PASS']}  [{time.time()-t:.0f}s]", flush=True)
        for k in KS_REPORT:
            a = p['arms'][f'audio_pca_k{k}']
            print(f"   audio_pca_k{k:<2d} accZA={a['accZA']:.4f} dacc={a['dacc']:+.4f} "
                  f"CI[{a['ci'][0]:+.4f},{a['ci'][1]:+.4f}]", flush=True)
        a = p['arms']['audio_full_cvC']
        print(f"   audio_full_cvC accZA={a['accZA']:.4f} dacc={a['dacc']:+.4f} "
              f"CI[{a['ci'][0]:+.4f},{a['ci'][1]:+.4f}]", flush=True)
        print(f"   shuffled(seed12345) k8={p['arms']['shuffled_audio_k8']['dacc']:+.4f} "
              f"k16={p['arms']['shuffled_audio_k16']['dacc']:+.4f}  "
              f"real_max_over_kdec={p['real_max_over_kdec']:+.4f}", flush=True)
    else:
        _, state = point_arms()  # rebuild state for a possible perm-null resume (deterministic)

    g = eval_gate(out['cell'])
    print(f"   -> GATE(point) C1={g['C1_point_ge_040']} C2={g['C2_ci_low_gt_0']} "
          f"calib={g['calib_pass']} verdict={g['verdict']}", flush=True)

    # perm null only to CONFIRM a would-be pass (K-APX-0 as written is 2-condition; C3 is extra rigor)
    if g['verdict'] == 'PASS_SIDE':
        real_maxdec = out['cell']['point']['real_max_over_kdec']
        existing = out['cell'].get('perm_null', {})
        if existing.get('n_seed', 0) < NSEED:
            for st in perm_null(state, real_maxdec, existing):
                out['cell']['perm_null'] = st; json.dump(out, open(OUT, 'w'), indent=1)
            st = out['cell']['perm_null']
            print(f"[{DS}] PERM n={st['n_seed']} maxk mean={st['maxk_mean']:+.4f} max={st['maxk_max']:+.4f} "
                  f"p(realmax>=permmax)={st['p_realmax_ge_permmax']:.4f} "
                  f"real_beats_all={st['real_beats_all_permmax']}", flush=True)

    out['gate_eval'] = eval_gate(out['cell'])
    v = out['gate_eval']['verdict']
    out['apx_verdict'] = {'K_APX_0_verdict': v,
                          'acoustic_axis': ('cleared_gate_download_ruling_next' if v == 'PASS_SIDE'
                                            else 'prior_slashed_no_download_escalation' if v == 'KILL'
                                            else 'machinery_invalid')}
    json.dump(out, open(OUT, 'w'), indent=1)
    ge = out['gate_eval']
    print('\n==== APX G0-cond audio gate — mechanical evaluation (HateMM) ====', flush=True)
    print(f"  best-k={ge['best_decision_k']} dacc={ge['best_dacc']:+.4f} "
          f"CI[{ge['best_ci'][0]:+.4f},{ge['best_ci'][1]:+.4f}] C1={ge['C1_point_ge_040']} "
          f"C2={ge['C2_ci_low_gt_0']} C3={ge['C3_real_beats_all_permmax']} calib={ge['calib_pass']} "
          f"=> {v}", flush=True)
    print(f"  APX acoustic axis: {out['apx_verdict']['acoustic_axis']}", flush=True)
    print(f'wrote {OUT}', flush=True)


if __name__ == '__main__':
    t0 = time.time(); main(); print(f'elapsed {time.time()-t0:.0f}s', flush=True)
