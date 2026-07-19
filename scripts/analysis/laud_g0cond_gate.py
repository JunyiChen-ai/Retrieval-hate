#!/usr/bin/env python3
"""LEARNED-AUDIO G0-cond conditional-information gate — the decisive $0 CPU screen (Stage C).

Fork of scripts/analysis/apx_g0cond_gate.py (which reuses scripts/analysis/c3_fusion_probe.py
VERBATIM — the C3-template conditional-info machinery that rendered the binding W2-A / APX / K9
verdicts). Design frozen in refine-logs/AUDIO_AXIS_FORENSIC_RECON.md section 4 (commit 166f9e2b),
followed VERBATIM. Pre-declared kill-switches (K-LAUD-0/1/2) frozen in refine-logs/LAUD_GATE_RECORD.md
BEFORE any number here was computed.

The ONLY data-layer changes vs the APX gate:
  (1) aux block = the LEARNED Whisper-large-v3 encoder video vector (2*d_model = 2560-d, mean(+)max
      pooled), replacing the classical eGeMAPS-88-d prosody vector (which F41/APX killed on HateMM).
  (2) generalized to THREE datasets, each with TWO conditioning Z arms:
        - deployed_7168 : the per-dataset DEPLOYED winning encoder img(+)text (the honest
          "does audio add over what we actually deploy?"). HateMM=LoRA-curric, EN=frozen-Qwen,
          ZH=LoRA.
        - strict_8960   : the exact W2-A/APX Z_best = CLIP img(+)text (+) frozen-Qwen img(+)text
          (guards the weak-deployed-encoder loophole). A PASS must clear BOTH arms (recon sec 4).

Question: does I(label; whisper-audio | Z) exceed +0.040 (K9 house bar)?  Honest prior LOW ~10-15%
(F31: the large-v3 TRANSCRIPT already banks spoken hate into text_feats; the blessed increment is the
non-lexical / non-speech residual only).

Datasets: HateMM (acoustic anchor / primary), MHC=EN (K-LAUD-2 blank-cell fill — audio conditional
info NEVER measured, even classically), MHC_zh=ZH (third leg). Zero test-touch (train (union) val
only). Gold labels used PROBE-ONLY (calibration arm + CV stratification). CPU-only.
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
OUT = f'{REPO}/refine-logs/LAUD_GATE_OUT.json'
MODEL_TAG = os.environ.get('LAUD_MODEL_TAG', 'whisper-large-v3')

# ---- constants: identical to c3_fusion_probe.py / apx_g0cond_gate.py (verbatim machinery) ----
N_SPLITS, N_REPEATS = 5, 5
C_GRID = [0.001, 0.01, 0.1, 1.0]
B_BOOT = 5000
SCALE_A = 50.0
MAX_ITER = 2000
BAR = 0.040                      # K-LAUD-0 promote bar (K9 house standard, == APX/W2-A)
HONEST_PARTIAL_LOW = 0.030       # +0.030..0.040 with CI-low>0 = documented HONEST-PARTIAL flag (recon sec 4)
SHUFFLE_SEED = 12345
KS_REPORT = [8, 16, 32, 64]      # point estimates reported (k32/k64 = context only)
KS_DECISION = [8, 16]            # pre-declared decision family + max-over-k correction family
BOOT_SEED = 20260714
NSEED = int(os.environ.get('NSEED', '150'))   # >=150 (only computed on a would-be pass)

CLIP = 'openai_clip-vit-large-patch14-336_HF'
FROZEN_QWEN = 'Qwen2.5-VL-7B-Instruct_HF'     # strict-8960 arm Qwen (== the exact W2-A/APX baseline)
DEPLOY = {                                    # per-dataset DEPLOYED winning encoder (7168-d img+text)
    'HateMM': 'Qwen2.5-VL-7B-Instruct-LoRA-curric_HF',
    'MHC':    'Qwen2.5-VL-7B-Instruct_HF',
    'MHC_zh': 'Qwen2.5-VL-7B-Instruct-LoRA_HF',
}
DATASETS = ['HateMM', 'MHC', 'MHC_zh']        # HateMM = acoustic anchor (decision); EN = K-LAUD-2; ZH = third leg
Z_ARMS = ['deployed_7168', 'strict_8960']


# ---------------- data (LAUD-specific loader; features-only, train U val) ----------------
def _pooled_id2row(ds, enc):
    """id -> (concat(img_feats, text_feats), label) over train (+) dev_seen. features-only."""
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


def load_cell(ds, z_arm):
    """Canonical order = the whisper aux cache id order (= gt train (+) val). Build Z aligned to it,
    assert label agreement across every cache. Zero test-touch."""
    w = torch.load(f'{REPO}/data/audio/{ds}/whisper_{MODEL_TAG}_trainval.pt',
                   map_location='cpu', weights_only=False)
    ids = list(w['ids'][0])
    A = w['emb'].numpy().astype(np.float64)              # [N, 2560]
    y = w['labels'].numpy().astype(int)                  # [N]
    N = len(ids)
    if z_arm == 'deployed_7168':
        dep = _pooled_id2row(ds, DEPLOY[ds])
        Zdim = 7168
        Z = np.zeros((N, Zdim), dtype=np.float64)
        for i, s in enumerate(ids):
            zv, yv = dep[s]
            assert yv == int(y[i]), f'{ds} label mismatch at {s}: deploy={yv} whisper={y[i]}'
            Z[i] = zv
    elif z_arm == 'strict_8960':
        clip = _pooled_id2row(ds, CLIP); qwen = _pooled_id2row(ds, FROZEN_QWEN)
        Zdim = 8960
        Z = np.zeros((N, Zdim), dtype=np.float64)
        for i, s in enumerate(ids):
            zc, yc = clip[s]; zq, yq = qwen[s]
            assert yc == yq == int(y[i]), f'{ds} label mismatch at {s}: clip={yc} qwen={yq} whisper={y[i]}'
            Z[i] = np.concatenate([zc, zq])
    else:
        raise ValueError(z_arm)
    assert Z.shape[1] == Zdim and A.shape[1] == int(w['pooled_dim']), (Z.shape, A.shape, w['pooled_dim'])
    n_zero = int((~A.any(axis=1)).sum())
    meta = {'N': N, 'n_pos': int(y.sum()), 'n_zero_audio_rows': n_zero,
            'aux_dim': int(A.shape[1]), 'Z_dim': int(Zdim), 'model_tag': w.get('model_tag'),
            'pool': w.get('pool'), 'd_model': int(w.get('d_model', 0))}
    return ids, y, Z, A, meta


# ---------------- probe machinery (VERBATIM from c3_fusion_probe.py / apx_g0cond_gate.py) ----------------
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


# ---------------- per-cell driver ----------------
def point_arms(ds, z_arm):
    ids, y, Z, A, meta = load_cell(ds, z_arm)
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
    return {'ds': ds, 'z_arm': z_arm, 'n': n, 'n_pos': int(y.sum()), 'meta': meta,
            'Z_dim': int(Z.shape[1]), 'aux_dim': int(A.shape[1]),
            'C_Z': C_Z, 'C_Z_cv_acc': float(cacc), 'C_full': C_full, 'C_full_cv_acc': float(cfacc),
            'baseline_accZ': accZ, 'calibration': cal, 'arms': arms,
            'real_max_over_kdec': float(max(arms[f'audio_pca_k{k}']['dacc'] for k in KS_DECISION))}


def perm_null(ds, z_arm, real_maxdec, existing):
    ids, y, Z, A, meta = load_cell(ds, z_arm); n = len(y)
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
            yield _perm_stats(perk, maxk, real_maxdec)
    yield _perm_stats(perk, maxk, real_maxdec)


def eval_arm(point, perm):
    """K-LAUD-0 (kill: point<+0.040 OR CI-low<=0; would-be pass also needs perm>all) + K-LAUD-1 (calib).
    HONEST-PARTIAL flag when +0.030<=point<+0.040 with CI-low>0. Binding point = best of {k8,k16}."""
    cal = point['calibration']; calib_pass = bool(cal['PASS'])
    best_k = max(KS_DECISION, key=lambda k: point['arms'][f'audio_pca_k{k}']['dacc'])
    best = point['arms'][f'audio_pca_k{best_k}']
    C1 = bool(best['dacc'] >= BAR)
    C2 = bool(best['ci'][0] > 0)
    C3 = bool(perm['real_beats_all_permmax']) if perm else None
    honest_partial = bool(C2 and (HONEST_PARTIAL_LOW <= best['dacc'] < BAR))
    if not calib_pass:
        verdict = 'MACHINERY_INVALID'
    elif not C1 or not C2:
        verdict = 'KILL'
    else:
        verdict = 'PASS_SIDE'          # cleared K-LAUD-0's 2 point conditions; perm-null confirms
    return {'calib_pass': calib_pass, 'best_decision_k': best_k, 'best_dacc': best['dacc'],
            'best_ci': best['ci'], 'C1_point_ge_040': C1, 'C2_ci_low_gt_0': C2,
            'C3_real_beats_all_permmax': C3, 'honest_partial_flag': honest_partial, 'verdict': verdict}


def dataset_ruling(arm_evals):
    """PASS requires BOTH Z arms clear (recon sec 4). Returns KILL / HONEST_PARTIAL / PASS /
    MACHINERY_INVALID for the dataset."""
    vs = {a: e['verdict'] for a, e in arm_evals.items()}
    if any(v == 'MACHINERY_INVALID' for v in vs.values()):
        return 'MACHINERY_INVALID'
    both_pass = all(e['verdict'] == 'PASS_SIDE' and e.get('C3_real_beats_all_permmax') is True
                    for e in arm_evals.values())
    if both_pass:
        return 'PASS'
    # HONEST-PARTIAL: both arms retain a positive CI-lower AND both points >= +0.030 (documented near-miss)
    both_partial = all(e['C2_ci_low_gt_0'] and (e['best_dacc'] >= HONEST_PARTIAL_LOW)
                       for e in arm_evals.values())
    if both_partial:
        return 'HONEST_PARTIAL'
    return 'KILL'


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {
        'design': {'gate': 'LEARNED-AUDIO G0-cond (recon 166f9e2b)',
                   'aux': f'Whisper {MODEL_TAG} encoder mean(+)max video vector (2*d_model)',
                   'z_arms': {'deployed_7168': 'per-dataset deployed encoder img+text',
                              'strict_8960': 'CLIP img+text (+) frozen-Qwen img+text (== W2-A/APX Z_best)'},
                   'deploy': DEPLOY, 'datasets': DATASETS,
                   'ks_decision': KS_DECISION, 'ks_report': KS_REPORT, 'bar': BAR,
                   'honest_partial_low': HONEST_PARTIAL_LOW, 'scale_A': SCALE_A,
                   'n_splits': N_SPLITS, 'n_repeats': N_REPEATS, 'B_boot': B_BOOT, 'nseed_perm': NSEED,
                   'kill_switch': ('K-LAUD-0: best-decision-k point<+0.040 OR CI-lower<=0 -> KILL '
                                   '(would-be pass also needs real>all >=150 perm maxima); '
                                   'K-LAUD-1: calib accZA<0.99 -> MACHINERY_INVALID; '
                                   'K-LAUD-2: identical EN blank-cell screen. PASS clears BOTH Z arms.'),
                   'machinery': 'VERBATIM c3_fusion_probe.py / apx_g0cond_gate.py'},
        'cells': {}}

    for ds in DATASETS:
        for z_arm in Z_ARMS:
            key = f'{ds}|{z_arm}'
            cell = out['cells'].get(key, {})
            if 'point' not in cell:
                t = time.time(); cell['point'] = point_arms(ds, z_arm)
                out['cells'][key] = cell; json.dump(out, open(OUT, 'w'), indent=1)
                p = cell['point']; cal = p['calibration']
                print(f"[{key}] Zdim={p['Z_dim']} auxdim={p['aux_dim']} C_Z={p['C_Z']} "
                      f"accZ={p['baseline_accZ']:.4f} label_accZA={cal['label_accZA']:.4f} "
                      f"(hfrac={cal['headroom_fraction']:.3f}) CALIB_PASS={cal['PASS']}  [{time.time()-t:.0f}s]",
                      flush=True)
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
            # perm-null ONLY to confirm a would-be pass (best point >= BAR and CI-low>0)
            pe = eval_arm(cell['point'], cell.get('perm_null'))
            if pe['C1_point_ge_040'] and pe['C2_ci_low_gt_0']:
                real_maxdec = cell['point']['real_max_over_kdec']
                existing = cell.get('perm_null', {})
                if existing.get('n_seed', 0) < NSEED:
                    for st in perm_null(ds, z_arm, real_maxdec, existing):
                        cell['perm_null'] = st; out['cells'][key] = cell
                        json.dump(out, open(OUT, 'w'), indent=1)
                    st = cell['perm_null']
                    print(f"[{key}] PERM n={st['n_seed']} maxk mean={st['maxk_mean']:+.4f} "
                          f"max={st['maxk_max']:+.4f} p(realmax>=permmax)={st['p_realmax_ge_permmax']:.4f} "
                          f"real_beats_all={st['real_beats_all_permmax']}", flush=True)
            cell['arm_eval'] = eval_arm(cell['point'], cell.get('perm_null'))
            out['cells'][key] = cell

    # ---- per-dataset rulings (PASS clears BOTH arms) ----
    rulings = {}
    for ds in DATASETS:
        arm_evals = {a: out['cells'][f'{ds}|{a}']['arm_eval'] for a in Z_ARMS}
        rulings[ds] = {'ruling': dataset_ruling(arm_evals),
                       'arms': {a: {'verdict': arm_evals[a]['verdict'],
                                    'best_decision_k': arm_evals[a]['best_decision_k'],
                                    'best_dacc': arm_evals[a]['best_dacc'],
                                    'best_ci': arm_evals[a]['best_ci'],
                                    'honest_partial_flag': arm_evals[a]['honest_partial_flag'],
                                    'C3': arm_evals[a]['C3_real_beats_all_permmax']} for a in Z_ARMS}}
    out['rulings'] = rulings
    out['laud_verdict'] = {
        'HateMM_anchor': rulings['HateMM']['ruling'],
        'MHC_EN_K_LAUD_2': rulings['MHC']['ruling'],
        'MHC_zh_third_leg': rulings['MHC_zh']['ruling'],
        'promote_head_gpu': bool(rulings['HateMM']['ruling'] == 'PASS')}
    json.dump(out, open(OUT, 'w'), indent=1)

    print('\n==== LEARNED-AUDIO G0-cond gate — mechanical evaluation ====', flush=True)
    for ds in DATASETS:
        r = rulings[ds]
        print(f"  [{ds}]  RULING = {r['ruling']}", flush=True)
        for a in Z_ARMS:
            ar = r['arms'][a]
            print(f"      {a:<14s} verdict={ar['verdict']:<17s} best-k{ar['best_decision_k']} "
                  f"dacc={ar['best_dacc']:+.4f} CI[{ar['best_ci'][0]:+.4f},{ar['best_ci'][1]:+.4f}] "
                  f"honest_partial={ar['honest_partial_flag']} C3={ar['C3']}", flush=True)
    print(f"\n  HateMM(anchor)={out['laud_verdict']['HateMM_anchor']}  "
          f"EN(K-LAUD-2)={out['laud_verdict']['MHC_EN_K_LAUD_2']}  "
          f"ZH={out['laud_verdict']['MHC_zh_third_leg']}  "
          f"=> promote_head_gpu={out['laud_verdict']['promote_head_gpu']}", flush=True)
    print(f'wrote {OUT}', flush=True)


if __name__ == '__main__':
    t0 = time.time(); main(); print(f'elapsed {time.time()-t0:.0f}s', flush=True)
