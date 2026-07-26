#!/usr/bin/env python3
"""CLAP general-audio axis — G0-cond conditional-information gate (the decisive $0 CPU screen).

Fork of scripts/analysis/laud_g0cond_gate.py (sha256 b601013a...), which forks
apx_g0cond_gate.py (sha256 c338de8c...), which reuses c3_fusion_probe.py (sha256 9091e2c3...)
VERBATIM — the same C3-template conditional-information machinery that rendered the binding
W2-A / K9 / APX / LAUD verdicts. Bars FROZEN in refine-logs/CLAP_GATE_SPEC_2026-07-27.md
(commit 6c8929d) BEFORE any CLAP feature existed.

The ONLY data-layer change vs the LAUD gate: the aux block becomes the CLAP video vector.
Every constant, seed, CV scheme and bar is UNCHANGED, so the number is directly comparable to the
F41 (eGeMAPS, -0.0038) and F64 (Whisper-encoder, +0.0014) graveyard entries.

Aux blocks (spec sec 2):
  proj   1024-d  mean(+)max over 10 s windows of the L2-normalised projected joint-space embedding
                 == ClapModel.get_audio_features().  ** BINDING PRIMARY ** (this IS the CLAP object)
  hidden 2048-d  mean(+)max of the pre-projection HTSAT pooler_output.  SECONDARY (spec sec 4.4):
                 CANNOT produce a PASS; a lone pass here is recorded as DISCORDANT.

Z arms (spec sec 3), HateMM only, N=851 train-union-val (341 pos):
  deployed_7168  LoRA-curric Qwen img(+)text — "does CLAP add over what we actually deploy?"
  strict_8960    CLIP img(+)text (+) frozen-Qwen img(+)text == the exact W2-A/APX/LAUD Z_best
  A PASS must clear BOTH.

K-CLAP-3 (spec sec 5): the FN1-targeted stratum read on train-union-val items with <=25 transcript
words (N=232, 42 hate), AUC-based, with a free head-to-head against the already-banked (killed)
Whisper block and a conditional dAUC leg.

Zero test-touch: only the *_trainval cache is opened; no test label is read; no test-set metric is
computed. Gold labels are used PROBE-ONLY (calibration arm + CV stratification). CPU-only.
"""
import os, json, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
import numpy as np
import torch
import warnings
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

REPO = '/data/jehc223/RGCL'
OUT = f'{REPO}/refine-logs/CLAP_G0COND_GATE_OUT.json'
DS = 'HateMM'
MODEL_TAG = os.environ.get('CLAP_MODEL_TAG', 'larger_clap_general')

# ---- constants: identical to c3_fusion_probe.py / apx / laud (verbatim machinery) ----
N_SPLITS, N_REPEATS = 5, 5
C_GRID = [0.001, 0.01, 0.1, 1.0]
B_BOOT = 5000
SCALE_A = 50.0
MAX_ITER = 2000
BAR = 0.040                      # K-CLAP-0 promote bar (K9 house standard, == APX/LAUD)
HONEST_PARTIAL_LOW = 0.030       # K-CLAP-2 documented near-miss band
SHUFFLE_SEED = 12345
KS_REPORT = [8, 16, 32, 64]
KS_DECISION = [8, 16]
BOOT_SEED = 20260714
NSEED = int(os.environ.get('NSEED', '150'))   # >=150 (only computed on a would-be pass)

CLIP = 'openai_clip-vit-large-patch14-336_HF'
FROZEN_QWEN = 'Qwen2.5-VL-7B-Instruct_HF'
DEPLOY = 'Qwen2.5-VL-7B-Instruct-LoRA-curric_HF'
Z_ARMS = ['deployed_7168', 'strict_8960']
AUX_BLOCKS = ['proj', 'hidden']               # 'proj' BINDING, 'hidden' SECONDARY (spec sec 4.4)

# ---- K-CLAP-3 stratum (spec sec 5, FROZEN: the identical FN1 rule + word-count fn as ERRPAT) ----
STRATUM_WORDS = 25               # FN1 rule: y=1 and <=25 words (errpat_hatemm_clusters.py:131)
CONTEXT_WORDS = 1                # empty-transcript analogue, reported as UNDERPOWERED context only
AUC_KILL = 0.60                  # K-CLAP-3 KILL-side iff AUC <= 0.60 OR boot CI-low <= 0.50
AUC_KILL_CILOW = 0.50
AUC_GO = 0.65                    # NARROW-GO needs AUC >= 0.65 AND CI-low > 0.55 AND ...
AUC_GO_CILOW = 0.55
WHISPER_MARGIN = 0.05            # ... AND CLAP stratum AUC - Whisper stratum AUC >= +0.05 AND ...
                                 # ... AND conditional dAUC > 0 with boot CI-low > 0.


# ---------------- data (features-only, train U val; the _trainval cache ONLY) ----------------
def _pooled_id2row(enc):
    """id -> (concat(img_feats, text_feats), label) over train (+) dev_seen. features-only."""
    d = {}
    for split in ('train', 'dev_seen'):
        o = torch.load(f'{REPO}/data/CLIP_Embedding/{DS}/{split}_{enc}.pt',
                       map_location='cpu', weights_only=False)
        ids = o['ids'][0]
        img = o['img_feats'].numpy().astype(np.float64)
        txt = o['text_feats'].numpy().astype(np.float64)
        lab = o['labels'].numpy().astype(int)
        for i, s in enumerate(ids):
            d[s] = (np.concatenate([img[i], txt[i]]), int(lab[i]))
    return d


_CACHE = {}


def load_clap():
    if 'clap' not in _CACHE:
        _CACHE['clap'] = torch.load(f'{REPO}/data/audio/{DS}/clap_{MODEL_TAG}_trainval.pt',
                                    map_location='cpu', weights_only=False)
    return _CACHE['clap']


def load_words():
    """id -> transcript word count, using the IDENTICAL function ERRPAT used
    (errpat_hatemm_clusters.py:131  nw = len(gt[v]["text"].split())). train + val only."""
    if 'words' not in _CACHE:
        w = {}
        for split in ('train', 'val'):
            with open(f'{REPO}/data/gt/{DS}/{split}.jsonl') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    o = json.loads(line)
                    w[str(o['id'])] = len(str(o['text']).split())
        _CACHE['words'] = w
    return _CACHE['words']


def load_cell(z_arm, aux_block):
    """Canonical order = the CLAP aux cache id order (= gt train (+) val). Build Z aligned to it,
    assert label agreement across every cache. Zero test-touch."""
    c = load_clap()
    ids = list(c['ids'][0])
    A = c[aux_block].numpy().astype(np.float64)
    y = c['labels'].numpy().astype(int)
    N = len(ids)
    if z_arm == 'deployed_7168':
        dep = _pooled_id2row(DEPLOY)
        Zdim = 7168
        Z = np.zeros((N, Zdim), dtype=np.float64)
        for i, s in enumerate(ids):
            zv, yv = dep[s]
            assert yv == int(y[i]), f'label mismatch at {s}: deploy={yv} clap={y[i]}'
            Z[i] = zv
    elif z_arm == 'strict_8960':
        clip = _pooled_id2row(CLIP); qwen = _pooled_id2row(FROZEN_QWEN)
        Zdim = 8960
        Z = np.zeros((N, Zdim), dtype=np.float64)
        for i, s in enumerate(ids):
            zc, yc = clip[s]; zq, yq = qwen[s]
            assert yc == yq == int(y[i]), f'label mismatch at {s}: clip={yc} qwen={yq} clap={y[i]}'
            Z[i] = np.concatenate([zc, zq])
    else:
        raise ValueError(z_arm)
    assert Z.shape[1] == Zdim
    meta = {'N': N, 'n_pos': int(y.sum()), 'n_zero_audio_rows': int((~A.any(axis=1)).sum()),
            'aux_block': aux_block, 'aux_dim': int(A.shape[1]), 'Z_dim': int(Zdim),
            'model': c.get('model'), 'model_tag': c.get('model_tag'), 'pool': c.get('pool'),
            'window_s': float(c.get('window_s', 0)), 'sample_rate': int(c.get('sample_rate', 0))}
    return ids, y, Z, A, meta


# ---------------- probe machinery (VERBATIM from c3_fusion_probe.py / apx / laud) ----------------
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


# ---------------- K-CLAP-3: stratum AUC machinery (spec sec 5) ----------------
def oof_scores(X, y, C):
    """5x5 RepeatedStratifiedKFold out-of-fold P(y=1); per-item mean over the 5 repeats."""
    n = len(y); s = np.zeros(n); cnt = np.zeros(n)
    for rep in range(N_REPEATS):
        for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=1000 + rep).split(X, y):
            sc = StandardScaler().fit(X[tr]); cnt[te] += 1
            lr = LogisticRegression(C=C, max_iter=MAX_ITER).fit(sc.transform(X[tr]), y[tr])
            s[te] += lr.predict_proba(sc.transform(X[te]))[:, 1]
    return s / cnt


def oof_scores_cond(Z, A, y, C_Z, ks):
    """Z (+) PCA-k(aux)xs=50 at C_Z -> out-of-fold scores per k (the main-gate aux protocol)."""
    n = len(y); s = {k: np.zeros(n) for k in ks}; cnt = np.zeros(n); kmax = max(ks)
    for rep in range(N_REPEATS):
        for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=1000 + rep).split(Z, y):
            scZ = StandardScaler().fit(Z[tr]); Ztr, Zte = scZ.transform(Z[tr]), scZ.transform(Z[te]); cnt[te] += 1
            scS = StandardScaler().fit(A[tr]); Str, Ste = scS.transform(A[tr]), scS.transform(A[te])
            kk = min(kmax, len(tr) - 1, A.shape[1]); pca = PCA(n_components=kk, random_state=0).fit(Str)
            Ptr, Pte = pca.transform(Str), pca.transform(Ste)
            for k in ks:
                j = min(k, kk); scP = StandardScaler().fit(Ptr[:, :j])
                Btr = scP.transform(Ptr[:, :j]) * SCALE_A; Bte = scP.transform(Pte[:, :j]) * SCALE_A
                lr = LogisticRegression(C=C_Z, max_iter=MAX_ITER).fit(
                    np.concatenate([Ztr, Btr], 1), y[tr])
                s[k][te] += lr.predict_proba(np.concatenate([Zte, Bte], 1))[:, 1]
    return {k: s[k] / cnt for k in ks}


def auc_boot(score, y, seed):
    """AUC + per-video-clustered bootstrap CI (B=5000). Resamples rows; skips degenerate draws."""
    a = float(roc_auc_score(y, score)); n = len(y); rng = np.random.default_rng(seed); bs = []
    for _ in range(B_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        bs.append(roc_auc_score(y[idx], score[idx]))
    bs = np.array(bs)
    return a, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], len(bs)


def auc_diff_boot(s1, s2, y, seed):
    """paired bootstrap of AUC(s1) - AUC(s2) on the same rows."""
    d = float(roc_auc_score(y, s1) - roc_auc_score(y, s2)); n = len(y)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        bs.append(roc_auc_score(y[idx], s1[idx]) - roc_auc_score(y[idx], s2[idx]))
    bs = np.array(bs)
    return d, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], len(bs)


def stratum_read():
    """K-CLAP-3 (spec sec 5) + the sec 5.1 confound diagnostics. HateMM, train U val only."""
    c = load_clap(); ids = list(c['ids'][0]); y_all = c['labels'].numpy().astype(int)
    A_all = c['proj'].numpy().astype(np.float64)
    words = load_words()
    nw = np.array([words[i] for i in ids], dtype=float)

    wh = torch.load(f'{REPO}/data/audio/{DS}/whisper_whisper-large-v3_trainval.pt',
                    map_location='cpu', weights_only=False)
    wh_ids = list(wh['ids'][0])
    assert wh_ids == ids, 'whisper cache id order differs from CLAP cache — head-to-head invalid'
    W_all = wh['emb'].numpy().astype(np.float64)
    assert (wh['labels'].numpy().astype(int) == y_all).all(), 'whisper/clap label mismatch'

    dep = _pooled_id2row(DEPLOY)
    Z_all = np.stack([dep[i][0] for i in ids])

    res = {'n_all': len(ids), 'n_pos_all': int(y_all.sum()),
           'whisper_cache': f'data/audio/{DS}/whisper_whisper-large-v3_trainval.pt',
           'strata': {}, 'diagnostics': {}}

    # ---- global CLAP-alone / Whisper-alone marginal reads (DIAGNOSTIC, spec sec 5.1; move no bar) ----
    Cg, _ = pick_C(A_all, y_all)
    sg = oof_scores(A_all, y_all, Cg)
    ag, agci, _ = auc_boot(sg, y_all, BOOT_SEED + 4001)
    Cgw, _ = pick_C(W_all, y_all)
    sgw = oof_scores(W_all, y_all, Cgw)
    agw, agwci, _ = auc_boot(sgw, y_all, BOOT_SEED + 4002)
    rho_g = spearmanr(sg, nw)
    res['diagnostics']['global_marginal'] = {
        'clap_alone_auc': ag, 'clap_alone_auc_ci': agci, 'clap_C': Cg,
        'whisper_alone_auc': agw, 'whisper_alone_auc_ci': agwci, 'whisper_C': Cgw,
        'clap_minus_whisper_auc': float(ag - agw),
        'spearman_clapscore_vs_nwords': {'rho': float(rho_g.statistic), 'p': float(rho_g.pvalue)},
        'note': 'DIAGNOSTIC ONLY (spec sec 5.1) — moves no bar.'}
    print(f"[diag/global] CLAP-alone AUC={ag:.4f} CI[{agci[0]:.4f},{agci[1]:.4f}]  "
          f"Whisper-alone AUC={agw:.4f} CI[{agwci[0]:.4f},{agwci[1]:.4f}]  "
          f"diff={ag-agw:+.4f}  rho(score,nwords)={rho_g.statistic:+.4f} (p={rho_g.pvalue:.3g})", flush=True)

    # ---- stratum reads ----
    for name, thr, binding in (('le25_FN1_rule', STRATUM_WORDS, True),
                               ('le1_empty_context', CONTEXT_WORDS, False)):
        m = nw <= thr
        y = y_all[m]; A = A_all[m]; W = W_all[m]; Z = Z_all[m]
        blk = {'threshold_words': thr, 'n': int(m.sum()), 'n_pos': int(y.sum()),
               'base_rate': float(y.mean()), 'binding': binding}
        if len(np.unique(y)) < 2 or y.sum() < 5:
            blk['status'] = 'DEGENERATE_SKIP (fewer than 5 positives or single class)'
            res['strata'][name] = blk
            print(f"[K-CLAP-3/{name}] n={blk['n']} pos={blk['n_pos']} -> {blk['status']}", flush=True)
            continue

        # (a) marginal CLAP-alone
        Ca, _ = pick_C(A, y); sa = oof_scores(A, y, Ca)
        auc_a, ci_a, nb = auc_boot(sa, y, BOOT_SEED + 5001)
        # (b) head-to-head vs the banked (killed) Whisper block
        Cw, _ = pick_C(W, y); sw = oof_scores(W, y, Cw)
        auc_w, ci_w, _ = auc_boot(sw, y, BOOT_SEED + 5002)
        dmw, ci_mw, _ = auc_diff_boot(sa, sw, y, BOOT_SEED + 5003)
        # (c) conditional dAUC over deployed Z, restricted to the stratum
        Cz, _ = pick_C(Z, y); sz = oof_scores(Z, y, Cz)
        auc_z, ci_z, _ = auc_boot(sz, y, BOOT_SEED + 5004)
        scond = oof_scores_cond(Z, A, y, Cz, KS_DECISION)
        cond = {}
        for k in KS_DECISION:
            akc, cikc, _ = auc_boot(scond[k], y, BOOT_SEED + 5010 + k)
            dk, cik, _ = auc_diff_boot(scond[k], sz, y, BOOT_SEED + 5020 + k)
            cond[f'k{k}'] = {'auc': akc, 'auc_ci': cikc, 'dauc_vs_Z': dk, 'dauc_ci': cik}
        best_k = max(KS_DECISION, key=lambda k: cond[f'k{k}']['dauc_vs_Z'])
        best_cond = cond[f'k{best_k}']
        rho = spearmanr(sa, nw[m])

        blk.update({
            'a_marginal_clap': {'auc': auc_a, 'ci': ci_a, 'C': Ca, 'n_boot_valid': nb},
            'b_head_to_head': {'whisper_auc': auc_w, 'whisper_ci': ci_w, 'whisper_C': Cw,
                               'clap_minus_whisper': dmw, 'diff_ci': ci_mw},
            'c_conditional': {'Z_alone_auc': auc_z, 'Z_alone_ci': ci_z, 'C_Z': Cz,
                              'per_k': cond, 'best_k': best_k,
                              'best_dauc_vs_Z': best_cond['dauc_vs_Z'],
                              'best_dauc_ci': best_cond['dauc_ci']},
            'confound_spearman_clapscore_vs_nwords': {'rho': float(rho.statistic), 'p': float(rho.pvalue)},
        })
        if binding:
            c1 = bool(auc_a >= AUC_GO and ci_a[0] > AUC_GO_CILOW)
            c2 = bool(dmw >= WHISPER_MARGIN)
            c3 = bool(best_cond['dauc_vs_Z'] > 0 and best_cond['dauc_ci'][0] > 0)
            killside = bool(auc_a <= AUC_KILL or ci_a[0] <= AUC_KILL_CILOW)
            verdict = 'NARROW_GO' if (c1 and c2 and c3) else ('KILL' if killside else 'INCONCLUSIVE_NARROW')
            blk['K_CLAP_3'] = {'C1_auc_ge_065_and_cilow_gt_055': c1,
                               'C2_beats_whisper_by_005': c2,
                               'C3_conditional_dauc_gt_0_cilow_gt_0': c3,
                               'kill_side': killside, 'verdict': verdict}
        res['strata'][name] = blk
        print(f"[K-CLAP-3/{name}] n={blk['n']} pos={blk['n_pos']} base={blk['base_rate']:.4f}\n"
              f"    (a) CLAP-alone   AUC={auc_a:.4f} CI[{ci_a[0]:.4f},{ci_a[1]:.4f}]\n"
              f"    (b) Whisper      AUC={auc_w:.4f} CI[{ci_w[0]:.4f},{ci_w[1]:.4f}]  "
              f"CLAP-Whisper={dmw:+.4f} CI[{ci_mw[0]:+.4f},{ci_mw[1]:+.4f}]\n"
              f"    (c) Z_alone      AUC={auc_z:.4f}  best-k{best_k} dAUC={best_cond['dauc_vs_Z']:+.4f} "
              f"CI[{best_cond['dauc_ci'][0]:+.4f},{best_cond['dauc_ci'][1]:+.4f}]\n"
              f"    rho(clap_score,nwords)={rho.statistic:+.4f} (p={rho.pvalue:.3g})"
              + (f"\n    K-CLAP-3 verdict = {blk['K_CLAP_3']['verdict']}" if binding else ''), flush=True)
    return res


# ---------------- per-cell driver ----------------
def point_arms(z_arm, aux_block):
    ids, y, Z, A, meta = load_cell(z_arm, aux_block)
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
    return {'ds': DS, 'z_arm': z_arm, 'aux_block': aux_block, 'n': n, 'n_pos': int(y.sum()),
            'meta': meta, 'Z_dim': int(Z.shape[1]), 'aux_dim': int(A.shape[1]),
            'C_Z': C_Z, 'C_Z_cv_acc': float(cacc), 'C_full': C_full, 'C_full_cv_acc': float(cfacc),
            'baseline_accZ': accZ, 'calibration': cal, 'arms': arms,
            'real_max_over_kdec': float(max(arms[f'audio_pca_k{k}']['dacc'] for k in KS_DECISION))}


def perm_null(z_arm, aux_block, real_maxdec, existing):
    ids, y, Z, A, meta = load_cell(z_arm, aux_block); n = len(y)
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
    """K-CLAP-0 (kill: point<+0.040 OR CI-low<=0; would-be pass also needs perm>all)
    + K-CLAP-1 (calib) + K-CLAP-2 honest-partial band. Binding point = best of {k8,k16}."""
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
        verdict = 'PASS_SIDE'
    return {'calib_pass': calib_pass, 'best_decision_k': best_k, 'best_dacc': best['dacc'],
            'best_ci': best['ci'], 'C1_point_ge_040': C1, 'C2_ci_low_gt_0': C2,
            'C3_real_beats_all_permmax': C3, 'honest_partial_flag': honest_partial, 'verdict': verdict}


def block_ruling(arm_evals):
    """PASS requires BOTH Z arms clear (spec sec 3/4.2)."""
    vs = [e['verdict'] for e in arm_evals.values()]
    if any(v == 'MACHINERY_INVALID' for v in vs):
        return 'MACHINERY_INVALID'
    if all(e['verdict'] == 'PASS_SIDE' and e.get('C3_real_beats_all_permmax') is True
           for e in arm_evals.values()):
        return 'PASS'
    if all(e['C2_ci_low_gt_0'] and e['best_dacc'] >= HONEST_PARTIAL_LOW for e in arm_evals.values()):
        return 'HONEST_PARTIAL'
    return 'KILL'


def main():
    out = json.load(open(OUT)) if os.path.exists(OUT) else {
        'design': {'gate': 'CLAP general-audio G0-cond (spec CLAP_GATE_SPEC_2026-07-27.md, 6c8929d)',
                   'model': f'laion/{MODEL_TAG}',
                   'aux_blocks': {'proj': 'BINDING PRIMARY — mean(+)max over 10s windows of the '
                                          'L2-normalised projected joint-space embedding (1024-d)',
                                  'hidden': 'SECONDARY (spec sec 4.4) — pre-projection HTSAT '
                                            'pooler_output mean(+)max (2048-d); CANNOT produce a PASS'},
                   'z_arms': {'deployed_7168': 'LoRA-curric Qwen img+text (deployed)',
                              'strict_8960': 'CLIP img+text (+) frozen-Qwen img+text (== W2-A/APX/LAUD Z_best)'},
                   'dataset': DS, 'ks_decision': KS_DECISION, 'ks_report': KS_REPORT, 'bar': BAR,
                   'honest_partial_low': HONEST_PARTIAL_LOW, 'scale_A': SCALE_A,
                   'n_splits': N_SPLITS, 'n_repeats': N_REPEATS, 'B_boot': B_BOOT, 'nseed_perm': NSEED,
                   'kill_switch': ('K-CLAP-0: best-decision-k point<+0.040 OR CI-lower<=0 -> KILL '
                                   '(would-be pass also needs real>all >=150 perm maxima); '
                                   'K-CLAP-1: calib accZA<0.99 -> MACHINERY_INVALID; '
                                   'K-CLAP-2: +0.030..0.040 with CI-low>0 = HONEST-PARTIAL (not a promote); '
                                   'K-CLAP-3: FN1 stratum AUC read (spec sec 5). '
                                   'PASS clears BOTH Z arms, on the proj block only.'),
                   'machinery': 'VERBATIM c3_fusion_probe.py / apx_g0cond_gate.py / laud_g0cond_gate.py',
                   'test_touch': 'NONE — only the _trainval cache is opened; no test label read; '
                                 'no test-set metric computed anywhere in this lane.'},
        'cells': {}}

    for aux_block in AUX_BLOCKS:
        for z_arm in Z_ARMS:
            key = f'{aux_block}|{z_arm}'
            cell = out['cells'].get(key, {})
            if 'point' not in cell:
                t = time.time(); cell['point'] = point_arms(z_arm, aux_block)
                out['cells'][key] = cell; json.dump(out, open(OUT, 'w'), indent=1)
                p = cell['point']; cal = p['calibration']
                print(f"[{key}] Zdim={p['Z_dim']} auxdim={p['aux_dim']} C_Z={p['C_Z']} "
                      f"accZ={p['baseline_accZ']:.4f} label_accZA={cal['label_accZA']:.4f} "
                      f"(hfrac={cal['headroom_fraction']:.3f}) CALIB_PASS={cal['PASS']}  [{time.time()-t:.0f}s]",
                      flush=True)
                for k in KS_REPORT:
                    a = p['arms'][f'audio_pca_k{k}']
                    print(f"   clap_pca_k{k:<2d} accZA={a['accZA']:.4f} dacc={a['dacc']:+.4f} "
                          f"CI[{a['ci'][0]:+.4f},{a['ci'][1]:+.4f}]", flush=True)
                a = p['arms']['audio_full_cvC']
                print(f"   clap_full_cvC accZA={a['accZA']:.4f} dacc={a['dacc']:+.4f} "
                      f"CI[{a['ci'][0]:+.4f},{a['ci'][1]:+.4f}]", flush=True)
                print(f"   shuffled(seed12345) k8={p['arms']['shuffled_audio_k8']['dacc']:+.4f} "
                      f"k16={p['arms']['shuffled_audio_k16']['dacc']:+.4f}  "
                      f"real_max_over_kdec={p['real_max_over_kdec']:+.4f}", flush=True)
            pe = eval_arm(cell['point'], cell.get('perm_null'))
            if pe['C1_point_ge_040'] and pe['C2_ci_low_gt_0']:
                existing = cell.get('perm_null', {})
                if existing.get('n_seed', 0) < NSEED:
                    for st in perm_null(z_arm, aux_block, cell['point']['real_max_over_kdec'], existing):
                        cell['perm_null'] = st; out['cells'][key] = cell
                        json.dump(out, open(OUT, 'w'), indent=1)
                    st = cell['perm_null']
                    print(f"[{key}] PERM n={st['n_seed']} maxk mean={st['maxk_mean']:+.4f} "
                          f"max={st['maxk_max']:+.4f} p(realmax>=permmax)={st['p_realmax_ge_permmax']:.4f} "
                          f"real_beats_all={st['real_beats_all_permmax']}", flush=True)
            cell['arm_eval'] = eval_arm(cell['point'], cell.get('perm_null'))
            out['cells'][key] = cell
            json.dump(out, open(OUT, 'w'), indent=1)

    rulings = {}
    for aux_block in AUX_BLOCKS:
        ae = {a: out['cells'][f'{aux_block}|{a}']['arm_eval'] for a in Z_ARMS}
        rulings[aux_block] = {'ruling': block_ruling(ae),
                              'arms': {a: {'verdict': ae[a]['verdict'],
                                           'best_decision_k': ae[a]['best_decision_k'],
                                           'best_dacc': ae[a]['best_dacc'],
                                           'best_ci': ae[a]['best_ci'],
                                           'honest_partial_flag': ae[a]['honest_partial_flag'],
                                           'C3': ae[a]['C3_real_beats_all_permmax']} for a in Z_ARMS}}
    out['rulings'] = rulings

    if 'stratum' not in out:
        t = time.time()
        print('\n==== K-CLAP-3 — FN1-targeted stratum read (spec sec 5) ====', flush=True)
        out['stratum'] = stratum_read()
        out['stratum']['elapsed_s'] = round(time.time() - t, 1)
        json.dump(out, open(OUT, 'w'), indent=1)

    k3 = out['stratum']['strata'].get('le25_FN1_rule', {}).get('K_CLAP_3', {})
    primary = rulings['proj']['ruling']
    secondary = rulings['hidden']['ruling']
    discordant = bool(secondary == 'PASS' and primary != 'PASS')
    out['clap_verdict'] = {
        'primary_proj': primary,
        'secondary_hidden': secondary,
        'DISCORDANT': discordant,
        'K_CLAP_3_stratum': k3.get('verdict'),
        'promote_head_gpu': bool(primary == 'PASS'),
        'narrow_prereg_draft_authorized': bool(k3.get('verdict') == 'NARROW_GO'
                                               and primary != 'MACHINERY_INVALID'),
    }
    json.dump(out, open(OUT, 'w'), indent=1)

    print('\n==== CLAP general-audio G0-cond gate — mechanical evaluation ====', flush=True)
    for aux_block in AUX_BLOCKS:
        r = rulings[aux_block]
        role = 'BINDING PRIMARY' if aux_block == 'proj' else 'SECONDARY (cannot PASS)'
        print(f"  [{aux_block}]  {role}  RULING = {r['ruling']}", flush=True)
        for a in Z_ARMS:
            ar = r['arms'][a]
            print(f"      {a:<14s} verdict={ar['verdict']:<17s} best-k{ar['best_decision_k']} "
                  f"dacc={ar['best_dacc']:+.4f} CI[{ar['best_ci'][0]:+.4f},{ar['best_ci'][1]:+.4f}] "
                  f"honest_partial={ar['honest_partial_flag']} C3={ar['C3']}", flush=True)
    v = out['clap_verdict']
    print(f"\n  primary(proj)={v['primary_proj']}  secondary(hidden)={v['secondary_hidden']}  "
          f"DISCORDANT={v['DISCORDANT']}  K-CLAP-3={v['K_CLAP_3_stratum']}", flush=True)
    print(f"  => promote_head_gpu={v['promote_head_gpu']}  "
          f"narrow_prereg_draft_authorized={v['narrow_prereg_draft_authorized']}", flush=True)
    print(f'wrote {OUT}', flush=True)


if __name__ == '__main__':
    t0 = time.time(); main(); print(f'elapsed {time.time()-t0:.0f}s', flush=True)
