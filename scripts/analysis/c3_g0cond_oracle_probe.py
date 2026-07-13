#!/usr/bin/env python3
"""C3 G0-cond ORACLE conditional-information probe — HateMM train (744).

Purpose: bound the TARGET/COMMUNITY-SEMANTIC content of any future C3 MLLM
dense-text channel BEFORE any GPU is spent. Reuses the A-line G0-cond machinery
(refine-logs/lb_scgp_global/M1_G0COND_PROBE.py): capacity-matched linear probes
g(Z) vs g'([Z, A]), same regularization C for every arm, RepeatedStratifiedKFold
CV, MDL held-out codelength in bits, example-clustered (per-video) bootstrap CI,
and a Fano bits->acc projection. Decision bar = +0.030 + 0.010 = +0.040
(REFLECTION_mllm_integration_failures.md §4).

Gold usage is PROBE-ONLY (target labels are appended features + probe targets;
never in-method). HateMM is the ONLY dataset with gold target annotations
(data/gt/HateMM/target_map.json, built during TARC). MHC has no target field.

Arms (per encoder, coverage 1.0 for the oracle — gold target exists for all 744):
  (0) baseline           g(Z)                              [Z-only reference]
  (1) oracle_target      g'([Z, A_target])   A = 9-way one-hot(8 gold targets + none)
  (2) shuffled_target    g'([Z, A_shuf])     A rows permuted (null control; expect ~0)
  (3) label_oracle       g'([Z, A_label])    A = 2-way one-hot(gold LABEL)  [DIAGNOSTIC
                                              upper bound of ANY signal; NOT a decision arm]

Z = concat(img_feats, text_feats) raw frozen features (StandardScaler per fold);
DPI => conditioning on raw pre-projection Z is maximally generous to A => a KILL
here is maximally fail-closed. Probe = L2 logistic regression (RGCL final linear
classifier proxy). Codelength bits = sum -log2 p(y_true) on held-out folds.
"""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import torch

RNG = np.random.default_rng(20260714)
N_SPLITS, N_REPEATS = 5, 5          # 25 fits/model; >= 5 seeds (satisfies protocol)
C_GRID = [0.001, 0.01, 0.1, 1.0]
B_BOOT = 5000
EPS = 1e-12
SHUFFLE_SEED = 12345

DS = 'HateMM'
ENCODERS = [('CLIP', 'openai_clip-vit-large-patch14-336_HF'),
            ('Qwen', 'Qwen2.5-VL-7B-Instruct_HF')]


def load_features(enc):
    o = torch.load(f'data/CLIP_Embedding/{DS}/train_{enc}.pt', map_location='cpu', weights_only=False)
    ids = o['ids'][0]
    Z = np.concatenate([o['img_feats'].numpy(), o['text_feats'].numpy()], axis=1).astype(np.float64)
    y = o['labels'].numpy().astype(int)
    return ids, Z, y


def load_target_oracle(ids):
    """A_target = 9-way one-hot(gold primary target) with a dedicated 'none'(-1) column.
    Coverage 1.0 (gold target present for every train id). Returns (A_onehot, primary, code_dict)."""
    tm = json.load(open(f'data/gt/{DS}/target_map.json'))
    code_dict = tm['_meta']['code_dict']          # 8 named targets -> {0..7}
    n_tgt = tm['_meta']['num_targets']            # 8
    vids = {k: v for k, v in tm.items() if not k.startswith('_')}
    missing = [s for s in ids if s not in vids]
    assert not missing, f'{len(missing)} train ids missing in target_map: {missing[:5]}'
    prim = np.array([vids[s]['primary'] for s in ids], dtype=int)   # -1 == none
    n = len(ids)
    # column layout: [0..n_tgt-1] = named targets, [n_tgt] = none
    A = np.zeros((n, n_tgt + 1), dtype=np.float64)
    for i, p in enumerate(prim):
        A[i, p if p != -1 else n_tgt] = 1.0
    return A, prim, code_dict


def pick_C(Z, y):
    best_c, best_acc = C_GRID[0], -1.0
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    for c in C_GRID:
        accs = []
        for tr, te in skf.split(Z, y):
            pipe = Pipeline([('sc', StandardScaler()),
                             ('lr', LogisticRegression(C=c, max_iter=2000))])
            pipe.fit(Z[tr], y[tr])
            accs.append((pipe.predict(Z[te]) == y[te]).mean())
        if np.mean(accs) > best_acc:
            best_acc, best_c = float(np.mean(accs)), c
    return best_c, best_acc


def cv_eval(X, y, C):
    """Per-video held-out mean NLL (bits) and mean accuracy over N_REPEATS repeats."""
    n = len(y)
    nll = np.zeros(n); cor = np.zeros(n); cnt = np.zeros(n)
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=1000 + rep)
        for tr, te in skf.split(X, y):
            pipe = Pipeline([('sc', StandardScaler()),
                             ('lr', LogisticRegression(C=C, max_iter=2000))])
            pipe.fit(X[tr], y[tr])
            p = pipe.predict_proba(X[te])[:, 1]
            p = np.clip(p, EPS, 1 - EPS)
            pt = np.where(y[te] == 1, p, 1 - p)
            nll[te] += -np.log2(pt)
            cor[te] += ((p >= 0.5).astype(int) == y[te])
            cnt[te] += 1
    return nll / cnt, cor / cnt


def boot_ci(delta, B=B_BOOT):
    """Example-clustered (per-video) percentile bootstrap. Each video is one cluster."""
    n = len(delta)
    means = np.array([delta[RNG.integers(0, n, n)].mean() for _ in range(B)])
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def hb(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def hb_inv(L):
    """Invert binary entropy on [0, 0.5]: smallest p with Hb(p) = L (L in bits, clipped to [0,1])."""
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
    """Fano accuracy ceiling for binary Y: cross-entropy(bits) >= H(Y|X) >= Hb(Pe)
    => Pe_floor = Hb^{-1}(min(bits,1)); acc ceiling = 1 - Pe_floor."""
    return 1.0 - hb_inv(mean_bits)


def run_arm(name, Z, A, y, C):
    XZ = Z
    XZA = np.concatenate([Z, A], axis=1)
    nll_z, acc_z = cv_eval(XZ, y, C)
    nll_za, acc_za = cv_eval(XZA, y, C)
    dnll = nll_z - nll_za              # bits saved per video by A (positive = A helps)
    dacc = acc_za - acc_z             # accuracy gain per video by A
    m_bits, lo_bits, hi_bits = boot_ci(dnll)
    m_acc, lo_acc, hi_acc = boot_ci(dacc)
    fano_z = fano_ceiling(nll_z.mean())
    fano_za = fano_ceiling(nll_za.mean())
    return {
        'arm': name, 'n': int(len(y)), 'A_dim': int(A.shape[1]),
        'acc_Z': float(acc_z.mean()), 'acc_ZA': float(acc_za.mean()),
        'dacc_mean': m_acc, 'dacc_ci': [lo_acc, hi_acc],
        'bits_Z_mean': float(nll_z.mean()), 'bits_ZA_mean': float(nll_za.mean()),
        'dbits_per_video_mean': m_bits, 'dbits_ci': [lo_bits, hi_bits],
        'dbits_total': float(dnll.sum()),
        'fano_acc_ceil_Z': float(fano_z), 'fano_acc_ceil_ZA': float(fano_za),
        'fano_dacc_proj': float(fano_za - fano_z),
    }


def main():
    out = {'dataset': DS, 'design': {
        'n_splits': N_SPLITS, 'n_repeats': N_REPEATS, 'C_grid': C_GRID,
        'B_boot': B_BOOT, 'rng_seed': 20260714, 'shuffle_seed': SHUFFLE_SEED,
        'bar_dacc': 0.040}, 'encoders': {}}

    # shared: build oracle target on the CLIP ids (both encoders share the same ids order — verified)
    ids_ref = None
    for enc_name, enc in ENCODERS:
        ids, Z, y = load_features(enc)
        if ids_ref is None:
            ids_ref = ids
            A_tgt, prim, code_dict = load_target_oracle(ids)
            # null control: permute rows (break video<->target alignment, preserve marginal)
            perm = np.random.default_rng(SHUFFLE_SEED).permutation(len(ids))
            A_shuf = A_tgt[perm]
            # diagnostic label oracle: 2-way one-hot(gold label)
            A_lab = np.zeros((len(y), 2)); A_lab[np.arange(len(y)), y] = 1.0
            from collections import Counter
            out['target_facts'] = {
                'n': int(len(ids)), 'base_rate_pos': float(y.mean()),
                'code_dict': code_dict,
                'primary_dist': {str(k): int(v) for k, v in sorted(Counter(prim.tolist()).items())},
                'coverage_non_none': float((prim != -1).mean()),
                'n_none': int((prim == -1).sum()),
                'none_all_nonhate': bool(((prim == -1) & (y == 1)).sum() == 0),
            }
        else:
            assert ids == ids_ref, 'encoder ids order mismatch — oracle alignment would break'

        C, cacc = pick_C(Z, y)
        arms = {}
        arms['oracle_target'] = run_arm('oracle_target (9-way one-hot, cov 1.0)', Z, A_tgt, y, C)
        arms['shuffled_target'] = run_arm('shuffled_target (null control)', Z, A_shuf, y, C)
        arms['label_oracle'] = run_arm('label_oracle (DIAGNOSTIC any-signal ceiling)', Z, A_lab, y, C)
        # baseline acc is arms' acc_Z (identical across arms — same Z, same C)
        out['encoders'][enc_name] = {
            'encoder_file': f'data/CLIP_Embedding/{DS}/train_{enc}.pt',
            'Z_dim': int(Z.shape[1]), 'C': C, 'C_cv_acc': float(cacc),
            'baseline_acc_Z': float(arms['oracle_target']['acc_Z']),
            'arms': arms}
        print(f'[{DS}/{enc_name}] C={C} Zdim={Z.shape[1]} baseline_acc={arms["oracle_target"]["acc_Z"]:.4f}',
              flush=True)
        for k, r in arms.items():
            print(f'   {r["arm"]:44s} accZ={r["acc_Z"]:.4f} accZA={r["acc_ZA"]:.4f} '
                  f'dacc={r["dacc_mean"]:+.4f} CI[{r["dacc_ci"][0]:+.4f},{r["dacc_ci"][1]:+.4f}] '
                  f'dbits/vid={r["dbits_per_video_mean"]:+.5f} CI[{r["dbits_ci"][0]:+.5f},{r["dbits_ci"][1]:+.5f}] '
                  f'fano_dacc={r["fano_dacc_proj"]:+.4f}', flush=True)

    outpath = sys.argv[1] if len(sys.argv) > 1 else 'c3_g0cond_oracle_out.json'
    json.dump(out, open(outpath, 'w'), indent=1)
    print(f'\nwrote {outpath}', flush=True)


if __name__ == '__main__':
    main()
